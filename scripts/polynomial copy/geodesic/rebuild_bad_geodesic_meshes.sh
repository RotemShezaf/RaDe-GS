#!/bin/bash
#
# Rebuild geodesic meshes that have non-manifold edges or duplicate faces.
#
# Scans all mesh_statistics.json files under the data root, identifies
# meshes with topology issues (non-manifold edges, duplicate faces),
# and re-runs compute_geodesic_mesh_for_gaussians.py for each.
#
# Uses the same parameters as build_geodesic_mesh_polynomial_all.sh.
#
# USAGE:
#   bash scripts/polynomial/geodesic/rebuild_bad_geodesic_meshes.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR   Data root (default: TrainData/Polynomial/SyntheticColmapData)
#   --surfaces LIST         Comma-separated surfaces (default: all)
#   --textures LIST         Comma-separated textures (default: blue)
#   --levels LIST           Comma-separated levels (default: 02,03,04)
#   --light_ids LIST        Comma-separated light IDs (default: 0,1,2,3,4)
#   --outputs LIST          Output folder names (default: output)
#   --jobs N                Parallel workers (default: nproc)
#   --dry_run               Print commands without executing
#   --verbose               Enable verbose output
#   --force                 Rebuild all meshes, not just bad ones

set -uo pipefail

# ============================================================================
# Defaults
# ============================================================================
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="02,03,04"
OUTPUTS="output"
LIGHT_IDS="0,1,2,3,4"
SEED="42"
DRY_RUN=false
VERBOSE=""
FORCE=false
MESH_METHOD="grid"
N_POINTS="400000"

MAX_JOBS=$(nproc 2>/dev/null || echo 1)

# Per-level/surface maps (same as build_geodesic_mesh_polynomial_all.sh)
declare -A MAX_EDGE_LENGTH_MAP=(
    ["02_Paraboloid"]="0.003" ["02_Saddle"]="0.004" ["02_HyperbolicParaboloid"]="0.004"
    ["03_Paraboloid"]="0.005" ["03_Saddle"]="0.005" ["03_HyperbolicParaboloid"]="0.005"
    ["04_Paraboloid"]="0.005" ["04_Saddle"]="0.005" ["04_HyperbolicParaboloid"]="0.005"
)
declare -A MAX_EDGE_LENGTH_GAUSSIANS_MAP=(
    ["02_Paraboloid"]="0.005" ["02_Saddle"]="0.005" ["02_HyperbolicParaboloid"]="0.005"
    ["03_Paraboloid"]="0.005" ["03_Saddle"]="0.005" ["03_HyperbolicParaboloid"]="0.005"
    ["04_Paraboloid"]="0.005" ["04_Saddle"]="0.005" ["04_HyperbolicParaboloid"]="0.005"
)
declare -A MAX_AREA_FACTOR_MAP=(
    ["02_Paraboloid"]="2" ["02_Saddle"]="2" ["02_HyperbolicParaboloid"]="2"
    ["03_Paraboloid"]="2" ["03_Saddle"]="2" ["03_HyperbolicParaboloid"]="2"
    ["04_Paraboloid"]="2" ["04_Saddle"]="2" ["04_HyperbolicParaboloid"]="2"
)
declare -A MAX_AREA_FACTOR_GAUSSIANS_MAP=(
    ["02_Paraboloid"]="1.5" ["02_Saddle"]="1.5" ["02_HyperbolicParaboloid"]="1.5"
    ["03_Paraboloid"]="1.5" ["03_Saddle"]="1.5" ["03_HyperbolicParaboloid"]="1.5"
    ["04_Paraboloid"]="1.5" ["04_Saddle"]="1.5" ["04_HyperbolicParaboloid"]="1.5"
)
declare -A CURVATURE_ALPHA_MAP=(
    ["02_Paraboloid"]="1.2" ["02_Saddle"]="1.4" ["02_HyperbolicParaboloid"]="1.4"
    ["03_Paraboloid"]="1.4" ["03_Saddle"]="1.4" ["03_HyperbolicParaboloid"]="1.4"
    ["04_Paraboloid"]="1.4" ["04_Saddle"]="1.4" ["04_HyperbolicParaboloid"]="1.4"
)

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base) SYNTH_DATA_BASE="$2"; shift 2 ;;
        --surfaces) SURFACES="$2"; shift 2 ;;
        --textures) TEXTURES="$2"; shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --outputs) OUTPUTS="$2"; shift 2 ;;
        --light_ids) LIGHT_IDS="$2"; shift 2 ;;
        --jobs|-j) MAX_JOBS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        --force) FORCE=true; shift ;;
        --help|-h) sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Convert lists
# ============================================================================
IFS=',' read -ra SURFACE_ARRAY <<< "$SURFACES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"
IFS=',' read -ra OUTPUT_ARRAY <<< "$OUTPUTS"
IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/GenerateData/compute_geodesic_mesh_for_gaussians.py"

# Use geo_splat conda env for running compute script (needs open3d, etc.)
CONDA_PREFIX="${CONDA_PREFIX:-/home/rotem.shezaf/miniconda3/envs/geo_splat}"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

# Activate venv for scanning (needs numpy + geodesic_mesh_utils)
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python not found: $PYTHON_BIN (set CONDA_PREFIX to geo_splat env)"
    exit 1
fi

# ============================================================================
# Scan for bad meshes
# ============================================================================
echo "============================================================"
echo "Scanning for meshes with topology issues ..."
echo "============================================================"
echo ""

declare -a BAD_LABELS=()
declare -a BAD_CMDS=()
GOOD=0
MISSING=0

for texture in "${TEXTURE_ARRAY[@]}"; do
    for level in "${LEVEL_ARRAY[@]}"; do
        for light_id in "${LIGHT_ID_ARRAY[@]}"; do
            for output_name in "${OUTPUT_ARRAY[@]}"; do
                for surface in "${SURFACE_ARRAY[@]}"; do
                    GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/light_${light_id}/$output_name"
                    MESH_NPZ="$GAUSSIAN_OUTPUT/geodesic_mesh/geodesic_mesh_data.npz"
                    LABEL="$texture/$surface/level_$level/light_$light_id/$output_name"

                    if [ ! -f "$MESH_NPZ" ]; then
                        MISSING=$((MISSING + 1))
                        continue
                    fi

                    # Check topology with Python
                    NEEDS_REBUILD=false
                    if [ "$FORCE" = true ]; then
                        NEEDS_REBUILD=true
                    else
                        RESULT=$(python3 -c "
import numpy as np, sys, os
os.chdir('$PROJECT_ROOT')
sys.path.insert(0, '.')
from GenerateData.utils.geodesic_mesh_utils import _detect_non_manifold_edges, _detect_duplicate_faces, _count_mesh_holes
d = np.load('$MESH_NPZ')
faces = d['faces']
n_nm, _ = _detect_non_manifold_edges(faces)
n_dup, _ = _detect_duplicate_faces(faces)
n_holes = _count_mesh_holes(faces)
print(f'{n_nm},{n_dup},{n_holes}')
" 2>&1)
                        RC=$?
                        if [ "$RC" -ne 0 ]; then
                            echo "  WARN $LABEL: topology check failed, skipping"
                            GOOD=$((GOOD + 1))
                            continue
                        fi
                        NM=$(echo "$RESULT" | cut -d, -f1)
                        DUP=$(echo "$RESULT" | cut -d, -f2)
                        HOLES=$(echo "$RESULT" | cut -d, -f3)
                        if [ "$NM" != "0" ] || [ "$DUP" != "0" ] || [ "$HOLES" != "0" ]; then
                            NEEDS_REBUILD=true
                            echo "  BAD  $LABEL: $NM non-manifold, $DUP duplicates, $HOLES holes"
                        fi
                    fi

                    if [ "$NEEDS_REBUILD" = true ]; then
                        MAP_KEY="${level}_${surface}"
                        LEVEL_MAX_EDGE="${MAX_EDGE_LENGTH_MAP[$MAP_KEY]:-0.005}"
                        LEVEL_MAX_EDGE_GAUSS="${MAX_EDGE_LENGTH_GAUSSIANS_MAP[$MAP_KEY]:-0.004}"
                        LEVEL_MAX_AREA="${MAX_AREA_FACTOR_MAP[$MAP_KEY]:-3.0}"
                        LEVEL_MAX_AREA_GAUSS="${MAX_AREA_FACTOR_GAUSSIANS_MAP[$MAP_KEY]:-2.5}"
                        LEVEL_CURVATURE_ALPHA="${CURVATURE_ALPHA_MAP[$MAP_KEY]:-1.4}"

                        CMD="$PYTHON_BIN $COMPUTE_SCRIPT \\
                            --gaussian_output $GAUSSIAN_OUTPUT \
                            --surface $surface \
                            --max_edge_length 0 \
                            --seed $SEED \
                            --n_points $N_POINTS \
                            --curvature_adaptive --curvature_alpha $LEVEL_CURVATURE_ALPHA \
                            --max_edge_length $LEVEL_MAX_EDGE \
                            --refine --mesh_method $MESH_METHOD \
                            --refine_gauss_max_edge_length $LEVEL_MAX_EDGE_GAUSS \
                            --refine_max_area_factor $LEVEL_MAX_AREA \
                            --refine_gauss_max_area_factor $LEVEL_MAX_AREA_GAUSS \
                            --refine_iterations 1 \
                            --refine_warmup_iterations 1 \
                            --refine_ring_fix \
                            --refine_ring_fix_iterations 1 \
                            --refine_delaunay_flip \
                            --refine_patience 8 \
                            --local_refinement \
                            $VERBOSE"

                        BAD_LABELS+=("$LABEL")
                        BAD_CMDS+=("$CMD")
                    else
                        GOOD=$((GOOD + 1))
                    fi
                done
            done
        done
    done
done

TOTAL_BAD=${#BAD_LABELS[@]}

echo ""
echo "Results: $GOOD good, $TOTAL_BAD need rebuild, $MISSING missing"
echo ""

if [ "$TOTAL_BAD" -eq 0 ]; then
    echo "All meshes are topologically clean. Nothing to rebuild."
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN — commands that would run:"
    for i in "${!BAD_LABELS[@]}"; do
        echo ""
        echo "  [$((i+1))/$TOTAL_BAD] ${BAD_LABELS[$i]}"
        echo "  ${BAD_CMDS[$i]}"
    done
    exit 0
fi

# ============================================================================
# Parallel rebuild
# ============================================================================
echo "============================================================"
echo "Rebuilding $TOTAL_BAD meshes with up to $MAX_JOBS workers ..."
echo "============================================================"
echo ""

FAIL_FILE=$(mktemp /tmp/rebuild_failures_XXXXXX)
SUCCESS_COUNT=0

for i in "${!BAD_LABELS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  [$((i+1))/$TOTAL_BAD] ${BAD_LABELS[$i]}"
    eval "${BAD_CMDS[$i]}" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "  ✓ Done"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ✗ FAILED (exit $rc)"
        echo "${BAD_LABELS[$i]}" >> "$FAIL_FILE"
    fi
done

# ============================================================================
# Summary
# ============================================================================
FAILED=()
if [ -s "$FAIL_FILE" ]; then
    while IFS= read -r line; do FAILED+=("$line"); done < "$FAIL_FILE"
fi
rm -f "$FAIL_FILE"

echo ""
echo "============================================================"
echo "Rebuild Complete"
echo "============================================================"
echo "  Rebuilt:  $TOTAL_BAD"
echo "  Failed:   ${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
