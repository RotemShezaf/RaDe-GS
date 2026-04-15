#!/bin/bash
#
# Rebuild the 2 meshes that had valence-0 Gaussian vertices:
#   - HyperbolicParaboloid / level_02 / light_0
#   - Paraboloid / level_02 / light_3
#
# Uses the same parameters as build_geodesic_mesh_polynomial_all.sh
# for level 02 meshes, but filters to only the two affected combos.
#
# USAGE:
#   bash scripts/polynomial/geodesic/rebuild_orphan_meshes_all.sh [--verbose] [--dry_run]

set -uo pipefail

# ============================================================================
# Defaults (matching build_geodesic_mesh_polynomial_all.sh for level 02)
# ============================================================================
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
TEXTURE="blue"
SEED="42"
N_POINTS="400000"
DRY_RUN=false
VERBOSE=""

MESH_METHOD="grid"
CURVATURE_ALPHA="1.3"
MAX_EDGE="0.0022"
MAX_EDGE_GAUSS="0.0008"
MAX_AREA_FACTOR="2"
MAX_AREA_FACTOR_GAUSS="1.5"
REFINE_ITERATIONS="1"
REFINE_WARMUP_ITERATIONS="1"
REFINE_PATIENCE="8"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)  VERBOSE="--verbose"; shift ;;
        --dry_run)  DRY_RUN=true;        shift ;;
        --help|-h)  sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_geodesic_mesh_for_gaussians.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Define the 2 problematic meshes
# ============================================================================
declare -a SURFACES=("HyperbolicParaboloid" "Paraboloid")
declare -a LIGHTS=("0" "3")

# Per-surface curvature alpha overrides (from the main build script)
declare -A ALPHA_MAP=(
    ["HyperbolicParaboloid"]="1.3"
    ["Paraboloid"]="1.2"
)

echo "============================================================"
echo "Rebuild Orphan-Affected Meshes"
echo "============================================================"
echo ""

FAILED=0
for i in "${!SURFACES[@]}"; do
    surface="${SURFACES[$i]}"
    light="${LIGHTS[$i]}"
    alpha="${ALPHA_MAP[$surface]:-$CURVATURE_ALPHA}"

    GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${TEXTURE}_texture/$surface/level_02/light_${light}/output"
    LABEL="${TEXTURE}/$surface/level_02/light_${light}"

    if [ ! -d "$GAUSSIAN_OUTPUT" ] || [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
        echo "  SKIP $LABEL — output not found"
        continue
    fi

    CMD="python $COMPUTE_SCRIPT \
        --gaussian_output $GAUSSIAN_OUTPUT \
        --surface $surface \
        --max_edge_length $MAX_EDGE \
        --seed $SEED \
        --n_points $N_POINTS \
        --curvature_adaptive --curvature_alpha $alpha \
        --refine --mesh_method $MESH_METHOD \
        --refine_gauss_max_edge_length $MAX_EDGE_GAUSS \
        --refine_max_area_factor $MAX_AREA_FACTOR \
        --refine_gauss_max_area_factor $MAX_AREA_FACTOR_GAUSS \
        --refine_iterations $REFINE_ITERATIONS \
        --refine_warmup_iterations $REFINE_WARMUP_ITERATIONS \
        --refine_ring_fix --refine_ring_fix_iterations 1 \
        --refine_delaunay_flip \
        --refine_patience $REFINE_PATIENCE \
        --local_refinement"

    if [ -n "$VERBOSE" ]; then
        CMD="$CMD $VERBOSE"
    fi

    echo "  [$((i+1))/2] $LABEL"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] $CMD"
        echo ""
        continue
    fi

    echo "    Running..."
    if eval "$CMD"; then
        echo "    ✓ Done"
    else
        echo "    ✗ FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

if [ "$FAILED" -gt 0 ]; then
    echo "  ⚠ $FAILED job(s) failed"
    exit 1
fi

echo "All done."
