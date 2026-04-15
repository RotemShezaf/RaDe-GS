#!/bin/bash
#
# Build geodesic meshes for ALL polynomial Gaussian outputs.
#
# Mirrors the iteration structure of compute_geodesic_polynomial_all.sh:
#   texture × level × light_id × output_name × surface
#
# Each Gaussian output gets its own geodesic mesh saved under
#   <gaussian_output>/geodesic_mesh/geodesic_mesh.ply
#
# USAGE:
#   ./build_geodesic_mesh_polynomial_all.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR   Synthetic COLMAP data dir (default: TrainData/Polynomial/SyntheticColmapData)
#   --surfaces LIST         Comma-separated surface types (default: Paraboloid,Saddle,HyperbolicParaboloid)
#   --textures LIST         Comma-separated texture names (default: blue)
#   --levels LIST           Comma-separated colmap levels (default: 02,03,04)
#   --outputs LIST          Comma-separated output folder names (default: output)
#   --light_ids LIST        Comma-separated light IDs (default: 0,1,2,3,4)
#   --n_points N            Add N surface samples per mesh (default: 300000)
#   --min_radius R          Minimum (x,y) spacing for extra samples (optional)
#   --curvature_adaptive    Use curvature-adaptive sampling (default: enabled)
#   --no_curvature          Use uniform sampling instead of curvature-adaptive
#   --curvature_alpha F     Global curvature sensitivity fallback (default: 1.4)
#   --seed N                RNG seed (default: 42)
#   --jobs N                Parallel workers (default: nproc)
#   --dry_run               Print commands without executing
#   --verbose               Enable verbose output
#
# EXAMPLES:
#   # Build meshes for everything (Gaussian vertices only, no extra samples)
#   ./build_geodesic_mesh_polynomial_all.sh
#
#   # With uniform sampling instead of curvature-adaptive
#   ./build_geodesic_mesh_polynomial_all.sh --no_curvature
#
#   # Only Paraboloid, 500k samples
#   ./build_geodesic_mesh_polynomial_all.sh --surfaces Paraboloid --n_points 500000
#
#   # Dry run
#   ./build_geodesic_mesh_polynomial_all.sh --dry_run

# Do NOT use set -e here — it interferes with background-job exit-code collection.
set -uo pipefail

# ============================================================================
# Default values
# ============================================================================
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="02,03,04"
OUTPUTS="output"
LIGHT_IDS="0,1,2,3,4"
N_POINTS="400000"
MIN_RADIUS=""
CURVATURE_ADAPTIVE=true
CURVATURE_ALPHA="1.4"
SEED="42"
DRY_RUN=false
VERBOSE=""
REFINE="--refine"
REFINE_ITERATIONS="1"
REFINE_WARMUP_ITERATIONS="1"
REFINE_RING_FIX=true
REFINE_DELAUNAY_FLIP=true
REFINE_SURFACE_AWARE=false
REFINE_PATIENCE="8"
LOCAL_REFINEMENT="--local_refinement"
BEST_PARAMS_JSON=""

MESH_METHOD="grid"

# Parallelism: default to all available CPUs
MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

# Per-level and per-surface maps for mesh quality thresholds.
# Keys are formatted as "${level}_${surface}".
# Adjust these to control triangle quality per level and surface.
declare -A MAX_EDGE_LENGTH_MAP=(
    ["02_Paraboloid"]="0.0022"
    ["02_Saddle"]="0.0022"
    ["02_HyperbolicParaboloid"]="0.0022"
    ["03_Paraboloid"]="0.0028"
    ["03_Saddle"]="0.0028"
    ["03_HyperbolicParaboloid"]="0.0028"
    ["04_Paraboloid"]="0.0028"
    ["04_Saddle"]="0.0028"
    ["04_HyperbolicParaboloid"]="0.0028"
)

declare -A MAX_EDGE_LENGTH_GAUSSIANS_MAP=(
    ["02_Paraboloid"]="0.0008"
    ["02_Saddle"]="0.0008"
    ["02_HyperbolicParaboloid"]="0.0008"
    ["03_Paraboloid"]="0.0008"
    ["03_Saddle"]="0.0008"
    ["03_HyperbolicParaboloid"]="0.0008"
    ["04_Paraboloid"]="0.0008"
    ["04_Saddle"]="0.0008"
    ["04_HyperbolicParaboloid"]="0.0008"
)

declare -A MAX_AREA_FACTOR_MAP=(
    ["02_Paraboloid"]="2"
    ["02_Saddle"]="2"
    ["02_HyperbolicParaboloid"]="2"
    ["03_Paraboloid"]="2"
    ["03_Saddle"]="2"
    ["03_HyperbolicParaboloid"]="2"
    ["04_Paraboloid"]="2"
    ["04_Saddle"]="2"
    ["04_HyperbolicParaboloid"]="2"
)

declare -A MAX_AREA_FACTOR_GAUSSIANS_MAP=(
    ["02_Paraboloid"]="1.5"
    ["02_Saddle"]="1.5"
    ["02_HyperbolicParaboloid"]="1.5"
    ["03_Paraboloid"]="1.5"
    ["03_Saddle"]="1.5"
    ["03_HyperbolicParaboloid"]="1.5"
    ["04_Paraboloid"]="1.5"
    ["04_Saddle"]="1.5"
    ["04_HyperbolicParaboloid"]="1.5"
)

declare -A CURVATURE_ALPHA_MAP=(
    ["02_Paraboloid"]="1.2"
    ["02_Saddle"]="1.3"
    ["02_HyperbolicParaboloid"]="1.3"
    ["03_Paraboloid"]="1.3"
    ["03_Saddle"]="1.3"
    ["03_HyperbolicParaboloid"]="1.3"
    ["04_Paraboloid"]="1.3"
    ["04_Saddle"]="1.3"
    ["04_HyperbolicParaboloid"]="1.3"
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
        --n_points) N_POINTS="$2"; shift 2 ;;
        --min_radius) MIN_RADIUS="$2"; shift 2 ;;
        --curvature_adaptive) CURVATURE_ADAPTIVE=true; shift ;;
        --no_curvature) CURVATURE_ADAPTIVE=false; shift ;;
        --curvature_alpha) CURVATURE_ALPHA="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --jobs|-j) MAX_JOBS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --refine_iterations) REFINE_ITERATIONS="$2"; shift 2 ;;
        --refine_ring_fix) REFINE_RING_FIX=true; shift ;;
        --refine_delaunay_flip) REFINE_DELAUNAY_FLIP=true; shift ;;
        --refine_surface_aware) REFINE_SURFACE_AWARE=true; shift ;;
        --refine_patience) REFINE_PATIENCE="$2"; shift 2 ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        --best_params) BEST_PARAMS_JSON="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Convert comma-separated lists to arrays
# ============================================================================
IFS=',' read -ra SURFACE_ARRAY <<< "$SURFACES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra LEVEL_ARRAY <<< "$LEVELS"
IFS=',' read -ra OUTPUT_ARRAY <<< "$OUTPUTS"

if [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Locate the Python script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_geodesic_mesh_for_gaussians.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Pre-scan: build work list, count skipped upfront
# ============================================================================
declare -a JOB_LABELS=()
declare -a JOB_CMDS=()
declare -a JOB_OUTPUTS=()

SKIPPED=0

for texture in "${TEXTURE_ARRAY[@]}"; do
    for level in "${LEVEL_ARRAY[@]}"; do
        for light_id in "${LIGHT_ID_ARRAY[@]}"; do
            for output_name in "${OUTPUT_ARRAY[@]}"; do
                for surface in "${SURFACE_ARRAY[@]}"; do

                    if [ -n "$light_id" ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/light_${light_id}/$output_name"
                    else
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/default_light/$output_name"
                    fi

                    LABEL="$texture/$surface/level_$level/light_${light_id:-default}/$output_name"

                    if [ ! -d "$GAUSSIAN_OUTPUT" ] || [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    # Create the composite key
                    MAP_KEY="${level}_${surface}"

                    # Defaults from per-level/surface maps
                    LEVEL_MAX_EDGE="${MAX_EDGE_LENGTH_MAP[$MAP_KEY]:-0.005}"
                    LEVEL_MAX_EDGE_GAUSS="${MAX_EDGE_LENGTH_GAUSSIANS_MAP[$MAP_KEY]:-0.004}"
                    LEVEL_MAX_AREA="${MAX_AREA_FACTOR_MAP[$MAP_KEY]:-3.0}"
                    LEVEL_MAX_AREA_GAUSS="${MAX_AREA_FACTOR_GAUSSIANS_MAP[$MAP_KEY]:-2.5}"
                    LEVEL_CURVATURE_ALPHA="${CURVATURE_ALPHA_MAP[$MAP_KEY]:-$CURVATURE_ALPHA}"
                    RING_FIX_ITERS="1"

                    # Override from per-output best params JSON if provided
                    if [ -n "$BEST_PARAMS_JSON" ] && [ -f "$BEST_PARAMS_JSON" ]; then
                        BP_KEY="${texture}/${surface}/L${level}/light_${light_id}"
                        BP_EDGE=$(python3 -c "import json,sys; d=json.load(open('$BEST_PARAMS_JSON')); p=d.get('$BP_KEY',{}); print(p.get('edge_length',''))" 2>/dev/null)
                        BP_RING=$(python3 -c "import json,sys; d=json.load(open('$BEST_PARAMS_JSON')); p=d.get('$BP_KEY',{}); print(p.get('ring_fix_iterations',''))" 2>/dev/null)
                        if [ -n "$BP_EDGE" ]; then
                            LEVEL_MAX_EDGE="$BP_EDGE"
                            LEVEL_MAX_EDGE_GAUSS=$(python3 -c "print(round($BP_EDGE * 0.75, 5))")
                        fi
                        if [ -n "$BP_RING" ]; then
                            RING_FIX_ITERS="$BP_RING"
                        fi
                    fi

                    CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --surface $surface \
                        --max_edge_length 0 \
                        --seed $SEED"

                    if [ -n "$N_POINTS" ]; then
                        CMD="$CMD --n_points $N_POINTS"
                    fi

                    if [ -n "$MIN_RADIUS" ]; then
                        CMD="$CMD --min_radius $MIN_RADIUS"
                    fi

                    if [ "$CURVATURE_ADAPTIVE" = true ]; then
                        CMD="$CMD --curvature_adaptive --curvature_alpha $LEVEL_CURVATURE_ALPHA --max_edge_length $LEVEL_MAX_EDGE $REFINE --mesh_method $MESH_METHOD"
                        if [ -n "$REFINE" ]; then
                            CMD="$CMD --refine_gauss_max_edge_length $LEVEL_MAX_EDGE_GAUSS"
                            CMD="$CMD --refine_max_area_factor $LEVEL_MAX_AREA"
                            CMD="$CMD --refine_gauss_max_area_factor $LEVEL_MAX_AREA_GAUSS"
                            CMD="$CMD --refine_iterations $REFINE_ITERATIONS"
                            CMD="$CMD --refine_warmup_iterations $REFINE_WARMUP_ITERATIONS"
                            if [ "$REFINE_RING_FIX" = true ]; then
                                CMD="$CMD --refine_ring_fix"
                                CMD="$CMD --refine_ring_fix_iterations $RING_FIX_ITERS"
                            fi
                            if [ "$REFINE_DELAUNAY_FLIP" = true ]; then
                                CMD="$CMD --refine_delaunay_flip"
                            fi
                            if [ "$REFINE_SURFACE_AWARE" = true ]; then
                                CMD="$CMD --refine_surface_aware"
                            fi
                            if [ -n "$REFINE_PATIENCE" ]; then
                                CMD="$CMD --refine_patience $REFINE_PATIENCE"
                            fi
                        fi
                    fi

                    if [ -n "$VERBOSE" ]; then
                        CMD="$CMD $VERBOSE"
                    fi
                    CMD="$CMD $LOCAL_REFINEMENT"

                    JOB_LABELS+=("$LABEL")
                    JOB_CMDS+=("$CMD")
                    JOB_OUTPUTS+=("$GAUSSIAN_OUTPUT")
                done
            done
        done
    done
done

TOTAL=${#JOB_LABELS[@]}

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Build Geodesic Meshes — All Polynomial Surfaces"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Synth data base: $SYNTH_DATA_BASE"
echo "  Textures:        ${TEXTURE_ARRAY[*]}"
echo "  Levels:          ${LEVEL_ARRAY[*]}"
echo "  Outputs:         ${OUTPUT_ARRAY[*]}"
echo "  Light IDs:       ${LIGHT_ID_ARRAY[*]:-<standard lighting>}"
echo "  Surfaces:        ${SURFACE_ARRAY[*]}"
echo "  N points:        ${N_POINTS:-<none>}"
echo "  Min radius:      ${MIN_RADIUS:-<none>}"
echo "  Curvature adapt: $CURVATURE_ADAPTIVE"
echo "  Seed:            $SEED"
echo "  Parallel jobs:   $MAX_JOBS"
echo "  Refine iters:    $REFINE_ITERATIONS"
echo "  Dry run:         $DRY_RUN"
echo "  Total outputs:   $TOTAL  (skipped: $SKIPPED)"
echo ""
echo "  Per-level and surface thresholds:"
# Print the updated map configurations nicely sorted by key
for key in $(printf '%s\n' "${!MAX_EDGE_LENGTH_MAP[@]}" | sort); do
    echo "    [${key}]: max_edge=${MAX_EDGE_LENGTH_MAP[$key]}" \
         " max_edge_gauss=${MAX_EDGE_LENGTH_GAUSSIANS_MAP[$key]}" \
         " area_factor=${MAX_AREA_FACTOR_MAP[$key]}" \
         " area_factor_gauss=${MAX_AREA_FACTOR_GAUSSIANS_MAP[$key]}" \
         " curvature_alpha=${CURVATURE_ALPHA_MAP[$key]:-$CURVATURE_ALPHA}"
done
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to do — all outputs skipped or not found."
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "============================================================"
    echo "DRY RUN — commands that would be executed:"
    echo "============================================================"
    for i in "${!JOB_LABELS[@]}"; do
        echo ""
        echo "  [$(( i + 1 ))/$TOTAL] ${JOB_LABELS[$i]}"
        echo "  ${JOB_CMDS[$i]}"
    done
    echo ""
    exit 0
fi

# ============================================================================
# Parallel pool helpers
# ============================================================================
_FAIL_FILE=$(mktemp /tmp/geodesic_failures_XXXXXX)
declare -a _PIDS=()
declare -a _LABELS=()
declare -a _LOGS=()

_pool_launch() {
    local label="$1"
    local gout="$2"
    local cmd="$3"
    local log
    log=$(mktemp /tmp/geodesic_job_XXXXXX.log)

    (
        set +e
        echo "  Path: $gout"
        echo ""
        eval "$cmd"
        rc=$?
        exit $rc
    ) > "$log" 2>&1 &

    local pid=$!
    _PIDS+=("$pid")
    _LABELS+=("$label")
    _LOGS+=("$log")
}

_pool_reap_one() {
    wait -n 2>/dev/null || true

    local new_pids=() new_labels=() new_logs=()
    local reaped=0

    for i in "${!_PIDS[@]}"; do
        local pid="${_PIDS[$i]}"
        if [ "$reaped" -eq 0 ] && ! kill -0 "$pid" 2>/dev/null; then
            local rc=0
            wait "$pid" 2>/dev/null || rc=$?
            local label="${_LABELS[$i]}"
            local log="${_LOGS[$i]}"

            {
                echo "------------------------------------------------------------"
                echo "  ${label}"
                cat "$log"
                if [ "$rc" -eq 0 ]; then
                    echo "  ✓ Mesh saved"
                else
                    echo "  ✗ FAILED (exit $rc)"
                    echo "$label" >> "$_FAIL_FILE"
                fi
                echo ""
            }

            rm -f "$log"
            reaped=1
        else
            new_pids+=("$pid")
            new_labels+=("${_LABELS[$i]}")
            new_logs+=("${_LOGS[$i]}")
        fi
    done

    _PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    _LABELS=("${new_labels[@]+"${new_labels[@]}"}")
    _LOGS=("${new_logs[@]+"${new_logs[@]}"}")
}

_pool_throttle() {
    while [ "${#_PIDS[@]}" -ge "$MAX_JOBS" ]; do
        _pool_reap_one
    done
}

_pool_drain() {
    while [ "${#_PIDS[@]}" -gt 0 ]; do
        _pool_reap_one
    done
}

_cleanup() {
    for pid in "${_PIDS[@]+"${_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    for log in "${_LOGS[@]+"${_LOGS[@]}"}"; do
        rm -f "$log"
    done
    rm -f "$_FAIL_FILE"
}
trap _cleanup EXIT INT TERM

# ============================================================================
# Main parallel dispatch
# ============================================================================
START_TIME=$(date +%s)
COMPLETED=0

echo "============================================================"
echo "Dispatching $TOTAL jobs with $MAX_JOBS parallel workers ..."
echo "============================================================"
echo ""

for i in "${!JOB_LABELS[@]}"; do
    _pool_throttle
    label="${JOB_LABELS[$i]}"
    cmd="${JOB_CMDS[$i]}"
    gout="${JOB_OUTPUTS[$i]}"
    COMPLETED=$(( i + 1 ))
    echo "  → queued [${COMPLETED}/${TOTAL}] ${label}"
    _pool_launch "$label" "$gout" "$cmd"
done

echo ""
echo "All jobs queued — waiting for completion ..."
echo ""
_pool_drain

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

FAILED=()
if [ -s "$_FAIL_FILE" ]; then
    while IFS= read -r line; do
        FAILED+=("$line")
    done < "$_FAIL_FILE"
fi
rm -f "$_FAIL_FILE"

echo "============================================================"
echo "All Done"
echo "============================================================"
echo "  Total time:  ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Jobs run:    $TOTAL"
echo "  Skipped:     $SKIPPED"
echo "  Failed:      ${#FAILED[@]}"

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo "Failed outputs:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi

echo ""