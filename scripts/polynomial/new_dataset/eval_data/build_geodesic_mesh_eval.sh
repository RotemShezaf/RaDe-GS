#!/bin/bash
#
# Build geodesic meshes for level 05 polynomial Gaussian outputs (evaluation).
#
# Mirrors build_geodesic_mesh_polynomial_all.sh but configured for level 05
# evaluation data. These meshes are used for computing geodesic distances
# on Gaussians that were NOT used for training (levels 02, 03, 04).
#
# USAGE:
#   bash scripts/polynomial/new_dataset/eval_data/build_geodesic_mesh_eval.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR   Synthetic COLMAP data dir (default: TrainData/Polynomial/SyntheticColmapData)
#   --surfaces LIST         Comma-separated surface types (default: Paraboloid,Saddle,HyperbolicParaboloid)
#   --textures LIST         Comma-separated texture names (default: blue)
#   --levels LIST           Comma-separated colmap levels (default: 05)
#   --outputs LIST          Comma-separated output folder names (default: output)
#   --light_ids LIST        Comma-separated light IDs (default: 0,1,2,3,4)
#   --n_points N            Add N surface samples per mesh (default: 400000)
#   --curvature_adaptive    Use curvature-adaptive sampling (default: enabled)
#   --no_curvature          Use uniform sampling instead of curvature-adaptive
#   --curvature_alpha F     Curvature sensitivity factor (default: 1.8)
#   --seed N                RNG seed (default: 42)
#   --jobs N                Parallel workers (default: nproc)
#   --dry_run               Print commands without executing
#   --verbose               Enable verbose output
#
# EXAMPLES:
#   # Build meshes for all level 05 evaluation outputs
#   bash scripts/polynomial/new_dataset/eval_data/build_geodesic_mesh_eval.sh
#
#   # Only Paraboloid
#   bash scripts/polynomial/new_dataset/eval_data/build_geodesic_mesh_eval.sh --surfaces Paraboloid
#
#   # Dry run
#   bash scripts/polynomial/new_dataset/eval_data/build_geodesic_mesh_eval.sh --dry_run

# Do NOT use set -e here — it interferes with background-job exit-code collection.
set -uo pipefail

# ============================================================================
# Default values
# ============================================================================
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="05"
OUTPUTS="output"
LIGHT_IDS="0,1,2,3,4"
N_POINTS="400000"
MIN_RADIUS=""
CURVATURE_ADAPTIVE=true
CURVATURE_ALPHA="1.8"
SEED="42"
DRY_RUN=false
VERBOSE=""
REFINE="--refine"
REFINE_WARMUP_ITERATIONS="8"
MESH_METHOD="grid"

MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

declare -A MAX_EDGE_LENGTH_MAP=(
    ["05"]="0.0005"
)
declare -A MAX_EDGE_LENGTH_GAUSSIANS_MAP=(
    ["05"]="0.0002"
)
declare -A MAX_AREA_FACTOR_MAP=(
    ["05"]="2"
)
declare -A MAX_AREA_FACTOR_GAUSSIANS_MAP=(
    ["05"]="1.5"
)

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base)  SYNTH_DATA_BASE="$2"; shift 2 ;;
        --surfaces)         SURFACES="$2";         shift 2 ;;
        --textures)         TEXTURES="$2";         shift 2 ;;
        --levels)           LEVELS="$2";           shift 2 ;;
        --outputs)          OUTPUTS="$2";          shift 2 ;;
        --light_ids)        LIGHT_IDS="$2";        shift 2 ;;
        --n_points)         N_POINTS="$2";         shift 2 ;;
        --min_radius)       MIN_RADIUS="$2";       shift 2 ;;
        --curvature_adaptive) CURVATURE_ADAPTIVE=true; shift ;;
        --no_curvature)     CURVATURE_ADAPTIVE=false; shift ;;
        --curvature_alpha)  CURVATURE_ALPHA="$2";  shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --jobs|-j)          MAX_JOBS="$2";         shift 2 ;;
        --dry_run)          DRY_RUN=true;          shift   ;;
        --verbose)          VERBOSE="--verbose";   shift   ;;
        --help|-h)
            sed -n '2,/^set -uo/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/GenerateData/compute_geodesic_mesh_for_gaussians.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# Pre-scan: build work list
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

                    LEVEL_MAX_EDGE="${MAX_EDGE_LENGTH_MAP[$level]:-0.005}"
                    LEVEL_MAX_EDGE_GAUSS="${MAX_EDGE_LENGTH_GAUSSIANS_MAP[$level]:-0.004}"
                    LEVEL_MAX_AREA="${MAX_AREA_FACTOR_MAP[$level]:-3.0}"
                    LEVEL_MAX_AREA_GAUSS="${MAX_AREA_FACTOR_GAUSSIANS_MAP[$level]:-2.5}"

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
                        CMD="$CMD --curvature_adaptive --curvature_alpha $CURVATURE_ALPHA --max_edge_length $LEVEL_MAX_EDGE $REFINE --mesh_method $MESH_METHOD"
                        if [ -n "$REFINE" ]; then
                            CMD="$CMD --refine_gauss_max_edge_length $LEVEL_MAX_EDGE_GAUSS"
                            CMD="$CMD --refine_max_area_factor $LEVEL_MAX_AREA"
                            CMD="$CMD --refine_gauss_max_area_factor $LEVEL_MAX_AREA_GAUSS"
                            CMD="$CMD --refine_warmup_iterations $REFINE_WARMUP_ITERATIONS"
                        fi
                    fi

                    if [ -n "$VERBOSE" ]; then
                        CMD="$CMD $VERBOSE"
                    fi

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
echo "Build Geodesic Meshes — Evaluation (Level 05)"
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
echo "  Curvature adapt: $CURVATURE_ADAPTIVE"
echo "  Curvature alpha: $CURVATURE_ALPHA"
echo "  Seed:            $SEED"
echo "  Parallel jobs:   $MAX_JOBS"
echo "  Dry run:         $DRY_RUN"
echo "  Total outputs:   $TOTAL  (skipped: $SKIPPED)"
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
_FAIL_FILE=$(mktemp /tmp/geodesic_eval_failures_XXXXXX)

declare -a _PIDS=()
declare -a _LABELS=()
declare -a _LOGS=()

_pool_launch() {
    local label="$1"
    local gout="$2"
    local cmd="$3"
    local log
    log=$(mktemp /tmp/geodesic_eval_job_XXXXXX.log)

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
                    echo "  Mesh saved"
                else
                    echo "  FAILED (exit $rc)"
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
    echo "  -> queued [${COMPLETED}/${TOTAL}] ${label}"
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
