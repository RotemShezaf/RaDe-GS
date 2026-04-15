#!/bin/bash
#
# Autotune geodesic meshes for ALL polynomial Gaussian outputs.
#
# For each output, tries 6 parameter configurations in parallel, picks
# the best by rank (balanced multi-factor), d3, or composite score, and saves all
# variants as backups so you can switch later.
#
# This is the autotune version of build_geodesic_mesh_polynomial_all.sh.
#
# USAGE:
#   ./build_geodesic_mesh_polynomial_all_autotune.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR   Synthetic COLMAP data dir
#   --surfaces LIST         Comma-separated surface types
#   --textures LIST         Comma-separated texture names
#   --levels LIST           Comma-separated colmap levels
#   --outputs LIST          Comma-separated output folder names
#   --light_ids LIST        Comma-separated light IDs
#   --metric rank|d3|composite   Selection metric (default: rank)
#   --workers_per_output N  Workers per autotune (default: 6)
#   --jobs N                Concurrent autotune runs (default: nproc/6)
#   --skip_existing         Skip outputs already autotuned
#   --dry_run               Print commands without executing
#   --verbose               Enable verbose output
#
# EXAMPLES:
#   # Autotune all surfaces:
#   ./build_geodesic_mesh_polynomial_all_autotune.sh
#
#   # Only HyperbolicParaboloid:
#   ./build_geodesic_mesh_polynomial_all_autotune.sh --surfaces HyperbolicParaboloid
#
#   # Dry run:
#   ./build_geodesic_mesh_polynomial_all_autotune.sh --dry_run

# Do NOT use set -e — it interferes with background-job exit-code collection.
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
METRIC="rank"
WORKERS_PER_OUTPUT="6"
SEED="42"
DRY_RUN=false
VERBOSE=""
SKIP_EXISTING=""

# Parallelism: number of concurrent autotune runs.
# Each autotune uses WORKERS_PER_OUTPUT CPUs, so total = MAX_JOBS × WORKERS_PER_OUTPUT.
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 6)
MAX_JOBS=$(( NCPU / 6 ))
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base)      SYNTH_DATA_BASE="$2";      shift 2 ;;
        --surfaces)             SURFACES="$2";             shift 2 ;;
        --textures)             TEXTURES="$2";             shift 2 ;;
        --levels)               LEVELS="$2";               shift 2 ;;
        --outputs)              OUTPUTS="$2";              shift 2 ;;
        --light_ids)            LIGHT_IDS="$2";            shift 2 ;;
        --metric)               METRIC="$2";               shift 2 ;;
        --workers_per_output)   WORKERS_PER_OUTPUT="$2";   shift 2 ;;
        --jobs|-j)              MAX_JOBS="$2";             shift 2 ;;
        --skip_existing)        SKIP_EXISTING="--skip_existing"; shift ;;
        --dry_run)              DRY_RUN=true;              shift   ;;
        --verbose)              VERBOSE="--verbose";       shift   ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --synth_data_base DIR     Synthetic COLMAP data (default: TrainData/Polynomial/SyntheticColmapData)"
            echo "  --surfaces LIST           Comma-separated surfaces (default: all three)"
            echo "  --textures LIST           Comma-separated textures (default: blue)"
            echo "  --levels LIST             Comma-separated levels (default: 02,03,04)"
            echo "  --outputs LIST            Comma-separated output names (default: output)"
            echo "  --light_ids LIST          Comma-separated light IDs (default: 0,1,2,3,4)"
            echo "  --metric rank|d3|composite Selection metric (default: rank)"
            echo "  --workers_per_output N    Parallel workers per autotune (default: 6)"
            echo "  --jobs N                  Concurrent autotune runs (default: nproc/6 = $MAX_JOBS)"
            echo "  --skip_existing           Skip already-autotuned outputs"
            echo "  --dry_run                 Print commands without executing"
            echo "  --verbose                 Verbose output"
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
# Locate the autotune script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AUTOTUNE_SCRIPT="$SCRIPT_DIR/GenerateData/autotune_geodesic_mesh.py"

if [ ! -f "$AUTOTUNE_SCRIPT" ]; then
    echo "Error: Script not found: $AUTOTUNE_SCRIPT"
    exit 1
fi

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

                    CMD="python $AUTOTUNE_SCRIPT \
                        --output_dir $GAUSSIAN_OUTPUT \
                        --surface $surface \
                        --workers $WORKERS_PER_OUTPUT \
                        --metric $METRIC \
                        $SKIP_EXISTING"

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
echo "Autotune Geodesic Meshes — All Polynomial Surfaces"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Synth data base:     $SYNTH_DATA_BASE"
echo "  Textures:            ${TEXTURE_ARRAY[*]}"
echo "  Levels:              ${LEVEL_ARRAY[*]}"
echo "  Outputs:             ${OUTPUT_ARRAY[*]}"
echo "  Light IDs:           ${LIGHT_ID_ARRAY[*]:-<standard>}"
echo "  Surfaces:            ${SURFACE_ARRAY[*]}"
echo "  Metric:              $METRIC"
echo "  Workers per output:  $WORKERS_PER_OUTPUT"
echo "  Concurrent jobs:     $MAX_JOBS"
echo "  Total CPUs used:     $(( MAX_JOBS * WORKERS_PER_OUTPUT ))"
echo "  Skip existing:       ${SKIP_EXISTING:-no}"
echo "  Dry run:             $DRY_RUN"
echo "  Total outputs:       $TOTAL  (skipped: $SKIPPED)"
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
# Parallel pool helpers (same pattern as build_geodesic_mesh_polynomial_all.sh)
# ============================================================================
_FAIL_FILE=$(mktemp /tmp/autotune_failures_XXXXXX)

declare -a _PIDS=()
declare -a _LABELS=()
declare -a _LOGS=()

_pool_launch() {
    local label="$1"
    local gout="$2"
    local cmd="$3"
    local log
    log=$(mktemp /tmp/autotune_job_XXXXXX.log)

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
                    echo "  ✓ Autotune complete"
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
echo "Dispatching $TOTAL autotune jobs ($MAX_JOBS concurrent) ..."
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
echo "To list backups for a specific output:"
echo "  python $AUTOTUNE_SCRIPT --output_dir <path> --list_backups"
echo ""
echo "To switch active mesh to a different backup:"
echo "  python $AUTOTUNE_SCRIPT --output_dir <path> --switch <config_name>"
echo ""
