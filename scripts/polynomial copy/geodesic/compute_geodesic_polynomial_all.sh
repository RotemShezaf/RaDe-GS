#!/bin/bash
#
# Compute geodesic distances for all polynomial surfaces and outputs
#
# This script iterates over:
#   - Multiple batch splits (source range batches)
#   - Multiple textures
#   - Multiple colmap levels
#   - Multiple output names (e.g., different training runs)
#   - Multiple surface types (Paraboloid, Saddle, HyperbolicParaboloid)
#
# It assumes the directory structure created by create_synthetic_colmap_dataset_from_mesh.py
# and that training has been performed with outputs stored in the specified output folder.
#
# USAGE:
#   ./compute_geodesic_polynomial_all.sh [options]
#
# OPTIONS:
#   --data_root DIR       Raw polynomial mesh data (default: TrainData/Polynomial/raw)
#   --output_base DIR     Base directory for Gaussian outputs (default: output/polynomial)
#   --n_jobs N            Parallel jobs per batch (default: number of cpu cores-1 divided by totsl amout of batches)
#   --n_batches N         Number of batches per surface/output (default: 8)
#   --resolution N        Source mesh resolution (default: 8)
#   --mesh_level N        Ground truth mesh level (default: 0)
#   --surfaces LIST       Comma-separated surface types (default: all)
#   --textures LIST       Comma-separated texture names (default: colors)
#   --levels LIST         Comma-separated colmap levels (default: 02,03,04)
#   --outputs LIST        Comma-separated output names (default: output)
#   --dry_run             Print commands without executing
#   --sequential          Run batches sequentially instead of parallel
#
# EXAMPLE:
#   # Process all surfaces with default settings
#   ./compute_geodesic_polynomial_all.sh
#
#   # Process only Paraboloid with specific textures and multiple outputs
#   ./compute_geodesic_polynomial_all.sh \
#       --surfaces Paraboloid \
#       --textures "colors,wood,marble" \
#       --levels "01,02,03" \
#       --outputs "run1,run2,baseline"
#
#   # Dry run to see what would be executed
#   ./compute_geodesic_polynomial_all.sh --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
DATA_ROOT="TrainData/Polynomial/raw"
OUTPUT_BASE="output/polynomial"
RAW_MESH_BASE="TrainData/Polynomial/raw"
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"

NUM_PARALLEL_OUTPUTS=4   # Number of outputs to process concurrently
N_JOBS=""                # Per-output internal parallelism (auto if empty)

MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi
RESOLUTION=8  # 8x8=64 sources
MESH_LEVEL=1
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="02,04,03"
OUTPUTS="output"  # Comma-separated list of output folder names
DRY_RUN=false
SEQUENTIAL=false
SKIP_EXISTING=false
USE_MAHALANOBIS=""
USE_GAUSSIAN_MESH="--use_gaussian_mesh"
VERBOSE=""
LIGHT_IDS="0,1,2,3,4"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --synth_data_base)
            SYNTH_DATA_BASE="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --num_parallel_outputs)
            NUM_PARALLEL_OUTPUTS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --mesh_level)
            MESH_LEVEL="$2"
            shift 2
            ;;
        --surfaces)
            SURFACES="$2"
            shift 2
            ;;
        --textures)
            TEXTURES="$2"
            shift 2
            ;;
        --levels)
            LEVELS="$2"
            shift 2
            ;;
        --outputs)
            OUTPUTS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --use_mahalanobis)
            USE_MAHALANOBIS="--use_mahalanobis"
            shift
            ;;
        --use_gaussian_mesh)
            USE_GAUSSIAN_MESH="--use_gaussian_mesh"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --light_ids)
            LIGHT_IDS="$2"
            shift 2
            ;;
        --surfaces)
            SURFACES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR       Raw polynomial mesh data (default: TrainData/Polynomial/raw)"
            echo "  --output_base DIR     Base for Gaussian outputs (default: output/polynomial)"
            echo "  --synth_data_base DIR Synthetic COLMAP data (default: TrainData/Polynomial/SyntheticColmapData)"
            echo "  --n_jobs N            Parallel jobs per output (default: auto)"
            echo "  --num_parallel_outputs N  Number of outputs to run concurrently (default: 4)"
            echo "  --resolution N        Source mesh resolution (default: 8, gives NxN sources)"
            echo "  --mesh_level N        Ground truth mesh level (default: 0)"
            echo "  --surfaces LIST       Comma-separated surfaces (default: all)"
            echo "  --textures LIST       Comma-separated textures (default: colors)"
            echo "  --levels LIST         Comma-separated levels (default: 02,03,04)"
            echo "  --outputs LIST        Comma-separated output folder names (default: output)"
            echo "  --dry_run             Print commands without executing"
            echo "  --sequential          Run batches sequentially"
            echo "  --use_mahalanobis     Use Mahalanobis distance"
            echo "  --use_gaussian_mesh   Use pre-built Gaussian mesh (no interpolation)"
            echo "  --skip_existing       Skip outputs that already have gt_geodesic.npz"
            echo "  --verbose             Enable verbose output"
            echo "  --light_ids LIST      Comma-separated light IDs to process (default: 0,1,2,3,4)"
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

# Handle light IDs: empty string means single iteration without light_id
if [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Path to scripts
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/GenerateData/compute_gaussian_geodesic_distances.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Compute script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Compute job parameters
# ============================================================================
TOTAL_SOURCES=$((RESOLUTION * RESOLUTION))
# Calculate N_JOBS (per-output internal parallelism) only if not explicitly set.
# Divide available CPUs evenly among concurrent outputs.
if [ -z "$N_JOBS" ]; then
    N_JOBS=$(( MAX_PARALLEL / NUM_PARALLEL_OUTPUTS ))
    if [ $N_JOBS -lt 1 ]; then
        N_JOBS=1
    fi
fi
# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Geodesic Distance Computation - All Polynomial Surfaces"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Data root:         $DATA_ROOT"
echo "  Output base:       $OUTPUT_BASE"
echo "  Synth data base:   $SYNTH_DATA_BASE"
echo "  Textures:          ${TEXTURE_ARRAY[*]}"
echo "  Levels:            ${LEVEL_ARRAY[*]}"
echo "  Outputs:           ${OUTPUT_ARRAY[*]}"
echo "  Light IDs:         ${LIGHT_ID_ARRAY[*]:-<standard lighting>}"
echo "  Surfaces:          ${SURFACE_ARRAY[*]}"
echo "  Source resolution: ${RESOLUTION}x${RESOLUTION} = $TOTAL_SOURCES sources"
echo "  Parallel outputs: $NUM_PARALLEL_OUTPUTS"
echo "  Jobs per output:  $N_JOBS"
echo "  Mesh level:        $MESH_LEVEL"
echo "  Sequential:        $SEQUENTIAL"
echo "  Skip existing:     $SKIP_EXISTING"
echo "  Dry run:           $DRY_RUN"
echo ""

# ============================================================================
# Pre-scan: build work list
# ============================================================================
declare -a JOB_LABELS=()
declare -a JOB_CMDS=()
declare -a JOB_MERGE_CMDS=()
SKIPPED=0

for texture in "${TEXTURE_ARRAY[@]}"; do
    for level in "${LEVEL_ARRAY[@]}"; do
        for light_id in "${LIGHT_ID_ARRAY[@]}"; do
            for output_name in "${OUTPUT_ARRAY[@]}"; do
                for surface in "${SURFACE_ARRAY[@]}"; do
                    DATASET_PATH="$DATA_ROOT/$surface"
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

                    if [ "$SKIP_EXISTING" = true ] && [ -f "$GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --data_root $DATA_ROOT \
                        --surface $surface \
                        --source_mesh_resolution $RESOLUTION \
                        --mesh_level $MESH_LEVEL \
                        --geodesic_method vtp \
                        --n_jobs $N_JOBS \
                        $USE_MAHALANOBIS \
                        $USE_GAUSSIAN_MESH \
                        $VERBOSE"

                    MERGE_CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --merge_only \
                        $VERBOSE"

                    JOB_LABELS+=("$LABEL")
                    JOB_CMDS+=("$CMD")
                    JOB_MERGE_CMDS+=("$MERGE_CMD")
                done
            done
        done
    done
done

TOTAL=${#JOB_LABELS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to do — all outputs skipped or not found."
    exit 0
fi

echo "  Total outputs:   $TOTAL  (skipped: $SKIPPED)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "============================================================"
    echo "DRY RUN — commands that would be executed:"
    echo "============================================================"
    for i in "${!JOB_LABELS[@]}"; do
        echo ""
        echo "  [$(( i + 1 ))/$TOTAL] ${JOB_LABELS[$i]}"
        echo "  ${JOB_CMDS[$i]}"
        echo "  merge: ${JOB_MERGE_CMDS[$i]}"
    done
    exit 0
fi

# ============================================================================
# Main processing — parallelize across outputs
# ============================================================================
START_TIME=$(date +%s)
CURRENT_OUTPUT=0
FAILED_OUTPUTS=()

# Pool state
declare -a _PIDS=()
declare -a _LABELS=()
declare -a _MERGE_CMDS=()

_reap_one() {
    # Wait for any one background job to finish
    wait -n 2>/dev/null || true
    local new_pids=() new_labels=() new_merges=()
    local reaped=0
    local _j
    for _j in "${!_PIDS[@]}"; do
        local pid="${_PIDS[$_j]}"
        if [ "$reaped" -eq 0 ] && ! kill -0 "$pid" 2>/dev/null; then
            local rc=0
            wait "$pid" 2>/dev/null || rc=$?
            local label="${_LABELS[$_j]}"
            local merge_cmd="${_MERGE_CMDS[$_j]}"
            if [ "$rc" -eq 0 ]; then
                echo "  ✓ $label — compute done, merging ..."
                if ! eval "$merge_cmd"; then
                    echo "  ✗ $label — merge FAILED"
                    FAILED_OUTPUTS+=("$label (merge failed)")
                fi
            else
                echo "  ✗ $label — compute FAILED (exit $rc)"
                FAILED_OUTPUTS+=("$label (compute failed)")
            fi
            reaped=1
        else
            new_pids+=("$pid")
            new_labels+=("${_LABELS[$_j]}")
            new_merges+=("${_MERGE_CMDS[$_j]}")
        fi
    done
    _PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    _LABELS=("${new_labels[@]+"${new_labels[@]}"}")
    _MERGE_CMDS=("${new_merges[@]+"${new_merges[@]}"}")
}

_throttle() {
    while [ "${#_PIDS[@]}" -ge "$NUM_PARALLEL_OUTPUTS" ]; do
        _reap_one
    done
}

_drain() {
    while [ "${#_PIDS[@]}" -gt 0 ]; do
        _reap_one
    done
}

_cleanup() {
    for pid in "${_PIDS[@]+"${_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap _cleanup EXIT INT TERM

echo "============================================================"
echo "Dispatching $TOTAL outputs ($NUM_PARALLEL_OUTPUTS concurrent)"
echo "============================================================"
echo ""

for i in "${!JOB_LABELS[@]}"; do
    _throttle
    label="${JOB_LABELS[$i]}"
    cmd="${JOB_CMDS[$i]}"

    CURRENT_OUTPUT=$(( i + 1 ))
    if [ "$SEQUENTIAL" = true ]; then
        echo "  → [$CURRENT_OUTPUT/$TOTAL] $label"
        if eval "$cmd"; then
            echo "  ✓ $label — compute done, merging ..."
            if ! eval "${JOB_MERGE_CMDS[$i]}"; then
                echo "  ✗ $label — merge FAILED"
                FAILED_OUTPUTS+=("$label (merge failed)")
            fi
        else
            echo "  ✗ $label — compute FAILED"
            FAILED_OUTPUTS+=("$label (compute failed)")
        fi
    else
        echo "  → queued [$CURRENT_OUTPUT/$TOTAL] $label"
        eval "$cmd" &
        _PIDS+=("$!")
        _LABELS+=("$label")
        _MERGE_CMDS+=("${JOB_MERGE_CMDS[$i]}")
    fi
done

if [ "$SEQUENTIAL" = false ]; then
    echo ""
    echo "All jobs queued — waiting for completion ..."
    _drain
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "All Processing Complete"
echo "============================================================"
echo "  Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Total outputs processed: $TOTAL"
echo "  Skipped:     $SKIPPED"
echo "  Failed outputs: ${#FAILED_OUTPUTS[@]}"

if [ ${#FAILED_OUTPUTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed outputs:"
    for failed in "${FAILED_OUTPUTS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
echo "Done!"
