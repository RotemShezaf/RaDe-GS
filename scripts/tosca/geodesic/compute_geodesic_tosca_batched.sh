#!/bin/bash
#
# Compute geodesic distances for a single TOSCA shape with batched source processing
#
# This script splits the source computation into batches and runs them
# in parallel. After all batches complete, it merges the results.
# It calls compute_gaussian_geodesic_distances_tosca.py (not the polynomial version).
#
# USAGE:
#   ./scripts/compute_geodesic_tosca_batched.sh <gaussian_output> <shape> [options]
#
# ARGUMENTS:
#   gaussian_output   Path to Gaussian splatting output folder
#   shape             TOSCA shape name (e.g., cat0, dog0, horse0)
#
# OPTIONS:
#   --data_root DIR     Preprocessed TOSCA data root (default: TrainData/TOSCA/processed)
#   --mesh_type TYPE    gt or reconstructed (default: gt)
#   --mesh_resolution R high_res or low_res (default: high_res)
#   --mesh_path PATH    Explicit mesh PLY path (overrides --mesh_type/--shape)
#   --num_sources N     Number of source vertices to sample (default: 200)
#   --n_batches N       Number of batches to split sources into (default: 8)
#   --n_jobs N          Number of parallel jobs per batch (default: auto)
#   --proximity_factor F  Proximity multiplier for candidate filtering (default: 3.0)
#   --iteration N       Gaussian training iteration (default: highest)
#   --dry_run           Print commands without executing
#   --sequential        Run batches sequentially instead of in parallel
#   --verbose           Enable verbose output
#
# EXAMPLES:
#   # Process cat0 with default settings
#   ./scripts/compute_geodesic_tosca_batched.sh \
#       TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
#       cat0
#
#   # Dry run with 400 sources in 16 batches
#   ./scripts/compute_geodesic_tosca_batched.sh \
#       TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
#       cat0 \
#       --num_sources 400 --n_batches 16 --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

N_BATCHES=8
DATA_ROOT="TrainData/TOSCA/processed"
MESH_TYPE="reconstructed"
MESH_RESOLUTION="high_res"
MESH_PATH=""
NUM_SOURCES=32
PROXIMITY_FACTOR=""
ITERATION=""
DRY_RUN=false
SEQUENTIAL=false
VERBOSE=""
SEED=42

# ============================================================================
# Parse arguments
# ============================================================================
if [ $# -lt 2 ]; then
    echo "Usage: $0 <gaussian_output> <shape> [options]"
    echo ""
    echo "Arguments:"
    echo "  gaussian_output   Path to Gaussian splatting output folder"
    echo "  shape             TOSCA shape name (e.g., cat0, dog0, horse0)"
    echo ""
    echo "Options:"
    echo "  --data_root DIR     Preprocessed TOSCA data root (default: TrainData/TOSCA/processed)"
    echo "  --mesh_type TYPE    gt or reconstructed (default: gt)"
    echo "  --mesh_resolution R high_res or low_res (default: high_res)"
    echo "  --mesh_path PATH    Explicit mesh PLY path (overrides --mesh_type/--shape)"
    echo "  --num_sources N     Number of source vertices (default: 200)"
    echo "  --n_batches N       Number of batches (default: 8)"
    echo "  --n_jobs N          Parallel jobs per batch (default: auto)"
    echo "  --proximity_factor F Proximity multiplier (default: 3.0)"
    echo "  --iteration N       Gaussian training iteration (default: highest)"
    echo "  --dry_run           Print commands without executing"
    echo "  --sequential        Run batches sequentially"
    echo "  --verbose           Enable verbose output"
    echo ""
    echo "Use --help for this message."
    exit 1
fi

GAUSSIAN_OUTPUT="$1"
SHAPE="$2"
shift 2

N_JOBS_OVERRIDE=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --mesh_type)
            MESH_TYPE="$2"
            shift 2
            ;;
        --mesh_resolution)
            MESH_RESOLUTION="$2"
            shift 2
            ;;
        --mesh_path)
            MESH_PATH="$2"
            shift 2
            ;;
        --num_sources)
            NUM_SOURCES="$2"
            shift 2
            ;;
        --n_batches)
            N_BATCHES="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS_OVERRIDE="$2"
            shift 2
            ;;
        --proximity_factor)
            PROXIMITY_FACTOR="--proximity_factor $2"
            shift 2
            ;;
        --iteration)
            ITERATION="--iteration $2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate inputs
# ============================================================================
if [ ! -d "$GAUSSIAN_OUTPUT" ]; then
    echo "Error: Gaussian output directory not found: $GAUSSIAN_OUTPUT"
    exit 1
fi

if [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
    echo "Error: No point_cloud directory found in: $GAUSSIAN_OUTPUT"
    exit 1
fi

if [ "$MESH_TYPE" = "gt" ] && [ -z "$MESH_PATH" ]; then
    if [ ! -d "$DATA_ROOT/$SHAPE" ]; then
        echo "Error: TOSCA shape directory not found: $DATA_ROOT/$SHAPE"
        exit 1
    fi
    if ! ls "$DATA_ROOT/$SHAPE"/mesh_${MESH_RESOLUTION}_*.ply 2>/dev/null | head -1 | grep -q .; then
        echo "Error: No ground truth mesh found at $DATA_ROOT/$SHAPE/mesh_${MESH_RESOLUTION}_*.ply"
        echo "       Run preprocess_tosca.py first."
        exit 1
    fi
fi

# Auto-detect reconstructed mesh if mesh_type=reconstructed and no explicit path
if [ "$MESH_TYPE" = "reconstructed" ] && [ -z "$MESH_PATH" ]; then
    if [ -f "$GAUSSIAN_OUTPUT/recon.ply" ]; then
        MESH_PATH="$GAUSSIAN_OUTPUT/recon.ply"
        echo "Auto-detected reconstructed mesh: $MESH_PATH"
    else
        echo "Error: No recon.ply found in $GAUSSIAN_OUTPUT"
        echo "       Provide --mesh_path or run mesh extraction first."
        exit 1
    fi
fi

# ============================================================================
# Compute batch parameters
# ============================================================================
SOURCES_PER_BATCH=$(( (NUM_SOURCES + N_BATCHES - 1) / N_BATCHES ))

# N_JOBS: each batch spawns a parent Python process + N_JOBS pool workers,
# so total = N_BATCHES * (1 + N_JOBS). Reserve one core per batch for the parent.
if [ -n "$N_JOBS_OVERRIDE" ]; then
    N_JOBS=$N_JOBS_OVERRIDE
else
    N_JOBS=$(( (MAX_PARALLEL - N_BATCHES) / N_BATCHES ))
    if [ $N_JOBS -lt 1 ]; then
        N_JOBS=1
    fi
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "TOSCA Geodesic Distance Computation - Batched Mode"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Gaussian output:  $GAUSSIAN_OUTPUT"
echo "  Shape:            $SHAPE"
echo "  Data root:        $DATA_ROOT"
echo "  Mesh type:        $MESH_TYPE"
echo "  Mesh resolution:  $MESH_RESOLUTION"
if [ -n "$MESH_PATH" ]; then
echo "  Mesh path:        $MESH_PATH"
fi
echo "  Num sources:      $NUM_SOURCES"
echo "  Number of batches: $N_BATCHES"
echo "  Sources per batch: ~$SOURCES_PER_BATCH"
echo "  Jobs per batch:    $N_JOBS"
echo "  Sequential:        $SEQUENTIAL"
echo "  Seed:              $SEED"
echo ""

# Path to the TOSCA compute script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_gaussian_geodesic_distances_tosca.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: TOSCA compute script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# Build common arguments
COMMON_ARGS="--gaussian_output $GAUSSIAN_OUTPUT \
    --shape $SHAPE \
    --data_root $DATA_ROOT \
    --mesh_type $MESH_TYPE \
    --mesh_resolution $MESH_RESOLUTION \
    --num_sources $NUM_SOURCES \
    --geodesic_method mmp \
    --embed_gaussians \
    --seed $SEED \
    --n_jobs $N_JOBS \
    $PROXIMITY_FACTOR \
    $ITERATION \
    $VERBOSE"

if [ -n "$MESH_PATH" ]; then
    COMMON_ARGS="$COMMON_ARGS --mesh_path $MESH_PATH"
fi

# ============================================================================
# Run batches
# ============================================================================
PIDS=()
START_TIME=$(date +%s)

echo "Starting $N_BATCHES batches..."
echo ""

for ((batch=0; batch<N_BATCHES; batch++)); do
    SOURCE_START=$((batch * SOURCES_PER_BATCH))
    SOURCE_END=$(( (batch + 1) * SOURCES_PER_BATCH ))

    # Clamp to total sources
    if [ $SOURCE_END -gt $NUM_SOURCES ]; then
        SOURCE_END=$NUM_SOURCES
    fi

    # Skip empty batches
    if [ $SOURCE_START -ge $NUM_SOURCES ]; then
        continue
    fi

    echo "  Batch $((batch+1))/$N_BATCHES: sources $SOURCE_START to $SOURCE_END"

    CMD="python $COMPUTE_SCRIPT \
        $COMMON_ARGS \
        --source_start $SOURCE_START \
        --source_end $SOURCE_END"

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] $CMD"
    elif [ "$SEQUENTIAL" = true ]; then
        echo "    Running sequentially..."
        if ! eval $CMD; then
            echo "    Warning: Batch $((batch+1)) failed"
        fi
    else
        # Run in background
        eval $CMD &
        PIDS+=($!)
    fi
done

# ============================================================================
# Wait for all batches to complete
# ============================================================================
if [ "$DRY_RUN" = false ] && [ "$SEQUENTIAL" = false ]; then
    echo ""
    echo "Waiting for all batches to complete..."

    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait $pid; then
            echo "  Warning: Batch with PID $pid failed"
            FAILED=$((FAILED + 1))
        fi
    done

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "============================================================"
    echo "Batch Processing Complete"
    echo "============================================================"
    echo "  Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
    echo "  Failed batches: $FAILED"

    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "Warning: Some batches failed. Check logs for details."
        echo "You may need to re-run failed source ranges manually."
    fi
fi

# ============================================================================
# Merge results
# ============================================================================
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Merging partial results..."

    MERGE_CMD="python $COMPUTE_SCRIPT \
        --gaussian_output $GAUSSIAN_OUTPUT \
        --merge_only \
        $VERBOSE"

    if eval $MERGE_CMD; then
        echo ""
        echo "Done! Results saved to:"
        echo "  $GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz"
    else
        echo ""
        echo "Warning: Merge failed. Partial results may still be in:"
        echo "  $GAUSSIAN_OUTPUT/geodesic_distance/gt_partial/"
    fi
fi
