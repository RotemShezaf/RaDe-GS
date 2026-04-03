#!/bin/bash
#
# Compute geodesic distances with batched source mesh processing
#
# This script splits the source mesh computation into batches and runs them
# in parallel. After all batches complete, it merges the results.
#
# USAGE:
#   ./compute_geodesic_batched.sh <gaussian_output> <data_root> <surface> [options]
#
# ARGUMENTS:
#   gaussian_output   Path to Gaussian splatting output folder
#   data_root         Base directory containing raw mesh data
#   surface           Surface type (Paraboloid, Saddle, HyperbolicParaboloid)
#
# OPTIONS:
#   --n_jobs N        Number of parallel jobs per batch (default: number of cpu coures-1 divided by batches)
#   --n_batches N     Number of batches to split sources into (default: 8)
#   --resolution N    Source mesh resolution (default: 8, gives N*N sources)
#   --mesh_level N    Ground truth mesh level (default: 0)
#   --use_mahalanobis Use Mahalanobis distance for mapping
#   --dry_run         Print commands without executing
#
# EXAMPLE:
#   ./compute_geodesic_batched.sh \
#       output/polynomial/Paraboloid \
#       TrainData/Polynomial/raw \
#       Paraboloid \
#       --n_jobs 4 \
#       --n_batches 8 \
#       --resolution 50

set -e

# ============================================================================
# Default values
# ============================================================================
MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

N_BATCHES=8
N_JOBS=""  # empty means auto-compute after argument parsing

RESOLUTION=8  # 8x8=64 sources, split into 8 batches of 8 sources each (0-7, 8-15, ...)
MESH_LEVEL=1
USE_MAHALANOBIS=""
USE_GAUSSIAN_MESH=""
DRY_RUN=false
VERBOSE=""

# ============================================================================
# Parse arguments
# ============================================================================
if [ $# -lt 3 ]; then
    echo "Usage: $0 <gaussian_output> <data_root> <surface> [options]"
    echo ""
    echo "Options:"
    echo "  --n_jobs N        Number of parallel jobs per batch (default: number of cpu cores-1 divided by batches)"
    echo "  --n_batches N     Number of batches (default: 8)"
    echo "  --resolution N    Source mesh resolution (default: 20)"
    echo "  --mesh_level N    Ground truth mesh level (default: 0)"
    echo "  --use_mahalanobis Use Mahalanobis distance"
    echo "  --use_gaussian_mesh Use pre-built Gaussian mesh (no interpolation)"
    echo "  --dry_run         Print commands without executing"
    echo "  --verbose         Enable verbose output"
    exit 1
fi

GAUSSIAN_OUTPUT="$1"
DATA_ROOT="$2"
SURFACE="$3"
shift 3

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --n_batches)
            N_BATCHES="$2"
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
        --use_mahalanobis)
            USE_MAHALANOBIS="--use_mahalanobis"
            shift
            ;;
        --use_gaussian_mesh)
            USE_GAUSSIAN_MESH="--use_gaussian_mesh"
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
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

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory not found: $DATA_ROOT"
    exit 1
fi

# Validate surface type
case $SURFACE in
    Paraboloid|Saddle|HyperbolicParaboloid)
        ;;
    *)
        echo "Error: Invalid surface type: $SURFACE"
        echo "Valid options: Paraboloid, Saddle, HyperbolicParaboloid"
        exit 1
        ;;
esac

# ============================================================================
# Compute batch parameters
# ============================================================================
TOTAL_SOURCES=$((RESOLUTION * RESOLUTION))
SOURCES_PER_BATCH=$(( (TOTAL_SOURCES + N_BATCHES - 1) / N_BATCHES ))

# Auto-compute N_JOBS if not explicitly set via --n_jobs.
# Each batch spawns a parent Python process + N_JOBS pool workers,
# so total = N_BATCHES * (1 + N_JOBS). Reserve one core per batch for the parent.
if [ -z "$N_JOBS" ]; then
    N_JOBS=$(( (MAX_PARALLEL - N_BATCHES) / N_BATCHES ))
    if [ $N_JOBS -lt 1 ]; then
        N_JOBS=1
    fi
fi

echo "============================================================"
echo "Geodesic Distance Computation - Batched Mode"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Gaussian output:  $GAUSSIAN_OUTPUT"
echo "  Data root:        $DATA_ROOT"
echo "  Surface:          $SURFACE"
echo "  Source resolution: ${RESOLUTION}x${RESOLUTION} = $TOTAL_SOURCES sources"
echo "  Number of batches: $N_BATCHES"
echo "  Sources per batch: ~$SOURCES_PER_BATCH"
echo "  Jobs per batch:    $N_JOBS"
echo "  Mesh level:        $MESH_LEVEL"
echo "  Use Mahalanobis:   ${USE_MAHALANOBIS:-no}"
echo "  Use Gaussian mesh: ${USE_GAUSSIAN_MESH:-no}"
echo ""

# Path to the main script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_gaussian_geodesic_distances.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Compute script not found: $COMPUTE_SCRIPT"
    exit 1
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
    if [ $SOURCE_END -gt $TOTAL_SOURCES ]; then
        SOURCE_END=$TOTAL_SOURCES
    fi
    
    # Skip empty batches
    if [ $SOURCE_START -ge $TOTAL_SOURCES ]; then
        continue
    fi
    
    echo "  Batch $((batch+1))/$N_BATCHES: sources $SOURCE_START to $SOURCE_END"
    
    CMD="python $COMPUTE_SCRIPT \
        --gaussian_output $GAUSSIAN_OUTPUT \
        --data_root $DATA_ROOT \
        --surface $SURFACE \
        --source_mesh_resolution $RESOLUTION \
        --mesh_level $MESH_LEVEL \
        --source_start $SOURCE_START \
        --source_end $SOURCE_END \
        --n_jobs $N_JOBS \
        $USE_MAHALANOBIS \
        $USE_GAUSSIAN_MESH \
        $VERBOSE"
    
    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] $CMD"
    else
        # Run in background
        $CMD &
        PIDS+=($!)
    fi
done

# ============================================================================
# Wait for all batches to complete
# ============================================================================
if [ "$DRY_RUN" = false ]; then
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
    echo "  Total time: ${ELAPSED}s"
    echo "  Failed batches: $FAILED"
    
    if [ $FAILED -gt 0 ]; then
        echo ""
        echo "Warning: Some batches failed. Check logs for details."
        echo "You may need to re-run failed source ranges manually."
    fi
    
    # ============================================================================
    # Merge results
    # ============================================================================
    echo ""
    echo "Merging partial results..."
    
    MERGE_CMD="python $COMPUTE_SCRIPT \
        --gaussian_output $GAUSSIAN_OUTPUT \
        --merge_only \
        $VERBOSE"
    
    $MERGE_CMD
    
    echo ""
    echo "Done! Results saved to:"
    echo "  $GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz"
fi
