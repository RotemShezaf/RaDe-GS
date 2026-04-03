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

N_BATCHES=8
N_JOBS=""  # empty means auto-compute later

MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi
RESOLUTION=8  # 8x8=64 sources, split into 8 batches of 8 sources each (0-7, 8-15, ...)
MESH_LEVEL=1
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="02,04,03"
OUTPUTS="output"  # Comma-separated list of output folder names
DRY_RUN=false
SEQUENTIAL=false
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
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR       Raw polynomial mesh data (default: TrainData/Polynomial/raw)"
            echo "  --output_base DIR     Base for Gaussian outputs (default: output/polynomial)"
            echo "  --synth_data_base DIR Synthetic COLMAP data (default: TrainData/Polynomial/SyntheticColmapData)"
            echo "  --n_jobs N            Parallel jobs per batch (default: 4)"
            echo "  --n_batches N         Number of batches (default: 8)"
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
# Compute batch parameters
# ============================================================================
TOTAL_SOURCES=$((RESOLUTION * RESOLUTION))
SOURCES_PER_BATCH=$(( (TOTAL_SOURCES + N_BATCHES - 1) / N_BATCHES ))
# Calculate N_JOBS only if not explicitly set by the user via --n_jobs.
# Each batch launches a parent Python process that spawns N_JOBS pool workers,
# so total processes = N_BATCHES * (1 + N_JOBS). Reserve one core per batch
# for the parent process to avoid oversubscribing CPUs.
if [ -z "$N_JOBS" ]; then
    N_JOBS=$(( (MAX_PARALLEL - N_BATCHES) / N_BATCHES ))
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
echo "  Number of batches: $N_BATCHES"
echo "  Sources per batch: ~$SOURCES_PER_BATCH"
echo "  Jobs per batch:    $N_JOBS"
echo "  Mesh level:        $MESH_LEVEL"
echo "  Sequential:        $SEQUENTIAL"
echo "  Dry run:           $DRY_RUN"
echo ""

# ============================================================================
# Count total work
# ============================================================================
TOTAL_OUTPUTS=0
for texture in "${TEXTURE_ARRAY[@]}"; do
    for level in "${LEVEL_ARRAY[@]}"; do
        for light_id in "${LIGHT_ID_ARRAY[@]}"; do
            for output_name in "${OUTPUT_ARRAY[@]}"; do
                for surface in "${SURFACE_ARRAY[@]}"; do
                    TOTAL_OUTPUTS=$((TOTAL_OUTPUTS + 1))
                done
            done
        done
    done
done

echo "Total outputs to process: $TOTAL_OUTPUTS"
echo ""

# ============================================================================
# Main processing loop
# Loop order: texture -> level -> output_name -> surface (surface is innermost)
# ============================================================================
START_TIME=$(date +%s)
CURRENT_OUTPUT=0
FAILED_OUTPUTS=()

for texture in "${TEXTURE_ARRAY[@]}"; do
    for level in "${LEVEL_ARRAY[@]}"; do
        for light_id in "${LIGHT_ID_ARRAY[@]}"; do
            for output_name in "${OUTPUT_ARRAY[@]}"; do
                for surface in "${SURFACE_ARRAY[@]}"; do
                    CURRENT_OUTPUT=$((CURRENT_OUTPUT + 1))
                    
                    # Construct paths
                    # Mesh dataset: {data_root}/{surface}
                    # Gaussian output: {synth_data_base}/{texture}_texture/{surface}/level_{level}/light_{light_id}/{output_name}
                    #                or {synth_data_base}/{texture}_texture/{surface}/level_{level}/default_light/{output_name} if no light_id
                    
                    DATASET_PATH="$DATA_ROOT/$surface"
                    if [ -n "$light_id" ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/light_${light_id}/$output_name"
                    else
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/default_light/$output_name"
                    fi
                
                    echo "============================================================"
                    echo "[$CURRENT_OUTPUT/$TOTAL_OUTPUTS] Processing:"
                    echo "  Texture:  $texture"
                    echo "  Level:    $level"
                    if [ -n "$light_id" ]; then
                        echo "  Light ID: $light_id"
                    fi
                    echo "  Output:   $output_name"
                    echo "  Surface:  $surface"
                    echo "  Dataset:  $DATASET_PATH"
                    echo "  Gaussian output: $GAUSSIAN_OUTPUT"
                    echo "============================================================"
                
                # Check if Gaussian output exists
                if [ ! -d "$GAUSSIAN_OUTPUT" ]; then
                    echo "  Warning: Gaussian output not found, skipping..."
                    echo "           Expected: $GAUSSIAN_OUTPUT"
                    FAILED_OUTPUTS+=("$texture/$surface/level_$level/$output_name (missing output)")
                    continue
                fi
                
                # Check if point cloud exists
                if [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
                    echo "  Warning: No point_cloud directory found, skipping..."
                    FAILED_OUTPUTS+=("$texture/$surface/level_$level/$output_name (no point cloud)")
                    continue
                fi
                
                # ================================================================
                # Run batched computation
                # ================================================================
                PIDS=()
                
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
                    
                    echo "  Batch $((batch+1))/$N_BATCHES: sources $SOURCE_START-$SOURCE_END"
                    
                    CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --data_root $DATA_ROOT \
                        --surface $surface \
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
                    elif [ "$SEQUENTIAL" = true ]; then
                        # Run sequentially
                        if ! $CMD; then
                            echo "    Warning: Batch $((batch+1)) failed"
                        fi
                    else
                        # Run in background
                        $CMD &
                        PIDS+=($!)
                    fi
                done
                
                # Wait for all batches of this output to complete
                if [ "$DRY_RUN" = false ] && [ "$SEQUENTIAL" = false ]; then
                    echo ""
                    echo "  Waiting for batches to complete..."
                    
                    BATCH_FAILED=0
                    for pid in "${PIDS[@]}"; do
                        if ! wait $pid; then
                            BATCH_FAILED=$((BATCH_FAILED + 1))
                        fi
                    done
                    
                    if [ $BATCH_FAILED -gt 0 ]; then
                        echo "  Warning: $BATCH_FAILED batches failed"
                        FAILED_OUTPUTS+=("$texture/$surface/level_$level/$output_name ($BATCH_FAILED batches failed)")
                    fi
                fi
            
                # ================================================================
                # Merge partial results for this output
                # ================================================================
                if [ "$DRY_RUN" = false ]; then
                    echo ""
                    echo "  Merging partial results..."
                    
                    MERGE_CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --merge_only \
                        $VERBOSE"
                    
                    if ! $MERGE_CMD; then
                        echo "  Warning: Merge failed for $texture/$surface/level_$level/$output_name"
                        FAILED_OUTPUTS+=("$texture/$surface/level_$level/$output_name (merge failed)")
                    else
                        echo "  Done! Results saved to: $GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz"
                    fi
                fi
                
                    echo ""
                done
            done
        done
    done
done

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "All Processing Complete"
echo "============================================================"
echo "  Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Total outputs processed: $CURRENT_OUTPUT"
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
