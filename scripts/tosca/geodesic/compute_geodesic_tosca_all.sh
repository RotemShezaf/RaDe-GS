#!/bin/bash
#
# Compute geodesic distances for all TOSCA shapes and Gaussian outputs
#
# This script iterates over:
#   - Multiple batch splits (source range batches)
#   - Multiple textures
#   - Multiple colmap resolutions (high_res, low_res)
#   - Multiple output names (e.g., different training runs)
#   - Multiple TOSCA shapes (auto-detected or explicitly specified)
#
# It assumes the directory structure created by create_synthetic_colmap_dataset_from_mesh_tosca.py
# and that training has been performed with outputs stored in the specified output folder.
#
# NOTE: compute_gaussian_geodesic_distances.py currently supports only polynomial surfaces
# via --surface. For TOSCA support, the Python script needs to be extended with a --shape
# argument that loads ground truth meshes from TrainData/TOSCA/processed/{shape}/mesh_high_res_*.ply.
# This script is designed to work with that extension.
#
# USAGE:
#   ./scripts/compute_geodesic_tosca_all.sh [options]
#
# OPTIONS:
#   --data_root DIR            Preprocessed TOSCA data root (default: TrainData/TOSCA/processed)
#   --synth_data_base DIR      Synthetic COLMAP data base (default: TrainData/TOSCA/SyntheticColmapData)
#   --shapes LIST              Comma-separated shape names (default: auto-detect from data_root)
#   --textures LIST            Comma-separated texture names (default: colors)
#   --colmap_resolutions LIST  Comma-separated COLMAP resolution levels (default: high_res)
#   --outputs LIST             Comma-separated output names (default: output)
#   --light_ids LIST           Comma-separated light IDs (default: 0,1,2,3,4)
#   --n_batches N              Number of batches per shape/output (default: 8)
#   --resolution N             Source mesh resolution (NxN) (default: 8)
#   --use_mahalanobis          Use Mahalanobis distance
#   --dry_run                  Print commands without executing
#   --sequential               Run batches sequentially instead of parallel
#   --verbose                  Enable verbose output
#
# EXAMPLES:
#   # Process all auto-detected shapes with default settings
#   ./scripts/compute_geodesic_tosca_all.sh
#
#   # Process only cat shapes
#   ./scripts/compute_geodesic_tosca_all.sh --shapes "cat0,cat1,cat2"
#
#   # Dry run
#   ./scripts/compute_geodesic_tosca_all.sh --dry_run

set -e

# Load animal→index map and expand_animals() helper
# Edit scripts/tosca/tosca_animal_map.sh to control which shapes are processed.
source "$(dirname "${BASH_SOURCE[0]}")/../tosca_animal_map.sh"

# ============================================================================
# Default values
# ============================================================================
DATA_ROOT="TrainData/TOSCA/processed"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"

N_BATCHES=8
MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

NUM_SOURCES=64 # Number of source vertices for FPS sampling
SHAPES=""     # Empty = auto-detect from DATA_ROOT
ANIMALS=""    # Animal names without index; expands to all indexed shapes found in DATA_ROOT
TEXTURES="colors"
COLMAP_RESOLUTIONS="high_res"
OUTPUTS="output"
LIGHT_IDS="0,1,2,3,4"
USE_DECOUPLED_APPEARANCE=false  # When true, use 'decoupled_appearance' subdir instead of 'light_{id}'
DRY_RUN=false
SEQUENTIAL=false
USE_MAHALANOBIS=""
VERBOSE="" #--mesh_type
MESH_TYPE="reconstructed"
# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --synth_data_base)
            SYNTH_DATA_BASE="$2"
            shift 2
            ;;
        --shapes)
            SHAPES="$2"
            shift 2
            ;;
        --animals)
            ANIMALS="$2"
            shift 2
            ;;
        --textures)
            TEXTURES="$2"
            shift 2
            ;;
        --colmap_resolutions)
            COLMAP_RESOLUTIONS="$2"
            shift 2
            ;;
        --outputs)
            OUTPUTS="$2"
            shift 2
            ;;
        --light_ids)
            LIGHT_IDS="$2"
            shift 2
            ;;
        --use_decoupled_appearance)
            USE_DECOUPLED_APPEARANCE=true
            shift
            ;;
        --n_jobs)
            N_JOBS_OVERRIDE="$2"
            shift 2
            ;;
        --n_batches)
            N_BATCHES="$2"
            shift 2
            ;;
        --num_sources)
            NUM_SOURCES="$2"
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
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR            Preprocessed TOSCA data (default: TrainData/TOSCA/processed)"
            echo "  --synth_data_base DIR       Synthetic COLMAP data (default: TrainData/TOSCA/SyntheticColmapData)"
            echo "  --shapes LIST               Comma-separated TOSCA shape names (default: auto-detect)"
            echo "  --textures LIST             Comma-separated textures (default: colors)"
            echo "  --colmap_resolutions LIST   Comma-separated COLMAP resolutions (default: high_res)"
            echo "  --outputs LIST              Comma-separated output folder names (default: output)"
            echo "  --light_ids LIST            Comma-separated light IDs (default: 0,1,2,3,4)"
            echo "  --n_jobs N                  Parallel jobs per batch"
            echo "  --n_batches N               Number of batches per shape (default: 8)"
            echo "  --num_sources N             Number of source vertices (default: 200)"
            echo "  --dry_run                   Print commands without executing"
            echo "  --sequential                Run batches sequentially"
            echo "  --use_mahalanobis           Use Mahalanobis distance"
            echo "  --verbose                   Enable verbose output"
            echo ""
            echo "NOTE: Requires compute_gaussian_geodesic_distances.py to support --shape"
            echo "      for TOSCA (extend beyond polynomial --surface argument)."
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
# Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_gaussian_geodesic_distances_tosca.py"
if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: TOSCA compute script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Expand --animals to indexed shape names using ANIMAL_INDEX_MAP
# (edit scripts/tosca/tosca_animal_map.sh to add/remove indices per animal)
# ============================================================================
if [ -n "$ANIMALS" ]; then
    SHAPES="$(expand_animals "$ANIMALS")"
    if [ -z "$SHAPES" ]; then
        echo "Error: No shapes found for animals: $ANIMALS"
        exit 1
    fi
    echo "Expanded animals '$ANIMALS' -> shapes: $SHAPES"
fi
# ============================================================================
# Auto-detect shapes if not specified
# ============================================================================
if [ -z "$SHAPES" ]; then
    echo "Auto-detecting TOSCA shapes from $DATA_ROOT ..."
    detected_shapes=()
    if [ -d "$DATA_ROOT" ]; then
        for shape_dir in "$DATA_ROOT"/*/; do
            [ -d "$shape_dir" ] || continue
            shape_name="$(basename "$shape_dir")"
            # Only include if it contains PLY mesh files
            if ls "$shape_dir"/mesh_high_res_*.ply 2>/dev/null | grep -q .; then
                detected_shapes+=("$shape_name")
            fi
        done
    fi
    if [ ${#detected_shapes[@]} -eq 0 ]; then
        echo "Error: No processed TOSCA shapes found in $DATA_ROOT"
        echo "       Please run preprocess_tosca.py first, or specify shapes with --shapes."
        exit 1
    fi
    SHAPES=$(IFS=','; echo "${detected_shapes[*]}")
    echo "  Found ${#detected_shapes[@]} shapes: $SHAPES"
fi

# ============================================================================
# Convert comma-separated lists to arrays
# ============================================================================
IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra RESOLUTION_ARRAY <<< "$COLMAP_RESOLUTIONS"
IFS=',' read -ra OUTPUT_ARRAY <<< "$OUTPUTS"

if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    LIGHT_ID_ARRAY=("__decoupled__")
elif [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Compute batch parameters
# ============================================================================
TOTAL_SOURCES=$NUM_SOURCES
SOURCES_PER_BATCH=$(( (TOTAL_SOURCES + N_BATCHES - 1) / N_BATCHES ))

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
echo "Geodesic Distance Computation - All TOSCA Shapes"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Data root:         $DATA_ROOT"
echo "  Synth data base:   $SYNTH_DATA_BASE"
echo "  Shapes:            ${SHAPE_ARRAY[*]}"
echo "  Textures:          ${TEXTURE_ARRAY[*]}"
echo "  COLMAP resolutions:${RESOLUTION_ARRAY[*]}"
echo "  Outputs:           ${OUTPUT_ARRAY[*]}"
if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    echo "  Lighting mode:     decoupled_appearance"
elif [ ${#LIGHT_ID_ARRAY[@]} -gt 0 ] && [ -n "${LIGHT_ID_ARRAY[0]}" ]; then
    echo "  Light IDs:         ${LIGHT_ID_ARRAY[*]}"
else
    echo "  Lighting mode:     default_light"
fi
echo "  Num sources:       $NUM_SOURCES"
echo "  Number of batches: $N_BATCHES"
echo "  Sources per batch: ~$SOURCES_PER_BATCH"
echo "  Jobs per batch:    $N_JOBS"
echo "  Sequential:        $SEQUENTIAL"
echo "  Dry run:           $DRY_RUN"
echo ""
echo "NOTE: This script calls compute_gaussian_geodesic_distances.py with"
echo "      --shape (TOSCA shape name) instead of --surface. Ensure the"
echo "      Python script has been extended to support TOSCA shapes."
echo ""

# ============================================================================
# Count total work
# ============================================================================
TOTAL_OUTPUTS=0
for shape in "${SHAPE_ARRAY[@]}"; do
    for texture in "${TEXTURE_ARRAY[@]}"; do
        for resolution in "${RESOLUTION_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                for output_name in "${OUTPUT_ARRAY[@]}"; do
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
# ============================================================================
START_TIME=$(date +%s)
CURRENT_OUTPUT=0
FAILED_OUTPUTS=()

for shape in "${SHAPE_ARRAY[@]}"; do
    for texture in "${TEXTURE_ARRAY[@]}"; do
        for resolution in "${RESOLUTION_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                for output_name in "${OUTPUT_ARRAY[@]}"; do
                    CURRENT_OUTPUT=$((CURRENT_OUTPUT + 1))

                    # Construct paths
                    # Gaussian output: {synth_data_base}/{texture}_texture/{shape}/{resolution}/light_{light_id}/{output_name}
                    if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/decoupled_appearance/$output_name"
                    elif [ -n "$light_id" ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${light_id}/$output_name"
                    else
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/default_light/$output_name"
                    fi

                    echo "============================================================"
                    echo "[$CURRENT_OUTPUT/$TOTAL_OUTPUTS] Processing:"
                    echo "  Shape:   $shape"
                    echo "  Texture: $texture"
                    echo "  Resolution: $resolution"
                    if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                        echo "  Lighting: decoupled_appearance"
                    elif [ -n "$light_id" ]; then
                        echo "  Light ID: $light_id"
                    fi
                    echo "  Output:  $output_name"
                    echo "  Gaussian output: $GAUSSIAN_OUTPUT"
                    echo "  Ground truth mesh: $DATA_ROOT/$shape/mesh_high_res_*.ply"
                    echo "============================================================"

                    # Check if Gaussian output exists
                    if [ ! -d "$GAUSSIAN_OUTPUT" ]; then
                        echo "  Warning: Gaussian output not found, skipping..."
                        echo "           Expected: $GAUSSIAN_OUTPUT"
                        FAILED_OUTPUTS+=("$texture/$shape/$resolution/$output_name (missing output)")
                        echo ""
                        continue
                    fi

                    # Check if point cloud exists
                    if [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
                        echo "  Warning: No point_cloud directory found, skipping..."
                        FAILED_OUTPUTS+=("$texture/$shape/$resolution/$output_name (no point cloud)")
                        echo ""
                        continue
                    fi

                    # Check if TOSCA ground truth mesh exists
                    if ! ls "$DATA_ROOT/$shape"/mesh_high_res_*.ply 2>/dev/null | grep -q .; then
                        echo "  Warning: No ground truth mesh found at $DATA_ROOT/$shape/mesh_high_res_*.ply"
                        echo "           Please run preprocess_tosca.py first."
                        FAILED_OUTPUTS+=("$texture/$shape/$resolution/$output_name (missing GT mesh)")
                        echo ""
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

                        # NOTE: compute_gaussian_geodesic_distances.py currently uses
                        # --surface for polynomial types. For TOSCA, the script needs to
                        # be extended with --shape and --tosca_data_root parameters.
                        # Replace --surface with --shape once TOSCA support is added.
                        CMD="python $COMPUTE_SCRIPT \
                            --gaussian_output $GAUSSIAN_OUTPUT \
                            --data_root $DATA_ROOT \
                            --shape $shape \
                            --mesh_type $MESH_TYPE \
                            --num_sources $NUM_SOURCES \
                            --geodesic_method mmp \
                            --embed_gaussians \
                            --source_start $SOURCE_START \
                            --source_end $SOURCE_END \
                            --n_jobs $N_JOBS \
                            $USE_MAHALANOBIS \
                            $VERBOSE"

                        if [ "$DRY_RUN" = true ]; then
                            echo "    [DRY RUN] $CMD"
                        elif [ "$SEQUENTIAL" = true ]; then
                            if ! $CMD; then
                                echo "    Warning: Batch $((batch+1)) failed"
                            fi
                        else
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
                            FAILED_OUTPUTS+=("$texture/$shape/$resolution/$output_name ($BATCH_FAILED batches failed)")
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
                            echo "  Warning: Merge failed for $texture/$shape/$resolution/$output_name"
                            FAILED_OUTPUTS+=("$texture/$shape/$resolution/$output_name (merge failed)")
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
echo "  Total time:              ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Total outputs processed: $CURRENT_OUTPUT"
echo "  Failed outputs:          ${#FAILED_OUTPUTS[@]}"

if [ ${#FAILED_OUTPUTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed outputs:"
    for failed in "${FAILED_OUTPUTS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
echo "Done!"
