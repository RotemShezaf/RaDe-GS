#!/bin/bash
#
# Render all TOSCA shapes with synthetic COLMAP datasets
#
# This script generates synthetic COLMAP-style datasets by rendering multiple
# TOSCA shapes at various mesh resolutions with different textures.
# Each combination can be processed in parallel.
#
# Features:
#   - Supports all preprocessed TOSCA shapes (cat, centaur, david, dog, gorilla, horse, ...)
#   - Loop over colmap resolutions and textures
#   - Parallel rendering with automatic job management
#   - Auto-detect available CPUs or manually specify --max_parallel
#   - Output organized by texture and shape
#
# USAGE:
#   ./render_all_tosca.sh [options]
#
# OPTIONS:
#   --shapes LIST               Comma-separated shape names with index (default: cat0,centaur0,dog0,horse0)
#   --animals LIST              Comma-separated animal names WITHOUT index (e.g. "cat,dog").
#                               Expands to all indexed shapes for those animals found in data_root.
#   --all_shapes                Process all shapes found in data_root
#   --textures LIST             Comma-separated textures (default: colors)
#   --colmap_resolutions LIST   Comma-separated resolutions: high_res,low_res (default: high_res)
#   --image_mesh_resolution S   Image mesh resolution: high_res or low_res (default: low_res)
#   --num_views N               Number of rendering views (default: 800)
#   --image_width N             Image width in pixels (default: 960)
#   --image_height N            Image height in pixels (default: 720)
#   --camera_radius N           Camera orbit radius (default: 6)
#   --auto_camera_radius        Auto-compute smallest radius that fits the mesh in all images
#   --data_root DIR             Root directory with preprocessed TOSCA shapes
#   --output_root DIR           Output root directory
#   --light_ids LIST            Comma-separated light IDs (default: 0,1,2,3,4)
#   --use_decoupled_appearance  Use --use_decoupled_appearance flag instead of --light_id.
#                               Overrides --light_ids; output subdir becomes 'decoupled_appearance'.
#   --max_parallel N            Maximum parallel jobs (default: auto-detect nproc-1)
#   --dry_run                   Print commands without executing
#   --sequential                Run jobs sequentially instead of parallel
#   --help                      Show this help message
#
# EXAMPLES:
#   # Render with defaults (a few representative shapes)
#   ./render_all_tosca.sh
#
#   # Render all shapes with both resolutions
#   ./render_all_tosca.sh \
#       --all_shapes \
#       --colmap_resolutions "high_res,low_res"
#
#   # Render specific animals (all indices) with decoupled appearance
#   ./render_all_tosca.sh \
#       --animals "cat,centaur,david" \
#       --textures "colors" \
#       --use_decoupled_appearance
#
#   # Render specific shapes (with index) using light IDs
#   ./render_all_tosca.sh \
#       --shapes "cat0,centaur0,david0" \
#       --textures "colors" \
#       --light_ids "0,1,2,3,4"
#
#   # Dry run to see what will be executed
#   ./render_all_tosca.sh --dry_run

set -e

# Load animal→index map and expand_animals() helper
# Edit scripts/tosca/tosca_animal_map.sh to control which shapes are processed.
source "$(dirname "${BASH_SOURCE[0]}")/../tosca_animal_map.sh"

# ============================================================================
# Default values
# ============================================================================
SHAPES="cat0,centaur0,dog0,horse0"
ANIMALS="cat,centaur,david,dog,gorilla,horse,lioness,michael,seahorse,shark,victoria,wolf"
ALL_SHAPES=false
TEXTURES="colors"
COLMAP_RESOLUTIONS="high_res"
IMAGE_MESH_RESOLUTION="high_res"
LIGHT_IDS="0,1,2,3,4"  # Empty means default_light, otherwise comma-separated like "0,1,2,3,4"
USE_DECOUPLED_APPEARANCE=false  # When true, passes --use_decoupled_appearance and ignores LIGHT_IDS
NUM_VIEWS=600
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
CAMERA_RADIUS=250
AUTO_CAMERA_RADIUS=false
DATA_ROOT="TrainData/TOSCA/processed"
OUTPUT_ROOT="TrainData/TOSCA/SyntheticColmapData"

# Use geo_splat conda Python explicitly to avoid .venv taking precedence
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then
    PYTHON3=python3  # fallback
fi

# Auto-detect number of CPUs, use (nproc - 1) as default
MAX_PARALLEL=4 #$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

DRY_RUN=false
SEQUENTIAL=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --shapes)
            SHAPES="$2"
            shift 2
            ;;
        --animals)
            ANIMALS="$2"
            shift 2
            ;;
        --all_shapes)
            ALL_SHAPES=true
            shift
            ;;
        --textures)
            TEXTURES="$2"
            shift 2
            ;;
        --colmap_resolutions)
            COLMAP_RESOLUTIONS="$2"
            shift 2
            ;;
        --image_mesh_resolution)
            IMAGE_MESH_RESOLUTION="$2"
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
        --num_views)
            NUM_VIEWS="$2"
            shift 2
            ;;
        --image_width)
            IMAGE_WIDTH="$2"
            shift 2
            ;;
        --image_height)
            IMAGE_HEIGHT="$2"
            shift 2
            ;;
        --camera_radius)
            CAMERA_RADIUS="$2"
            shift 2
            ;;
        --auto_camera_radius)
            AUTO_CAMERA_RADIUS=true
            shift
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --max_parallel)
            MAX_PARALLEL="$2"
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
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --shapes LIST               Shape names with index (default: cat0,centaur0,dog0,horse0)"
            echo "  --animals LIST              Animal names without index (e.g. \"cat,dog\"); expands to all indexed shapes"
            echo "  --all_shapes                Process all shapes in data_root"
            echo "  --textures LIST             Textures (default: colors)"
            echo "  --colmap_resolutions LIST   COLMAP resolutions: high_res,low_res (default: high_res)"
            echo "  --image_mesh_resolution S   Image mesh resolution (default: low_res)"
            echo "  --num_views N               Number of views (default: 800)"
            echo "  --image_width N             Image width (default: 960)"
            echo "  --image_height N            Image height (default: 720)"
            echo "  --camera_radius N           Camera radius (default: 6)"
            echo "  --auto_camera_radius        Auto-compute smallest radius that fits the mesh"
            echo "  --data_root DIR             Data root directory"
            echo "  --output_root DIR           Output root directory"
            echo "  --light_ids LIST            Light IDs (default: 0,1,2,3,4)"
            echo "  --use_decoupled_appearance  Use decoupled appearance mode (overrides --light_ids)"
            echo "  --max_parallel N            Max parallel jobs (default: nproc-1)"
            echo "  --dry_run                   Print without executing"
            echo "  --sequential                Run sequentially"
            echo "  --help                      Show this message"
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
# Determine project root directory
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

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
# Resolve shape list
# ============================================================================
if [ "$ALL_SHAPES" = true ]; then
    if [ ! -d "$DATA_ROOT" ]; then
        echo "Error: data_root '$DATA_ROOT' does not exist!"
        exit 1
    fi
    # Collect all subdirectories that contain TOSCA mesh files
    SHAPES=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -exec sh -c 'ls "$1"/mesh_high_res_*.ply "$1"/mesh_low_res_*.ply 2>/dev/null | head -1 | grep -q .' _ {} \; -printf '%f\n' | sort | tr '\n' ',')
    SHAPES="${SHAPES%,}"  # Remove trailing comma
    if [ -z "$SHAPES" ]; then
        echo "Error: No valid TOSCA shapes found in '$DATA_ROOT'!"
        exit 1
    fi
    echo "Auto-detected shapes: $SHAPES"
fi

# ============================================================================
# Convert comma-separated lists to arrays
# ============================================================================
IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra COLMAP_RES_ARRAY <<< "$COLMAP_RESOLUTIONS"

# Handle light configuration:
#   --use_decoupled_appearance => single pass with sentinel "__decoupled__"
#   --light_ids ''             => single pass with no --light_id (default_light)
#   --light_ids '0,1,2,3,4'   => iterate over each id
if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    LIGHT_ITER_ARRAY=("__decoupled__")
elif [ -z "$LIGHT_IDS" ]; then
    LIGHT_ITER_ARRAY=("")  # Empty string means no --light_id flag
else
    IFS=',' read -ra LIGHT_ITER_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Count total jobs
# ============================================================================
TOTAL_JOBS=0
for texture in "${TEXTURE_ARRAY[@]}"; do
    for colmap_res in "${COLMAP_RES_ARRAY[@]}"; do
        for shape in "${SHAPE_ARRAY[@]}"; do
            for light_iter in "${LIGHT_ITER_ARRAY[@]}"; do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        done
    done
done

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Synthetic COLMAP Dataset Generation - All TOSCA Shapes"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project root:           $PROJECT_ROOT"
echo "  Data root:              $DATA_ROOT"
echo "  Output root:            $OUTPUT_ROOT"
echo "  Shapes:                 ${SHAPE_ARRAY[*]}"
echo "  Textures:               ${TEXTURE_ARRAY[*]}"
echo "  COLMAP resolutions:     ${COLMAP_RES_ARRAY[*]}"
echo "  Image mesh resolution:  $IMAGE_MESH_RESOLUTION"
if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    echo "  Lighting mode:          decoupled_appearance"
elif [ -n "$LIGHT_IDS" ]; then
    echo "  Light IDs:              ${LIGHT_ITER_ARRAY[*]}"
else
    echo "  Light IDs:              (default lighting)"
fi
echo "  Number of views:        $NUM_VIEWS"
echo "  Image size:             ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
if [ "$AUTO_CAMERA_RADIUS" = true ]; then
    echo "  Camera radius:          auto"
else
    echo "  Camera radius:          $CAMERA_RADIUS"
fi
echo "  Max parallel jobs:      $MAX_PARALLEL"
echo "  Sequential:             $SEQUENTIAL"
echo "  Dry run:                $DRY_RUN"
echo ""
echo "Total rendering jobs:     $TOTAL_JOBS"
echo ""

if [ $TOTAL_JOBS -eq 0 ]; then
    echo "Error: No jobs to process!"
    exit 1
fi

# ============================================================================
# Main processing loop
# ============================================================================
START_TIME=$(date +%s)
CURRENT_JOB=0
PIDS=()
FAILED_JOBS=()

for texture in "${TEXTURE_ARRAY[@]}"; do
    for colmap_res in "${COLMAP_RES_ARRAY[@]}"; do
        for shape in "${SHAPE_ARRAY[@]}"; do
            for light_iter in "${LIGHT_ITER_ARRAY[@]}"; do
                CURRENT_JOB=$((CURRENT_JOB + 1))

                # Construct output directory
                if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                    OUTPUT_DIR="$OUTPUT_ROOT/${texture}_texture/$shape/${colmap_res}/decoupled_appearance"
                    LIGHT_ID_DESC="Decoupled appearance"
                elif [ -n "$light_iter" ]; then
                    OUTPUT_DIR="$OUTPUT_ROOT/${texture}_texture/$shape/${colmap_res}/light_${light_iter}"
                    LIGHT_ID_DESC="Light ID: $light_iter"
                else
                    OUTPUT_DIR="$OUTPUT_ROOT/${texture}_texture/$shape/${colmap_res}/default_light"
                    LIGHT_ID_DESC="Default lighting"
                fi

                echo "[$CURRENT_JOB/$TOTAL_JOBS] Rendering:"
                echo "  Shape:                  $shape"
                echo "  Texture:                $texture"
                echo "  COLMAP resolution:      $colmap_res"
                echo "  Image mesh resolution:  $IMAGE_MESH_RESOLUTION"
                echo "  $LIGHT_ID_DESC"
                if [ "$AUTO_CAMERA_RADIUS" = true ]; then
                    echo "  Camera radius:          auto"
                else
                    echo "  Camera radius:          $CAMERA_RADIUS"
                fi
                echo "  Output:                 $OUTPUT_DIR"

                # Construct command
                CMD="$PYTHON3 GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py \
                    --shape $shape \
                    --texture_name $texture \
                    --colmap_resolution $colmap_res \
                    --image_mesh_resolution $IMAGE_MESH_RESOLUTION \
                    --num_views $NUM_VIEWS \
                    --image_width $IMAGE_WIDTH \
                    --image_height $IMAGE_HEIGHT \
                    --data_root $DATA_ROOT \
                    --output_root $OUTPUT_ROOT"

                # Add camera radius flag
                if [ "$AUTO_CAMERA_RADIUS" = true ]; then
                    CMD="$CMD --auto_camera_radius"
                else
                    CMD="$CMD --camera_radius $CAMERA_RADIUS"
                fi

                # Add lighting flag
                if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                    CMD="$CMD --use_decoupled_appearance"
                elif [ -n "$light_iter" ]; then
                    CMD="$CMD --light_id $light_iter"
                fi

                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] $CMD"
                elif [ "$SEQUENTIAL" = true ]; then
                    # Run sequentially
                    if ! eval "$CMD"; then
                        echo "  ✗ Failed"
                        FAILED_JOBS+=("$shape/$texture/${colmap_res}")
                    else
                        echo "  ✓ Success"
                    fi
                else
                    # Run in background and manage job queue
                    # Stagger launches so Open3D EGL contexts don't collide on
                    # init (Python startup + OffscreenRenderer() takes ~8-12s)
                    sleep 15
                    eval "$CMD" &
                    PIDS+=($!)

                    # Wait if we've reached max parallel jobs
                    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
                        # Wait for any job to complete
                        for i in "${!PIDS[@]}"; do
                            pid=${PIDS[$i]}
                            if ! kill -0 $pid 2>/dev/null; then
                                if ! wait $pid 2>/dev/null; then
                                    echo "  ✗ Job failed (pid $pid)"
                                fi
                                unset 'PIDS[$i]'
                                break
                            fi
                        done
                        # Re-index array
                        PIDS=("${PIDS[@]}")
                    fi
                fi

                echo ""
            done
        done
    done
done

# ============================================================================
# Wait for remaining background jobs
# ============================================================================
if [ "$DRY_RUN" = false ] && [ "$SEQUENTIAL" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for remaining jobs to complete..."
    FAILED_COUNT=0

    for pid in "${PIDS[@]}"; do
        if ! wait $pid; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done

    if [ $FAILED_COUNT -gt 0 ]; then
        echo "Warning: $FAILED_COUNT jobs failed"
    fi
fi

# ============================================================================
# Print summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed (no datasets generated)"
else
    echo "All TOSCA shapes rendered!"
    echo "Total time: ${ELAPSED}s"

    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        echo ""
        echo "Failed jobs:"
        for job in "${FAILED_JOBS[@]}"; do
            echo "  - $job"
        done
    fi
fi
echo "============================================================"
