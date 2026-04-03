#!/bin/bash
#
# Render polynomial surfaces at level 5 for evaluation.
#
# Creates a separate set of Gaussian data (level 05) that can be used to
# evaluate geodesic prediction models on Gaussians that were NOT used
# for training (which used levels 02, 03, 04).
#
# USAGE:
#   bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh [options]
#
# OPTIONS:
#   --surfaces LIST           Comma-separated surfaces (default: Paraboloid,Saddle,HyperbolicParaboloid)
#   --textures LIST           Comma-separated textures (default: blue)
#   --colmap_levels LIST      Comma-separated COLMAP levels (default: 5)
#   --light_ids LIST          Comma-separated light IDs (default: 0,1,2,3,4)
#   --num_views N             Number of rendering views (default: 400)
#   --image_width N           Image width in pixels (default: 640)
#   --image_height N          Image height in pixels (default: 480)
#   --output_base DIR         Base output directory (default: TrainData/Polynomial/SyntheticColmapData)
#   --max_parallel N          Maximum parallel jobs (default: auto-detect nproc-1)
#   --dry_run                 Print commands without executing
#   --sequential              Run jobs sequentially instead of parallel
#
# EXAMPLES:
#   # Render level 5 evaluation data (all surfaces, all lights)
#   bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh
#
#   # Render only Paraboloid level 5
#   bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh --surfaces Paraboloid
#
#   # Dry run
#   bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
COLMAP_LEVELS="5"
IMAGE_MESH_LEVEL=0
LIGHT_IDS="0,1,2,3,4"
NUM_VIEWS=400
IMAGE_WIDTH=640
IMAGE_HEIGHT=480
CAMERA_RADIUS=2.6
OUTPUT_BASE="TrainData/Polynomial/SyntheticColmapData"
DATA_ROOT="TrainData/Polynomial/raw"

MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

DRY_RUN=false
SEQUENTIAL=false

declare -A CAMERA_RADIUS_MAP=(
    [Paraboloid]=2.8
    [Saddle]=2.8
    [HyperbolicParaboloid]=3.0
)

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --surfaces)         SURFACES="$2";       shift 2 ;;
        --textures)         TEXTURES="$2";       shift 2 ;;
        --colmap_levels)    COLMAP_LEVELS="$2";  shift 2 ;;
        --light_ids)        LIGHT_IDS="$2";      shift 2 ;;
        --image_mesh_level) IMAGE_MESH_LEVEL="$2"; shift 2 ;;
        --num_views)        NUM_VIEWS="$2";      shift 2 ;;
        --image_width)      IMAGE_WIDTH="$2";    shift 2 ;;
        --image_height)     IMAGE_HEIGHT="$2";   shift 2 ;;
        --camera_radius)    CAMERA_RADIUS="$2";  shift 2 ;;
        --output_base)      OUTPUT_BASE="$2";    shift 2 ;;
        --data_root)        DATA_ROOT="$2";      shift 2 ;;
        --max_parallel)     MAX_PARALLEL="$2";   shift 2 ;;
        --dry_run)          DRY_RUN=true;        shift   ;;
        --sequential)       SEQUENTIAL=true;     shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
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
IFS=',' read -ra COLMAP_LEVEL_ARRAY <<< "$COLMAP_LEVELS"

if [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Determine project root directory
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# Count total jobs
# ============================================================================
TOTAL_JOBS=0
for texture in "${TEXTURE_ARRAY[@]}"; do
    for colmap_level in "${COLMAP_LEVEL_ARRAY[@]}"; do
        for surface in "${SURFACE_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        done
    done
done

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Render Evaluation Surfaces (Level 5) – new_dataset"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project root:        $PROJECT_ROOT"
echo "  Output base:         $OUTPUT_BASE"
echo "  Surfaces:            ${SURFACE_ARRAY[*]}"
echo "  Textures:            ${TEXTURE_ARRAY[*]}"
echo "  COLMAP levels:       ${COLMAP_LEVEL_ARRAY[*]}"
echo "  Image mesh level:    ${IMAGE_MESH_LEVEL}"
if [ -n "$LIGHT_IDS" ]; then
    echo "  Light IDs:           ${LIGHT_ID_ARRAY[*]}"
else
    echo "  Light IDs:           (default lighting)"
fi
echo "  Number of views:     $NUM_VIEWS"
echo "  Image size:          ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "  Camera radius:       $CAMERA_RADIUS"
echo "  Max parallel jobs:   $MAX_PARALLEL"
echo "  Sequential:          $SEQUENTIAL"
echo "  Dry run:             $DRY_RUN"
echo ""
echo "Total rendering jobs:  $TOTAL_JOBS"
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
    for colmap_level in "${COLMAP_LEVEL_ARRAY[@]}"; do
        for surface in "${SURFACE_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                CURRENT_JOB=$((CURRENT_JOB + 1))

                RADIUS=${CAMERA_RADIUS_MAP[$surface]:-$CAMERA_RADIUS}

                if [ -n "$light_id" ]; then
                    OUTPUT_DIR="$OUTPUT_BASE/${texture}_texture/$surface/level_0${colmap_level}/light_${light_id}"
                    LIGHT_ID_DESC="Light ID: $light_id"
                else
                    OUTPUT_DIR="$OUTPUT_BASE/${texture}_texture/$surface/level_0${colmap_level}/default_light"
                    LIGHT_ID_DESC="Default lighting"
                fi

                echo "[$CURRENT_JOB/$TOTAL_JOBS] Rendering:"
                echo "  Surface:         $surface"
                echo "  Texture:         $texture"
                echo "  COLMAP level:    $colmap_level"
                echo "  $LIGHT_ID_DESC"
                echo "  Camera radius:   $RADIUS"
                echo "  Output:          $OUTPUT_DIR"

                CMD="python3 GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
                    --surface $surface \
                    --texture_name $texture \
                    --colmap_level $colmap_level \
                    --image_mesh_level $IMAGE_MESH_LEVEL \
                    --num_views $NUM_VIEWS \
                    --image_width $IMAGE_WIDTH \
                    --image_height $IMAGE_HEIGHT \
                    --camera_radius $RADIUS \
                    --output_root $OUTPUT_BASE \
                    --data_root $DATA_ROOT"

                if [ -n "$light_id" ]; then
                    CMD="$CMD --light_id $light_id"
                fi

                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] $CMD"
                elif [ "$SEQUENTIAL" = true ]; then
                    if ! eval "$CMD"; then
                        echo "  Failed"
                        FAILED_JOBS+=("$surface/$texture/level_0${colmap_level}/light_${light_id}")
                    else
                        echo "  Success"
                    fi
                else
                    eval "$CMD" &
                    PIDS+=($!)

                    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
                        for i in "${!PIDS[@]}"; do
                            pid=${PIDS[$i]}
                            if ! kill -0 $pid 2>/dev/null; then
                                wait $pid 2>/dev/null || true
                                unset 'PIDS[$i]'
                                break
                            fi
                        done
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
    echo "Evaluation surfaces rendered!"
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
