#!/bin/bash
#
# Run the FULL TOSCA pipeline end-to-end for a SINGLE Gaussian output:
#   1. Render synthetic COLMAP dataset (CPU)
#   2. Train Gaussian Splatting + extract mesh (GPU)
#   3. Evaluate mesh quality
#   4. Compute geodesic distances (CPU)
#   5. Generate training patches (CPU)
#
# Each stage runs sequentially. Based on the render parameters from
# render_all_blue.sh (blue texture, 600 views, 1024x1024, auto camera radius).
#
# USAGE:
#   bash scripts/tosca/pipeline/full_pipeline_single.sh <shape> [options]
#
# ARGUMENTS:
#   shape              TOSCA shape name (e.g., cat0, dog0, horse5)
#
# OPTIONS:
#   --light_id N       Light ID (default: 1). Use -1 for decoupled_appearance.
#   --texture NAME     Texture name (default: blue)
#   --resolution RES   COLMAP resolution (default: high_res)
#   --num_views N      Number of camera views (default: 600)
#   --iterations N     Training iterations (default: 30000)
#   --num_sources N    Geodesic FPS source vertices (default: 32)
#   --output NAME      Output folder name (default: output)
#   --skip_render      Skip stage 1 (rendering)
#   --skip_train       Skip stage 2 (training + mesh)
#   --skip_eval_mesh   Skip stage 3 (mesh evaluation)
#   --skip_geodesic    Skip stage 4 (geodesic)
#   --skip_patches     Skip stage 5 (patches)
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   # Full pipeline for cat0 with light_id=1
#   bash scripts/tosca/pipeline/full_pipeline_single.sh cat0 --light_id 1
#
#   # Skip render and train (already done)
#   bash scripts/tosca/pipeline/full_pipeline_single.sh cat0 --skip_render --skip_train
#
#   # Decoupled appearance mode
#   bash scripts/tosca/pipeline/full_pipeline_single.sh cat0 --light_id -1

set -e

# ============================================================================
# Defaults
# ============================================================================
TEXTURE="blue"
RESOLUTION="high_res"
LIGHT_ID=1
NUM_VIEWS=600
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
ITERATIONS=30000
NUM_SOURCES=32
OUTPUT_NAME="output"
SKIP_RENDER=false
SKIP_TRAIN=false
SKIP_EVAL_MESH=false
SKIP_GEODESIC=false
SKIP_PATCHES=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <shape> [options]"
    echo "  e.g.: $0 cat0 --light_id 1"
    echo "  Use --help for all options."
    exit 1
fi

SHAPE="$1"; shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --light_id)       LIGHT_ID="$2";       shift 2 ;;
        --texture)        TEXTURE="$2";        shift 2 ;;
        --resolution)     RESOLUTION="$2";     shift 2 ;;
        --num_views)      NUM_VIEWS="$2";      shift 2 ;;
        --iterations)     ITERATIONS="$2";     shift 2 ;;
        --num_sources)    NUM_SOURCES="$2";    shift 2 ;;
        --output)         OUTPUT_NAME="$2";    shift 2 ;;
        --skip_render)    SKIP_RENDER=true;    shift   ;;
        --skip_train)     SKIP_TRAIN=true;     shift   ;;
        --skip_eval_mesh) SKIP_EVAL_MESH=true; shift   ;;
        --skip_geodesic)  SKIP_GEODESIC=true;  shift   ;;
        --skip_patches)   SKIP_PATCHES=true;   shift   ;;
        --dry_run)        DRY_RUN=true;        shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Setup paths
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Unset PYTHONPATH to avoid import conflicts — Python scripts manage their own sys.path
unset PYTHONPATH

DATA_ROOT="TrainData/TOSCA/processed"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"

# Determine appearance subdirectory
if [ "$LIGHT_ID" -eq -1 ]; then
    APPEARANCE="decoupled_appearance"
    TRAIN_EXTRA="--use_decoupled_appearance"
    LIGHT_FLAG="--use_decoupled_appearance"
else
    APPEARANCE="light_${LIGHT_ID}"
    TRAIN_EXTRA=""
    LIGHT_FLAG="--light_id $LIGHT_ID"
fi

DATASET_DIR="$SYNTH_DATA_BASE/${TEXTURE}_texture/$SHAPE/$RESOLUTION/$APPEARANCE"
MODEL_DIR="$DATASET_DIR/$OUTPUT_NAME"

# Resolve Python interpreter
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
[ -x "$PYTHON3" ] || PYTHON3=python3

EVAL_SCRIPT="$PROJECT_ROOT/scripts/evaluate_gaussian_mesh_quality.py"
GEODESIC_SCRIPT="$PROJECT_ROOT/GenerateData/compute_gaussian_geodesic_distances_tosca.py"
RENDER_SCRIPT="$PROJECT_ROOT/GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py"

echo "============================================================"
echo "TOSCA Full Pipeline — Single Shape"
echo "============================================================"
echo "  Shape:        $SHAPE"
echo "  Texture:      $TEXTURE"
echo "  Resolution:   $RESOLUTION"
echo "  Appearance:   $APPEARANCE"
echo "  Light ID:     $LIGHT_ID"
echo "  Views:        $NUM_VIEWS"
echo "  Image size:   ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "  Iterations:   $ITERATIONS"
echo "  Num sources:  $NUM_SOURCES"
echo "  Dataset:      $DATASET_DIR"
echo "  Model:        $MODEL_DIR"
echo ""
echo "  Stages:"
[ "$SKIP_RENDER" = false ]    && echo "    1. Render"             || echo "    1. Render [SKIPPED]"
[ "$SKIP_TRAIN" = false ]     && echo "    2. Train + Mesh"       || echo "    2. Train [SKIPPED]"
[ "$SKIP_EVAL_MESH" = false ] && echo "    3. Evaluate Mesh"      || echo "    3. Eval Mesh [SKIPPED]"
[ "$SKIP_GEODESIC" = false ]  && echo "    4. Geodesic Distances" || echo "    4. Geodesic [SKIPPED]"
[ "$SKIP_PATCHES" = false ]   && echo "    5. Training Patches"   || echo "    5. Patches [SKIPPED]"
echo ""

# ============================================================================
# Stage 1: Render
# ============================================================================
if [ "$SKIP_RENDER" = false ]; then
    echo "========== Stage 1/5: Rendering =========="
    CMD="$PYTHON3 $RENDER_SCRIPT \
        --shape $SHAPE \
        --texture_name $TEXTURE \
        --colmap_resolution $RESOLUTION \
        --image_mesh_resolution $RESOLUTION \
        --num_views $NUM_VIEWS \
        --image_width $IMAGE_WIDTH \
        --image_height $IMAGE_HEIGHT \
        --auto_camera_radius \
        --data_root $DATA_ROOT \
        --output_root $SYNTH_DATA_BASE \
        $LIGHT_FLAG"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
        echo "Rendering complete."
    else
        echo "  [DRY RUN]"
    fi
    echo ""
fi

# ============================================================================
# Stage 2: Train + Mesh Extract
# ============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo "========== Stage 2/5: Training + Mesh =========="

    if [ ! -d "$DATASET_DIR" ]; then
        echo "ERROR: Dataset not found: $DATASET_DIR"
        echo "       Run rendering first (remove --skip_render)."
        exit 1
    fi

    # Train
    CMD="$PYTHON3 train.py -s $DATASET_DIR -m $MODEL_DIR --eval --iterations $ITERATIONS -r 2 $TRAIN_EXTRA"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
        echo "Training complete."
    else
        echo "  [DRY RUN]"
    fi

    # Mesh extract
    CMD="$PYTHON3 mesh_extract_tetrahedra.py -s $DATASET_DIR -m $MODEL_DIR --eval"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
        echo "Mesh extraction complete."
    else
        echo "  [DRY RUN]"
    fi
    echo ""
fi

# ============================================================================
# Stage 3: Evaluate Mesh Quality
# ============================================================================
if [ "$SKIP_EVAL_MESH" = false ]; then
    echo "========== Stage 3/5: Evaluate Mesh Quality =========="

    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Output directory not found: $MODEL_DIR"
        exit 1
    fi

    CMD="$PYTHON3 $EVAL_SCRIPT --output_dir $MODEL_DIR --iteration $ITERATIONS"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
        echo "Mesh evaluation complete."
    else
        echo "  [DRY RUN]"
    fi
    echo ""
fi

# ============================================================================
# Stage 4: Geodesic Distances
# ============================================================================
if [ "$SKIP_GEODESIC" = false ]; then
    echo "========== Stage 4/5: Geodesic Distances =========="

    GAUSS_OUTPUT="$MODEL_DIR"
    CMD="$PYTHON3 $GEODESIC_SCRIPT \
        --gaussian_output $GAUSS_OUTPUT \
        --shape $SHAPE \
        --mesh_type reconstructed \
        --num_sources $NUM_SOURCES"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
        echo "Geodesic distances complete."
    else
        echo "  [DRY RUN]"
    fi
    echo ""
fi

# ============================================================================
# Stage 5: Training Patches (optional — requires config)
# ============================================================================
if [ "$SKIP_PATCHES" = false ]; then
    echo "========== Stage 5/5: Training Patches =========="

    # Extract animal name from shape (e.g., cat0 -> cat)
    ANIMAL="${SHAPE%%[0-9]*}"
    CONFIG_DIR="DataSets/configs/tosca"
    CONFIG="$CONFIG_DIR/tosca_${ANIMAL}.yaml"

    if [ -f "$CONFIG" ]; then
        CMD="$PYTHON3 DataSets/create_gaussian_training_patches.py --config $CONFIG"
        echo "  $CMD"
        if [ "$DRY_RUN" = false ]; then
            eval "$CMD"
            echo "Training patches complete."
        else
            echo "  [DRY RUN]"
        fi
    else
        echo "  Warning: Config not found: $CONFIG — skipping patches."
        echo "  To create patches, create a config file or run generate_training_patches.sh."
    fi
    echo ""
fi

# ============================================================================
# Done
# ============================================================================
echo "========== TOSCA Pipeline Complete =========="
echo "Results: $MODEL_DIR"
if [ -f "$MODEL_DIR/quality_report.json" ]; then
    echo ""
    echo "Mesh quality report:"
    cat "$MODEL_DIR/quality_report.json"
fi
