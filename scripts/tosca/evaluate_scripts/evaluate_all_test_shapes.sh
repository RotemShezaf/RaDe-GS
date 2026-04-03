#!/bin/bash
#
# Run evaluation scripts in parallel across all TOSCA test shapes.
#
# Test shapes are high-resolution TOSCA shapes that are NOT in our training
# set (gorilla8, horse10, michael16).  For each shape we launch the full
# evaluation pipeline (run_all_evaluations.sh) inside a dedicated tmux
# session so all shapes are evaluated concurrently.
#
# USAGE:
#   bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep7)
#   --gpus N              Number of GPUs per job (default: 1)
#   --time HH:MM:SS       SLURM time limit per job (default: 4:00:00)
#   --config PATH         Testing config YAML (default: combined_tosca_ring3.yaml)
#   --model_path PATH     Model checkpoint path (optional)
#   --train_config PATH   Training config YAML (optional)
#   --dataset_config PATH Dataset config YAML (optional)
#   --shapes LIST         Comma-separated shape names to evaluate
#                         (default: gorilla8,horse10,michael16)
#   --texture NAME        Texture folder name (default: blue)
#   --resolution NAME     Resolution folder (default: high_res)
#   --output_name NAME    Gaussian output folder name (default: output)
#   --synth_data_base DIR Synthetic COLMAP data base
#                         (default: TrainData/TOSCA/SyntheticColmapData)
#   --sequential          Run each evaluation sequentially (default: parallel)
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep7"
GPUS=1
TIME="4:00:00"
CONFIG=""
MODEL_PATH=""
TRAIN_CONFIG=""
DATASET_CONFIG=""
SHAPES="gorilla8,horse10,michael16"
TEXTURE="blue"
RESOLUTION="high_res"
OUTPUT_NAME="output"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
SEQUENTIAL=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)              NODE="$2";            shift 2 ;;
        --gpus)              GPUS="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --config)            CONFIG="$2";          shift 2 ;;
        --model_path)        MODEL_PATH="$2";      shift 2 ;;
        --train_config)      TRAIN_CONFIG="$2";    shift 2 ;;
        --dataset_config)    DATASET_CONFIG="$2";  shift 2 ;;
        --shapes)            SHAPES="$2";          shift 2 ;;
        --texture)           TEXTURE="$2";         shift 2 ;;
        --resolution)        RESOLUTION="$2";      shift 2 ;;
        --output_name)       OUTPUT_NAME="$2";     shift 2 ;;
        --synth_data_base)   SYNTH_DATA_BASE="$2"; shift 2 ;;
        --sequential)        SEQUENTIAL=true;      shift   ;;
        --dry_run)           DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# ============================================================================
# Build shape array
# ============================================================================
IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Evaluate All TOSCA Test Shapes"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Node:              $NODE"
echo "  GPUs per job:      $GPUS"
echo "  Time limit:        $TIME"
echo "  Config:            ${CONFIG:-<default>}"
echo "  Model path:        ${MODEL_PATH:-<default>}"
echo "  Train config:      ${TRAIN_CONFIG:-<default>}"
echo "  Dataset config:    ${DATASET_CONFIG:-<default>}"
echo "  Synth data base:   $SYNTH_DATA_BASE"
echo "  Texture:           $TEXTURE"
echo "  Resolution:        $RESOLUTION"
echo "  Output name:       $OUTPUT_NAME"
echo "  Shapes:            ${SHAPE_ARRAY[*]}"
echo "  Sequential:        $SEQUENTIAL"
echo "  Dry run:           $DRY_RUN"
echo ""
echo "Total shapes to evaluate: ${#SHAPE_ARRAY[@]}"
echo ""

DRY_FLAG=""
if [ "$DRY_RUN" = true ]; then
    DRY_FLAG="--dry_run"
fi

# ============================================================================
# Launch evaluations for each shape
# ============================================================================
LAUNCHED=0

for shape in "${SHAPE_ARRAY[@]}"; do
    shape="$(echo "$shape" | tr -d '[:space:]')"

    GAUSSIAN_DIR="$SYNTH_DATA_BASE/${TEXTURE}_texture/${shape}/${RESOLUTION}/decoupled_appearance/${OUTPUT_NAME}"

    echo "------------------------------------------------------------"
    echo "[$((LAUNCHED + 1))/${#SHAPE_ARRAY[@]}] Launching evaluations for: $shape"
    echo "  Gaussian dir: $GAUSSIAN_DIR"
    echo ""

    # Verify the gaussian directory exists
    if [ ! -d "$PROJECT_ROOT/$GAUSSIAN_DIR" ]; then
        echo "  WARNING: Gaussian directory not found at $PROJECT_ROOT/$GAUSSIAN_DIR – skipping."
        echo ""
        continue
    fi

    # Build arguments for run_all_evaluations.sh
    EVAL_ARGS="--node $NODE --gpus $GPUS --time $TIME"
    EVAL_ARGS="$EVAL_ARGS --gaussian_dir $GAUSSIAN_DIR"
    if [[ -n "$CONFIG" ]];         then EVAL_ARGS="$EVAL_ARGS --config $CONFIG"; fi
    if [[ -n "$MODEL_PATH" ]];     then EVAL_ARGS="$EVAL_ARGS --model_path $MODEL_PATH"; fi
    if [[ -n "$TRAIN_CONFIG" ]];   then EVAL_ARGS="$EVAL_ARGS --train_config $TRAIN_CONFIG"; fi
    if [[ -n "$DATASET_CONFIG" ]]; then EVAL_ARGS="$EVAL_ARGS --dataset_config $DATASET_CONFIG"; fi
    if [[ -n "$DRY_FLAG" ]];       then EVAL_ARGS="$EVAL_ARGS $DRY_FLAG"; fi

    # Run the full evaluation pipeline for this shape
    # Each sub-script inside run_all_evaluations.sh creates its own tmux
    # session, so different shapes' sessions will be named differently
    # by appending the shape name.
    #
    # We override SESSION_NAME_SUFFIX via env variable so tmux sessions
    # don't collide across shapes.
    export EVAL_SHAPE_SUFFIX="_${shape}"
    bash "$SCRIPT_DIR/run_all_evaluations.sh" $EVAL_ARGS

    LAUNCHED=$((LAUNCHED + 1))

    if [ "$SEQUENTIAL" = true ] && [ $LAUNCHED -lt ${#SHAPE_ARRAY[@]} ]; then
        echo "  Waiting before launching next shape..."
        sleep 10
    fi
done

echo ""
echo "============================================================"
echo "All test shape evaluations launched!"
echo "  Total launched: $LAUNCHED / ${#SHAPE_ARRAY[@]}"
echo ""
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null | grep -E "eval_|sweep_" || echo "  (none yet, check again in a moment)"
echo ""
echo "Attach to a session with:"
echo "  tmux attach -t <SESSION_NAME>"
echo ""
