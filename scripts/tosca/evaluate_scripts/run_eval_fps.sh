#!/bin/bash
#
# Geodesic propagation evaluation with optional FPS downsampling and GPU support.
# (TOSCA variant – uses combined_tosca_ring3 config & checkpoint)
#
# USAGE:
#   bash scripts/tosca/evaluate_scripts/run_eval_fps.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep10)
#   --gpus N              Number of GPUs (default: 1)
#   --time HH:MM:SS       SLURM time limit (default: 4:00:00)
#   --config PATH         Testing config
#   --model_path PATH     Model checkpoint path
#   --train_config PATH   Training config YAML path
#   --dataset_config PATH Dataset config YAML path
#   --gaussian_dir PATH   Gaussian output directory
#   --fps_target N        FPS target number (0 = no downsampling, default: 0)
#   --device DEVICE       Compute device: cpu or cuda (default: cuda)
#   --extra "ARGS"        Extra CLI overrides
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep10"
GPUS=1
TIME="4:00:00"
CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/testing.yaml"
MODEL_PATH=""
TRAIN_CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/tosca/combined_tosca_ring3.yaml"
DATASET_CONFIG=""
GAUSSIAN_DIR="TrainData/TOSCA/SyntheticColmapData/blue_texture/cat2/high_res/decoupled_appearance/output"
FPS_TARGET=""
DEVICE=""
EXTRA=""
DRY_RUN=false
SESSION_NAME="eval_fps_tosca${EVAL_SHAPE_SUFFIX:-}"
CONDA_ENV="geo_splat"
CPUS_PER_GPU=6

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
        --gaussian_dir)      GAUSSIAN_DIR="$2";    shift 2 ;;
        --fps_target)        FPS_TARGET="$2";      shift 2 ;;
        --device)            DEVICE="$2";          shift 2 ;;
        --extra)             EXTRA="$2";           shift 2 ;;
        --dry_run)           DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve project root (three levels up from scripts/tosca/evaluate_scripts/)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# ============================================================================
# Build the evaluation command
# ============================================================================
EVAL_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
python geodesic_propagation/run_eval_fps.py"

if [[ -n "$MODEL_PATH" ]]; then
    EVAL_CMD="$EVAL_CMD --model_path $MODEL_PATH"
fi
if [[ -n "$TRAIN_CONFIG" ]]; then
    EVAL_CMD="$EVAL_CMD --train_config $TRAIN_CONFIG"
fi
if [[ -n "$DATASET_CONFIG" ]]; then
    EVAL_CMD="$EVAL_CMD --dataset_config $DATASET_CONFIG"
fi
if [[ -n "$GAUSSIAN_DIR" ]]; then
    EVAL_CMD="$EVAL_CMD --gaussian_dir $GAUSSIAN_DIR"
fi
if [[ -n "$FPS_TARGET" ]]; then
    EVAL_CMD="$EVAL_CMD --fps_target $FPS_TARGET"
fi
if [[ -n "$DEVICE" ]]; then
    EVAL_CMD="$EVAL_CMD --device $DEVICE"
fi
if [[ -n "$EXTRA" ]]; then
    EVAL_CMD="$EVAL_CMD $EXTRA"
fi

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS_PER_GPU --time=$TIME --pty bash -c '$EVAL_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Evaluate Geodesic Propagation with FPS (TOSCA)"
echo "============================================================"
echo "  tmux session:     $SESSION_NAME"
echo "  Node:             $NODE"
echo "  GPUs:             $GPUS"
echo "  Time limit:       $TIME"
echo "  Config:           $CONFIG"
echo "  Model path:       ${MODEL_PATH:-<default>}"
echo "  Train config:     ${TRAIN_CONFIG:-<default>}"
echo "  Dataset config:   ${DATASET_CONFIG:-<default>}"
echo "  Gaussian dir:     ${GAUSSIAN_DIR:-<default>}"
echo "  FPS target:       ${FPS_TARGET:-<default>}"
echo "  Device:           ${DEVICE:-<default: cuda>}"
echo "  Conda env:        $CONDA_ENV"
echo "  Extra args:       ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

# ============================================================================
# Create / attach tmux session and launch
# ============================================================================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists – sending command..."
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
else
    echo "Creating tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    sleep 1
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
fi

echo ""
echo "Evaluation launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
