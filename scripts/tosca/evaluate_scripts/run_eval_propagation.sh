#!/bin/bash
#
# Debug evaluation: run FM propagation on full scene, stop after N visited.
# (TOSCA variant – uses combined_tosca_ring3 config & checkpoint)
#
# USAGE:
#   bash scripts/tosca/evaluate_scripts/run_eval_propagation.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep10)
#   --gpus N              Number of GPUs (default: 1)
#   --time HH:MM:SS       SLURM time limit (default: 4:00:00)
#   --model_path PATH     Model checkpoint path
#   --train_config PATH   Training config YAML path
#   --dataset_config PATH Dataset config YAML path
#   --gaussian_dir PATH   Gaussian output directory
#   --max_visited N       Stop after N points visited (default: 2000)
#   --refine_passes N     Post-FM iterative refinement passes (default: 0)
#   --device DEVICE       Compute device: cpu or cuda (default: cuda)
#   --extra "ARGS"        Extra CLI overrides
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep8"
GPUS=1
TIME="4:00:00"
MODEL_PATH=""
TRAIN_CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/tosca/combined_tosca_ring3.yaml"
DATASET_CONFIG=""
GAUSSIAN_DIR="TrainData/TOSCA/SyntheticColmapData/blue_texture/cat2/high_res/decoupled_appearance/output"
MAX_VISITED="5000"
REFINE_PASSES="0"
DEVICE=""
EXTRA=""
DRY_RUN=false
SESSION_NAME="eval_propagation_tosca${EVAL_SHAPE_SUFFIX:-}"
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
        --model_path)        MODEL_PATH="$2";      shift 2 ;;
        --train_config)      TRAIN_CONFIG="$2";    shift 2 ;;
        --dataset_config)    DATASET_CONFIG="$2";  shift 2 ;;
        --gaussian_dir)      GAUSSIAN_DIR="$2";    shift 2 ;;
        --max_visited)       MAX_VISITED="$2";     shift 2 ;;
        --refine_passes)     REFINE_PASSES="$2";   shift 2 ;;
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
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

EVAL_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
$PYTHON3 geodesic_propagation/run_eval_propagation.py \
  --train_config $TRAIN_CONFIG \
  --max_visited $MAX_VISITED"

if [[ -n "$MODEL_PATH" ]]; then
    EVAL_CMD="$EVAL_CMD --model_path $MODEL_PATH"
fi
if [[ -n "$DATASET_CONFIG" ]]; then
    EVAL_CMD="$EVAL_CMD --dataset_config $DATASET_CONFIG"
fi
if [[ -n "$GAUSSIAN_DIR" ]]; then
    EVAL_CMD="$EVAL_CMD --gaussian_dir $GAUSSIAN_DIR"
fi
if [[ "$REFINE_PASSES" != "0" ]]; then
    EVAL_CMD="$EVAL_CMD --refine_passes $REFINE_PASSES"
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
echo "Debug: FM Propagation Evaluation (TOSCA, first N visited)"
echo "============================================================"
echo "  tmux session:     $SESSION_NAME"
echo "  Node:             $NODE"
echo "  GPUs:             $GPUS"
echo "  Time limit:       $TIME"
echo "  Model path:       $MODEL_PATH"
echo "  Train config:     $TRAIN_CONFIG"
echo "  Dataset config:   $DATASET_CONFIG"
echo "  Gaussian dir:     $GAUSSIAN_DIR"
echo "  Max visited:      $MAX_VISITED"
echo "  Refine passes:    $REFINE_PASSES"
echo "  Device:           ${DEVICE:-<auto>}"
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
