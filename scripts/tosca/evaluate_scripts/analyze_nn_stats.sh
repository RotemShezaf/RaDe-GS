#!/bin/bash
#
# Analyze k-NN distance statistics from Gaussian PLY files.
# (TOSCA variant – uses combined_tosca_ring3 config & checkpoint)
#
# USAGE:
#   bash scripts/tosca/evaluate_scripts/analyze_nn_stats.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep7)
#   --time HH:MM:SS       SLURM time limit (default: 2:00:00)
#   --config PATH         Model config YAML
#   --model_path PATH     Model checkpoint path
#   --gaussian_dir PATH   Gaussian output directory
#   --extra "ARGS"        Extra CLI overrides
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep7"
TIME="2:00:00"
CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/tosca/combined_tosca_ring3.yaml"
MODEL_PATH="/home/rotem.shezaf/RaDe-GS/checkpoints/combined_tosca_ring3/best_model.pth"
GAUSSIAN_DIR=""
EXTRA=""
DRY_RUN=false
SESSION_NAME="eval_nn_stats_tosca${EVAL_SHAPE_SUFFIX:-}"
CONDA_ENV="geo_splat"
CPUS_PER_GPU=6

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)              NODE="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --config)            CONFIG="$2";          shift 2 ;;
        --model_path)        MODEL_PATH="$2";      shift 2 ;;
        --gaussian_dir)      GAUSSIAN_DIR="$2";    shift 2 ;;
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
$PYTHON3 geodesic_propagation/analyze_nn_stats.py \
  --config $CONFIG \
  --model_path $MODEL_PATH"

if [[ -n "$GAUSSIAN_DIR" ]]; then
    EVAL_CMD="$EVAL_CMD --gaussian_dir $GAUSSIAN_DIR"
fi
if [[ -n "$EXTRA" ]]; then
    EVAL_CMD="$EVAL_CMD $EXTRA"
fi

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:1 --cpus-per-task=$CPUS_PER_GPU --time=$TIME --pty bash -c '$EVAL_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Analyze NN Statistics (TOSCA)"
echo "============================================================"
echo "  tmux session:     $SESSION_NAME"
echo "  Node:             $NODE"
echo "  Time limit:       $TIME"
echo "  Config:           $CONFIG"
echo "  Model path:       ${MODEL_PATH:-<default>}"
echo "  Gaussian dir:     ${GAUSSIAN_DIR:-<default>}"
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
echo "Analysis launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
