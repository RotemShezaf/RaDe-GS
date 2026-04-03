#!/bin/bash
#
# Evaluate geodesic distance propagation using Fast Marching.
# (geodesic_attention variant – uses geodesic_attention config & checkpoint)
#
# USAGE:
#   bash scripts/polynomial/geodesic_attention/evaluate_scripts/evaluate_geodesic.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep7)
#   --gpus N              Number of GPUs (default: 1)
#   --time HH:MM:SS       SLURM time limit (default: 4:00:00)
#   --config PATH         Testing config (default: models/configs/testing.yaml)
#   --model_path PATH     Model checkpoint path
#   --gaussian_output PATH Path to Gaussian output dir
#   --dataset_config PATH Dataset config YAML path
#   --source_indices IDX  Comma-separated source indices
#   --refine_passes N     Post-FM iterative refinement passes (default: 0)
#   --extra "ARGS"        Extra CLI overrides
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep7"
GPUS=1
TIME="4:00:00"
CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/testing.yaml"
MODEL_PATH="/home/rotem.shezaf/RaDe-GS/checkpoints/geodesic_attention/combined_polynomial_ring3/best_model.pth"
GAUSSIAN_OUTPUT=""
DATASET_CONFIG="/home/rotem.shezaf/RaDe-GS/DataSets/configs/polynomial/new_dataset/combined_polynomial_all_one_source.yaml"
SOURCE_INDICES=""
REFINE_PASSES="0"
EXTRA=""
DRY_RUN=false
SESSION_NAME="eval_geodesic_ga"
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
        --gaussian_output)   GAUSSIAN_OUTPUT="$2"; shift 2 ;;
        --dataset_config)    DATASET_CONFIG="$2";  shift 2 ;;
        --source_indices)    SOURCE_INDICES="$2";  shift 2 ;;
        --refine_passes)     REFINE_PASSES="$2";   shift 2 ;;
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
# Resolve project root (four levels up from scripts/polynomial/geodesic_attention/evaluate_scripts/)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"

# ============================================================================
# Build the evaluation command
# ============================================================================
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

EVAL_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
$PYTHON3 geodesic_propagation/evaluate_geodesic.py"

if [[ -n "$MODEL_PATH" ]]; then
    EVAL_CMD="$EVAL_CMD --model_path $MODEL_PATH"
fi
if [[ -n "$GAUSSIAN_OUTPUT" ]]; then
    EVAL_CMD="$EVAL_CMD --gaussian_output $GAUSSIAN_OUTPUT"
fi
if [[ -n "$DATASET_CONFIG" ]]; then
    EVAL_CMD="$EVAL_CMD --dataset_config $DATASET_CONFIG"
fi
if [[ -n "$CONFIG" ]]; then
    EVAL_CMD="$EVAL_CMD --config $CONFIG"
fi
if [[ -n "$SOURCE_INDICES" ]]; then
    EVAL_CMD="$EVAL_CMD --source_indices $SOURCE_INDICES"
fi
if [[ "$REFINE_PASSES" != "0" ]]; then
    EVAL_CMD="$EVAL_CMD --refine_passes $REFINE_PASSES"
fi
if [[ -n "$EXTRA" ]]; then
    EVAL_CMD="$EVAL_CMD $EXTRA"
fi

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS_PER_GPU --time=$TIME --pty bash -c '$EVAL_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Evaluate Geodesic Propagation (geodesic_attention)"
echo "============================================================"
echo "  tmux session:     $SESSION_NAME"
echo "  Node:             $NODE"
echo "  GPUs:             $GPUS"
echo "  Time limit:       $TIME"
echo "  Config:           $CONFIG"
echo "  Model path:       ${MODEL_PATH:-<default>}"
echo "  Gaussian output:  ${GAUSSIAN_OUTPUT:-<default>}"
echo "  Dataset config:   ${DATASET_CONFIG:-<default>}"
echo "  Source indices:    ${SOURCE_INDICES:-<default>}"
echo "  Refine passes:    $REFINE_PASSES"
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
