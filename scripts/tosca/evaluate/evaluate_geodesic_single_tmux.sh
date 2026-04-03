#!/bin/bash
# ============================================================================
# tmux launcher: Evaluate geodesic on a single TOSCA test shape (GPU)
#
# Usage:
#   bash scripts/tosca/evaluate/evaluate_geodesic_single_tmux.sh --model_path <path> --shape cat10
#   bash scripts/tosca/evaluate/evaluate_geodesic_single_tmux.sh --model_path <path> --shape cat10 --node gipdeep7
#   bash scripts/tosca/evaluate/evaluate_geodesic_single_tmux.sh --dry_run --model_path <path> --shape cat10
# ============================================================================

set -e

# --- Defaults ---
NODE="gipdeep7"
GPUS=1
CPUS=10
TIME="04:00:00"
SESSION="eval_geodesic_single"
CONDA_ENV="geo_splat"
MODEL_PATH=""
SHAPE=""
EXTRA_ARGS=""
DRY_RUN=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)         NODE="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --session)      SESSION="$2"; shift 2 ;;
        --conda_env)    CONDA_ENV="$2"; shift 2 ;;
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --shape)        SHAPE="$2"; shift 2 ;;
        --extra)        EXTRA_ARGS="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 --model_path <path> --shape <name> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path PATH    Path to trained geodesic model checkpoint"
            echo "  --shape NAME         Shape name (e.g., cat10, horse18)"
            echo ""
            echo "Options:"
            echo "  --node NAME          SLURM node (default: gipdeep7)"
            echo "  --gpus N             GPUs to request (default: 1)"
            echo "  --cpus N             CPUs to request (default: 10)"
            echo "  --time HH:MM:SS     SLURM time limit (default: 04:00:00)"
            echo "  --session NAME       tmux session name (default: eval_geodesic_single)"
            echo "  --conda_env NAME     Conda environment (default: geo_splat)"
            echo "  --extra \"ARGS\"       Extra args forwarded to evaluate_geodesic_single.sh"
            echo "  --dry_run            Print without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model_path is required"; exit 1
fi
if [[ -z "$SHAPE" ]]; then
    echo "ERROR: --shape is required"; exit 1
fi

# --- Build command ---
INNER_CMD="bash scripts/tosca/evaluate/evaluate_geodesic_single.sh --model_path $MODEL_PATH --shape $SHAPE $EXTRA_ARGS"

FULL_CMD="srun --nodelist=${NODE} --gres=gpu:${GPUS} --cpus-per-task=${CPUS} --time=${TIME} --pty bash -c '\
    conda activate ${CONDA_ENV} && \
    cd ~/RaDe-GS && \
    ${INNER_CMD}'"

echo "=========================================="
echo "tmux: Evaluate Geodesic - Single Shape"
echo "=========================================="
echo "Node:     ${NODE}"
echo "GPUs:     ${GPUS}"
echo "CPUs:     ${CPUS}"
echo "Time:     ${TIME}"
echo "Session:  ${SESSION}"
echo "Model:    ${MODEL_PATH}"
echo "Shape:    ${SHAPE}"
echo "=========================================="

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would execute in tmux session '${SESSION}':"
    echo "$FULL_CMD"
    exit 0
fi

# Create or reuse tmux session
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Reusing existing tmux session: ${SESSION}"
    tmux send-keys -t "$SESSION" "$FULL_CMD" C-m
else
    echo "Creating new tmux session: ${SESSION}"
    tmux new-session -d -s "$SESSION" "$FULL_CMD"
fi

echo ""
echo "Attach with: tmux attach -t ${SESSION}"
echo "Detach with: Ctrl+B, then D"
