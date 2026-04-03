#!/bin/bash
# ============================================================================
# tmux launcher: Evaluate geodesic on ALL TOSCA test shapes (GPU)
#
# Test set = last pose per animal (12 shapes):
#   cat10, centaur5, david14, dog10, gorilla20, horse18,
#   lioness16, michael19, seahorse5, shark0, victoria25, wolf2
#
# Usage:
#   bash scripts/tosca/evaluate/evaluate_geodesic_all_tmux.sh --model_path <path>
#   bash scripts/tosca/evaluate/evaluate_geodesic_all_tmux.sh --model_path <path> --node gipdeep7
#   bash scripts/tosca/evaluate/evaluate_geodesic_all_tmux.sh --model_path <path> --shapes "cat10,dog10"
#   bash scripts/tosca/evaluate/evaluate_geodesic_all_tmux.sh --dry_run --model_path <path>
# ============================================================================

set -e

# --- Defaults ---
NODE="gipdeep10"
GPUS=1
CPUS=70
TIME="24:00:00"
SESSION="eval_geodesic_all"
CONDA_ENV="geo_splat"
MODEL_PATH=""
SHAPES=""
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
        --shapes)       SHAPES="$2"; shift 2 ;;
        --extra)        EXTRA_ARGS="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Launches evaluation of all TOSCA test shapes in a tmux session on a GPU node."
            echo ""
            echo "Required:"
            echo "  --model_path PATH    Path to trained geodesic model checkpoint"
            echo ""
            echo "Options:"
            echo "  --shapes LIST        Comma-separated shapes (default: all 12 test shapes)"
            echo "  --node NAME          SLURM node (default: gipdeep7)"
            echo "  --gpus N             GPUs to request (default: 1)"
            echo "  --cpus N             CPUs to request (default: 10)"
            echo "  --time HH:MM:SS     SLURM time limit (default: 24:00:00)"
            echo "  --session NAME       tmux session name (default: eval_geodesic_all)"
            echo "  --conda_env NAME     Conda environment (default: geo_splat)"
            echo "  --extra \"ARGS\"       Extra args forwarded to evaluate_geodesic_all.sh"
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

# --- Build inner command ---
INNER_CMD="bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --model_path $MODEL_PATH"

if [[ -n "$SHAPES" ]]; then
    INNER_CMD="$INNER_CMD --shapes $SHAPES"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    INNER_CMD="$INNER_CMD $EXTRA_ARGS"
fi

FULL_CMD="srun --nodelist=${NODE} --gres=gpu:${GPUS} --cpus-per-task=${CPUS} --time=${TIME} --pty bash -c '\
    conda activate ${CONDA_ENV} && \
    cd ~/RaDe-GS && \
    ${INNER_CMD}'"

echo "=========================================="
echo "tmux: Evaluate Geodesic - All Test Shapes"
echo "=========================================="
echo "Node:     ${NODE}"
echo "GPUs:     ${GPUS}"
echo "CPUs:     ${CPUS}"
echo "Time:     ${TIME}"
echo "Session:  ${SESSION}"
echo "Model:    ${MODEL_PATH}"
if [[ -n "$SHAPES" ]]; then
    echo "Shapes:   ${SHAPES}"
else
    echo "Shapes:   all 12 test shapes"
fi
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
