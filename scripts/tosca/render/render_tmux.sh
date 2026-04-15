#!/bin/bash
#
# Launch TOSCA rendering inside a tmux session on a SLURM node.
#
# USAGE:
#   bash scripts/tosca/render/render_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep9)
#   --cpus N           CPUs to request (default: 20)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: tosca_render)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --shapes LIST      Comma-separated shapes with index (default: all)
#   --animals LIST     Animal names WITHOUT index (e.g. "cat,dog")
#   --extra "ARGS"     Extra args forwarded to render_all_blue.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/render/render_tmux.sh
#   bash scripts/tosca/render/render_tmux.sh --node gipdeep12 --shapes "cat0,cat1,dog0"
#   bash scripts/tosca/render/render_tmux.sh --dry_run

set -e

NODE="gipdeep10"
CPUS=1
GPUS=2
TIME="24:00:00"
SESSION_NAME="tosca_render"
CONDA_ENV="geo_splat"
SHAPES=""
ANIMALS=""
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)       NODE="$2";         shift 2 ;;
        --cpus)       CPUS="$2";         shift 2 ;;
        --gpus)       GPUS="$2";         shift 2 ;;
        --time)       TIME="$2";         shift 2 ;;
        --session)    SESSION_NAME="$2"; shift 2 ;;
        --conda_env)  CONDA_ENV="$2";    shift 2 ;;
        --shapes)     SHAPES="$2";       shift 2 ;;
        --animals)    ANIMALS="$2";      shift 2 ;;
        --extra)      EXTRA="$2";        shift 2 ;;
        --dry_run)    DRY_RUN=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SHAPE_ARG=""
[ -n "$ANIMALS" ] && SHAPE_ARG="--animals $ANIMALS"
[ -z "$SHAPE_ARG" ] && [ -n "$SHAPES" ] && SHAPE_ARG="--shapes $SHAPES"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/render/render_all_blue.sh $SHAPE_ARG $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --gres=gpu:$GPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Rendering – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  GPUs:            $GPUS"
echo "  Time limit:      $TIME"
  echo "  Conda env:       $CONDA_ENV"
  echo "  Animals:         ${ANIMALS:-<not set>}"
  echo "  Shapes:          ${SHAPES:-all}"
echo "  Extra:           ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' exists – sending command..."
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
else
    echo "Creating tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    sleep 1
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
fi

echo ""
echo "Launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
