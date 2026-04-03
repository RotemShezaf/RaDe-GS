#!/bin/bash
#
# Launch TOSCA training patch generation inside a tmux session.
#
# USAGE:
#   bash scripts/tosca/patches/generate_patches_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep12)
#   --cpus N           CPUs to request (default: 40)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: tosca_patches)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --animals LIST     Comma-separated animals (default: all 12)
#   --scale_opacity    Generate patches with scale+opacity
#   --extra "ARGS"     Extra args forwarded to generate_training_patches.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/patches/generate_patches_tmux.sh
#   bash scripts/tosca/patches/generate_patches_tmux.sh --scale_opacity
#   bash scripts/tosca/patches/generate_patches_tmux.sh --animals "cat,dog"

set -e

NODE="gipdeep12"
CPUS=35
TIME="24:00:00"
SESSION_NAME="tosca_patches"
CONDA_ENV="geo_splat"
ANIMALS=""
SCALE_OPACITY=""
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)          NODE="$2";                     shift 2 ;;
        --cpus)          CPUS="$2";                     shift 2 ;;
        --time)          TIME="$2";                     shift 2 ;;
        --session)       SESSION_NAME="$2";             shift 2 ;;
        --conda_env)     CONDA_ENV="$2";                shift 2 ;;
        --animals)       ANIMALS="$2";                  shift 2 ;;
        --scale_opacity) SCALE_OPACITY="--scale_opacity"; shift ;;
        --extra)         EXTRA="$2";                    shift 2 ;;
        --dry_run)       DRY_RUN=true;                  shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

ANIMAL_ARG=""
[ -n "$ANIMALS" ] && ANIMAL_ARG="--animals $ANIMALS"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/patches/generate_training_patches.sh $ANIMAL_ARG $SCALE_OPACITY $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Training Patches – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Animals:         ${ANIMALS:-all}"
echo "  Scale+Opacity:   ${SCALE_OPACITY:-no}"
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
