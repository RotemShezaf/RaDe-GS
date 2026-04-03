#!/bin/bash
#
# Launch TOSCA training patch generation (outlier filtering) inside a tmux session.
#
# USAGE:
#   bash scripts/tosca/outlier_filtering/patches/generate_patches_outlier_filtering_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           CPUs to request (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: tosca_of_patches)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --animals LIST     Comma-separated animals (default: all 9)
#   --extra "ARGS"     Extra args forwarded to generate_tosca_training_patches_outlier_filtering.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/outlier_filtering/patches/generate_patches_outlier_filtering_tmux.sh
#   bash scripts/tosca/outlier_filtering/patches/generate_patches_outlier_filtering_tmux.sh --animals "cat,dog"
#   bash scripts/tosca/outlier_filtering/patches/generate_patches_outlier_filtering_tmux.sh --dry_run

set -e

NODE="gipdeep10"
CPUS=65
TIME="24:00:00"
SESSION_NAME="tosca_of_patches"
CONDA_ENV="geo_splat"
ANIMALS=""
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)          NODE="$2";         shift 2 ;;
        --cpus)          CPUS="$2";         shift 2 ;;
        --time)          TIME="$2";         shift 2 ;;
        --session)       SESSION_NAME="$2"; shift 2 ;;
        --conda_env)     CONDA_ENV="$2";    shift 2 ;;
        --animals)       ANIMALS="$2";      shift 2 ;;
        --extra)         EXTRA="$2";        shift 2 ;;
        --dry_run)       DRY_RUN=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

ANIMAL_ARG=""
[ -n "$ANIMALS" ] && ANIMAL_ARG="--animals $ANIMALS"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/outlier_filtering/patches/generate_tosca_training_patches_outlier_filtering.sh $ANIMAL_ARG $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Generate TOSCA Patches (outlier filtering) – tmux"
echo "============================================================"
echo "  tmux session:  $SESSION_NAME"
echo "  Node:          $NODE"
echo "  CPUs:          $CPUS"
echo "  Time limit:    $TIME"
echo "  Conda env:     $CONDA_ENV"
echo "  Animals:       ${ANIMALS:-all}"
echo "  Extra args:    ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

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
echo "Patch generation launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
