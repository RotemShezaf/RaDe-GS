#!/bin/bash
#
# Rebuild the 2 orphan-affected meshes inside a tmux session on a SLURM node.
#
# USAGE:
#   bash scripts/polynomial/geodesic/rebuild_orphan_meshes_tmux.sh [options]
#
# OPTIONS:
#   --node NAME     SLURM node (default: gipdeep1)
#   --cpus N        Number of CPUs (default: 20)
#   --time HH:MM:SS Time limit (default: 06:00:00)
#   --session NAME  tmux session name (default: rebuild_orphan)
#   --dry_run       Print commands without executing

set -e

NODE="gipdeep1"
CPUS=20
TIME="06:00:00"
SESSION_NAME="rebuild_orphan"
CONDA_ENV="geo_splat"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)    NODE="$2";          shift 2 ;;
        --cpus)    CPUS="$2";          shift 2 ;;
        --time)    TIME="$2";          shift 2 ;;
        --session) SESSION_NAME="$2";  shift 2 ;;
        --dry_run) DRY_RUN=true;       shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/scripts/polynomial/geodesic/rebuild_orphan_meshes_all.sh"

if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "Error: Script not found: $BUILD_SCRIPT"
    exit 1
fi

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $BUILD_SCRIPT --verbose"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Rebuild Orphan Meshes – tmux launcher"
echo "============================================================"
echo "  tmux session:  $SESSION_NAME"
echo "  Node:          $NODE"
echo "  CPUs:          $CPUS"
echo "  Time limit:    $TIME"
echo "  Build script:  $BUILD_SCRIPT"
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
echo "Launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
