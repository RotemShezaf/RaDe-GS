#!/bin/bash
#
# Generate polynomial training patches (new_dataset) inside a tmux session.
#
# Creates (or reuses) a tmux session called "gen_patches_nd", allocates
# CPUs via srun, then runs generate_polynomial_training_patches_new_dataset.sh.
#
# USAGE:
#   bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           Number of CPUs to request (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: gen_patches_nd)
#   --extra "ARGS"     Extra args forwarded to generate_polynomial_training_patches_new_dataset.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh
#   bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh --node gipdeep12 --cpus 40
#   bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh --extra "--num_iterations 5000"
#   bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep10"
CPUS=70
TIME="24:00:00"
SESSION_NAME="gen_patches_nd"
CONDA_ENV="geo_splat"
EXTRA=""
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)    NODE="$2";         shift 2 ;;
        --cpus)    CPUS="$2";         shift 2 ;;
        --time)    TIME="$2";         shift 2 ;;
        --session) SESSION_NAME="$2"; shift 2 ;;
        --extra)   EXTRA="$2";        shift 2 ;;
        --dry_run) DRY_RUN=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve project root
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GENERATE_SCRIPT="$SCRIPT_DIR/scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh"

# ============================================================================
# Build the command that runs inside srun
# ============================================================================
INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $GENERATE_SCRIPT $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Generate Polynomial Training Patches (new_dataset) – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Generate script: $GENERATE_SCRIPT"
echo "  Extra args:      ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

# ============================================================================
# Create / reuse tmux session and launch
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
echo "Launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
