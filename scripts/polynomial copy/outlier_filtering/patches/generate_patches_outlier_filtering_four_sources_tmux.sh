#!/bin/bash
#
# Generate polynomial training patches (outlier filtering, 4-source) inside a tmux session.
#
# Same as generate_patches_outlier_filtering_tmux.sh but defaults to 4-source
# configs (paraboloid_all.yaml, saddle_all.yaml, hyperbolic_paraboloid_all.yaml).
#
# USAGE:
#   bash scripts/polynomial/outlier_filtering/patches/generate_patches_outlier_filtering_four_sources_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           Number of CPUs to request (default: 70)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: gen_patches_of_4s)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --extra "ARGS"     Extra args forwarded to the generation script
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/outlier_filtering/patches/generate_patches_outlier_filtering_four_sources_tmux.sh
#   bash scripts/polynomial/outlier_filtering/patches/generate_patches_outlier_filtering_four_sources_tmux.sh --node gipdeep12

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep12"
CPUS=35
TIME="24:00:00"
SESSION_NAME="gen_patches_of_4s"
CONDA_ENV="geo_splat"
EXTRA=""
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)      NODE="$2";         shift 2 ;;
        --cpus)      CPUS="$2";         shift 2 ;;
        --time)      TIME="$2";         shift 2 ;;
        --session)   SESSION_NAME="$2"; shift 2 ;;
        --conda_env) CONDA_ENV="$2";    shift 2 ;;
        --extra)     EXTRA="$2";        shift 2 ;;
        --dry_run)   DRY_RUN=true;      shift   ;;
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
GENERATE_SCRIPT="$SCRIPT_DIR/scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh"

# ============================================================================
# Build the command that runs inside srun — always passes --four_sources
# ============================================================================
INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $GENERATE_SCRIPT --four_sources $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Generate Polynomial Patches (outlier filtering, 4-source) – tmux"
echo "============================================================"
echo "  tmux session:  $SESSION_NAME"
echo "  Node:          $NODE"
echo "  CPUs:          $CPUS"
echo "  Time limit:    $TIME"
echo "  Conda env:     $CONDA_ENV"
echo "  Extra args:    ${EXTRA:-<none>}"
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
echo "Patch generation launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
