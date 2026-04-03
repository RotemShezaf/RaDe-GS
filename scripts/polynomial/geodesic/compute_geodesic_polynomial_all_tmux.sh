#!/bin/bash
#
# Launch compute_geodesic_polynomial_all.sh inside a tmux session
# on a SLURM node.
#
# Creates (or reuses) a tmux session called "compute_geo", allocates
# CPUs on the target node via srun, then runs
# compute_geodesic_polynomial_all.sh.
#
# USAGE:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep9)
#   --cpus N           Number of CPUs to request (default: 20)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: compute_geo)
#   --skip_existing    Skip combinations that already have gt_geodesic.npz on disk
#   --extra "ARGS"     Extra args forwarded to compute_geodesic_polynomial_all.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all_tmux.sh
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all_tmux.sh --node gipdeep10 --cpus 55
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all_tmux.sh \
#       --extra "--surfaces Paraboloid --textures blue"
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all_tmux.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep10"
CPUS=60
TIME="24:00:00"
SESSION_NAME="compute_geo"
CONDA_ENV="geo_splat"
EXTRA=""
DRY_RUN=false
SKIP_EXISTING=""

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)    NODE="$2";         shift 2 ;;
        --cpus)    CPUS="$2";         shift 2 ;;
        --time)    TIME="$2";         shift 2 ;;
        --session)        SESSION_NAME="$2"; shift 2 ;;
        --skip_existing)  SKIP_EXISTING="--skip_existing";  shift   ;;
        --extra)          EXTRA="$2";         shift 2 ;;
        --dry_run)        DRY_RUN=true;        shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve project root and compute script
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/scripts/polynomial/geodesic/compute_geodesic_polynomial_all.sh"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Build the command that runs inside srun
# ============================================================================
INNER_CMD="cd $PROJECT_ROOT && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $COMPUTE_SCRIPT --verbose $SKIP_EXISTING $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Compute Geodesic Distances – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Compute script:  $COMPUTE_SCRIPT"
echo "  Skip existing:   ${SKIP_EXISTING:-no}"
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
