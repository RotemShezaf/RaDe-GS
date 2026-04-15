#!/bin/bash
#
# Launch compute_geodesic_polynomial_all.sh for level 04 only,
# inside a tmux session on a SLURM node.
#
# USAGE:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_level04_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           Number of CPUs to request (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: compute_geo_l04)
#   --extra "ARGS"     Extra args forwarded to compute_geodesic_polynomial_all.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_level04_tmux.sh
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_level04_tmux.sh --node gipdeep9 --cpus 55
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_level04_tmux.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep1"
CPUS=80
TIME="24:00:00"
SESSION_NAME="compute_geo_l04_copy"
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
        --session)        SESSION_NAME="$2"; shift 2 ;;
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
bash $COMPUTE_SCRIPT --verbose --levels 04 --surfaces HyperbolicParaboloid  --light_ids 1 --num_parallel_outputs 1 $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Compute Geodesic Distances (level 04) – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Compute script:  $COMPUTE_SCRIPT"
echo "  Level:           04"
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
