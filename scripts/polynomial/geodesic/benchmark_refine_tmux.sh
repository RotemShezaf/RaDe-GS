#!/bin/bash
#
# Run the benchmark_refine parameter sweep inside a tmux session on a SLURM node.
#
# Creates (or reuses) a tmux session, allocates CPUs on a SLURM node via
# srun, then runs benchmark_refine_all.sh.
#
# Analogous to build_geodesic_mesh_tmux_autotune.sh.
#
# USAGE:
#   bash scripts/polynomial/geodesic/benchmark_refine_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep1)
#   --cpus N           Number of CPUs to request (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 04:00:00)
#   --session NAME     tmux session name (default: bench_refine)
#   --extra "ARGS"     Extra args forwarded to benchmark_refine_all.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/benchmark_refine_tmux.sh
#   bash scripts/polynomial/geodesic/benchmark_refine_tmux.sh --node gipdeep6 --cpus 30
#   bash scripts/polynomial/geodesic/benchmark_refine_tmux.sh --extra "--surfaces Paraboloid"
#   bash scripts/polynomial/geodesic/benchmark_refine_tmux.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep1"
CPUS=60
TIME="08:00:00"
SESSION_NAME="bench_refine"
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/scripts/polynomial/geodesic/benchmark_refine_all.sh"

if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "Error: Script not found: $BUILD_SCRIPT"
    exit 1
fi

# ============================================================================
# Build the command that runs inside srun
# ============================================================================
INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $BUILD_SCRIPT --workers $CPUS $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Benchmark refine_bad_triangles — tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Build script:    $BUILD_SCRIPT"
echo "  Extra args:      ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] — nothing executed."
    exit 0
fi

# ============================================================================
# Create / reuse tmux session and launch
# ============================================================================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists — sending command..."
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
