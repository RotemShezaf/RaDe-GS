#!/bin/bash
#
# Autotune geodesic meshes for all polynomial surfaces inside a tmux session
# on a SLURM node.
#
# Creates (or reuses) a tmux session, allocates CPUs on a SLURM node via
# srun, then runs build_geodesic_mesh_polynomial_all_autotune.sh.
#
# This is the autotune version of build_geodesic_mesh_tmux.sh.
#
# USAGE:
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_tmux_autotune.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep12)
#   --cpus N           Number of CPUs to request (default: 36)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: autotune_geo_mesh)
#   --extra "ARGS"     Extra args forwarded to the all-outputs script
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_tmux_autotune.sh
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_tmux_autotune.sh --node gipdeep6 --cpus 20
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_tmux_autotune.sh --extra "--surfaces Paraboloid --skip_existing"
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_tmux_autotune.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep12"
CPUS=36
TIME="24:00:00"
SESSION_NAME="autotune_geo_mesh"
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
BUILD_SCRIPT="$SCRIPT_DIR/scripts/polynomial/geodesic/build_geodesic_mesh_polynomial_all_autotune.sh"

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
bash $BUILD_SCRIPT $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Autotune Geodesic Meshes – tmux launcher"
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
