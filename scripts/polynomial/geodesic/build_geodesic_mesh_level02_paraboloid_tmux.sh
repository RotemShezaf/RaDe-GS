#!/bin/bash
#
# Build geodesic meshes for level 02 only, inside a tmux session on a SLURM node.
#
# USAGE:
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_level02_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           Number of CPUs to request (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: build_geo_mesh_l02)
#   --skip_existing    Skip outputs that already have a geodesic_mesh directory
#   --extra "ARGS"     Extra args forwarded to build_geodesic_mesh_polynomial_all.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_level02_tmux.sh
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_level02_tmux.sh --node gipdeep9 --cpus 55
#   bash scripts/polynomial/geodesic/build_geodesic_mesh_level02_tmux.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep1"
CPUS=80
TIME="24:00:00"
SESSION_NAME="build_geo_mesh_l02"
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
        --skip_existing) SKIP_EXISTING="--skip_existing";  shift   ;;
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
# Resolve project root
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/scripts/polynomial/geodesic/build_geodesic_mesh_polynomial_all.sh"

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
bash $BUILD_SCRIPT --verbose --levels 02 --surfaces Paraboloid --light_ids 04 $SKIP_EXISTING $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Build Geodesic Meshes (level 02) – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Build script:    $BUILD_SCRIPT"
echo "  Level:           02"
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
