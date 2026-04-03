#!/bin/bash
#
# Launch TOSCA geodesic computation inside a tmux session on a SLURM node.
#
# Creates (or reuses) a tmux session, allocates CPUs on a compute node
# via srun, then runs compute_geodesic_tosca_batched.sh.
#
# USAGE:
#   bash scripts/compute_geodesic_tosca_tmux.sh <gaussian_output> <shape> [options]
#
# ARGUMENTS:
#   gaussian_output   Path to Gaussian splatting output folder
#   shape             TOSCA shape name (e.g., cat0, dog0, horse0)
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep9)
#   --cpus N           Number of CPUs to request (default: 20)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: tosca_geodesic)
#   --conda_env NAME   Conda environment name (default: geo_splat)
#   --extra "ARGS"     Extra args forwarded to compute_geodesic_tosca_batched.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   # Run geodesic for cat0 decoupled_appearance on gipdeep9
#   bash scripts/compute_geodesic_tosca_tmux.sh \
#       TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
#       cat0
#
#   # Custom node and more sources
#   bash scripts/compute_geodesic_tosca_tmux.sh \
#       TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
#       cat0 \
#       --node gipdeep12 --cpus 40 \
#       --extra "--num_sources 400 --n_batches 16"
#
#   # Dry run
#   bash scripts/compute_geodesic_tosca_tmux.sh \
#       TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
#       cat0 --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep9"
CPUS=20
TIME="24:00:00"
SESSION_NAME="tosca_geodesic"
CONDA_ENV="geo_splat"
EXTRA=""
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
if [ $# -lt 2 ]; then
    echo "Usage: $0 <gaussian_output> <shape> [options]"
    echo ""
    echo "Arguments:"
    echo "  gaussian_output   Path to Gaussian splatting output folder"
    echo "  shape             TOSCA shape name (e.g., cat0)"
    echo ""
    echo "Options:"
    echo "  --node NAME        SLURM node (default: gipdeep9)"
    echo "  --cpus N           CPUs to request (default: 20)"
    echo "  --time HH:MM:SS    SLURM time limit (default: 24:00:00)"
    echo "  --session NAME     tmux session name (default: tosca_geodesic)"
    echo "  --conda_env NAME   Conda environment (default: geo_splat)"
    echo "  --extra \"ARGS\"     Extra args for compute_geodesic_tosca_batched.sh"
    echo "  --dry_run          Print commands without executing"
    exit 1
fi

GAUSSIAN_OUTPUT="$1"
SHAPE="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)       NODE="$2";         shift 2 ;;
        --cpus)       CPUS="$2";         shift 2 ;;
        --time)       TIME="$2";         shift 2 ;;
        --session)    SESSION_NAME="$2"; shift 2 ;;
        --conda_env)  CONDA_ENV="$2";    shift 2 ;;
        --extra)      EXTRA="$2";        shift 2 ;;
        --dry_run)    DRY_RUN=true;      shift   ;;
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
BATCHED_SCRIPT="$SCRIPT_DIR/scripts/tosca/geodesic/compute_geodesic_tosca_batched.sh"

if [ ! -f "$BATCHED_SCRIPT" ]; then
    echo "Error: Batched script not found: $BATCHED_SCRIPT"
    exit 1
fi

# ============================================================================
# Build the command that runs inside srun
# ============================================================================
INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $BATCHED_SCRIPT $GAUSSIAN_OUTPUT $SHAPE $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "TOSCA Geodesic Computation – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Shape:           $SHAPE"
echo "  Gaussian output: $GAUSSIAN_OUTPUT"
echo "  Batched script:  $BATCHED_SCRIPT"
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
