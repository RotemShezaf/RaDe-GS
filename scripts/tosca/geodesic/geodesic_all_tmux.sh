#!/bin/bash
#
# Launch TOSCA geodesic computation for ALL shapes inside a tmux session.
#
# USAGE:
#   bash scripts/tosca/geodesic/geodesic_all_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep9)
#   --cpus N           CPUs to request (default: 20)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: tosca_geodesic_all)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --shapes LIST      Comma-separated shapes (default: all)
#   --extra "ARGS"     Extra args forwarded to compute_geodesic_all_blue.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/geodesic/geodesic_all_tmux.sh
#   bash scripts/tosca/geodesic/geodesic_all_tmux.sh --node gipdeep12 --cpus 40
#   bash scripts/tosca/geodesic/geodesic_all_tmux.sh --shapes "cat0,cat1"

set -e

NODE="gipdeep8"
CPUS=70
TIME="24:00:00"
SESSION_NAME="tosca_geodesic_all"
CONDA_ENV="geo_splat"
SHAPES=""
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)       NODE="$2";         shift 2 ;;
        --cpus)       CPUS="$2";         shift 2 ;;
        --time)       TIME="$2";         shift 2 ;;
        --session)    SESSION_NAME="$2"; shift 2 ;;
        --conda_env)  CONDA_ENV="$2";    shift 2 ;;
        --shapes)     SHAPES="$2";       shift 2 ;;
        --extra)      EXTRA="$2";        shift 2 ;;
        --dry_run)    DRY_RUN=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SHAPE_ARG=""
[ -n "$SHAPES" ] && SHAPE_ARG="--shapes $SHAPES"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh $SHAPE_ARG $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Geodesic (all shapes) – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Shapes:          ${SHAPES:-all}"
echo "  Extra:           ${EXTRA:-<none>}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' exists – sending command..."
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
