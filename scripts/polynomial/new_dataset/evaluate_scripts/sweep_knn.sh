#!/bin/bash
#
# Sweep kNN k values for polynomial surfaces.
# (new_dataset variant – uses new_dataset config & checkpoint)
#
# Finds the best k that eliminates "no-closer-neighbor" fallback points
# while keeping ring-3 neighbor counts within the 128 cap.
#
# USAGE:
#   bash scripts/polynomial/new_dataset/evaluate_scripts/sweep_knn.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep10)
#   --gpus N              Number of GPUs (default: 1)
#   --time HH:MM:SS       SLURM time limit (default: 4:00:00)
#   --k_values LIST       Comma-separated k values (default: 8,10,11,12,13,14,15,16,18,20)
#   --ring N              Ring level (default: 3)
#   --surfaces LIST       Comma-separated surfaces (default: all)
#   --levels LIST         Comma-separated levels (default: level_03)
#   --lights LIST         Comma-separated lights (default: light_0)
#   --extra "ARGS"        Extra CLI overrides
#   --dry_run             Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/new_dataset/evaluate_scripts/sweep_knn.sh
#   bash scripts/polynomial/new_dataset/evaluate_scripts/sweep_knn.sh --k_values 8,10,12,14 --levels level_04

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep10"
GPUS=1
TIME="4:00:00"
K_VALUES="8,10,11,12,13,14,15,16,18,20"
RING="3"
SURFACES=""
LEVELS="level_03"
LIGHTS="light_0"
EXTRA=""
DRY_RUN=false
SESSION_NAME="sweep_knn_nd"
CONDA_ENV="geo_splat"
CPUS_PER_GPU=6

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)              NODE="$2";            shift 2 ;;
        --gpus)              GPUS="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --k_values)          K_VALUES="$2";        shift 2 ;;
        --ring)              RING="$2";            shift 2 ;;
        --surfaces)          SURFACES="$2";        shift 2 ;;
        --levels)            LEVELS="$2";          shift 2 ;;
        --lights)            LIGHTS="$2";          shift 2 ;;
        --extra)             EXTRA="$2";           shift 2 ;;
        --dry_run)           DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve project root (four levels up from scripts/polynomial/new_dataset/evaluate_scripts/)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"

# ============================================================================
# Build the sweep command
# ============================================================================
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

EVAL_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
$PYTHON3 geodesic_propagation/sweep_knn_for_polynomial.py \
  --k_values $K_VALUES \
  --ring $RING"

if [[ -n "$SURFACES" ]]; then
    EVAL_CMD="$EVAL_CMD --surfaces $SURFACES"
fi
if [[ -n "$LEVELS" ]]; then
    EVAL_CMD="$EVAL_CMD --levels $LEVELS"
fi
if [[ -n "$LIGHTS" ]]; then
    EVAL_CMD="$EVAL_CMD --lights $LIGHTS"
fi
if [[ -n "$EXTRA" ]]; then
    EVAL_CMD="$EVAL_CMD $EXTRA"
fi

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS_PER_GPU --time=$TIME --pty bash -c '$EVAL_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Sweep kNN k for Polynomial Surfaces (new_dataset)"
echo "============================================================"
echo "  tmux session:     $SESSION_NAME"
echo "  Node:             $NODE"
echo "  GPUs:             $GPUS"
echo "  Time limit:       $TIME"
echo "  k values:         $K_VALUES"
echo "  Ring:             $RING"
echo "  Surfaces:         ${SURFACES:-<all>}"
echo "  Levels:           $LEVELS"
echo "  Lights:           $LIGHTS"
echo "  Conda env:        $CONDA_ENV"
echo "  Extra args:       ${EXTRA:-<none>}"
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
echo "Sweep launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
