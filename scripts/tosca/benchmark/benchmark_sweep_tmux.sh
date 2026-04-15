#!/bin/bash
#
# Launch TOSCA benchmark sweep inside a tmux session on a SLURM GPU node.
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_sweep_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --gpus N           GPUs to request (default: 2)
#   --cpus N           CPUs to request (default: 6)
#   --cpus_eval N      CPUs for evaluation (no GPU, default: 8)
#   --time HH:MM:SS    SLURM time limit (default: 48:00:00)
#   --session NAME     tmux session name (default: tosca_sweep)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --shapes LIST      Comma-separated shapes (default: all)
#   --animals LIST     Animal groups (default: all)
#   --max_parallel N   Max parallel training (default: 2)
#   --max_parallel_cpu N Max parallel CPU eval jobs (default: 4)
#   --skip_existing    Skip shapes with existing benchmark_report.json
#   --evaluate_only    Only run CPU evaluation (no training/mesh), no GPU needed
#   --no_wandb         Disable wandb logging
#   --extra "ARGS"     Extra args forwarded to benchmark_sweep.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/benchmark/benchmark_sweep_tmux.sh
#   bash scripts/tosca/benchmark/benchmark_sweep_tmux.sh --node gipdeep7 --shapes cat0,dog0
#   bash scripts/tosca/benchmark/benchmark_sweep_tmux.sh --skip_existing --no_wandb

set -e

NODE="gipdeep8"
GPUS=3
CPUS=38
CPUS_EVAL=31
TIME="48:00:00"
SESSION_NAME="tosca_sweep"
CONDA_ENV="geo_splat"
SHAPES=""
ANIMALS=""
MAX_PARALLEL=3


MAX_PARALLEL_CPU=37
SKIP_EXISTING="--skip_existing" #"--skip_existing"
EVALUATE_ONLY="" #--evaluate_only"
NO_WANDB="--no_wandb"
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)          NODE="$2";                     shift 2 ;;
        --gpus)          GPUS="$2";                     shift 2 ;;
        --cpus)          CPUS="$2";                     shift 2 ;;
        --cpus_eval)     CPUS_EVAL="$2";                 shift 2 ;;
        --time)          TIME="$2";                     shift 2 ;;
        --session)       SESSION_NAME="$2";             shift 2 ;;
        --conda_env)     CONDA_ENV="$2";                shift 2 ;;
        --shapes)        SHAPES="$2";                   shift 2 ;;
        --animals)       ANIMALS="$2";                  shift 2 ;;
        --max_parallel)  MAX_PARALLEL="$2";             shift 2 ;;
        --max_parallel_cpu) MAX_PARALLEL_CPU="$2";       shift 2 ;;
        --skip_existing) SKIP_EXISTING="--skip_existing"; shift ;;
        --evaluate_only) EVALUATE_ONLY="--evaluate_only"; shift ;;
        --no_wandb)      NO_WANDB="--no_wandb";        shift   ;;
        --extra)         EXTRA="$2";                    shift 2 ;;
        --dry_run)       DRY_RUN=true;                  shift   ;;
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
ANIMAL_ARG=""
[ -n "$ANIMALS" ] && ANIMAL_ARG="--animals $ANIMALS"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/benchmark/benchmark_sweep_r11.sh \
    $SHAPE_ARG $ANIMAL_ARG \
    --max_parallel $MAX_PARALLEL \
    --max_parallel_cpu $MAX_PARALLEL_CPU \
    $SKIP_EXISTING $EVALUATE_ONLY $NO_WANDB $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Benchmark Sweep – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  GPUs:            $GPUS"
echo "  CPUs (train):    $CPUS"
echo "  CPUs (eval):     $CPUS_EVAL"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Shapes:          ${SHAPES:-all}"
echo "  Animals:         ${ANIMALS:-all}"
echo "  Max parallel:    $MAX_PARALLEL"
echo "  Skip existing:   ${SKIP_EXISTING:-no}"
echo "  Evaluate only:   ${EVALUATE_ONLY:-no}"
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
    sleep 0.5
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
fi

echo ""
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo "Detach with:  Ctrl+b d"
