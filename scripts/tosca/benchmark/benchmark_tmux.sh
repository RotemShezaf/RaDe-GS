#!/bin/bash
#
# Launch TOSCA parameter benchmark inside a tmux session on a SLURM GPU node.
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --gpus N           GPUs to request (default: 2)
#   --cpus N           CPUs to request (default: 6)
#   --cpus_eval N      CPUs for evaluation (no GPU, default: 8)
#   --time HH:MM:SS    SLURM time limit (default: 48:00:00)
#   --session NAME     tmux session name (default: tosca_benchmark)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --shape SHAPE      TOSCA shape to benchmark (default: cat0)
#   --max_parallel N   Max parallel training (default: 2)
#   --max_parallel_cpu N Max parallel CPU eval jobs (default: 4)
#   --no_wandb         Disable wandb logging
#   --skip_existing    Skip already completed runs
#   --extra "ARGS"     Extra args forwarded to benchmark_params.sh
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/benchmark/benchmark_tmux.sh
#   bash scripts/tosca/benchmark/benchmark_tmux.sh --node gipdeep7 --shape dog0
#   bash scripts/tosca/benchmark/benchmark_tmux.sh --skip_existing --no_wandb

set -e

NODE="gipdeep10"
GPUS=2
CPUS=32
CPUS_EVAL=32
TIME="48:00:00"
SESSION_NAME="tosca_benchmark"
CONDA_ENV="geo_splat"
SHAPE="cat0,cat2,centaur0,centaur1,centaur5,david0,dog0,gorilla5,gorilla8,horse0,horse10,michael0,michael2,michael16,victoria0,victoria2,wolf0"
MAX_PARALLEL=2
MAX_PARALLEL_CPU=31
EXTRA=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)          NODE="$2";              shift 2 ;;
        --gpus)          GPUS="$2";              shift 2 ;;
        --cpus)          CPUS="$2";              shift 2 ;;
        --cpus_eval)     CPUS_EVAL="$2";          shift 2 ;;
        --time)          TIME="$2";              shift 2 ;;
        --session)       SESSION_NAME="$2";      shift 2 ;;
        --conda_env)     CONDA_ENV="$2";         shift 2 ;;
        --shape)         SHAPE="$2";             shift 2 ;;
        --max_parallel)  MAX_PARALLEL="$2";      shift 2 ;;
        --max_parallel_cpu) MAX_PARALLEL_CPU="$2"; shift 2 ;;
        --no_wandb)      EXTRA="$EXTRA --no_wandb"; shift ;;
        --skip_existing) EXTRA="$EXTRA --skip_existing"; shift ;;
        --extra)         EXTRA="$EXTRA $2";      shift 2 ;;
        --dry_run)       DRY_RUN=true;           shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash scripts/tosca/benchmark/benchmark_round5.sh --shape $SHAPE --max_parallel $MAX_PARALLEL --max_parallel_cpu $MAX_PARALLEL_CPU $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Parameter Benchmark – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  GPUs:            $GPUS"
echo "  CPUs (train):    $CPUS"
echo "  CPUs (eval):     $CPUS_EVAL"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Shape:           $SHAPE"
echo "  Max parallel:    $MAX_PARALLEL"
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
    echo "tmux session '$SESSION_NAME' already exists – sending command..."
else
    echo "Creating tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    sleep 0.5
fi

tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter

echo ""
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo "Detach with:  Ctrl+b d"
