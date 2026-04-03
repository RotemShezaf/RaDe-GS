#!/bin/bash
#
# Run the FULL TOSCA pipeline end-to-end inside a tmux session:
#   1. Render synthetic COLMAP datasets (CPU)
#   2. Train Gaussian Splatting + extract mesh (GPU)
#   3. Compute geodesic distances (CPU)
#   4. Generate training patches (CPU)
#
# Each stage runs sequentially. Use the individual tmux scripts
# (render_tmux.sh, train_tmux.sh, etc.) if you want to run stages
# independently or on different nodes.
#
# USAGE:
#   bash scripts/tosca/pipeline/full_pipeline_tmux.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep7)
#   --gpus N           GPUs for training stage (default: 1)
#   --cpus N           CPUs to request (default: 20)
#   --time HH:MM:SS    SLURM time limit (default: 72:00:00)
#   --session NAME     tmux session name (default: tosca_pipeline)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --shapes LIST      Comma-separated shapes (default: all)
#   --skip_render      Skip stage 1 (rendering)
#   --skip_train       Skip stage 2 (training + mesh)
#   --skip_geodesic    Skip stage 3 (geodesic)
#   --skip_patches     Skip stage 4 (patches)
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   # Full pipeline for all shapes
#   bash scripts/tosca/pipeline/full_pipeline_tmux.sh
#
#   # Only geodesic + patches (already rendered & trained)
#   bash scripts/tosca/pipeline/full_pipeline_tmux.sh --skip_render --skip_train
#
#   # Specific shapes
#   bash scripts/tosca/pipeline/full_pipeline_tmux.sh --shapes "cat0,cat1,dog0"

set -e

NODE="gipdeep7"
GPUS=1
CPUS=20
TIME="72:00:00"
SESSION_NAME="tosca_pipeline"
CONDA_ENV="geo_splat"
SHAPES=""
SKIP_RENDER=false
SKIP_TRAIN=false
SKIP_GEODESIC=false
SKIP_PATCHES=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)          NODE="$2";          shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --cpus)          CPUS="$2";          shift 2 ;;
        --time)          TIME="$2";          shift 2 ;;
        --session)       SESSION_NAME="$2";  shift 2 ;;
        --conda_env)     CONDA_ENV="$2";     shift 2 ;;
        --shapes)        SHAPES="$2";        shift 2 ;;
        --skip_render)   SKIP_RENDER=true;   shift   ;;
        --skip_train)    SKIP_TRAIN=true;    shift   ;;
        --skip_geodesic) SKIP_GEODESIC=true; shift   ;;
        --skip_patches)  SKIP_PATCHES=true;  shift   ;;
        --dry_run)       DRY_RUN=true;       shift   ;;
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

# Build inner pipeline command
PIPELINE_CMDS=""

if [ "$SKIP_RENDER" = false ]; then
    PIPELINE_CMDS+="echo '========== Stage 1/4: Rendering ==========' && "
    PIPELINE_CMDS+="bash scripts/tosca/render/render_all_blue.sh $SHAPE_ARG && "
fi

if [ "$SKIP_TRAIN" = false ]; then
    PIPELINE_CMDS+="echo '========== Stage 2/4: Training + Mesh ==========' && "
    PIPELINE_CMDS+="bash scripts/tosca/train_gaussians/train_all_blue.sh $SHAPE_ARG --skip_existing && "
fi

if [ "$SKIP_GEODESIC" = false ]; then
    PIPELINE_CMDS+="echo '========== Stage 3/4: Geodesic Distances ==========' && "
    PIPELINE_CMDS+="bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh $SHAPE_ARG && "
fi

if [ "$SKIP_PATCHES" = false ]; then
    PIPELINE_CMDS+="echo '========== Stage 4/4: Training Patches ==========' && "
    PIPELINE_CMDS+="bash scripts/tosca/patches/generate_training_patches.sh && "
fi

PIPELINE_CMDS+="echo '========== TOSCA Pipeline Complete =========='"

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
$PIPELINE_CMDS"

SRUN_CMD="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "TOSCA Full Pipeline – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  GPUs:            $GPUS"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Shapes:          ${SHAPES:-all}"
echo ""
echo "  Stages:"
[ "$SKIP_RENDER" = false ]   && echo "    1. Render (blue_texture)"      || echo "    1. Render [SKIPPED]"
[ "$SKIP_TRAIN" = false ]    && echo "    2. Train + Mesh Extract"       || echo "    2. Train [SKIPPED]"
[ "$SKIP_GEODESIC" = false ] && echo "    3. Geodesic Distances (MMP)"   || echo "    3. Geodesic [SKIPPED]"
[ "$SKIP_PATCHES" = false ]  && echo "    4. Training Patches"           || echo "    4. Patches [SKIPPED]"
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
