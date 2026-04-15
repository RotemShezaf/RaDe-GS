#!/bin/bash
#
# Generate training patches with _aug_normals and then train.
#
# Two-phase pipeline:
#   Phase 1: Generate training patches for all 3 polynomial surfaces
#            using per-surface configs with _aug_normals attribute.
#   Phase 2: Train GaussianPatchTransformer using the aug_normals
#            training config.
#
# USAGE:
#   bash scripts/polynomial/new_dataset/train/generate_and_train_aug_normals.sh [options]
#
# OPTIONS:
#   --node NAME        SLURM node (default: gipdeep10)
#   --gpus N           GPUs for training (default: 1)
#   --cpus N           CPUs for generation (default: 60)
#   --time HH:MM:SS    SLURM time limit (default: 24:00:00)
#   --session NAME     tmux session name (default: aug_normals)
#   --skip_generate    Skip data generation, train only
#   --generate_only    Generate data only, don't train
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/polynomial/new_dataset/train/generate_and_train_aug_normals.sh
#   bash scripts/polynomial/new_dataset/train/generate_and_train_aug_normals.sh --skip_generate
#   bash scripts/polynomial/new_dataset/train/generate_and_train_aug_normals.sh --node gipdeep12 --gpus 1

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep10"
GPUS=1
CPUS=60
TIME="24:00:00"
SESSION_NAME="aug_normals"
CONDA_ENV="geo_splat"
SKIP_GENERATE=false
GENERATE_ONLY=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)           NODE="$2";          shift 2 ;;
        --gpus)           GPUS="$2";          shift 2 ;;
        --cpus)           CPUS="$2";          shift 2 ;;
        --time)           TIME="$2";          shift 2 ;;
        --session)        SESSION_NAME="$2";  shift 2 ;;
        --skip_generate)  SKIP_GENERATE=true; shift   ;;
        --generate_only)  GENERATE_ONLY=true; shift   ;;
        --dry_run)        DRY_RUN=true;       shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

CONFIG_DIR="DataSets/configs/polynomial/new_dataset"
GENERATE_SCRIPT="DataSets/create_gaussian_training_patches.py"
TRAIN_SCRIPT="models/train_gaussian_patch_transformer.py"
TRAIN_CONFIG="models/configs/new_dataset/combined_polynomial_ring3_noisy_geo_aug_normals.yaml"

# Per-surface dataset configs (with _aug_normals)
PARABOLOID_CONFIG="$CONFIG_DIR/paraboloid_all_one_source_aug_normals.yaml"
SADDLE_CONFIG="$CONFIG_DIR/saddle_all_one_source_aug_normals.yaml"
HYPPAR_CONFIG="$CONFIG_DIR/hyperbolic_paraboloid_all_one_source_aug_normals.yaml"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")

# ============================================================================
# Build the inner command to run inside srun
# ============================================================================
INNER_CMD="cd $SCRIPT_DIR && \\
source \$(conda info --base)/etc/profile.d/conda.sh && \\
conda activate $CONDA_ENV && \\
export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH"

# Phase 1: Generate patches (sequential to avoid memory pressure)
if [ "$SKIP_GENERATE" = false ]; then
    INNER_CMD="$INNER_CMD && \\
echo \"============================================================\" && \\
echo \"Phase 1: Generating training patches with _aug_normals\" && \\
echo \"============================================================\" && \\
echo && \\
echo \"--- Paraboloid ---\" && \\
python $GENERATE_SCRIPT --config $PARABOLOID_CONFIG && \\
echo && \\
echo \"--- Saddle ---\" && \\
python $GENERATE_SCRIPT --config $SADDLE_CONFIG && \\
echo && \\
echo \"--- HyperbolicParaboloid ---\" && \\
python $GENERATE_SCRIPT --config $HYPPAR_CONFIG && \\
echo && \\
echo \"Phase 1 complete: all patches generated.\""
fi

# Phase 2: Train
if [ "$GENERATE_ONLY" = false ]; then
    SRUN_TRAIN="srun --nodelist=$NODE --gres=gpu:$GPUS --cpus-per-task=6 --time=$TIME --pty bash -c 'cd $SCRIPT_DIR && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python $TRAIN_SCRIPT --train_config $TRAIN_CONFIG --ring 3'"

    if [ "$SKIP_GENERATE" = true ]; then
        # Train only — just use srun directly with GPU
        FULL_CMD="$SRUN_TRAIN"
    else
        # After generation (CPU srun), launch training (GPU srun)
        INNER_CMD="$INNER_CMD && \\
echo && \\
echo \"============================================================\" && \\
echo \"Phase 2: Launching training\" && \\
echo \"============================================================\" && \\
echo \"Training config: $TRAIN_CONFIG\""
        # Generation runs on CPU srun, training needs separate GPU srun
        FULL_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD' && $SRUN_TRAIN"
    fi
else
    FULL_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"
fi

# If skip_generate and not generate_only, FULL_CMD is just the training srun
if [ "$SKIP_GENERATE" = true ] && [ "$GENERATE_ONLY" = false ]; then
    FULL_CMD="$SRUN_TRAIN"
fi

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Generate & Train – aug_normals directed perturbation"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs (gen):      $CPUS"
echo "  GPUs (train):    $GPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Skip generate:   $SKIP_GENERATE"
echo "  Generate only:   $GENERATE_ONLY"
echo ""
echo "  Dataset configs:"
echo "    Paraboloid:     $PARABOLOID_CONFIG"
echo "    Saddle:         $SADDLE_CONFIG"
echo "    HypPar:         $HYPPAR_CONFIG"
echo "  Training config:  $TRAIN_CONFIG"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Commands that would be executed:"
    echo "$FULL_CMD"
    exit 0
fi

# ============================================================================
# Launch in tmux
# ============================================================================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists – sending command..."
    tmux send-keys -t "$SESSION_NAME" "$FULL_CMD" Enter
else
    echo "Creating tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    sleep 1
    tmux send-keys -t "$SESSION_NAME" "$FULL_CMD" Enter
fi

echo ""
echo "Launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
