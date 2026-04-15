#!/bin/bash
#
# Master script to run all evaluation scripts sequentially.
# (one_source variant – uses combined_polynomial_ring3 config & checkpoint)
#
# This script runs the evaluation steps in separate tmux sessions:
#  1. evaluate_geodesic.sh     – Geodesic distance propagation
#  2. evaluate_model_vs_gt.sh  – Model accuracy vs ground truth
#  3. analyze_nn_stats.sh      – Nearest-neighbor statistics
#  4. run_eval_fps.sh          – FPS-downsampled evaluation
#  5. run_eval_propagation.sh  – Debug FM propagation (first N visited)
#  6. sweep_knn.sh             – kNN k-value sweep
#
# USAGE:
#   bash scripts/polynomial/evaluate_scripts/run_all_evaluations.sh [options]
#
# OPTIONS:
#   --node NAME           SLURM node (default: gipdeep7)
#   --gpus N              Number of GPUs (default: 1)
#   --time HH:MM:SS       SLURM time limit (default: 4:00:00)
#   --config PATH         Testing config (default: models/configs/testing.yaml)
#   --gaussian_dir PATH   Path to Gaussian output dir (optional)
#   --model_path PATH     Model checkpoint path (optional)
#   --train_config PATH   Training config YAML (optional)
#   --dataset_config PATH Dataset config YAML (optional)
#   --sequential          Run scripts one after another (default: all in parallel tmux sessions)
#   --dry_run             Print commands without executing

set -e

# ============================================================================
# Defaults
# ============================================================================
NODE="gipdeep7"
GPUS=1
TIME="4:00:00"
CONFIG="/home/rotem.shezaf/RaDe-GS/models/configs/testing.yaml"
GAUSSIAN_DIR=""
MODEL_PATH=""
TRAIN_CONFIG=""
DATASET_CONFIG=""
SEQUENTIAL=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)              NODE="$2";            shift 2 ;;
        --gpus)              GPUS="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --config)            CONFIG="$2";          shift 2 ;;
        --gaussian_dir)      GAUSSIAN_DIR="$2";    shift 2 ;;
        --model_path)        MODEL_PATH="$2";      shift 2 ;;
        --train_config)      TRAIN_CONFIG="$2";    shift 2 ;;
        --dataset_config)    DATASET_CONFIG="$2";  shift 2 ;;
        --sequential)        SEQUENTIAL=true;      shift   ;;
        --dry_run)           DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Master Evaluation Pipeline – All Geodesic Tests (one_source)"
echo "============================================================"
echo "  Node:            $NODE"
echo "  GPUs:            $GPUS"
echo "  Time limit:      $TIME"
echo "  Config:          $CONFIG"
echo "  Gaussian dir:    ${GAUSSIAN_DIR:-<default>}"
echo "  Model path:      ${MODEL_PATH:-<default>}"
echo "  Train config:    ${TRAIN_CONFIG:-<default>}"
echo "  Dataset config:  ${DATASET_CONFIG:-<default>}"
echo "  Sequential:      $SEQUENTIAL"
echo ""
echo "  Scripts to run:"
echo "    1. evaluate_geodesic.sh"
echo "    2. evaluate_model_vs_gt.sh"
echo "    3. analyze_nn_stats.sh"
echo "    4. run_eval_fps.sh"
echo "    5. run_eval_propagation.sh"
echo "    6. sweep_knn.sh"
echo ""

DRY_FLAG=""
if [ "$DRY_RUN" = true ]; then
    DRY_FLAG="--dry_run"
    echo "[DRY RUN] – printing commands only."
    echo ""
fi

# ============================================================================
# Launch all evaluations
# ============================================================================
echo "Launching evaluations..."
echo ""

# Build common arguments
COMMON_ARGS="--node $NODE --time $TIME --config $CONFIG"
if [[ -n "$MODEL_PATH" ]]; then COMMON_ARGS="$COMMON_ARGS --model_path $MODEL_PATH"; fi
if [[ -n "$TRAIN_CONFIG" ]]; then COMMON_ARGS="$COMMON_ARGS --train_config $TRAIN_CONFIG"; fi
if [[ -n "$DATASET_CONFIG" ]]; then COMMON_ARGS="$COMMON_ARGS --dataset_config $DATASET_CONFIG"; fi

# 1. Evaluate geodesic
echo "[1/6] Launching evaluate_geodesic.sh..."
bash "$SCRIPT_DIR/evaluate_geodesic.sh" --gpus $GPUS $COMMON_ARGS $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

# 2. Evaluate model vs GT
echo "[2/6] Launching evaluate_model_vs_gt.sh..."
bash "$SCRIPT_DIR/evaluate_model_vs_gt.sh" --gpus $GPUS $COMMON_ARGS $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

# 3. Analyze NN stats
echo "[3/6] Launching analyze_nn_stats.sh..."
STATS_ARGS="--node $NODE --time $TIME --config $CONFIG"
if [[ -n "$GAUSSIAN_DIR" ]]; then STATS_ARGS="$STATS_ARGS --gaussian_dir $GAUSSIAN_DIR"; fi
bash "$SCRIPT_DIR/analyze_nn_stats.sh" $STATS_ARGS $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

# 4. Run eval FPS
echo "[4/6] Launching run_eval_fps.sh..."
bash "$SCRIPT_DIR/run_eval_fps.sh" --gpus $GPUS $COMMON_ARGS $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

# 5. Run eval propagation (does not accept --config)
echo "[5/6] Launching run_eval_propagation.sh..."
PROP_ARGS="--node $NODE --time $TIME --gpus $GPUS"
if [[ -n "$MODEL_PATH" ]]; then PROP_ARGS="$PROP_ARGS --model_path $MODEL_PATH"; fi
if [[ -n "$TRAIN_CONFIG" ]]; then PROP_ARGS="$PROP_ARGS --train_config $TRAIN_CONFIG"; fi
if [[ -n "$DATASET_CONFIG" ]]; then PROP_ARGS="$PROP_ARGS --dataset_config $DATASET_CONFIG"; fi
bash "$SCRIPT_DIR/run_eval_propagation.sh" $PROP_ARGS $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

# 6. Sweep kNN (does not accept --config)
echo "[6/6] Launching sweep_knn.sh..."
bash "$SCRIPT_DIR/sweep_knn.sh" --node $NODE --time $TIME $DRY_FLAG
if [ "$SEQUENTIAL" = true ]; then sleep 5; fi

echo ""
echo "============================================================"
echo "All evaluations launched!"
echo ""
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null | grep -E "eval_|sweep_" || echo "  (none yet, check again in a moment)"
echo ""
echo "Attach to a session with:"
echo "  tmux attach -t <SESSION_NAME>"
echo ""
