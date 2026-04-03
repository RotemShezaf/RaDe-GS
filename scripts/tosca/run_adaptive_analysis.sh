#!/usr/bin/env bash
# Run geodesic valid-neighbor analysis on TOSCA (and optionally Polynomial) datasets.
# For each shape and source, counts how many ring-k neighbors have geodesic
# distance ≤ the center point (i.e. are "valid" during FM propagation).
#
# Usage:
#   bash scripts/tosca/run_adaptive_analysis.sh [options]
#
# Options:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           CPUs to request (default: 50)
#   --gpu              Request a GPU for acceleration
#   --time HH:MM:SS    SLURM time limit (default: 06:00:00)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --dataset NAME     Dataset: tosca|polynomial|both (default: tosca)
#   --shapes LIST      Comma-separated shape names (default: all)
#   --ring N           Ring level (default: 3)
#   --n_neighbors N    Ring-1 kNN k (default: 10)
#   --num_sources N    Multi-source source count (default: 1)
#   --num_trials N     Random source trials (default: 5)
#   --use_mahalanobis  Use Mahalanobis kNN
#   --dry_run          Print commands without executing
set -euo pipefail

NODE="gipdeep12"
CPUS=30
GPU_FLAG=""
TIME="06:00:00"
CONDA_ENV="geo_splat"
DATASET="tosca"
SHAPES=""
RING=3
N_NEIGHBORS=10
NUM_SOURCES=4
NUM_TRIALS=50
USE_MAHALANOBIS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)            NODE="$2";              shift 2 ;;
        --cpus)            CPUS="$2";              shift 2 ;;
        --gpu)             GPU_FLAG="--gres=gpu:1"; shift ;;
        --time)            TIME="$2";              shift 2 ;;
        --conda_env)       CONDA_ENV="$2";         shift 2 ;;
        --dataset)         DATASET="$2";           shift 2 ;;
        --shapes)          SHAPES="$2";            shift 2 ;;
        --ring)            RING="$2";              shift 2 ;;
        --n_neighbors)     N_NEIGHBORS="$2";       shift 2 ;;
        --num_sources)     NUM_SOURCES="$2";       shift 2 ;;
        --num_trials)      NUM_TRIALS="$2";        shift 2 ;;
        --use_mahalanobis) USE_MAHALANOBIS="--use_mahalanobis"; shift ;;
        --dry_run)         DRY_RUN=true;           shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SHAPES_ARG=""
if [[ -n "$SHAPES" ]]; then
    SHAPES_ARG="--shapes $SHAPES"
fi

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
python DataSets/analyze_geodesic_valid_neighbors.py \
    --mode adaptive \
    --ring $RING \
    --dataset $DATASET \
    --n_neighbors $N_NEIGHBORS \
    --num_sources $NUM_SOURCES \
    --num_trials $NUM_TRIALS \
    --adaptive_target_ring $RING \
    --adaptive_target_neighbors 16\
    --adaptive_k_boost 6\
    --adaptive_max_mean_cut 0.00 \
    --adaptive_max_steps 10 \
    --workers $CPUS \
    $USE_MAHALANOBIS \
    $SHAPES_ARG"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS $GPU_FLAG --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Geodesic Valid-Neighbor Analysis — srun launcher"
echo "============================================================"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  GPU:             ${GPU_FLAG:-none}"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Dataset:         $DATASET"
echo "  Shapes:          ${SHAPES:-all}"
echo "  Ring:            $RING"
echo "  kNN k:           $N_NEIGHBORS"
echo "  Num sources:     $NUM_SOURCES"
echo "  Num trials:      $NUM_TRIALS"
echo "  Mahalanobis:     ${USE_MAHALANOBIS:-no}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

eval "$SRUN_CMD"
