#!/usr/bin/env bash
# Run adaptive kNN over-cap analysis on TOSCA + Polynomial datasets.
# Uses sparse matrix ring counts; parallelised across available CPUs.
# Runs via srun on a SLURM compute node (see scripts/tosca/generate_patches_tmux.sh).
#
# Usage:
#   bash scripts/run_adaptive_analysis.sh [options]
#
# Options:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           CPUs to request (default: 50)
#   --gpu              Request a GPU for sparse matmul acceleration
#   --time HH:MM:SS    SLURM time limit (default: 06:00:00)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --dataset NAME     tosca|polynomial|both (default: both)
#   --tosca_neighbors N  Target ring-3 neighbors for TOSCA (default: 192)
#   --poly_neighbors N   Target ring-3 neighbors for Polynomial (default: 128)
#   --dry_run          Print commands without executing
set -euo pipefail

NODE="gipdeep10"
CPUS=50
GPU_FLAG=""
GPU_ARG=""
TIME="06:00:00"
CONDA_ENV="geo_splat"
DATASET="both"
TOSCA_NEIGHBORS=192
POLY_NEIGHBORS=128
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)             NODE="$2";            shift 2 ;;
        --cpus)             CPUS="$2";            shift 2 ;;
        --gpu)              GPU_FLAG="--gres=gpu:1"; GPU_ARG=""; shift ;;
        --time)             TIME="$2";            shift 2 ;;
        --conda_env)        CONDA_ENV="$2";       shift 2 ;;
        --dataset)          DATASET="$2";         shift 2 ;;
        --tosca_neighbors)  TOSCA_NEIGHBORS="$2"; shift 2 ;;
        --poly_neighbors)   POLY_NEIGHBORS="$2";  shift 2 ;;
        --dry_run)          DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INNER_CMD="cd $PROJECT_ROOT && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
python DataSets/analyze_ring_statistics.py \
    --mode adaptive \
    --dataset $DATASET \
    --adaptive_target_ring 3 \
    --adaptive_target_neighbors $POLY_NEIGHBORS \
    --tosca_adaptive_target_neighbors $TOSCA_NEIGHBORS \
    --adaptive_k_boost 18\
    --adaptive_max_mean_cut 0.00 \
    --adaptive_max_steps 12 \
    --n_neighbors 6"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS $GPU_FLAG --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Adaptive kNN Analysis — srun launcher"
echo "============================================================"
echo "  Node:                $NODE"
echo "  CPUs:                $CPUS"
echo "  GPU:                 ${GPU_FLAG:-none}"
echo "  Time limit:          $TIME"
echo "  Conda env:           $CONDA_ENV"
echo "  Dataset:             $DATASET"
echo "  TOSCA neighbors:     $TOSCA_NEIGHBORS"
echo "  Poly neighbors:      $POLY_NEIGHBORS"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

eval "$SRUN_CMD"
