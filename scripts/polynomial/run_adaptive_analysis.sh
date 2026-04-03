#!/usr/bin/env bash
# Run adaptive kNN over-cap analysis on Polynomial dataset only.
# Uses sparse matrix ring counts; parallelised across available CPUs.
#
# Usage:
#   bash scripts/polynomial/run_adaptive_analysis.sh [options]
#
# Options:
#   --node NAME        SLURM node (default: gipdeep10)
#   --cpus N           CPUs to request (default: 50)
#   --gpu              Request a GPU for sparse matmul acceleration
#   --time HH:MM:SS    SLURM time limit (default: 06:00:00)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --neighbors N      Target ring-3 neighbors for Polynomial (default: 128)
#   --dry_run          Print commands without executing
set -euo pipefail

NODE="gipdeep12"
CPUS=30
GPU_FLAG=""
TIME="06:00:00"
CONDA_ENV="geo_splat"
POLY_NEIGHBORS=128
N_NEIGHBORS=14
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)       NODE="$2";            shift 2 ;;
        --cpus)       CPUS="$2";            shift 2 ;;
        --gpu)        GPU_FLAG="--gres=gpu:1"; shift ;;
        --time)       TIME="$2";            shift 2 ;;
        --conda_env)  CONDA_ENV="$2";       shift 2 ;;
        --neighbors)  POLY_NEIGHBORS="$2";  shift 2 ;;
        --n_neighbors) N_NEIGHBORS="$2";     shift 2 ;;
        --dry_run)    DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Compute max_steps as the difference between target_neighbors and n_neighbors
MAX_STEPS=$(( POLY_NEIGHBORS - N_NEIGHBORS ))
if [ "$MAX_STEPS" -lt 1 ]; then
    MAX_STEPS=1
fi

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
python DataSets/analyze_geodesic_valid_neighbors.py \
    --mode adaptive \
    --dataset polynomial \
    --adaptive_target_ring 3 \
    --adaptive_target_neighbors $POLY_NEIGHBORS \
    --adaptive_k_boost 20 \
    --adaptive_max_mean_cut 0.00 \
    --adaptive_max_steps 6 \
    --num_sources 4 \
    --num_trials 4 \
    --n_neighbors $N_NEIGHBORS"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS $GPU_FLAG --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Adaptive kNN Analysis (Polynomial only) — srun launcher"
echo "============================================================"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  GPU:             ${GPU_FLAG:-none}"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Dataset:         polynomial"
echo "  Poly neighbors:  $POLY_NEIGHBORS"
echo "  n_neighbors:     $N_NEIGHBORS"
echo "  max_steps:       $MAX_STEPS"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

eval "$SRUN_CMD"
