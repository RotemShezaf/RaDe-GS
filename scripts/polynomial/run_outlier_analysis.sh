#!/usr/bin/env bash
# Run outlier-filtering analysis on Polynomial dataset.
# For each shape, samples points and shows how many ring-k neighbors
# survive the geo-discrepancy outlier filter (same logic as training).
#
# Usage:
#   bash scripts/polynomial/run_outlier_analysis.sh [options]
#
# Options:
#   --node NAME        SLURM node (default: gipdeep12)
#   --cpus N           CPUs to request (default: 30)
#   --gpu              Request a GPU
#   --time HH:MM:SS    SLURM time limit (default: 02:00:00)
#   --conda_env NAME   Conda environment (default: geo_splat)
#   --surfaces LIST    Comma-separated surface names (default: all)
#   --ring N           Ring level (default: 3)
#   --n_neighbors N    Ring-1 kNN k (default: 14)
#   --neighbors N      Target ring-3 neighbors for adaptive (default: 128)
#   --num_sources N    Number of geodesic sources (default: 4)
#   --num_sample_points N  Max points per shape (0=all, default: 0)
#   --outlier_median_multiplier F  (default: 3.0)
#   --outlier_threshold_floor F    (default: 2.0)
#   --outlier_hard_cap F           (default: 500.0)
#   --config PATH      Use a dataset config YAML for outlier params
#   --dry_run          Print commands without executing
set -euo pipefail

NODE="gipdeep10"
CPUS=60
GPU_FLAG=""
TIME="02:00:00"
CONDA_ENV="geo_splat"
SURFACES=""
RING=3
N_NEIGHBORS=20
POLY_NEIGHBORS=128
NUM_SOURCES=4
NUM_SAMPLE_POINTS=0
CONFIG=""
DRY_RUN=false

# Outlier param overrides (empty = use defaults)
OUTLIER_MEDIAN_MULT=""
OUTLIER_FLOOR=""
OUTLIER_HARD_CAP=""
OUTLIER_FALLBACK_MULT=""
OUTLIER_FALLBACK_FLOOR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --node)             NODE="$2";                    shift 2 ;;
        --cpus)             CPUS="$2";                    shift 2 ;;
        --gpu)              GPU_FLAG="--gres=gpu:1";      shift   ;;
        --time)             TIME="$2";                    shift 2 ;;
        --conda_env)        CONDA_ENV="$2";               shift 2 ;;
        --surfaces)         SURFACES="$2";                shift 2 ;;
        --ring)             RING="$2";                    shift 2 ;;
        --n_neighbors)      N_NEIGHBORS="$2";             shift 2 ;;
        --neighbors)        POLY_NEIGHBORS="$2";          shift 2 ;;
        --num_sources)      NUM_SOURCES="$2";             shift 2 ;;
        --num_sample_points) NUM_SAMPLE_POINTS="$2";      shift 2 ;;
        --config)           CONFIG="$2";                  shift 2 ;;
        --outlier_median_multiplier)  OUTLIER_MEDIAN_MULT="--outlier_median_multiplier $2"; shift 2 ;;
        --outlier_threshold_floor)    OUTLIER_FLOOR="--outlier_threshold_floor $2";         shift 2 ;;
        --outlier_hard_cap)           OUTLIER_HARD_CAP="--outlier_hard_cap $2";             shift 2 ;;
        --outlier_fallback_multiplier) OUTLIER_FALLBACK_MULT="--outlier_fallback_multiplier $2"; shift 2 ;;
        --outlier_fallback_floor)     OUTLIER_FALLBACK_FLOOR="--outlier_fallback_floor $2";     shift 2 ;;
        --dry_run)          DRY_RUN=true;                 shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SURFACES_ARG=""
if [[ -n "$SURFACES" ]]; then
    SURFACES_ARG="--surfaces $SURFACES"
fi

CONFIG_ARG=""
if [[ -n "$CONFIG" ]]; then
    CONFIG_ARG="--config $CONFIG"
fi

INNER_CMD="cd $SCRIPT_DIR && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
python DataSets/analyze_outlier_filtering.py \
    --dataset polynomial \
    --ring $RING \
    --n_neighbors $N_NEIGHBORS \
    --num_sources $NUM_SOURCES \
    --num_sample_points $NUM_SAMPLE_POINTS \
    --adaptive_target_ring $RING \
    --adaptive_target_neighbors $POLY_NEIGHBORS \
    --adaptive_k_boost 22 \
    --adaptive_max_mean_cut 0.00 \
    --adaptive_max_steps 6 \
    --workers $CPUS \
    $SURFACES_ARG \
    $CONFIG_ARG \
    $OUTLIER_MEDIAN_MULT \
    $OUTLIER_FLOOR \
    $OUTLIER_HARD_CAP \
    $OUTLIER_FALLBACK_MULT \
    $OUTLIER_FALLBACK_FLOOR"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS $GPU_FLAG --time=$TIME --pty bash -c '$INNER_CMD'"

echo "============================================================"
echo "Outlier Filtering Analysis (Polynomial) — srun launcher"
echo "============================================================"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  GPU:             ${GPU_FLAG:-none}"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Dataset:         polynomial"
echo "  Surfaces:        ${SURFACES:-all}"
echo "  Ring:            $RING"
echo "  kNN k:           $N_NEIGHBORS"
echo "  Poly neighbors:  $POLY_NEIGHBORS"
echo "  Num sources:     $NUM_SOURCES"
echo "  Sample points:   ${NUM_SAMPLE_POINTS:-all}"
echo "  Config:          ${CONFIG:-none}"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

eval "$SRUN_CMD"
