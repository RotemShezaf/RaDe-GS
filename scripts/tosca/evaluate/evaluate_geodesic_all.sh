#!/bin/bash
# ============================================================================
# Evaluate geodesic distance propagation on all TOSCA test shapes
#
# Test set = last pose per animal (12 shapes total):
#   cat10, centaur5, david14, dog10, gorilla20, horse18,
#   lioness16, michael19, seahorse5, shark0, victoria25, wolf2
#
# Note: shark has only 1 pose (shark0), so it appears in test set only.
#
# Usage:
#   bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --model_path checkpoints/my_model/best_model.pth
#   bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --model_path <path> --shapes "cat10,dog10,horse18"
#   bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --model_path <path> --dry_run
#   bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --help
# ============================================================================

set -e

# --- Test set: last pose per animal ---
# Format: "animal:pose"
declare -A TEST_SHAPES
TEST_SHAPES=(
    [cat]="cat10"
    [centaur]="centaur5"
    [david]="david14"
    [dog]="dog10"
    [gorilla]="gorilla20"
    [horse]="horse18"
    [lioness]="lioness16"
    [michael]="michael19"
    [seahorse]="seahorse5"
    [shark]="shark0"
    [victoria]="victoria25"
    [wolf]="wolf2"
)
ALL_TEST_SHAPES="cat10 centaur5 david14 dog10 gorilla20 horse18 lioness16 michael19 seahorse5 shark0 victoria25 wolf2"

# --- Defaults ---
MODEL_PATH=""
SHAPES=""  # empty = all test shapes
TEXTURE="blue_texture"
RESOLUTION="high_res"
APPEARANCE="decoupled_appearance"
NUM_SOURCES=5
SEED=42
N_NEIGHBORS=16
RING=2
BATCH_SIZE=32
DEVICE="cuda"
USE_MAHALANOBIS=""
NO_PLY=""
OUTPUT_BASE="geodesic_eval_results/tosca"
ITERATION=""
EXTRA_ARGS=""
DRY_RUN=false
SKIP_EXISTING=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2"; shift 2 ;;
        --shapes)           SHAPES="$2"; shift 2 ;;
        --texture)          TEXTURE="$2"; shift 2 ;;
        --resolution)       RESOLUTION="$2"; shift 2 ;;
        --appearance)       APPEARANCE="$2"; shift 2 ;;
        --num_sources)      NUM_SOURCES="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --n_neighbors)      N_NEIGHBORS="$2"; shift 2 ;;
        --ring)             RING="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --no_mahalanobis)   USE_MAHALANOBIS="--no_mahalanobis"; shift ;;
        --no_ply)           NO_PLY="--no_ply"; shift ;;
        --output_base)      OUTPUT_BASE="$2"; shift 2 ;;
        --iteration)        ITERATION="$2"; shift 2 ;;
        --extra)            EXTRA_ARGS="$2"; shift 2 ;;
        --dry_run)          DRY_RUN=true; shift ;;
        --skip_existing)    SKIP_EXISTING=true; shift ;;
        --help|-h)
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Evaluates geodesic propagation on all TOSCA test shapes."
            echo "Test set = last pose per animal: ${ALL_TEST_SHAPES}"
            echo ""
            echo "Required:"
            echo "  --model_path PATH    Path to trained geodesic model checkpoint"
            echo ""
            echo "Options:"
            echo "  --shapes LIST        Comma-separated shapes to evaluate (default: all test shapes)"
            echo "  --texture NAME       Texture type (default: blue_texture)"
            echo "  --resolution NAME    Resolution (default: high_res)"
            echo "  --appearance NAME    Appearance mode (default: decoupled_appearance)"
            echo "  --num_sources N      Number of random source points per shape (default: 5)"
            echo "  --seed N             Random seed for source selection (default: 42)"
            echo "  --n_neighbors N      Number of ring-1 neighbors (default: 16)"
            echo "  --ring N             Ring level [1-4] (default: 2)"
            echo "  --batch_size N       Batch size for predictions (default: 32)"
            echo "  --device DEVICE      cuda or cpu (default: cuda)"
            echo "  --no_mahalanobis     Use Euclidean instead of Mahalanobis distance"
            echo "  --no_ply             Skip PLY visualization export"
            echo "  --output_base PATH   Base output directory (default: geodesic_eval_results/tosca)"
            echo "  --iteration N        Gaussian iteration (default: highest)"
            echo "  --skip_existing      Skip shapes that already have results"
            echo "  --extra \"ARGS\"       Extra args forwarded to evaluate_geodesic.py"
            echo "  --dry_run            Print commands without executing"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Validate ---
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model_path is required"
    echo "Usage: $0 --model_path <path>"
    exit 1
fi

# --- Resolve project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# --- Determine shapes to evaluate ---
if [[ -n "$SHAPES" ]]; then
    # User specified shapes (comma-separated)
    SHAPE_LIST=$(echo "$SHAPES" | tr ',' ' ')
else
    SHAPE_LIST="$ALL_TEST_SHAPES"
fi

# --- Summary ---
TOTAL=$(echo $SHAPE_LIST | wc -w)
echo "=========================================="
echo "TOSCA Geodesic Evaluation - All Test Shapes"
echo "=========================================="
echo "Model:       $MODEL_PATH"
echo "Shapes:      $SHAPE_LIST"
echo "Total:       $TOTAL shapes"
echo "Output base: $OUTPUT_BASE"
echo "Num sources: $NUM_SOURCES per shape"
echo "Ring:        $RING"
echo "Device:      $DEVICE"
echo "=========================================="
echo ""

# --- Run evaluation for each shape ---
PASS=0
FAIL=0
SKIP=0
RESULTS_SUMMARY=""

for SHAPE in $SHAPE_LIST; do
    SHAPE_OUTPUT_DIR="${OUTPUT_BASE}/${SHAPE}"

    # Skip existing?
    if $SKIP_EXISTING && [[ -d "$SHAPE_OUTPUT_DIR" ]] && [[ -f "$SHAPE_OUTPUT_DIR/evaluation_results.npz" || -f "$SHAPE_OUTPUT_DIR/propagation_results.npz" ]]; then
        echo "[SKIP] ${SHAPE} - results already exist in ${SHAPE_OUTPUT_DIR}"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Evaluating: ${SHAPE} ($((PASS + FAIL + SKIP + 1))/${TOTAL})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build args for single script
    SINGLE_ARGS="--model_path $MODEL_PATH --shape $SHAPE"
    SINGLE_ARGS="$SINGLE_ARGS --texture $TEXTURE --resolution $RESOLUTION --appearance $APPEARANCE"
    SINGLE_ARGS="$SINGLE_ARGS --num_sources $NUM_SOURCES --seed $SEED"
    SINGLE_ARGS="$SINGLE_ARGS --n_neighbors $N_NEIGHBORS --ring $RING --batch_size $BATCH_SIZE"
    SINGLE_ARGS="$SINGLE_ARGS --device $DEVICE"
    SINGLE_ARGS="$SINGLE_ARGS --output_dir $SHAPE_OUTPUT_DIR"

    if [[ -n "$USE_MAHALANOBIS" ]]; then
        SINGLE_ARGS="$SINGLE_ARGS $USE_MAHALANOBIS"
    fi
    if [[ -n "$NO_PLY" ]]; then
        SINGLE_ARGS="$SINGLE_ARGS $NO_PLY"
    fi
    if [[ -n "$ITERATION" ]]; then
        SINGLE_ARGS="$SINGLE_ARGS --iteration $ITERATION"
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        SINGLE_ARGS="$SINGLE_ARGS --extra \"$EXTRA_ARGS\""
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] bash scripts/tosca/evaluate_geodesic_single.sh $SINGLE_ARGS"
        PASS=$((PASS + 1))
        continue
    fi

    # Run evaluation
    if bash scripts/tosca/evaluate_geodesic_single.sh $SINGLE_ARGS; then
        PASS=$((PASS + 1))
        RESULTS_SUMMARY="${RESULTS_SUMMARY}\n  ✓ ${SHAPE}"
    else
        FAIL=$((FAIL + 1))
        RESULTS_SUMMARY="${RESULTS_SUMMARY}\n  ✗ ${SHAPE} (FAILED)"
        echo "WARNING: Evaluation failed for ${SHAPE}, continuing..."
    fi
done

# --- Print summary ---
echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Total:   ${TOTAL} shapes"
echo "Passed:  ${PASS}"
echo "Failed:  ${FAIL}"
echo "Skipped: ${SKIP}"
echo -e "Results:${RESULTS_SUMMARY}"
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo "=========================================="

# --- Aggregate metrics ---
if ! $DRY_RUN && [[ $PASS -gt 0 ]]; then
    echo ""
    echo "Aggregating metrics across all shapes..."
    python3 -c "
import json, os, sys
from pathlib import Path
import numpy as np

base = Path('${OUTPUT_BASE}')
all_metrics = {}
shapes = '${SHAPE_LIST}'.split()

for shape in shapes:
    report = base / shape / 'summary_report_evaluation_results.json'
    if report.exists():
        with open(report) as f:
            data = json.load(f)
        metrics = data.get('metrics', {})
        if metrics:
            all_metrics[shape] = metrics

if not all_metrics:
    print('No evaluation metrics found (ground truth may be missing).')
    sys.exit(0)

# Print table
print()
print(f'{'Shape':<15} {'MAE':>10} {'RMSE':>10} {'RelErr%':>10} {'MaxErr':>10} {'P50':>10} {'P90':>10}')
print('-' * 75)

all_mae, all_rmse, all_rel = [], [], []
for shape in sorted(all_metrics.keys()):
    m = all_metrics[shape]
    mae = m.get('mae', float('nan'))
    rmse = m.get('rmse', float('nan'))
    rel = m.get('relative_error_pct', float('nan'))
    maxe = m.get('max_error', float('nan'))
    p50 = m.get('p50_error', float('nan'))
    p90 = m.get('p90_error', float('nan'))
    print(f'{shape:<15} {mae:>10.4f} {rmse:>10.4f} {rel:>10.2f} {maxe:>10.4f} {p50:>10.4f} {p90:>10.4f}')
    all_mae.append(mae)
    all_rmse.append(rmse)
    all_rel.append(rel)

print('-' * 75)
print(f'{'Mean':<15} {np.nanmean(all_mae):>10.4f} {np.nanmean(all_rmse):>10.4f} {np.nanmean(all_rel):>10.2f}')
print(f'{'Median':<15} {np.nanmedian(all_mae):>10.4f} {np.nanmedian(all_rmse):>10.4f} {np.nanmedian(all_rel):>10.2f}')
print()

# Save aggregate
aggregate = {
    'per_shape': all_metrics,
    'aggregate': {
        'mean_mae': float(np.nanmean(all_mae)),
        'mean_rmse': float(np.nanmean(all_rmse)),
        'mean_relative_error_pct': float(np.nanmean(all_rel)),
        'median_mae': float(np.nanmedian(all_mae)),
        'median_rmse': float(np.nanmedian(all_rmse)),
        'median_relative_error_pct': float(np.nanmedian(all_rel)),
    }
}
agg_path = base / 'aggregate_metrics.json'
with open(agg_path, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f'Aggregate metrics saved to: {agg_path}')
" 2>&1 || echo "WARNING: Metric aggregation failed (non-critical)"
fi

# Exit with failure if any shape failed
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
