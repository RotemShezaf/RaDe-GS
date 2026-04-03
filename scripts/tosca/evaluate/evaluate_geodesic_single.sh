#!/bin/bash
# ============================================================================
# Evaluate geodesic distance propagation on a single TOSCA shape
#
# This script runs evaluate_geodesic.py on one shape from the TOSCA dataset.
# It uses the extracted Gaussian splatting output and ground truth geodesic
# distances to compute evaluation metrics.
#
# Usage:
#   bash scripts/tosca/evaluate/evaluate_geodesic_single.sh --model_path <path> --shape cat0
#   bash scripts/tosca/evaluate/evaluate_geodesic_single.sh --model_path checkpoints/my_model/best_model.pth --shape horse18
#   bash scripts/tosca/evaluate/evaluate_geodesic_single.sh --model_path <path> --shape cat0 --num_sources 10
#   bash scripts/tosca/evaluate/evaluate_geodesic_single.sh --help
# ============================================================================

set -e

# --- Defaults ---
SHAPE=""
MODEL_PATH=""
TEXTURE="blue_texture"
RESOLUTION="high_res"
APPEARANCE="decoupled_appearance"
NUM_SOURCES=5
SEED=42
N_NEIGHBORS=16
RING=2
BATCH_SIZE=32
DEVICE="cuda"
USE_MAHALANOBIS="--use_mahalanobis"
EXPORT_PLY=""  # empty = export PLY by default
OUTPUT_DIR=""  # auto-compute if empty
ITERATION=""   # empty = use highest available
EXTRA_ARGS=""
DRY_RUN=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2"; shift 2 ;;
        --shape)            SHAPE="$2"; shift 2 ;;
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
        --no_ply)           EXPORT_PLY="--no_ply"; shift ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --iteration)        ITERATION="$2"; shift 2 ;;
        --extra)            EXTRA_ARGS="$2"; shift 2 ;;
        --dry_run)          DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 --model_path <path> --shape <shape_name> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path PATH    Path to trained geodesic model checkpoint"
            echo "  --shape NAME         TOSCA shape name (e.g., cat0, horse18, dog10)"
            echo ""
            echo "Options:"
            echo "  --texture NAME       Texture type (default: blue_texture)"
            echo "  --resolution NAME    Resolution (default: high_res)"
            echo "  --appearance NAME    Appearance mode (default: decoupled_appearance)"
            echo "  --num_sources N      Number of random source points (default: 5)"
            echo "  --seed N             Random seed for source selection (default: 42)"
            echo "  --n_neighbors N      Number of ring-1 neighbors (default: 16)"
            echo "  --ring N             Ring level [1-4] (default: 2)"
            echo "  --batch_size N       Batch size for predictions (default: 32)"
            echo "  --device DEVICE      cuda or cpu (default: cuda)"
            echo "  --no_mahalanobis     Use Euclidean instead of Mahalanobis distance"
            echo "  --no_ply             Skip PLY visualization export"
            echo "  --output_dir PATH    Override output directory"
            echo "  --iteration N        Gaussian iteration (default: highest)"
            echo "  --extra \"ARGS\"       Extra args forwarded to evaluate_geodesic.py"
            echo "  --dry_run            Print command without executing"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Validate required arguments ---
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model_path is required"
    echo "Usage: $0 --model_path <path> --shape <shape_name>"
    exit 1
fi

if [[ -z "$SHAPE" ]]; then
    echo "ERROR: --shape is required"
    echo "Usage: $0 --model_path <path> --shape <shape_name>"
    exit 1
fi

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

GAUSSIAN_OUTPUT="TrainData/TOSCA/SyntheticColmapData/${TEXTURE}/${SHAPE}/${RESOLUTION}/${APPEARANCE}/output"
GEODESIC_DATA="${GAUSSIAN_OUTPUT}/geodesic_distance/gt_geodesic.npz"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="geodesic_eval_results/tosca/${SHAPE}"
fi

# --- Validate paths ---
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -d "$GAUSSIAN_OUTPUT" ]]; then
    echo "ERROR: Gaussian output not found: $GAUSSIAN_OUTPUT"
    echo "Have you trained Gaussian splatting on ${SHAPE}?"
    exit 1
fi

# Check for ground truth geodesic data
GEODESIC_FLAG=""
if [[ -f "$GEODESIC_DATA" ]]; then
    GEODESIC_FLAG="--geodesic_data $GEODESIC_DATA"
    echo "Ground truth geodesic data found: $GEODESIC_DATA"
else
    echo "WARNING: No ground truth geodesic data at $GEODESIC_DATA"
    echo "  Evaluation will run without ground truth comparison."
fi

# --- Generate random source indices ---
# Use Python to generate random source indices based on the number of Gaussians
SOURCE_INDICES=$(python3 -c "
import numpy as np
from pathlib import Path
import sys, glob

# Find point cloud directory
pc_dir = Path('${GAUSSIAN_OUTPUT}/point_cloud')
if not pc_dir.exists():
    print('ERROR: point_cloud directory not found', file=sys.stderr)
    sys.exit(1)

# Find the highest iteration
iters = sorted([d.name for d in pc_dir.iterdir() if d.is_dir()])
if not iters:
    print('ERROR: no iteration directories found', file=sys.stderr)
    sys.exit(1)

# Load the ply to count Gaussians
from plyfile import PlyData
ply_path = pc_dir / iters[-1] / 'point_cloud.ply'
if not ply_path.exists():
    print(f'ERROR: {ply_path} not found', file=sys.stderr)
    sys.exit(1)

plydata = PlyData.read(str(ply_path))
num_gaussians = len(plydata['vertex'])

# Generate random source indices
rng = np.random.RandomState(${SEED})
sources = rng.choice(num_gaussians, size=min(${NUM_SOURCES}, num_gaussians), replace=False)
print(' '.join(str(s) for s in sorted(sources)))
")

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to generate source indices"
    exit 1
fi

echo ""
echo "=========================================="
echo "Geodesic Evaluation: ${SHAPE}"
echo "=========================================="
echo "Model:           $MODEL_PATH"
echo "Gaussian output: $GAUSSIAN_OUTPUT"
echo "Output dir:      $OUTPUT_DIR"
echo "Source indices:   $SOURCE_INDICES"
echo "Num sources:     $NUM_SOURCES"
echo "Ring:            $RING"
echo "N neighbors:     $N_NEIGHBORS"
echo "Device:          $DEVICE"
echo "=========================================="

# --- Build command ---
ITER_FLAG=""
if [[ -n "$ITERATION" ]]; then
    ITER_FLAG="--iteration $ITERATION"
fi

CMD="python geodesic_propagation/evaluate_geodesic.py \
    --model_path $MODEL_PATH \
    --gaussian_output $GAUSSIAN_OUTPUT \
    --source_indices $SOURCE_INDICES \
    --output_dir $OUTPUT_DIR \
    --n_neighbors $N_NEIGHBORS \
    --ring $RING \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    $USE_MAHALANOBIS \
    $GEODESIC_FLAG \
    $ITER_FLAG \
    $EXPORT_PLY \
    $EXTRA_ARGS"

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would execute:"
    echo "$CMD"
    exit 0
fi

echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation complete for: ${SHAPE}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
