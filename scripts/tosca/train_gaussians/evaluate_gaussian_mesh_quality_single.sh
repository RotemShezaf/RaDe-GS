#!/bin/bash
#
# Evaluate Gaussian-mesh quality for a single TOSCA shape.
# Writes quality_report.json to the output directory.
#
# USAGE:
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_single.sh <shape> [options]
#
# ARGUMENTS:
#   shape              TOSCA shape name (e.g., cat0, dog0, horse5)
#
# OPTIONS:
#   --texture NAME     Texture name (default: blue)
#   --resolution RES   COLMAP resolution (default: high_res)
#   --appearance NAME  Appearance mode: decoupled_appearance | light_N (default: decoupled_appearance)
#   --output NAME      Output folder name (default: output)
#   --iteration N      Iteration to evaluate (default: 30000)
#   --gt_mesh PATH     Path to GT mesh for Chamfer distance evaluation
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_single.sh cat0
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_single.sh dog3 --iteration 15000

set -e

# ============================================================================
# Defaults
# ============================================================================
TEXTURE="blue"
RESOLUTION="high_res"
APPEARANCE="decoupled_appearance"
OUTPUT_NAME="output"
ITERATION=30000
GT_MESH=""
DRY_RUN=false
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"

# ============================================================================
# Parse arguments
# ============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <shape> [options]"
    echo "  e.g.: $0 cat0"
    exit 1
fi

SHAPE="$1"; shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --texture)      TEXTURE="$2";      shift 2 ;;
        --resolution)   RESOLUTION="$2";   shift 2 ;;
        --appearance)   APPEARANCE="$2";   shift 2 ;;
        --output)       OUTPUT_NAME="$2";  shift 2 ;;
        --iteration)    ITERATION="$2";    shift 2 ;;
        --gt_mesh)      GT_MESH="$2";      shift 2 ;;
        --dry_run)      DRY_RUN=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/evaluate_gaussian_mesh_quality.py"

SOURCE_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/$SHAPE/$RESOLUTION/$APPEARANCE"
MODEL_PATH="$SOURCE_PATH/$OUTPUT_NAME"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
[ -x "$PYTHON3" ] || PYTHON3=python3

# ============================================================================
# Validate
# ============================================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Output directory not found: $MODEL_PATH"
    exit 1
fi

# ============================================================================
# Run
# ============================================================================
CMD="$PYTHON3 $EVAL_SCRIPT --output_dir $MODEL_PATH --iteration $ITERATION"

if [ -n "$GT_MESH" ]; then
    CMD="$CMD --gt_mesh $GT_MESH"
fi

echo "============================================================"
echo "Evaluating: $SHAPE ($TEXTURE, $RESOLUTION, $APPEARANCE)"
echo "  Output: $MODEL_PATH"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] $CMD"
else
    eval "$CMD"
fi
