#!/bin/bash
#
# Evaluate Gaussian-mesh quality for ALL TOSCA shapes (blue_texture, decoupled_appearance).
# Wraps evaluate_gaussian_mesh_quality_single.sh across all shapes.
#
# Writes quality_report.json into each output directory.
#
# USAGE:
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_all.sh [options]
#
# OPTIONS:
#   --shapes LIST      Comma-separated shapes (default: auto-detect from blue_texture)
#   --animals LIST     Animal names WITHOUT index; expands to all indexed shapes
#   --iteration N      Iteration to evaluate (default: 30000)
#   --dry_run          Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_all.sh
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_all.sh --shapes "cat0,dog0"
#   bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_all.sh --dry_run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../tosca_animal_map.sh"

# ============================================================================
# Defaults
# ============================================================================
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
SHAPES=""
ANIMALS="cat,centaur,david,dog,gorilla,horse,lioness,michael,seahorse,shark,victoria,wolf"
TEXTURE="blue"
RESOLUTION="high_res"
APPEARANCE="decoupled_appearance"
OUTPUT_NAME="output"
ITERATION=30000
DRY_RUN=""

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --shapes)       SHAPES="$2";       shift 2 ;;
        --animals)      ANIMALS="$2";      shift 2 ;;
        --iteration)    ITERATION="$2";    shift 2 ;;
        --dry_run)      DRY_RUN="--dry_run"; shift ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Expand animals -> shapes
# ============================================================================
if [ -z "$SHAPES" ] && [ -n "$ANIMALS" ]; then
    SHAPES="$(expand_animals "$ANIMALS")"
fi

IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"

# ============================================================================
# Run
# ============================================================================
TOTAL=${#SHAPE_ARRAY[@]}
COMPLETED=0
FAILED=0
SKIPPED=0

echo "============================================================"
echo "Evaluate Gaussian-Mesh Quality — ALL TOSCA"
echo "  Shapes: ${SHAPE_ARRAY[*]}"
echo "  Texture: $TEXTURE"
echo "  Resolution: $RESOLUTION"
echo "  Iteration: $ITERATION"
echo "============================================================"
echo ""

for shape in "${SHAPE_ARRAY[@]}"; do
    MODEL_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/$shape/$RESOLUTION/$APPEARANCE/$OUTPUT_NAME"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "[SKIP] No output dir: $MODEL_PATH"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "------------------------------------------------------------"
    if bash "$SCRIPT_DIR/evaluate_gaussian_mesh_quality_single.sh" \
        "$shape" \
        --texture "$TEXTURE" \
        --resolution "$RESOLUTION" \
        --appearance "$APPEARANCE" \
        --output "$OUTPUT_NAME" \
        --iteration "$ITERATION" \
        $DRY_RUN; then
        COMPLETED=$((COMPLETED + 1))
    else
        echo "[FAILED] $shape"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "  Total: $TOTAL  Completed: $COMPLETED  Skipped: $SKIPPED  Failed: $FAILED"
echo "============================================================"
