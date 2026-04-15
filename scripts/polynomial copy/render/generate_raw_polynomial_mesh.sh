#!/bin/bash
#
# Generate raw polynomial meshes and point clouds
#
# This script runs GenerateRawPolynomialMesh.py to create triangulated meshes
# and point clouds for polynomial surfaces:
#   - Paraboloid: z = x^2 + y^2
#   - Saddle: z = x^2 - y^2
#   - HyperbolicParaboloid: z = x^2 - y^2 + xy
#
# Each surface is generated at multiple resolution levels.
#
# USAGE:
#   ./generate_raw_polynomial_mesh.sh [options]
#
# OPTIONS:
#   --base_resolution N    Highest resolution grid size (default: 1000)
#   --num_levels N         Number of resolution levels (default: 6)
#   --output_dir DIR       Output directory (default: TrainData/Polynomial/raw)
#   --x_min FLOAT          Minimum x value (default: -0.8)
#   --x_max FLOAT          Maximum x value (default: 0.8)
#   --y_min FLOAT          Minimum y value (default: -0.8)
#   --y_max FLOAT          Maximum y value (default: 0.8)
#   --adaptive             Use adaptive sampling for uniform arc length spacing
#   --dry_run              Print command without executing
#
# EXAMPLE:
#   # Generate with default settings
#   ./generate_raw_polynomial_mesh.sh
#
#   # Generate with custom resolution
#   ./generate_raw_polynomial_mesh.sh --base_resolution 500 --num_levels 4
#
#   # Generate with adaptive sampling
#   ./generate_raw_polynomial_mesh.sh --adaptive
#
#   # Dry run to see what would be executed
#   ./generate_raw_polynomial_mesh.sh --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
BASE_RESOLUTION=1024
NUM_LEVELS=6
OUTPUT_DIR="TrainData/Polynomial/raw"
X_MIN=-0.8
X_MAX=0.8
Y_MIN=-0.8
Y_MAX=0.8
ADAPTIVE=""
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_resolution)
            BASE_RESOLUTION="$2"
            shift 2
            ;;
        --num_levels)
            NUM_LEVELS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --x_min)
            X_MIN="$2"
            shift 2
            ;;
        --x_max)
            X_MAX="$2"
            shift 2
            ;;
        --y_min)
            Y_MIN="$2"
            shift 2
            ;;
        --y_max)
            Y_MAX="$2"
            shift 2
            ;;
        --adaptive)
            ADAPTIVE="--adaptive"
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --base_resolution N    Highest resolution grid size (default: 1000)"
            echo "  --num_levels N         Number of resolution levels (default: 6)"
            echo "  --output_dir DIR       Output directory (default: TrainData/Polynomial/raw)"
            echo "  --x_min FLOAT          Minimum x value (default: -0.8)"
            echo "  --x_max FLOAT          Maximum x value (default: 0.8)"
            echo "  --y_min FLOAT          Minimum y value (default: -0.8)"
            echo "  --y_max FLOAT          Maximum y value (default: 0.8)"
            echo "  --adaptive             Use adaptive sampling for uniform arc length spacing"
            echo "  --dry_run              Print command without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Path to Python script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/GenerateData/GenerateRawPolynomialMesh.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Generate Raw Polynomial Meshes"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Base resolution:   $BASE_RESOLUTION"
echo "  Number of levels:  $NUM_LEVELS"
echo "  Output directory:  $OUTPUT_DIR"
echo "  X range:           [$X_MIN, $X_MAX]"
echo "  Y range:           [$Y_MIN, $Y_MAX]"
echo "  Adaptive sampling: ${ADAPTIVE:-disabled}"
echo "  Dry run:           $DRY_RUN"
echo ""

# ============================================================================
# Build and run command
# ============================================================================
CMD="python $PYTHON_SCRIPT \
    --base_resolution $BASE_RESOLUTION \
    --num_levels $NUM_LEVELS \
    --output_dir $OUTPUT_DIR \
    --x_min $X_MIN \
    --x_max $X_MAX \
    --y_min $Y_MIN \
    --y_max $Y_MAX \
    $ADAPTIVE"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] $CMD"
else
    echo "Running mesh generation..."
    echo ""
    $CMD
    echo ""
    echo "============================================================"
    echo "Done! Meshes saved to: $OUTPUT_DIR"
    echo "============================================================"
fi
