#!/bin/bash
#
# Generate polynomial training patches (outlier filtering) for all surfaces
#
# Uses the outlier_filtering configs which include:
#   - Outlier filtering enabled (disable_outlier_filtering: false)
#   - Configurable outlier params (median_multiplier, threshold_floor, etc.)
#   - n_neighbors=14, adaptive_k_boost=20, adaptive_max_steps=6
#   - normalize_per_patch: true
#
# Dataset configs used:
#   DataSets/configs/polynomial/outlier_filtering/paraboloid_all_one_source.yaml
#   DataSets/configs/polynomial/outlier_filtering/saddle_all_one_source.yaml
#   DataSets/configs/polynomial/outlier_filtering/hyperbolic_paraboloid_all_one_source.yaml
# (or 4-source variants when --four_sources is passed)
#
# USAGE:
#   ./scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh [options]
#
# OPTIONS:
#   --surfaces LIST       Comma-separated surfaces (default: all three)
#   --four_sources        Use 4-source configs instead of one-source
#   --num_iterations N    Number of training iterations per surface (default: from config)
#   --num_sources N       Override geodesic sources per iteration (default: from config)
#   --num_train_points N  Training points per iteration (default: from config)
#   --seed N              Random seed (default: 42)
#   --num_output_workers N Parallel workers per surface for KNN (default: 5)
#   --dry_run             Print commands without executing
#   --sequential          Run surfaces one at a time (default: parallel)
#
# EXAMPLES:
#   # Generate all polynomial datasets (parallel, one source)
#   ./scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh
#
#   # Generate 4-source variants
#   ./scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh --four_sources
#
#   # Generate only Paraboloid with more iterations
#   ./scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh --surfaces Paraboloid --num_iterations 5000
#
#   # Dry run
#   ./scripts/polynomial/outlier_filtering/patches/generate_polynomial_training_patches_outlier_filtering.sh --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
FOUR_SOURCES=false
NUM_ITERATIONS=""
NUM_SOURCES=""
NUM_TRAIN_POINTS=""
SEED=""
NUM_OUTPUT_WORKERS=""
DRY_RUN=false
SEQUENTIAL=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --surfaces)
            SURFACES="$2"
            shift 2
            ;;
        --four_sources)
            FOUR_SOURCES=true
            shift
            ;;
        --num_iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --num_sources)
            NUM_SOURCES="$2"
            shift 2
            ;;
        --num_train_points)
            NUM_TRAIN_POINTS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_output_workers)
            NUM_OUTPUT_WORKERS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate training patches (outlier filtering) for polynomial surfaces."
            echo "Each surface combines all levels (02,03,04) and lights (0-4)."
            echo ""
            echo "Options:"
            echo "  --surfaces LIST       Comma-separated surfaces (default: all)"
            echo "  --four_sources        Use 4-source configs instead of one-source"
            echo "  --num_iterations N    Iterations per surface (default: from config)"
            echo "  --num_sources N       Geodesic sources per iteration (default: from config)"
            echo "  --num_train_points N  Training points per iteration (default: from config)"
            echo "  --seed N              Random seed (default: from config)"
            echo "  --dry_run             Print commands without executing"
            echo "  --sequential          Run surfaces sequentially (default: parallel)"
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
# Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="DataSets/configs/polynomial/outlier_filtering"
GENERATE_SCRIPT="DataSets/create_gaussian_training_patches.py"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "  PYTHONPATH:    $PYTHONPATH"
echo "  Working dir:   $(pwd)"

if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: Generation script not found: $GENERATE_SCRIPT"
    exit 1
fi

# Map surface names to outlier_filtering config files
declare -A SURFACE_CONFIGS
if [ "$FOUR_SOURCES" = true ]; then
    SURFACE_CONFIGS["Paraboloid"]="$CONFIG_DIR/paraboloid_all.yaml"
    SURFACE_CONFIGS["Saddle"]="$CONFIG_DIR/saddle_all.yaml"
    SURFACE_CONFIGS["HyperbolicParaboloid"]="$CONFIG_DIR/hyperbolic_paraboloid_all.yaml"
else
    SURFACE_CONFIGS["Paraboloid"]="$CONFIG_DIR/paraboloid_all_one_source.yaml"
    SURFACE_CONFIGS["Saddle"]="$CONFIG_DIR/saddle_all_one_source.yaml"
    SURFACE_CONFIGS["HyperbolicParaboloid"]="$CONFIG_DIR/hyperbolic_paraboloid_all_one_source.yaml"
fi

IFS=',' read -ra SURFACE_ARRAY <<< "$SURFACES"

# Build optional CLI overrides
OVERRIDES=""
if [ -n "$NUM_ITERATIONS" ]; then
    OVERRIDES="$OVERRIDES --num_iterations $NUM_ITERATIONS"
fi
if [ -n "$NUM_SOURCES" ]; then
    OVERRIDES="$OVERRIDES --num_sources $NUM_SOURCES"
fi
if [ -n "$NUM_TRAIN_POINTS" ]; then
    OVERRIDES="$OVERRIDES --num_train_points $NUM_TRAIN_POINTS"
fi
if [ -n "$SEED" ]; then
    OVERRIDES="$OVERRIDES --seed $SEED"
fi
if [ -n "$NUM_OUTPUT_WORKERS" ]; then
    OVERRIDES="$OVERRIDES --num_output_workers $NUM_OUTPUT_WORKERS"
fi

# ============================================================================
# Generate patches for each surface
# ============================================================================
echo ""
echo "============================================================"
echo "Generate Polynomial Training Patches (outlier filtering)"
echo "============================================================"
echo "  Surfaces:      ${SURFACE_ARRAY[*]}"
echo "  Config dir:    $CONFIG_DIR"
echo "  Four sources:  $FOUR_SOURCES"
echo "  Overrides:     ${OVERRIDES:-<none>}"
echo ""

PIDS=()
for SURFACE in "${SURFACE_ARRAY[@]}"; do
    CONFIG="${SURFACE_CONFIGS[$SURFACE]}"
    if [ -z "$CONFIG" ]; then
        echo "Warning: No config found for surface '$SURFACE'. Skipping."
        continue
    fi
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file not found: $CONFIG"
        continue
    fi

    CMD="python $GENERATE_SCRIPT --config $CONFIG $OVERRIDES"
    echo "  [$SURFACE] $CMD"

    if [ "$DRY_RUN" = true ]; then
        continue
    fi

    if [ "$SEQUENTIAL" = true ]; then
        eval "$CMD"
    else
        eval "$CMD" &
        PIDS+=($!)
    fi
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

# Wait for parallel jobs
if [ "$SEQUENTIAL" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Waiting for ${#PIDS[@]} parallel jobs..."
    FAIL=0
    for PID in "${PIDS[@]}"; do
        wait "$PID" || ((FAIL++))
    done
    if [ "$FAIL" -gt 0 ]; then
        echo "Warning: $FAIL job(s) failed."
        exit 1
    fi
fi

echo ""
echo "All polynomial patch generation (outlier filtering) complete."
