#!/bin/bash
#
# Generate training patches (new_dataset) for all polynomial surfaces
#
# Uses the new_dataset configs which include:
#   - num_sources=1 (one geodesic source)
#   - Updated KNN params (n_neighbors=6, adaptive_k_boost=18, adaptive_max_steps=12)
#   - Near-source oversampling (0.3)
#   - disable_outlier_filtering: true
#
# Dataset configs used:
#   DataSets/configs/polynomial/new_dataset/paraboloid_all_one_source.yaml
#   DataSets/configs/polynomial/new_dataset/saddle_all_one_source.yaml
#   DataSets/configs/polynomial/new_dataset/hyperbolic_paraboloid_all_one_source.yaml
#
# USAGE:
#   ./scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh [options]
#
# OPTIONS:
#   --surfaces LIST       Comma-separated surfaces (default: all three)
#   --num_iterations N    Number of training iterations per surface (default: from config)
#   --num_sources N       Override geodesic sources per iteration (default: 1, from config)
#   --num_train_points N  Training points per iteration (default: from config)
#   --seed N              Random seed (default: 42)
#   --num_output_workers N Parallel workers per surface for KNN (default: 5)
#   --dry_run             Print commands without executing
#   --sequential          Run surfaces one at a time (default: parallel)
#
# EXAMPLES:
#   # Generate all polynomial datasets (parallel)
#   ./scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh
#
#   # Generate only Paraboloid with more iterations
#   ./scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh --surfaces Paraboloid --num_iterations 5000
#
#   # Dry run
#   ./scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh --dry_run

set -e

# ============================================================================
# Default values
# ============================================================================
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
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
            echo "Generate training patches (new_dataset) for polynomial surfaces."
            echo "Each surface combines all levels (02,03,04) and lights (0-4)."
            echo ""
            echo "Options:"
            echo "  --surfaces LIST       Comma-separated surfaces (default: all)"
            echo "  --num_iterations N    Iterations per surface (default: from config)"
            echo "  --num_sources N       Geodesic sources per iteration (default: 1)"
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

CONFIG_DIR="DataSets/configs/polynomial/new_dataset"
GENERATE_SCRIPT="DataSets/create_gaussian_training_patches.py"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "  PYTHONPATH:    $PYTHONPATH"
echo "  Working dir:   $(pwd)"

if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: Generation script not found: $GENERATE_SCRIPT"
    exit 1
fi

# Map surface names to new_dataset config files
declare -A SURFACE_CONFIGS
SURFACE_CONFIGS["Paraboloid"]="$CONFIG_DIR/paraboloid_all_one_source.yaml"
SURFACE_CONFIGS["Saddle"]="$CONFIG_DIR/saddle_all_one_source.yaml"
SURFACE_CONFIGS["HyperbolicParaboloid"]="$CONFIG_DIR/hyperbolic_paraboloid_all_one_source.yaml"

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
# Print configuration
# ============================================================================
echo "============================================================"
echo "Generate Polynomial Training Patches (new_dataset)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Surfaces:    ${SURFACE_ARRAY[*]}"
echo "  Config dir:  $CONFIG_DIR"
echo "  Sequential:  $SEQUENTIAL"
echo "  Dry run:     $DRY_RUN"
if [ -n "$OVERRIDES" ]; then
    echo "  Overrides:  $OVERRIDES"
fi
echo ""

# ============================================================================
# Generate training patches for each surface
# ============================================================================
START_TIME=$(date +%s)
PIDS=()
FAILED=()

for surface in "${SURFACE_ARRAY[@]}"; do
    config_file="${SURFACE_CONFIGS[$surface]}"

    if [ -z "$config_file" ]; then
        echo "Error: Unknown surface '$surface'. Valid: Paraboloid, Saddle, HyperbolicParaboloid"
        FAILED+=("$surface (unknown)")
        continue
    fi

    if [ ! -f "$config_file" ]; then
        echo "Error: Config file not found: $config_file"
        FAILED+=("$surface (config missing)")
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Processing: $surface"
    echo "  Config: $config_file"
    echo "------------------------------------------------------------"

    CMD="python $GENERATE_SCRIPT --config $config_file $OVERRIDES"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
    elif [ "$SEQUENTIAL" = true ]; then
        echo "  Running sequentially..."
        if ! $CMD; then
            echo "  Warning: $surface generation failed"
            FAILED+=("$surface")
        fi
    else
        echo "  Running in background..."
        $CMD &
        PIDS+=("$!:$surface")
    fi

    echo ""
done

# ============================================================================
# Wait for background processes
# ============================================================================
if [ "$DRY_RUN" = false ] && [ "$SEQUENTIAL" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for all surfaces to complete..."
    echo ""

    for entry in "${PIDS[@]}"; do
        pid="${entry%%:*}"
        surface="${entry##*:}"
        if ! wait "$pid"; then
            echo "  Warning: $surface generation failed"
            FAILED+=("$surface")
        else
            echo "  $surface completed successfully"
        fi
    done
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Generation Complete"
echo "============================================================"
echo "  Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Surfaces processed: ${#SURFACE_ARRAY[@]}"
echo "  Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed surfaces:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi

echo ""
echo "Output directories:"
for surface in "${SURFACE_ARRAY[@]}"; do
    config_file="${SURFACE_CONFIGS[$surface]}"
    if [ -n "$config_file" ] && [ -f "$config_file" ]; then
        output_dir=$(grep "^output_dir:" "$config_file" | awk '{print $2}' | tr -d '"')
        echo "  $surface: $output_dir"
    fi
done

echo ""
echo "To load the combined new_dataset:"
echo "  from DataSets.gaussian_dataset import CombinedGaussianPatchDataset"
echo "  dataset = CombinedGaussianPatchDataset("
echo "      config='DataSets/configs/polynomial/new_dataset/combined_polynomial_all_one_source.yaml',"
echo "      attributes=['xyz'], ring=3"
echo "  )"
echo ""
echo "Done!"
