#!/bin/bash
#
# Generate polynomial training patches (split_enc_dec) for all surfaces
#
# Uses the split_enc_dec configs which include:
#   - Outlier filtering disabled (disable_outlier_filtering: true)
#   - n_neighbors=10, adaptive_k_boost=16
#   - normalize_all_neighbors: true
#
# Dataset configs used:
#   DataSets/configs/polynomial/split_enc_dec/paraboloid_all_one_source.yaml
#   DataSets/configs/polynomial/split_enc_dec/saddle_all_one_source.yaml
#   DataSets/configs/polynomial/split_enc_dec/hyperbolic_paraboloid_all_one_source.yaml
# (or 4-source variants when --four_sources is passed)
#
# USAGE:
#   ./scripts/polynomial/split_enc_dec/patches/generate_polynomial_training_patches_split_enc_dec.sh [options]
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
#   ./scripts/polynomial/split_enc_dec/patches/generate_polynomial_training_patches_split_enc_dec.sh
#
#   # Generate 4-source variants
#   ./scripts/polynomial/split_enc_dec/patches/generate_polynomial_training_patches_split_enc_dec.sh --four_sources
#
#   # Dry run
#   ./scripts/polynomial/split_enc_dec/patches/generate_polynomial_training_patches_split_enc_dec.sh --dry_run

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
        --surfaces)            SURFACES="$2";            shift 2 ;;
        --four_sources)        FOUR_SOURCES=true;         shift   ;;
        --num_iterations)      NUM_ITERATIONS="$2";       shift 2 ;;
        --num_sources)         NUM_SOURCES="$2";          shift 2 ;;
        --num_train_points)    NUM_TRAIN_POINTS="$2";     shift 2 ;;
        --seed)                SEED="$2";                 shift 2 ;;
        --num_output_workers)  NUM_OUTPUT_WORKERS="$2";   shift 2 ;;
        --dry_run)             DRY_RUN=true;              shift   ;;
        --sequential)          SEQUENTIAL=true;           shift   ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate training patches (split_enc_dec) for polynomial surfaces."
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
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/DataSets/configs/polynomial/split_enc_dec"

# ============================================================================
# Map surface names to config files
# ============================================================================
declare -A SURFACE_CONFIG
if [ "$FOUR_SOURCES" = true ]; then
    SURFACE_CONFIG["Paraboloid"]="$CONFIG_DIR/paraboloid_all.yaml"
    SURFACE_CONFIG["Saddle"]="$CONFIG_DIR/saddle_all.yaml"
    SURFACE_CONFIG["HyperbolicParaboloid"]="$CONFIG_DIR/hyperbolic_paraboloid_all.yaml"
else
    SURFACE_CONFIG["Paraboloid"]="$CONFIG_DIR/paraboloid_all_one_source.yaml"
    SURFACE_CONFIG["Saddle"]="$CONFIG_DIR/saddle_all_one_source.yaml"
    SURFACE_CONFIG["HyperbolicParaboloid"]="$CONFIG_DIR/hyperbolic_paraboloid_all_one_source.yaml"
fi

# ============================================================================
# Build override arguments
# ============================================================================
OVERRIDE_ARGS=""
[ -n "$NUM_ITERATIONS" ]     && OVERRIDE_ARGS="$OVERRIDE_ARGS --num_iterations $NUM_ITERATIONS"
[ -n "$NUM_SOURCES" ]        && OVERRIDE_ARGS="$OVERRIDE_ARGS --num_sources $NUM_SOURCES"
[ -n "$NUM_TRAIN_POINTS" ]   && OVERRIDE_ARGS="$OVERRIDE_ARGS --num_train_points $NUM_TRAIN_POINTS"
[ -n "$SEED" ]               && OVERRIDE_ARGS="$OVERRIDE_ARGS --seed $SEED"
[ -n "$NUM_OUTPUT_WORKERS" ] && OVERRIDE_ARGS="$OVERRIDE_ARGS --num_output_workers $NUM_OUTPUT_WORKERS"

# ============================================================================
# Generate patches
# ============================================================================
echo "============================================================"
echo "Generate Polynomial Patches (split_enc_dec)"
echo "============================================================"
echo "  Surfaces:    $SURFACES"
echo "  Four sources: $FOUR_SOURCES"
echo "  Overrides:   ${OVERRIDE_ARGS:-<none>}"
echo ""

IFS=',' read -ra SURFACE_LIST <<< "$SURFACES"
PIDS=()

for surface in "${SURFACE_LIST[@]}"; do
    config="${SURFACE_CONFIG[$surface]}"
    if [ -z "$config" ]; then
        echo "ERROR: Unknown surface '$surface'"
        exit 1
    fi

    CMD="cd $SCRIPT_DIR && python DataSets/create_gaussian_training_patches.py --config $config $OVERRIDE_ARGS"

    echo "[$surface] Config: $config"
    echo "[$surface] Command: $CMD"
    echo ""

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
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

# Wait for parallel jobs
if [ "$SEQUENTIAL" = false ]; then
    echo "Waiting for all surfaces to finish..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done
    if [ "$FAILED" -gt 0 ]; then
        echo "WARNING: $FAILED surface(s) failed."
        exit 1
    fi
fi

echo ""
echo "All polynomial patches (split_enc_dec) generated successfully."
