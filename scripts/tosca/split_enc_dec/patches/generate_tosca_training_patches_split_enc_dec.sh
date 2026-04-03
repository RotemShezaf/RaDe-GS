#!/bin/bash
#
# Generate TOSCA training patches (split_enc_dec) for all animals
#
# This script generates Gaussian training patches for each TOSCA shape
# with the split_enc_dec configuration:
#   - Outlier filtering ENABLED (disable_outlier_filtering: false)
#   - n_neighbors=10, adaptive_k_boost=16
#   - normalize_all_neighbors: true
#
# For each shape it:
#   1. Discovers all available Gaussian outputs under SyntheticColmapData
#   2. Writes a .txt list file of those output paths
#   3. Generates a YAML config referencing the .txt list
#   4. Runs create_gaussian_training_patches.py with that config
#
# Dataset configs used from DataSets/configs/tosca/split_enc_dec/
#
# USAGE:
#   ./scripts/tosca/split_enc_dec/patches/generate_tosca_training_patches_split_enc_dec.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR      Synthetic COLMAP data base (default: TrainData/TOSCA/SyntheticColmapData)
#   --data_root DIR            Preprocessed TOSCA data root (default: TrainData/TOSCA/processed)
#   --animals LIST             Animal names WITHOUT index (e.g. "cat,dog"); expands to all indexed
#                              shapes found in data_root. (default: all 9 animals)
#   --textures LIST            Comma-separated texture names (default: colors)
#   --colmap_resolutions LIST  Comma-separated COLMAP resolutions (default: high_res)
#   --light_ids LIST           Comma-separated light IDs (default: 0,1,2,3,4)
#   --num_iterations N         Training iterations per shape (default: from config)
#   --num_sources N            Geodesic sources per iteration (default: from config)
#   --num_train_points N       Training points per iteration (default: from config)
#   --seed N                   Random seed (default: 42)
#   --num_output_workers N     Parallel workers for KNN (default: 5)
#   --dry_run                  Print commands without executing
#   --sequential               Run shapes one at a time (default: parallel)
#
# EXAMPLES:
#   # Generate patches for all TOSCA animals (parallel)
#   ./scripts/tosca/split_enc_dec/patches/generate_tosca_training_patches_split_enc_dec.sh
#
#   # Generate only for specific animals
#   ./scripts/tosca/split_enc_dec/patches/generate_tosca_training_patches_split_enc_dec.sh --animals "cat,dog"
#
#   # Dry run
#   ./scripts/tosca/split_enc_dec/patches/generate_tosca_training_patches_split_enc_dec.sh --dry_run

set -e

# Load animal→index map and expand_animals() helper
source "$(dirname "${BASH_SOURCE[0]}")/../../tosca_animal_map.sh"

# ============================================================================
# Default values
# ============================================================================
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
DATA_ROOT="TrainData/TOSCA/processed"
ANIMALS=""
TEXTURES="colors"
COLMAP_RESOLUTIONS="high_res"
LIGHT_IDS="0,1,2,3,4"
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
        --synth_data_base)     SYNTH_DATA_BASE="$2";     shift 2 ;;
        --data_root)           DATA_ROOT="$2";           shift 2 ;;
        --animals)             ANIMALS="$2";             shift 2 ;;
        --textures)            TEXTURES="$2";            shift 2 ;;
        --colmap_resolutions)  COLMAP_RESOLUTIONS="$2";  shift 2 ;;
        --light_ids)           LIGHT_IDS="$2";           shift 2 ;;
        --num_iterations)      NUM_ITERATIONS="$2";      shift 2 ;;
        --num_sources)         NUM_SOURCES="$2";         shift 2 ;;
        --num_train_points)    NUM_TRAIN_POINTS="$2";    shift 2 ;;
        --seed)                SEED="$2";                shift 2 ;;
        --num_output_workers)  NUM_OUTPUT_WORKERS="$2";  shift 2 ;;
        --dry_run)             DRY_RUN=true;             shift   ;;
        --sequential)          SEQUENTIAL=true;          shift   ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate TOSCA training patches with split_enc_dec configuration."
            echo ""
            echo "Options:"
            echo "  --animals LIST       Animal names (default: all 9)"
            echo "  --num_iterations N   Iterations per shape (default: from config)"
            echo "  --num_sources N      Geodesic sources per iteration (default: from config)"
            echo "  --dry_run            Print commands without executing"
            echo "  --sequential         Run shapes sequentially (default: parallel)"
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

CONFIG_DIR="DataSets/configs/tosca/split_enc_dec"
GENERATE_SCRIPT="DataSets/create_gaussian_training_patches.py"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "  PYTHONPATH:    $PYTHONPATH"
echo "  Working dir:   $(pwd)"

if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: Generation script not found: $GENERATE_SCRIPT"
    exit 1
fi

# ============================================================================
# Auto-detect animals if not specified
# ============================================================================
ALL_ANIMALS="cat centaur david dog gorilla horse michael victoria wolf"
if [ -n "$ANIMALS" ]; then
    IFS=',' read -ra ANIMAL_ARRAY <<< "$ANIMALS"
else
    IFS=' ' read -ra ANIMAL_ARRAY <<< "$ALL_ANIMALS"
fi

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
# Generate patches for each animal
# ============================================================================
echo ""
echo "============================================================"
echo "Generate TOSCA Training Patches (split_enc_dec)"
echo "============================================================"
echo "  Animals:       ${ANIMAL_ARRAY[*]}"
echo "  Config dir:    $CONFIG_DIR"
echo "  Overrides:     ${OVERRIDES:-<none>}"
echo ""

PIDS=()
for ANIMAL in "${ANIMAL_ARRAY[@]}"; do
    CONFIG="$CONFIG_DIR/tosca_${ANIMAL}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "Warning: Config file not found: $CONFIG. Skipping $ANIMAL."
        continue
    fi

    CMD="python $GENERATE_SCRIPT --config $CONFIG $OVERRIDES"
    echo "  [$ANIMAL] $CMD"

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
echo "All TOSCA patches (split_enc_dec) generated successfully."
