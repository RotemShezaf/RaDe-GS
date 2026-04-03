#!/bin/bash
#
# Generate training patches for all TOSCA shapes
#
# This script generates Gaussian training patches for each TOSCA shape.
# For each shape (and texture combination), it:
#   1. Discovers all available Gaussian outputs under SyntheticColmapData
#   2. Writes a .txt list file of those output paths
#   3. Generates a YAML config referencing the .txt list
#   4. Runs create_gaussian_training_patches.py with that config
#
# USAGE:
#   ./scripts/generate_tosca_training_patches.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR      Synthetic COLMAP data base (default: TrainData/TOSCA/SyntheticColmapData)
#   --data_root DIR            Preprocessed TOSCA data root (used for shape auto-detection)
#                              (default: TrainData/TOSCA/processed)
#   --shapes LIST              Comma-separated shape names with index (default: auto-detect)
#   --animals LIST             Animal names WITHOUT index (e.g. "cat,dog"); expands to all indexed
#                              shapes found in data_root. Takes priority over --shapes when set.
#   --textures LIST            Comma-separated texture names (default: colors)
#   --colmap_resolutions LIST  Comma-separated COLMAP resolutions to include (default: high_res)
#   --light_ids LIST           Comma-separated light IDs to include (default: 0,1,2,3,4)
#   --use_decoupled_appearance  Scan for 'decoupled_appearance' subdirs instead of 'light_{id}'
#   --output_name NAME         Gaussian training output folder name (default: output)
#   --config_dir DIR           Directory to write generated YAML configs (default: DataSets/configs/tosca)
#   --num_iterations N         Training iterations per shape (default: 1000)
#   --num_sources N            Geodesic sources per iteration (default: 4)
#   --num_train_points N       Training points per iteration (default: 50)
#   --seed N                   Random seed (default: 42)
#   --num_output_workers N     Parallel workers for KNN (default: 5)
#   --dry_run                  Print commands without executing
#   --sequential               Run shapes one at a time (default: parallel)
#   --combined_config          Also generate a combined YAML for all shapes
#
# EXAMPLES:
#   # Generate patches for all auto-detected TOSCA shapes (parallel)
#   ./scripts/generate_tosca_training_patches.sh
#
#   # Generate only for specific shapes
#   ./scripts/generate_tosca_training_patches.sh --shapes "cat0,cat1,centaur1"
#
#   # Generate with specific texture and also produce combined config
#   ./scripts/generate_tosca_training_patches.sh --textures colors --combined_config
#
#   # Dry run
#   ./scripts/generate_tosca_training_patches.sh --dry_run

set -e

# Load animal→index map and expand_animals() helper
# Edit scripts/tosca/tosca_animal_map.sh to control which shapes are processed.
source "$(dirname "${BASH_SOURCE[0]}")/../tosca_animal_map.sh"

# ============================================================================
# Default values
# ============================================================================
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
DATA_ROOT="TrainData/TOSCA/processed"
SHAPES=""            # Empty = auto-detect
ANIMALS=""           # Animal names without index; overrides SHAPES when set
TEXTURES="colors"
COLMAP_RESOLUTIONS="high_res"
LIGHT_IDS="0,1,2,3,4"
USE_DECOUPLED_APPEARANCE=false  # When true, scan for 'decoupled_appearance' subdir
OUTPUT_NAME="output"
CONFIG_DIR="DataSets/configs/tosca"
NUM_ITERATIONS=""
NUM_SOURCES=""
NUM_TRAIN_POINTS=""
SEED=""
NUM_OUTPUT_WORKERS=""
DRY_RUN=false
SEQUENTIAL=false
COMBINED_CONFIG=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base)
            SYNTH_DATA_BASE="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --shapes)
            SHAPES="$2"
            shift 2
            ;;
        --animals)
            ANIMALS="$2"
            shift 2
            ;;
        --textures)
            TEXTURES="$2"
            shift 2
            ;;
        --colmap_resolutions)
            COLMAP_RESOLUTIONS="$2"
            shift 2
            ;;
        --light_ids)
            LIGHT_IDS="$2"
            shift 2
            ;;
        --use_decoupled_appearance)
            USE_DECOUPLED_APPEARANCE=true
            shift
            ;;
        --output_name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --config_dir)
            CONFIG_DIR="$2"
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
        --combined_config)
            COMBINED_CONFIG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Generate TOSCA training patches for all (or selected) shapes."
            echo "Configs and source lists are auto-generated per shape+texture."
            echo ""
            echo "Options:"
            echo "  --synth_data_base DIR      Synthetic COLMAP data (default: TrainData/TOSCA/SyntheticColmapData)"
            echo "  --data_root DIR            Preprocessed TOSCA root for auto-detection (default: TrainData/TOSCA/processed)"
            echo "  --shapes LIST              Comma-separated shape names with index (default: auto-detect)"
            echo "  --animals LIST             Animal names without index (e.g. \"cat,dog\"); expands to all indexed shapes"
            echo "  --textures LIST            Comma-separated textures (default: colors)"
            echo "  --colmap_resolutions LIST  Comma-separated COLMAP resolutions (default: high_res)"
            echo "  --light_ids LIST           Comma-separated light IDs (default: 0,1,2,3,4)"
            echo "  --use_decoupled_appearance Scan for 'decoupled_appearance' subdir (overrides light_ids)"
            echo "  --output_name NAME         Gaussian training output name (default: output)"
            echo "  --config_dir DIR           Output dir for generated YAML configs (default: DataSets/configs/tosca)"
            echo "  --num_iterations N         Iterations per shape (default: from config)"
            echo "  --num_sources N            Geodesic sources per iteration"
            echo "  --num_train_points N       Training points per iteration"
            echo "  --seed N                   Random seed"
            echo "  --num_output_workers N     KNN parallel workers"
            echo "  --dry_run                  Print commands without executing"
            echo "  --sequential               Run shapes sequentially (default: parallel)"
            echo "  --combined_config          Also create a combined config for all shapes"
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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "  PYTHONPATH:    $PYTHONPATH"
echo "  Working dir:   $(pwd)"

GENERATE_SCRIPT="DataSets/create_tosca_training_patches.py"
if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: Generation script not found: $GENERATE_SCRIPT"
    exit 1
fi

# Create config and sources directories
GAUSSIAN_SOURCES_DIR="$CONFIG_DIR/gaussian_sources"
mkdir -p "$CONFIG_DIR"
mkdir -p "$GAUSSIAN_SOURCES_DIR"

# ============================================================================
# Convert comma-separated lists to arrays
# ============================================================================
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra RESOLUTION_ARRAY <<< "$COLMAP_RESOLUTIONS"
if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    LIGHT_ID_ARRAY=("__decoupled__")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Expand --animals to indexed shape names using ANIMAL_INDEX_MAP
# (edit scripts/tosca/tosca_animal_map.sh to add/remove indices per animal)
# ============================================================================
if [ -n "$ANIMALS" ]; then
    SHAPES="$(expand_animals "$ANIMALS")"
    if [ -z "$SHAPES" ]; then
        echo "Error: No shapes found for animals: $ANIMALS"
        exit 1
    fi
    echo "Expanded animals '$ANIMALS' -> shapes: $SHAPES"
fi
# ============================================================================
# Auto-detect shapes if not specified
# ============================================================================
if [ -z "$SHAPES" ]; then
    echo "Auto-detecting TOSCA shapes ..."
    detected_shapes=()

    # Try from DATA_ROOT first (processed meshes)
    if [ -d "$DATA_ROOT" ]; then
        for shape_dir in "$DATA_ROOT"/*/; do
            [ -d "$shape_dir" ] || continue
            if ls "$shape_dir"/mesh_high_res_*.ply 2>/dev/null | grep -q .; then
                detected_shapes+=("$(basename "$shape_dir")")
            fi
        done
    fi

    # Fall back: detect from synth_data_base using first texture
    if [ ${#detected_shapes[@]} -eq 0 ] && [ -d "$SYNTH_DATA_BASE" ]; then
        first_texture="${TEXTURE_ARRAY[0]}"
        texture_dir="$SYNTH_DATA_BASE/${first_texture}_texture"
        if [ -d "$texture_dir" ]; then
            for shape_dir in "$texture_dir"/*/; do
                [ -d "$shape_dir" ] || continue
                detected_shapes+=("$(basename "$shape_dir")")
            done
        fi
    fi

    if [ ${#detected_shapes[@]} -eq 0 ]; then
        echo "Error: No TOSCA shapes found."
        echo "       Checked: $DATA_ROOT"
        echo "       Checked: $SYNTH_DATA_BASE"
        echo "       Please specify shapes with --shapes, or run preprocess_tosca.py and"
        echo "       create_synthetic_colmap_dataset_from_mesh_tosca.py first."
        exit 1
    fi
    SHAPES=$(IFS=','; echo "${detected_shapes[*]}")
    echo "  Found ${#detected_shapes[@]} shapes: $SHAPES"
fi

IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"

# Build optional CLI overrides for create_gaussian_training_patches.py
OVERRIDES=""
[ -n "$NUM_ITERATIONS" ]    && OVERRIDES="$OVERRIDES --num_iterations $NUM_ITERATIONS"
[ -n "$NUM_SOURCES" ]       && OVERRIDES="$OVERRIDES --num_sources $NUM_SOURCES"
[ -n "$NUM_TRAIN_POINTS" ]  && OVERRIDES="$OVERRIDES --num_train_points $NUM_TRAIN_POINTS"
[ -n "$SEED" ]              && OVERRIDES="$OVERRIDES --seed $SEED"
[ -n "$NUM_OUTPUT_WORKERS" ] && OVERRIDES="$OVERRIDES --num_output_workers $NUM_OUTPUT_WORKERS"

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Generate TOSCA Training Patches"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Shapes:             ${SHAPE_ARRAY[*]}"
echo "  Textures:           ${TEXTURE_ARRAY[*]}"
echo "  COLMAP resolutions: ${RESOLUTION_ARRAY[*]}"
echo "  Light IDs:          ${LIGHT_ID_ARRAY[*]}"
echo "  Output name:        $OUTPUT_NAME"
echo "  Config dir:         $CONFIG_DIR"
echo "  Sequential:         $SEQUENTIAL"
echo "  Dry run:            $DRY_RUN"
echo "  Combined config:    $COMBINED_CONFIG"
[ -n "$OVERRIDES" ] && echo "  Overrides:          $OVERRIDES"
echo ""

# ============================================================================
# Generate training patches for each shape × texture combination
# ============================================================================
START_TIME=$(date +%s)
PIDS=()
FAILED=()
ALL_SHAPE_CONFIGS=()  # Collect all per-shape config paths for combined config

for shape in "${SHAPE_ARRAY[@]}"; do
    for texture in "${TEXTURE_ARRAY[@]}"; do
        # ------------------------------------------------------------------
        # Step 1: Build the list of available Gaussian outputs for this
        #         shape + texture combination
        # ------------------------------------------------------------------
        TXT_FILE="$GAUSSIAN_SOURCES_DIR/tosca_${shape}_${texture}_all.txt"
        output_paths=()

        for resolution in "${RESOLUTION_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                    candidate="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/decoupled_appearance/$OUTPUT_NAME"
                else
                    candidate="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${light_id}/$OUTPUT_NAME"
                fi
                if [ -d "$candidate/point_cloud" ]; then
                    output_paths+=("$candidate")
                elif [ "$DRY_RUN" = true ]; then
                    # In dry run, include even missing paths so config is generated
                    output_paths+=("$candidate")
                fi
            done
        done

        if [ ${#output_paths[@]} -eq 0 ] && [ "$DRY_RUN" = false ]; then
            echo "Warning: No trained Gaussian outputs found for shape=$shape texture=$texture"
            echo "         Expected under: $SYNTH_DATA_BASE/${texture}_texture/$shape/"
            echo "         Skipping..."
            FAILED+=("$shape/$texture (no Gaussian outputs)")
            continue
        fi

        # Write .txt file listing all output paths
        if [ "$DRY_RUN" = false ]; then
            {
                echo "# Gaussian output paths for TOSCA shape: $shape, texture: $texture"
                if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                    echo "# Resolutions: ${COLMAP_RESOLUTIONS}, lighting: decoupled_appearance"
                else
                    echo "# Resolutions: ${COLMAP_RESOLUTIONS}, light_ids: ${LIGHT_IDS}"
                fi
                for p in "${output_paths[@]}"; do
                    echo "$p"
                done
            } > "$TXT_FILE"
            echo "  [$shape/$texture] Written source list: $TXT_FILE (${#output_paths[@]} outputs)"
        else
            echo "  [DRY RUN] Would write source list: $TXT_FILE"
            echo "    Outputs:"
            for resolution in "${RESOLUTION_ARRAY[@]}"; do
                if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                    echo "      $SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/decoupled_appearance/$OUTPUT_NAME"
                else
                    for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                        echo "      $SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${light_id}/$OUTPUT_NAME"
                    done
                fi
            done
        fi

        # ------------------------------------------------------------------
        # Step 2: Generate YAML config for this shape + texture
        # ------------------------------------------------------------------
        YAML_FILE="$CONFIG_DIR/tosca_${shape}_${texture}.yaml"
        PATCH_OUTPUT_DIR="TrainData/datasets/gaussian_patches/tosca_${shape}_${texture}"

        if [ "$DRY_RUN" = false ]; then
            cat > "$YAML_FILE" <<YAML_EOF
# Auto-generated config for TOSCA shape: ${shape}, texture: ${texture}
# Generated by: scripts/generate_tosca_training_patches.sh
#
# Usage:
#   python DataSets/create_gaussian_training_patches.py --config ${YAML_FILE}

# Dataset class for loading training data
dataset_class: GaussianPatchDataset

# Paths
# gaussian_output points to a .txt file listing all Gaussian output folders
gaussian_output: "${TXT_FILE}"
geodesic_data: null  # Auto-detect from {gaussian_output}/geodesic_distance/gt_geodesic.npz
output_dir: "${PATCH_OUTPUT_DIR}"
tosca_data_root: "${DATA_ROOT}"
shape: "${shape}"
iteration: null      # Use highest available iteration

# Data generation parameters
num_iterations: ${NUM_ITERATIONS:-1000}
num_sources: ${NUM_SOURCES:-4}
num_train_points: ${NUM_TRAIN_POINTS:-50}
seed: ${SEED:-42}

# Neighborhood computation
use_mahalanobis: false
n_neighbors: 10
normalize_per_patch: true

# Ring configuration
rings: [2, 3]

# Ring size mapping
ring_size_mapping:
  euclidean:
    2: 64
    3: 192
    4: 512
  mahalanobis:
    2: 90
    3: 250
    4: 600

# Feature attributes
attributes:
  - xyz

# Normalization and padding
nn_mean: 1
use_r1_min_val: true
mask_constant: -10.0

# Dataset info
dataset:
  name: "TOSCA ${shape} (${texture} texture)"
  type: "TOSCA"
  description: "Training patches for TOSCA shape '${shape}' with ${texture} texture across ${COLMAP_RESOLUTIONS} resolution(s) and ${LIGHT_IDS} light condition(s)"
YAML_EOF
            echo "  [$shape/$texture] Written config: $YAML_FILE"
        else
            echo "  [DRY RUN] Would write config: $YAML_FILE"
        fi

        ALL_SHAPE_CONFIGS+=("$YAML_FILE:$shape:$texture:$PATCH_OUTPUT_DIR")

        # ------------------------------------------------------------------
        # Step 3: Run create_gaussian_training_patches.py
        # ------------------------------------------------------------------
        echo "  [$shape/$texture] Processing..."

        CMD="python $GENERATE_SCRIPT --config $YAML_FILE $OVERRIDES"

        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN] $CMD"
        elif [ "$SEQUENTIAL" = true ]; then
            echo "    Running sequentially..."
            if ! $CMD; then
                echo "    Warning: $shape/$texture generation failed"
                FAILED+=("$shape/$texture")
            fi
        else
            echo "    Running in background..."
            $CMD &
            PIDS+=("$!:$shape/$texture")
        fi

        echo ""
    done
done

# ============================================================================
# Wait for background processes
# ============================================================================
if [ "$DRY_RUN" = false ] && [ "$SEQUENTIAL" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for all shape/texture combinations to complete..."
    echo ""

    for entry in "${PIDS[@]}"; do
        pid="${entry%%:*}"
        name="${entry##*:}"
        if ! wait "$pid"; then
            echo "  Warning: $name generation failed"
            FAILED+=("$name")
        else
            echo "  $name completed successfully"
        fi
    done
fi

# ============================================================================
# Generate combined config (if requested)
# ============================================================================
if [ "$COMBINED_CONFIG" = true ] && [ ${#ALL_SHAPE_CONFIGS[@]} -gt 0 ]; then
    COMBINED_YAML="$CONFIG_DIR/combined_tosca_all.yaml"
    COMBINED_OUTPUT="TrainData/datasets/gaussian_patches/tosca_combined_all"

    echo ""
    echo "------------------------------------------------------------"
    echo "Generating combined TOSCA config: $COMBINED_YAML"
    echo "------------------------------------------------------------"

    if [ "$DRY_RUN" = false ]; then
        {
            echo "# Auto-generated combined config for ALL TOSCA shapes"
            echo "# Generated by: scripts/generate_tosca_training_patches.sh"
            echo "#"
            echo "# Usage:"
            echo "#   from DataSets.gaussian_dataset import CombinedGaussianPatchDataset"
            echo "#   dataset = CombinedGaussianPatchDataset("
            echo "#       config='${COMBINED_YAML}',"
            echo "#       attributes=['xyz'], ring=2"
            echo "#   )"
            echo ""
            echo "dataset_class: CombinedGaussianPatchDataset"
            echo ""
            echo "output_dir: \"${COMBINED_OUTPUT}\""
            echo ""
            echo "data_sources:"

            for entry in "${ALL_SHAPE_CONFIGS[@]}"; do
                yaml_file="${entry%%:*}"
                rest="${entry#*:}"
                shape="${rest%%:*}"
                rest="${rest#*:}"
                texture="${rest%%:*}"
                patch_output_dir="${rest##*:}"
                txt_file="$GAUSSIAN_SOURCES_DIR/tosca_${shape}_${texture}_all.txt"

                echo "  - name: \"tosca_${shape}_${texture}\""
                echo "    gaussian_output: \"${txt_file}\""
                echo "    geodesic_data: null"
                echo "    output_dir: \"${patch_output_dir}\""
                echo "    weight: 1.0"
                echo "    iteration: null"
                echo ""
            done

            echo "# Data generation parameters (shared across all sources)"
            echo "num_iterations: ${NUM_ITERATIONS:-1000}"
            echo "num_sources: ${NUM_SOURCES:-4}"
            echo "num_train_points: ${NUM_TRAIN_POINTS:-50}"
            echo "seed: ${SEED:-42}"
            echo ""
            echo "use_mahalanobis: false"
            echo "n_neighbors: 10"
            echo "normalize_per_patch: true"
            echo ""
            echo "rings: [2, 3]"
            echo ""
            echo "ring_size_mapping:"
            echo "  euclidean:"
            echo "    2: 64"
            echo "    3: 192"
            echo "    4: 512"
            echo ""
            echo "attributes:"
            echo "  - xyz"
            echo ""
            echo "nn_mean: 1"
            echo "use_r1_min_val: true"
            echo "mask_constant: -10.0"
        } > "$COMBINED_YAML"

        echo "  Written combined config: $COMBINED_YAML"
    else
        echo "  [DRY RUN] Would write combined config: $COMBINED_YAML"
    fi
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
echo "  Total time:          ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Shapes processed:    ${#SHAPE_ARRAY[@]}"
echo "  Textures:            ${#TEXTURE_ARRAY[@]}"
echo "  Combinations:        $(( ${#SHAPE_ARRAY[@]} * ${#TEXTURE_ARRAY[@]} ))"
echo "  Failed:              ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed combinations:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi

echo ""
echo "Generated config files:"
for entry in "${ALL_SHAPE_CONFIGS[@]}"; do
    yaml_file="${entry%%:*}"
    rest="${entry#*:}"
    shape="${rest%%:*}"
    rest="${rest#*:}"
    texture="${rest%%:*}"
    patch_output_dir="${rest##*:}"
    echo "  $shape/$texture: $yaml_file"
    echo "    -> patches: $patch_output_dir"
done

if [ "$COMBINED_CONFIG" = true ]; then
    echo ""
    echo "Combined config: $CONFIG_DIR/combined_tosca_all.yaml"
    echo ""
    echo "To load the combined TOSCA dataset:"
    echo "  from DataSets.gaussian_dataset import CombinedGaussianPatchDataset"
    echo "  dataset = CombinedGaussianPatchDataset("
    echo "      config='$CONFIG_DIR/combined_tosca_all.yaml',"
    echo "      attributes=['xyz'], ring=2"
    echo "  )"
fi

echo ""
echo "Done!"
