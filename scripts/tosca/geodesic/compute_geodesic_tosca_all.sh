#!/bin/bash
#
# Compute geodesic distances for all TOSCA shapes and Gaussian outputs
#
# This script iterates over:
#   - Multiple textures
#   - Multiple colmap resolutions (high_res, low_res)
#   - Multiple output names (e.g., different training runs)
#   - Multiple TOSCA shapes (auto-detected or explicitly specified)
#
# Outputs are processed concurrently (up to NUM_PARALLEL_OUTPUTS at a time).
# Each output runs a single Python process that handles internal batch
# parallelism via the compute_and_save_geodesic_pipeline.
#
# It assumes the directory structure created by create_synthetic_colmap_dataset_from_mesh_tosca.py
# and that training has been performed with outputs stored in the specified output folder.
#
# NOTE: compute_gaussian_geodesic_distances.py currently supports only polynomial surfaces
# via --surface. For TOSCA support, the Python script needs to be extended with a --shape
# argument that loads ground truth meshes from TrainData/TOSCA/processed/{shape}/mesh_high_res_*.ply.
# This script is designed to work with that extension.
#
# USAGE:
#   ./scripts/compute_geodesic_tosca_all.sh [options]
#
# OPTIONS:
#   --data_root DIR            Preprocessed TOSCA data root (default: TrainData/TOSCA/processed)
#   --synth_data_base DIR      Synthetic COLMAP data base (default: TrainData/TOSCA/SyntheticColmapData)
#   --shapes LIST              Comma-separated shape names (default: auto-detect from data_root)
#   --textures LIST            Comma-separated texture names (default: colors)
#   --colmap_resolutions LIST  Comma-separated COLMAP resolution levels (default: high_res)
#   --outputs LIST             Comma-separated output names (default: output)
#   --light_ids LIST           Comma-separated light IDs (default: 0,1,2,3,4)
#   --n_batches N              (deprecated alias for --num_parallel_outputs)
#   --num_parallel_outputs N   Number of outputs to run concurrently (default: 4)
#   --resolution N             Source mesh resolution (NxN) (default: 8)
#   --use_mahalanobis          Use Mahalanobis distance
#   --dry_run                  Print commands without executing
#   --sequential               Run batches sequentially instead of parallel
#   --verbose                  Enable verbose output
#
# EXAMPLES:
#   # Process all auto-detected shapes with default settings
#   ./scripts/compute_geodesic_tosca_all.sh
#
#   # Process only cat shapes
#   ./scripts/compute_geodesic_tosca_all.sh --shapes "cat0,cat1,cat2"
#
#   # Dry run
#   ./scripts/compute_geodesic_tosca_all.sh --dry_run

set -e

# Load animal→index map and expand_animals() helper
# Edit scripts/tosca/tosca_animal_map.sh to control which shapes are processed.
source "$(dirname "${BASH_SOURCE[0]}")/../tosca_animal_map.sh"

# ============================================================================
# Default values
# ============================================================================
DATA_ROOT="TrainData/TOSCA/processed"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"

NUM_PARALLEL_OUTPUTS=4   # Number of outputs to process concurrently
MAX_PARALLEL=$(($(nproc) - 1))
if [ $MAX_PARALLEL -lt 1 ]; then
    MAX_PARALLEL=1
fi

NUM_SOURCES=64 # Number of source vertices for FPS sampling
SHAPES=""     # Empty = auto-detect from DATA_ROOT
ANIMALS=""    # Animal names without index; expands to all indexed shapes found in DATA_ROOT
TEXTURES="colors"
COLMAP_RESOLUTIONS="high_res"
OUTPUTS="output"
LIGHT_IDS="0,1,2,3,4"
USE_DECOUPLED_APPEARANCE=false  # When true, use 'decoupled_appearance' subdir instead of 'light_{id}'
DRY_RUN=false
SEQUENTIAL=false
USE_MAHALANOBIS=""
VERBOSE="" #--mesh_type
MESH_TYPE="reconstructed"
# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --synth_data_base)
            SYNTH_DATA_BASE="$2"
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
        --outputs)
            OUTPUTS="$2"
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
        --n_jobs)
            N_JOBS_OVERRIDE="$2"
            shift 2
            ;;
        --n_batches|--num_parallel_outputs)
            NUM_PARALLEL_OUTPUTS="$2"
            shift 2
            ;;
        --num_sources)
            NUM_SOURCES="$2"
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
        --use_mahalanobis)
            USE_MAHALANOBIS="--use_mahalanobis"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR            Preprocessed TOSCA data (default: TrainData/TOSCA/processed)"
            echo "  --synth_data_base DIR       Synthetic COLMAP data (default: TrainData/TOSCA/SyntheticColmapData)"
            echo "  --shapes LIST               Comma-separated TOSCA shape names (default: auto-detect)"
            echo "  --textures LIST             Comma-separated textures (default: colors)"
            echo "  --colmap_resolutions LIST   Comma-separated COLMAP resolutions (default: high_res)"
            echo "  --outputs LIST              Comma-separated output folder names (default: output)"
            echo "  --light_ids LIST            Comma-separated light IDs (default: 0,1,2,3,4)"
            echo "  --n_jobs N                  Parallel jobs per output"
            echo "  --num_parallel_outputs N    Number of outputs to run concurrently (default: 4)"
            echo "  --num_sources N             Number of source vertices (default: 64)"
            echo "  --dry_run                   Print commands without executing"
            echo "  --sequential                Run batches sequentially"
            echo "  --use_mahalanobis           Use Mahalanobis distance"
            echo "  --verbose                   Enable verbose output"
            echo ""
            echo "NOTE: Requires compute_gaussian_geodesic_distances.py to support --shape"
            echo "      for TOSCA (extend beyond polynomial --surface argument)."
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

COMPUTE_SCRIPT="$SCRIPT_DIR/GenerateData/compute_gaussian_geodesic_distances_tosca.py"
if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: TOSCA compute script not found: $COMPUTE_SCRIPT"
    exit 1
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
    echo "Auto-detecting TOSCA shapes from $DATA_ROOT ..."
    detected_shapes=()
    if [ -d "$DATA_ROOT" ]; then
        for shape_dir in "$DATA_ROOT"/*/; do
            [ -d "$shape_dir" ] || continue
            shape_name="$(basename "$shape_dir")"
            # Only include if it contains PLY mesh files
            if ls "$shape_dir"/mesh_high_res_*.ply 2>/dev/null | grep -q .; then
                detected_shapes+=("$shape_name")
            fi
        done
    fi
    if [ ${#detected_shapes[@]} -eq 0 ]; then
        echo "Error: No processed TOSCA shapes found in $DATA_ROOT"
        echo "       Please run preprocess_tosca.py first, or specify shapes with --shapes."
        exit 1
    fi
    SHAPES=$(IFS=','; echo "${detected_shapes[*]}")
    echo "  Found ${#detected_shapes[@]} shapes: $SHAPES"
fi

# ============================================================================
# Convert comma-separated lists to arrays
# ============================================================================
IFS=',' read -ra SHAPE_ARRAY <<< "$SHAPES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra RESOLUTION_ARRAY <<< "$COLMAP_RESOLUTIONS"
IFS=',' read -ra OUTPUT_ARRAY <<< "$OUTPUTS"

if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    LIGHT_ID_ARRAY=("__decoupled__")
elif [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

# ============================================================================
# Compute job parameters
# ============================================================================
TOTAL_SOURCES=$NUM_SOURCES
# Calculate N_JOBS (per-output internal parallelism) only if not explicitly set.
# Divide available CPUs evenly among concurrent outputs.
if [ -n "$N_JOBS_OVERRIDE" ]; then
    N_JOBS=$N_JOBS_OVERRIDE
else
    N_JOBS=$(( MAX_PARALLEL / NUM_PARALLEL_OUTPUTS ))
    if [ $N_JOBS -lt 1 ]; then
        N_JOBS=1
    fi
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Geodesic Distance Computation - All TOSCA Shapes"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Data root:         $DATA_ROOT"
echo "  Synth data base:   $SYNTH_DATA_BASE"
echo "  Shapes:            ${SHAPE_ARRAY[*]}"
echo "  Textures:          ${TEXTURE_ARRAY[*]}"
echo "  COLMAP resolutions:${RESOLUTION_ARRAY[*]}"
echo "  Outputs:           ${OUTPUT_ARRAY[*]}"
if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
    echo "  Lighting mode:     decoupled_appearance"
elif [ ${#LIGHT_ID_ARRAY[@]} -gt 0 ] && [ -n "${LIGHT_ID_ARRAY[0]}" ]; then
    echo "  Light IDs:         ${LIGHT_ID_ARRAY[*]}"
else
    echo "  Lighting mode:     default_light"
fi
echo "  Num sources:       $NUM_SOURCES"
echo "  Parallel outputs:  $NUM_PARALLEL_OUTPUTS"
echo "  Jobs per output:   $N_JOBS"
echo "  Sequential:        $SEQUENTIAL"
echo "  Dry run:           $DRY_RUN"
echo ""
echo "NOTE: This script calls compute_gaussian_geodesic_distances.py with"
echo "      --shape (TOSCA shape name) instead of --surface. Ensure the"
echo "      Python script has been extended to support TOSCA shapes."
echo ""

# ============================================================================
# Count total work
# ============================================================================
# ============================================================================
# Pre-scan: build work list
# ============================================================================
declare -a JOB_LABELS=()
declare -a JOB_CMDS=()
declare -a JOB_MERGE_CMDS=()
SKIPPED=0

for shape in "${SHAPE_ARRAY[@]}"; do
    for texture in "${TEXTURE_ARRAY[@]}"; do
        for resolution in "${RESOLUTION_ARRAY[@]}"; do
            for light_id in "${LIGHT_ID_ARRAY[@]}"; do
                for output_name in "${OUTPUT_ARRAY[@]}"; do
                    # Construct paths
                    if [ "$USE_DECOUPLED_APPEARANCE" = true ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/decoupled_appearance/$output_name"
                    elif [ -n "$light_id" ] && [ "$light_id" != "__decoupled__" ]; then
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${light_id}/$output_name"
                    else
                        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/default_light/$output_name"
                    fi

                    LABEL="$texture/$shape/$resolution/light_${light_id:-default}/$output_name"

                    # Validate prerequisites
                    if [ ! -d "$GAUSSIAN_OUTPUT" ] || [ ! -d "$GAUSSIAN_OUTPUT/point_cloud" ]; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                    if ! ls "$DATA_ROOT/$shape"/mesh_high_res_*.ply 2>/dev/null | grep -q .; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --data_root $DATA_ROOT \
                        --shape $shape \
                        --mesh_type $MESH_TYPE \
                        --num_sources $NUM_SOURCES \
                        --geodesic_method mmp \
                        --embed_gaussians \
                        --n_jobs $N_JOBS \
                        $USE_MAHALANOBIS \
                        $VERBOSE"

                    MERGE_CMD="python $COMPUTE_SCRIPT \
                        --gaussian_output $GAUSSIAN_OUTPUT \
                        --merge_only \
                        $VERBOSE"

                    JOB_LABELS+=("$LABEL")
                    JOB_CMDS+=("$CMD")
                    JOB_MERGE_CMDS+=("$MERGE_CMD")
                done
            done
        done
    done
done

TOTAL=${#JOB_LABELS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to do — all outputs skipped or not found."
    exit 0
fi

echo "  Total outputs:   $TOTAL  (skipped: $SKIPPED)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "============================================================"
    echo "DRY RUN — commands that would be executed:"
    echo "============================================================"
    for i in "${!JOB_LABELS[@]}"; do
        echo ""
        echo "  [$(( i + 1 ))/$TOTAL] ${JOB_LABELS[$i]}"
        echo "  ${JOB_CMDS[$i]}"
        echo "  merge: ${JOB_MERGE_CMDS[$i]}"
    done
    exit 0
fi

# ============================================================================
# Main processing — parallelize across outputs
# ============================================================================
START_TIME=$(date +%s)
CURRENT_OUTPUT=0
FAILED_OUTPUTS=()

# Pool state
declare -a _PIDS=()
declare -a _LABELS=()
declare -a _MERGE_CMDS=()

_reap_one() {
    # Wait for any one background job to finish
    wait -n 2>/dev/null || true
    local new_pids=() new_labels=() new_merges=()
    local reaped=0
    for i in "${!_PIDS[@]}"; do
        local pid="${_PIDS[$i]}"
        if [ "$reaped" -eq 0 ] && ! kill -0 "$pid" 2>/dev/null; then
            local rc=0
            wait "$pid" 2>/dev/null || rc=$?
            local label="${_LABELS[$i]}"
            local merge_cmd="${_MERGE_CMDS[$i]}"
            if [ "$rc" -eq 0 ]; then
                echo "  ✓ $label — compute done, merging ..."
                if ! eval "$merge_cmd"; then
                    echo "  ✗ $label — merge FAILED"
                    FAILED_OUTPUTS+=("$label (merge failed)")
                fi
            else
                echo "  ✗ $label — compute FAILED (exit $rc)"
                FAILED_OUTPUTS+=("$label (compute failed)")
            fi
            reaped=1
        else
            new_pids+=("$pid")
            new_labels+=("${_LABELS[$i]}")
            new_merges+=("${_MERGE_CMDS[$i]}")
        fi
    done
    _PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    _LABELS=("${new_labels[@]+"${new_labels[@]}"}")
    _MERGE_CMDS=("${new_merges[@]+"${new_merges[@]}"}")
}

_throttle() {
    while [ "${#_PIDS[@]}" -ge "$NUM_PARALLEL_OUTPUTS" ]; do
        _reap_one
    done
}

_drain() {
    while [ "${#_PIDS[@]}" -gt 0 ]; do
        _reap_one
    done
}

_cleanup() {
    for pid in "${_PIDS[@]+"${_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap _cleanup EXIT INT TERM

echo "============================================================"
echo "Dispatching $TOTAL outputs ($NUM_PARALLEL_OUTPUTS concurrent)"
echo "============================================================"
echo ""

for i in "${!JOB_LABELS[@]}"; do
    _throttle
    label="${JOB_LABELS[$i]}"
    cmd="${JOB_CMDS[$i]}"

    CURRENT_OUTPUT=$(( i + 1 ))
    if [ "$SEQUENTIAL" = true ]; then
        echo "  → [$CURRENT_OUTPUT/$TOTAL] $label"
        if eval "$cmd"; then
            echo "  ✓ $label — compute done, merging ..."
            if ! eval "${JOB_MERGE_CMDS[$i]}"; then
                echo "  ✗ $label — merge FAILED"
                FAILED_OUTPUTS+=("$label (merge failed)")
            fi
        else
            echo "  ✗ $label — compute FAILED"
            FAILED_OUTPUTS+=("$label (compute failed)")
        fi
    else
        echo "  → queued [$CURRENT_OUTPUT/$TOTAL] $label"
        eval "$cmd" &
        _PIDS+=("$!")
        _LABELS+=("$label")
        _MERGE_CMDS+=("${JOB_MERGE_CMDS[$i]}")
    fi
done

if [ "$SEQUENTIAL" = false ]; then
    echo ""
    echo "All jobs queued — waiting for completion ..."
    _drain
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "All Processing Complete"
echo "============================================================"
echo "  Total time:              ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Total outputs processed: $TOTAL"
echo "  Skipped:                 $SKIPPED"
echo "  Failed outputs:          ${#FAILED_OUTPUTS[@]}"

if [ ${#FAILED_OUTPUTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed outputs:"
    for failed in "${FAILED_OUTPUTS[@]}"; do
        echo "  - $failed"
    done
fi

echo ""
echo "Done!"
