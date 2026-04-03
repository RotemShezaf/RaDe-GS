#!/bin/bash
#
# mesh_extract_tosca_all.sh
#
# Extract meshes for all trained TOSCA outputs that don't yet have a recon.ply.
# Scans for directories containing point_cloud/ but missing recon.ply and runs
# mesh_extract_tetrahedra.py on each one.
#
# USAGE:
#   bash scripts/tosca/train_gaussians/mesh_extract_tosca_all.sh [options]
#
# OPTIONS:
#   --synth_data_base DIR   Base directory for synthetic COLMAP data
#                           (default: TrainData/TOSCA/SyntheticColmapData)
#   --output_name NAME      Output subdirectory name to look for (default: output)
#   --max_parallel N        Max parallel mesh extraction jobs (default: 1)
#   --gpu_ids LIST          Comma-separated GPU IDs to round-robin (default: auto-detect)
#   --force                 Re-extract even if recon.ply already exists
#   --dry_run               Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/train_gaussians/mesh_extract_tosca_all.sh
#   bash scripts/tosca/train_gaussians/mesh_extract_tosca_all.sh --max_parallel 3
#   bash scripts/tosca/train_gaussians/mesh_extract_tosca_all.sh --force --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
OUTPUT_NAME="output"
MAX_PARALLEL=1
GPU_IDS=""
FORCE=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base) SYNTH_DATA_BASE="$2"; shift 2 ;;
        --output_name)     OUTPUT_NAME="$2";     shift 2 ;;
        --max_parallel)    MAX_PARALLEL="$2";    shift 2 ;;
        --gpu_ids)         GPU_IDS="$2";         shift 2 ;;
        --force)           FORCE=true;           shift   ;;
        --dry_run)         DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve paths
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

MESH_EXTRACT_SCRIPT="$PROJECT_ROOT/mesh_extract_tetrahedra.py"
if [ ! -f "$MESH_EXTRACT_SCRIPT" ]; then
    echo "Error: Mesh extraction script not found: $MESH_EXTRACT_SCRIPT"
    exit 1
fi

# Resolve geo_splat Python interpreter
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

# ============================================================================
# Determine available GPUs
# ============================================================================
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
    GPU_ARRAY=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_ARRAY+=("$i"); done
fi
NUM_GPUS=${#GPU_ARRAY[@]}
echo "Using ${NUM_GPUS} GPU(s): ${GPU_ARRAY[*]}"

# ============================================================================
# Collect targets: output dirs with point_cloud/ but no recon.ply
# ============================================================================
declare -a TARGETS_MODEL=()
declare -a TARGETS_SOURCE=()

echo "Scanning $SYNTH_DATA_BASE for trained outputs..."
for model_path in "$SYNTH_DATA_BASE"/*/*/*/*/"$OUTPUT_NAME" \
                  "$SYNTH_DATA_BASE"/*/*/*/"$OUTPUT_NAME"; do
    [ -d "$model_path/point_cloud" ] || continue
    if [ "$FORCE" = false ] && [ -f "$model_path/recon.ply" ]; then
        continue
    fi
    # source path is the parent of the output directory
    source_path="$(dirname "$model_path")"
    TARGETS_MODEL+=("$model_path")
    TARGETS_SOURCE+=("$source_path")
done

TOTAL=${#TARGETS_MODEL[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No outputs need mesh extraction."
    exit 0
fi
echo "Found $TOTAL output(s) needing mesh extraction."
echo ""

# ============================================================================
# Parallel tracking
# ============================================================================
RUNNING_PIDS=()
RUNNING_JOBS=()
COMPLETED=0
FAILED_JOBS=()
GPU_SLOT=0

wait_for_slot() {
    while [ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]; do
        NEW_PIDS=()
        NEW_JOBS=()
        for i in "${!RUNNING_PIDS[@]}"; do
            pid="${RUNNING_PIDS[$i]}"
            job="${RUNNING_JOBS[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
                NEW_JOBS+=("$job")
            else
                if wait "$pid"; then
                    echo "  [DONE] $job"
                    COMPLETED=$((COMPLETED + 1))
                else
                    echo "  [FAILED] $job"
                    FAILED_JOBS+=("$job")
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_JOBS=("${NEW_JOBS[@]}")
        [ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ] && sleep 5
    done
}

wait_for_all() {
    for i in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[$i]}"
        job="${RUNNING_JOBS[$i]}"
        if wait "$pid"; then
            echo "  [DONE] $job"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  [FAILED] $job"
            FAILED_JOBS+=("$job")
        fi
    done
    RUNNING_PIDS=()
    RUNNING_JOBS=()
}

# ============================================================================
# Main loop
# ============================================================================
START_TIME=$(date +%s)
for idx in "${!TARGETS_MODEL[@]}"; do
    model_path="${TARGETS_MODEL[$idx]}"
    source_path="${TARGETS_SOURCE[$idx]}"
    JOB_NUM=$((idx + 1))
    JOB_NAME="${model_path#$SYNTH_DATA_BASE/}"

    echo "============================================================"
    echo "[$JOB_NUM/$TOTAL] $JOB_NAME"
    echo "  Source: $source_path"
    echo "  Model:  $model_path"
    echo "============================================================"

    # Auto-detect the highest available checkpoint iteration
    ITER=30000
    if [ ! -d "$model_path/point_cloud/iteration_${ITER}" ]; then
        LATEST_ITER=$(ls -d "$model_path/point_cloud/iteration_"* 2>/dev/null \
            | sed 's/.*iteration_//' | sort -n | tail -1)
        if [ -n "$LATEST_ITER" ]; then
            echo "  Note: iteration_30000 not found, using iteration_$LATEST_ITER"
            ITER=$LATEST_ITER
        else
            echo "  Warning: No checkpoint found in $model_path/point_cloud/, skipping."
            echo ""
            continue
        fi
    fi

    ASSIGNED_GPU="${GPU_ARRAY[$((GPU_SLOT % NUM_GPUS))]}"
    CMD="CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU $PYTHON3 $MESH_EXTRACT_SCRIPT -s $source_path -m $model_path --eval --iteration $ITER"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
        echo ""
        continue
    fi

    if [ $MAX_PARALLEL -gt 1 ]; then
        wait_for_slot
        eval $CMD > "$model_path.mesh.log" 2>&1 &
        RUNNING_PIDS+=($!)
        RUNNING_JOBS+=("$JOB_NAME")
        GPU_SLOT=$((GPU_SLOT + 1))
        echo "  Running in background (PID: $!, GPU: $ASSIGNED_GPU)"
    else
        echo "  Extracting mesh (GPU: $ASSIGNED_GPU)..."
        if eval $CMD 2>&1 | tee "$model_path.mesh.log" | tail -5; then
            echo "  [DONE] $JOB_NAME"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  [FAILED] $JOB_NAME"
            FAILED_JOBS+=("$JOB_NAME")
        fi
    fi
    echo ""
done

# Wait for remaining background jobs
if [ $MAX_PARALLEL -gt 1 ]; then
    echo "Waiting for remaining mesh extraction jobs..."
    wait_for_all
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "Mesh Extraction Complete"
echo "============================================================"
echo "  Total time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Total:      $TOTAL"
echo "  Completed:  $COMPLETED"
echo "  Failed:     ${#FAILED_JOBS[@]}"

if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Failed:"
    for f in "${FAILED_JOBS[@]}"; do
        echo "  - $f"
    done
fi

echo ""
echo "Done!"
