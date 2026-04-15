#!/usr/bin/env bash
# ============================================================================
# Run all missing benchmark evaluations on gipdeep9 via srun + parallel procs
# Single srun allocation, 90 parallel eval processes inside
# ============================================================================
set -euo pipefail

PROJECT_ROOT="/home/rotem.shezaf/RaDe-GS"
PYTHON3="/home/rotem.shezaf/miniconda3/envs/geo_splat/bin/python3"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/tosca/benchmark/evaluate_benchmark.py"
GT_ROOT="$PROJECT_ROOT/TrainData/TOSCA/processed"
DATA_ROOT="$PROJECT_ROOT/TrainData/TOSCA/SyntheticColmapData/blue_texture"
ITERATIONS=45000
MAX_PARALLEL=4
LOGDIR="$PROJECT_ROOT/logs/benchmark_eval_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOGDIR"

# Build job list file: sweep_dir|gt_mesh
JOBFILE="$LOGDIR/jobs.txt"
find "$DATA_ROOT" -maxdepth 5 -type d -name 'sweep_*' | sort | while read d; do
    if ! find "$d" \( -name '*.ply' -o -name 'cfg_args' \) 2>/dev/null | head -1 | grep -q .; then
        continue
    fi
    if [ -f "$d/benchmark_report.json" ]; then
        continue
    fi
    shape=$(echo "$d" | sed 's|.*/blue_texture/||;s|/high_res.*||')
    gt_mesh=$(find "$GT_ROOT/$shape" -name "mesh_high_res_*.ply" 2>/dev/null | sort | head -1)
    if [ -z "$gt_mesh" ]; then
        echo "WARNING: No GT mesh for $shape, skipping $d" >&2
        continue
    fi
    echo "$d|$gt_mesh"
done > "$JOBFILE"

TOTAL=$(wc -l < "$JOBFILE")
echo "============================================================"
echo "Benchmark Evaluation: $TOTAL jobs, max $MAX_PARALLEL parallel"
echo "  Log dir: $LOGDIR"
echo "  Job file: $JOBFILE"
echo "============================================================"

if [ "$TOTAL" -eq 0 ]; then
    echo "No sweeps need benchmarking. All done!"
    exit 0
fi

START_TIME=$(date +%s)
COMPLETED=0
FAILED=0
RUNNING_PIDS=()
RUNNING_NAMES=()

wait_for_slot() {
    while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; do
        local NEW_PIDS=() NEW_NAMES=()
        for i in "${!RUNNING_PIDS[@]}"; do
            if kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
                NEW_PIDS+=("${RUNNING_PIDS[$i]}")
                NEW_NAMES+=("${RUNNING_NAMES[$i]}")
            else
                if wait "${RUNNING_PIDS[$i]}" 2>/dev/null; then
                    COMPLETED=$((COMPLETED + 1))
                    echo "  [DONE $COMPLETED/$TOTAL] ${RUNNING_NAMES[$i]}"
                else
                    FAILED=$((FAILED + 1))
                    echo "  [FAIL] ${RUNNING_NAMES[$i]}"
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_NAMES=("${NEW_NAMES[@]}")
        if [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; then
            sleep 2
        fi
    done
}

while IFS='|' read -r sweep_dir gt_mesh; do
    short_name=$(echo "$sweep_dir" | sed 's|.*/blue_texture/||;s|/high_res/light_0/sweep_|/|')
    log_file="$LOGDIR/${short_name//\//_}.log"

    wait_for_slot

    (
        CUDA_VISIBLE_DEVICES="" "$PYTHON3" "$EVAL_SCRIPT" \
            --output_dir "$sweep_dir" \
            --gt_mesh "$gt_mesh" \
            --iteration "$ITERATIONS" \
        > "$log_file" 2>&1
    ) &
    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$short_name")
done < "$JOBFILE"

# Wait for remaining
for i in "${!RUNNING_PIDS[@]}"; do
    if wait "${RUNNING_PIDS[$i]}" 2>/dev/null; then
        COMPLETED=$((COMPLETED + 1))
        echo "  [DONE $COMPLETED/$TOTAL] ${RUNNING_NAMES[$i]}"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] ${RUNNING_NAMES[$i]}"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
echo "Benchmark Evaluation Complete"
echo "============================================================"
echo "  Total time:   ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
echo "  Completed:    $COMPLETED / $TOTAL"
echo "  Failed:       $FAILED"
echo "  Logs:         $LOGDIR"
echo "============================================================"
