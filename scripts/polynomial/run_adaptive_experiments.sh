#!/usr/bin/env bash
# Run adaptive analysis experiments with different (n_neighbors, target_neighbors) combos.
# adaptive_max_steps = target_neighbors - n_neighbors for each run.
#
# Results are saved to DataSets/adaptive_experiments/<run_name>.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/DataSets/adaptive_experiments"
mkdir -p "$OUT_DIR"

# Parameter grid: (n_neighbors, target_neighbors)
COMBOS=(
    "10 14"
    "10 16"
    "12 16"
    "12 18"
    "14 18"
    "14 20"
)

for combo in "${COMBOS[@]}"; do
    read -r N_NBR TGT <<< "$combo"
    MAX_STEPS=$(( TGT - N_NBR ))
    RUN_NAME="n${N_NBR}_t${TGT}_s${MAX_STEPS}"
    LOG="$OUT_DIR/${RUN_NAME}.log"

    echo "============================================================"
    echo "  Experiment: n_neighbors=$N_NBR  target=$TGT  max_steps=$MAX_STEPS"
    echo "  Log: $LOG"
    echo "============================================================"

    bash "$SCRIPT_DIR/run_adaptive_analysis.sh" \
        --node gipdeep10 --cpus 75 \
        --n_neighbors "$N_NBR" \
        --neighbors "$TGT" \
        --dry_run 2>&1 | head -20

    echo ""
    echo "Launching..."
    bash "$SCRIPT_DIR/run_adaptive_analysis.sh" \
        --node gipdeep10 --cpus 75 \
        --n_neighbors "$N_NBR" \
        --neighbors "$TGT" 2>&1 | tee "$LOG"

    echo ""
    echo "Done: $RUN_NAME"
    echo ""
done

echo "============================================================"
echo "All experiments complete. Logs in: $OUT_DIR"
echo "============================================================"
