#!/bin/bash
#
# Generate training patches for ALL TOSCA shapes using pre-created configs.
#
# Iterates over all per-animal configs in DataSets/configs/tosca/ and runs
# create_gaussian_training_patches.py for each.
#
# USAGE:
#   bash scripts/tosca/patches/generate_training_patches.sh [options]
#
# OPTIONS:
#   --animals LIST       Comma-separated animal names (default: all 12)
#   --scale_opacity      Use scale_opacity configs instead of xyz-only
#   --max_parallel N     Parallel generation jobs (default: 4)
#   --num_output_workers N  KNN parallel workers (default: 5)
#   --dry_run            Print commands without executing
#   --sequential         Run one at a time
#
# EXAMPLES:
#   bash scripts/tosca/patches/generate_training_patches.sh
#   bash scripts/tosca/patches/generate_training_patches.sh --animals "cat,dog" --scale_opacity
#   bash scripts/tosca/patches/generate_training_patches.sh --dry_run

set -e

# ============================================================================
# Defaults
# ============================================================================
ALL_ANIMALS="cat centaur david dog gorilla horse michael victoria wolf"
ANIMALS=""
SCALE_OPACITY=false
MAX_PARALLEL=$(( $(nproc) - 1 ))
NUM_OUTPUT_WORKERS=5
DRY_RUN=false
SEQUENTIAL=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --animals)           ANIMALS="$2";            shift 2 ;;
        --scale_opacity)     SCALE_OPACITY=true;      shift   ;;
        --max_parallel)      MAX_PARALLEL="$2";       shift 2 ;;
        --num_output_workers) NUM_OUTPUT_WORKERS="$2"; shift 2 ;;
        --dry_run)           DRY_RUN=true;            shift   ;;
        --sequential)        SEQUENTIAL=true;         shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_DIR="DataSets/configs/tosca"
GENERATE_SCRIPT="DataSets/create_gaussian_training_patches.py"

if [ ! -f "$GENERATE_SCRIPT" ]; then
    # Fall back to create_tosca_training_patches.py
    GENERATE_SCRIPT="DataSets/create_tosca_training_patches.py"
fi

if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: Generation script not found."
    echo "       Expected: DataSets/create_gaussian_training_patches.py"
    echo "              or DataSets/create_tosca_training_patches.py"
    exit 1
fi

# Build animal list
if [ -n "$ANIMALS" ]; then
    IFS=',' read -ra ANIMAL_ARRAY <<< "$ANIMALS"
else
    ANIMAL_ARRAY=($ALL_ANIMALS)
fi

# Config suffix
SUFFIX=""
if [ "$SCALE_OPACITY" = true ]; then
    SUFFIX="_scale_opacity"
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================"
echo "Generate TOSCA Training Patches"
echo "============================================================"
echo ""
echo "  Animals:           ${ANIMAL_ARRAY[*]}"
echo "  Scale+Opacity:     $SCALE_OPACITY"
echo "  Config dir:        $CONFIG_DIR"
echo "  Max parallel:      $MAX_PARALLEL"
echo "  Output workers:    $NUM_OUTPUT_WORKERS"
echo "  Dry run:           $DRY_RUN"
echo ""

# ============================================================================
# Main loop
# ============================================================================
TOTAL=${#ANIMAL_ARRAY[@]}
CURRENT=0
PIDS=()
FAILED=()
SKIPPED=0

for animal in "${ANIMAL_ARRAY[@]}"; do
    CURRENT=$((CURRENT + 1))
    CONFIG="$CONFIG_DIR/tosca_${animal}${SUFFIX}.yaml"

    echo "[$CURRENT/$TOTAL] $animal  ->  $CONFIG"

    if [ ! -f "$CONFIG" ]; then
        echo "  Warning: Config not found, skipping: $CONFIG"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    CMD="python $GENERATE_SCRIPT --config $CONFIG --num_output_workers $NUM_OUTPUT_WORKERS"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
        continue
    fi

    if [ "$SEQUENTIAL" = true ] || [ "$MAX_PARALLEL" -le 1 ]; then
        if eval "$CMD"; then
            echo "  [DONE] $animal"
        else
            echo "  [FAILED] $animal"
            FAILED+=("$animal")
        fi
    else
        eval "$CMD" &
        PIDS+=($!)

        # Throttle
        while [ ${#PIDS[@]} -ge "$MAX_PARALLEL" ]; do
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                else
                    wait "$pid" 2>/dev/null || FAILED+=("pid:$pid")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            [ ${#PIDS[@]} -ge "$MAX_PARALLEL" ] && sleep 2
        done
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAILED+=("pid:$pid")
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "Training Patch Generation Complete"
echo "============================================================"
echo "  Total:    $TOTAL"
echo "  Skipped:  $SKIPPED"
echo "  Failed:   ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo ""
