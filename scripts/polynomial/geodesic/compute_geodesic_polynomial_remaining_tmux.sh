#!/bin/bash
#
# Launch compute_geodesic_polynomial_all.sh for REMAINING geodesic combinations
# inside a tmux session on a SLURM node.
#
# By default recomputes ALL combinations (trusting pane/run-based progress, not
# stale filesystem results). Use --skip_existing to skip any combination that
# already has geodesic_distance/gt_geodesic.npz on disk.
# Use --skip_levels / --skip_light_ids to explicitly exclude finished work.
#
# Based on: compute_geodesic_polynomial_all_tmux.sh
#
# USAGE:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh [options]
#
# OPTIONS:
#   --node NAME              SLURM node (default: gipdeep10)
#   --cpus N                 Number of CPUs to request (default: 70)
#   --time HH:MM:SS          SLURM time limit (default: 72:00:00)
#   --session NAME           tmux session name (default: compute_geo_remaining)
#   --skip_levels  LEVELS    Comma-separated levels to skip (default: 03)
#   --skip_light_ids IDS     Comma-separated light IDs to skip (default: none)
#   --skip_existing          Skip combinations with existing gt_geodesic.npz on disk
#   --dry_run                Print what would be run without executing
#
# EXAMPLES:
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh --skip_existing
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh --skip_levels 02,03
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh --skip_levels "" --skip_light_ids ""
#   bash scripts/polynomial/geodesic/compute_geodesic_polynomial_remaining_tmux.sh --node gipdeep9 --cpus 55

set -e

# ============================================================================
# Defaults  (must match the original compute_geodesic_polynomial_all.sh defaults)
# ============================================================================
NODE="gipdeep10"
CPUS=70
TIME="72:00:00"
SESSION_NAME="compute_geo_remaining"
CONDA_ENV="geo_splat"
DRY_RUN=false
SKIP_EXISTING=false
# Pre-populate with already-finished level 03 (all light IDs completed)
SKIP_LEVELS="03"
SKIP_LIGHT_IDS=""

# Must match defaults in compute_geodesic_polynomial_all.sh
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
TEXTURES="blue"
LEVELS="02 04 03"
LIGHT_IDS="0 1 2 3 4"
SURFACES="Paraboloid Saddle HyperbolicParaboloid"
OUTPUTS="output"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --node)    NODE="$2";         shift 2 ;;
        --cpus)    CPUS="$2";         shift 2 ;;
        --time)    TIME="$2";         shift 2 ;;
        --session)        SESSION_NAME="$2";   shift 2 ;;
        --skip_levels)    SKIP_LEVELS="$2";    shift 2 ;;
        --skip_light_ids) SKIP_LIGHT_IDS="$2"; shift 2 ;;
        --skip_existing)  SKIP_EXISTING=true;   shift   ;;
        --dry_run)        DRY_RUN=true;         shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Resolve project root
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/scripts/polynomial/geodesic/compute_geodesic_polynomial_all.sh"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

# ============================================================================
# Determine which combinations are still missing gt_geodesic.npz
# ============================================================================
REMAINING_SURFACES=()
REMAINING_LEVELS=()
REMAINING_LIGHTS=()

TOTAL=0
DONE=0
MISSING=0

# Build skip lookup sets from comma-separated inputs
declare -A SKIP_LEVEL_SET SKIP_LIGHT_SET
if [ -n "$SKIP_LEVELS" ]; then
    IFS=',' read -ra _arr <<< "$SKIP_LEVELS"
    for v in "${_arr[@]}"; do SKIP_LEVEL_SET["$v"]=1; done
fi
if [ -n "$SKIP_LIGHT_IDS" ]; then
    IFS=',' read -ra _arr <<< "$SKIP_LIGHT_IDS"
    for v in "${_arr[@]}"; do SKIP_LIGHT_SET["$v"]=1; done
fi

echo "============================================================"
echo "Scanning for remaining geodesic computations..."
echo "  skip_levels:    ${SKIP_LEVELS:-<none>}"
echo "  skip_light_ids: ${SKIP_LIGHT_IDS:-<none>}"
echo "  skip_existing:  $SKIP_EXISTING"
echo "============================================================"
echo ""

for texture in $TEXTURES; do
    for level in $LEVELS; do
        for light_id in $LIGHT_IDS; do
            for output_name in $OUTPUTS; do
                for surface in $SURFACES; do
                    TOTAL=$((TOTAL + 1))
                    GEO_FILE="$PROJECT_ROOT/$SYNTH_DATA_BASE/${texture}_texture/${surface}/level_${level}/light_${light_id}/${output_name}/geodesic_distance/gt_geodesic.npz"

                    # Check explicit skip lists
                    if [ -n "${SKIP_LEVEL_SET[$level]+x}" ] || [ -n "${SKIP_LIGHT_SET[$light_id]+x}" ]; then
                        DONE=$((DONE + 1))
                        echo "  [SKIP]    ${texture}/${surface}/level_${level}/light_${light_id}/${output_name}  (skip list)"
                        continue
                    fi

                    if [ "$SKIP_EXISTING" = true ] && [ -f "$GEO_FILE" ]; then
                        DONE=$((DONE + 1))
                        echo "  [SKIP]    ${texture}/${surface}/level_${level}/light_${light_id}/${output_name}  (exists on disk)"
                    else
                        MISSING=$((MISSING + 1))
                        REMAINING_SURFACES+=("$surface")
                        REMAINING_LEVELS+=("$level")
                        REMAINING_LIGHTS+=("$light_id")
                        if [ -f "$GEO_FILE" ]; then
                            echo "  [RERUN]   ${texture}/${surface}/level_${level}/light_${light_id}/${output_name}  (old result exists)"
                        else
                            echo "  [MISSING] ${texture}/${surface}/level_${level}/light_${light_id}/${output_name}"
                        fi
                    fi
                done
            done
        done
    done
done

echo ""
echo "Total:      $TOTAL"
echo "Skipped:    $DONE"
echo "To compute: $MISSING"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "All geodesic computations are already complete. Nothing to do."
    exit 0
fi

# ============================================================================
# Build --surfaces / --levels / --light_ids lists for missing combinations
# (pass as --extra to compute_geodesic_polynomial_all.sh)
# ============================================================================
# Collect unique values that appear in REMAINING_*
declare -A NEED_SURFACES NEED_LEVELS NEED_LIGHTS
for i in "${!REMAINING_SURFACES[@]}"; do
    NEED_SURFACES["${REMAINING_SURFACES[$i]}"]=1
    NEED_LEVELS["${REMAINING_LEVELS[$i]}"]=1
    NEED_LIGHTS["${REMAINING_LIGHTS[$i]}"]=1
done

SURF_LIST=$(IFS=,; s=""; for k in "${!NEED_SURFACES[@]}"; do s="$s,$k"; done; echo "${s#,}")
LEVEL_LIST=$(IFS=,; s=""; for k in "${!NEED_LEVELS[@]}"; do s="$s,$k"; done; echo "${s#,}")
LIGHT_LIST=$(IFS=,; s=""; for k in "${!NEED_LIGHTS[@]}"; do s="$s,$k"; done; echo "${s#,}")

EXTRA="--surfaces $SURF_LIST --levels $LEVEL_LIST --light_ids $LIGHT_LIST"

echo "Passing to compute script:"
echo "  --surfaces  $SURF_LIST"
echo "  --levels    $LEVEL_LIST"
echo "  --light_ids $LIGHT_LIST"
echo ""
echo "Note: The compute script will attempt all combinations of the above."
echo "      Any already-completed entries will be recomputed unless the"
echo "      compute script has its own skip logic."
echo ""

# ============================================================================
# Build srun command
# ============================================================================
INNER_CMD="cd $PROJECT_ROOT && \
source \$(conda info --base)/etc/profile.d/conda.sh && \
conda activate $CONDA_ENV && \
bash $COMPUTE_SCRIPT --verbose $EXTRA"

SRUN_CMD="srun --nodelist=$NODE --cpus-per-task=$CPUS --time=$TIME --pty bash -c '$INNER_CMD'"

# ============================================================================
# Print plan
# ============================================================================
echo "============================================================"
echo "Remaining Geodesic Distances – tmux launcher"
echo "============================================================"
echo "  tmux session:    $SESSION_NAME"
echo "  Node:            $NODE"
echo "  CPUs:            $CPUS"
echo "  Time limit:      $TIME"
echo "  Conda env:       $CONDA_ENV"
echo "  Compute script:  $COMPUTE_SCRIPT"
echo ""
echo "  srun command:"
echo "    $SRUN_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] – nothing executed."
    exit 0
fi

# ============================================================================
# Create / reuse tmux session and launch
# ============================================================================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists – sending command..."
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
else
    echo "Creating tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    sleep 1
    tmux send-keys -t "$SESSION_NAME" "$SRUN_CMD" Enter
fi

echo ""
echo "Launched in tmux session '$SESSION_NAME'."
echo "Attach with:  tmux attach -t $SESSION_NAME"
echo ""
