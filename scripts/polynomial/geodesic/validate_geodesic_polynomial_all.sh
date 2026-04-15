#!/bin/bash
#
# Validate geodesic distances for all polynomial surfaces and outputs.
#
# Iterates over surfaces, textures, colmap levels, light IDs, and output
# names, running verify_geodesic_distances.py with the --surface flag to
# check projection correctness.
#
# USAGE:
#   ./validate_geodesic_polynomial_all.sh [options]
#
# OPTIONS:
#   --surfaces LIST    Comma-separated (default: Paraboloid,Saddle,HyperbolicParaboloid)
#   --textures LIST    Comma-separated (default: blue)
#   --levels LIST      Comma-separated (default: 02,04,03)
#   --light_ids LIST   Comma-separated (default: 0,1,2,3,4)
#   --outputs LIST     Comma-separated output folder names (default: output)
#   --synth_data_base  Base directory for SyntheticColmapData (default: TrainData/Polynomial/SyntheticColmapData)
#   --verbose          Enable verbose verification output
#

set -e

# ── defaults ─────────────────────────────────────────────────────────────
SYNTH_DATA_BASE="TrainData/Polynomial/SyntheticColmapData"
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
TEXTURES="blue"
LEVELS="02,04,03"
LIGHT_IDS="0,1,2,3,4"
OUTPUTS="output"
VERBOSE=""

# ── parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --synth_data_base) SYNTH_DATA_BASE="$2"; shift 2 ;;
        --surfaces)        SURFACES="$2";        shift 2 ;;
        --textures)        TEXTURES="$2";        shift 2 ;;
        --levels)          LEVELS="$2";          shift 2 ;;
        --light_ids)       LIGHT_IDS="$2";       shift 2 ;;
        --outputs)         OUTPUTS="$2";         shift 2 ;;
        --verbose)         VERBOSE="--verbose";  shift   ;;
        --help|-h)
            echo "Usage: $0 [--surfaces S] [--textures T] [--levels L] [--light_ids I] [--outputs O] [--verbose]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra SURFACE_ARRAY <<< "$SURFACES"
IFS=',' read -ra TEXTURE_ARRAY <<< "$TEXTURES"
IFS=',' read -ra LEVEL_ARRAY   <<< "$LEVELS"
IFS=',' read -ra OUTPUT_ARRAY  <<< "$OUTPUTS"
if [ -z "$LIGHT_IDS" ]; then
    LIGHT_ID_ARRAY=("")
else
    IFS=',' read -ra LIGHT_ID_ARRAY <<< "$LIGHT_IDS"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SINGLE_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/validate_geodesic_single.sh"

if [ ! -f "$SINGLE_SCRIPT" ]; then
    echo "Error: single script not found: $SINGLE_SCRIPT"
    exit 1
fi

# ── iterate ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Geodesic Validation — All Polynomial Surfaces"
echo "============================================================"
echo ""
echo "  Surfaces:   ${SURFACE_ARRAY[*]}"
echo "  Textures:   ${TEXTURE_ARRAY[*]}"
echo "  Levels:     ${LEVEL_ARRAY[*]}"
echo "  Light IDs:  ${LIGHT_ID_ARRAY[*]:-<default>}"
echo "  Outputs:    ${OUTPUT_ARRAY[*]}"
echo ""

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
declare -a FAIL_LIST=()

for texture in "${TEXTURE_ARRAY[@]}"; do
for level in "${LEVEL_ARRAY[@]}"; do
for light_id in "${LIGHT_ID_ARRAY[@]}"; do
for output_name in "${OUTPUT_ARRAY[@]}"; do
for surface in "${SURFACE_ARRAY[@]}"; do
    if [ -n "$light_id" ]; then
        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/light_${light_id}/$output_name"
    else
        GAUSSIAN_OUTPUT="$SYNTH_DATA_BASE/${texture}_texture/$surface/level_${level}/default_light/$output_name"
    fi
    LABEL="$texture/$surface/level_$level/light_${light_id:-default}/$output_name"
    GT="$GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz"

    TOTAL=$((TOTAL + 1))

    if [ ! -f "$GT" ]; then
        echo "  SKIP  $LABEL  (no gt_geodesic.npz)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "────────────────────────────────────────"
    echo "  Validating: $LABEL"

    if bash "$SINGLE_SCRIPT" "$GAUSSIAN_OUTPUT" "$surface" $VERBOSE; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAIL_LIST+=("$LABEL")
    fi
done
done
done
done
done

echo ""
echo "============================================================"
echo "  VALIDATION SUMMARY"
echo "============================================================"
echo "  Total:   $TOTAL"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed outputs:"
    for f in "${FAIL_LIST[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"
