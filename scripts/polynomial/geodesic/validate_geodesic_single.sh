#!/bin/bash
#
# Validate geodesic distances for a single polynomial Gaussian output.
#
# USAGE:
#   ./validate_geodesic_single.sh <gaussian_output> <surface_type> [--verbose]
#
# EXAMPLE:
#   ./validate_geodesic_single.sh \
#       TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_02/light_0/output \
#       Paraboloid --verbose
#

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <gaussian_output> <surface_type> [--verbose]"
    echo "  surface_type: Paraboloid | Saddle | HyperbolicParaboloid"
    exit 1
fi

GAUSSIAN_OUTPUT="$1"
SURFACE="$2"
VERBOSE="${3:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERIFY_SCRIPT="$PROJECT_ROOT/GenerateData/verify_geodesic_distances.py"
PYTHON="${PYTHON:-$(which python3 2>/dev/null || which python 2>/dev/null)}"

if [ ! -f "$VERIFY_SCRIPT" ]; then
    echo "Error: verify script not found: $VERIFY_SCRIPT"
    exit 1
fi

GT="$GAUSSIAN_OUTPUT/geodesic_distance/gt_geodesic.npz"
if [ ! -f "$GT" ]; then
    echo "SKIP: no gt_geodesic.npz in $GAUSSIAN_OUTPUT"
    exit 0
fi

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$VERIFY_SCRIPT" \
    --gaussian_output "$GAUSSIAN_OUTPUT" \
    --surface "$SURFACE" \
    $VERBOSE
