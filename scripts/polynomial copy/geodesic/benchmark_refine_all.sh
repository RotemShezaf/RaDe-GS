#!/bin/bash
#
# Run the benchmark_refine.py sweep on ALL polynomial Gaussian outputs.
#
# Discovers outputs using the same directory structure as
# build_geodesic_mesh_polynomial_all.sh and benchmarks several
# max_edge_length values to find parameters giving ~700K-1M vertices.
#
# USAGE:
#   bash scripts/polynomial/geodesic/benchmark_refine_all.sh [options]
#
# OPTIONS:
#   --surfaces LIST       Comma-separated surfaces (default: Paraboloid,Saddle,HyperbolicParaboloid)
#   --levels LIST         Comma-separated levels (default: 02,03,04)
#   --light_ids LIST      Comma-separated light IDs (default: 0,1,2,3,4)
#   --edge_lengths LIST   Comma-separated edge lengths (default: 0.005,0.006,0.007,0.008,0.010)
#   --ring_steps LIST     Comma-separated ring_fix_iterations values (default: 1,2)
#   --workers N           Parallel workers (default: nproc)
#   --output_dir DIR      Directory for JSON report (default: script dir)
#   --dry_run             Print command without executing

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
SURFACES="Paraboloid,Saddle,HyperbolicParaboloid"
LEVELS="02,03,04"
LIGHT_IDS="0,1,2,3,4"
EDGE_LENGTHS="0.005,0.006,0.007,0.008,0.010"
RING_STEPS="1,2"
WORKERS=""
OUTPUT_DIR=""
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --surfaces)      SURFACES="$2";      shift 2 ;;
        --levels)        LEVELS="$2";        shift 2 ;;
        --light_ids)     LIGHT_IDS="$2";     shift 2 ;;
        --edge_lengths)  EDGE_LENGTHS="$2";  shift 2 ;;
        --ring_steps)    RING_STEPS="$2";    shift 2 ;;
        --workers)       WORKERS="$2";       shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --dry_run)       DRY_RUN=true;       shift   ;;
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/scripts/polynomial/geodesic/benchmark_refine.py"

if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "Error: Script not found: $BENCHMARK_SCRIPT"
    exit 1
fi

# Auto-detect CPU count
if [ -z "$WORKERS" ]; then
    WORKERS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# Default output_dir
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$SCRIPT_DIR/scripts/polynomial/geodesic"
fi

# ============================================================================
# Build command
# ============================================================================
CMD="python $BENCHMARK_SCRIPT \
    --surfaces $SURFACES \
    --levels $LEVELS \
    --light_ids $LIGHT_IDS \
    --edge_lengths $EDGE_LENGTHS \
    --ring_steps $RING_STEPS \
    --workers $WORKERS \
    --output_dir $OUTPUT_DIR"

# ============================================================================
# Print plan and run
# ============================================================================
echo "============================================================"
echo "Benchmark: Real Gaussian Mesh Building"
echo "============================================================"
echo "  Surfaces:     $SURFACES"
echo "  Levels:       $LEVELS"
echo "  Light IDs:    $LIGHT_IDS"
echo "  Edge lengths: $EDGE_LENGTHS"
echo "  Ring steps:   $RING_STEPS"
echo "  Workers:      $WORKERS"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Script:       $BENCHMARK_SCRIPT"
echo ""
echo "  Command:"
echo "    $CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] -- nothing executed."
    exit 0
fi

eval "$CMD"
