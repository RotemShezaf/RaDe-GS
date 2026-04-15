#!/bin/bash
#
# Compute geodesic distances for ALL TOSCA shapes
# (blue_texture, high_res, decoupled_appearance, MMP backend + Gaussian embedding).
#
# Wrapper around compute_geodesic_tosca_all.sh with blue_texture defaults.
#
# USAGE:
#   bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh [options]
#
# OPTIONS:
#   --shapes LIST        Comma-separated shapes (default: auto-detect)
#   --num_parallel_outputs N  Number of outputs to run concurrently (default: 4)
#   --num_sources N      FPS source vertices (default: 32)
#   --dry_run            Print commands without executing
#   --sequential         Run sequentially
#
# EXAMPLES:
#   bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh --shapes "cat0,cat1,dog0"
#   bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh --dry_run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHAPES=""
ANIMALS=""
NUM_PARALLEL_OUTPUTS=""
NUM_SOURCES=""
DRY_RUN=""
SEQUENTIAL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --shapes)       SHAPES="--shapes $2";           shift 2 ;;
        --animals)      ANIMALS="--animals $2";          shift 2 ;;
        --n_batches|--num_parallel_outputs)  NUM_PARALLEL_OUTPUTS="--num_parallel_outputs $2"; shift 2 ;;
        --num_sources)  NUM_SOURCES="--num_sources $2"; shift 2 ;;
        --dry_run)      DRY_RUN="--dry_run";            shift   ;;
        --sequential)   SEQUENTIAL="--sequential";      shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CMD="bash $SCRIPT_DIR/compute_geodesic_tosca_all.sh \
    --textures blue \
    --colmap_resolutions high_res \
    --use_decoupled_appearance \
    --outputs output \
    $SHAPES $ANIMALS $NUM_PARALLEL_OUTPUTS $NUM_SOURCES $DRY_RUN $SEQUENTIAL"

echo "============================================================"
echo "TOSCA Geodesic All (blue_texture, decoupled_appearance)"
echo "============================================================"
echo ""
eval "$CMD"
