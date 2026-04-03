#!/bin/bash
#
# Render synthetic COLMAP datasets for ALL TOSCA shapes
# (blue_texture, high_res, decoupled_appearance by default).
#
# Thin wrapper around render_all_tosca.sh with blue_texture defaults.
#
# USAGE:
#   bash scripts/tosca/render/render_all_blue.sh [options]
#
# OPTIONS:
#   --shapes LIST        Comma-separated shapes with index (default: all)
#   --animals LIST       Animal names WITHOUT index (e.g. "cat,dog"); expands via data_root
#   --max_parallel N     Parallel rendering jobs (default: nproc-1)
#   --num_views N        Camera views (default: 400)
#   --dry_run            Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/render/render_all_blue.sh --shapes "cat0,cat1" --dry_run
#   bash scripts/tosca/render/render_all_blue.sh --all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHAPES=""
ANIMALS=""
ALL_SHAPES=""
MAX_PARALLEL=""
NUM_VIEWS=600
DRY_RUN=""
USE_DECOUPLED_APPEARANCE=false  # When true, passes --use_decoupled_appearance and ignores LIGHT_IDS
while [[ $# -gt 0 ]]; do
    case $1 in
        --shapes)       SHAPES="$2";                    shift 2 ;;
        --animals)      ANIMALS="$2";                   shift 2 ;;
        --all)          ALL_SHAPES="--all_shapes";       shift   ;;
        --max_parallel) MAX_PARALLEL="--max_parallel $2"; shift 2 ;;
        --num_views)    NUM_VIEWS="$2";                  shift 2 ;;
        --dry_run)      DRY_RUN="--dry_run";             shift   ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CMD="bash $SCRIPT_DIR/render_all_tosca.sh \
    --textures blue \
    --colmap_resolutions high_res \
    --image_mesh_resolution  high_res \
    --num_views $NUM_VIEWS \
    --image_width 1024 \
    --image_height 1024 \
    --auto_camera_radius \
    $ALL_SHAPES $MAX_PARALLEL $DRY_RUN"

if [ -n "$ANIMALS" ]; then
    CMD="$CMD --animals $ANIMALS"
elif [ -n "$SHAPES" ]; then
    CMD="$CMD --shapes $SHAPES"
else
    CMD="$CMD --all_shapes"
fi

echo "============================================================"
echo "TOSCA Render All (blue_texture, decoupled_appearance)"
echo "============================================================"
echo ""
eval "$CMD"
