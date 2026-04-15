#!/bin/bash
#
# Autotune geodesic mesh for a SINGLE Gaussian output.
#
# Tries 6 parameter configurations in parallel, evaluates mesh quality,
# selects the best one, and saves all variants as backups for later switching.
#
# This is the autotune version of build_geodesic_mesh.sh.
#
# USAGE:
#   ./build_geodesic_mesh_autotune.sh --gaussian_output <path> --surface <type> [options]
#
# EXAMPLES:
#   # Autotune (try 6 configs, pick best):
#   ./build_geodesic_mesh_autotune.sh \
#       --gaussian_output TrainData/.../Paraboloid/level_02/light_0/output \
#       --surface Paraboloid
#
#   # With custom metric:
#   ./build_geodesic_mesh_autotune.sh \
#       --gaussian_output output/HyperbolicParaboloid \
#       --surface HyperbolicParaboloid \
#       --metric composite
#
#   # List available backups:
#   ./build_geodesic_mesh_autotune.sh \
#       --gaussian_output output/Paraboloid \
#       --list_backups
#
#   # Switch active mesh to a backup:
#   ./build_geodesic_mesh_autotune.sh \
#       --gaussian_output output/Paraboloid \
#       --switch a2.00_e0.006_light

set -e

# ============================================================================
# Defaults
# ============================================================================
GAUSSIAN_OUTPUT=""
SURFACE=""
WORKERS="6"
METRIC="rank"
LIST_BACKUPS=false
SWITCH=""
DRY_RUN=false
SKIP_EXISTING=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gaussian_output) GAUSSIAN_OUTPUT="$2"; shift 2 ;;
        --surface)         SURFACE="$2";         shift 2 ;;
        --workers)         WORKERS="$2";         shift 2 ;;
        --metric)          METRIC="$2";          shift 2 ;;
        --list_backups)    LIST_BACKUPS=true;     shift   ;;
        --switch)          SWITCH="$2";          shift 2 ;;
        --dry_run)         DRY_RUN=true;         shift   ;;
        --skip_existing)   SKIP_EXISTING=true;   shift   ;;
        --help|-h)
            echo "Usage: $0 --gaussian_output <path> --surface <type> [options]"
            echo ""
            echo "Required (for build mode):"
            echo "  --gaussian_output PATH   Gaussian splatting output folder"
            echo "  --surface TYPE           Paraboloid | Saddle | HyperbolicParaboloid"
            echo ""
            echo "Optional:"
            echo "  --workers N              Parallel workers (default: 6 = one per config)"
            echo "  --metric d3|composite    Selection metric (default: d3)"
            echo "  --skip_existing          Skip if already autotuned"
            echo "  --dry_run                Print configs without building"
            echo ""
            echo "Backup management:"
            echo "  --list_backups           List available backups"
            echo "  --switch CONFIG          Switch active mesh to named backup"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Validate
# ============================================================================
if [ -z "$GAUSSIAN_OUTPUT" ]; then
    echo "Error: --gaussian_output is required"
    exit 1
fi

if [ "$LIST_BACKUPS" = false ] && [ -z "$SWITCH" ] && [ -z "$SURFACE" ]; then
    echo "Error: --surface is required for build mode"
    exit 1
fi

# ============================================================================
# Locate Python autotune script
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AUTOTUNE_SCRIPT="$PROJECT_ROOT/GenerateData/autotune_geodesic_mesh.py"

if [ ! -f "$AUTOTUNE_SCRIPT" ]; then
    echo "Error: Script not found: $AUTOTUNE_SCRIPT"
    exit 1
fi

# ============================================================================
# Build command
# ============================================================================
CMD="python $AUTOTUNE_SCRIPT --output_dir $GAUSSIAN_OUTPUT"

if [ "$LIST_BACKUPS" = true ]; then
    CMD="$CMD --list_backups"
elif [ -n "$SWITCH" ]; then
    CMD="$CMD --switch $SWITCH"
else
    CMD="$CMD --surface $SURFACE --workers $WORKERS --metric $METRIC"
    if [ "$DRY_RUN" = true ]; then
        CMD="$CMD --dry_run"
    fi
    if [ "$SKIP_EXISTING" = true ]; then
        CMD="$CMD --skip_existing"
    fi
fi

# ============================================================================
# Run
# ============================================================================
echo "============================================================"
echo "Autotune Geodesic Mesh"
echo "============================================================"
echo "  Output:  $GAUSSIAN_OUTPUT"
echo "  Surface: ${SURFACE:-N/A}"
echo "  Workers: $WORKERS"
echo "  Metric:  $METRIC"
echo "============================================================"
echo ""

$CMD

echo ""
echo "Done!"
