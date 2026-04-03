#!/bin/bash
#
# Build a geodesic mesh for a SINGLE Gaussian output.
#
# This script wraps compute_geodesic_mesh_for_gaussians.py for one
# specific Gaussian output folder / surface pair.
#
# USAGE:
#   ./build_geodesic_mesh.sh --gaussian_output <path> --surface <type> [options]
#
# EXAMPLES:
#   # Minimal (Gaussians only, no extra samples):
#   ./build_geodesic_mesh.sh \
#       --gaussian_output TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_02/light_0/output \
#       --surface Paraboloid
#
#   # With 500 extra uniform samples:
#   ./build_geodesic_mesh.sh \
#       --gaussian_output TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_03/light_1/output \
#       --surface Saddle \
#       --n_points 500
#
#   # With minimum spacing:
#   ./build_geodesic_mesh.sh \
#       --gaussian_output output/polynomial/HyperbolicParaboloid \
#       --surface HyperbolicParaboloid \
#       --min_radius 0.05 \
#       --x_range "-1.0 1.0" --y_range "-1.0 1.0"

set -e

# ============================================================================
# Defaults
# ============================================================================
GAUSSIAN_OUTPUT=""
SURFACE=""
N_POINTS=""
MIN_RADIUS=""
X_RANGE=""
Y_RANGE=""
ITERATION=""
SEED="42"
VERBOSE=""
REFINE="--refine"
REFINE_ITERATIONS=""
REFINE_RING_FIX=true
REFINE_DELAUNAY_FLIP=true
REFINE_SURFACE_AWARE=false
REFINE_PATIENCE="3"
MAX_EDGE_LENGTH="0.05"
MESH_METHOD="grid"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gaussian_output)
            GAUSSIAN_OUTPUT="$2"
            shift 2
            ;;
        --surface)
            SURFACE="$2"
            shift 2
            ;;
        --n_points)
            N_POINTS="$2"
            shift 2
            ;;
        --min_radius)
            MIN_RADIUS="$2"
            shift 2
            ;;
        --x_range)
            X_RANGE="$2"
            shift 2
            ;;
        --y_range)
            Y_RANGE="$2"
            shift 2
            ;;
        --iteration)
            ITERATION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --mesh_method)
            MESH_METHOD="$2"
            shift 2
            ;;
        --refine_iterations)
            REFINE_ITERATIONS="$2"
            shift 2
            ;;
        --refine_ring_fix)
            REFINE_RING_FIX=true
            shift
            ;;
        --refine_delaunay_flip)
            REFINE_DELAUNAY_FLIP=true
            shift
            ;;
        --refine_surface_aware)
            REFINE_SURFACE_AWARE=true
            shift
            ;;
        --refine_patience)
            REFINE_PATIENCE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --gaussian_output <path> --surface <type> [options]"
            echo ""
            echo "Required:"
            echo "  --gaussian_output PATH   Gaussian splatting output folder"
            echo "  --surface TYPE           Paraboloid | Saddle | HyperbolicParaboloid"
            echo ""
            echo "Optional (mutually exclusive sampling):"
            echo "  --n_points N             Add N uniform surface samples"
            echo "  --min_radius R           Minimum (x,y) spacing for dart-throwing"
            echo ""
            echo "Optional:"
            echo "  --x_range 'XMIN XMAX'   Domain bounds in x (auto-detected if omitted)"
            echo "  --y_range 'YMIN YMAX'   Domain bounds in y (auto-detected if omitted)"
            echo "  --iteration N            Training iteration (default: highest available)"
            echo "  --seed N                 RNG seed (default: 42)"
            echo "  --verbose                Enable verbose output"
            echo "  --mesh_method METHOD     delaunay | ball_pivoting | grid (default: delaunay)"
            echo "  --refine_iterations N    Maximum refinement passes (default: 20)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate
# ============================================================================
if [ -z "$GAUSSIAN_OUTPUT" ]; then
    echo "Error: --gaussian_output is required"
    exit 1
fi

if [ -z "$SURFACE" ]; then
    echo "Error: --surface is required"
    exit 1
fi

if [ ! -d "$GAUSSIAN_OUTPUT" ]; then
    echo "Error: Gaussian output directory not found: $GAUSSIAN_OUTPUT"
    exit 1
fi

# ============================================================================
# Build command
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPUTE_SCRIPT="$PROJECT_ROOT/GenerateData/compute_geodesic_mesh_for_gaussians.py"

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Script not found: $COMPUTE_SCRIPT"
    exit 1
fi

CMD="python $COMPUTE_SCRIPT \
    --gaussian_output $GAUSSIAN_OUTPUT \
    --surface $SURFACE \
    --max_edge_length $MAX_EDGE_LENGTH \
    --mesh_method $MESH_METHOD \
    --seed $SEED \
    $REFINE"

if [ -n "$N_POINTS" ]; then
    CMD="$CMD --n_points $N_POINTS"
fi

if [ -n "$MIN_RADIUS" ]; then
    CMD="$CMD --min_radius $MIN_RADIUS"
fi

if [ -n "$X_RANGE" ]; then
    CMD="$CMD --x_range $X_RANGE"
fi

if [ -n "$Y_RANGE" ]; then
    CMD="$CMD --y_range $Y_RANGE"
fi

if [ -n "$ITERATION" ]; then
    CMD="$CMD --iteration $ITERATION"
fi

if [ -n "$REFINE_ITERATIONS" ]; then
    CMD="$CMD --refine_iterations $REFINE_ITERATIONS"
fi

if [ "$REFINE_RING_FIX" = true ]; then
    CMD="$CMD --refine_ring_fix"
fi

if [ "$REFINE_DELAUNAY_FLIP" = true ]; then
    CMD="$CMD --refine_delaunay_flip"
fi

if [ "$REFINE_SURFACE_AWARE" = true ]; then
    CMD="$CMD --refine_surface_aware"
fi

if [ -n "$REFINE_PATIENCE" ]; then
    CMD="$CMD --refine_patience $REFINE_PATIENCE"
fi

if [ -n "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

# ============================================================================
# Run
# ============================================================================
echo "============================================================"
echo "Building Geodesic Mesh"
echo "============================================================"
echo "  Gaussian output: $GAUSSIAN_OUTPUT"
echo "  Surface:         $SURFACE"
echo "  Mesh method:     $MESH_METHOD"
echo "  N points:        ${N_POINTS:-<none>}"
echo "  Min radius:      ${MIN_RADIUS:-<none>}"
echo "  Seed:            $SEED"
echo "============================================================"
echo ""

$CMD

echo ""
echo "Done! Mesh saved to: $GAUSSIAN_OUTPUT/geodesic_mesh/"
