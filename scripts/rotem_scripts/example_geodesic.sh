#!/bin/bash
#
# Demonstration script for geodesic distance computation
# Shows various usage patterns of the geodesic.py script
#

set -e  # Exit on error

# Determine the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GEODESIC_PY="$(python3 -c "import os; print(os.path.relpath(os.path.join('$PROJECT_ROOT', 'geodesic.py'), os.getcwd()))")"

echo "=========================================="
echo "Geodesic Distance Computation Demo"
echo "=========================================="
echo ""

# ============================================
# Example 1: Auto-generated sphere mesh
# ============================================
echo "----------------------------------------"
echo "Example 1: Auto-generated sphere"
echo "----------------------------------------"
echo "Command: python \$GEODESIC_PY --generate sphere --source 0 --output-dir ./output_sphere"
echo ""
python "$GEODESIC_PY" --generate sphere --source 0 --output-dir ./output_sphere
echo ""
echo "✓ Example 1 completed!"
echo ""
echo ""

# ============================================
# Example 2: Auto-generated plane mesh with post-processing
# ============================================
echo "----------------------------------------"
echo "Example 2: Auto-generated plane with post-processing"
echo "----------------------------------------"
echo "Command: python \$GEODESIC_PY --generate plane --source 100 --post-process --output-dir ./output_plane --save-mesh"
echo ""
python "$GEODESIC_PY" --generate plane --source 100 --post-process --output-dir ./output_plane --save-mesh
echo ""
echo "✓ Example 2 completed!"
echo ""
echo ""

# ============================================
# Example 3: User-provided PLY file (if exists)
# ============================================
echo "----------------------------------------"
echo "Example 3: User-provided PLY file"
echo "----------------------------------------"

# Check if a sample PLY file exists in common locations
SAMPLE_PLY=""
if [ -f "data/sample.ply" ]; then
    SAMPLE_PLY="data/sample.ply"
elif [ -f "mesh.ply" ]; then
    SAMPLE_PLY="mesh.ply"
elif [ -f "input.ply" ]; then
    SAMPLE_PLY="input.ply"
fi

if [ -n "$SAMPLE_PLY" ]; then
    echo "Found sample PLY file: $SAMPLE_PLY"
    echo "Command: python \$GEODESIC_PY --input \$SAMPLE_PLY --source 0 --post-process --output-dir ./output_custom"
    echo ""
    python "$GEODESIC_PY" --input "$SAMPLE_PLY" --source 0 --post-process --output-dir ./output_custom
    echo ""
    echo "✓ Example 3 completed!"
else
    echo "No sample PLY file found. Skipping this example."
    echo "To run with your own mesh, use:"
    echo "  python geodesic.py --input your_mesh.ply --source 0 --post-process --output-dir ./output_custom"
fi
echo ""
echo ""

# ============================================
# Example 4: Default behavior (no arguments)
# ============================================
echo "----------------------------------------"
echo "Example 4: Default behavior"
echo "----------------------------------------"
echo "Command: python \$GEODESIC_PY"
echo "(This will generate a default sphere with source vertex 0)"
echo ""
python "$GEODESIC_PY"
echo ""
echo "✓ Example 4 completed!"
echo ""
echo ""

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "All examples completed successfully!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  - Example 1 (sphere):         ./output_sphere/"
echo "  - Example 2 (plane):          ./output_plane/"
if [ -n "$SAMPLE_PLY" ]; then
    echo "  - Example 3 (custom mesh):    ./output_custom/"
fi
echo "  - Example 4 (default):        ./geodesic_output/"
echo ""
echo "Each output directory contains:"
echo "  - *_exact_distances.npy    : Exact geodesic distances (MMP algorithm)"
echo "  - *_fmm_distances.npy      : Fast Marching Method distances"
echo "  - *_comparison.npz         : Comparison data between both methods"
echo "  - *_processed.ply          : Processed mesh (if --save-mesh was used)"
echo ""
echo "=========================================="
echo "Usage examples:"
echo "=========================================="
echo ""
echo "1. Generate test sphere:"
echo "   python \$GEODESIC_PY --generate sphere --source 0"
echo ""
echo "2. Generate test plane:"
echo "   python \$GEODESIC_PY --generate plane --source 100"
echo ""
echo "3. Load custom PLY file:"
echo "   python \$GEODESIC_PY --input your_mesh.ply --source 0"
echo ""
echo "4. Enable post-processing:"
echo "   python \$GEODESIC_PY --input mesh.ply --post-process"
echo ""
echo "5. Specify output directory:"
echo "   python \$GEODESIC_PY --generate sphere --output-dir ./my_results"
echo ""
echo "6. Save processed mesh:"
echo "   python \$GEODESIC_PY --input mesh.ply --post-process --save-mesh"
echo ""
echo "7. Change source vertex:"
echo "   python \$GEODESIC_PY --generate sphere --source 50"
echo ""
echo "8. Full example:"
echo "   python \$GEODESIC_PY --input mesh.ply --source 10 --post-process --save-mesh --output-dir ./results"
echo ""
echo "For help:"
echo "   python \$GEODESIC_PY --help"
echo ""
