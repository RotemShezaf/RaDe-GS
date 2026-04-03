#!/bin/bash
# Script to render Barn ground truth mesh to images

# Configuration
INPUT_PLY="data/tnt/TNT_GOF/ground_truth/Barn/Barn.ply"
OUTPUT_PREFIX="data/tnt/TNT_GOF/ground_truth/Barn/render"
MAX_POINTS=${1:-100000}  # Default to 100000 if not specified

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate radegs

# Run rendering script
echo "Rendering mesh with max_points=${MAX_POINTS}..."
python scripts/render_mesh_simple.py "$INPUT_PLY" "$OUTPUT_PREFIX" --max_points "$MAX_POINTS"

echo "Done! Check output files at: ${OUTPUT_PREFIX}_*.png"
