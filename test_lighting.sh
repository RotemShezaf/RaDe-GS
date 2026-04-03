#!/bin/bash

# Test new lighting on Saddle surface
# Quick render with fewer views for testing

echo "=========================================="
echo "Testing improved lighting on Saddle"
echo "=========================================="

# The script outputs to default location
OUTPUT_DIR="TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_03"

echo ""
echo "Rendering Saddle with new balanced lighting setup..."
echo ""

# Render with new lighting
python3 GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Saddle \
    --texture_name blue \
    --colmap_level 3 \
    --image_mesh_level 1 \
    --camera_radius 2.6 \
    --num_views 50

echo ""
echo "=========================================="
echo "Testing complete! Analyzing lighting..."
echo "=========================================="

# Run lighting analysis
python3 debug_lighting.py "${OUTPUT_DIR}/images" --samples 25

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
