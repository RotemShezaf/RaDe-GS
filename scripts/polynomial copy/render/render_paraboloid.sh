#!/bin/bash

# Script to render Paraboloid surface with synthetic COLMAP dataset
# Can be run from project root or from GenerateData/scripts directory

# Determine project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "Running from project root: $PROJECT_ROOT"
echo "Rendering Paraboloid surface..."

# Generate synthetic COLMAP dataset from polynomial mesh
# --colmap_level: Resolution for COLMAP point cloud (higher = denser points3D)
# --image_mesh_level: Resolution for rendering images (can be lower for speed)
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Paraboloid \
    --colmap_level 4 \
    --image_mesh_level 1 \
    --num_views 150 \
    --image_width 640 \
    --image_height 480 \
    --camera_radius 2.6 \
    --texture_name blue \

echo "Paraboloid rendering complete!"
