#!/bin/bash

# Script to render a single TOSCA shape (cat0) with synthetic COLMAP dataset
# Can be run from project root or from scripts directory

# Determine project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "Running from project root: $PROJECT_ROOT"
echo "Rendering TOSCA shape: cat0..."


# Generate synthetic COLMAP dataset from TOSCA mesh
# --colmap_resolution: Resolution for COLMAP point cloud ('high_res' = denser points3D)
# --image_mesh_resolution: Resolution for rendering images (can be 'low_res' for speed)
python3 GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py \
    --shape cat0 \
    --colmap_resolution high_res \
    --image_mesh_resolution high_res \
    --num_views 400 \
    --image_width 1024 \
    --image_height 1024 \
    --auto_camera_radius \
    --texture_name blue \
    --use_decoupled_appearance


echo "cat0 rendering complete!"
