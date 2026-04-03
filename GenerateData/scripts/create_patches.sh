#!/bin/bash

# Script to render Saddle surface with synthetic COLMAP dataset
# Can be run from project root or from GenerateData/scripts directory

# Determine project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ "$SCRIPT_DIR" == */GenerateData/scripts ]]; then
    PROJECT_ROOT="$SCRIPT_DIR/../.."
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT" || exit 1

echo "Running from project root: $PROJECT_ROOT"
# Automatic (finds geodesic data in output folder)
python GenerateData/create_gaussian_training_patches.py \
    --gaussian_output TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_02/output/sparse \
    --output_dir TrainData/datasets/gaussian_patches/blue_texture/Saddle \
    --num_iterations 100

python GenerateData/compute_gaussian_geodesic_distances.py \
    --gaussian_output TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_02/output/sparse \
    --surface Saddle \
    --merge_only
    --source_mesh_resolution 8\
    --mesh_level 1 
    
