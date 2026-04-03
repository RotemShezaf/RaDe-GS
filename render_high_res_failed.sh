#!/bin/bash
set -e
FAILED="$(cat /home/rotem.shezaf/RaDe-GS/failed_shapes_high_res.txt)"
cd /home/rotem.shezaf/RaDe-GS
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate geo_splat
bash scripts/tosca/render_all_tosca.sh \
    --textures blue \
    --colmap_resolutions high_res \
    --image_mesh_resolution high_res \
    --num_views 400 \
    --image_width 960 \
    --image_height 960 \
    --camera_radius 250 \
    --use_decoupled_appearance \
    --max_parallel 3 \
    --shapes "$FAILED"
