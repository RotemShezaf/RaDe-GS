#!/bin/bash
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate geo_splat
cd /home/rotem.shezaf/RaDe-GS

SURFACE="HyperbolicParaboloid"
TEXTURE="checkerboard_blue"
COLMAP_LEVEL=4
IMAGE_MESH_LEVEL=0
NUM_VIEWS=400
IMAGE_WIDTH=640
IMAGE_HEIGHT=480
CAMERA_RADIUS=3.0
LIGHT_ID=0

OUTPUT_BASE="TrainData/Polynomial/SyntheticColmapData"
DATA_ROOT="TrainData/Polynomial/raw"

echo "=========================================="
echo "Step 1: Render synthetic dataset"
echo "=========================================="
echo "Surface: ${SURFACE}"
echo "Texture: ${TEXTURE}"
echo ""

python3 GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface ${SURFACE} \
    --texture_name ${TEXTURE} \
    --colmap_level ${COLMAP_LEVEL} \
    --image_mesh_level ${IMAGE_MESH_LEVEL} \
    --num_views ${NUM_VIEWS} \
    --image_width ${IMAGE_WIDTH} \
    --image_height ${IMAGE_HEIGHT} \
    --camera_radius ${CAMERA_RADIUS} \
    --output_root ${OUTPUT_BASE} \
    --data_root ${DATA_ROOT} \
    --light_id ${LIGHT_ID}

DATASET_DIR="${OUTPUT_BASE}/${TEXTURE}_texture/${SURFACE}/level_0${COLMAP_LEVEL}/light_${LIGHT_ID}"

echo ""
echo "=========================================="
echo "Step 2: Train Gaussian Splatting"
echo "=========================================="
echo "Dataset: ${DATASET_DIR}"

OUTPUT_DIR="${DATASET_DIR}/output_highfreq_texture"
python train.py -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" --eval

echo ""
echo "=========================================="
echo "Step 3: Extract Mesh"
echo "=========================================="

python mesh_extract_tetrahedra.py -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" --eval

echo ""
echo "=========================================="
echo "Pipeline completed"
echo "Results: ${OUTPUT_DIR}"
echo "=========================================="
