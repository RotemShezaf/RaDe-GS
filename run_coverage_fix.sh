#!/bin/bash
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate geo_splat
cd /home/rotem.shezaf/RaDe-GS

DATASET_DIR="TrainData/Polynomial/SyntheticColmapData/blue_texture/HyperbolicParaboloid/level_04/light_0"
OUTPUT_DIR="${DATASET_DIR}/output_coverage_ratio_fix_early_reg"

echo "=========================================="
echo "Training with coverage_ratio fix + early regularization (7k)"
echo "=========================================="
echo "Dataset: ${DATASET_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

python train.py -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" --eval

echo ""
echo "=========================================="
echo "Extracting Mesh with Tetrahedra"
echo "=========================================="

python mesh_extract_tetrahedra.py -s "${DATASET_DIR}" -m "${OUTPUT_DIR}" --eval

echo ""
echo "=========================================="
echo "Pipeline completed"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
