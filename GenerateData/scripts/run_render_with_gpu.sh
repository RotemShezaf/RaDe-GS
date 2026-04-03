#!/bin/bash

# Complete script to run rendering with GPU allocation and proper environment
# This script can be run from anywhere and will handle GPU allocation + conda activation

echo "=== RaDe-GS Rendering Pipeline with GPU ==="
echo ""
echo "Step 1: Requesting GPU allocation on gipdeep6..."
echo "This will allocate a GPU node for 5 hours"
echo ""
echo "Step 2: Inside the GPU node, run these commands:"
echo "  conda activate geo_splat"
echo "  cd /home/rotem.shezaf/RaDe-GS"
echo "  ./GenerateData/scripts/render_saddle.sh"
echo ""
echo "Starting srun now..."
echo ""

srun --nodelist=gipdeep6 --gres=gpu:1 --time=05:00:00 --pty bash -c '
    echo "=== GPU node allocated ==="
    echo "Node: $(hostname)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo ""
    
    echo "Activating conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate geo_splat
    
    echo "Environment: $CONDA_DEFAULT_ENV"
    echo "Python: $(which python)"
    echo ""
    
    echo "Navigating to project directory..."
    cd /home/rotem.shezaf/RaDe-GS
    
    echo "Starting render_saddle.sh..."
    echo ""
    ./GenerateData/scripts/render_saddle.sh
    
    echo ""
    echo "=== Rendering complete ==="
    echo "Press Ctrl+D or type exit to release GPU"
    
    # Keep session open
    exec bash
'
