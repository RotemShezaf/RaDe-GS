#!/usr/bin/env python3
"""
Dense reconstruction using COLMAP.
Converts sparse COLMAP reconstruction to dense point cloud.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_colmap_dense_reconstruction(workspace_path, output_path=None):
    """
    Run COLMAP dense reconstruction pipeline.
    
    Args:
        workspace_path: Path to the COLMAP workspace (contains sparse/0/)
        output_path: Optional output path for dense point cloud
    """
    workspace_path = Path(workspace_path)
    sparse_path = workspace_path / "sparse" / "0"
    
    if not sparse_path.exists():
        print(f"Error: Sparse reconstruction not found at {sparse_path}")
        return False
    
    # Create dense folder
    dense_path = workspace_path / "dense"
    dense_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("COLMAP Dense Reconstruction Pipeline")
    print("="*60)
    print(f"Workspace: {workspace_path}")
    print(f"Sparse model: {sparse_path}")
    print(f"Dense output: {dense_path}")
    print()
    
    # Check if COLMAP is installed
    try:
        result = subprocess.run(['colmap', '-h'], capture_output=True, text=True)
        print("✓ COLMAP found")
    except FileNotFoundError:
        print("✗ COLMAP not found. Please install COLMAP first:")
        print("  Ubuntu: sudo apt install colmap")
        print("  Or build from source: https://colmap.github.io/install.html")
        return False
    
    print("\n" + "="*60)
    print("Step 1: Undistorting images...")
    print("="*60)
    
    # Step 1: Undistort images
    cmd_undistort = [
        'colmap', 'image_undistorter',
        '--image_path', str(workspace_path / 'images'),
        '--input_path', str(sparse_path),
        '--output_path', str(dense_path),
        '--output_type', 'COLMAP'
    ]
    
    print(f"Running: {' '.join(cmd_undistort)}")
    result = subprocess.run(cmd_undistort, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in undistortion: {result.stderr}")
        return False
    print("✓ Image undistortion complete")
    
    print("\n" + "="*60)
    print("Step 2: Computing stereo...")
    print("="*60)
    
    # Step 2: Stereo matching
    cmd_stereo = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', str(dense_path),
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true'
    ]
    
    print(f"Running: {' '.join(cmd_stereo)}")
    print("Note: This step may take a long time...")
    result = subprocess.run(cmd_stereo, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in stereo matching: {result.stderr}")
        return False
    print("✓ Stereo matching complete")
    
    print("\n" + "="*60)
    print("Step 3: Fusing stereo maps to point cloud...")
    print("="*60)
    
    # Step 3: Stereo fusion
    cmd_fusion = [
        'colmap', 'stereo_fusion',
        '--workspace_path', str(dense_path),
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', str(dense_path / 'fused.ply')
    ]
    
    print(f"Running: {' '.join(cmd_fusion)}")
    result = subprocess.run(cmd_fusion, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in stereo fusion: {result.stderr}")
        return False
    print("✓ Stereo fusion complete")
    
    # Check output
    fused_ply = dense_path / 'fused.ply'
    if fused_ply.exists():
        file_size = fused_ply.stat().st_size / (1024*1024)
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Dense point cloud saved to: {fused_ply}")
        print(f"File size: {file_size:.2f} MB")
        
        # Copy to output path if specified
        if output_path:
            import shutil
            shutil.copy(fused_ply, output_path)
            print(f"Copied to: {output_path}")
        
        return True
    else:
        print("\nError: Dense point cloud not generated")
        return False

if __name__ == "__main__":
    # Default path for Barn dataset
    workspace = "/home/rotem.shezaf/RaDe-GS/data/tnt/TNT_GOF/TrainingSet/Barn"
    output = "/home/rotem.shezaf/RaDe-GS/data/tnt/barn_dense_pointcloud.ply"
    
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]
    
    success = run_colmap_dense_reconstruction(workspace, output)
    sys.exit(0 if success else 1)
