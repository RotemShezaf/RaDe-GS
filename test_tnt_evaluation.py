#!/usr/bin/env python3
"""
Test script for TNT (Tanks and Temples) evaluation

This script helps you test the evaluation pipeline on the TNT dataset.
It checks your setup and provides example commands.
"""

import os
import sys
import argparse
from pathlib import Path

def check_file_exists(filepath, name):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✓ {name} found: {filepath}")
        return True
    else:
        print(f"✗ {name} NOT found: {filepath}")
        return False

def check_tnt_setup(dataset_dir, scene_name):
    """Check if TNT dataset files are properly set up"""
    print(f"\n{'='*60}")
    print(f"Checking TNT setup for scene: {scene_name}")
    print(f"{'='*60}\n")
    
    all_good = True
    
    # Required files for TNT evaluation
    required_files = {
        "Ground truth PLY": os.path.join(dataset_dir, f"{scene_name}.ply"),
        "Crop JSON": os.path.join(dataset_dir, f"{scene_name}.json"),
        "Transformation matrix": os.path.join(dataset_dir, f"{scene_name}_trans.txt"),
        "COLMAP reference log": os.path.join(dataset_dir, f"{scene_name}_COLMAP_SfM.log"),
    }
    
    for name, filepath in required_files.items():
        if not check_file_exists(filepath, name):
            all_good = False
    
    if all_good:
        print(f"\n✓ All required files found for {scene_name}!")
    else:
        print(f"\n✗ Some required files are missing. Please check the TNT dataset setup.")
        print("\nYou need to download the evaluation data from:")
        print("https://drive.google.com/open?id=1UoKPiUUsKa0AVHFOrnMRhc5hFngjkE-t")
    
    return all_good

def print_usage_examples(scene_name):
    """Print example usage commands"""
    print(f"\n{'='*60}")
    print("Usage Examples")
    print(f"{'='*60}\n")
    
    print("Step 1: Cull the mesh (remove unseen parts)")
    print(f"python eval_tnt/cull_mesh.py \\")
    print(f"    --traj-path /path/to/your/camera_trajectory.log \\")
    print(f"    --mesh-path /path/to/your/reconstruction.ply \\")
    print(f"    --out-dir /path/to/output \\")
    print(f"    --dataset-dir /path/to/TNT/data/{scene_name}")
    
    print("\nStep 2: Evaluate the culled mesh")
    print(f"python eval_tnt/run.py \\")
    print(f"    --dataset-dir /path/to/TNT/data/{scene_name} \\")
    print(f"    --traj-path /path/to/your/camera_trajectory.log \\")
    print(f"    --ply-path /path/to/culled_mesh.ply \\")
    print(f"    --out-dir /path/to/evaluation_results")
    
    print("\n" + "="*60)
    print("Available TNT Scenes:")
    print("="*60)
    scenes = ["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    for scene in scenes:
        print(f"  - {scene}")

def check_dependencies():
    """Check if required Python packages are installed"""
    print(f"\n{'='*60}")
    print("Checking Python Dependencies")
    print(f"{'='*60}\n")
    
    required_packages = {
        'open3d': 'Open3D',
        'numpy': 'NumPy',
        'trimesh': 'Trimesh',
        'matplotlib': 'Matplotlib',
        'torch': 'PyTorch',
        'pyrender': 'PyRender (for mesh culling)',
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\nTo install missing packages:")
        print("pip install open3d numpy trimesh matplotlib torch pyrender")
    
    return all_installed

def find_mesh_files(search_dir):
    """Find potential mesh files to evaluate"""
    if not os.path.exists(search_dir):
        return []
    
    mesh_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.ply') and 'recon' in file.lower():
                mesh_files.append(os.path.join(root, file))
    
    return mesh_files

def main():
    parser = argparse.ArgumentParser(
        description="Test TNT evaluation setup",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to TNT dataset directory (e.g., /path/to/TNT/data/Barn)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"],
        help="Scene name"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if required dependencies are installed"
    )
    parser.add_argument(
        "--find-meshes",
        type=str,
        help="Search directory for reconstruction mesh files"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("TNT (Tanks and Temples) Evaluation Test Script")
    print(f"{'='*60}")
    
    # Check dependencies
    if args.check_deps or not args.dataset_dir:
        check_dependencies()
    
    # Find mesh files
    if args.find_meshes:
        print(f"\n{'='*60}")
        print(f"Searching for mesh files in: {args.find_meshes}")
        print(f"{'='*60}\n")
        mesh_files = find_mesh_files(args.find_meshes)
        if mesh_files:
            print(f"Found {len(mesh_files)} mesh file(s):")
            for mesh_file in mesh_files:
                print(f"  - {mesh_file}")
        else:
            print("No mesh files found.")
    
    # Check TNT setup
    if args.dataset_dir and args.scene:
        check_tnt_setup(args.dataset_dir, args.scene)
    elif args.dataset_dir:
        # Try to infer scene name from path
        scene = os.path.basename(os.path.normpath(args.dataset_dir))
        if scene in ["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]:
            check_tnt_setup(args.dataset_dir, scene)
        else:
            print(f"\n✗ Could not infer scene name from path: {args.dataset_dir}")
            print("Please specify --scene argument")
    
    # Print usage examples
    scene_name = args.scene or "Barn"
    print_usage_examples(scene_name)
    
    print("\n" + "="*60)
    print("Quick Start Guide")
    print("="*60)
    print("""
1. Download TNT evaluation data:
   https://drive.google.com/open?id=1UoKPiUUsKa0AVHFOrnMRhc5hFngjkE-t

2. Extract to a directory (e.g., /data/TNT/evaluation/data/)

3. Train your model and generate a mesh

4. Run this test script to verify setup:
   python test_tnt_evaluation.py --dataset-dir /data/TNT/evaluation/data/Barn --scene Barn --check-deps

5. Cull your mesh (remove unseen geometry)

6. Run evaluation to get precision, recall, and F-score

For more details, see eval_tnt/README.md
    """)

if __name__ == "__main__":
    main()
