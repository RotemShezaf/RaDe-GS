#!/usr/bin/env python3
"""
Geodesic distance computation on triangular meshes.

This script loads a PLY mesh file (or generates a test mesh), computes geodesic
distances from a source vertex using both exact (MMP) and approximate (Fast Marching)
methods, and saves the results to separate files.

Usage:
    python geodesic.py --input mesh.ply --source 0 [--post-process] [--output-dir ./output]
    python geodesic.py --generate sphere --source 0  # Use auto-generated test mesh
"""

import argparse
import time
import os
import numpy as np
import torch
from tqdm import tqdm

def main():
    # Import inside the function to avoid circular import at module level
    from utils.geodesic_utils import (
        load_ply, save_ply, build_geodesic_mesh,
        compute_exact_geodesic, compute_fmm_geodesic,
        post_process_mesh, generate_test_mesh
    )
    
    parser = argparse.ArgumentParser(
        description="Compute geodesic distances on triangular meshes"
    )
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default=None,
        help='Input PLY mesh file path'
    )
    parser.add_argument(
        '--generate', '-g',
        type=str,
        choices=['sphere', 'plane'],
        default=None,
        help='Generate a test mesh instead of loading from file (sphere or plane)'
    )
    parser.add_argument(
        '--source', '-s',
        type=int,
        default=0,
        help='Source vertex index for geodesic computation (default: 0)'
    )
    parser.add_argument(
        '--post-process', '-p',
        action='store_true',
        help='Enable mesh post-processing (centering and normalization)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./assets/geodesic',
        help='Output directory for results (default: ./assets/geodesic)'
    )
    parser.add_argument(
        '--save-mesh',
        action='store_true',
        help='Save the processed mesh to output directory'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")
    
    # ========== Load or Generate Mesh ==========
    print("=" * 60)
    print("STEP 1: Loading Mesh")
    print("=" * 60)
    
    if args.input is not None:
        # Load mesh from file
        print(f"Loading mesh from: {args.input}")
        start_time = time.time()
        vertices, faces = load_ply(args.input)
        load_time = time.time() - start_time
        print(f"Loaded mesh in {load_time:.4f} seconds")
        mesh_name = os.path.splitext(os.path.basename(args.input))[0]
    elif args.generate is not None:
        # Generate test mesh
        print(f"Generating test mesh: {args.generate}")
        start_time = time.time()
        vertices, faces = generate_test_mesh(args.generate)
        load_time = time.time() - start_time
        print(f"Generated mesh in {load_time:.4f} seconds")
        mesh_name = f"test_{args.generate}"
    else:
        # Default: generate a sphere
        print("No input specified. Generating default test sphere...")
        start_time = time.time()
        vertices, faces = generate_test_mesh("sphere")
        load_time = time.time() - start_time
        print(f"Generated mesh in {load_time:.4f} seconds")
        mesh_name = "test_sphere"
    
    print(f"Mesh info: {len(vertices)} vertices, {len(faces)} faces\n")
    
    # ========== Post-Processing ==========
    postprocess_time = 0
    if args.post_process:
        print("=" * 60)
        print("STEP 2: Post-Processing Mesh")
        print("=" * 60)
        start_time = time.time()
        vertices, faces = post_process_mesh(vertices, faces, cluster_to_keep=1000)
        postprocess_time = time.time() - start_time
        print(f"Post-processing completed in {postprocess_time:.4f} seconds\n")
        
        if args.save_mesh:
            processed_mesh_path = os.path.join(args.output_dir, f"{mesh_name}_processed.ply")
            save_ply(processed_mesh_path, vertices, faces)
    else:
        print("Post-processing disabled (use --post-process to enable)\n")
    
    # ========== Validate Source Vertex ==========
    if args.source < 0 or args.source >= len(vertices):
        print(f"ERROR: Source vertex index {args.source} is out of range!")
        print(f"Valid range: 0 to {len(vertices) - 1}")
        return
    
    print(f"Source vertex: {args.source}")
    print(f"Source coordinates: {vertices[args.source]}\n")
    
    # ========== Build Geodesic Mesh ==========
    print("=" * 60)
    print("STEP 3: Building Geodesic Mesh Structure")
    print("=" * 60)
    start_time = time.time()

    
    # ========== Compute Exact Geodesic (MMP) ==========
    print("=" * 60)
    print("STEP 4: Computing Exact Geodesic Distances (MMP Algorithm)")
    print("=" * 60)
    start_time = time.time()
    exact_distances = compute_exact_geodesic(vertices, faces, args.source)
    exact_time = time.time() - start_time
    
    print(f"Computed exact distances in {exact_time:.4f} seconds")
    print(f"Distance statistics:")
    print(f"  - Min distance: {np.min(exact_distances):.6f}")
    print(f"  - Max distance: {np.max(exact_distances):.6f}")
    print(f"  - Mean distance: {np.mean(exact_distances):.6f}")
    print(f"  - Std deviation: {np.std(exact_distances):.6f}")
    
    # Save exact distances
    exact_output_path = os.path.join(args.output_dir, f"{mesh_name}_exact_distances.npy")
    np.save(exact_output_path, exact_distances)
    print(f"Saved exact distances to: {exact_output_path}\n")
    
    # ========== Compute Fast Marching Geodesic ==========
    print("=" * 60)
    print("STEP 5: Computing Approximate Geodesic Distances (Fast Marching)")
    print("=" * 60)
    start_time = time.time()
    fmm_distances = compute_fmm_geodesic(vertices, faces, args.source)
    fmm_time = time.time() - start_time
    
    print(f"Computed FMM distances in {fmm_time:.4f} seconds")
    print(f"Distance statistics:")
    print(f"  - Min distance: {np.min(fmm_distances):.6f}")
    print(f"  - Max distance: {np.max(fmm_distances):.6f}")
    print(f"  - Mean distance: {np.mean(fmm_distances):.6f}")
    print(f"  - Std deviation: {np.std(fmm_distances):.6f}")
    
    # Save FMM distances
    fmm_output_path = os.path.join(args.output_dir, f"{mesh_name}_fmm_distances.npy")
    np.save(fmm_output_path, fmm_distances)
    print(f"Saved FMM distances to: {fmm_output_path}\n")
    
    # ========== Compare Methods ==========
    print("=" * 60)
    print("STEP 6: Comparing Methods")
    print("=" * 60)
    
    # Compute differences
    abs_diff = np.abs(exact_distances - fmm_distances)
    rel_diff = abs_diff / (exact_distances + 1e-10)  # Avoid division by zero
    
    print(f"Absolute difference statistics:")
    print(f"  - Mean: {np.mean(abs_diff):.6f}")
    print(f"  - Max: {np.max(abs_diff):.6f}")
    print(f"  - Std: {np.std(abs_diff):.6f}")
    print(f"\nRelative difference statistics:")
    print(f"  - Mean: {np.mean(rel_diff):.6f}")
    print(f"  - Max: {np.max(rel_diff):.6f}")
    print(f"  - Std: {np.std(rel_diff):.6f}")
    
    # Save comparison
    comparison_output_path = os.path.join(args.output_dir, f"{mesh_name}_comparison.npz")
    np.savez(
        comparison_output_path,
        exact=exact_distances,
        fmm=fmm_distances,
        abs_diff=abs_diff,
        rel_diff=rel_diff
    )
    print(f"\nSaved comparison data to: {comparison_output_path}\n")
    
    # ========== Summary ==========
    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Mesh loading/generation: {load_time:.4f} seconds")
    if args.post_process:
        print(f"Post-processing:         {postprocess_time:.4f} seconds")
    print(f"Geodesic mesh build:     {build_time:.4f} seconds")
    print(f"Exact geodesic (MMP):    {exact_time:.4f} seconds")
    print(f"Fast Marching (FMM):     {fmm_time:.4f} seconds")
    print(f"Speedup (Exact/FMM):     {exact_time/fmm_time:.2f}x")
    
    total_time = load_time + build_time + exact_time + fmm_time
    if args.post_process:
        total_time += postprocess_time
    print(f"\nTotal computation time:  {total_time:.4f} seconds")
    print("=" * 60)
    print("\n✓ Geodesic computation completed successfully!")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
