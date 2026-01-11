#!/usr/bin/env python3
"""
Compute geodesic distances on Gaussian splats from polynomial surfaces.

This script calculates ground truth geodesic distances on a Gaussian splatting 
reconstruction by projecting each Gaussian to the nearest vertex on a high-resolution 
ground truth mesh, then using the mesh's exact geodesic distances.

WORKFLOW:
---------
1. Load Gaussian splat from output folder (point_cloud/iteration_X/point_cloud.ply)
2. Load ground truth mesh at level 0 (highest resolution) from raw data
3. Generate a low-resolution source mesh using the same parametric surface definition
4. Map source mesh vertices to nearest ground truth mesh vertices
5. Compute exact geodesic distances on GT mesh using MMP algorithm (gdist)
6. Find closest mesh vertex for each Gaussian splat
7. Transfer geodesic distances from mesh vertices to Gaussians
8. Save results in a structured format for further analysis

The script supports:
- Batch processing of multiple source vertices
- Partial computation with source range specification
- Automatic merging of partial results into complete ground truth
- Caching of expensive geodesic computations

OUTPUT STRUCTURE:
-----------------
{output_folder}/
    geodesic_distance/
        gt_partial/
            sources_range_{start}_{end}.npz    # Partial results for source range
        gt_geodesic.npz                         # Complete merged ground truth

Each .npz file contains:
    - gaussian_positions: (N_gaussians, 3) Gaussian center positions
    - source_indices: (N_sources,) mesh vertex indices of sources
    - source_positions: (N_sources, 3) positions of source vertices
    - geodesic_distances: (N_sources, N_gaussians) geodesic distances
    - closest_mesh_indices: (N_gaussians,) nearest mesh vertex for each Gaussian
    - closest_mesh_distances: (N_gaussians,) Euclidean distance to nearest vertex
    - source_gaussian_indices: (N_sources,) Gaussian indices corresponding to each source

USAGE:
------
# Compute geodesic distances using a 20x20 source mesh:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 20

# Use a different source mesh resolution:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 50

# Compute partial results for a source range (recommended for parallelization):
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 20 \\
    --source_start 0 \\
    --source_end 200

# Merge partial results into complete ground truth:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --merge_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
import pickle
import json
from datetime import datetime
from tqdm import tqdm
# Ensure project root is in sys.path for module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from distance.dist import exact_geodesic_via_vtp_vertex_distance, vertex_dist
from plyfile import PlyData
from utils.geodesic_utils import load_ply
from GenerateData.GenerateRawPolynomialMesh import generate_surface_mesh
from GenerateData.utils.load_utils import load_ground_truth_mesh, find_available_iterations, load_gaussian_data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute geodesic distances on Gaussian splats from polynomial surfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments (unless merge_only is specified)
    parser.add_argument(
        '--gaussian_output',
        type=str,
        required=True,
        help='Path to Gaussian splatting output folder (containing point_cloud/iteration_*/)'
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="TrainData/Polynomial/raw",
        help="Base directory containing raw polynomial mesh data"
    )
    parser.add_argument(
        "--surface",
        type=str,
        choices=["Paraboloid", "Saddle", "HyperbolicParaboloid"],
        help="Surface type (required unless --merge_only is specified)"
    )
    
    # Source vertex selection
    parser.add_argument(
        "--num_sources",
        type=int,
        default=None,
        help="Total number of source vertices to use. If not specified, uses all mesh vertices as sources."
    )
    parser.add_argument(
        "--source_mesh_resolution",
        type=int,
        default=20,
        help="Resolution for generating source mesh (NxN grid). Sources will be vertices of this mesh."
    )
    parser.add_argument(
        "--source_start",
        type=int,
        default=None,
        help="Start index of source vertex range (for partial computation)"
    )
    parser.add_argument(
        "--source_end",
        type=int,
        default=None,
        help="End index of source vertex range (exclusive, for partial computation)"
    )
    parser.add_argument(
        "--source_selection",
        type=str,
        choices=["uniform", "random", "all"],
        default="uniform",
        help="Strategy for selecting source vertices: uniform spacing, random sampling, or all vertices"
    )
    
    # Mesh and iteration selection
    parser.add_argument(
        "--mesh_level",
        type=int,
        default=0,
        help="Mesh resolution level to use for ground truth (0 = highest resolution)"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Gaussian splatting training iteration to use. If not specified, uses the highest available iteration."
    )
    
    # Merge mode
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Only merge existing partial results, skip computation"
    )
    
    # Optional settings
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing mesh geodesic data and compute only for missing sources"
    )
    parser.add_argument(
        "--use_mahalanobis",
        action="store_true",
        help="Use Mahalanobis distance based on Gaussian covariance for distance computations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (used in random source selection)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.merge_only and args.surface is None:
        parser.error("--surface is required unless --merge_only is specified")
    
    if args.source_start is not None and args.source_end is None:
        parser.error("--source_end must be specified when --source_start is provided")
    
    if args.source_end is not None and args.source_start is None:
        parser.error("--source_start must be specified when --source_end is provided")
    
    if args.source_start is not None and args.source_end is not None:
        if args.source_start >= args.source_end:
            parser.error("--source_start must be less than --source_end")
    
    return args


def generate_source_mesh_and_map(
    surface_type: str,
    source_mesh_resolution: int,
    gt_mesh_vertices: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a low-resolution source mesh and map its vertices to nearest GT mesh vertices.
    
    Args:
        surface_type: Surface type (e.g., 'Paraboloid', 'Saddle', 'HyperbolicParaboloid')
        source_mesh_resolution: Resolution for source mesh (NxN grid)
        gt_mesh_vertices: (N, 3) array of ground truth mesh vertices
        x_range: (min, max) x range from ground truth mesh
        y_range: (min, max) y range from ground truth mesh
    
    Returns:
        Tuple of:
        - source_indices: (S,) array of GT mesh vertex indices corresponding to sources
        - source_positions: (S, 3) array of actual source positions from source mesh
    """
    print(f"\n{'='*80}")
    print(f"Generating Source Mesh")
    print(f"{'='*80}")
    print(f"  Surface type: {surface_type}")
    print(f"  Source mesh resolution: {source_mesh_resolution}x{source_mesh_resolution}")
    print(f"  X range: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
    print(f"  Y range: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
    
    # Generate source mesh using the same parametric definition
    source_mesh, source_normals, arc_length = generate_surface_mesh(
        surface_type=surface_type,
        nx=source_mesh_resolution,
        ny=source_mesh_resolution,
        x_range=x_range,
        y_range=y_range,
        adaptive=False
    )
    
    source_positions = np.asarray(source_mesh.vertices)
    print(f"  Generated {len(source_positions)} source vertices")
    print(f"  Arc length resolution: {arc_length:.6f}")
    
    # Map source positions to nearest GT mesh vertices
    print(f"\n  Mapping sources to ground truth mesh vertices...")
    tree = KDTree(gt_mesh_vertices)
    distances, source_indices = tree.query(source_positions)
    
    print(f"  Mapping statistics:")
    print(f"    Mean distance to nearest GT vertex: {distances.mean():.6f}")
    print(f"    Max distance to nearest GT vertex: {distances.max():.6f}")
    print(f"    Min distance to nearest GT vertex: {distances.min():.6f}")
    
    if distances.max() > 0.01:
        print(f"\n  Warning: Some sources are far from GT mesh vertices (max = {distances.max():.6f})")
        print(f"           Consider using a higher resolution ground truth mesh.")
    
    return source_indices, source_positions


def compute_geodesic_distances_for_sources(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute exact geodesic distances from each source to all vertices using MMP algorithm.
    
    Args:
        vertices: (N, 3) array of mesh vertex positions
        faces: (M, 3) array of triangle indices
        source_indices: (S,) array of source vertex indices
        verbose: Whether to show detailed progress
    
    Returns:
        (S, N) array of geodesic distances from each source to all vertices
    """
    print(f"\n{'='*80}")
    print(f"Computing Geodesic Distances (MMP Algorithm)")
    print(f"{'='*80}")
    print(f"  Sources: {len(source_indices)}")
    print(f"  Target vertices: {len(vertices)}")
    print(f"  Total computations: {len(source_indices)} x {len(vertices)} = {len(source_indices) * len(vertices):,}")
    print(f"\n  Note: This uses exact geodesic computation (gdist/MMP) which can be slow.")
    print(f"        Progress bar estimates remaining time based on completed sources.\n")
    
    num_sources = len(source_indices)
    num_vertices = len(vertices)
    #distances = np.zeros((num_sources, num_vertices), dtype=np.float32)
    
    #vertices_for_laybary = vertices.tolist()
    #faces_for_laybary = faces.tolist()
    # Compute distances with progress bar
    
    distances =exact_geodesic_via_vtp_vertex_distance(
        v=vertices,
        f=faces,
        src_vi=source_indices.tolist(),
        sources_are_disjoint=True
    )
    #breakpoint()
    
    print(f"\n  Distance statistics:")
    print(f"    Min: {distances.min():.6f}")
    print(f"    Max: {distances.max():.6f}")
    print(f"    Mean: {distances.mean():.6f}")
    print(f"    Std: {distances.std():.6f}")
    
    return distances


def save_mesh_geodesic_gt(
    data_root: Path,
    surface: str,
    mesh_level: int,
    source_indices: np.ndarray,
    source_positions: np.ndarray,
    geodesic_distances: np.ndarray,
    mesh_vertices: np.ndarray
) -> Path:
    """
    Save mesh geodesic ground truth to data_root/surface/level_{level}/geodesic/.
    
    Args:
        data_root: Base data directory
        surface: Surface name
        mesh_level: Mesh resolution level
        source_indices: (S,) source vertex indices
        source_positions: (S, 3) source positions
        geodesic_distances: (S, N) geodesic distances from sources to all vertices
        mesh_vertices: (N, 3) mesh vertex positions
    
    Returns:
        Path to saved file
    """
    output_dir = data_root / surface / f"level_{mesh_level}" / "geodesic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "mesh_geodesic_gt.npz"
    
    print(f"\n{'='*80}")
    print(f"Saving Mesh Geodesic Ground Truth")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")
    
    np.savez_compressed(
        output_path,
        source_indices=source_indices,
        source_positions=source_positions,
        geodesic_distances=geodesic_distances,
        mesh_vertices=mesh_vertices
    )
    
    print(f"  Saved data:")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    print(f"    mesh_vertices: {mesh_vertices.shape}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return output_path


def load_mesh_geodesic_gt(
    data_root: Path,
    surface: str,
    mesh_level: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load existing mesh geodesic ground truth if it exists.
    
    Args:
        data_root: Base data directory
        surface: Surface name
        mesh_level: Mesh resolution level
    
    Returns:
        Tuple of (source_indices, source_positions, geodesic_distances) or None if not found
    """
    geodesic_path = data_root / surface / f"level_{mesh_level}" / "geodesic" / "mesh_geodesic_gt.npz"
    
    if not geodesic_path.exists():
        print(f"\n  No existing mesh geodesic data found at: {geodesic_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Loading Existing Mesh Geodesic Ground Truth")
    print(f"{'='*80}")
    print(f"  Path: {geodesic_path}")
    
    data = np.load(geodesic_path)
    source_indices = data['source_indices']
    source_positions = data['source_positions']
    geodesic_distances = data['geodesic_distances']
    
    print(f"  Loaded data:")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    
    return source_indices, source_positions, geodesic_distances


def find_missing_sources(
    all_source_indices: np.ndarray,
    existing_source_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find which sources are missing from existing data.
    
    Args:
        all_source_indices: All source indices we want to compute
        existing_source_indices: Source indices already computed
    
    Returns:
        Tuple of (missing_indices, missing_mask) where missing_mask is boolean array
    """
    # Find which indices from all_source_indices are not in existing_source_indices
    missing_mask = ~np.isin(all_source_indices, existing_source_indices)
    missing_indices = all_source_indices[missing_mask]
    
    print(f"\n{'='*80}")
    print(f"Checking for Missing Sources")
    print(f"{'='*80}")
    print(f"  Total requested sources: {len(all_source_indices)}")
    print(f"  Existing sources: {len(existing_source_indices)}")
    print(f"  Missing sources: {len(missing_indices)}")
    
    return missing_indices, missing_mask


def merge_geodesic_data(
    existing_source_indices: np.ndarray,
    existing_source_positions: np.ndarray,
    existing_geodesic_distances: np.ndarray,
    new_source_indices: np.ndarray,
    new_source_positions: np.ndarray,
    new_geodesic_distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge existing and newly computed geodesic data.
    
    Args:
        existing_source_indices: Existing source indices
        existing_source_positions: Existing source positions
        existing_geodesic_distances: Existing geodesic distances
        new_source_indices: New source indices
        new_source_positions: New source positions
        new_geodesic_distances: New geodesic distances
    
    Returns:
        Tuple of merged (source_indices, source_positions, geodesic_distances)
    """
    print(f"\n{'='*80}")
    print(f"Merging Geodesic Data")
    print(f"{'='*80}")
    
    # Concatenate
    merged_source_indices = np.concatenate([existing_source_indices, new_source_indices])
    merged_source_positions = np.concatenate([existing_source_positions, new_source_positions])
    merged_geodesic_distances = np.concatenate([existing_geodesic_distances, new_geodesic_distances], axis=0)
    
    # Sort by source index
    sort_order = np.argsort(merged_source_indices)
    merged_source_indices = merged_source_indices[sort_order]
    merged_source_positions = merged_source_positions[sort_order]
    merged_geodesic_distances = merged_geodesic_distances[sort_order]
    
    print(f"  Merged total sources: {len(merged_source_indices)}")
    
    return merged_source_indices, merged_source_positions, merged_geodesic_distances


def map_indexes_between_gaussian_and_surfaces(
    source_idx,
    source_points,
    dest_surface,
    use_mahalanobis=False,
    dest_scales=None,
    dest_rotations=None
):
    """
    Map indexes from source surface to destination surface.
    
    Args:
        source_idx: Indices to map from source
        source_points: Source point positions
        dest_surface: Destination point positions
        use_mahalanobis: Whether to use Mahalanobis distance
        dest_scales: (N, 3) scales for destination Gaussians (required if use_mahalanobis=True)
        dest_rotations: (N, 4) rotations for destination Gaussians (required if use_mahalanobis=True)
    
    Returns:
        Array of closest destination indices
    """
    from GenerateData.utils.data_generation_utils import map_points_to_surface
    
    return map_points_to_surface(
        query_points=source_points[source_idx],
        target_points=dest_surface,
        use_mahalanobis=use_mahalanobis,
        target_scales=dest_scales,
        target_rotations=dest_rotations
    ) 

def find_closest_mesh_vertices(
    gaussian_centers: np.ndarray,
    mesh_vertices: np.ndarray,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the closest mesh vertex for each Gaussian center.
    
    Args:
        gaussian_centers: (G, 3) array of Gaussian center positions
        mesh_vertices: (N, 3) array of mesh vertex positions
        use_mahalanobis: Whether to use Mahalanobis distance
        gaussian_scales: (G, 3) Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (G, 4) Gaussian rotations (required if use_mahalanobis=True)
    
    Returns:
        Tuple of:
        - closest_indices: (G,) array of mesh vertex indices
        - closest_distances: (G,) array of distances to closest vertex
    """
    from GenerateData.utils.data_generation_utils import map_points_to_surface
    
    print(f"\n{'='*80}")
    print(f"Finding Closest Mesh Vertices")
    print(f"{'='*80}")
    print(f"  Method: {'Mahalanobis' if use_mahalanobis else 'Euclidean'} distance")
    print(f"  Querying nearest neighbors for {len(gaussian_centers)} Gaussians...")
    
    

    closest_indices, closest_distances = map_points_to_surface(
        query_points=gaussian_centers,
        target_points=mesh_vertices,
        use_mahalanobis=use_mahalanobis,
        query_scales=gaussian_scales,
        query_rotations=gaussian_rotations,
        return_distances=True
    )

   
    
    print(f"\n  Projection statistics:")
    print(f"    Mean distance to closest vertex: {closest_distances.mean():.6f}")
    print(f"    Max distance to closest vertex: {closest_distances.max():.6f}")
    print(f"    Min distance to closest vertex: {closest_distances.min():.6f}")
    print(f"    Std distance to closest vertex: {closest_distances.std():.6f}")
    
    # Check for potential issues
    if closest_distances.max() > 0.1:
        print(f"\n  Warning: Some Gaussians are far from the mesh (max distance = {closest_distances.max():.6f})")
        print(f"           This may indicate misalignment between Gaussian splat and ground truth mesh.")
    
    return closest_indices, closest_distances


def transfer_geodesic_to_gaussians(
    mesh_geodesic_distances: np.ndarray,
    gaussian_to_mesh_indices: np.ndarray,
    source_mesh_indices: np.ndarray,
    source_gaussian_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer geodesic distances from mesh vertices to Gaussian splats.
    Also maps source mesh indices to corresponding Gaussian indices.
    
    Args:
        mesh_geodesic_distances: (S, N_mesh) array of geodesic distances on mesh
        gaussian_to_mesh_indices: (N_gaussian,) array mapping each Gaussian to closest mesh vertex
        source_mesh_indices: (S,) array of mesh vertex indices that are sources
    
    Returns:
        Tuple of:
        - gaussian_geodesic_distances: (S, N_gaussian) array of geodesic distances for Gaussians
        - source_gaussian_indices: (S,) array of Gaussian indices corresponding to sources
    """
    print(f"\n{'='*80}")
    print(f"Transferring Geodesic Distances to Gaussians")
    print(f"{'='*80}")
    
    num_sources = mesh_geodesic_distances.shape[0]
    num_gaussians = len(gaussian_to_mesh_indices)
    
    print(f"  Sources: {num_sources}")
    print(f"  Gaussians: {num_gaussians}")

    sort_order_sources = np.argsort(source_gaussian_indices)

    # Transfer distances by indexing
    gaussian_geodesic_distances = \
    mesh_geodesic_distances[:, gaussian_to_mesh_indices][sort_order_sources, :]
    

    
    print(f"  Mapped {num_sources} sources to Gaussian indices")
    print(f"  Source Gaussian index range: [{source_gaussian_indices.min()}, {source_gaussian_indices.max()}]")

    print(f"\n  Transferred distance statistics:")
    print(f"    Min: {gaussian_geodesic_distances.min():.6f}")
    print(f"    Max: {gaussian_geodesic_distances.max():.6f}")
    print(f"    Mean: {gaussian_geodesic_distances.mean():.6f}")
    print(f"    Std: {gaussian_geodesic_distances.std():.6f}")
    
    return gaussian_geodesic_distances


def save_partial_results(
    output_path: Path,
    gaussian_positions: np.ndarray,
    source_indices: np.ndarray,
    source_positions: np.ndarray,
    geodesic_distances: np.ndarray,
    closest_mesh_indices: np.ndarray,
    closest_mesh_distances: np.ndarray,
    source_gaussian_indices: np.ndarray
) -> None:
    """
    Save partial geodesic distance results to NPZ file.
    
    Args:
        output_path: Path to output .npz file
        gaussian_positions: (N_gaussian, 3) Gaussian center positions
        source_indices: (S,) source vertex indices
        source_positions: (S, 3) source vertex positions
        geodesic_distances: (S, N_gaussian) geodesic distances
        closest_mesh_indices: (N_gaussian,) nearest mesh vertex for each Gaussian
        closest_mesh_distances: (N_gaussian,) Euclidean distance to nearest vertex
        source_gaussian_indices: (S,) Gaussian indices corresponding to sources
    """
    print(f"\n{'='*80}")
    print(f"Saving Partial Results")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        gaussian_positions=gaussian_positions,
        #indexes in mes
        source_indices=source_indices,
        source_positions=source_positions,
        geodesic_distances=geodesic_distances,
        closest_mesh_indices=closest_mesh_indices,
        closest_mesh_distances=closest_mesh_distances,
        source_gaussian_indices=source_gaussian_indices
    )
    
    print(f"  Saved data:")
    print(f"    gaussian_positions: {gaussian_positions.shape}")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    print(f"    closest_mesh_indices: {closest_mesh_indices.shape}")
    print(f"    closest_mesh_distances: {closest_mesh_distances.shape}")
    print(f"    source_gaussian_indices: {source_gaussian_indices.shape}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")


def merge_partial_results(output_folder: Path, verbose: bool = False) -> None:
    """
    Merge all partial results into a single complete ground truth file.
    
    Args:
        output_folder: Base output folder containing gt_partial subdirectory
        verbose: Whether to show detailed progress
    """
    print(f"\n{'='*80}")
    print(f"Merging Partial Results")
    print(f"{'='*80}")
    
    partial_dir = output_folder / "geodesic_distance" / "gt_partial"
    if not partial_dir.exists():
        print(f"  Error: No partial results directory found at {partial_dir}")
        return
    
    # Find all partial result files
    partial_files = sorted(partial_dir.glob("sources_range_*.npz"))
    if not partial_files:
        print(f"  Error: No partial result files found in {partial_dir}")
        return
    
    print(f"  Found {len(partial_files)} partial result files:")
    for f in partial_files:
        print(f"    - {f.name}")
    
    # Load all partial results
    all_source_indices = []
    all_source_positions = []
    all_geodesic_distances = []
    all_source_gaussian_indices = []
    gaussian_positions = None
    closest_mesh_indices = None
    closest_mesh_distances = None
    
    for partial_file in tqdm(partial_files, desc="  Loading partial files"):
        data = np.load(partial_file)
        
        # Verify consistency (should be same for all files)
        if gaussian_positions is None:
            gaussian_positions = data['gaussian_positions']
            closest_mesh_indices = data['closest_mesh_indices']
            closest_mesh_distances = data['closest_mesh_distances']
        else:
            # Sanity check
            assert np.allclose(gaussian_positions, data['gaussian_positions']), \
                f"Gaussian positions mismatch in {partial_file.name}"
        
        all_source_indices.append(data['source_indices'])
        all_source_positions.append(data['source_positions'])
        all_geodesic_distances.append(data['geodesic_distances'])
        all_source_gaussian_indices.append(data['source_gaussian_indices'])
        
        if verbose:
            print(f"    Loaded {len(data['source_indices'])} sources from {partial_file.name}")
    
    # Concatenate all sources
    merged_source_indices = np.concatenate(all_source_indices, axis=0)
    merged_source_positions = np.concatenate(all_source_positions, axis=0)
    merged_geodesic_distances = np.concatenate(all_geodesic_distances, axis=0)
    merged_source_gaussian_indices = np.concatenate(all_source_gaussian_indices, axis=0)
    
    # Sort by gaussian source index (the indexes of sources in the Gaussian splat)
    sort_order = np.argsort(merged_source_gaussian_indices)
    merged_source_indices = merged_source_indices[sort_order]
    merged_source_positions = merged_source_positions[sort_order]
    merged_geodesic_distances = merged_geodesic_distances[sort_order]
    merged_source_gaussian_indices = merged_source_gaussian_indices[sort_order]
    
    print(f"\n  Merged data:")
    print(f"    Total sources: {len(merged_source_indices)}")
    print(f"    Gaussians: {len(gaussian_positions)}")
    print(f"    Geodesic distances shape: {merged_geodesic_distances.shape}")
    print(f"    Source Gaussian indices shape: {merged_source_gaussian_indices.shape}")
    
    # Save merged results
    output_path = output_folder / "geodesic_distance" / "gt_geodesic.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        gaussian_positions=gaussian_positions,
        source_indices=merged_source_indices,
        source_positions=merged_source_positions,
        geodesic_distances=merged_geodesic_distances,
        closest_mesh_indices=closest_mesh_indices,
        closest_mesh_distances=closest_mesh_distances,
        source_gaussian_indices=merged_source_gaussian_indices
    )
    
    print(f"\n  Saved complete ground truth to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"\n  Distance statistics:")
    print(f"    Min: {merged_geodesic_distances.min():.6f}")
    print(f"    Max: {merged_geodesic_distances.max():.6f}")
    print(f"    Mean: {merged_geodesic_distances.mean():.6f}")
    print(f"    Std: {merged_geodesic_distances.std():.6f}")


def save_computation_metadata(
    output_folder: Path,
    args: argparse.Namespace,
    num_gaussians: int,
    num_sources: int,
    surface_name: Optional[str] = None
) -> None:
    """
    Save metadata about the geodesic distance computation.
    
    Args:
        output_folder: Output folder path
        args: Command line arguments
        num_gaussians: Number of Gaussians
        num_sources: Total number of sources
        surface_name: Surface name
    """
    metadata = {
        'computation_info': {
            'timestamp': datetime.now().isoformat(),
            'script': 'compute_gaussian_geodesic_distances.py',
            'description': 'Ground truth geodesic distances on Gaussian splats'
        },
        'surface': {
            'name': surface_name if surface_name else args.surface,
            'type': args.surface if args.surface else 'Unknown',
            'data_root': str(args.data_root),
            'mesh_level': args.mesh_level
        },
        'gaussian_data': {
            'source_folder': str(args.gaussian_output),
            'iteration': args.iteration if args.iteration else 'auto (highest)',
            'num_gaussians': num_gaussians
        },
        'source_generation': {
            'method': 'parametric_mesh',
            'source_mesh_resolution': args.source_mesh_resolution,
            'total_sources': num_sources,
            'source_selection': args.source_selection
        },
        'distance_computation': {
            'method': 'Mahalanobis' if args.use_mahalanobis else 'Euclidean',
            'description': 'Gaussian covariance-based distance' if args.use_mahalanobis else 'Standard Euclidean distance',
            'geodesic_algorithm': 'MMP (exact geodesic via VTP)'
        },
        'parameters': {
            'use_mahalanobis': args.use_mahalanobis,
            'seed': args.seed,
            'source_start': args.source_start,
            'source_end': args.source_end
        }
    }
    
    # Save metadata
    metadata_path = output_folder / "geodesic_distance" / "computation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Metadata saved to: {metadata_path}")


def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# Gaussian Splat Geodesic Distance Computation")
    print(f"{'#'*80}")
    
    output_folder = Path(args.gaussian_output)
    
    # Handle merge-only mode
    if args.merge_only:
        merge_partial_results(output_folder, verbose=args.verbose)
        return
    
    
    # Load Gaussian splat
    gaussian_positions, gaussian_scales, gaussian_rotations, gaussian_opacities \
        = load_gaussian_data(output_folder, args.iteration)
    
    # Step 2: Load ground truth mesh
    data_root = Path(args.data_root)
    mesh_vertices, mesh_faces = load_ground_truth_mesh(data_root, args.surface, args.mesh_level)
    
    # Step 3: Compute mesh range from ground truth vertices
    x_min, x_max = mesh_vertices[:, 0].min(), mesh_vertices[:, 0].max()
    y_min, y_max = mesh_vertices[:, 1].min(), mesh_vertices[:, 1].max()
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    
    print(f"\n  Ground truth mesh range:")
    print(f"    X: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
    print(f"    Y: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
    
    # Step 4: Generate source mesh and map to GT mesh vertices
    all_source_indices, all_source_positions = generate_source_mesh_and_map(
        surface_type=args.surface,
        source_mesh_resolution=args.source_mesh_resolution,
        gt_mesh_vertices=mesh_vertices,
        x_range=x_range,
        y_range=y_range,
    )
    
    # Step 4.5: Load existing mesh geodesic data if --load is specified
    existing_mesh_geodesic_data = None
    if args.load:
        result = load_mesh_geodesic_gt(
            data_root=data_root,
            surface=args.surface,
            mesh_level=args.mesh_level
        )
        if result is not None:
            existing_mesh_geodesic_data = result

    
    # Determine which sources need to be computed
    if existing_mesh_geodesic_data is not None:
        existing_source_indices, existing_source_positions, existing_geodesic_distances = existing_mesh_geodesic_data
        
        # Find missing sources
        missing_source_indices, missing_mask = find_missing_sources(
            all_source_indices=all_source_indices,
            existing_source_indices=existing_source_indices
        )
        
        if len(missing_source_indices) == 0:
            print(f"\n  All sources already computed! Using existing data.")
            mesh_geodesic_distances = existing_geodesic_distances
            source_indices = all_source_indices
            source_positions_for_mesh = all_source_positions
        else:
            # Compute only for missing sources
            print(f"\n  Will compute geodesic distances for {len(missing_source_indices)} missing sources")
            source_indices_to_compute = missing_source_indices
            source_positions_to_compute = all_source_positions[missing_mask]
    else:
        # No existing data, compute for all sources
        print(f"\n  No existing mesh geodesic data. Computing for all sources.")
        source_indices_to_compute = all_source_indices
        source_positions_to_compute = all_source_positions
    
    # Determine source range for this run
    if args.source_start is not None and args.source_end is not None:
        source_start = args.source_start
        source_end = min(args.source_end, len(all_source_indices))
        source_indices = all_source_indices[source_start:source_end]
        source_positions_subset = all_source_positions[source_start:source_end]
        
        # If loading, filter to only compute missing ones in this range
        if existing_mesh_geodesic_data is not None and len(missing_source_indices) > 0:
            # Find which sources in our range are missing
            range_mask = (source_indices_to_compute >= source_start) & (source_indices_to_compute < source_end)
            source_indices = source_indices_to_compute[range_mask]
            source_positions_subset = source_positions_to_compute[range_mask]
        
        print(f"\n  Processing source range: [{source_start}, {source_end}) = {len(source_indices)} sources")
    else:
        source_start = 0
        source_end = len(all_source_indices)
        
        # Use the computed list (either all or missing)
        if existing_mesh_geodesic_data is not None and len(missing_source_indices) > 0:
            source_indices = source_indices_to_compute
            source_positions_subset = source_positions_to_compute
        else:
            source_indices = all_source_indices
            source_positions_subset = all_source_positions
        
        print(f"\n  Processing all {len(source_indices)} sources")


    # Map source indices to Gaussian indices (sources on mesh -> Gaussians)
    print(f"\n  Mapping source mesh indices to Gaussian indices...")
    print(f"  Method: {'Mahalanobis' if args.use_mahalanobis else 'Euclidean'} distance")
    source_gaussian_indices = map_indexes_between_gaussian_and_surfaces(
        source_indices,
        mesh_vertices,
        gaussian_positions,
        use_mahalanobis=args.use_mahalanobis,
        dest_scales=gaussian_scales if args.use_mahalanobis else None,
        dest_rotations=gaussian_rotations if args.use_mahalanobis else None
    )

    
    # Step 5: Find closest mesh vertices for Gaussians
    gaussian_to_mesh_indices, gaussian_to_mesh_distances = find_closest_mesh_vertices(
        gaussian_centers=gaussian_positions,
        mesh_vertices=mesh_vertices,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=gaussian_scales if args.use_mahalanobis else None,
        gaussian_rotations=gaussian_rotations if args.use_mahalanobis else None
    )

    
    # Step 6: Compute geodesic distances on mesh
    if len(source_indices) > 0:
        mesh_geodesic_distances_new = compute_geodesic_distances_for_sources(
            vertices=mesh_vertices,
            faces=mesh_faces,
            source_indices=source_indices,
            verbose=args.verbose
        )
        
        # Merge with existing data if applicable
        if existing_mesh_geodesic_data is not None and len(missing_source_indices) > 0:
            existing_source_indices, existing_source_positions, existing_geodesic_distances = existing_mesh_geodesic_data
            source_indices_for_mesh, source_positions_for_mesh, mesh_geodesic_distances = merge_geodesic_data(
                existing_source_indices=existing_source_indices,
                existing_source_positions=existing_source_positions,
                existing_geodesic_distances=existing_geodesic_distances,
                new_source_indices=source_indices,
                new_source_positions=source_positions_subset,
                new_geodesic_distances=mesh_geodesic_distances_new
            )
        else:
            source_indices_for_mesh = source_indices
            source_positions_for_mesh = source_positions_subset
            mesh_geodesic_distances = mesh_geodesic_distances_new
        

        # Save mesh geodesic ground truth
        save_mesh_geodesic_gt(
            data_root=data_root,
            surface=args.surface,
            mesh_level=args.mesh_level,
            source_indices=source_indices_for_mesh,
            source_positions=source_positions_for_mesh,
            geodesic_distances=mesh_geodesic_distances,
            mesh_vertices=mesh_vertices
        )
        
        # For Gaussian computation, use only the current range
        mesh_geodesic_distances_for_gaussians = mesh_geodesic_distances_new
    else:
        print(f"\n  No new sources to compute. Skipping geodesic computation.")
        # Use existing data for this range
        if existing_mesh_geodesic_data is not None:
            existing_source_indices, existing_source_positions, existing_geodesic_distances = existing_mesh_geodesic_data
            # Find indices in existing data that match our range
            if args.source_start is not None:
                range_mask = (existing_source_indices >= source_start) & (existing_source_indices < source_end)
                source_indices = existing_source_indices[range_mask]
                source_positions_subset = existing_source_positions[range_mask]
                mesh_geodesic_distances_for_gaussians = existing_geodesic_distances[range_mask]
            else:
                source_indices = existing_source_indices
                source_positions_subset = existing_source_positions
                mesh_geodesic_distances_for_gaussians = existing_geodesic_distances
            print(f"  Using {len(source_indices)} sources from existing data")



    
    # Step 7: Transfer geodesic distances to Gaussians
    gaussian_geodesic_distances= transfer_geodesic_to_gaussians(
        mesh_geodesic_distances=mesh_geodesic_distances_for_gaussians,
        gaussian_to_mesh_indices=gaussian_to_mesh_indices,
        source_mesh_indices=source_indices,
        source_gaussian_indices=source_gaussian_indices
    )
    
    # Step 8: Save results
    output_path = output_folder / "geodesic_distance" / "gt_partial" / f"sources_range_{source_start}_{source_end}.npz"
    
    save_partial_results(
        output_path=output_path,
        gaussian_positions=gaussian_positions,
        source_indices=source_indices,
        source_positions=source_positions_subset,  # Use actual source mesh positions
        geodesic_distances=gaussian_geodesic_distances,
        closest_mesh_indices=gaussian_to_mesh_indices,
        closest_mesh_distances=gaussian_to_mesh_distances,
        source_gaussian_indices=source_gaussian_indices
    )
    
    # Save computation metadata
    save_computation_metadata(
        output_folder=output_folder,
        args=args,
        num_gaussians=len(gaussian_positions),
        num_sources=len(all_source_indices)
    )
    
    print(f"\n{'#'*80}")
    print(f"# Computation Complete")
    print(f"{'#'*80}")
    print(f"\nPartial results saved to:")
    print(f"  {output_path}")
    print(f"\nTo merge all partial results, run:")
    print(f"  python {Path(__file__).name} --gaussian_output {args.gaussian_output} --merge_only")


if __name__ == "__main__":
    main()
