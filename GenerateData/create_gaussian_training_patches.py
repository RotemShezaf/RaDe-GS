#!/usr/bin/env python3
"""
Generate training examples from Gaussian splats for learning geodesic distance computation.

This script creates training patches by:
1. Loading Gaussian splat data (positions, scales, rotations)
2. Computing neighborhood rings using Mahalanobis or Euclidean distance
3. Loading ground truth geodesic distances from precomputed data
4. Generating training examples with neighbor features and target distances

The training examples can be used to train a neural network to predict geodesic distances
from local neighborhood information on Gaussian splats.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from numpy import linalg as LA
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from utils.load_utils import find_available_iterations, load_gaussian_data

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.data_generation_utils import (
    get_all_points_nbrs_all_rings,
    ring1_neighbors_gaussians,
    get_neighborhood_by_ring
)
from plyfile import PlyData
from scene.gaussian_model import GaussianModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training examples from Gaussian splats"
    )
    
    parser.add_argument(
        "--gaussian_output",
        type=str,
        required=True,
        help="Path to Gaussian splatting output folder"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Training iteration to use (default: highest available)"
    )
    parser.add_argument(
        "--geodesic_data",
        type=str,
        default=None,
        help="Path to precomputed geodesic distance data (NPZ file). If not specified, will look for gt_geodesic.npz in {gaussian_output}/geodesic_distance/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save training examples"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1000,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--num_sources",
        type=int,
        default=1,
        help="Number of source points per example"
    )
    parser.add_argument(
        "--num_train_points",
        type=int,
        default=100,
        help="Number of training points per example"
    )
    parser.add_argument(
        "--use_mahalanobis",
        action="store_true",
        help="Use Mahalanobis distance for neighborhood computation"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=22,
        help="Number of ring-1 neighbors"
    )
    parser.add_argument(
        "--rings",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Ring levels to generate examples for (e.g., 2 3)"
    )
    parser.add_argument(
        "--add_normals",
        action="store_true",
        help="Include normal information in features"
    )
    parser.add_argument(
        "--add_euclidean_distance",
        action="store_true",
        help="Include Euclidean distance to neighbors in features"
    )
    parser.add_argument(
        "--constant_val",
        type=float,
        default=-10.0,
        help="Constant value for padding"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()




def extract_surface_and_texture_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract surface name and texture information from dataset path.
    
    Expected path patterns:
    - .../blue_texture/Saddle/level_02/output/...
    - .../Polynomial/SyntheticColmapData/red_texture/Paraboloid/...
    - .../output/polynomial/Paraboloid
    
    Args:
        path: Path to parse
    
    Returns:
        Tuple of (surface_name, texture) or (None, None) if not found
    """
    parts = path.parts
    surface_name = None
    texture = None
    
    # Common surface names to look for
    surface_names = ['Paraboloid', 'Saddle', 'HyperbolicParaboloid', 'Sphere', 'Torus']
    
    # Look for surface name in path parts
    for i, part in enumerate(parts):
        # Check if this part is a known surface name
        if part in surface_names:
            surface_name = part
            
            # Look backwards for texture (usually 1-2 parts before surface)
            for j in range(max(0, i-3), i):
                if 'texture' in parts[j].lower():
                    texture = parts[j]
                    break
            break
    
    # If not found by exact match, try to infer from path structure
    if surface_name is None:
        # Look for patterns like "polynomial/Paraboloid" or "output/Saddle"
        for i, part in enumerate(parts):
            if part.lower() in ['polynomial', 'synthetic', 'syntheticcolmapdata']:
                # Next capitalized word might be the surface
                for j in range(i+1, min(len(parts), i+4)):
                    if parts[j] and parts[j][0].isupper():
                        surface_name = parts[j]
                        break
    
    # Use last part as fallback for surface name
    if surface_name is None and len(parts) > 0:
        surface_name = parts[-1]
    
    return surface_name, texture


def compute_gaussian_normals(
    scales: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Compute approximate normals for Gaussians based on their principal axis.
    Uses the direction of smallest scale as the normal.
    
    Args:
        scales: (N, 3) array of scales
        rotations: (N, 4) array of quaternions
    
    Returns:
        (N, 3) array of normal vectors
    """
    num_gaussians = len(scales)
    normals = np.zeros((num_gaussians, 3))
    
    for i in range(num_gaussians):
        # Convert quaternion to rotation matrix
        q = rotations[i]
        q = q / np.linalg.norm(q)
        
        # Assume [x, y, z, w] format (common in Gaussian splatting)
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Get the axis corresponding to the smallest scale (most compressed direction)
        min_scale_idx = np.argmin(scales[i])
        normals[i] = R[:, min_scale_idx]
    
    return normals


def load_geodesic_distances(geodesic_path: Path) -> Dict:
    """
    Load precomputed geodesic distance data.
    
    Args:
        geodesic_path: Path to NPZ file with geodesic distances
    
    Returns:
        Dictionary with geodesic data
    """
    print(f"\nLoading geodesic distances from: {geodesic_path}")
    data = np.load(str(geodesic_path))
    
    print(f"  Sources: {len(data['source_indices'])}")
    print(f"  Gaussians: {data['geodesic_distances'].shape[1]}")
    
    return {
        'gaussian_positions': data['gaussian_positions'],
        'source_indices': data['source_indices'],
        'source_positions': data['source_positions'],
        'geodesic_distances': data['geodesic_distances'],
        'source_gaussian_indices': data.get('source_gaussian_indices', None)
    }


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_gaussians: int,
    scale_stats: Dict,
    examples_info: Dict,
    surface_name: Optional[str] = None,
    texture: Optional[str] = None
) -> None:
    """
    Save metadata about the data generation process.
    
    Args:
        output_dir: Output directory
        args: Command line arguments
        num_gaussians: Number of Gaussians in the dataset
        scale_stats: Statistics about Gaussian scales
        examples_info: Information about generated examples per ring
        surface_name: Name of the surface (if applicable)
        texture: Texture name (if applicable)
    """
    metadata = {
        'creation_info': {
            'timestamp': datetime.now().isoformat(),
            'script': 'create_gaussian_training_patches.py',
            'description': 'Training examples for geodesic distance learning on Gaussian splats'
        },
        'surface': {
            'name': surface_name if surface_name else 'Unknown',
            'type': 'Gaussian Splatting Reconstruction',
            'texture': texture if texture else 'Not specified'
        },
        'gaussian_data': {
            'source_folder': str(args.gaussian_output),
            'iteration': args.iteration if args.iteration else 'auto (highest)',
            'num_gaussians': num_gaussians,
            'scale_statistics': scale_stats
        },
        'geodesic_data': {
            'source_file': str(args.geodesic_data),
            'precomputed': True
        },
        'generation_parameters': {
            'num_iterations': args.num_iterations,
            'num_sources_per_iteration': args.num_sources,
            'num_train_points_per_iteration': args.num_train_points,
            'rings': args.rings,
            'seed': args.seed
        },
        'neighborhood': {
            'method': 'Mahalanobis' if args.use_mahalanobis else 'Euclidean',
            'n_neighbors_ring1': args.n_neighbors,
            'description': 'Neighborhood computed based on Gaussian covariance' if args.use_mahalanobis else 'Standard Euclidean distance'
        },
        'features': {
            'include_normals': args.add_normals,
            'include_euclidean_distance': args.add_euclidean_distance,
            'normalization_factor': scale_stats['mean'],
            'constant_val': args.constant_val,
            'description': 'Normals computed from Gaussian principal axes' if args.add_normals else 'Position-based features only'
        },
        'output_files': examples_info,
        'texture': {
            'type': 'Name',
            'description': 'Color image used to generate the mesh that created the gaussians',
        }
    }
    
    # Save as JSON
    metadata_path = output_dir / 'generation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Also save a human-readable text version
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GAUSSIAN TRAINING EXAMPLES METADATA\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {metadata['creation_info']['timestamp']}\n")
        f.write(f"Script: {metadata['creation_info']['script']}\n\n")
        
        f.write("SURFACE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Name: {metadata['surface']['name']}\n")
        f.write(f"  Type: {metadata['surface']['type']}\n")
        f.write(f"  Texture: {metadata['surface']['texture']}\n\n")
        
        f.write("GAUSSIAN DATA:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Source: {metadata['gaussian_data']['source_folder']}\n")
        f.write(f"  Iteration: {metadata['gaussian_data']['iteration']}\n")
        f.write(f"  Number of Gaussians: {metadata['gaussian_data']['num_gaussians']:,}\n")
        f.write(f"  Mean scale: {scale_stats['mean']:.6f}\n")
        f.write(f"  Scale range: [{scale_stats['min']:.6f}, {scale_stats['max']:.6f}]\n\n")
        
        f.write("GENERATION PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Iterations: {metadata['generation_parameters']['num_iterations']}\n")
        f.write(f"  Sources per iteration: {metadata['generation_parameters']['num_sources_per_iteration']}\n")
        f.write(f"  Training points per iteration: {metadata['generation_parameters']['num_train_points_per_iteration']}\n")
        f.write(f"  Rings: {metadata['generation_parameters']['rings']}\n")
        f.write(f"  Random seed: {metadata['generation_parameters']['seed']}\n\n")
        
        f.write("NEIGHBORHOOD COMPUTATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Method: {metadata['neighborhood']['method']}\n")
        f.write(f"  Ring-1 neighbors: {metadata['neighborhood']['n_neighbors_ring1']}\n")
        f.write(f"  Description: {metadata['neighborhood']['description']}\n\n")
        
        f.write("FEATURES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Include normals: {metadata['features']['include_normals']}\n")
        f.write(f"  Include Euclidean distances: {metadata['features']['include_euclidean_distance']}\n")
        f.write(f"  Normalization factor: {metadata['features']['normalization_factor']:.6f}\n")
        f.write(f"  Constant value (padding): {metadata['features']['constant_val']}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("-" * 40 + "\n")
        for ring, info in metadata['output_files'].items():
            f.write(f"  {ring}:\n")
            f.write(f"    File: {info['filename']}\n")
            f.write(f"    Examples: {info['num_examples']:,}\n")
            f.write(f"    Shape: {info['shape']}\n")
            f.write(f"    Size: {info['size_mb']:.2f} MB\n\n")
        
        f.write("TEXTURE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Type: {metadata['texture']['type']}\n")
        f.write(f"  Description: {metadata['texture']['description']}\n")
    
    print(f"README saved to: {readme_path}")


def get_ring_size_mapping(ring: int, use_mahalanobis: bool) -> int:
    """
    Get expected maximum number of neighbors for a given ring.
    These are empirical estimates based on typical Gaussian splat densities.
    
    Args:
        ring: Ring number (1-4)
        use_mahalanobis: Whether Mahalanobis distance is used
    
    Returns:
        Maximum number of neighbors
    """
    if use_mahalanobis:
        # Mahalanobis tends to have more variable neighborhood sizes
        mapping = {1: 25, 2: 90, 3: 250, 4: 600}
    else:
        # Euclidean distance has more uniform neighborhoods
        mapping = {1: 22, 2: 75, 3: 200, 4: 500}
    
    return mapping.get(ring, 100)


def create_train_example(
    point_idx: int,
    positions: np.ndarray,
    normals: Optional[np.ndarray],
    geodesic_distances: np.ndarray,
    ring_nbrs: Dict[int, np.ndarray],
    ring1_nbrs: Dict[int, np.ndarray],
    ring: int,
    normalization_factor: float,
    constant_val: float,
    add_normals: bool,
    add_euclidean_distance: bool
) -> Optional[np.ndarray]:
    """
    Create a single training example for a point.
    
    Args:
        point_idx: Index of the point to create example for
        positions: (N, 3) Gaussian positions
        normals: (N, 3) Gaussian normals (or None)
        geodesic_distances: (N,) geodesic distances from source
        ring_nbrs: Dictionary of ring-k neighbors
        ring1_nbrs: Dictionary of ring-1 neighbors
        ring: Ring level
        normalization_factor: Factor for normalizing coordinates
        constant_val: Value for padding
        add_normals: Whether to include normals
        add_euclidean_distance: Whether to include Euclidean distances
    
    Returns:
        Training example array or None if invalid
    """
    max_num_nbrs = get_ring_size_mapping(ring, False)  # Conservative estimate
    
    # Get neighbors
    nbrs = ring_nbrs[point_idx]
    r1_nbrs = ring1_nbrs[point_idx]
    
    # Get point data
    p_xyz = positions[point_idx]
    p_u = geodesic_distances[point_idx]
    
    # Get neighbor relative positions
    nbrs_xyz = positions[nbrs] - p_xyz
    nbrs_euclidean_distances = LA.norm(nbrs_xyz, axis=1)
    nbrs_u = geodesic_distances[nbrs]
    
    # Get ring-1 neighbor data
    r1_nbrs_xyz = positions[r1_nbrs] - p_xyz
    r1_nbrs_euclidean_distances = LA.norm(r1_nbrs_xyz, axis=1)
    r1_nbrs_u = geodesic_distances[r1_nbrs]
    
    # Build feature arrays
    if add_normals and normals is not None:
        nbrs_normals = normals[nbrs]
        r1_nbrs_normals = normals[r1_nbrs]
        
        if add_euclidean_distance:
            input_data = np.concatenate((
                nbrs_normals,
                nbrs_xyz,
                np.expand_dims(nbrs_euclidean_distances, axis=1),
                np.expand_dims(nbrs_u, axis=1)
            ), axis=1)
            r1_input_data = np.concatenate((
                r1_nbrs_normals,
                r1_nbrs_xyz,
                np.expand_dims(r1_nbrs_euclidean_distances, axis=1),
                np.expand_dims(r1_nbrs_u, axis=1)
            ), axis=1)
        else:
            input_data = np.concatenate((
                nbrs_normals,
                nbrs_xyz,
                np.expand_dims(nbrs_u, axis=1)
            ), axis=1)
            r1_input_data = np.concatenate((
                r1_nbrs_normals,
                r1_nbrs_xyz,
                np.expand_dims(r1_nbrs_u, axis=1)
            ), axis=1)
    else:
        if add_euclidean_distance:
            input_data = np.concatenate((
                nbrs_xyz,
                np.expand_dims(nbrs_euclidean_distances, axis=1),
                np.expand_dims(nbrs_u, axis=1)
            ), axis=1)
            r1_input_data = np.concatenate((
                r1_nbrs_xyz,
                np.expand_dims(r1_nbrs_euclidean_distances, axis=1),
                np.expand_dims(r1_nbrs_u, axis=1)
            ), axis=1)
        else:
            input_data = np.concatenate((
                nbrs_xyz,
                np.expand_dims(nbrs_u, axis=1)
            ), axis=1)
            r1_input_data = np.concatenate((
                r1_nbrs_xyz,
                np.expand_dims(r1_nbrs_u, axis=1)
            ), axis=1)
    
    # Filter: only keep neighbors with distance <= current point
    input_data = input_data[input_data[:, -1] <= p_u]
    r1_input_data = r1_input_data[r1_input_data[:, -1] > p_u]
    
    # Track which neighbors are closer (for masking)
    min1_vals = input_data[:, -1] > p_u
    
    if input_data.shape[0] > max_num_nbrs:
        print(f"Warning: ring size > max ({input_data.shape[0]} > {max_num_nbrs})")
        return None
    
    # Get ring-1 minimum for dropout augmentation
    r1_min_val = r1_nbrs_u.min() if len(r1_nbrs_u) > 0 else p_u
    
    # Normalize: shift to zero minimum
    min_input = input_data[:, -1].min() if len(input_data) > 0 else p_u
    input_data[:, -1] = input_data[:, -1] - min_input
    p_u = p_u - min_input
    r1_min_val = r1_min_val - min_input
    
    # Normalize coordinates and distances
    nn_mean = -1 * constant_val
    if add_normals and normals is not None:
        input_data[:, 3:] = (input_data[:, 3:] / normalization_factor) * nn_mean
    else:
        input_data = (input_data / normalization_factor) * nn_mean
    
    p_u = (p_u / normalization_factor) * nn_mean
    r1_min_val = (r1_min_val / normalization_factor) * nn_mean
    
    # Apply masking
    input_data[min1_vals, -1] = constant_val
    
    # Pad to fixed size
    const_rows = np.ones((max_num_nbrs - input_data.shape[0], input_data.shape[1])) * (2 * constant_val)
    input_data = np.vstack([input_data, const_rows])
    
    # Construct final example
    example = np.append(input_data.flatten(), p_u)
    example = np.append(example, r1_min_val)
    
    return example


def generate_training_examples(
    positions: np.ndarray,
    normals: Optional[np.ndarray],
    geodesic_data: Dict,
    ring_nbrs_dict: Dict,
    ring1_nbrs: Dict,
    ring: int,
    num_iterations: int,
    num_sources: int,
    num_train_points: int,
    normalization_factor: float,
    constant_val: float,
    add_normals: bool,
    add_euclidean_distance: bool,
    seed: int = 42
) -> np.ndarray:
    """
    Generate training examples by randomly sampling sources and training points.
    
    Returns:
        (M, D) array of training examples
    """
    np.random.seed(seed)
    
    num_gaussians = len(positions)
    all_source_indices = geodesic_data['source_gaussian_indices']
    all_geodesic_distances = geodesic_data['geodesic_distances']
    
    examples = []
    
    print(f"\nGenerating training examples (ring {ring}):")
    print(f"  Iterations: {num_iterations}")
    print(f"  Sources per iteration: {num_sources}")
    print(f"  Train points per iteration: {num_train_points}")
    
    for i in tqdm(range(num_iterations), desc="Generating examples"):
        # Randomly select sources from available precomputed sources
        if num_sources <= len(all_source_indices):
            selected_source_idxs = np.random.choice(
                len(all_source_indices),
                num_sources,
                replace=False
            )
        else:
            print(f"Warning: Requested {num_sources} sources but only {len(all_source_indices)} available")
            selected_source_idxs = np.arange(len(all_source_indices))
        
        # Get minimum distance from selected sources
        selected_distances = all_geodesic_distances[selected_source_idxs]
        min_distances = selected_distances.min(axis=0)
        
        # Randomly select training points (excluding sources)
        source_gaussian_idxs = all_source_indices[selected_source_idxs]
        available_points = np.setdiff1d(np.arange(num_gaussians), source_gaussian_idxs)
        
        if len(available_points) < num_train_points:
            train_points = available_points
        else:
            train_points = np.random.choice(
                available_points,
                num_train_points,
                replace=False
            )
        
        # Create examples for each training point
        for point_idx in train_points:
            example = create_train_example(
                point_idx,
                positions,
                normals,
                min_distances,
                ring_nbrs_dict[ring],
                ring1_nbrs,
                ring,
                normalization_factor,
                constant_val,
                add_normals,
                add_euclidean_distance
            )
            
            if example is not None:
                examples.append(example)
    
    if len(examples) == 0:
        raise ValueError("No valid examples generated!")
    
    return np.vstack(examples)


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Gaussian Training Example Generation")
    print("="*80)
    
    # Load Gaussian data
    output_folder = Path(args.gaussian_output)
    positions, scales, rotations, opacities = load_gaussian_data(
        output_folder,
        args.iteration
    )
    
    # Extract surface name and texture from path
    surface_name, texture = extract_surface_and_texture_from_path(output_folder)
    print(f"\nDetected from path:")
    print(f"  Surface: {surface_name if surface_name else 'Unknown'}")
    print(f"  Texture: {texture if texture else 'Not specified'}")
    
    # Compute normals if needed
    normals = None
    if args.add_normals:
        print("\nComputing Gaussian normals...")
        normals = compute_gaussian_normals(scales, rotations)
    
    # Determine geodesic data path
    if args.geodesic_data is None:
        # Look for geodesic data in standard location
        geodesic_path = output_folder / "geodesic_distance" / "gt_geodesic.npz"
        if not geodesic_path.exists():
            raise FileNotFoundError(
                f"Geodesic data not found at: {geodesic_path}\n"
                f"Please either:\n"
                f"  1. Run compute_gaussian_geodesic_distances.py first to generate geodesic distances, or\n"
                f"  2. Specify the geodesic data path with --geodesic_data"
            )
        print(f"\nUsing geodesic data from standard location: {geodesic_path}")
    else:
        geodesic_path = Path(args.geodesic_data)
        if not geodesic_path.exists():
            raise FileNotFoundError(f"Geodesic data not found at: {geodesic_path}")
    
    # Load geodesic distances
    geodesic_data = load_geodesic_distances(geodesic_path)
    
    # Compute normalization factor (mean edge length approximation)
    # For Gaussians, use mean scale as proxy
    mean_scale = scales.mean()
    min_scale = scales.min()
    max_scale = scales.max()
    normalization_factor = mean_scale
    print(f"\nNormalization factor (mean scale): {normalization_factor:.6f}")
    
    # Collect scale statistics for metadata
    scale_stats = {
        'mean': float(mean_scale),
        'min': float(min_scale),
        'max': float(max_scale),
        'std': float(scales.std())
    }
    
    # Compute neighborhood rings
    print(f"\nComputing neighborhood rings...")
    print(f"  Using {'Mahalanobis' if args.use_mahalanobis else 'Euclidean'} distance")
    print(f"  Ring-1 neighbors: {args.n_neighbors}")
    
    ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs = get_all_points_nbrs_all_rings(
        positions,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=scales if args.use_mahalanobis else None,
        gaussian_rotations=rotations if args.use_mahalanobis else None,
        n_neighbors_ring1=args.n_neighbors
    )
    
    ring_nbrs_dict = {
        1: ring1_nbrs,
        2: ring2_nbrs,
        3: ring3_nbrs,
        4: ring4_nbrs
    }
    
    # Generate training examples for each ring
    examples_info = {}
    
    for ring in args.rings:
        print(f"\n{'='*80}")
        print(f"Generating examples for ring {ring}")
        print(f"{'='*80}")
        
        examples = generate_training_examples(
            positions,
            normals,
            geodesic_data,
            ring_nbrs_dict,
            ring1_nbrs,
            ring,
            args.num_iterations,
            args.num_sources,
            args.num_train_points,
            normalization_factor,
            args.constant_val,
            args.add_normals,
            args.add_euclidean_distance,
            args.seed + ring  # Different seed per ring
        )
        
        # Save examples
        output_name = f"gaussian_examples_ring{ring}_n{len(examples)}.npy"
        output_path = output_dir / output_name
        np.save(output_path, examples)
        
        print(f"\nSaved {len(examples)} examples to: {output_path}")
        print(f"  Example shape: {examples.shape}")
        print(f"  Memory size: {examples.nbytes / 1e6:.2f} MB")
        
        # Collect info for metadata
        examples_info[f"ring_{ring}"] = {
            'filename': output_name,
            'num_examples': len(examples),
            'shape': str(examples.shape),
            'size_mb': examples.nbytes / 1e6
        }
    
    # Save metadata
    save_metadata(
        output_dir,
        args,
        len(positions),
        scale_stats,
        examples_info,
        surface_name,
        texture
    )
    
    print(f"\n{'='*80}")
    print("Training example generation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
