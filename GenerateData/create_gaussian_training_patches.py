#!/usr/bin/env python3
"""
Generate training examples from Gaussian splats for learning geodesic distance computation.

This script creates training patches by:
1. Loading Gaussian splat data (positions, scales, rotations)
2. Computing neighborhood rings using Mahalanobis or Euclidean distance
3. Loading ground truth geodesic distances from precomputed data
4. Generating patches training examples with neighbor features and target distances

The training examples can be used to train a neural network to predict geodesic distances
from local neighborhood information on Gaussian splats.
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from numpy import linalg as LA
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from utils.load_utils import load_gaussian_data, extract_surface_and_texture_from_path
from utils.data_generation_utils import build_rotation
from utils.data_transformation_utils import get_masked_entry
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


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file with command-line arguments.
    Command-line arguments take precedence over config file.
    
    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments
    
    Returns:
        Updated arguments namespace
    """
    # Map config keys to argument names
    config_mapping = {
        'gaussian_output': 'gaussian_output',
        'geodesic_data': 'geodesic_data',
        'output_dir': 'output_dir',
        'iteration': 'iteration',
        'num_iterations': 'num_iterations',
        'num_sources': 'num_sources',
        'num_train_points': 'num_train_points',
        'seed': 'seed',
        'use_mahalanobis': 'use_mahalanobis',
        'use_r1_min_val': 'use_r1_min_val',
        'n_neighbors': 'n_neighbors',
        'rings': 'rings',
        'attributes': 'attributes',
        'constant_val': 'constant_val',
        'mask_attributes': 'mask_attributes',
        'mask_constant': 'mask_constant',
        'normalize_per_patch': 'normalize_per_patch'
    }
    
    # Get default values from parser to detect which args were explicitly set
    parser = argparse.ArgumentParser()
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            defaults[action.dest] = action.default
    
    # Apply config values only if not explicitly set via command line
    for config_key, arg_name in config_mapping.items():
        if config_key in config and config[config_key] is not None:
            current_value = getattr(args, arg_name, None)
            
            # For paths and iteration, only use config if command-line arg is None or not set
            if arg_name in ['gaussian_output', 'geodesic_data', 'output_dir', 'iteration']:
                if current_value is None:
                    setattr(args, arg_name, config[config_key])
            # Check if argument has default value (wasn't set on command line)
            elif arg_name == 'rings' and current_value == [2, 3]:
                setattr(args, arg_name, config[config_key])
            elif arg_name == 'attributes' and current_value == ['xyz']:
                setattr(args, arg_name, config[config_key])
            elif arg_name == 'mask_attributes' and current_value == []:
                setattr(args, arg_name, config[config_key])
            elif arg_name in ['num_iterations', 'num_sources', 'num_train_points', 'seed', 'n_neighbors', 'constant_val', 'mask_constant']:
                # For numeric values, we can't easily detect if they were set explicitly
                # So we use a convention: if config exists and arg seems like default, use config
                setattr(args, arg_name, config[config_key])
            elif arg_name in ['use_mahalanobis', 'use_r1_min_val'] and not current_value:
                setattr(args, arg_name, config[config_key])
    
    # Store ring_size_mapping if present
    if 'ring_size_mapping' in config:
        args.ring_size_mapping = config['ring_size_mapping']
    
    # Store dataset info if present
    if 'dataset' in config:
        args.dataset_info = config['dataset']
    
    return args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training examples from Gaussian splats"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="GenerateData/configs/saddle.yaml",
        help="Path to YAML dataset configuration file (command-line args override config)"
    )
    parser.add_argument(
        "--gaussian_output",
        type=str,
        required=False,
        default=None,
        help="Path to Gaussian splatting output folder (can be set in config file)"
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
        required=False,
        default=None,
        help="Directory to save training examples (can be set in config file)"
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
        "--use_r1_min_val",
        action="store_true",
        help="Include ring-1 minimum distance value in training examples (for dropout augmentation)"
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
        "--attributes",
        type=str,
        nargs="+",
        default=["xyz"],
        choices=["xyz", "scale", "opacity", "rotation", "sh", "normals", "euclidean_distances"],
        help="Attributes to include for each neighbor (in order). Options: xyz, opacity, rotation, sh, normals, euclidean_distances"
    )
    parser.add_argument(
        "--add_normals",
        action="store_true",
        help="[DEPRECATED] Use --attributes normals instead. Include normal information in features"
    )
    parser.add_argument(
        "--add_euclidean_distance",
        action="store_true",
        help="[DEPRECATED] Use --attributes euclidean_distances instead. Include Euclidean distance to neighbors in features"
    )
    parser.add_argument(
        "--nn_mean",
        type=float,
        default=1.0,
        help="Constant value for padding"
    )
    parser.add_argument(
        "--normalize_per_patch",
        action="store_true",
        help="Normalize each patch by its own nearest neighbor distance instead of global mean"
    )
    parser.add_argument(        "--mask_attributes",
        type=str,
        nargs="+",
        default=[],
        choices=["xyz", "opacity", "rotation", "scale", "sh", "normals", "euclidean_distances", "geodesic_distance"],
        help="Attributes to mask for invalid neighbors (where neighbor distance > point distance)"
    )
    parser.add_argument(
        "--mask_constant",
        type=float,
        default=-10.0,
        help="Constant value to use for masking invalid neighbors"
    )
    parser.add_argument(        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()




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
    R = build_rotation(rotations)
    
    # Get the axis corresponding to the smallest scale (most compressed direction)
    min_scale_idx = np.argmin(scales, axis=1)
    normals = R[:, min_scale_idx]
    
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
            'use_r1_min_val': args.use_r1_min_val,
            'normalization_factor': scale_stats['mean'],
            'nn_mean': args.nn_mean,            'mask_attributes': args.mask_attributes,
            'mask_constant': args.mask_constant,            'description': 'Normals computed from Gaussian principal axes' if args.add_normals else 'Position-based features only',
            'normalize_per_patch': args.normalize_per_patch,  'description': 'Each patch normalized by its own nearest neighbor distance' if args.normalize_per_patch else 'Global normalization factor used'
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
        f.write(f"  Mean wanted nearest neighbor distance: {metadata['features']['nn_mean']:.6f}\n\n")
        
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


def get_ring_size_mapping(ring: int, use_mahalanobis: bool, custom_mapping: Optional[Dict] = None) -> int:
    """
    Get expected maximum number of neighbors for a given ring.
    These are empirical estimates based on typical Gaussian splat densities.
    
    Args:
        ring: Ring number (1-4)
        use_mahalanobis: Whether Mahalanobis distance is used
        custom_mapping: Optional custom ring size mapping from config file
    
    Returns:
        Maximum number of neighbors
    """
    if custom_mapping:
        method = 'mahalanobis' if use_mahalanobis else 'euclidean'
        if method in custom_mapping and ring in custom_mapping[method]:
            return custom_mapping[method][ring]
    
    # Default mappings
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
    nn_mean: float,
    attributes: list,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    sh_features: Optional[np.ndarray] = None,
    use_mahalanobis: bool = False,
    use_r1_min_val: bool = True,
    mask_attributes: list = [],
    mask_constant: float = -10.0,
    ring_size_mapping: Optional[Dict] = None,
    normalize_per_patch: bool = False,
    per_point_nn_distances: Optional[np.ndarray] = None
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
        nn_mean: wanted mean distance for nearest neighbor
        attributes: List of attributes to include ['xyz', 'opacity', 'rotation', 'sh', 'normals', 'euclidean_distances']
        scales: (N, 3) Gaussian scales (optional)
        rotations: (N, 4) Gaussian rotations (optional)
        opacities: (N, 1) Gaussian opacities (optional)
        sh_features: (N, K) Spherical harmonics features (optional)
    
    Returns:
        Training example array or None if invalid
    """
    max_num_nbrs = get_ring_size_mapping(ring, use_mahalanobis, ring_size_mapping)
    
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
    
    # Build feature arrays based on attributes
    nbrs_features = []
    r1_nbrs_features = []
    p_fetures = []
    if "xyz" in attributes:
        # Add relative positions
        nbrs_features.append(nbrs_xyz)
        r1_nbrs_features.append(r1_nbrs_xyz)
        p_fetures.append(np.zeros((3,)))  # Point relative position is zero
    if "opacity" in attributes:
        if opacities is None:
            raise ValueError("opacities data required for 'opacity' attribute")
        nbrs_features.append(opacities[nbrs])
        r1_nbrs_features.append(opacities[r1_nbrs])
        p_fetures.append(opacities[point_idx])
    if "scale" in attributes:
        if scales is None:
            raise ValueError("scales data required for 'scale' attribute")
        nbrs_features.append(scales[nbrs])
        r1_nbrs_features.append(scales[r1_nbrs])
        p_fetures.append(scales[point_idx])  
    if "rotation" in attributes:
        if rotations is None:
            raise ValueError("rotations data required for 'rotation' attribute")
        nbrs_features.append(rotations[nbrs])
        r1_nbrs_features.append(rotations[r1_nbrs])
        p_fetures.append(rotations[point_idx])
    if "sh" in attributes:
        if sh_features is None:
            raise ValueError("sh_features data required for 'sh' attribute")
        nbrs_features.append(sh_features[nbrs])
        r1_nbrs_features.append(sh_features[r1_nbrs])
        p_fetures.append(sh_features[point_idx])
    if "normals" in attributes:
        if normals is None:
            raise ValueError("normals data required for 'normals' attribute")
        nbrs_features.append(normals[nbrs])
        r1_nbrs_features.append(normals[r1_nbrs])
        p_fetures.append(normals[point_idx])
    if "euclidean_distances" in attributes:
        nbrs_features.append(np.expand_dims(nbrs_euclidean_distances, axis=1))
        r1_nbrs_features.append(np.expand_dims(r1_nbrs_euclidean_distances, axis=1))
        p_fetures.append(np.array([0.0]))  # Point to itself distance is zero
    
    # Add geodesic distances at the end (always included)
    nbrs_features.append(np.expand_dims(nbrs_u, axis=1))
    r1_nbrs_features.append(np.expand_dims(r1_nbrs_u, axis=1))
    
    # Concatenate all features
    neighborhood = np.concatenate(nbrs_features, axis=1)
    r1_neighborhood = np.concatenate(r1_nbrs_features, axis=1)
    p_fetures = np.concatenate(p_fetures, axis=0) if p_fetures else np.array([])
    
    
    # Filter: only keep neighbors with distance <= current point
    neighborhood = neighborhood[neighborhood[:, -1] <= p_u]
    #r1_neighborhood= r1_neighborhood[r1_neighborhood[:, -1] > p_u]
    
    
    if neighborhood.shape[0] > max_num_nbrs:
        print(f"Warning: ring size > max ({neighborhood.shape[0]} > {max_num_nbrs})")
        return None
    
    # Get ring-1 minimum for dropout augmentation
    r1_min_val = r1_nbrs_u.min() if len(r1_nbrs_u) > 0 else p_u
    
    # Normalize: shift to zero minimum the geodesin distances
    min_input = neighborhood[:, -1].min() if len(neighborhood) > 0 else p_u
    neighborhood[:, -1] = neighborhood[:, -1] - min_input
    p_u = p_u - min_input
    r1_min_val = r1_min_val - min_input
    
    # Determine normalization factor: per-patch or global
    if normalize_per_patch and per_point_nn_distances is not None:
        current_normalization = per_point_nn_distances[nbrs].mean() if len(nbrs) > 0 else normalization_factor
    else:
        current_normalization = normalization_factor
    
    # Normalize coordinates and distances (skip normals, opacity, rotation, sh at the beginning)
    #nn is the wanted mean distaice for nearens neighbors
    
    #
    
    #normalize required coordinates
    attr_index = 0
    for i, attr in enumerate(attributes):
        if attr in ["xyz"]:
            neighborhood[:, attr_index:attr_index+3] = (neighborhood[:, attr_index:attr_index+3] / current_normalization) * nn_mean
            r1_neighborhood[:, attr_index:attr_index+3] = (r1_neighborhood[:, attr_index:attr_index+3] / current_normalization) * nn_mean
            attr_index += 3
        elif attr == "scale":
            #alse normelize scales
            neighborhood[:, attr_index:attr_index+3] = (neighborhood[:, attr_index:attr_index+3] / current_normalization) * nn_mean
            r1_neighborhood[:, attr_index:attr_index+3] = (r1_neighborhood[:, attr_index:attr_index+3] / current_normalization) * nn_mean
            p_fetures[attr_index:attr_index+3] = (p_fetures[attr_index:attr_index+3] / current_normalization) * nn_mean
            attr_index += 3
        elif attr == "normals":
            attr_index += 3
        elif attr == "opacity":
            attr_index += 1
        elif attr == "rotation":
            attr_index += 4
        elif attr == "sh":
            attr_index += sh_features.shape[1] if sh_features is not None else 0
        elif attr == "euclidean_distances":
            neighborhood[:, attr_index:attr_index+1] = (neighborhood[:, attr_index:attr_index+1] / current_normalization) * nn_mean
            r1_neighborhood[:, attr_index:attr_index+1] = (r1_neighborhood[:, attr_index:attr_index+1] / current_normalization) * nn_mean
            attr_index += 1
    

        # Normalize geodesic distances
        neighborhood[:, -1] = (neighborhood[:, -1] / current_normalization) * nn_mean
    

    p_u = (p_u / current_normalization) * nn_mean
    r1_min_val = (r1_min_val / current_normalization) * nn_mean
    
    # Create masked entry for padding (same shape as a single neighbor)
    masked_entry = get_masked_entry(attributes, mask_constant).detach().cpu().numpy()
  
    # Pad to fixed size using masked entries
    pad_num = max_num_nbrs - neighborhood.shape[0]
    if pad_num > 0:
        padding = np.tile(masked_entry, (pad_num, 1))
        neighborhood = np.vstack([neighborhood, padding])
    
    assert neighborhood.shape[0] == max_num_nbrs, f"Expected {max_num_nbrs} neighbors, got {neighborhood.shape[0]}"
    # Construct final example: [neighborhood_features..., target, r1_min_val?]
    if use_r1_min_val:
        example = np.append(neighborhood.flatten(), p_fetures.flatten())
        example = np.append(example, r1_min_val)
        example = np.append(example, p_u) 
    else:
        example = np.append(neighborhood.flatten(), p_fetures.flatten())
        example = np.append(example, p_u)
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
    nn_mean: float,
    attributes: list,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    sh_features: Optional[np.ndarray] = None,
    use_mahalanobis: bool = False,
    use_r1_min_val: bool = True,
    mask_attributes: list = [],
    mask_constant: float = -10.0,
    ring_size_mapping: Optional[Dict] = None,
    normalize_per_patch: bool = False,
    per_point_nn_distances: Optional[np.ndarray] = None,
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
    print(f"  Attributes: {', '.join(attributes)}")
    
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
                nn_mean,
                attributes,
                scales,
                rotations,
                opacities,
                sh_features,
                use_mahalanobis,
                use_r1_min_val,
                mask_attributes,
                mask_constant,
                ring_size_mapping,
                normalize_per_patch,
                per_point_nn_distances
            )
            
            if example is not None:
                examples.append(example)
    
    if len(examples) == 0:
        raise ValueError("No valid examples generated!")
    
    return np.vstack(examples)


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        args = merge_config_with_args(config, args)
        print(f"Configuration loaded successfully")
    
    # Validate required arguments
    if args.gaussian_output is None:
        raise ValueError("--gaussian_output must be specified either via command line or config file")
    if args.output_dir is None:
        raise ValueError("--output_dir must be specified either via command line or config file")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Handle deprecated arguments
    if args.add_normals and "normals" not in args.attributes:
        print("Warning: --add_normals is deprecated. Use --attributes normals instead.")
        args.attributes.append("normals")
    if args.add_euclidean_distance and "euclidean_distances" not in args.attributes:
        print("Warning: --add_euclidean_distance is deprecated. Use --attributes euclidean_distances instead.")
        args.attributes.append("euclidean_distances")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Gaussian Training Example Generation")
    print("="*80)
    if hasattr(args, 'dataset_info') and args.dataset_info:
        print(f"\nDataset: {args.dataset_info.get('name', 'Unknown')}")
        if 'description' in args.dataset_info:
            print(f"Description: {args.dataset_info['description']}")
    print(f"\nAttributes to include: {', '.join(args.attributes)}")
    
    # Load Gaussian data
    output_folder = Path(args.gaussian_output)
    gaussian_model = load_gaussian_data(
        output_folder,
        args.iteration
    )
    
    # Extract attributes from GaussianModel
    positions = gaussian_model.get_xyz.detach().cpu().numpy()
    scales = gaussian_model.get_scaling.detach().cpu().numpy()
    rotations = gaussian_model.get_rotation.detach().cpu().numpy()
    opacities = gaussian_model.get_opacity.detach().cpu().numpy()
    
    # Extract surface name and texture from path
    surface_name, texture = extract_surface_and_texture_from_path(output_folder)
    print(f"\nDetected from path:")
    print(f"  Surface: {surface_name if surface_name else 'Unknown'}")
    print(f"  Texture: {texture if texture else 'Not specified'}")
    
    # Compute normals if needed
    normals = None
    if "normals" in args.attributes:
        print("\nComputing Gaussian normals...")
        normals = compute_gaussian_normals(scales, rotations)
    
    # Load SH features if needed
    sh_features = None
    if "sh" in args.attributes:
        print("\nLoading spherical harmonics features...")
        # Get all SH features (DC + rest)
        sh_features = gaussian_model.get_features.detach().cpu().numpy()
        # Flatten to (N, K) where K = 3 * (degree+1)^2
        sh_features = sh_features.reshape(sh_features.shape[0], -1)
        print(f"  SH features shape: {sh_features.shape}")
    
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
    
    ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs,\
    mean_dist, per_point_dist = get_all_points_nbrs_all_rings(
        positions,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=scales if args.use_mahalanobis else None,
        gaussian_rotations=rotations if args.use_mahalanobis else None,
        n_neighbors_ring1=args.n_neighbors
    )

    normalization_factor = mean_dist
    
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
            args.nn_mean,
            args.attributes,
            scales,
            rotations,
            opacities,
            sh_features,
            args.use_mahalanobis,
            args.use_r1_min_val,
            args.mask_attributes,
            args.mask_constant,
            getattr(args, 'ring_size_mapping', None),
            args.normalize_per_patch,
            per_point_dist,
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
            'size_mb': examples.nbytes / 1e6,
            'attributes': args.attributes
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
