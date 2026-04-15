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

Supports both single-source and multi-source configurations for combined datasets.
Each data source gets its own output directory for training patches.
"""

import os
import sys
import argparse
import json
import multiprocessing
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# Add project root to path (must be at position 0 to take priority over the
# script's own directory which Python auto-inserts.  This ensures that
# ``from utils.general_utils import ...`` resolves to the top-level utils
# package rather than DataSets/utils/).
project_root = Path(__file__).resolve().parent.parent
# Remove script directory if present so DataSets/utils doesn't shadow utils/
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.load_utils import load_gaussian_data_cpu, extract_surface_and_texture_from_path
from utils.misc import fps_gs

# Import shared utilities - canonical implementations
from DataSets.utils.config_utils import (
    load_config,
    get_data_sources,
    get_ring_size_mapping,
    resolve_geodesic_data_path,
    load_geodesic_distances,
    is_multi_source_config,
    is_gaussian_outputs_file,
    resolve_gaussian_outputs,
)

# Import helper functions
from DataSets.utils.training_patches_helpers import (
    compute_gaussian_normals,
    create_train_example,
    generate_training_examples,
    merge_config_with_args,
    compute_scale_stats,
    GaussianData,
    PatchConfig,
)

from GenerateData.utils.data_generation_utils import (
    get_all_points_nbrs_all_rings,
)


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
        help="Path to Gaussian splatting output folder, or a .txt file listing multiple "
             "Gaussian output folders (one per line). When a .txt file is given, samples "
             "from all listed outputs are combined into one dataset."
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
        "--near_source_oversample",
        type=float,
        default=0.0,
        help=(
            "Fraction (0-1) of num_train_points sampled with inverse-geodesic-"
            "distance weighting so that points closer to the source are more "
            "likely to be selected. 0.0 = uniform sampling (default)."
        ),
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
        "--adaptive_target_ring",
        type=int,
        default=None,
        help="Ring level to optimize with adaptive kNN (e.g. 3). Disabled if not set."
    )
    parser.add_argument(
        "--adaptive_target_neighbors",
        type=int,
        default=None,
        help="Desired ring-k neighbor count for adaptive kNN (e.g. 128)."
    )
    parser.add_argument(
        "--adaptive_k_boost",
        type=int,
        default=20,
        help="Maximum boosted k for deficient points in adaptive kNN mode"
    )
    parser.add_argument(
        "--adaptive_max_mean_cut",
        type=float,
        default=5.0,
        help="Stop adaptive binary search when mean cut across all points <= this value"
    )
    parser.add_argument(
        "--adaptive_max_steps",
        type=int,
        default=5,
        help="Maximum number of binary-search steps in adaptive kNN mode"
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
    parser.add_argument(
        "--num_output_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing Gaussian outputs within "
             "a single source. Each worker handles one Gaussian output (load data, "
             "compute KNN neighborhoods, generate examples). Set >1 to parallelize "
             "the heavy KNN computation across outputs. Default: 1 (sequential)."
    )
    parser.add_argument(
        "--fps_target",
        type=int,
        default=0,
        help="FPS downsample the Gaussian cloud to this many points before "
             "computing neighborhoods and generating patches. 0 = no downsampling."
    )
    parser.add_argument(
        "--fps_attributes",
        type=str,
        nargs="+",
        default=["xyz"],
        choices=["xyz", "scale", "opacity", "rotation", "sh", "normals"],
        help="Attributes to use for FPS distance metric (default: xyz only). "
             "Use multiple attributes for Gaussian-aware FPS."
    )
    
    return parser.parse_args()


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
    
    This saves both JSON metadata and a human-readable README.txt file.
    
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
            'source_folder': str(getattr(args, 'gaussian_output', 'Unknown')),
            'iteration': args.iteration if hasattr(args, 'iteration') and args.iteration else 'auto (highest)',
            'num_gaussians': num_gaussians,
            'scale_statistics': scale_stats
        },
        'geodesic_data': {
            'source_file': str(getattr(args, 'geodesic_data', 'Unknown')),
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
            'adaptive_target_ring': getattr(args, 'adaptive_target_ring', None),
            'adaptive_target_neighbors': getattr(args, 'adaptive_target_neighbors', None),
            'adaptive_k_boost': getattr(args, 'adaptive_k_boost', 20),
            'adaptive_max_mean_cut': getattr(args, 'adaptive_max_mean_cut', 5.0),
            'adaptive_max_steps': getattr(args, 'adaptive_max_steps', 5),
        },
        'features': {
            'attributes': args.attributes,
            'use_r1_min_val': args.use_r1_min_val,
            'nn_mean': getattr(args, 'nn_mean', 1.0),
            'mask_constant': args.mask_constant,
            'normalize_per_patch': getattr(args, 'normalize_per_patch', False),
        },
        'fps': {
            'fps_target': getattr(args, 'fps_target', 0),
            'fps_attributes': getattr(args, 'fps_attributes', ['xyz']),
        },
        'output_files': examples_info,
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
        f.write(f"  Ring-1 neighbors: {metadata['neighborhood']['n_neighbors_ring1']}\n\n")
        
        f.write("FEATURES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Attributes: {metadata['features']['attributes']}\n")
        f.write(f"  Use R1 min val: {metadata['features']['use_r1_min_val']}\n")
        f.write(f"  NN mean: {metadata['features']['nn_mean']}\n\n")
        
        fps_info = metadata.get('fps', {})
        if fps_info.get('fps_target', 0) > 0:
            f.write("FPS DOWNSAMPLING:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Target points: {fps_info['fps_target']}\n")
            f.write(f"  FPS attributes: {fps_info['fps_attributes']}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("-" * 40 + "\n")
        for ring, info in metadata['output_files'].items():
            f.write(f"  {ring}:\n")
            f.write(f"    File: {info['filename']}\n")
            f.write(f"    Examples: {info['num_examples']:,}\n")
            f.write(f"    Shape: {info['shape']}\n")
            f.write(f"    Size: {info['size_mb']:.2f} MB\n\n")
    
    print(f"README saved to: {readme_path}")


def process_single_gaussian_output(
    gaussian_output: Path,
    args: argparse.Namespace,
    geodesic_data_path: Optional[str] = None,
    iteration: Optional[int] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], 
           Optional[np.ndarray], Optional[np.ndarray], Dict, Dict, float, np.ndarray]:
    """
    Load data from a single Gaussian output and compute neighborhoods.
    
    Returns all the data needed for generating training examples from this output.
    
    Args:
        gaussian_output: Path to Gaussian splatting output folder
        args: Command line arguments with generation parameters
        geodesic_data_path: Optional explicit geodesic data path
        iteration: Specific iteration to load
    
    Returns:
        Tuple of (geodesic_data, positions, scales, rotations, opacities, normals,
                  sh_features, ring_nbrs_dict, ring1_nbrs, normalization_factor, per_point_dist,
                  surface_name, texture, scale_stats)
    """
    # Load Gaussian data (CPU mode - no CUDA required)
    print(f"\nLoading Gaussian data from: {gaussian_output}")
    load_sh = "sh" in args.attributes
    gaussian_data = load_gaussian_data_cpu(gaussian_output, iteration, load_sh_features=load_sh)
    
    # Extract attributes from GaussianDataCPU (already numpy arrays)
    positions = gaussian_data.get_xyz()
    scales = gaussian_data.get_scaling()
    rotations = gaussian_data.get_rotation()
    opacities = gaussian_data.get_opacity()
    
    # Extract surface name and texture from path
    surface_name, texture = extract_surface_and_texture_from_path(gaussian_output)
    print(f"  Surface: {surface_name if surface_name else 'Unknown'}")
    print(f"  Texture: {texture if texture else 'Not specified'}")
    print(f"  Gaussians: {len(positions)}")
    
    # Compute normals if needed
    normals = None
    if "normals" in args.attributes:
        print("  Computing normals...")
        normals = compute_gaussian_normals(scales, rotations)
    
    # Load SH features if needed
    sh_features = None
    if "sh" in args.attributes:
        sh_features = gaussian_data.get_features()
        if sh_features is not None:
            sh_features = sh_features.reshape(sh_features.shape[0], -1)
    
    # Resolve geodesic data path
    geodesic_path = resolve_geodesic_data_path(gaussian_output, geodesic_data_path)
    
    # Load geodesic distances
    geodesic_data = load_geodesic_distances(geodesic_path)
    
    # ── Optional FPS downsampling ───────────────────────────────────
    fps_target = getattr(args, 'fps_target', 0) or 0
    if fps_target > 0 and fps_target < len(positions):
        fps_attrs = getattr(args, 'fps_attributes', ['xyz'])
        # Protect source Gaussian indices so they are never dropped by FPS
        src_gauss_idx = geodesic_data.get('source_gaussian_indices')
        protected = src_gauss_idx if src_gauss_idx is not None else None
        n_protected = len(protected) if protected is not None else 0
        print(f"\nFPS downsampling: {len(positions)} -> {fps_target} "
              f"(attrs={fps_attrs}, {n_protected} source points protected)")
        fps_idx = fps_gs(
            positions, fps_target, attributes=fps_attrs,
            scales=scales, rotations=rotations, opacities=opacities,
            sh_features=sh_features, normals=normals,
            protected_indices=protected,
            device='cuda',
        )
        # Remap geodesic data to the FPS subset (all sources guaranteed present)
        old_to_new = {int(old): new for new, old in enumerate(fps_idx)}
        geo_dists = geodesic_data['geodesic_distances'][:, fps_idx]
        if src_gauss_idx is not None:
            new_src_gauss = np.array([old_to_new[int(s)] for s in src_gauss_idx])
        else:
            new_src_gauss = None
        geodesic_data = {
            'gaussian_positions': positions[fps_idx],
            'source_indices': geodesic_data.get('source_indices'),
            'source_positions': geodesic_data.get('source_positions'),
            'geodesic_distances': geo_dists,
            'source_gaussian_indices': new_src_gauss,
        }
        # Slice arrays
        positions = positions[fps_idx]
        scales = scales[fps_idx]
        rotations = rotations[fps_idx]
        opacities = opacities[fps_idx]
        if normals is not None:
            normals = normals[fps_idx]
        if sh_features is not None:
            sh_features = sh_features[fps_idx]
        print(f"  After FPS: {len(positions)} Gaussians, {len(geo_dists)} GT sources (all kept)")
    elif fps_target > 0:
        print(f"  FPS target {fps_target} >= cloud size {len(positions)}, skipping")
    
    # Collect scale statistics
    scale_stats = {
        'mean': float(scales.mean()),
        'min': float(scales.min()),
        'max': float(scales.max()),
        'std': float(scales.std())
    }
    
    # Compute neighborhood rings
    print(f"\nComputing neighborhood rings...")
    print(f"  Method: {'Mahalanobis' if args.use_mahalanobis else 'Euclidean'}")
    if getattr(args, 'adaptive_target_ring', None) is not None:
        print(f"  Adaptive kNN: target ring-{args.adaptive_target_ring} ≥ {args.adaptive_target_neighbors}, k_boost={args.adaptive_k_boost}")
    
    ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs, \
    mean_dist, per_point_dist = get_all_points_nbrs_all_rings(
        positions,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=scales if args.use_mahalanobis else None,
        gaussian_rotations=rotations if args.use_mahalanobis else None,
        n_neighbors_ring1=args.n_neighbors,
        adaptive_target_ring=getattr(args, 'adaptive_target_ring', None),
        adaptive_target_neighbors=getattr(args, 'adaptive_target_neighbors', None),
        adaptive_k_boost=getattr(args, 'adaptive_k_boost', 20),
        adaptive_max_mean_cut=getattr(args, 'adaptive_max_mean_cut', 5.0),
        adaptive_max_steps=getattr(args, 'adaptive_max_steps', 5),
    )
    
    ring_nbrs_dict = {1: ring1_nbrs, 2: ring2_nbrs, 3: ring3_nbrs, 4: ring4_nbrs}
    
    return (geodesic_data, positions, scales, rotations, opacities, normals,
            sh_features, ring_nbrs_dict, ring1_nbrs, mean_dist, per_point_dist,
            surface_name, texture, scale_stats)


def _process_output_worker(worker_args):
    """
    Worker function for parallel processing of a single Gaussian output.

    Runs the full pipeline for one output folder:
      load data → compute KNN neighborhoods → generate training examples.

    Must be a module-level function so that :mod:`multiprocessing` can pickle
    it when using the *fork* (default on Linux) or *spawn* start methods.

    Args:
        worker_args: Tuple of
            (gout_path_str, args, geodesic_data_path, iteration,
             rings, gout_idx, total_outputs)

    Returns:
        Dictionary with keys: gout_idx, surface_name, texture,
        num_positions, scale_stats, examples_per_ring.
    """
    (gout_path_str, args, geodesic_data_path, iteration,
     rings, gout_idx, total_outputs) = worker_args

    gout_path = Path(gout_path_str)
    print(f"\n--- [Worker {gout_idx+1}/{total_outputs}] Processing: {gout_path} ---")

    # Heavy step: load Gaussian data + compute Mahalanobis/Euclidean KNN
    (geodesic_data, positions, scales, rotations, opacities, normals,
     sh_features, ring_nbrs_dict, ring1_nbrs, normalization_factor, per_point_dist,
     surface_name, texture, scale_stats) = process_single_gaussian_output(
        gout_path, args, geodesic_data_path, iteration
    )

    # Generate training examples for each ring
    # Use num_workers=1 (sequential) to avoid nested multiprocessing pools
    gdata = GaussianData(
        positions=positions, scales=scales, rotations=rotations,
        opacities=opacities, normals=normals, sh_features=sh_features,
        per_point_nn_distances=per_point_dist,
    )
    examples_per_ring = {}
    for ring in rings:
        print(f"\n  Ring {ring} (output {gout_idx+1}/{total_outputs}):")

        pcfg = PatchConfig(
            ring=ring, normalization_factor=normalization_factor,
            nn_mean=args.nn_mean, attributes=args.attributes,
            use_mahalanobis=args.use_mahalanobis, use_r1_min_val=args.use_r1_min_val,
            mask_attributes=args.mask_attributes, mask_constant=args.mask_constant,
            ring_size_mapping=getattr(args, 'ring_size_mapping', None),
            normalize_per_patch=args.normalize_per_patch,
            disable_outlier_filtering=getattr(args, 'disable_outlier_filtering', False),
            surface_type=surface_name if '_aug_normals' in args.attributes else None,
            outlier_median_multiplier=getattr(args, 'outlier_median_multiplier', 3.0),
            outlier_threshold_floor=getattr(args, 'outlier_threshold_floor', 2.0),
            outlier_hard_cap=getattr(args, 'outlier_hard_cap', 500.0),
            outlier_fallback_multiplier=getattr(args, 'outlier_fallback_multiplier', 5.0),
            outlier_fallback_floor=getattr(args, 'outlier_fallback_floor', 3.0),
            outlier_max_removal_fraction=getattr(args, 'outlier_max_removal_fraction', None),
            mask_outliers_only=getattr(args, 'mask_outliers_only', False),
        )
        examples = generate_training_examples(
            geodesic_data, ring_nbrs_dict, ring1_nbrs,
            args.num_iterations, args.num_sources, args.num_train_points,
            gaussian_data=gdata, patch_config=pcfg,
            seed=args.seed + ring + gout_idx * 1000,
            num_workers=1,  # sequential – avoid nested pools
            near_source_oversample=getattr(args, 'near_source_oversample', 0.0),
        )

        examples_per_ring[ring] = examples
        print(f"    Generated {len(examples)} examples")

    print(f"\n--- [Worker {gout_idx+1}/{total_outputs}] Done: {surface_name} ---")

    return {
        'gout_idx': gout_idx,
        'surface_name': surface_name,
        'texture': texture,
        'num_positions': len(positions),
        'scale_stats': scale_stats,
        'examples_per_ring': examples_per_ring,
    }


def process_single_source(
    source: Dict,
    args: argparse.Namespace,
    config: Optional[Dict] = None
) -> Dict:
    """
    Process a single data source to generate training patches.
    
    If the source's gaussian_output is a .txt file listing multiple Gaussian
    output folders, this processes each folder and combines all examples into
    a single dataset.
    
    Args:
        source: Data source dictionary with gaussian_output, geodesic_data, output_dir, etc.
        args: Command line arguments with generation parameters
        config: Optional full config dictionary
    
    Returns:
        Dictionary with generation results and statistics
    """
    print("\n" + "="*80)
    print(f"Processing source: {source['name']}")
    print("="*80)
    
    # Get paths from source
    gaussian_output_raw = source['gaussian_output']
    output_dir = Path(source['output_dir'])
    iteration = source.get('iteration') or args.iteration
    geodesic_data_path = source.get('geodesic_data')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve gaussian outputs (single path or txt file with multiple paths)
    gaussian_output_paths = resolve_gaussian_outputs(gaussian_output_raw)
    is_multi_output = len(gaussian_output_paths) > 1
    
    if is_multi_output:
        print(f"\nMulti-output mode: {len(gaussian_output_paths)} Gaussian outputs from txt file")
        for i, p in enumerate(gaussian_output_paths):
            print(f"  [{i+1}] {p}")
    
    # Process each Gaussian output and collect examples
    all_examples_per_ring = {ring: [] for ring in args.rings}
    all_surface_names = []
    all_textures = []
    total_gaussians = 0
    combined_scale_stats = {'mean': [], 'min': [], 'max': [], 'std': []}
    
    # Determine number of parallel workers
    num_output_workers = getattr(args, 'num_output_workers', 1) or 1
    num_output_workers = max(1, min(num_output_workers, len(gaussian_output_paths)))
    
    if num_output_workers > 1 and len(gaussian_output_paths) > 1:
        # ------------------------------------------------------------------
        # Parallel processing of Gaussian outputs
        # ------------------------------------------------------------------
        print(f"\nParallel output processing: {num_output_workers} workers "
              f"for {len(gaussian_output_paths)} outputs")
        
        worker_args = [
            (str(gout_path), args, geodesic_data_path, iteration,
             args.rings, gout_idx, len(gaussian_output_paths))
            for gout_idx, gout_path in enumerate(gaussian_output_paths)
        ]
        
        with multiprocessing.Pool(processes=num_output_workers) as pool:
            for result in pool.imap_unordered(_process_output_worker, worker_args):
                idx = result['gout_idx']
                all_surface_names.append(result['surface_name'])
                all_textures.append(result['texture'])
                total_gaussians += result['num_positions']
                for k in combined_scale_stats:
                    combined_scale_stats[k].append(result['scale_stats'][k])
                for ring in args.rings:
                    all_examples_per_ring[ring].append(result['examples_per_ring'][ring])
                print(f"\n  Completed output [{idx+1}/{len(gaussian_output_paths)}]")
    else:
        # ------------------------------------------------------------------
        # Sequential processing (original behaviour)
        # ------------------------------------------------------------------
        for gout_idx, gout_path in enumerate(gaussian_output_paths):
            gout_path = Path(gout_path)
            
            if is_multi_output:
                print(f"\n--- Gaussian output [{gout_idx+1}/{len(gaussian_output_paths)}]: {gout_path} ---")
            
            # Load and process this Gaussian output
            (geodesic_data, positions, scales, rotations, opacities, normals,
             sh_features, ring_nbrs_dict, ring1_nbrs, normalization_factor, per_point_dist,
             surface_name, texture, scale_stats) = process_single_gaussian_output(
                gout_path, args, geodesic_data_path, iteration
            )
            
            all_surface_names.append(surface_name)
            all_textures.append(texture)
            total_gaussians += len(positions)
            for k in combined_scale_stats:
                combined_scale_stats[k].append(scale_stats[k])
            
            # Generate training examples for each ring from this Gaussian output
            gdata = GaussianData(
                positions=positions, scales=scales, rotations=rotations,
                opacities=opacities, normals=normals, sh_features=sh_features,
                per_point_nn_distances=per_point_dist,
            )
            for ring in args.rings:
                print(f"\n  Ring {ring}:")

                pcfg = PatchConfig(
                    ring=ring, normalization_factor=normalization_factor,
                    nn_mean=args.nn_mean, attributes=args.attributes,
                    use_mahalanobis=args.use_mahalanobis, use_r1_min_val=args.use_r1_min_val,
                    mask_attributes=args.mask_attributes, mask_constant=args.mask_constant,
                    ring_size_mapping=getattr(args, 'ring_size_mapping', None),
                    normalize_per_patch=args.normalize_per_patch,
                    disable_outlier_filtering=getattr(args, 'disable_outlier_filtering', False),
                    surface_type=surface_name if '_aug_normals' in args.attributes else None,
                    outlier_median_multiplier=getattr(args, 'outlier_median_multiplier', 3.0),
                    outlier_threshold_floor=getattr(args, 'outlier_threshold_floor', 2.0),
                    outlier_hard_cap=getattr(args, 'outlier_hard_cap', 500.0),
                    outlier_fallback_multiplier=getattr(args, 'outlier_fallback_multiplier', 5.0),
                    outlier_fallback_floor=getattr(args, 'outlier_fallback_floor', 3.0),
                    outlier_max_removal_fraction=getattr(args, 'outlier_max_removal_fraction', None),
                    mask_outliers_only=getattr(args, 'mask_outliers_only', False),
                )
                examples = generate_training_examples(
                    geodesic_data, ring_nbrs_dict, ring1_nbrs,
                    args.num_iterations, args.num_sources, args.num_train_points,
                    gaussian_data=gdata, patch_config=pcfg,
                    seed=args.seed + ring + gout_idx * 1000,
                    near_source_oversample=getattr(args, 'near_source_oversample', 0.0),
                )
                
                all_examples_per_ring[ring].append(examples)
                print(f"    Generated {len(examples)} examples")
    
    # Aggregate scale stats
    final_scale_stats = {
        'mean': float(np.mean(combined_scale_stats['mean'])),
        'min': float(np.min(combined_scale_stats['min'])),
        'max': float(np.max(combined_scale_stats['max'])),
        'std': float(np.mean(combined_scale_stats['std']))
    }
    
    # Combine and save examples for each ring
    examples_info = {}
    
    for ring in args.rings:
        print(f"\n--- Ring {ring} (combined) ---")
        
        combined_examples = np.vstack(all_examples_per_ring[ring])
        
        # Save examples
        output_name = f"gaussian_examples_ring{ring}_n{len(combined_examples)}.npy"
        output_path = output_dir / output_name
        np.save(output_path, combined_examples)
        
        if is_multi_output:
            per_output_counts = [len(ex) for ex in all_examples_per_ring[ring]]
            print(f"  Combined {len(combined_examples)} examples from {len(gaussian_output_paths)} outputs")
            print(f"  Per-output counts: {per_output_counts}")
        else:
            print(f"  Saved {len(combined_examples)} examples")
        print(f"  Saved to: {output_path}")
        
        examples_info[f"ring_{ring}"] = {
            'filename': output_name,
            'num_examples': len(combined_examples),
            'shape': str(combined_examples.shape),
            'size_mb': combined_examples.nbytes / 1e6,
            'attributes': args.attributes
        }
        if is_multi_output:
            examples_info[f"ring_{ring}"]['per_output_counts'] = [
                len(ex) for ex in all_examples_per_ring[ring]
            ]
    
    # Determine surface name for metadata
    if is_multi_output:
        surface_name = f"Combined ({', '.join(filter(None, all_surface_names))})"
        texture = ', '.join(filter(None, all_textures)) or None
    else:
        surface_name = all_surface_names[0] if all_surface_names else None
        texture = all_textures[0] if all_textures else None
    
    # Save source-specific metadata
    save_metadata(
        output_dir, args, total_gaussians, final_scale_stats, examples_info,
        surface_name, texture
    )
    
    # Also save source config for loading
    source_config = {
        'output_dir': str(output_dir),
        'gaussian_output': str(gaussian_output_raw),
        'gaussian_output_paths': [str(p) for p in gaussian_output_paths] if is_multi_output else None,
        'geodesic_data': str(geodesic_data_path) if geodesic_data_path else None,
        'attributes': args.attributes,
        'rings': args.rings,
        'use_mahalanobis': args.use_mahalanobis,
        'use_r1_min_val': args.use_r1_min_val,
        'ring_size_mapping': getattr(args, 'ring_size_mapping', None),
        'mask_constant': args.mask_constant,
        'surface_name': surface_name,
        'texture': texture,
        'num_gaussian_outputs': len(gaussian_output_paths),
        'disable_outlier_filtering': getattr(args, 'disable_outlier_filtering', False),
    }
    
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(source_config, f, default_flow_style=False)
    
    print(f"\nSource config saved to: {config_path}")
    
    return {
        'name': source['name'],
        'output_dir': str(output_dir),
        'num_gaussians': total_gaussians,
        'num_gaussian_outputs': len(gaussian_output_paths),
        'examples_info': examples_info,
        'surface_name': surface_name,
        'texture': texture
    }


def main():
    args = parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        args = merge_config_with_args(config, args)
        print(f"Configuration loaded successfully")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Handle deprecated arguments
    if args.add_normals and "normals" not in args.attributes:
        print("Warning: --add_normals is deprecated. Use --attributes normals instead.")
        args.attributes.append("normals")
    if args.add_euclidean_distance and "euclidean_distances" not in args.attributes:
        print("Warning: --add_euclidean_distance is deprecated. Use --attributes euclidean_distances instead.")
        args.attributes.append("euclidean_distances")
    
    print("="*80)
    print("Gaussian Training Example Generation")
    print("="*80)
    print(f"Attributes: {', '.join(args.attributes)}")
    print(f"Rings: {args.rings}")
    
    # Check if multi-source config
    if config and is_multi_source_config(config):
        # Multi-source mode: process each source independently
        sources = get_data_sources(config)
        print(f"\nMulti-source mode: {len(sources)} sources to process")
        
        all_results = []
        for source in sources:
            result = process_single_source(source, args, config)
            all_results.append(result)
        
        # Save combined summary
        base_output_dir = Path(config.get('output_dir', 'TrainData/datasets/gaussian_patches'))
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'multi_source': True,
            'num_sources': len(sources),
            'sources': all_results,
            'shared_config': {
                'attributes': args.attributes,
                'rings': args.rings,
                'use_mahalanobis': args.use_mahalanobis,
                'use_r1_min_val': args.use_r1_min_val
            }
        }
        
        summary_path = base_output_dir / 'multi_source_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Multi-source generation complete!")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}")
        
    else:
        # Single-source mode (backward compatible)
        if args.gaussian_output is None:
            raise ValueError("--gaussian_output must be specified either via command line or config file")
        if args.output_dir is None:
            raise ValueError("--output_dir must be specified either via command line or config file")
        
        source = {
            'name': 'default',
            'gaussian_output': args.gaussian_output,
            'output_dir': args.output_dir,
            'geodesic_data': args.geodesic_data,
            'iteration': args.iteration
        }
        
        process_single_source(source, args, config)
        
        print(f"\n{'='*80}")
        print("Training example generation complete!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
