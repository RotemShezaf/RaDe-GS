#!/usr/bin/env python3
"""
Shared utility functions for Gaussian training patch generation and loading.

This module contains common functions used by both:
- create_gaussian_training_patches.py (data generation)
- gaussian_dataset.py (data loading)

For attribute size calculations and masked entry generation, see:
- DataSets/utils/data_transformation_utils.py (canonical implementations)

Supports three configuration modes:
1. Single-source config: One gaussian_output path → one dataset
2. Single-source config with gaussian_outputs_file: A .txt file listing multiple
   gaussian_output paths → one unified dataset with samples from all listed outputs
3. Combined data config (data_sources): Multiple independent datasets combined
   via CombinedGaussianPatchDataset for training diversity
"""

import numpy as np
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Import canonical implementations from data_transformation_utils
from DataSets.utils.data_transformation_utils import (
    get_attribute_size,
    get_entry_size,
    get_point_feature_size,
    get_masked_entry
)


# =============================================================================
# Configuration Loading and Merging
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary with configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict, required_fields: List[str] = None) -> bool:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration dictionary
        required_fields: List of required field names (default: common required fields)
    
    Returns:
        True if valid
        
    Raises:
        ValueError if validation fails
    """
    if required_fields is None:
        required_fields = ['output_dir', 'attributes', 'rings']
    
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    
    return True


# =============================================================================
# Gaussian Outputs File (.txt) Support
# =============================================================================

def is_gaussian_outputs_file(path: Optional[Union[str, Path]]) -> bool:
    """
    Check if a path points to a .txt file listing multiple Gaussian output paths.
    
    Args:
        path: Path to check (can be None)
    
    Returns:
        True if path ends with .txt
    """
    if path is None:
        return False
    return str(path).endswith('.txt')


def load_gaussian_outputs_file(txt_path: Union[str, Path]) -> List[str]:
    """
    Load a list of Gaussian output paths from a .txt file.
    
    The file should contain one path per line. Empty lines and lines starting
    with '#' are ignored.
    
    Args:
        txt_path: Path to the .txt file
    
    Returns:
        List of Gaussian output paths (as strings)
    
    Raises:
        FileNotFoundError if the txt file doesn't exist
        ValueError if the txt file is empty or contains no valid paths
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Gaussian outputs file not found: {txt_path}")
    
    paths = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                paths.append(line)
    
    if not paths:
        raise ValueError(f"No valid paths found in {txt_path}")
    
    return paths


def resolve_gaussian_outputs(gaussian_output: Optional[Union[str, Path]]) -> List[str]:
    """
    Resolve gaussian_output to a list of paths.
    
    If gaussian_output is a .txt file, load paths from it.
    Otherwise, return it as a single-element list.
    
    Args:
        gaussian_output: Path to Gaussian output folder or .txt file listing multiple
    
    Returns:
        List of Gaussian output paths
    """
    if gaussian_output is None:
        return []
    
    if is_gaussian_outputs_file(gaussian_output):
        return load_gaussian_outputs_file(gaussian_output)
    else:
        return [str(gaussian_output)]


# =============================================================================
# Multi-file Dataset Support
# =============================================================================

def get_data_sources(config: Dict) -> List[Dict]:
    """
    Get list of data sources from config.
    
    Supports both single-source (backward compatible) and multi-source configs.
    In both modes, gaussian_output can be either a direct path or a .txt file
    listing multiple Gaussian output paths.
    
    Single-source config:
        gaussian_output: "path/to/output"       # single path
        # OR
        gaussian_output: "path/to/outputs.txt"   # txt file with multiple paths
        geodesic_data: "path/to/geodesic.npz"
        output_dir: "path/to/patches"
    
    Multi-source (combined data) config:
        data_sources:
          - name: "paraboloid_blue"
            gaussian_output: "path/to/output1"   # single or .txt file
            geodesic_data: "path/to/geodesic1.npz"  # Optional, auto-detect if null
            output_dir: "path/to/patches1"  # Per-source output directory
            weight: 1.0
          - name: "saddle_blue"
            gaussian_output: "path/to/output2"
            output_dir: "path/to/patches2"
            weight: 0.5
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of data source dictionaries, each with:
        - gaussian_output: Path to Gaussian splatting output
        - geodesic_data: Path to geodesic data (or None for auto-detect)
        - output_dir: Path to save/load training patches for this source
        - weight: Sampling weight (default 1.0)
        - name: Name for the source (for display/logging)
        - iteration: Specific iteration to use (or None for highest)
    """
    if 'data_sources' in config:
        # Multi-source config
        sources = []
        base_output_dir = config.get('output_dir', 'TrainData/datasets/gaussian_patches')
        
        for i, source in enumerate(config['data_sources']):
            name = source.get('name', f'source_{i}')
            # Per-source output_dir, or derive from base output_dir + name
            source_output_dir = source.get('output_dir')
            if source_output_dir is None:
                source_output_dir = str(Path(base_output_dir) / name)
            
            sources.append({
                'gaussian_output': source.get('gaussian_output'),
                'geodesic_data': source.get('geodesic_data'),
                'output_dir': source_output_dir,
                'weight': source.get('weight', 1.0),
                'name': name,
                'iteration': source.get('iteration')
            })
        return sources
    else:
        # Single-source config (backward compatible)
        return [{
            'gaussian_output': config.get('gaussian_output'),
            'geodesic_data': config.get('geodesic_data'),
            'output_dir': config.get('output_dir'),
            'weight': 1.0,
            'name': 'default',
            'iteration': config.get('iteration')
        }]


def is_multi_source_config(config: Dict) -> bool:
    """Check if a config is multi-source."""
    return 'data_sources' in config


# Shared settings that must be identical across all sources in a multi-source config
SHARED_CONFIG_KEYS = [
    'attributes',
    'rings',
    'ring_size_mapping',
    'use_mahalanobis',
    'use_r1_min_val',
    'mask_constant',
    'nn_mean',
    'normalize_per_patch',
]


def merge_source_config(master_config: Dict, source: Dict) -> Dict:
    """
    Merge shared settings from master config with per-source settings.
    
    This creates a complete config dict for a single source by combining:
    - Shared settings from master_config (attributes, rings, etc.)
    - Per-source settings (gaussian_output, output_dir, etc.)
    
    The resulting dict can be passed directly to GaussianPatchDataset.
    
    Args:
        master_config: The multi-source config containing shared settings
        source: A single source dict from data_sources list
    
    Returns:
        Complete config dict suitable for GaussianPatchDataset
    """
    merged = {}
    
    # Copy all shared settings from master config
    for key in SHARED_CONFIG_KEYS:
        if key in master_config:
            merged[key] = master_config[key]
    
    # Copy additional shared settings that may exist
    for key in ['dataset', 'num_iterations', 'num_sources', 'num_train_points', 'seed', 'n_neighbors']:
        if key in master_config:
            merged[key] = master_config[key]
    
    # Override with per-source settings
    merged['output_dir'] = source.get('output_dir')
    merged['gaussian_output'] = source.get('gaussian_output')
    merged['geodesic_data'] = source.get('geodesic_data')
    merged['source_name'] = source.get('name', 'unknown')
    merged['weight'] = source.get('weight', 1.0)
    merged['iteration'] = source.get('iteration')
    # Per-source surface_type (used by GaussianPatchSurfacePerturb)
    if 'surface_type' in source:
        merged['surface_type'] = source['surface_type']
    
    return merged


def validate_shared_settings(configs: List[Dict], setting_keys: List[str] = None) -> None:
    """
    Validate that shared settings are identical across all configs.
    
    Args:
        configs: List of config dictionaries to compare
        setting_keys: Keys to check (default: SHARED_CONFIG_KEYS)
    
    Raises:
        ValueError if any shared setting differs between configs
    """
    if len(configs) < 2:
        return
    
    if setting_keys is None:
        setting_keys = SHARED_CONFIG_KEYS
    
    reference = configs[0]
    
    for i, config in enumerate(configs[1:], start=2):
        for key in setting_keys:
            ref_val = reference.get(key)
            cfg_val = config.get(key)
            
            if ref_val != cfg_val:
                raise ValueError(
                    f"Shared setting '{key}' differs between sources:\n"
                    f"  Source 1: {ref_val}\n"
                    f"  Source {i}: {cfg_val}\n"
                    f"All sources must have identical values for: {setting_keys}"
                )


def get_all_source_output_dirs(config: Dict) -> List[Path]:
    """
    Get list of output directories from all data sources.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of Path objects to output directories
    """
    sources = get_data_sources(config)
    return [Path(s['output_dir']) for s in sources if s['output_dir']]


def get_output_files(data_dir: Union[str, Path], ring: int = None) -> List[Path]:
    """
    Get list of output data files from a data directory.
    
    Args:
        data_dir: Directory containing .npy data files
        ring: Specific ring to filter for (None for all)
    
    Returns:
        List of Path objects to data files
    """
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob('gaussian_examples_ring*.npy'))
    
    if ring is not None:
        # Filter for specific ring
        filtered = []
        for file_path in data_files:
            filename = file_path.stem
            parts = filename.split('_')
            ring_idx = next((i for i, p in enumerate(parts) if p.startswith('ring')), None)
            if ring_idx is not None:
                ring_num = int(parts[ring_idx].replace('ring', ''))
                if ring_num == ring:
                    filtered.append(file_path)
        return filtered
    
    return data_files


# =============================================================================
# Ring Size Mapping
# =============================================================================

def get_ring_size_mapping(ring: int, use_mahalanobis: bool, 
                          custom_mapping: Optional[Dict] = None) -> int:
    """
    Get expected maximum number of neighbors for a given ring.
    
    Args:
        ring: Ring number (1-4)
        use_mahalanobis: Whether Mahalanobis distance is used
        custom_mapping: Optional custom ring size mapping from config
    
    Returns:
        Maximum number of neighbors for the ring
    """
    if custom_mapping:
        method = 'mahalanobis' if use_mahalanobis else 'euclidean'
        if method in custom_mapping and ring in custom_mapping[method]:
            return custom_mapping[method][ring]
    
    # Default mappings
    if use_mahalanobis:
        mapping = {1: 25, 2: 90, 3: 250, 4: 600}
    else:
        mapping = {1: 22, 2: 75, 3: 200, 4: 500}
    
    return mapping.get(ring, 100)


def get_ring_size_from_config(config: Dict, ring: int) -> int:
    """
    Get ring size from config, handling both methods.
    
    Args:
        config: Configuration dictionary
        ring: Ring number
    
    Returns:
        Maximum number of neighbors
    """
    use_mahalanobis = config.get('use_mahalanobis', False)
    custom_mapping = config.get('ring_size_mapping')
    return get_ring_size_mapping(ring, use_mahalanobis, custom_mapping)


# Note: get_attribute_size, get_entry_size, get_point_feature_size, and get_masked_entry
# are imported from DataSets/utils/data_transformation_utils.py to avoid duplication


# =============================================================================
# Metadata Handling
# =============================================================================

def load_metadata(data_dir: Union[str, Path]) -> Optional[Dict]:
    """
    Load generation metadata from a data directory.
    
    Args:
        data_dir: Directory containing the metadata file
    
    Returns:
        Metadata dictionary or None if not found
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'generation_metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def save_metadata(
    output_dir: Union[str, Path],
    config: Dict,
    num_gaussians: int,
    scale_stats: Dict,
    examples_info: Dict,
    surface_name: Optional[str] = None,
    texture: Optional[str] = None,
    data_sources: Optional[List[Dict]] = None
) -> None:
    """
    Save metadata about the data generation process.
    
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        num_gaussians: Number of Gaussians in the dataset
        scale_stats: Statistics about Gaussian scales
        examples_info: Information about generated examples per ring
        surface_name: Name of the surface (if applicable)
        texture: Texture name (if applicable)
        data_sources: List of data sources (for multi-source configs)
    """
    output_dir = Path(output_dir)
    
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
        'data_sources': data_sources if data_sources else [{
            'gaussian_output': config.get('gaussian_output'),
            'geodesic_data': config.get('geodesic_data')
        }],
        'gaussian_data': {
            'num_gaussians': num_gaussians,
            'scale_statistics': scale_stats
        },
        'generation_parameters': {
            'num_iterations': config.get('num_iterations'),
            'num_sources_per_iteration': config.get('num_sources'),
            'num_train_points_per_iteration': config.get('num_train_points'),
            'rings': config.get('rings'),
            'seed': config.get('seed')
        },
        'neighborhood': {
            'method': 'Mahalanobis' if config.get('use_mahalanobis') else 'Euclidean',
            'n_neighbors_ring1': config.get('n_neighbors')
        },
        'features': {
            'attributes': config.get('attributes'),
            'use_r1_min_val': config.get('use_r1_min_val', False),
            'nn_mean': config.get('nn_mean', 1.0),
            'mask_constant': config.get('mask_constant', -10.0),
            'normalize_per_patch': config.get('normalize_per_patch', False)
        },
        'output_files': examples_info
    }
    
    # Save as JSON
    metadata_path = output_dir / 'generation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")


# =============================================================================
# Geodesic Data Path Resolution
# =============================================================================

def resolve_geodesic_data_path(
    gaussian_output: Union[str, Path],
    geodesic_data: Optional[Union[str, Path]] = None
) -> Path:
    """
    Resolve the path to geodesic data file.
    
    If geodesic_data is None, automatically looks for:
    {gaussian_output}/geodesic_distance/gt_geodesic.npz
    
    Args:
        gaussian_output: Path to Gaussian output folder
        geodesic_data: Optional explicit path to geodesic data
    
    Returns:
        Path to geodesic data file
        
    Raises:
        FileNotFoundError if file not found
    """
    gaussian_output = Path(gaussian_output)
    
    if geodesic_data is not None:
        geodesic_path = Path(geodesic_data)
    else:
        geodesic_path = gaussian_output / 'geodesic_distance' / 'gt_geodesic.npz'
    
    if not geodesic_path.exists():
        raise FileNotFoundError(
            f"Geodesic data not found at: {geodesic_path}\n"
            f"Run compute_gaussian_geodesic_distances.py first."
        )
    
    return geodesic_path


def load_geodesic_distances(geodesic_path: Union[str, Path]) -> Dict:
    """
    Load precomputed geodesic distance data.
    
    Args:
        geodesic_path: Path to NPZ file with geodesic distances
    
    Returns:
        Dictionary with geodesic data
    """
    geodesic_path = Path(geodesic_path)
    print(f"Loading geodesic distances from: {geodesic_path}")
    
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


# =============================================================================
# Multi-file Dataset Configuration Template
# =============================================================================

def create_multi_source_config_template() -> Dict:
    """
    Create a template configuration for multi-source datasets.
    
    Returns:
        Template configuration dictionary
    """
    return {
        'output_dir': 'TrainData/datasets/gaussian_patches/combined',
        
        # Multiple data sources
        'data_sources': [
            {
                'name': 'paraboloid',
                'gaussian_output': 'path/to/paraboloid/output',
                'geodesic_data': None,  # Auto-detect
                'weight': 1.0,
                'iteration': None
            },
            {
                'name': 'saddle',
                'gaussian_output': 'path/to/saddle/output',
                'geodesic_data': None,
                'weight': 1.0,
                'iteration': None
            }
        ],
        
        # Shared parameters
        'num_iterations': 1000,
        'num_sources': 3,
        'num_train_points': 15,
        'seed': 42,
        
        'use_mahalanobis': True,
        'n_neighbors': 10,
        'normalize_per_patch': True,
        
        'rings': [2, 3],
        
        'ring_size_mapping': {
            'euclidean': {2: 40, 3: 150, 4: 300},
            'mahalanobis': {2: 90, 3: 250, 4: 600}
        },
        
        'attributes': ['xyz'],
        'nn_mean': 1.0,
        'use_r1_min_val': True,
        'mask_constant': -10.0,
        
        'dataset': {
            'name': 'Combined Polynomial Surfaces',
            'type': 'Multi-source',
            'description': 'Combined training examples from multiple surfaces'
        }
    }
