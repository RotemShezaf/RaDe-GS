"""
DataSets utilities module.

This module contains shared utilities for:
- Configuration loading and management
- Data transformation utilities
- Rotation conversions
"""

from .config_utils import (
    load_config,
    save_config,
    validate_config,
    get_data_sources,
    get_output_files,
    get_ring_size_mapping,
    get_ring_size_from_config,
    load_metadata,
    save_metadata,
    resolve_geodesic_data_path,
    load_geodesic_distances,
    create_multi_source_config_template,
    is_multi_source_config,
    get_all_source_output_dirs
)

from .data_transformation_utils import (
    get_attribute_size,
    get_entry_size,
    get_point_feature_size,
    get_masked_entry,
    canonical_rotation,
    qvec2rotmat,
    rotmat2qvec,
    quaternion_multiply
)

__all__ = [
    # Config utilities
    'load_config',
    'save_config',
    'validate_config',
    'get_data_sources',
    'get_output_files',
    'get_ring_size_mapping',
    'get_ring_size_from_config',
    'get_attribute_size',
    'get_entry_size',
    'get_point_feature_size',
    'get_masked_entry',
    'load_metadata',
    'save_metadata',
    'resolve_geodesic_data_path',
    'load_geodesic_distances',
    'create_multi_source_config_template',
    'is_multi_source_config',
    'get_all_source_output_dirs',
    # Data transformation utilities
    'canonical_rotation',
    'qvec2rotmat',
    'rotmat2qvec',
    'quaternion_multiply',
]
