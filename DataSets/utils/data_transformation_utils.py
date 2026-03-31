"""
Data transformation utilities for Gaussian patch processing.

This module contains utility functions for:
- Attribute size calculations
- Masked entry generation for padding
- Rotation transformations (canonical rotation, quaternion operations)

These functions are the canonical implementations used throughout the codebase.
"""

import math
import torch
import numpy as np
from typing import List


# =============================================================================
# Attribute Size Calculations
# =============================================================================

def get_attribute_size(attr: str) -> int:
    """
    Get the size (number of values) for an attribute.
    
    Args:
        attr: Attribute name
    
    Returns:
        Number of values for the attribute
    """
    sizes = {
        'xyz': 3,
        'normals': 3,
        '_aug_normals': 3,
        'scale': 3,
        'sh': 3,  # Only first 3 SH coefficients typically used
        'rotation': 4,
        'opacity': 1,
        'euclidean_distances': 1,
        'geodesic_distance': 1
    }
    return sizes.get(attr, 1)


def get_entry_size(attributes: List[str], include_geodesic: bool = True) -> int:
    """
    Calculate the size of a single Gaussian entry with given attributes.
    
    This is the canonical implementation used for:
    - Data generation (create_gaussian_training_patches.py)
    - Dataset loading (gaussian_dataset.py)
    - Data augmentation (dropout, masking)
    
    Args:
        attributes: List of attribute names (e.g., ['xyz', 'normals', 'opacity'])
        include_geodesic: Whether to include geodesic distance (+1). Default True.
            Set to False when computing point features (no target distance).
        
    Returns:
        Size in number of features
        
    Example:
        >>> get_entry_size(['xyz', 'opacity'])  # 3 + 1 + 1 = 5
        5
        >>> get_entry_size(['xyz', 'opacity'], include_geodesic=False)  # 3 + 1 = 4
        4
    """
    entry_size = sum(get_attribute_size(attr) for attr in attributes)
    if include_geodesic:
        entry_size += 1  # +1 for geodesic distance
    return entry_size


def get_point_feature_size(attributes: List[str]) -> int:
    """
    Get the size of point features (no geodesic distance).
    
    Convenience wrapper around get_entry_size with include_geodesic=False.
    
    Args:
        attributes: List of attribute names
    
    Returns:
        Size of point feature vector
    """
    return get_entry_size(attributes, include_geodesic=False)


# =============================================================================
# Masked Entry Generation
# =============================================================================

def get_masked_entry(attributes: List[str], mask_constant: float = -10.0) -> torch.Tensor:
    """
    Get a masked Gaussian point entry for padding invalid/missing neighbors.
    
    Used in data augmentation (dropout, max neighbors) and padding during
    data generation when a ring has fewer neighbors than the maximum.
    
    Masked values are set as follows:
    - xyz: mask_constant (typically -10.0)
    - normals: 0.0 (neutral direction)
    - opacity: 0.0 (fully transparent)
    - scale: 1.0 (unit scale)
    - rotation: [1, 0, 0, 0] (identity quaternion)
    - sh: transformation of -0.5 (dark color)
    - euclidean_distances: mask_constant
    - geodesic_distance: mask_constant
    
    Args:
        attributes: List of attribute names
        mask_constant: Value used for masked entries (default: -10.0)
    
    Returns:
        Tensor of shape (1, entry_size) with masked values
    """
    entry_size = get_entry_size(attributes)
    masked_entry = torch.zeros(1, entry_size, dtype=torch.float32)
    
    attr_index = 0
    for attr in attributes:
        if attr == "xyz":
            masked_entry[0, attr_index:attr_index+3] = mask_constant
            attr_index += 3
        elif attr == "normals":
            # Keep normals as zero (neutral direction)
            attr_index += 3
        elif attr == "_aug_normals":
            # Augmentation normals: zero = no normal info
            attr_index += 3
        elif attr == "opacity":
            masked_entry[0, attr_index] = 0.0
            attr_index += 1
        elif attr == "scale":
            masked_entry[0, attr_index:attr_index+3] = 1.0
            attr_index += 3
        elif attr == "rotation":
            # Unit quaternion for masking (identity rotation)
            masked_entry[0, attr_index:attr_index+4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            attr_index += 4
        elif attr == "sh":
            masked_entry[0, attr_index:attr_index+3] = -0.5 * 2 / math.sqrt(3)
            attr_index += 3
        elif attr == "euclidean_distances":
            masked_entry[0, attr_index] = mask_constant
            attr_index += 1
    
    # Geodesic distance (always last)
    masked_entry[0, -1] = mask_constant
    
    return masked_entry


# =============================================================================
# Rotation Transformations
# ============================================================================= 

def canonical_rotation(A, b=None):
    """
    Compute a canonical rotation matrix that aligns vector 'a' with vector 'b'.
    
    This function computes a 3x3 rotation matrix R such that R @ a points in the direction of b.
    The rotation is "canonical" in the sense that it's constructed systematically using
    the Gram-Schmidt process and Rodrigues-like rotation formula.
    
    Mathematical background:
    - The rotation is constructed by building an orthonormal frame F using vectors a and b
    - A 2D rotation matrix G is constructed in the plane spanned by a and b
    - The final rotation R = F @ G @ F^(-1) transforms a towards b
    
    Args:
        A: torch.Tensor of shape (3,) - source vector to rotate
        b: torch.Tensor of shape (3,) or None - target direction vector
           If None, defaults to [0, 1, 0] (y-axis)
    
    Returns:
        r: torch.Tensor of shape (3, 3) - rotation matrix that rotates a towards b
    
    Example:
        >>> a = torch.tensor([1., 0., 0.])  # x-axis
        >>> b = torch.tensor([0., 1., 0.])  # y-axis
        >>> R = canonical_rotation_torch(a, b)
        >>> rotated = R @ a  # Should point towards y-axis
    """
    # Default target direction is y-axis if not specified
    if b is None:
        b = torch.tensor([0., 1., 0.], dtype=A.dtype, device=A.device)
    
    # Handle degenerate case: zero vector cannot be rotated
    if torch.norm(A) == 0:
        return torch.eye(3, dtype=A.dtype, device=A.device)
    
    # Normalize source vector
    a = A / torch.norm(A)
    
    # Handle case where b is already aligned with a (or opposite)
    # Check if the component of b perpendicular to a is zero
    b_perp = b - torch.dot(a, b) * a  # Projection of b perpendicular to a
    if torch.norm(b_perp) == 0:
        return torch.eye(3, dtype=a.dtype, device=a.device)

    # Compute rotation parameters
    ab_dot = torch.dot(a, b)              # cos(theta) where theta is angle between a and b
    ab_cross_norm = torch.norm(torch.linalg.cross(a, b))  # sin(theta) * |a| * |b|
    
    # Construct 2D rotation matrix G in the plane spanned by a and b
    # This is essentially a Givens rotation in 2D
    # G rotates the first basis vector towards the projection of b onto the (a, b_perp) plane
    g = torch.tensor([[ab_dot, -ab_cross_norm, 0.],
                      [ab_cross_norm, ab_dot, 0.],
                      [0., 0., 1.]], dtype=a.dtype, device=a.device)

    # Construct orthonormal frame F using Gram-Schmidt process
    # F[:, 0] = normalized a (first basis vector)
    # F[:, 1] = normalized component of b perpendicular to a (second basis vector)
    # F[:, 2] = cross product (third basis vector, perpendicular to both)
    f = torch.zeros((3, 3), dtype=a.dtype, device=a.device)
    f[:, 0] = a
    f[:, 1] = (b - torch.dot(a, b) * a) / torch.norm(b - torch.dot(a, b) * a)
    f[:, 2] = torch.linalg.cross(b, a)

    # Apply similarity transformation: R = F @ G @ F^(-1)
    # This rotates from standard basis -> F basis -> apply 2D rotation -> back to standard basis
    # The result is a rotation in 3D that aligns a with b
    r = f @ g @ torch.linalg.inv(f)

    return r  

def qvec2rotmat(qvec):
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = torch.tensor([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions q1 * q2.
        Quaternions are in (w, x, y, z) format.
        
        Args:
            q1: (N, 4) tensor of quaternions
            q2: (N, 4) tensor of quaternions or (4,) single quaternion
            
        Returns:
            (N, 4) tensor of quaternion products
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        
        if q2.dim() == 1:
            w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        else:
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)