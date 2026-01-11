"""
Utility functions for computing neighborhood rings on point clouds and meshes.

Supports both regular Euclidean distance and Mahalanobis distance for Gaussians.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Tuple
import torch

from utils.general_utils import  strip_symmetric, build_scaling_rotation


def mahalanobis_distance(points, center, cov_inv):
    """Compute Mahalanobis distance from points to a center.
    
    Args:
        points: (M, 3) array of points
        center: (3,) center point
        cov_inv: (3, 3) inverse covariance matrix
    
    Returns:
        (M,) distances
    """
    diff = points - center
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return distances


def ring1_neighbors_gaussians(
    vertices: np.ndarray,
    n_neighbors: int = 16,
    use_mahalanobis: bool = True,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None
) -> Dict[int, np.ndarray]:
    """
    Find ring-1 (immediate) neighbors for each point in a point cloud.
    
    Args:
        vertices: (N, 3) array of vertex positions
        n_neighbors: Number of neighbors to find for each point
        use_mahalanobis: If True, use Mahalanobis distance based on Gaussian covariance
        gaussian_scales: (N, 3) array of Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (N, 4) array of Gaussian rotations as quaternions (required if use_mahalanobis=True)
    
    Returns:
        Dictionary mapping vertex index to array of neighbor indices (excluding self)
    """
    num_v = vertices.shape[0]
    
    if use_mahalanobis:
        if gaussian_scales is None or gaussian_rotations is None:
            raise ValueError("gaussian_scales and gaussian_rotations must be provided when use_mahalanobis=True")
        
        # Compute Mahalanobis distance-based neighbors
        ring1_nbrs = _compute_mahalanobis_neighbors(
            vertices, gaussian_scales, gaussian_rotations, n_neighbors
        )
    else:
        # Use standard Euclidean distance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(vertices)
        _, indices = nbrs.kneighbors(vertices)
        ring1_nbrs = {index: indices[index, 1:] for index in range(num_v)}
    
    return ring1_nbrs


def _compute_mahalanobis_neighbors(
    vertices: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    k: int
) -> Dict[int, np.ndarray]:
    """
    Compute k-nearest neighbors using Mahalanobis distance based on Gaussian covariance.
    
    The Mahalanobis distance accounts for the anisotropic shape of each Gaussian,
    giving preference to neighbors that are close in the principal directions of the Gaussian.
    
    Args:
        vertices: (N, 3) array of Gaussian center positions
        scales: (N, 3) array of Gaussian scales
        rotations: (N, 4) array of Gaussian rotations as quaternions [w, x, y, z]
        k: Number of neighbors to find
    
    Returns:
        Dictionary mapping vertex index to array of k nearest neighbor indices
    """
    num_points = len(vertices)
    neighbors = {}
    
    # Build covariance matrices for each Gaussian
    covariances = _build_covariance_matrices(scales, rotations)
    
    for i in range(num_points):
        # Compute Mahalanobis distance from point i to all other points
        # D^2 = (x - μ)^T Σ^(-1) (x - μ)
        diff = vertices - vertices[i]  # (N, 3)
        
        # Use the covariance of the source point for distance computation
        try:
            cov_inv = np.linalg.inv(covariances[i])
        except np.linalg.LinAlgError:
            # If covariance is singular, fall back to Euclidean distance
            distances = np.linalg.norm(diff, axis=1)
        else:
            # Compute Mahalanobis distance
            distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        
        # Find k+1 nearest (including self) and exclude self
        nearest_indices = np.argpartition(distances, k)[:k+1]
        nearest_indices = nearest_indices[nearest_indices != i][:k]
        
        # Sort by distance
        nearest_distances = distances[nearest_indices]
        sorted_order = np.argsort(nearest_distances)
        neighbors[i] = nearest_indices[sorted_order]
    
    return neighbors


def _build_covariance_matrices(
    scales: np.ndarray,
    rotations: np.ndarray,
    scaling_modifier = 1.0,
) -> np.ndarray:
    """
    Build covariance matrices from Gaussian scales and rotations.
    
    Covariance matrix: Σ = R * S * S^T * R^T
    where R is rotation matrix from quaternion and S is diagonal scale matrix.
    
    Args:
        scales: (N, 3) array of scales
        rotations: (N, 4) array of quaternions [w, x, y, z]
    
    Returns:
        (N, 3, 3) array of covariance matrices
    """
    L = build_scaling_rotation(torch.tensor(scaling_modifier * scales), torch.tensor(rotations))
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm.cpu().numpy()
    
    
    return covariances


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [w, x, y, z] or [x, y, z, w] (will auto-detect)
    
    Returns:
        (3, 3) rotation matrix
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Check if format is [w, x, y, z] or [x, y, z, w]
    # Typically quaternions have w as the real part, which should be largest for small rotations
    if abs(q[0]) > abs(q[-1]):
        # Assume [w, x, y, z] format
        w, x, y, z = q
    else:
        # Assume [x, y, z, w] format
        x, y, z, w = q
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def get_neighborhood_by_ring(
    point: int,
    ring: int,
    ring1_nbrs: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Get all neighbors within a specified ring distance from a point.
    
    Ring k neighbors are all points reachable by k or fewer edge hops.
    Supports up to ring 4.
    
    Args:
        point: Index of the center point
        ring: Ring distance (1-4)
        ring1_nbrs: Dictionary mapping each point to its ring-1 neighbors
    
    Returns:
        Array of neighbor indices within the specified ring distance
    """
    nbrs = set(ring1_nbrs[point])
    
    if ring >= 2:
        second_nbrs = set()
        for nbr in nbrs:
            second_nbrs = second_nbrs.union(set(ring1_nbrs[nbr]))
        second_nbrs.discard(point)
        nbrs = nbrs.union(second_nbrs)
    
    if ring >= 3:
        third_nbrs = set()
        for nbr in second_nbrs:
            third_nbrs = third_nbrs.union(set(ring1_nbrs[nbr]))
        third_nbrs.discard(point)
        nbrs = nbrs.union(third_nbrs)
    
    if ring == 4:
        fourth_nbrs = set()
        for nbr in third_nbrs:
            fourth_nbrs = fourth_nbrs.union(set(ring1_nbrs[nbr]))
        fourth_nbrs.discard(point)
        nbrs = nbrs.union(fourth_nbrs)
    
    return np.array(list(nbrs))


def get_all_points_nbrs_all_rings(
    vertices: np.ndarray,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None,
    n_neighbors_ring1: int = 22
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute all neighborhood rings (1-4) for all points in a point cloud.
    
    Args:
        vertices: (N, 3) array of vertex positions
        use_mahalanobis: If True, use Mahalanobis distance for Gaussians
        gaussian_scales: (N, 3) Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (N, 4) Gaussian rotations as quaternions (required if use_mahalanobis=True)
        n_neighbors_ring1: Number of neighbors to use for ring-1 computation
    
    Returns:
        Tuple of (ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs) dictionaries
    """
    num_v = vertices.shape[0]
    
    # Compute ring-1 neighbors
    ring1_nbrs = ring1_neighbors_gaussians(
        vertices,
        n_neighbors=n_neighbors_ring1,
        use_mahalanobis=use_mahalanobis,
        gaussian_scales=gaussian_scales,
        gaussian_rotations=gaussian_rotations
    )
    
    # Build higher-order rings
    ring2_nbrs = {index: get_neighborhood_by_ring(index, 2, ring1_nbrs) for index in range(num_v)}
    ring3_nbrs = {index: get_neighborhood_by_ring(index, 3, ring1_nbrs) for index in range(num_v)}
    ring4_nbrs = {index: get_neighborhood_by_ring(index, 4, ring1_nbrs) for index in range(num_v)}
    
    return ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs


def get_all_points_nbrs_single_ring(
    vertices: np.ndarray,
    ring: int,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None,
    n_neighbors_ring1: int = 22
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute ring-1 and specified ring neighborhoods for all points.
    
    This is more efficient than get_all_points_nbrs_all_rings when you only need
    one specific ring level.
    
    Args:
        vertices: (N, 3) array of vertex positions
        ring: Ring distance to compute (1-4)
        use_mahalanobis: If True, use Mahalanobis distance for Gaussians
        gaussian_scales: (N, 3) Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (N, 4) Gaussian rotations as quaternions (required if use_mahalanobis=True)
        n_neighbors_ring1: Number of neighbors to use for ring-1 computation
    
    Returns:
        Tuple of (ring1_nbrs, ring_nbrs) dictionaries
    """
    num_v = vertices.shape[0]
    
    # Compute ring-1 neighbors
    ring1_nbrs = ring1_neighbors_gaussians(
        vertices,
        n_neighbors=n_neighbors_ring1,
        use_mahalanobis=use_mahalanobis,
        gaussian_scales=gaussian_scales,
        gaussian_rotations=gaussian_rotations
    )
    
    # Build specified ring
    ring_nbrs = {index: get_neighborhood_by_ring(index, ring, ring1_nbrs) for index in range(num_v)}
    
    return ring1_nbrs, ring_nbrs


def map_points_to_surface(
    query_points: np.ndarray,
    target_points: np.ndarray,
    use_mahalanobis: bool = False,
    query_scales: Optional[np.ndarray] = None,
    query_rotations: Optional[np.ndarray] = None,
    target_scales: Optional[np.ndarray] = None,
    target_rotations: Optional[np.ndarray] = None,
    return_distances: bool = False
) -> np.ndarray:
    """
    Map query points to nearest target points using Euclidean or Mahalanobis distance.
    the Mahalanobis distance can be computed from either the query points' or target points' covariances.
    that enables the quary points or the target points be gaussians.
    Args:
        query_points: (M, 3) points/gaussians to map
        target_points: (N, 3) target surface points/gaussians
        use_mahalanobis: Whether to use Mahalanobis distance
        query_scales: (M, 3) scales for query points (used if use_mahalanobis=True)
        query_rotations: (M, 4) rotations for query points (used if use_mahalanobis=True)
        target_scales: (N, 3) scales for target points (used if use_mahalanobis=True)
        target_rotations: (N, 4) rotations for target points (used if use_mahalanobis=True)
        return_distances: Whether to also return distances
    
    Returns:
        If return_distances=False: (M,) array of target indices
        If return_distances=True: tuple of ((M,) indices, (M,) distances)
    """
    num_query = len(query_points)
    
    if not use_mahalanobis:
        # Use standard KD-tree for Euclidean distance
        from scipy.spatial import KDTree
        tree = KDTree(target_points)
        distances, indices = tree.query(query_points)
        
        if return_distances:
            return indices, distances
        return indices
    
    # Use Mahalanobis distance
    if query_scales is not None and query_rotations is not None:
        # Distance from query points' perspective (query point covariance)
        covariances = _build_covariance_matrices(query_scales, query_rotations)
        indices = np.zeros(num_query, dtype=np.int32)
        distances_out = np.zeros(num_query)
        
        for i in range(num_query):
            try:
                cov_inv = np.linalg.inv(covariances[i])
            except np.linalg.LinAlgError:
                # Fall back to Euclidean
                dists = np.linalg.norm(target_points - query_points[i], axis=1)
            else:
                diff = target_points - query_points[i]
                dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
            
            indices[i] = np.argmin(dists)
            distances_out[i] = dists[indices[i]]
    
    elif target_scales is not None and target_rotations is not None:
        # Distance from target points' perspective (target point covariance)
        target_covariances = _build_covariance_matrices(target_scales, target_rotations)
        indices = np.zeros(num_query, dtype=np.int32)
        distances_out = np.zeros(num_query)
        
        for i in range(num_query):
            dists = np.zeros(len(target_points))
            for j in range(len(target_points)):
                try:
                    cov_inv = np.linalg.inv(target_covariances[j])
                except np.linalg.LinAlgError:
                    dists[j] = np.linalg.norm(query_points[i] - target_points[j])
                else:
                    diff = query_points[i] - target_points[j]
                    dists[j] = np.sqrt(diff @ cov_inv @ diff)
            
            indices[i] = np.argmin(dists)
            distances_out[i] = dists[indices[i]]
    else:
        raise ValueError("Either query or target scales/rotations must be provided for Mahalanobis distance")
    
    if return_distances:
        return indices, distances_out
    return indices
