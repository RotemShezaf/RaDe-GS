"""
Utility functions for computing neighborhood rings on point clouds and meshes.

Supports both regular Euclidean distance and Mahalanobis distance for Gaussians.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Tuple, Union
import torch

# Try knn_cuda first
try:
    from KNN_CUDA.knn_cuda import KNN
    KNN_CUDA_AVAILABLE = True
    SIMPLE_KNN_AVAILABLE = False
except (ImportError, ValueError, RuntimeError, Exception) as e:
    KNN_CUDA_AVAILABLE = False
    # Try simple-knn as fallback
    try:
        from simple_knn._C import SimpleKNN
        SIMPLE_KNN_AVAILABLE = True
        print("Using simple-knn from submodules (knn_cuda not available)")
    except (ImportError, Exception) as e2:
        SIMPLE_KNN_AVAILABLE = False
        print(f"Warning: Neither knn_cuda nor simple-knn available, falling back to CPU implementation")

from utils.general_utils import  strip_symmetric, build_scaling_rotation

KNN_CUDA_AVAILABLE = False #until fix bug
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
    vertices: Union[np.ndarray, torch.Tensor],
    n_neighbors: int = 16,
    use_mahalanobis: bool = True,
    gaussian_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gaussian_rotations: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[Dict[int, np.ndarray], float, np.ndarray]:
    """
    Find ring-1 (immediate) neighbors for each point in a point cloud.
    
    Args:
        vertices: (N, 3) array of vertex positions
        n_neighbors: Number of neighbors to find for each point
        use_mahalanobis: If True, use Mahalanobis distance based on Gaussian covariance
        gaussian_scales: (N, 3) array of Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (N, 4) array of Gaussian rotations as quaternions (required if use_mahalanobis=True)
    
    Returns:
        Tuple of (neighbor_dict, mean_nearest_distance, per_point_nearest_distances) where:
            neighbor_dict: Dictionary mapping vertex index to array of neighbor indices (excluding self)
            mean_nearest_distance: Mean distance to nearest neighbor across all points
            per_point_nearest_distances: (N,) array of distances to nearest neighbor for each point
    """
    # Convert to torch if needed

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).float()
    if gaussian_scales is not None and isinstance(gaussian_scales, np.ndarray):
        gaussian_scales = torch.from_numpy(gaussian_scales).float()
    if gaussian_rotations is not None and isinstance(gaussian_rotations, np.ndarray):
        gaussian_rotations = torch.from_numpy(gaussian_rotations).float()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vertices = vertices.to(device)
    
    num_v = vertices.shape[0]
    
    if use_mahalanobis:
        if gaussian_scales is None or gaussian_rotations is None:
            raise ValueError("gaussian_scales and gaussian_rotations must be provided when use_mahalanobis=True")
        
        # Compute Mahalanobis distance-based neighbors
        ring1_nbrs, mean_nn_dist, per_point_nn_dist = _compute_mahalanobis_neighbors_torch(
            vertices, gaussian_scales.to(device), gaussian_rotations.to(device), n_neighbors
        )
    else:
        # Use knn_cuda if available, otherwise fall back
        if KNN_CUDA_AVAILABLE and device.type == 'cuda':

            knn = KNN(k=10, transpose_mode=True)

            ref = torch.rand(32, 1000, 5).cuda()
            query = torch.rand(32, 50, 5).cuda()

            dist, indx = knn(ref, query)  # 32 x 50 x 10

           

            # knn_cuda expects [batch_size x dim x num_points]
            ref = vertices.unsqueeze(0).clone() # [1 x N x 3]
            query = ref.clone()
            
            
            knn = KNN(k=n_neighbors+1, transpose_mode=True)
            dist, indices = knn(ref, query)  # [1 x k+1 x N]
            # Remove batch dimension and exclude self (first neighbor)
            indices = indices.squeeze(0)[1:, :].cpu().numpy()  # [N x k]
            dist_tensor = dist.squeeze(0)[1:, :]  # [N x k] keep in torch
            dist = dist_tensor.cpu().numpy()  # [N x k]
            ring1_nbrs = {i: indices[i] for i in range(num_v)}
            # Per-point and mean distance to nearest neighbor (first column after excluding self)
            per_point_nn_dist = torch.sqrt(dist_tensor[:, 0]).cpu().numpy()  # (N,)
            mean_nn_dist = float(per_point_nn_dist.mean())
        elif SIMPLE_KNN_AVAILABLE and device.type == 'cuda':
            # Use simple-knn from submodules
            try:
                # SimpleKNN expects (N, 3) tensor and returns neighbor indices
                knn_simple = SimpleKNN(n_neighbors+1)
                knn_simple.build(vertices.contiguous())
                indices = knn_simple.query(vertices.contiguous())  # [N, k+1]
                indices = indices[:, 1:].cpu().numpy()  # Exclude self (first neighbor), (N, k)
                ring1_nbrs = {i: indices[i] for i in range(num_v)}
                # Compute distances to nearest neighbors
                nn_distances = torch.norm(vertices - vertices[indices[:, 0]], dim=1)
                per_point_nn_dist = nn_distances.cpu().numpy()  # (N,)
                mean_nn_dist = float(per_point_nn_dist.mean())
            except Exception as e:
                # If simple-knn fails, fall back to pure torch
                dist_matrix = torch.cdist(vertices, vertices, p=2.0)  # [N, N]
                dist, indices = torch.topk(dist_matrix, k=n_neighbors+1, largest=False, dim=1)
                indices = indices[:, 1:].cpu().numpy()  # Exclude self, (N, k)
                ring1_nbrs = {i: indices[i] for i in range(num_v)}
                per_point_nn_dist = dist[:, 1].cpu().numpy()  # (N,) Distance to nearest (first after self)
                mean_nn_dist = float(per_point_nn_dist.mean())
        else:
            # Fallback to sklearn
            vertices_np = vertices.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(vertices_np)
            dist, indices = nbrs.kneighbors(vertices_np)
            ring1_nbrs = {index: indices[index, 1:] for index in range(num_v)}
            # Per-point and mean distance to nearest neighbor (first column after self)
            per_point_nn_dist = dist[:, 1]  # (N,)
            mean_nn_dist = float(per_point_nn_dist.mean())
    
    return ring1_nbrs, mean_nn_dist, per_point_nn_dist


def _compute_mahalanobis_neighbors(
    vertices: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    k: int
) -> Dict[int, np.ndarray]:
    """
    Compute k-nearest neighbors using Mahalanobis distance based on Gaussian covariance.
    
    Legacy numpy implementation for backward compatibility.
    """
    # Convert to torch and use optimized version
    vertices_torch = torch.from_numpy(vertices).float()
    scales_torch = torch.from_numpy(scales).float()
    rotations_torch = torch.from_numpy(rotations).float()
    
    return _compute_mahalanobis_neighbors_torch(vertices_torch, scales_torch, rotations_torch, k)


def _compute_mahalanobis_neighbors_torch(
    vertices: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    k: int
) -> Tuple[Dict[int, np.ndarray], float, np.ndarray]:
    """
    Compute k-nearest neighbors using Mahalanobis distance based on Gaussian covariance.
    Optimized torch implementation.
    
    Args:
        vertices: (N, 3) tensor of Gaussian center positions
        scales: (N, 3) tensor of Gaussian scales
        rotations: (N, 4) tensor of Gaussian rotations as quaternions
        k: Number of neighbors to find
    
    Returns:
        Tuple of (neighbors_dict, mean_nearest_distance, per_point_nearest_distances)
    """
    num_points = len(vertices)
    neighbors = {}
    nearest_distances = []
    
    # Build covariance matrices for each Gaussian
    L = build_scaling_rotation(scales, rotations)
    actual_covariance = L @ L.transpose(1, 2)
    covariances = strip_symmetric(actual_covariance)  # (N, 3, 3)
    
    # Batch process for efficiency
    for i in range(num_points):
        # Compute Mahalanobis distance from point i to all other points
        diff = vertices - vertices[i]  # (N, 3)
        
        try:
            cov_inv = torch.linalg.inv(covariances[i])
            # Vectorized Mahalanobis distance: sqrt((x-μ)^T Σ^(-1) (x-μ))
            distances = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
        except:
            # Fallback to Euclidean
            distances = torch.norm(diff, dim=1)
        
        # Find k+1 nearest (including self) and exclude self
        sorted_distances, nearest_indices = torch.topk(distances, k+1, largest=False)
        nearest_indices = nearest_indices[nearest_indices != i][:k]
        
        neighbors[i] = nearest_indices.cpu().numpy()
        # Store nearest neighbor distance (first non-self)
        nearest_distances.append(sorted_distances[1].item())
    
    nearest_distances_array = np.array(nearest_distances)
    mean_nn_dist = float(nearest_distances_array.mean())
    return neighbors, mean_nn_dist, nearest_distances_array


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
        Tuple of (ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs, mean_dist, per_point_dist)
        where mean_dist is mean nearest neighbor distance and per_point_dist is (N,) array
    """
    num_v = vertices.shape[0]
    
    # Compute ring-1 neighbors
    ring1_nbrs, mean_dist, per_point_dist = ring1_neighbors_gaussians(
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
    
    return  ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs, mean_dist, per_point_dist


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
    ring1_nbrs, _ = ring1_neighbors_gaussians(
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
    query_points: Union[np.ndarray, torch.Tensor],
    target_points: Union[np.ndarray, torch.Tensor],
    use_mahalanobis: bool = False,
    query_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    query_rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
    return_distances: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Map query points to nearest target points using Euclidean or Mahalanobis distance.
    Optimized torch/knn_cuda implementation.
    
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
    # Convert to torch
    if isinstance(query_points, np.ndarray):
        query_points = torch.from_numpy(query_points).float()
    if isinstance(target_points, np.ndarray):
        target_points = torch.from_numpy(target_points).float()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_points = query_points.to(device)
    target_points = target_points.to(device)
    
    num_query = len(query_points)
    
    if not use_mahalanobis:
        # Use knn_cuda if available for fast Euclidean distance
        if KNN_CUDA_AVAILABLE and device.type == 'cuda':
            # knn_cuda expects [batch_size x dim x num_points]
            ref = target_points.T.unsqueeze(0)  # [1 x 3 x N]
            query = query_points.T.unsqueeze(0)  # [1 x 3 x M]
            
            knn = KNN(k=1, transpose_mode=False)
            dist, indices = knn(ref, query)  # [1 x 1 x M]
            
            # Remove batch and k dimensions
            indices = indices.squeeze().cpu().numpy()  # (M,)
            distances = dist.squeeze().sqrt().cpu().numpy()  # (M,)
            
            if return_distances:
                return indices, distances
            return indices
        elif SIMPLE_KNN_AVAILABLE and device.type == 'cuda':
            # Use simple-knn from submodules
            try:
                knn_simple = SimpleKNN(1)
                knn_simple.build(target_points.contiguous())
                indices = knn_simple.query(query_points.contiguous())  # [M, 1]
                indices = indices.squeeze(1).cpu().numpy()  # [M]
                
                if return_distances:
                    # Compute distances for the found neighbors
                    distances_out = torch.norm(
                        query_points - target_points[torch.from_numpy(indices).to(device)],
                        dim=1
                    ).cpu().numpy()
                    return indices, distances_out
                return indices
            except Exception as e:
                # Fallback to torch.cdist
                dist_matrix = torch.cdist(query_points, target_points, p=2.0)  # [M, N]
                indices = torch.argmin(dist_matrix, dim=1)  # [M]
                indices_np = indices.cpu().numpy()
                
                if return_distances:
                    distances_out = dist_matrix[torch.arange(num_query), indices].cpu().numpy()
                    return indices_np, distances_out
                return indices_np
        else:
            # Fallback to scipy KDTree for CPU
            from scipy.spatial import KDTree
            tree = KDTree(target_points.cpu().numpy())
            distances, indices = tree.query(query_points.cpu().numpy())
            
            if return_distances:
                return indices, distances
            return indices
    
    # Use Mahalanobis distance
    if query_scales is not None and query_rotations is not None:
        # Convert to torch if needed
        if isinstance(query_scales, np.ndarray):
            query_scales = torch.from_numpy(query_scales).float()
        if isinstance(query_rotations, np.ndarray):
            query_rotations = torch.from_numpy(query_rotations).float()
        
        query_scales = query_scales.to(device)
        query_rotations = query_rotations.to(device)
        
        # Distance from query points' perspective (query point covariance)
        L = build_scaling_rotation(query_scales, query_rotations)
        actual_covariance = L @ L.transpose(1, 2)
        covariances = strip_symmetric(actual_covariance)
        
        indices = torch.zeros(num_query, dtype=torch.long, device=device)
        distances_out = torch.zeros(num_query, device=device)
        
        for i in range(num_query):
            try:
                cov_inv = torch.linalg.inv(covariances[i])
                diff = target_points - query_points[i]
                dists = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
            except:
                # Fall back to Euclidean
                dists = torch.norm(target_points - query_points[i], dim=1)
            
            indices[i] = torch.argmin(dists)
            distances_out[i] = dists[indices[i]]
        
        indices = indices.cpu().numpy()
        distances_out = distances_out.cpu().numpy()
    
    elif target_scales is not None and target_rotations is not None:
        # Convert to torch if needed
        if isinstance(target_scales, np.ndarray):
            target_scales = torch.from_numpy(target_scales).float()
        if isinstance(target_rotations, np.ndarray):
            target_rotations = torch.from_numpy(target_rotations).float()
        
        target_scales = target_scales.to(device)
        target_rotations = target_rotations.to(device)
        
        # Distance from target points' perspective (target point covariance)
        L = build_scaling_rotation(target_scales, target_rotations)
        actual_covariance = L @ L.transpose(1, 2)
        target_covariances = strip_symmetric(actual_covariance)
        
        indices = torch.zeros(num_query, dtype=torch.long, device=device)
        distances_out = torch.zeros(num_query, device=device)
        
        for i in range(num_query):
            dists = torch.zeros(len(target_points), device=device)
            for j in range(len(target_points)):
                try:
                    cov_inv = torch.linalg.inv(target_covariances[j])
                    diff = query_points[i] - target_points[j]
                    dists[j] = torch.sqrt(diff @ cov_inv @ diff)
                except:
                    dists[j] = torch.norm(query_points[i] - target_points[j])
            
            indices[i] = torch.argmin(dists)
            distances_out[i] = dists[indices[i]]
        
        indices = indices.cpu().numpy()
        distances_out = distances_out.cpu().numpy()
    else:
        raise ValueError("Either query or target scales/rotations must be provided for Mahalanobis distance")
    
    if return_distances:
        return indices, distances_out
    return indices

def build_rotation(r):
    norm = np.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = np.zeros((q.shape[0], 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
