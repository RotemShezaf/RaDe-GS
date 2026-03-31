"""
Utility functions for computing neighborhood rings on point clouds and meshes.

Supports both regular Euclidean distance and Mahalanobis distance for Gaussians.

Adaptive KNN
~~~~~~~~~~~~
The :func:`adaptive_ring1_neighbors` function performs a **per-point binary
search** over each point's ring-1 *k* value (in the range
``[n_neighbors_base, k_boost]``).  A single kNN query with ``k_boost`` is
computed once; then at each binary-search step the **true** ring-
``target_ring`` counts are computed from the actual heterogeneous per-point
*k* values via :func:`_ring_counts_from_mid` (GPU sparse matmul with
``torch.sparse`` on CUDA when available, ``scipy.sparse`` on CPU otherwise).

The search terminates when **any** of three criteria is met:

* The mean cut across all points is ≤ ``adaptive_max_mean_cut``.
* The number of binary-search steps reaches ``adaptive_max_steps``.
* The per-point *mid* values have converged (no change from previous step).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from typing import Dict, List, Optional, Tuple, Union
import torch

# Try knn_cuda first
KNN_CUDA_AVAILABLE = False
SIMPLE_KNN_AVAILABLE = False

try:
    from knn_cuda import KNN
    KNN_CUDA_AVAILABLE = True
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


def _compute_knn_euclidean(
    vertices: torch.Tensor,
    n_neighbors: int,
    query_points: Optional[torch.Tensor] = None,
    exclude_self: bool = True,
    return_dict: bool = False,
    return_stats: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]],
           Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], float, np.ndarray]]:
    """
    Compute K-nearest neighbors using Euclidean distance with multiple backend support.
    
    Args:
        vertices: (N, 3) reference points tensor
        n_neighbors: Number of neighbors to find
        query_points: (M, 3) query points tensor. If None, uses vertices as query (self-query)
        exclude_self: If True and query_points is None, excludes self from neighbors
        return_dict: If True, also returns neighbor_dict
        return_stats: If True, also returns mean_nn_dist and per_point_nn_dist (implies return_dict)
    
    Returns:
        Base: (indices, distances) where indices is (M, k) and distances is (M, k)
        If return_dict: adds neighbor_dict mapping point index to neighbor indices
        If return_stats: adds mean_nn_dist (float) and per_point_nn_dist (M,) array
    """
    device = vertices.device
    is_self_query = query_points is None
    
    if is_self_query:
        query_points = vertices
        num_points = len(vertices)
        k_total = n_neighbors + 1 if exclude_self else n_neighbors
    else:
        num_points = len(query_points)
        k_total = n_neighbors
    
    # Try knn_cuda first
    if KNN_CUDA_AVAILABLE and device.type == 'cuda':
        ref = vertices.unsqueeze(0).contiguous()  # [1 x N x 3]
        query = query_points.unsqueeze(0).contiguous()  # [1 x M x 3]
        
        knn = KNN(k=k_total, transpose_mode=True)
        dist, indices = knn(ref, query)  # [1 x M x k]
        
        # Remove batch dimension
        indices_tensor = indices.squeeze(0)  # [M x k]
        dist_tensor = dist.squeeze(0)  # [M x k]
        
        if is_self_query and exclude_self:
            indices_np = indices_tensor[:, 1:].cpu().numpy()
            dist_np = dist_tensor[:, 1:].cpu().numpy()
            nn_dist_col = 0  # After slicing, column 0 is nearest non-self neighbor
        else:
            indices_np = indices_tensor.cpu().numpy()
            dist_np = dist_tensor.cpu().numpy()
            nn_dist_col = 0
            
    # Try simple-knn
    elif SIMPLE_KNN_AVAILABLE and device.type == 'cuda' and is_self_query:
        try:
            knn_simple = SimpleKNN(k_total)
            knn_simple.build(vertices.contiguous())
            indices_tensor = knn_simple.query(vertices.contiguous())
            
            if exclude_self:
                indices_np = indices_tensor[:, 1:].cpu().numpy()
                nn_dist_col = 0  # After slicing, column 0 is nearest non-self
            else:
                indices_np = indices_tensor.cpu().numpy()
                nn_dist_col = 0
            
            # Compute distances
            dist_list = []
            for i in range(num_points):
                dists = torch.norm(vertices[i] - vertices[indices_np[i]], dim=1)
                dist_list.append(dists.cpu().numpy())
            dist_np = np.stack(dist_list)
        except Exception:
            # Fall through to torch.cdist
            dist_matrix = torch.cdist(query_points, vertices, p=2.0)
            dist_tensor, indices_tensor = torch.topk(dist_matrix, k=k_total, largest=False, dim=1)
            
            if is_self_query and exclude_self:
                indices_np = indices_tensor[:, 1:].cpu().numpy()
                dist_np = dist_tensor[:, 1:].cpu().numpy()
                nn_dist_col = 0
            else:
                indices_np = indices_tensor.cpu().numpy()
                dist_np = dist_tensor.cpu().numpy()
                nn_dist_col = 0
    
    # Fallback to torch.cdist for CUDA
    elif device.type == 'cuda':
        dist_matrix = torch.cdist(query_points, vertices, p=2.0)
        dist_tensor, indices_tensor = torch.topk(dist_matrix, k=k_total, largest=False, dim=1)
        
        if is_self_query and exclude_self:
            indices_np = indices_tensor[:, 1:].cpu().numpy()
            dist_np = dist_tensor[:, 1:].cpu().numpy()
            nn_dist_col = 0
        else:
            indices_np = indices_tensor.cpu().numpy()
            dist_np = dist_tensor.cpu().numpy()
            nn_dist_col = 0
    
    # CPU fallback with sklearn
    else:
        vertices_np = vertices.cpu().numpy()
        query_np = query_points.cpu().numpy()
        
        nbrs = NearestNeighbors(n_neighbors=k_total, algorithm='ball_tree').fit(vertices_np)
        dist_np, indices_np = nbrs.kneighbors(query_np)
        
        if is_self_query and exclude_self:
            indices_np = indices_np[:, 1:]
            dist_np = dist_np[:, 1:]
        nn_dist_col = 0
    
    # Build optional outputs
    if not return_dict and not return_stats:
        return indices_np, dist_np
    
    neighbor_dict = {i: indices_np[i] for i in range(num_points)}
    
    if not return_stats:
        return indices_np, dist_np, neighbor_dict
    
    per_point_nn_dist = dist_np[:, nn_dist_col]
    mean_nn_dist = float(per_point_nn_dist.mean())
    
    return indices_np, dist_np, neighbor_dict, mean_nn_dist, per_point_nn_dist


def _compute_knn_mahalanobis(
    vertices: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    n_neighbors: int,
    return_dict: bool = False,
    return_stats: bool = False,
    batch_size: Optional[int] = 1024
) -> Union[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]],
           Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], float, np.ndarray]]:
    """
    Compute K-nearest neighbors using Mahalanobis distance based on Gaussian covariance.
    
    Args:
        vertices: (N, 3) tensor of Gaussian center positions
        scales: (N, 3) tensor of Gaussian scales
        rotations: (N, 4) tensor of Gaussian rotations as quaternions
        n_neighbors: Number of neighbors to find
        return_dict: If True, also returns neighbor_dict
        return_stats: If True, also returns mean_nn_dist and per_point_nn_dist (implies return_dict)
        batch_size: If provided, process distance matrix in batches to reduce memory usage.
                   If None, processes all at once (faster but more memory).
    
    Returns:
        Base: (indices, distances) where indices is (N, k) and distances is (N, k)
        If return_dict: adds neighbor_dict mapping point index to neighbor indices
        If return_stats: adds mean_nn_dist (float) and per_point_nn_dist (N,) array
    """
    num_points = len(vertices)
    device = vertices.device
    
    # Build covariance matrices for each Gaussian
    L = build_scaling_rotation(scales, rotations)
    actual_covariance = L @ L.transpose(1, 2)  # (N, 3, 3)
    
    # Compute inverse covariances in batch
    try:
        cov_inv_batch = torch.linalg.inv(actual_covariance)  # (N, 3, 3)
        use_mahalanobis = True
    except:
        use_mahalanobis = False
    
    # Determine k for topk (need k+1 to exclude self)
    k_topk = min(n_neighbors + 1, num_points)
    
    if use_mahalanobis:
        if batch_size is None or batch_size >= num_points:
            # Fully vectorized: compute all distances at once
            # diff[i, j] = vertices[j] - vertices[i], shape (N, N, 3)
            diff = vertices.unsqueeze(0) - vertices.unsqueeze(1)
            
            # Compute temp[i, j, :] = diff[i, j, :] @ cov_inv_batch[i, :, :]
            temp = torch.einsum('ijk,ikl->ijl', diff, cov_inv_batch)  # (N, N, 3)
            
            # Compute squared Mahalanobis distance and take sqrt
            distances = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (N, N)
            
            # Find k+1 nearest for each point
            sorted_distances, nearest_indices = torch.topk(distances, k_topk, largest=False, dim=1)
        else:
            # Batched computation to reduce memory
            # We'll compute topk per batch and merge results
            all_distances = []
            all_indices = []
            
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                batch_vertices = vertices[start_idx:end_idx]  # (B, 3)
                batch_cov_inv = cov_inv_batch[start_idx:end_idx]  # (B, 3, 3)
                
                # diff[b, j] = vertices[j] - batch_vertices[b], shape (B, N, 3)
                diff = vertices.unsqueeze(0) - batch_vertices.unsqueeze(1)
                
                # temp[b, j, :] = diff[b, j, :] @ batch_cov_inv[b, :, :]
                temp = torch.einsum('bjk,bkl->bjl', diff, batch_cov_inv)  # (B, N, 3)
                
                # Compute Mahalanobis distances for this batch
                batch_distances = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (B, N)
                
                # Get topk for this batch
                batch_sorted_dist, batch_nearest_idx = torch.topk(batch_distances, k_topk, largest=False, dim=1)
                
                all_distances.append(batch_sorted_dist)
                all_indices.append(batch_nearest_idx)
            
            sorted_distances = torch.cat(all_distances, dim=0)  # (N, k+1)
            nearest_indices = torch.cat(all_indices, dim=0)  # (N, k+1)
    else:
        # Euclidean distance fallback (also supports batching)
        if batch_size is None or batch_size >= num_points:
            distances = torch.cdist(vertices, vertices, p=2.0)  # (N, N)
            sorted_distances, nearest_indices = torch.topk(distances, k_topk, largest=False, dim=1)
        else:
            all_distances = []
            all_indices = []
            
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                batch_vertices = vertices[start_idx:end_idx]
                
                batch_distances = torch.cdist(batch_vertices, vertices, p=2.0)  # (B, N)
                batch_sorted_dist, batch_nearest_idx = torch.topk(batch_distances, k_topk, largest=False, dim=1)
                
                all_distances.append(batch_sorted_dist)
                all_indices.append(batch_nearest_idx)
            
            sorted_distances = torch.cat(all_distances, dim=0)
            nearest_indices = torch.cat(all_indices, dim=0)
    
    # Exclude self (always first in sorted order since distance to self is 0)
    indices_np = nearest_indices[:, 1:].cpu().numpy()  # (N, k)
    dist_np = sorted_distances[:, 1:].cpu().numpy()  # (N, k)
    per_point_nn_dist = sorted_distances[:, 1].cpu().numpy()  # (N,) nearest non-self
    
    if not return_dict and not return_stats:
        return indices_np, dist_np
    
    neighbor_dict = {i: indices_np[i] for i in range(num_points)}
    
    if not return_stats:
        return indices_np, dist_np, neighbor_dict
    
    mean_nn_dist = float(per_point_nn_dist.mean())
    return indices_np, dist_np, neighbor_dict, mean_nn_dist, per_point_nn_dist


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
        _, _, ring1_nbrs, mean_nn_dist, per_point_nn_dist = _compute_knn_mahalanobis(
            vertices, gaussian_scales.to(device), gaussian_rotations.to(device), 
            n_neighbors, return_stats=True
        )
    else:
        # Use unified KNN function with multiple backend support
        _, _, ring1_nbrs, mean_nn_dist, per_point_nn_dist = _compute_knn_euclidean(
            vertices, n_neighbors, query_points=None, exclude_self=True, return_stats=True
        )
    
    return ring1_nbrs, mean_nn_dist, per_point_nn_dist


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
        scaling_modifier: Scalar to multiply scales by
    
    Returns:
        (N, 6) array of symmetric covariance matrices in packed format
        [C00, C01, C02, C11, C12, C22] for each matrix
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


def ring_counts_sparse(
    ring1_nbrs: Dict[int, np.ndarray],
    N: int,
    target_ring: int,
) -> np.ndarray:
    """Count ring-``target_ring`` neighbours for every point via sparse matrix ops.

    Builds a CSR adjacency matrix from *ring1_nbrs* and computes reachability
    via sparse matrix powers — same counts as calling
    :func:`get_neighborhood_by_ring` per point but without a Python per-point
    loop.

    Args:
        ring1_nbrs: Mapping from point index to ring-1 neighbour indices.
        N: Total number of points.
        target_ring: Ring level (e.g. 3).

    Returns:
        (N,) int32 array of ring-``target_ring`` neighbour counts.
    """
    lengths = np.array([len(ring1_nbrs[i]) for i in range(N)], dtype=np.int64)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.concatenate([ring1_nbrs[i] for i in range(N)]).astype(np.int32)
    A = csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(N, N),
    )

    reach = A.copy()
    power = A.copy()
    for r in range(2, target_ring + 1):
        power = power @ A
        reach = reach + power
    reach.setdiag(0)
    reach.eliminate_zeros()
    return np.diff(reach.indptr).astype(np.int32)


def _ring_counts_from_mid(
    boost_array: np.ndarray,
    mid: np.ndarray,
    N: int,
    target_ring: int,
) -> np.ndarray:
    """Compute TRUE ring-``target_ring`` counts from heterogeneous per-point k.

    Builds a sparse adjacency matrix where point *i* is connected to
    ``boost_array[i, :mid[i]]`` neighbours, then computes reachability
    via sparse matrix powers.

    Uses GPU sparse matmul (``torch.sparse`` on CUDA) when available,
    falls back to ``scipy.sparse`` on CPU otherwise.

    Args:
        boost_array: ``(N, k_boost)`` int32 array of precomputed kNN indices.
        mid: ``(N,)`` int32 array of per-point ring-1 sizes.
        N: Number of points.
        target_ring: Ring level (e.g. 3).

    Returns:
        ``(N,)`` int32 array of ring-``target_ring`` neighbour counts.
    """
    max_k = int(mid.max())
    # Vectorised construction: column j valid for point i iff j < mid[i]
    col_range = np.arange(max_k, dtype=np.int32)
    mask = col_range[np.newaxis, :] < mid[:, np.newaxis]  # (N, max_k)
    rows = np.repeat(np.arange(N, dtype=np.int32), mid)
    cols = boost_array[:, :max_k][mask].astype(np.int32)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        rows_t = torch.from_numpy(rows).to(device, dtype=torch.long)
        cols_t = torch.from_numpy(cols).to(device, dtype=torch.long)
        idx = torch.stack([rows_t, cols_t])
        vals = torch.ones(len(rows), device=device, dtype=torch.float32)
        A = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()

        power = A
        acc_idx = [A.indices()]
        for r in range(2, target_ring + 1):
            power = torch.sparse.mm(power, A).coalesce()
            acc_idx.append(power.indices())

        all_idx = torch.cat(acc_idx, dim=1)
        all_val = torch.ones(all_idx.shape[1], device=device, dtype=torch.float32)
        reach = torch.sparse_coo_tensor(all_idx, all_val, (N, N)).coalesce()

        ri = reach.indices()
        diag_mask = ri[0] != ri[1]
        counts = torch.bincount(
            ri[0][diag_mask], minlength=N
        ).cpu().numpy().astype(np.int32)

        del rows_t, cols_t, idx, vals, A, power, reach
        torch.cuda.empty_cache()
        return counts
    else:
        A = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(N, N),
        )
        reach = A.copy()
        power = A.copy()
        for r in range(2, target_ring + 1):
            power = power @ A
            reach = reach + power
        reach.setdiag(0)
        reach.eliminate_zeros()
        return np.diff(reach.indptr).astype(np.int32)


def adaptive_ring1_neighbors(
    vertices: Union[np.ndarray, torch.Tensor],
    target_ring: int,
    target_ring_neighbors: int,
    n_neighbors_base: int = 10,
    k_boost: int = 20,
    adaptive_max_mean_cut: float = 2.0,
    adaptive_max_steps: int = 5,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gaussian_rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[Dict[int, np.ndarray], float, np.ndarray]:
    """
    Build per-point adaptive ring-1 neighbors via binary search so that each
    point's ring-``target_ring`` neighbourhood size stays close to
    ``target_ring_neighbors``.

    **Algorithm — true-count binary search**

    1. Compute kNN with ``k_boost`` once → ``(N, k_boost)`` neighbour array.
    2. Binary search: each point *i* has its own ``[lo_i, hi_i]`` interval.
       At every step ``mid_i = (lo_i + hi_i) // 2`` is computed and the
       **true** ring-``target_ring`` count is computed from the actual
       heterogeneous per-point *mid* values via
       :func:`_ring_counts_from_mid` (GPU sparse matmul when CUDA is
       available, scipy sparse on CPU).
    3. If the count exceeds ``target_ring_neighbors``, ``hi_i`` is lowered;
       otherwise ``lo_i`` is raised.
    4. Stops when mean cut ≤ ``adaptive_max_mean_cut``, after
       ``adaptive_max_steps`` iterations, or when *mid* converges.

    Args:
        vertices: (N, 3) point positions.
        target_ring: Ring level to optimise (e.g. 3).
        target_ring_neighbors: Desired maximum ring-``target_ring`` cap
            (e.g. 128).
        n_neighbors_base: Lower bound of the per-point binary search.
        k_boost: Upper bound (and kNN *k* used to precompute all candidates).
        adaptive_max_mean_cut: Stop when the average cut is at most this.
        adaptive_max_steps: Maximum binary-search iterations.
        use_mahalanobis: Whether to use Mahalanobis distance for kNN.
        gaussian_scales: (N, 3) — required if ``use_mahalanobis``.
        gaussian_rotations: (N, 4) — required if ``use_mahalanobis``.

    Returns:
        Same triple as :func:`ring1_neighbors_gaussians`:
        ``(ring1_nbrs, mean_nn_dist, per_point_nn_dist)``
    """
    if k_boost <= n_neighbors_base:
        return ring1_neighbors_gaussians(
            vertices, n_neighbors=n_neighbors_base,
            use_mahalanobis=use_mahalanobis,
            gaussian_scales=gaussian_scales,
            gaussian_rotations=gaussian_rotations,
        )

    N = len(vertices) if isinstance(vertices, np.ndarray) else vertices.shape[0]

    # ── Precompute full k_boost kNN once ─────────────────────────
    ring1_boost, mean_nn_dist, per_point_nn_dist = ring1_neighbors_gaussians(
        vertices, n_neighbors=k_boost,
        use_mahalanobis=use_mahalanobis,
        gaussian_scales=gaussian_scales,
        gaussian_rotations=gaussian_rotations,
    )

    # Convert dict → 2-D array for vectorised truncation
    boost_array = np.array([ring1_boost[i] for i in range(N)], dtype=np.int32)

    # ── Per-point binary search with true heterogeneous counts ───
    lo = np.full(N, n_neighbors_base, dtype=np.int32)
    hi = np.full(N, k_boost, dtype=np.int32)
    mid = np.full(N, k_boost, dtype=np.int32)

    for step in range(adaptive_max_steps):
        #prev_mid = mid.copy()
        #mid = np.clip((lo + hi) // 2, n_neighbors_base, k_boost)

        # Early exit when mid has converged
       # if np.array_equal(mid, prev_mid):
        #    break

        # TRUE ring counts from heterogeneous per-point k
        counts = _ring_counts_from_mid(boost_array, mid, N, target_ring)

        cuts = np.maximum(counts - target_ring_neighbors, 0)
        mean_cut = float(cuts.sum()) / N
        over = counts > target_ring_neighbors
        mid[over] = np.clip(mid[over] - 1, n_neighbors_base, k_boost)
        #hi[over] = hi[over] - 1
        #lo[~over] = mid[~over] + 1

        if mean_cut <= adaptive_max_mean_cut:
            break

    # Build final ring-1 dict from converged mid values
    ring1_final = {i: boost_array[i, :mid[i]] for i in range(N)}

    return ring1_final, mean_nn_dist, per_point_nn_dist


def get_all_points_nbrs_all_rings(
    vertices: np.ndarray,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None,
    n_neighbors_ring1: int = 22,
    adaptive_target_ring: Optional[int] = None,
    adaptive_target_neighbors: Optional[int] = None,
    adaptive_k_boost: int = 20,
    adaptive_max_mean_cut: float = 2.0,
    adaptive_max_steps: int = 5,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute all neighborhood rings (1-4) for all points in a point cloud.
    
    Args:
        vertices: (N, 3) array of vertex positions
        use_mahalanobis: If True, use Mahalanobis distance for Gaussians
        gaussian_scales: (N, 3) Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (N, 4) Gaussian rotations as quaternions (required if use_mahalanobis=True)
        n_neighbors_ring1: Number of neighbors to use for ring-1 computation
        adaptive_target_ring: If set, use adaptive kNN to target this ring level.
        adaptive_target_neighbors: Desired ring-k count for the adaptive ring.
        adaptive_k_boost: Maximum boosted k for deficient points (adaptive mode).
    
    Returns:
        Tuple of (ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs, mean_dist, per_point_dist)
        where mean_dist is mean nearest neighbor distance and per_point_dist is (N,) array
    """
    num_v = vertices.shape[0]
    
    # Compute ring-1 neighbors (adaptive or fixed)
    if adaptive_target_ring is not None and adaptive_target_neighbors is not None:
        ring1_nbrs, mean_dist, per_point_dist = adaptive_ring1_neighbors(
            vertices,
            target_ring=adaptive_target_ring,
            target_ring_neighbors=adaptive_target_neighbors,
            n_neighbors_base=n_neighbors_ring1,
            k_boost=adaptive_k_boost,
            adaptive_max_mean_cut=adaptive_max_mean_cut,
            adaptive_max_steps=adaptive_max_steps,
            use_mahalanobis=use_mahalanobis,
            gaussian_scales=gaussian_scales,
            gaussian_rotations=gaussian_rotations,
        )
    else:
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
    n_neighbors_ring1: int = 22,
    adaptive_target_ring: Optional[int] = None,
    adaptive_target_neighbors: Optional[int] = None,
    adaptive_k_boost: int = 20,
    adaptive_max_mean_cut: float = 5.0,
    adaptive_max_steps: int = 5,
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
        adaptive_target_ring: If set, use adaptive kNN to target this ring level.
        adaptive_target_neighbors: Desired ring-k count for the adaptive ring.
        adaptive_k_boost: Maximum boosted k for deficient points (adaptive mode).
    
    Returns:
        Tuple of (ring1_nbrs, ring_nbrs) dictionaries
    """
    num_v = vertices.shape[0]
    
    # Compute ring-1 neighbors (adaptive or fixed)
    if adaptive_target_ring is not None and adaptive_target_neighbors is not None:
        ring1_nbrs, _, _ = adaptive_ring1_neighbors(
            vertices,
            target_ring=adaptive_target_ring,
            target_ring_neighbors=adaptive_target_neighbors,
            n_neighbors_base=n_neighbors_ring1,
            k_boost=adaptive_k_boost,
            adaptive_max_mean_cut=adaptive_max_mean_cut,
            adaptive_max_steps=adaptive_max_steps,
            use_mahalanobis=use_mahalanobis,
            gaussian_scales=gaussian_scales,
            gaussian_rotations=gaussian_rotations,
        )
    else:
        ring1_nbrs, _, _ = ring1_neighbors_gaussians(
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
    return_distances: bool = False,
    batch_size: Optional[int] = 1024
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Map query points to nearest target points using Euclidean or Mahalanobis distance.
    Optimized torch/knn_cuda implementation with batching support.
    
    Args:
        query_points: (M, 3) points/gaussians to map
        target_points: (N, 3) target surface points/gaussians
        use_mahalanobis: Whether to use Mahalanobis distance
        query_scales: (M, 3) scales for query points (used if use_mahalanobis=True)
        query_rotations: (M, 4) rotations for query points (used if use_mahalanobis=True)
        target_scales: (N, 3) scales for target points (used if use_mahalanobis=True)
        target_rotations: (N, 4) rotations for target points (used if use_mahalanobis=True)
        return_distances: Whether to also return distances
        batch_size: If provided, process in batches to reduce memory usage.
                   If None, processes all at once (faster but more memory).
    
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
    num_target = len(target_points)
    
    if not use_mahalanobis:
        # Use unified KNN function for Euclidean distance (just indices and distances)
        indices_np, dist_np = _compute_knn_euclidean(
            target_points, n_neighbors=1, query_points=query_points, exclude_self=False
        )
        # Extract single nearest neighbor
        indices = indices_np[:, 0]
        distances = dist_np[:, 0]
        
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
        
        # Build covariance matrices and invert
        L = build_scaling_rotation(query_scales, query_rotations)
        actual_covariance = L @ L.transpose(1, 2)  # (M, 3, 3)
        
        try:
            cov_inv_batch = torch.linalg.inv(actual_covariance)  # (M, 3, 3)
            use_maha = True
        except:
            use_maha = False
        
        if use_maha:
            if batch_size is None or batch_size >= num_query:
                # Fully vectorized: diff[q, t] = target[t] - query[q], shape (M, N, 3)
                diff = target_points.unsqueeze(0) - query_points.unsqueeze(1)
                
                # temp[q, t, :] = diff[q, t, :] @ cov_inv_batch[q, :, :]
                temp = torch.einsum('qtk,qkl->qtl', diff, cov_inv_batch)  # (M, N, 3)
                
                # Mahalanobis distances
                distances_matrix = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (M, N)
                
                # Find nearest target for each query
                distances_out, indices = torch.min(distances_matrix, dim=1)
            else:
                # Batched computation
                indices = torch.zeros(num_query, dtype=torch.long, device=device)
                distances_out = torch.zeros(num_query, device=device)
                
                for start_idx in range(0, num_query, batch_size):
                    end_idx = min(start_idx + batch_size, num_query)
                    batch_query = query_points[start_idx:end_idx]  # (B, 3)
                    batch_cov_inv = cov_inv_batch[start_idx:end_idx]  # (B, 3, 3)
                    
                    # diff[b, t] = target[t] - batch_query[b], shape (B, N, 3)
                    diff = target_points.unsqueeze(0) - batch_query.unsqueeze(1)
                    
                    temp = torch.einsum('btk,bkl->btl', diff, batch_cov_inv)  # (B, N, 3)
                    batch_distances = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (B, N)
                    
                    batch_min_dist, batch_min_idx = torch.min(batch_distances, dim=1)
                    indices[start_idx:end_idx] = batch_min_idx
                    distances_out[start_idx:end_idx] = batch_min_dist
        else:
            # Euclidean fallback
            distances_matrix = torch.cdist(query_points, target_points, p=2.0)
            distances_out, indices = torch.min(distances_matrix, dim=1)
        
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
        
        # Build covariance matrices for targets and invert
        L = build_scaling_rotation(target_scales, target_rotations)
        actual_covariance = L @ L.transpose(1, 2)  # (N, 3, 3)
        
        try:
            cov_inv_batch = torch.linalg.inv(actual_covariance)  # (N, 3, 3)
            use_maha = True
        except:
            use_maha = False
        
        if use_maha:
            if batch_size is None or batch_size >= num_query:
                # Fully vectorized: diff[q, t] = query[q] - target[t], shape (M, N, 3)
                diff = query_points.unsqueeze(1) - target_points.unsqueeze(0)
                
                # temp[q, t, :] = diff[q, t, :] @ cov_inv_batch[t, :, :]
                temp = torch.einsum('qtk,tkl->qtl', diff, cov_inv_batch)  # (M, N, 3)
                
                # Mahalanobis distances
                distances_matrix = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (M, N)
                
                # Find nearest target for each query
                distances_out, indices = torch.min(distances_matrix, dim=1)
            else:
                # Batched computation over query points
                indices = torch.zeros(num_query, dtype=torch.long, device=device)
                distances_out = torch.zeros(num_query, device=device)
                
                for start_idx in range(0, num_query, batch_size):
                    end_idx = min(start_idx + batch_size, num_query)
                    batch_query = query_points[start_idx:end_idx]  # (B, 3)
                    
                    # diff[b, t] = batch_query[b] - target[t], shape (B, N, 3)
                    diff = batch_query.unsqueeze(1) - target_points.unsqueeze(0)
                    
                    temp = torch.einsum('btk,tkl->btl', diff, cov_inv_batch)  # (B, N, 3)
                    batch_distances = torch.sqrt(torch.clamp(torch.sum(temp * diff, dim=2), min=0))  # (B, N)
                    
                    batch_min_dist, batch_min_idx = torch.min(batch_distances, dim=1)
                    indices[start_idx:end_idx] = batch_min_idx
                    distances_out[start_idx:end_idx] = batch_min_dist
        else:
            # Euclidean fallback
            distances_matrix = torch.cdist(query_points, target_points, p=2.0)
            distances_out, indices = torch.min(distances_matrix, dim=1)
        
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
