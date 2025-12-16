"""
Extract mesh from Gaussian centers using KNN based on Mahalanobis distance.
Connects Gaussian centers that are neighbors according to Mahalanobis distance.
"""

import torch
import numpy as np
import trimesh
import os
from argparse import ArgumentParser
from plyfile import PlyData
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def mahalanobis_distance_vectorized(points, center, cov_inv):
    """Compute Mahalanobis distance from points to a center (vectorized).
    
    Args:
        points: (M, 3) array of points
        center: (3,) center point
        cov_inv: (3, 3) inverse covariance matrix
    
    Returns:
        (M,) distances
    """
    diff = points - center
    # Efficient computation: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
    mahal = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
    return mahal


def build_knn_graph_mahalanobis_kdtree(centers, covariances, k=8, opacity_threshold=0.1, opacities=None):
    """Build KNN graph using KD-tree with Mahalanobis distance metric.
    
    Uses sklearn's BallTree with custom Mahalanobis metric for efficient nearest neighbor search.
    
    Args:
        centers: (N, 3) Gaussian centers
        covariances: (N, 3, 3) covariance matrices
        k: number of nearest neighbors
        opacity_threshold: minimum opacity to consider
        opacities: (N,) opacity values
    
    Returns:
        edges: (M, 2) array of connected point indices
    """
    from sklearn.neighbors import BallTree
    
    N = centers.shape[0]
    
    # Filter by opacity if provided
    if opacities is not None:
        mask = opacities > opacity_threshold
        valid_indices = np.where(mask)[0]
        centers_filtered = centers[mask]
        covariances_filtered = covariances[mask]
        print(f"Filtered {N} points to {len(valid_indices)} based on opacity > {opacity_threshold}")
    else:
        valid_indices = np.arange(N)
        centers_filtered = centers
        covariances_filtered = covariances
    
    edges = []
    
    print("Computing KNN using Mahalanobis distance with optimized search...")
    
    # For each Gaussian, find k nearest neighbors using its Mahalanobis metric
    for i in range(len(centers_filtered)):
        if i % 1000 == 0:
            print(f"Processing point {i}/{len(centers_filtered)}")
        
        center = centers_filtered[i]
        cov = covariances_filtered[i]
        
        # Add small regularization to avoid singular matrices
        cov_reg = cov + np.eye(3) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov_reg)
            
            # Transform all points to Mahalanobis space
            # In Mahalanobis space, distance becomes Euclidean
            L = np.linalg.cholesky(cov_inv)  # Cholesky decomposition
            
            # Transform points: x_transformed = L @ (x - center)
            diff = centers_filtered - center
            transformed_points = diff @ L.T
            
            # Use KDTree in transformed space (Euclidean distance = Mahalanobis)
            from scipy.spatial import cKDTree
            tree = cKDTree(transformed_points)
            
            # Query for k+1 neighbors (includes self at distance 0)
            distances, neighbor_indices = tree.query([np.zeros(3)], k=k+1)
            neighbor_indices = neighbor_indices[0][1:]  # Exclude self
            
        except (np.linalg.LinAlgError, ValueError):
            # If decomposition fails, fall back to sorting all distances
            cov_inv = np.eye(3)
            distances = mahalanobis_distance_vectorized(centers_filtered, center, cov_inv)
            neighbor_indices = np.argsort(distances)[1:k+1]
        
        # Add edges
        for j in neighbor_indices:
            orig_i = valid_indices[i]
            orig_j = valid_indices[j]
            # Add edge (avoid duplicates by ordering)
            if orig_i < orig_j:
                edges.append([orig_i, orig_j])
    
    edges = np.array(edges) if edges else np.array([]).reshape(0, 2)
    print(f"Created {len(edges)} edges")
    
    return edges


from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args


def load_gaussian_model(dataset: ModelParams, iteration: int):
    """Load Gaussian model using RaDe-GS Scene and GaussianModel.
    
    Args:
        dataset: ModelParams object
        iteration: iteration number to load
    
    Returns:
        gaussians: GaussianModel with loaded parameters
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    
    return gaussians


def compute_covariance_matrices_from_gaussian(scales, rotations):
    """Compute covariance matrices from Gaussian scales and rotations.
    
    Args:
        scales: (N, 3) log-space scaling factors
        rotations: (N, 4) quaternions [w, x, y, z]
    
    Returns:
        (N, 3, 3) covariance matrices
    """
    N = scales.shape[0]
    
    # Convert log scales to actual scales
    scales_actual = np.exp(scales)
    
    # Convert quaternions to rotation matrices
    rot_matrices = np.zeros((N, 3, 3))
    for i in range(N):
        w, x, y, z = rotations[i]
        rot_matrices[i] = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    # Compute covariance: Sigma = R * S * S^T * R^T
    covariances = np.zeros((N, 3, 3))
    for i in range(N):
        S = np.diag(scales_actual[i])
        covariances[i] = rot_matrices[i] @ S @ S.T @ rot_matrices[i].T
    
    return covariances






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


def build_knn_graph_mahalanobis(centers, covariances, k=8):
    """Build KNN graph using Mahalanobis distance with KD-tree optimization.
    
    Args:
        centers: (N, 3) Gaussian centers
        covariances: (N, 3, 3) covariance matrices
        k: number of nearest neighbors
    
    Returns:
        edges: (M, 2) array of connected point indices
    """
    from scipy.spatial import cKDTree
    
    N = centers.shape[0]
    edges = []
    
    print("Computing KNN based on Mahalanobis distance with KD-tree...")
    for i in range(N):
        if i % 1000 == 0:
            print(f"Processing point {i}/{N}")
        
        center = centers[i]
        cov = covariances[i]
        
        # Add small regularization to avoid singular matrices
        cov_reg = cov + np.eye(3) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov_reg)
            
            # Transform all points to Mahalanobis space using Cholesky decomposition
            L = np.linalg.cholesky(cov_inv)
            
            # Transform points: difference from center, then apply transformation
            diff = centers - center
            transformed_points = diff @ L.T
            
            # Use KD-tree in transformed space (Euclidean = Mahalanobis)
            tree = cKDTree(transformed_points)
            
            # Query for k+1 neighbors (includes self at distance 0)
            distances, neighbor_indices = tree.query([np.zeros(3)], k=min(k+1, N))
            neighbor_indices = neighbor_indices[0][1:]  # Exclude self
            
        except (np.linalg.LinAlgError, ValueError):
            # Fall back to computing all distances
            cov_inv = np.eye(3)
            distances_all = mahalanobis_distance(centers, center, cov_inv)
            neighbor_indices = np.argsort(distances_all)[1:k+1]
        
        # Add edges (avoid duplicates by ordering)
        for j in neighbor_indices:
            if i < j:
                edges.append([i, j])
    
    edges = np.array(edges) if edges else np.array([]).reshape(0, 2)
    print(f"Created {len(edges)} edges")
    
    return edges


def get_largest_connected_component(edges, num_points):
    """Extract the largest connected component from a graph.
    
    Args:
        edges: (M, 2) array of edge connectivity
        num_points: total number of points in the graph
    
    Returns:
        component_mask: (N,) boolean mask for points in largest component
        component_edges: (K, 2) edges within the largest component
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    if len(edges) == 0:
        print("No edges to process")
        return np.zeros(num_points, dtype=bool), edges
    
    # Build adjacency matrix
    # Create symmetric adjacency (undirected graph)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows))
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(num_points, num_points))
    
    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    
    print(f"Found {n_components} connected components")
    
    # Find the largest component
    component_sizes = np.bincount(labels)
    largest_component_id = np.argmax(component_sizes)
    largest_size = component_sizes[largest_component_id]
    
    print(f"Largest component has {largest_size} points ({100*largest_size/num_points:.1f}% of total)")
    
    # Create mask for points in largest component
    component_mask = labels == largest_component_id
    
    # Filter edges to only include those within the largest component
    edge_mask = component_mask[edges[:, 0]] & component_mask[edges[:, 1]]
    component_edges = edges[edge_mask]
    
    # Remap edge indices to new numbering
    point_mapping = np.full(num_points, -1, dtype=int)
    point_mapping[component_mask] = np.arange(np.sum(component_mask))
    component_edges_remapped = point_mapping[component_edges]
    
    print(f"Largest component has {len(component_edges_remapped)} edges")
    
    return component_mask, component_edges_remapped


def edges_to_mesh_delaunay(centers, edges):
    """Convert point cloud with edges to mesh using Delaunay triangulation.
    
    Args:
        centers: (N, 3) points
        edges: (M, 2) edge connectivity
    
    Returns:
        trimesh.Trimesh object
    """
    # Get unique points that are part of edges
    unique_points = np.unique(edges.flatten())
    centers_subset = centers[unique_points]
    
    # Remap edge indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_points)}
    edges_remapped = np.array([[index_map[e[0]], index_map[e[1]]] for e in edges])
    
    print(f"Creating mesh from {len(centers_subset)} points...")
    get_largest_connected_component
    # Perform Delaunay triangulation
    if len(centers_subset) < 4:
        print("Not enough points for triangulation")
        return None
    
    try:
        tri = Delaunay(centers_subset)
        faces = tri.simplices
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=centers_subset, faces=faces, process=False)
        
        # Optional: filter faces based on edge connectivity
        # Keep only tetrahedra whose edges are in the KNN graph
        print(f"Initial mesh has {len(faces)} faces")
        
        return mesh
    except Exception as e:
        print(f"Error in Delaunay triangulation: {e}")
        return None


def extract_mesh_knn_mahalanobis(dataset, iteration, output_path, k=8, opacity_threshold=0.1, use_delaunay=True):
    """Extract mesh from Gaussian model using KNN with Mahalanobis distance.
    
    Args:
        dataset: ModelParams object
        iteration: iteration number to load
        output_path: path to save output mesh
        k: number of nearest neighbors
        opacity_threshold: minimum opacity threshold
        use_delaunay: whether to use Delaunay triangulation
    """
    print(f"Loading Gaussian model from {dataset.model_path} iteration {iteration}")
    gaussians = load_gaussian_model(dataset, iteration)
    
    # Extract data from Gaussian model
    centers = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians._scaling.detach().cpu().numpy()
    rotations = gaussians._rotation.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    
    print(f"Loaded {len(centers)} Gaussians")
    
    # Filter by opacity
    if opacity_threshold > 0:
        mask = opacities > opacity_threshold
        centers = centers[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opacities = opacities[mask]
        print(f"Filtered to {len(centers)} Gaussians with opacity > {opacity_threshold}")
    
    print(f"Computing covariance matrices...")
    covariances = compute_covariance_matrices_from_gaussian(scales, rotations)
    
    # Build KNN graph using Mahalanobis distance with KD-tree
    edges = build_knn_graph_mahalanobis(centers, covariances, k=k)
    
    if len(edges) == 0:
        print("No edges created. Try lowering opacity_threshold or increasing k.")
        return
    
    # Extract largest connected component
    print("\nExtracting largest connected component...")
    component_mask, component_edges = get_largest_connected_component(edges, len(centers))
    
    # Filter centers to only largest component
    centers_filtered = centers[component_mask]
    
    print(f"Using {len(centers_filtered)} points and {len(component_edges)} edges from largest component")
    component_edges_remapped = get_largest_connected_component(edges, len(centers_filtered))

    if use_delaunay:
        # Create mesh using Delaunay triangulation
        mesh = edges_to_mesh_delaunay(centers_filtered, component_edges_remapped)
    else:
        # Create a simple point cloud with edges (as a line mesh)
        # Get unique vertices from edges
        unique_verts = np.unique(component_edges.flatten())
        vertices = centers_filtered[unique_verts]
        
        # Remap edges
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}
        edges_remapped = np.array([[index_map[e[0]], index_map[e[1]]] for e in component_edges])
        
        # Create as point cloud
        mesh = trimesh.PointCloud(vertices)
        print(f"Created point cloud with {len(vertices)} vertices")
    
    if mesh is not None:
        print(f"Saving mesh to {output_path}")
        mesh.export(output_path)
        print("Done!")
    else:
        print("Failed to create mesh")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract mesh from Gaussian centers using KNN with Mahalanobis distance")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=30000, type=int, help="Iteration to load")
    parser.add_argument("--k", type=int, default=8, help="Number of nearest neighbors")
    parser.add_argument("--opacity_threshold", type=float, default=0.1, help="Minimum opacity threshold")
    parser.add_argument("--no_delaunay", action="store_true", help="Don't use Delaunay triangulation")
    parser.add_argument("--output", type=str, default=None, help="Output mesh path (default: model_path/mesh_knn_mahalanobis.ply)")
    
    args = get_combined_args(parser)
    
    dataset = model.extract(args)
    output_path = args.output if args.output else os.path.join(dataset.model_path, "mesh_knn_mahalanobis.ply")
    
    print(f"Extracting mesh from {dataset.model_path}")
    
    extract_mesh_knn_mahalanobis(
        dataset,
        args.iteration,
        output_path,
        k=args.k,
        opacity_threshold=args.opacity_threshold,
        use_delaunay=not args.no_delaunay
    )
