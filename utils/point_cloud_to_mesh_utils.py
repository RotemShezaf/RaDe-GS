"""
Point Cloud to Mesh Extraction Utility

This module provides functions to extract meshes from point clouds using
compatible methods from the existing RaDe-GS utilities.
"""

import numpy as np
import torch
import open3d as o3d
import trimesh
from skimage import measure
from tqdm import tqdm
from utils.graphics_utils import BasicPointCloud


def point_cloud_to_mesh_poisson(
    pcd: BasicPointCloud,
    depth=9,
    width=0,
    scale=1.1,
    linear_fit=False,
    density_threshold=0.01
):
    """
    Extract mesh from point cloud using Poisson surface reconstruction.
    
    Args:
        pcd: BasicPointCloud with points, colors, and normals
        depth: Maximum depth of the tree used for reconstruction (higher = more detail)
        width: Target width of the finest level octree cells
        scale: Ratio between the diameter of the cube used for reconstruction and the diameter of the samples bounding cube
        linear_fit: If true, the reconstructor will use linear interpolation
        density_threshold: Threshold for filtering low-density vertices
        
    Returns:
        o3d.geometry.TriangleMesh: Reconstructed mesh
    """
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    
    # Estimate normals if not provided
    if pcd.normals is None or np.all(pcd.normals == 0):
        print("Estimating normals...")
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d_pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals)
    
    print(f"Running Poisson reconstruction with depth={depth}...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    # Filter out low-density vertices
    if density_threshold > 0:
        print(f"Filtering vertices with density < {density_threshold}...")
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, density_threshold)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh.compute_vertex_normals()
    
    return mesh


def point_cloud_to_mesh_ball_pivoting(
    pcd: BasicPointCloud,
    radii=[0.005, 0.01, 0.02, 0.04]
):
    """
    Extract mesh from point cloud using Ball Pivoting algorithm.
    
    Args:
        pcd: BasicPointCloud with points, colors, and normals
        radii: List of radii for ball pivoting
        
    Returns:
        o3d.geometry.TriangleMesh: Reconstructed mesh
    """
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    
    # Estimate normals if not provided
    if pcd.normals is None or np.all(pcd.normals == 0):
        print("Estimating normals...")
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d_pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals)
    
    print(f"Running Ball Pivoting with radii={radii}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        o3d_pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    mesh.compute_vertex_normals()
    
    return mesh


def point_cloud_to_mesh_alpha_shape(
    pcd: BasicPointCloud,
    alpha=0.03
):
    """
    Extract mesh from point cloud using Alpha Shape algorithm.
    
    Args:
        pcd: BasicPointCloud with points, colors, and normals
        alpha: Parameter for alpha shape (smaller = tighter fit)
        
    Returns:
        o3d.geometry.TriangleMesh: Reconstructed mesh
    """
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    
    print(f"Running Alpha Shape with alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        o3d_pcd, alpha
    )
    
    mesh.compute_vertex_normals()
    
    return mesh


def point_cloud_to_mesh_marching_cubes(
    pcd: BasicPointCloud,
    voxel_size=0.01,
    sdf_trunc=0.04,
    resolution=128
):
    """
    Extract mesh from point cloud using volumetric approach with marching cubes.
    Similar to the TSDF fusion method used in mesh_utils.py
    
    Args:
        pcd: BasicPointCloud with points, colors, and normals
        voxel_size: Size of voxels in the volume
        sdf_trunc: Truncation distance for signed distance function
        resolution: Resolution of the voxel grid
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh
    """
    points = pcd.points
    colors = pcd.colors if pcd.colors is not None else np.ones_like(points)
    
    # Compute bounding box
    bbox_min = points.min(axis=0) - sdf_trunc * 2
    bbox_max = points.max(axis=0) + sdf_trunc * 2
    
    print(f"Computing volumetric SDF with resolution {resolution}...")
    
    # Create grid
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Compute SDF using nearest neighbor distance
    print("Computing signed distance field...")
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances, indices = tree.query(grid_points, k=1)
    
    # Reshape to grid
    sdf = distances.reshape(resolution, resolution, resolution)
    
    # Run marching cubes
    print("Running marching cubes...")
    vertices, faces, normals, _ = measure.marching_cubes(
        sdf,
        level=voxel_size,
        spacing=(
            (bbox_max[0] - bbox_min[0]) / (resolution - 1),
            (bbox_max[1] - bbox_min[1]) / (resolution - 1),
            (bbox_max[2] - bbox_min[2]) / (resolution - 1),
        )
    )
    
    vertices = vertices + bbox_min
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    
    # Color vertices by nearest point cloud color
    if colors is not None:
        _, color_indices = tree.query(vertices, k=1)
        vertex_colors = colors[color_indices]
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    return mesh


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts.
    Compatible with the existing post_process_mesh in mesh_utils.py
    
    Args:
        mesh: o3d.geometry.TriangleMesh to process
        cluster_to_keep: Minimum number of triangles to keep a cluster
        
    Returns:
        o3d.geometry.TriangleMesh: Processed mesh
    """
    import copy
    print(f"Post-processing mesh to filter clusters (min size: {cluster_to_keep})...")
    
    mesh_processed = copy.deepcopy(mesh)
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_processed.cluster_connected_triangles()
        )
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    # Find threshold
    n_cluster = np.sort(cluster_n_triangles)[-min(cluster_to_keep, len(cluster_n_triangles)):]
    n_cluster = max(n_cluster.min(), 50)  # Filter meshes smaller than 50
    
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_processed.remove_triangles_by_mask(triangles_to_remove)
    mesh_processed.remove_unreferenced_vertices()
    mesh_processed.remove_degenerate_triangles()
    
    print(f"Vertices before: {len(mesh.vertices)}, after: {len(mesh_processed.vertices)}")
    
    return mesh_processed


def extract_mesh_from_point_cloud(
    input_path,
    output_path,
    method='poisson',
    post_process=True,
    **kwargs
):
    """
    Main function to extract mesh from a point cloud file.
    
    Args:
        input_path: Path to input point cloud (.ply, .pcd, .xyz, etc.)
        output_path: Path to save the output mesh (.ply)
        method: Method to use ('poisson', 'ball_pivoting', 'alpha_shape', 'marching_cubes')
        post_process: Whether to post-process the mesh
        **kwargs: Additional arguments for the specific method
        
    Returns:
        mesh: The extracted mesh
    """
    print(f"Loading point cloud from {input_path}...")
    
    # Load point cloud
    if input_path.endswith('.ply'):
        pcd_o3d = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd_o3d.points)
        colors = np.asarray(pcd_o3d.colors) if pcd_o3d.has_colors() else None
        normals = np.asarray(pcd_o3d.normals) if pcd_o3d.has_normals() else None
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    # Create BasicPointCloud
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    print(f"Point cloud has {len(points)} points")
    
    # Extract mesh using specified method
    if method == 'poisson':
        mesh = point_cloud_to_mesh_poisson(pcd, **kwargs)
        is_o3d_mesh = True
    elif method == 'ball_pivoting':
        mesh = point_cloud_to_mesh_ball_pivoting(pcd, **kwargs)
        is_o3d_mesh = True
    elif method == 'alpha_shape':
        mesh = point_cloud_to_mesh_alpha_shape(pcd, **kwargs)
        is_o3d_mesh = True
    elif method == 'marching_cubes':
        mesh = point_cloud_to_mesh_marching_cubes(pcd, **kwargs)
        is_o3d_mesh = False
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Post-process if requested
    if post_process and is_o3d_mesh:
        mesh = post_process_mesh(mesh)
    
    # Save mesh
    print(f"Saving mesh to {output_path}...")
    if is_o3d_mesh:
        o3d.io.write_triangle_mesh(output_path, mesh)
    else:
        mesh.export(output_path)
    
    print("Done!")
    
    return mesh


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract mesh from point cloud")
    parser.add_argument("--input", type=str, required=True, help="Input point cloud path (.ply)")
    parser.add_argument("--output", type=str, required=True, help="Output mesh path (.ply)")
    parser.add_argument(
        "--method", 
        type=str, 
        default='poisson',
        choices=['poisson', 'ball_pivoting', 'alpha_shape', 'marching_cubes'],
        help="Mesh extraction method"
    )
    parser.add_argument("--depth", type=int, default=9, help="Poisson depth (for poisson method)")
    parser.add_argument("--alpha", type=float, default=0.03, help="Alpha parameter (for alpha_shape method)")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size (for marching_cubes method)")
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution (for marching_cubes method)")
    parser.add_argument("--no-post-process", action="store_true", help="Skip post-processing")
    
    args = parser.parse_args()
    
    # Prepare kwargs based on method
    kwargs = {}
    if args.method == 'poisson':
        kwargs['depth'] = args.depth
    elif args.method == 'alpha_shape':
        kwargs['alpha'] = args.alpha
    elif args.method == 'marching_cubes':
        kwargs['voxel_size'] = args.voxel_size
        kwargs['resolution'] = args.resolution
    
    extract_mesh_from_point_cloud(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        post_process=not args.no_post_process,
        **kwargs
    )



