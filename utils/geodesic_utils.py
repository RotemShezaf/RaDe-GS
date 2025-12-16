"""
Utility functions for geodesic distance computation on triangular meshes.
Provides functions for loading/saving PLY files, building geodesic meshes,
computing distances, and generating test meshes.
"""

import numpy as np
import trimesh
import pygeodesic.geodesic as geodesic
from typing import Tuple, Optional
import open3d as o3d


def load_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a PLY mesh file and return vertices and faces.
    
    Args:
        path: Path to the PLY file
        
    Returns:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
    """
    mesh = trimesh.load(path, process=False)
    
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def save_ply(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Save a mesh to a PLY file.
    
    Args:
        path: Output path for the PLY file
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(path)
    print(f"Mesh saved to {path}")


def build_geodesic_mesh(vertices: np.ndarray, faces: np.ndarray) -> geodesic.PyGeodesicAlgorithmExact:
    """
    Build a geodesic mesh object from vertex and face arrays.
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
        
    Returns:
        Geodesic mesh object ready for distance computation
    """
    # Flatten arrays for the geodesic library
    points = vertices.flatten().tolist()
    faces_flat = faces.flatten().tolist()
    
    # Create geodesic mesh
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces_flat)
    
    return geoalg


def compute_exact_geodesic(vertices: np.ndarray,
                           faces: np.ndarray,
                           source_id: int,
                           target_ids: Optional[list] = None) -> np.ndarray:
    """
    Compute exact geodesic distances using the MMP algorithm.

    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
        source_id: Index of the source vertex
        target_ids: Optional list of target vertex indices. If None, computes to all vertices.

    Returns:
        Array of geodesic distances from source to all/specified vertices
    """
    points = vertices.tolist()
    faces_flat = faces.tolist()
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces_flat)

    if target_ids is not None and len(target_ids) == 1:
        # Compute geodesic distance and path between source and single target
        distance, path = geoalg.geodesicDistance(source_id, target_ids[0])
        return np.array([distance])
    else:
        # Compute distances from source to all/specified vertices
        source_indices = [source_id]
        distances, _ = geoalg.geodesicDistances(source_indices, target_ids)
        return np.array(distances)


def compute_fmm_geodesic(vertices: np.ndarray,
                        faces: np.ndarray,
                        source_id: int,
                        target_ids: Optional[list] = None) -> np.ndarray:
    """
    Compute approximate geodesic distances using the Fast Marching Method.
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
        source_id: Index of the source vertex
        target_ids: Optional list of target vertex indices. If None, computes to all vertices.
        
    Returns:
        Array of geodesic distances from source to all/specified vertices
    """
    # Flatten arrays for the geodesic library
    points = vertices.tolist()
    faces_flat = faces.tolist()
    
    # Create Fast Marching Method algorithm
    geoalg = geodesic.PyGeodesicAlgorithmFastMarching(points, faces_flat)
    
    distances, _ = geoalg.geodesicDistances([source_id], target_ids)
    
    return np.array(distances)


def post_process_mesh(vertices: np.ndarray, faces: np.ndarray, cluster_to_keep: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-process a mesh: normalize scale, center, and clean up disconnected components.
    Similar to mesh_utils.post_process_mesh but adapted for geodesic computation.
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of triangle indices
        cluster_to_keep: Number of largest clusters to keep
        
    Returns:
        Tuple of (processed_vertices, faces)
    """
    import copy
    print("Post-processing mesh...")
    print(f"  - Original: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to Open3D mesh for cleaning
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    
    # Remove disconnected components
    mesh_clean = copy.deepcopy(mesh_o3d)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_clean.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        n_cluster = np.sort(cluster_n_triangles.copy())[-min(cluster_to_keep, len(cluster_n_triangles))]
        n_cluster = max(n_cluster, 50)  # Filter meshes smaller than 50 triangles
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_clean.remove_triangles_by_mask(triangles_to_remove)
    
    mesh_clean.remove_unreferenced_vertices()
    mesh_clean.remove_degenerate_triangles()
    
    # Get cleaned vertices and faces
    vertices_clean = np.asarray(mesh_clean.vertices)
    faces_clean = np.asarray(mesh_clean.triangles)
    
    # Center the mesh
    centroid = np.mean(vertices_clean, axis=0)
    vertices_centered = vertices_clean - centroid
    
    # Normalize scale to fit in unit sphere
    max_dist = np.max(np.linalg.norm(vertices_centered, axis=1))
    if max_dist > 0:
        vertices_normalized = vertices_centered / max_dist
    else:
        vertices_normalized = vertices_centered
    
    print(f"  - Cleaned: {len(vertices_clean)} vertices, {len(faces_clean)} faces")
    print(f"  - Centered mesh at origin")
    print(f"  - Normalized scale (max radius: {max_dist:.4f} -> 1.0)")
    
    return vertices_normalized, faces_clean


def generate_test_mesh(mesh_type: str = "sphere") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple test mesh for debugging using trimesh primitives.
    
    Args:
        mesh_type: Type of mesh to generate ("sphere" or "plane")
        
    Returns:
        Tuple of (vertices, faces)
    """
    if mesh_type == "sphere":
        # Create an icosphere using trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        print(f"Generated test sphere: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    elif mesh_type == "plane":
        # Create a grid plane using trimesh
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 0.01])
        # Flatten to make it more like a plane
        mesh.vertices[:, 2] = 0
        # Subdivide for more vertices
        mesh = mesh.subdivide()
        print(f"Generated test plane: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)
