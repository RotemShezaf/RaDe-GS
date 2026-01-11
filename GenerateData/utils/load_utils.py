#!/usr/bin/env python3
from pathlib import Path
import sys
from typing import List
from pathlib import Path
import numpy as np
from plyfile import PlyData
from typing import Optional, Tuple
import trimesh
from typing import Tuple, Optional
import open3d as o3d

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#/home/rotem.shezaf/RaDe-GS/TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_02/output/sparse
def find_available_iterations(output_folder: Path) -> list[int]:
    """
    Find all available iteration directories in the output folder.
    
    Args:
        output_folder: Base output folder (e.g., output/polynomial/Paraboloid)
    
    Returns:
        Sorted list of available iteration numbers
    """
    point_cloud_dir = output_folder / "point_cloud"
    if not point_cloud_dir.exists():
        return []
    #/home/rotem.shezaf/RaDe-GS/TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_02/output/sparse/point_cloud
    iterations = []
    
    for iter_dir in point_cloud_dir.glob("iteration_*"):
        
        if iter_dir.is_dir():
            try:
                iter_num = int(iter_dir.name.split("_")[1])
                # Check if point_cloud.ply exists
                if (iter_dir / "point_cloud.ply").exists():
                    iterations.append(iter_num)
            except (ValueError, IndexError):
                continue
    
    return sorted(iterations)



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


def load_gaussian_data(
    output_folder: Path,
    iteration: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Gaussian splat data from PLY file.
    
    Args:
        output_folder: Base output folder
        iteration: Iteration number (None = highest available)
    
    Returns:
        Tuple of (positions, scales, rotations, opacities)
    """
    # Find iteration
    point_cloud_dir = output_folder / "point_cloud"
    
    if iteration is None:
        iterations = find_available_iterations(output_folder)
        iteration = max(iterations)
    
    ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
    print(f"Loading Gaussian data from: {ply_path}")
    
    plydata = PlyData.read(str(ply_path))
    
    
    ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
    print(f"Loading Gaussian data from: {ply_path}")
    
    plydata = PlyData.read(str(ply_path))
    
    # Extract positions
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1)
    
    # Extract scales (stored as log scales in PLY)
    scales = np.stack((
        np.asarray(plydata.elements[0]["scale_0"]),
        np.asarray(plydata.elements[0]["scale_1"]),
        np.asarray(plydata.elements[0]["scale_2"])
    ), axis=1)
    scales = np.exp(scales)  # Convert from log space
    
    # Extract rotations (quaternions)
    rotations = np.stack((
        np.asarray(plydata.elements[0]["rot_0"]),
        np.asarray(plydata.elements[0]["rot_1"]),
        np.asarray(plydata.elements[0]["rot_2"]),
        np.asarray(plydata.elements[0]["rot_3"])
    ), axis=1)
    
    # Extract opacities
    opacities = np.asarray(plydata.elements[0]["opacity"])
    
    print(f"  Loaded {len(xyz)} Gaussians")
    print(f"  Position range: [{xyz.min():.4f}, {xyz.max():.4f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    
    return xyz, scales, rotations, opacities


def load_ground_truth_mesh(
    data_root: Path,
    surface: str,
    level: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the ground truth mesh at specified resolution level.
    
    Args:
        data_root: Base directory containing polynomial mesh data
        surface: Surface type (e.g., 'Paraboloid', 'Saddle', 'HyperbolicParaboloid')
        level: Resolution level (0 = highest resolution)
    
    Returns:
        Tuple of (vertices, faces) arrays
    """
    print(f"\n{'='*80}")
    print(f"Loading Ground Truth Mesh")
    print(f"{'='*80}")
    
    surface_dir = data_root / surface
    mesh_candidates = sorted(surface_dir.glob(f"mesh_level{level}_*.ply"))
    
    if not mesh_candidates:
        raise FileNotFoundError(
            f"No mesh found for surface={surface} level={level} in {surface_dir}"
        )
    
    mesh_path = mesh_candidates[0]
    print(f"Path: {mesh_path}")
    
    vertices, faces = load_ply(str(mesh_path))
    
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Vertex range:")
    print(f"    X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"    Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"    Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    
    return vertices, faces

