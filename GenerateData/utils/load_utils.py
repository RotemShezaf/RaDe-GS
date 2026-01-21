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
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scene.gaussian_model import GaussianModel

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
    iteration: Optional[int] = None,
    sh_degree: int = 3
) -> GaussianModel:
    """
    Load Gaussian splat data from PLY file using GaussianModel.
    
    Args:
        output_folder: Base output folder
        iteration: Iteration number (None = highest available)
        sh_degree: Spherical harmonics degree (default: 3)
    
    Returns:
        GaussianModel instance with loaded data
    """
    # Find iteration
    point_cloud_dir = output_folder / "point_cloud"
    
    if iteration is None:
        iterations = find_available_iterations(output_folder)
        iteration = max(iterations)
    
    ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
    print(f"Loading Gaussian data from: {ply_path}")
    
    # Create GaussianModel and load from ply
    gaussian_model = GaussianModel(sh_degree=sh_degree)
    gaussian_model.load_ply(str(ply_path))
    
    # Get basic stats
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    scales = gaussian_model.get_scaling.detach().cpu().numpy()
    
    print(f"  Loaded {len(xyz)} Gaussians")
    print(f"  Position range: [{xyz.min():.4f}, {xyz.max():.4f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
    
    return gaussian_model


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


def extract_surface_and_texture_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract surface name and texture information from dataset path.
    
    Expected path patterns:
    - .../blue_texture/Saddle/level_02/output/...
    - .../Polynomial/SyntheticColmapData/red_texture/Paraboloid/...
    - .../output/polynomial/Paraboloid
    
    Args:
        path: Path to parse
    
    Returns:
        Tuple of (surface_name, texture) or (None, None) if not found
    """
    parts = path.parts
    surface_name = None
    texture = None
    
    # Common surface names to look for
    surface_names = ['Paraboloid', 'Saddle', 'HyperbolicParaboloid', 'Sphere', 'Torus']
    
    # Look for surface name in path parts
    for i, part in enumerate(parts):
        # Check if this part is a known surface name
        if part in surface_names:
            surface_name = part
            
            # Look backwards for texture (usually 1-2 parts before surface)
            for j in range(max(0, i-3), i):
                if 'texture' in parts[j].lower():
                    texture = parts[j]
                    break
            break
    
    # If not found by exact match, try to infer from path structure
    if surface_name is None:
        # Look for patterns like "polynomial/Paraboloid" or "output/Saddle"
        for i, part in enumerate(parts):
            if part.lower() in ['polynomial', 'synthetic', 'syntheticcolmapdata']:
                # Next capitalized word might be the surface
                for j in range(i+1, min(len(parts), i+4)):
                    if parts[j] and parts[j][0].isupper():
                        surface_name = parts[j]
                        break
    
    # Use last part as fallback for surface name
    if surface_name is None and len(parts) > 0:
        surface_name = parts[-1]
    
    return surface_name, texture

