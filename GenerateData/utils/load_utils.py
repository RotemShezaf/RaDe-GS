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
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import GaussianModel only when needed (requires CUDA)
# from scene.gaussian_model import GaussianModel

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
):
    """
    Load Gaussian splat data from PLY file using GaussianModel (requires CUDA).
    
    Args:
        output_folder: Base output folder
        iteration: Iteration number (None = highest available)
        sh_degree: Spherical harmonics degree (default: 3)
    
    Returns:
        GaussianModel instance with loaded data
    """
    # Import here to avoid CUDA requirement when not using this function
    from scene.gaussian_model import GaussianModel
    
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


class GaussianDataCPU:
    """Simple container for Gaussian splat data that can run on CPU."""
    def __init__(self, xyz, scales, rotations, opacities, features_dc=None, features_rest=None, filter_3D=None):
        self._xyz = xyz
        self._scaling = scales
        self._rotation = rotations
        self._opacity = opacities
        self._features_dc = features_dc  # (N, 3, 1) or (N, 3) - DC component of SH
        self._features_rest = features_rest  # (N, 3, SH_coeffs-1) - rest of SH coefficients
        self.filter_3D = filter_3D
    
    def get_xyz(self):
        return self._xyz
    
    def get_scaling(self):
        """Apply exp activation to scales."""
        return np.exp(self._scaling)
    
    def get_rotation(self):
        """Normalize quaternions and ensure first component is positive."""
        # Normalize quaternions
        rots = self._rotation / (np.linalg.norm(self._rotation, axis=1, keepdims=True) + 1e-9)
        # Always set the first component to be positive (standard convention)
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]
        return rots
    
    def get_opacity(self):
        """Apply sigmoid activation to opacity with numerical stability."""
        # Use numerically stable sigmoid to avoid overflow
        # For large positive x: sigmoid(x) ≈ 1
        # For large negative x: sigmoid(x) ≈ 0
        x = self._opacity
        # Clip to avoid overflow in exp
        x_clipped = np.clip(x, -88.0, 88.0)  # exp(88) is near float32 max
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def get_features_dc(self):
        """Get DC component of spherical harmonics (RGB color)."""
        if self._features_dc is None:
            return None
        # Return as (N, 3) if stored as (N, 3, 1)
        if self._features_dc.ndim == 3:
            return self._features_dc.squeeze(-1)
        return self._features_dc
    
    def get_features_rest(self):
        """Get rest of spherical harmonic coefficients."""
        return self._features_rest
    
    def get_features(self):
        """Get all spherical harmonic features concatenated."""
        if self._features_dc is None:
            return None
        if self._features_rest is None:
            # Only DC component
            if self._features_dc.ndim == 3:
                return self._features_dc
            return self._features_dc[:, :, None]
        # Concatenate DC and rest components
        features_dc = self._features_dc if self._features_dc.ndim == 3 else self._features_dc[:, :, None]
        return np.concatenate([features_dc, self._features_rest], axis=2)


def load_gaussian_data_cpu(
    output_folder: Path,
    iteration: Optional[int] = None,
    load_sh_features: bool = False,
    max_sh_degree: int = 3
) -> GaussianDataCPU:
    """
    Load Gaussian splat data from PLY file directly (CPU-compatible, no CUDA required).
    This reads the PLY file structure matching save_ply() in GaussianModel.
    
    Args:
        output_folder: Base output folder
        iteration: Iteration number (None = highest available)
        load_sh_features: Whether to load spherical harmonic features (default: False)
        max_sh_degree: Maximum spherical harmonic degree to load (default: 3)
    
    Returns:
        GaussianDataCPU instance with loaded data (numpy arrays)
    """
    # Find iteration
    point_cloud_dir = output_folder / "point_cloud"
    
    if iteration is None:
        iterations = find_available_iterations(output_folder)
        if not iterations:
            raise FileNotFoundError(f"No iterations found in {point_cloud_dir}")
        iteration = max(iterations)
    
    ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
    print(f"Loading Gaussian data (CPU mode) from: {ply_path}")
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    # Load PLY file
    plydata = PlyData.read(str(ply_path))
    vertex = plydata.elements[0]
    
    # Extract positions (x, y, z)
    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"])
    ], axis=1).astype(np.float32)
    
    # Extract opacities
    opacities = np.asarray(vertex["opacity"])[..., np.newaxis].astype(np.float32)
    
    # Extract scales (scale_0, scale_1, scale_2)
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(vertex[attr_name])
    
    # Extract rotations (rot_0, rot_1, rot_2, rot_3 - quaternions)
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rotations = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rotations[:, idx] = np.asarray(vertex[attr_name])
    
    # Extract spherical harmonic features if requested
    features_dc = None
    features_rest = None
    if load_sh_features:
        # Extract DC component (f_dc_0, f_dc_1, f_dc_2)
        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"])
        
        # Extract rest of SH coefficients if they exist
        extra_f_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
        if extra_f_names:
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
            # Calculate expected number of coefficients
            expected_coeffs = 3 * (max_sh_degree + 1) ** 2 - 3
            
            if len(extra_f_names) >= expected_coeffs:
                features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
                for idx, attr_name in enumerate(extra_f_names[:expected_coeffs]):
                    features_extra[:, idx] = np.asarray(vertex[attr_name])
                # Reshape to (N, 3, SH_coeffs-1)
                features_rest = features_extra.reshape((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    
    # Extract filter_3D if available
    filter_3D = None
    if "filter_3D" in [p.name for p in vertex.properties]:
        filter_3D = np.asarray(vertex["filter_3D"])[..., np.newaxis].astype(np.float32)
    
    # Create CPU-compatible Gaussian data container
    gaussian_data = GaussianDataCPU(xyz, scales, rotations, opacities, features_dc, features_rest, filter_3D)
    
    # Get basic stats (after activations)
    activated_scales = gaussian_data.get_scaling()
    activated_opacities = gaussian_data.get_opacity()
    activated_rotations = gaussian_data.get_rotation()
    
    print(f"  Loaded {len(xyz)} Gaussians")
    print(f"  Position range: [{xyz.min():.4f}, {xyz.max():.4f}]")
    print(f"  Scale range: [{activated_scales.min():.4f}, {activated_scales.max():.4f}]")
    print(f"  Opacity range: [{activated_opacities.min():.4f}, {activated_opacities.max():.4f}]")
    print(f"  Rotation (normalized) range: [{activated_rotations.min():.4f}, {activated_rotations.max():.4f}]")
    
    if load_sh_features and features_dc is not None:
        print(f"  Loaded SH features: DC shape {features_dc.shape}", end="")
        if features_rest is not None:
            print(f", Rest shape {features_rest.shape}")
        else:
            print(" (DC only)")
    
    return gaussian_data


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

