"""Generate a synthetic COLMAP-style dataset from the polynomial meshes.

This script creates a synthetic dataset that mimics the structure of a real COLMAP reconstruction.
It loads meshes produced by ``GenerateRawPolynomialMesh.py`` and renders them from multiple 
virtual cameras placed on an orbit.

DUAL-LEVEL MESH SYSTEM:
-----------------------
The script uses two separate mesh resolution levels for different purposes:

1. IMAGE_MESH_LEVEL (--image_mesh_level): 
   - Used for rendering the training images from virtual cameras
   - Can be lower resolution for faster rendering without quality loss
   - Affects rendering speed but not the final COLMAP point cloud
   - Default: level 1 (medium resolution)

2. COLMAP_LEVEL (--colmap_level):
   - Used for generating the COLMAP sparse point cloud (points3D.*)
   - Should be higher resolution to provide denser geometric information
   - Affects the quality and density of the point cloud used for reconstruction
   - Default: level 2 (high resolution)

This separation allows you to optimize rendering speed independently from point cloud quality.
For example, you can render images from a simpler mesh while providing a dense point cloud
for accurate geometric reconstruction.

OUTPUT STRUCTURE:
-----------------
For every surface, the script writes:
- ``images/``: JPG renders with simple shading from virtual cameras
- ``sparse/0/cameras.txt`` + ``cameras.bin``: Camera intrinsics in COLMAP format
- ``sparse/0/images.txt`` + ``images.bin``: Camera extrinsics (poses) in COLMAP format
- ``sparse/0/points3D.{txt,bin,ply}``: Sparse 3D point cloud sampled from the mesh

The resulting folder can be consumed directly by ``train.py`` via the standard
``-s <dataset_root>`` argument, just like a real COLMAP reconstruction.
"""


from __future__ import annotations

# Ensure project root is in sys.path for module imports
import os
import sys
from pathlib import Path

# CRITICAL: Set rendering backend BEFORE importing Open3D to prevent EGL segfault
# OSMesa provides software rendering that works reliably in headless environments
os.environ['OPEN3D_CPU_RENDERING'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import imageio.v2 as imageio
import cv2
import numpy as np
import struct
import trimesh
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial_normal
from GenerateData.convert_to_logfile import convert_COLMAP_to_log
from GenerateData.utils.camera_utils import CameraSample, CameraIntrinsics
from GenerateData.utils.io_utils import (   
    _ensure_dirs,
    _load_texture_image,
    _find_texture_file,
    _validate_dataset,
    _write_cameras_bin,
    _write_cameras_txt,
    _write_images_bin,

    _write_images_txt,
    _write_points3d_bin,
    _write_points3d_ply,
    _write_points3d_txt,
    _create_training_script
)

from GenerateData.utils.rendering_utils import(
    _render_images,
    _generate_uv_coordinates,
    _sample_points,
    _sample_colors_from_texture,
)

from GenerateData.utils.camera_utils import (
    _compute_camera_centers,
    _rotation_matrix_from_quaternion,
    _random_rotation_matrix,
    _fibonacci_sphere,
)



from scene.colmap_loader import (
    rotmat2qvec,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)

SURFACES = ("Paraboloid", "Saddle", "HyperbolicParaboloid")





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render polynomial meshes into a synthetic COLMAP dataset")
    parser.add_argument("--surface", choices=SURFACES, help="Surface type to render", default=SURFACES[0])
    
    # Mesh resolution levels: This script uses a two-level system for flexibility
    parser.add_argument(
        "--colmap_level", 
        type=int, 
        default=2,
        help="Resolution level for COLMAP point cloud sampling. Higher values = denser mesh used for points3D.* files. "
             "This mesh is sampled to generate the sparse point cloud that COLMAP would typically produce."
    )
    parser.add_argument(
        "--image_mesh_level", 
        type=int, 
        default=1,
        help="Resolution level for rendering images. Can be lower resolution than colmap_level for faster rendering. "
             "The mesh at this level is used to generate the synthetic training images from virtual cameras."
    )
    
    parser.add_argument("--data_root", type=str, default="TrainData/Polynomial/raw", help="Base directory containing generated surfaces")
    parser.add_argument("--output_root", type=str, default="TrainData/Polynomial/SyntheticColmapData", help="Destination root for the rendered dataset")
    parser.add_argument("--num_views", type=int, default=50, help="Number of rendered viewpoints along the orbit")
    parser.add_argument("--image_width", type=int, default=960, help="Rendered image width in pixels")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height in pixels")
    parser.add_argument("--vertical_fov", type=float, default=45.0, help="Camera vertical field of view in degrees")
    parser.add_argument("--orbit_radius_scale", type=float, default=1.4, help="Orbit radius as a multiple of the mesh bounding radius")
    parser.add_argument("--elevation_deg", type=float, default=25.0, help="Camera elevation angle in degrees")
    parser.add_argument("--camera_radius", type=float, default=6, help="Absolute radius for camera placement (overrides orbit_radius_scale)")
    parser.add_argument(
        "--camera_distribution",
        choices=["uniform_sphere", "orbit"],
        default="uniform_sphere",
        help="Strategy for sampling camera centers",
    )
    parser.add_argument("--light_intensity", type=float, default=3.5, help="Directional light intensity for pyrender")
    parser.add_argument("--points3d_thresh", type=float, default=None, help="Downsample density threshold for COLMAP points3D (minimum distance between points). If None, no downsampling is performed.")
    parser.add_argument("--color_scheme", choices=["height", "surface"], default="height", help="Fallback color scheme when the mesh has no vertex colors")
    parser.add_argument("--texture_folder", type=str, default="textures", help="Folder name containing texture images (relative to GenerateData directory)")
    parser.add_argument("--texture_name", type=str, default="colors", help="Texture image filename (extension optional, will search for .png, .jpg, .jpeg, .bmp, .tif)")
    parser.add_argument("--use_decoupled_appearance", action="store_true", help="Vary lighting across view groups to simulate appearance variations for training appearance networks")
    parser.add_argument("--light_id", type=int, default=None, help="Fixed lighting configuration ID (0-4) for consistent lighting across all views. 0=default balanced, 1-4=variations. Ignored if --use_decoupled_appearance is set.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for viewpoint shuffling and point sampling")
    return parser.parse_args()




def _load_mesh(data_root: Path, surface: str, level: int, color_scheme: str, texture_folder: str, texture_name: str) -> o3d.geometry.TriangleMesh:
    mesh_candidates = sorted((data_root / surface).glob(f"mesh_level{level}_*.ply"))
    if not mesh_candidates:
        raise FileNotFoundError(f"No mesh for surface={surface} level={level} under {data_root/surface}")
    
    # Load with trimesh first to handle colors
    trimesh_mesh = trimesh.load(mesh_candidates[0], process=False)
    if not isinstance(trimesh_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a trimesh.Trimesh, got {type(trimesh_mesh)} from {mesh_candidates[0]}")
    
    # Set up colors if needed
    if trimesh_mesh.visual is None or not hasattr(trimesh_mesh.visual, "vertex_colors") or len(trimesh_mesh.visual.vertex_colors) == 0:
        trimesh_mesh.visual.vertex_colors = _build_colors(np.asarray(trimesh_mesh.vertices), surface, color_scheme)
    
    # Convert to Open3D mesh
    vertices = np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    faces = np.asarray(trimesh_mesh.faces, dtype=np.int32)
    
    # Generate UV coordinates
    uv_coords = _generate_uv_coordinates(vertices)
    
    # Sample colors from texture if available, otherwise use procedural colors
    texture_folder_path = Path(__file__).parent / texture_folder
    texture_path = _find_texture_file(texture_folder_path, texture_name)
    if texture_path is not None:
        print(f"Sampling vertex colors from texture: {texture_path}")
        colors = _sample_colors_from_texture(texture_path, uv_coords)
    else:
        print(f"Texture not found in {texture_folder_path}, using procedural colors")
        colors = np.asarray(trimesh_mesh.visual.vertex_colors[:, :3], dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
    
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # For triangle_uvs, we need 3 UV coordinates per triangle
    # Flatten faces and index into uv_coords to get per-triangle-vertex UVs
    triangle_uvs = uv_coords[faces.flatten()]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    
    return mesh


def _load_normals_if_available(data_root: Path, surface: str, level: int, mesh: o3d.geometry.TriangleMesh) -> None:
    """Load analytical normals saved by GenerateRawPolynomialMesh if present, or calculate them.

    Looks for normals_level{level}_*.ply next to the meshes. The normals file has the same
    structure as the mesh file but with normals stored as vertex data.
    If not found, calculates normals from the mesh.
    Modifies the mesh in place by setting vertex_normals.
    """
    surface_dir = data_root / surface
    candidates = sorted(surface_dir.glob(f"normals_level{level}_*.ply"))
    
    if candidates:
        try:
            # Load the normals mesh - it should have the same vertices as the original mesh
            normals_mesh = trimesh.load(str(candidates[0]), process=False)
            if isinstance(normals_mesh, trimesh.Trimesh):
                # Try to get normals from vertex_normals attribute
                if hasattr(normals_mesh, 'vertex_normals') and len(normals_mesh.vertex_normals) > 0:
                    normals = np.asarray(normals_mesh.vertex_normals, dtype=np.float64)
                    if len(normals) == len(mesh.vertices):
                        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
                        print(f"Loaded analytical normals for {surface} level {level}")
                        return
        except Exception as e:
            print(f"Warning: Failed to load normals from {candidates[0]}: {e}")
    
    # Fall back to calculating normals from the mesh
    print(f"Calculating normals for {surface} level {level}")
    mesh.compute_vertex_normals()


def _build_colors(vertices: np.ndarray, surface: str, scheme: str) -> np.ndarray:
    palette = {
        "Paraboloid": np.array([0.95, 0.45, 0.1]),
        "Saddle": np.array([0.2, 0.7, 0.95]),
        "HyperbolicParaboloid": np.array([0.3, 0.9, 0.35]),
    }
    colors = np.empty((len(vertices), 4), dtype=np.uint8)
    if scheme == "surface":
        base = palette.get(surface, np.array([0.8, 0.8, 0.8]))
        colors[:, :3] = np.clip(base * 255, 0, 255).astype(np.uint8)
    else:
        z = vertices[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-6)
        grad = np.stack([z_norm, 0.4 + 0.4 * (1 - np.abs(z_norm - 0.5)), 1.0 - z_norm], axis=1)
        colors[:, :3] = np.clip(grad * 255, 0, 255).astype(np.uint8)
    colors[:, 3] = 255
    return colors




def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    # Load mesh at image_mesh_level for rendering synthetic training images
    # This can be a lower-resolution mesh for faster rendering without sacrificing quality
    image_mesh = _load_mesh(data_root, args.surface, args.image_mesh_level, args.color_scheme, args.texture_folder, args.texture_name)

    # Load analytical normals if available, otherwise calculate from mesh
    _load_normals_if_available(data_root, args.surface, args.image_mesh_level, image_mesh)

    # Output directory includes texture, level, and lighting mode subdirectory
    base_dir = Path(args.output_root) / f"{args.texture_name}_texture" / args.surface / f"level_{args.colmap_level:02d}"
    
    # Determine subdirectory based on lighting mode
    if args.use_decoupled_appearance:
        dataset_dir = base_dir / "decoupled_appearance"
    elif args.light_id is not None:
        dataset_dir = base_dir / f"light_{args.light_id}"
    else:
        dataset_dir = base_dir / "default_light"
    
    _ensure_dirs(dataset_dir)

    # Compute camera positions and render images using the image_mesh_level mesh
    centers, targets = _compute_camera_centers(image_mesh, args.num_views, args, args.seed)
    samples, camera_intrinsics = _render_images(image_mesh, centers, targets, args, dataset_dir, args.texture_folder, args.texture_name)

    # Load mesh at colmap_level for generating COLMAP point cloud (points3D.*)
    # This is typically a higher-resolution mesh to provide a denser point cloud
    # that better represents the surface geometry for reconstruction
    colmap_mesh = _load_mesh(data_root, args.surface, args.colmap_level, args.color_scheme, args.texture_folder, args.texture_name)
    # Load analytical normals if available, otherwise calculate from mesh
    _load_normals_if_available(data_root, args.surface, args.colmap_level, colmap_mesh)
    points_xyz_colmap, points_rgb_colmap, points_normals_colmap = _sample_points(colmap_mesh, args.points3d_thresh, args.seed)

    sparse_dir = dataset_dir / "sparse" / "0"
    _write_cameras_txt(sparse_dir / "cameras.txt", camera_intrinsics)
    _write_cameras_bin(sparse_dir / "cameras.bin", camera_intrinsics)
    _write_images_txt(sparse_dir / "images.txt", samples)
    _write_images_bin(sparse_dir / "images.bin", samples)
    _write_points3d_txt(sparse_dir / "points3D.txt", points_xyz_colmap, points_rgb_colmap)
    _write_points3d_bin(sparse_dir / "points3D.bin", points_xyz_colmap, points_rgb_colmap)
    _write_points3d_ply(sparse_dir / "points3D.ply", points_xyz_colmap, points_rgb_colmap, points_normals_colmap)

    # Generate COLMAP_SFM.log file using convert_COLMAP_to_log
    logfile_name = f"{args.surface}_COLMAP_SfM.log"
    logfile_path = dataset_dir / logfile_name
    convert_COLMAP_to_log(
        filename=str(sparse_dir / "cameras.bin"),
        logfile_out=str(logfile_path),
        input_images=str(dataset_dir / "images"),
        formatp="png"  # Changed from "jpg" to "png" to support alpha channel
    )
    print(f"Generated COLMAP log file: {logfile_path}")

    # Create training script in the dataset directory
    _create_training_script(dataset_dir, args.use_decoupled_appearance)

    print("Synthetic dataset created at", dataset_dir)
    if args.use_decoupled_appearance:
        print("Note: Dataset generated with appearance variation (different lighting per group)")
        print("      Training script includes --use_decoupled_appearance flag")
    print("Next step: run ./train_and_extract_mesh.sh in", dataset_dir)
    _validate_dataset(dataset_dir)



if __name__ == "__main__":
    main()
