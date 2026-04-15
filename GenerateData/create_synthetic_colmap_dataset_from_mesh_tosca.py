"""Generate a synthetic COLMAP-style dataset from TOSCA meshes.

This script creates a synthetic dataset that mimics the structure of a real COLMAP reconstruction.
It loads TOSCA meshes (preprocessed by preprocess_tosca.py) and renders them from multiple 
virtual cameras placed on an orbit.

TOSCA MESH LOADING:
-------------------
The script expects TOSCA meshes preprocessed by preprocess_tosca.py with the following structure:
TrainData/TOSCA/processed/
    ├── cat0/
    │   ├── mesh_high_res_arc*.ply
    │   ├── mesh_low_res_arc*.ply
    │   ├── normals_high_res_arc*.ply
    │   ├── normals_low_res_arc*.ply
    │   └── ...
    └── ...

DUAL-LEVEL MESH SYSTEM:
-----------------------
The script uses two separate mesh resolution levels for different purposes:

1. IMAGE_MESH_RESOLUTION (--image_mesh_resolution): 
   - Used for rendering the training images from virtual cameras
   - Can be lower resolution ('low_res') for faster rendering without quality loss
   - Affects rendering speed but not the final COLMAP point cloud
   - Default: 'low_res'

2. COLMAP_RESOLUTION (--colmap_resolution):
   - Used for generating the COLMAP sparse point cloud (points3D.*)
   - Should be higher resolution ('high_res') to provide denser geometric information
   - Affects the quality and density of the point cloud used for reconstruction
   - Default: 'high_res'

This separation allows you to optimize rendering speed independently from point cloud quality.
For example, you can render images from a simpler mesh while providing a dense point cloud
for accurate geometric reconstruction.

OUTPUT STRUCTURE:
-----------------
For every TOSCA shape, the script writes:
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
    align_mesh_principal_axes,
    apply_rotation_to_mesh,
)

from scene.colmap_loader import (
    rotmat2qvec,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)

# TOSCA shapes available after preprocessing
TOSCA_SHAPES = []  # Will be populated from data directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TOSCA meshes into a synthetic COLMAP dataset")
    parser.add_argument("--shape", type=str, help="TOSCA shape name to render (e.g., 'cat0', 'centaur1'). If not specified, will process all shapes in data_root.", default=None)
    
    # Mesh resolution levels: This script uses a two-level system for flexibility
    parser.add_argument(
        "--colmap_resolution", 
        type=str,
        choices=['high_res', 'low_res'],
        default='high_res',
        help="Resolution level for COLMAP point cloud sampling. 'high_res' = denser mesh used for points3D.* files. "
             "This mesh is sampled to generate the sparse point cloud that COLMAP would typically produce."
    )
    parser.add_argument(
        "--image_mesh_resolution", 
        type=str,
        choices=['high_res', 'low_res'],
        default='low_res',
        help="Resolution level for rendering images. Can be lower resolution than colmap_resolution for faster rendering. "
             "The mesh at this level is used to generate the synthetic training images from virtual cameras."
    )
    
    parser.add_argument("--data_root", type=str, default="TrainData/TOSCA/processed", help="Base directory containing preprocessed TOSCA shapes")
    parser.add_argument("--output_root", type=str, default="TrainData/TOSCA/SyntheticColmapData", help="Destination root for the rendered dataset")
    parser.add_argument("--num_views", type=int, default=50, help="Number of rendered viewpoints along the orbit")
    parser.add_argument("--image_width", type=int, default=960, help="Rendered image width in pixels")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height in pixels")
    parser.add_argument("--vertical_fov", type=float, default=45.0, help="Camera vertical field of view in degrees")
    parser.add_argument("--orbit_radius_scale", type=float, default=1.4, help="Orbit radius as a multiple of the mesh bounding radius")
    parser.add_argument("--elevation_deg", type=float, default=25.0, help="Camera elevation angle in degrees")
    parser.add_argument("--camera_radius", type=float, default=6, help="Absolute radius for camera placement (overrides orbit_radius_scale)")
    parser.add_argument("--auto_camera_radius", action="store_true", help="Automatically calculate the smallest camera radius that makes the entire mesh visible in all images. Overrides --camera_radius and --orbit_radius_scale.")
    parser.add_argument(
        "--camera_distribution",
        choices=["uniform_sphere", "orbit", "n_circles"],
        default="uniform_sphere",
        help="Strategy for sampling camera centers",
    )
    parser.add_argument(
        "--circle_elevations",
        type=str,
        default="25,55,85",
        help="When --camera_distribution=n_circles, provide comma-separated elevation angles in degrees, e.g. '25,55,85'",
    )
    parser.add_argument("--light_intensity", type=float, default=3.5, help="Directional light intensity for pyrender")
    parser.add_argument("--points3d_thresh", type=float, default=None, help="Downsample density threshold for COLMAP points3D (minimum distance between points). If None, no downsampling is performed.")
    parser.add_argument("--color_scheme", choices=["height", "surface", "vertex"], default="vertex", help="Color scheme: 'height' = height-based gradient, 'surface' = shape-based color, 'vertex' = use vertex colors from mesh")
    parser.add_argument("--texture_folder", type=str, default="textures", help="Folder name containing texture images (relative to GenerateData directory)")
    parser.add_argument("--texture_name", type=str, default="colors", help="Texture image filename (extension optional, will search for .png, .jpg, .jpeg, .bmp, .tif)")
    parser.add_argument("--use_decoupled_appearance", action="store_true", help="Vary lighting across view groups to simulate appearance variations for training appearance networks")
    parser.add_argument("--light_id", type=int, default=None, help="Fixed lighting configuration ID (0-4) for consistent lighting across all views. 0=default balanced, 1-4=variations. Ignored if --use_decoupled_appearance is set.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for viewpoint shuffling and point sampling")
    parser.add_argument("--align_principal_axes", action="store_true", help="Rotate mesh so principal axes align with world axes before rendering/sampling")
    parser.add_argument("--major_to", choices=["x", "y", "z"], default="z", help="When --align_principal_axes is set, map the largest principal axis to this world axis (default: z)")
    parser.add_argument("--all", action='store_true', help="Process all shapes in data_root directory")
    return parser.parse_args()


def _get_available_shapes(data_root: Path) -> List[str]:
    """Get list of all available TOSCA shape names in data_root.
    
    Args:
        data_root: Path to processed TOSCA directory
        
    Returns:
        List of shape names (folder names)
    """
    if not data_root.exists():
        return []
    
    shapes = []
    for item in data_root.iterdir():
        if item.is_dir():
            # Check if it contains TOSCA mesh files
            high_res_meshes = list(item.glob("mesh_high_res_*.ply"))
            low_res_meshes = list(item.glob("mesh_low_res_*.ply"))
            if high_res_meshes or low_res_meshes:
                shapes.append(item.name)
    
    return sorted(shapes)


def _load_tosca_mesh(data_root: Path, shape: str, resolution: str, color_scheme: str, texture_folder: str, texture_name: str) -> o3d.geometry.TriangleMesh:
    """Load a TOSCA mesh at specified resolution.
    
    Args:
        data_root: Root directory containing TOSCA shapes
        shape: Shape name (e.g., 'cat0', 'centaur1')
        resolution: 'high_res' or 'low_res'
        color_scheme: 'height', 'surface', or 'vertex' for color assignment
        texture_folder: Folder containing texture images
        texture_name: Texture image filename
        
    Returns:
        Open3D TriangleMesh with colors
    """
    shape_dir = data_root / shape
    mesh_candidates = sorted(shape_dir.glob(f"mesh_{resolution}_*.ply"))
    
    if not mesh_candidates:
        raise FileNotFoundError(f"No mesh for shape={shape} resolution={resolution} under {shape_dir}")
    
    # Load with trimesh first to handle colors
    trimesh_mesh = trimesh.load(mesh_candidates[0], process=False)
    if not isinstance(trimesh_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a trimesh.Trimesh, got {type(trimesh_mesh)} from {mesh_candidates[0]}")
    
    # Set up colors based on scheme
    if color_scheme == "vertex":
        # Use vertex colors from mesh if available
        if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, "vertex_colors") and len(trimesh_mesh.visual.vertex_colors) > 0:
            colors = np.asarray(trimesh_mesh.visual.vertex_colors[:, :3], dtype=np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            # Fallback to height-based colors
            print(f"No vertex colors available for {shape}, using height-based coloring")
            colors = _build_tosca_colors(np.asarray(trimesh_mesh.vertices), shape, "height")
    else:
        colors = _build_tosca_colors(np.asarray(trimesh_mesh.vertices), shape, color_scheme)
    
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
        print(f"Texture not found in {texture_folder_path}, using {color_scheme} coloring")
    
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    
    # For triangle_uvs, we need 3 UV coordinates per triangle
    # Flatten faces and index into uv_coords to get per-triangle-vertex UVs
    triangle_uvs = uv_coords[faces.flatten()]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    
    return mesh


def _load_tosca_normals_if_available(data_root: Path, shape: str, resolution: str, mesh: o3d.geometry.TriangleMesh) -> None:
    """Load normals saved by preprocess_tosca.py if present, or calculate them.

    Looks for normals_{resolution}_*.ply next to the meshes. The normals file has the same
    structure as the mesh file but with normals stored as vertex data.
    If not found, calculates normals from the mesh.
    Modifies the mesh in place by setting vertex_normals.
    
    Args:
        data_root: Root directory containing TOSCA shapes
        shape: Shape name (e.g., 'cat0')
        resolution: 'high_res' or 'low_res'
        mesh: Open3D mesh to attach normals to
    """
    shape_dir = data_root / shape
    candidates = sorted(shape_dir.glob(f"normals_{resolution}_*.ply"))
    
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
                        print(f"Loaded precomputed normals for {shape} {resolution}")
                        return
        except Exception as e:
            print(f"Warning: Failed to load normals from {candidates[0]}: {e}")
    
    # Fall back to calculating normals from the mesh
    print(f"Calculating normals for {shape} {resolution}")
    mesh.compute_vertex_normals()


def _build_tosca_colors(vertices: np.ndarray, shape: str, scheme: str) -> np.ndarray:
    """Build vertex colors for TOSCA mesh based on color scheme.
    
    Args:
        vertices: Nx3 array of vertex positions
        shape: Shape name
        scheme: 'height', 'surface', or 'vertex'
        
    Returns:
        Nx3 array of RGB colors in [0, 1]
    """
    colors = np.empty((len(vertices), 3), dtype=np.float64)
    
    if scheme == "surface":
        # Assign colors based on shape category
        shape_lower = shape.lower()
        if 'cat' in shape_lower:
            base = np.array([0.95, 0.7, 0.3])  # Orange for cats
        elif 'dog' in shape_lower:
            base = np.array([0.6, 0.4, 0.2])  # Brown for dogs
        elif 'horse' in shape_lower:
            base = np.array([0.5, 0.3, 0.15])  # Dark brown for horses
        elif 'centaur' in shape_lower:
            base = np.array([0.8, 0.6, 0.4])  # Tan for centaurs
        elif 'gorilla' in shape_lower:
            base = np.array([0.2, 0.2, 0.2])  # Dark gray for gorillas
        elif 'michael' in shape_lower or 'victoria' in shape_lower or 'david' in shape_lower:
            base = np.array([0.9, 0.75, 0.65])  # Skin tone for humans
        else:
            base = np.array([0.7, 0.7, 0.7])  # Default gray
        
        colors[:, :] = base
        
    elif scheme == "height":
        # Height-based gradient
        z = vertices[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-6)
        # Create a blue-to-red gradient
        colors[:, 0] = z_norm  # Red channel
        colors[:, 1] = 0.4 + 0.4 * (1 - np.abs(z_norm - 0.5))  # Green channel
        colors[:, 2] = 1.0 - z_norm  # Blue channel
    else:
        # Default gray
        colors[:, :] = 0.7
    
    return colors


def _process_single_shape(args: argparse.Namespace, shape: str, data_root: Path) -> None:
    """Process a single TOSCA shape and generate synthetic COLMAP dataset.
    
    Args:
        args: Command-line arguments
        shape: Shape name to process
        data_root: Root directory containing TOSCA shapes
    """
    print("\n" + "=" * 80)
    print(f"Processing TOSCA shape: {shape}")
    print("=" * 80)
    
    # Load mesh at image_mesh_resolution for rendering synthetic training images
    # This can be a lower-resolution mesh for faster rendering without sacrificing quality
    actual_image_mesh_resolution = args.image_mesh_resolution
    try:
        image_mesh = _load_tosca_mesh(data_root, shape, actual_image_mesh_resolution, args.color_scheme, args.texture_folder, args.texture_name)
    except FileNotFoundError:
        if actual_image_mesh_resolution == 'high_res' and args.colmap_resolution == 'low_res':
            print(f"Warning: No high_res mesh for {shape}, falling back to low_res for image_mesh rendering")
            actual_image_mesh_resolution = 'low_res'
            image_mesh = _load_tosca_mesh(data_root, shape, actual_image_mesh_resolution, args.color_scheme, args.texture_folder, args.texture_name)
        else:
            raise

    # Load analytical normals if available, otherwise calculate from mesh
    _load_tosca_normals_if_available(data_root, shape, actual_image_mesh_resolution, image_mesh)

    # Optionally align principal axes of the image mesh and record rotation
    rot_R = None
    rot_centroid = None
    if getattr(args, 'align_principal_axes', False):
        major_to = getattr(args, 'major_to', 'z')
        print(f"Aligning principal axes for shape {shape} (major -> {major_to})...")
        R, centroid = align_mesh_principal_axes(image_mesh, return_matrix=True, major_to=major_to)
        rot_R = R
        rot_centroid = centroid

    # Output directory includes texture, resolution, and lighting mode subdirectory
    base_dir = Path(args.output_root) / f"{args.texture_name}_texture" / shape / f"{args.colmap_resolution}"
    
    # Determine subdirectory based on lighting mode
    if args.use_decoupled_appearance:
        dataset_dir = base_dir / "decoupled_appearance"
    elif args.light_id is not None:
        dataset_dir = base_dir / f"light_{args.light_id}"
    else:
        dataset_dir = base_dir / "default_light"
    
    _ensure_dirs(dataset_dir)

    # Compute camera positions and render images using the image_mesh_resolution mesh
    centers, targets = _compute_camera_centers(image_mesh, args.num_views, args, args.seed)
    samples, camera_intrinsics = _render_images(image_mesh, centers, targets, args, dataset_dir, args.texture_folder, args.texture_name)

    # Load mesh at colmap_resolution for generating COLMAP point cloud (points3D.*)
    # This is typically a higher-resolution mesh to provide a denser point cloud
    # that better represents the surface geometry for reconstruction
    colmap_mesh = _load_tosca_mesh(data_root, shape, args.colmap_resolution, args.color_scheme, args.texture_folder, args.texture_name)
    # Load analytical normals if available, otherwise calculate from mesh
    _load_tosca_normals_if_available(data_root, shape, args.colmap_resolution, colmap_mesh)
    # If image mesh was aligned, apply same rotation to colmap mesh for consistency
    if rot_R is not None and rot_centroid is not None:
        apply_rotation_to_mesh(colmap_mesh, rot_R, rot_centroid)
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
    logfile_name = f"{shape}_COLMAP_SfM.log"
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

    print(f"Synthetic dataset created at {dataset_dir}")
    if args.use_decoupled_appearance:
        print(f"Note: Dataset generated with appearance variation (different lighting per group)")
        print(f"      Training script includes --use_decoupled_appearance flag")
    print(f"Next step: run ./train_and_extract_mesh.sh in {dataset_dir}")
    _validate_dataset(dataset_dir)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        print("\nPlease run preprocess_tosca.py first to prepare the TOSCA meshes.")
        sys.exit(1)
    
    # Get available shapes
    available_shapes = _get_available_shapes(data_root)
    
    if not available_shapes:
        print(f"Error: No TOSCA shapes found in {data_root}")
        print("\nPlease run preprocess_tosca.py first to prepare the TOSCA meshes.")
        sys.exit(1)
    
    print(f"Found {len(available_shapes)} TOSCA shapes in {data_root}")
    
    # Determine which shapes to process
    if args.all:
        shapes_to_process = available_shapes
        print(f"Processing all {len(shapes_to_process)} shapes")
    elif args.shape:
        if args.shape not in available_shapes:
            print(f"Error: Shape '{args.shape}' not found in {data_root}")
            print(f"\nAvailable shapes: {', '.join(available_shapes)}")
            sys.exit(1)
        shapes_to_process = [args.shape]
        print(f"Processing single shape: {args.shape}")
    else:
        print("Error: Please specify either --shape <shape_name> or --all")
        print(f"\nAvailable shapes: {', '.join(available_shapes)}")
        sys.exit(1)
    
    # Process each shape
    for i, shape in enumerate(shapes_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(shapes_to_process)}")
        print(f"{'='*80}")
        try:
            _process_single_shape(args, shape, data_root)
        except Exception as e:
            print(f"Error processing shape {shape}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("All shapes processed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
