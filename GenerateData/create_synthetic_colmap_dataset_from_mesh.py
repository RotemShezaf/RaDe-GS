"""Generate a synthetic COLMAP-style dataset from the polynomial meshes.

The script loads the meshes produced by ``GenerateRawPolynomialMesh.py`` and
renders them from multiple virtual cameras placed on an orbit.  For every
surface/level pair it writes:

- ``images/``: JPG renders with simple shading
- ``sparse/0/cameras.txt`` + ``cameras.bin``
- ``sparse/0/images.txt`` + ``images.bin``
- ``sparse/0/points3D.{txt,bin,ply}``

The resulting folder can be consumed directly by ``train.py`` via the standard
``-s <dataset_root>`` argument.
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

from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial_normal

from scene.colmap_loader import (
    rotmat2qvec,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)

SURFACES = ("Paraboloid", "Saddle", "HyperbolicParaboloid")


@dataclass
class CameraSample:
    image_id: int
    camera_id: int
    image_name: str
    qvec: np.ndarray  # (4,)
    tvec: np.ndarray  # (3,)


@dataclass
class CameraIntrinsics:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render polynomial meshes into a synthetic COLMAP dataset")
    parser.add_argument("--surface", choices=SURFACES, help="Surface type to render", default=SURFACES[0])
    parser.add_argument("--level", type=int, help="Resolution level index for polynomial surface (if used)", default=2)
    parser.add_argument("--mesh_level", type=int, help="Mesh resolution level index to render image (PLY)", default=1)
    parser.add_argument("--data_root", type=str, default="TrainData/raw", help="Base directory containing generated surfaces")
    parser.add_argument("--output_root", type=str, default="SyntheticData", help="Destination root for the rendered dataset")
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
    parser.add_argument("--points3d_count", type=int, default=70000, help="Number of vertices to dump into COLMAP points3D")
    parser.add_argument("--points3d_colmap_count", type=int, default=70000, help="Number of 3D points for COLMAP files (points3D.*)")
    parser.add_argument("--color_scheme", choices=["height", "surface"], default="height", help="Fallback color scheme when the mesh has no vertex colors")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for viewpoint shuffling and point sampling")
    return parser.parse_args()


def _load_mesh(data_root: Path, surface: str, level: int, color_scheme: str) -> o3d.geometry.TriangleMesh:
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
    texture_path = Path(__file__).parent / "textures" / "colors.png"
    if texture_path.exists():
        print(f"Sampling vertex colors from texture: {texture_path}")
        colors = _sample_colors_from_texture(texture_path, uv_coords)
    else:
        print(f"Texture not found, using procedural colors")
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


def _generate_uv_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Generate UV coordinates for vertices based on x,y positions.
    
    Maps x,y coordinates to [0,1] range for texture mapping.
    
    Args:
        vertices: (N, 3) array of vertex positions
    
    Returns:
        uv_coords: (N, 2) array of UV coordinates in [0, 1]
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Normalize x and y to [0, 1] range
    u = (x - x.min()) / (np.ptp(x) + 1e-8)
    v = (y - y.min()) / (np.ptp(y) + 1e-8)
    return np.column_stack([u, v])


def _load_texture_image(texture_path: Path) -> o3d.geometry.Image:
    """Load texture image and convert to Open3D format.
    
    Args:
        texture_path: Path to texture image file
    
    Returns:
        o3d_image: Open3D Image object
    """
    # Load with PIL
    pil_image = Image.open(texture_path).convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.uint8)
    
    # Convert to Open3D Image
    o3d_image = o3d.geometry.Image(img_array)
    
    return o3d_image


def _sample_colors_from_texture(texture_path: Path, uv_coords: np.ndarray) -> np.ndarray:
    """Sample RGB colors from texture image at given UV coordinates.
    
    Args:
        texture_path: Path to texture image file
        uv_coords: (N, 2) array of UV coordinates in [0, 1] range
    
    Returns:
        colors: (N, 3) array of RGB colors in [0, 1] range
    """
    # Load texture image
    pil_image = Image.open(texture_path).convert('RGB')
    img_array = np.array(pil_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    height, width = img_array.shape[:2]
    
    # Convert UV coordinates to pixel coordinates
    # UV (0,0) is typically bottom-left, but image (0,0) is top-left
    # So we flip V coordinate
    u = np.clip(uv_coords[:, 0], 0, 1)
    v = np.clip(1.0 - uv_coords[:, 1], 0, 1)  # Flip V
    
    # Convert to pixel indices
    x_pixels = (u * (width - 1)).astype(np.int32)
    y_pixels = (v * (height - 1)).astype(np.int32)
    
    # Sample colors from texture
    colors = img_array[y_pixels, x_pixels, :]
    
    return colors



def _rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _random_rotation_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ]
    )
    # Reorder to w, x, y, z for conversion
    quat = np.array([q[3], q[0], q[1], q[2]])
    R = _rotation_matrix_from_quaternion(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0)
    return _rotation_matrix_from_quaternion(quat)


def _fibonacci_sphere(num_views: int) -> np.ndarray:
    """
    Generate points uniformly distributed on the surface of a unit sphere
    using the Fibonacci sphere algorithm.

    Args:
        num_views (int): Number of points to generate on the sphere.

    Returns:
        np.ndarray: Array of shape (num_views, 3) with (x, y, z) coordinates.
    """
    # Create indices 0.5, 1.5, 2.5, ..., num_views - 0.5
    # The 0.5 offset centers points in each interval to avoid clustering at poles
    i = np.arange(num_views, dtype=np.float64) + 0.5

    # Compute polar angle (from z-axis) for each point
    # 1 - 2*i/num_views linearly spaces values from 1 to -1
    # arccos converts to angle phi
    phi = np.arccos(1 - 2 * i / num_views)

    # Compute azimuthal angle using the golden angle to evenly distribute points
    # The golden angle ~137.5° ensures minimal overlap/spiral pattern
    theta = math.pi * (1 + math.sqrt(5)) * i

    # Convert spherical coordinates (phi, theta) to Cartesian coordinates (x, y, z)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Stack coordinates into a (num_views, 3) array
    return np.stack([x, y, z], axis=1)


def _compute_camera_centers(
    mesh: o3d.geometry.TriangleMesh,
    num_views: int,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound)
    bbox_max = np.asarray(bbox.max_bound)
    center = (bbox_min + bbox_max) * 0.5
    extent = np.linalg.norm(bbox_max - bbox_min)
    radius = args.camera_radius if args.camera_radius is not None else extent * args.orbit_radius_scale

    if args.camera_distribution == "uniform_sphere":
        directions = _fibonacci_sphere(num_views)
        rot = _random_rotation_matrix(seed)
        directions = directions @ rot.T
        centers = center[None, :] + radius * directions
    else:
        theta = np.linspace(0, 2 * math.pi, num_views, endpoint=False)
        rng = np.random.default_rng(seed)
        rng.shuffle(theta)
        elevation = math.radians(args.elevation_deg)
        centers = []
        for angle in theta:
            x = center[0] + radius * math.cos(angle) * math.cos(elevation)
            y = center[1] + radius * math.sin(angle) * math.cos(elevation)
            z = center[2] + radius * math.sin(elevation)
            centers.append([x, y, z])
        centers = np.asarray(centers)

    targets = np.repeat(center[None, :], num_views, axis=0)
    return centers, targets






def _render_images(mesh: o3d.geometry.TriangleMesh, centers: np.ndarray, targets: np.ndarray, args: argparse.Namespace, output_dir: Path) -> List[CameraSample]:
    """Render images using Open3D renderer.

    The camera sampling (centers/targets) is preserved from the original
    implementation; rendering is done directly with Open3D mesh.
    """

    width = args.image_width
    height = args.image_height
    yfov = math.radians(args.vertical_fov)
    fy = (height / 2.0) / math.tan(yfov / 2.0)
    fx = fy * (width / height)
    cx = width / 2.0
    cy = height / 2.0

    samples: List[CameraSample] = []
    camera_intrinsics: List[CameraIntrinsics] = []

    # Load texture if it exists
    texture_path = Path(__file__).parent / "textures" / "colors.png"
    has_texture = texture_path.exists()
    
    # Create renderer once outside the loop
    renderer = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

   
    
    # Apply texture if available
    if has_texture:
        print(f"Loading texture from {texture_path}")
        texture_image = _load_texture_image(texture_path)
        material.albedo_img = texture_image
        #print(f"Texture loaded: {texture_image.width}x{texture_image.height}")
    
    # Add geometry once
    renderer.scene.add_geometry("mesh", mesh, material)
    
    # Set camera intrinsics once
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    renderer.setup_camera(intrinsic, np.eye(4))
    
    # Debug mesh before rendering
    print(f"Mesh has {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Mesh has colors: {mesh.has_vertex_colors()}")
    if mesh.has_vertex_colors():
        colors_arr = np.asarray(mesh.vertex_colors)
        print(f"Color range: [{colors_arr.min():.4f}, {colors_arr.max():.4f}]")
    
    # Set lighting once - reduced from 75000 to 5000
    renderer.scene.scene.set_sun_light([1,-1, 1], [1,1,1], 10000)
    renderer.scene.scene.enable_sun_light(True)

    for idx, (eye, target) in enumerate(zip(centers, targets), start=1):

        
        # Set the camera pose in the renderer using the c2w matrix
        renderer.scene.camera.look_at(target, eye, [0,1,0])

        # Get c2w camera ematrix from the renderer
        c2w = np.linalg.inv(renderer.scene.camera.get_view_matrix())

       
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
    
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R_w2c = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        tvec = w2c[:3, 3]


        
        # Transform mesh vertices to camera coordinates and check minimum z value
        vertices_world = np.asarray(mesh.vertices)
        vertices_camera = (R_w2c @ vertices_world.T).T + tvec
        min_z = vertices_camera[:, 2].min()
        max_z = vertices_camera[:, 2].max()
        print(f"Camera {idx}: Z range = [{min_z:.4f}, {max_z:.4f}]")
        assert min_z >= 0.2, f"Camera {idx}: Minimum z value ({min_z:.4f}) is less than 0.2 - mesh too close to camera!"
        
        # Render image
        img = renderer.render_to_image()
        
        # Debug first image
        if idx == 1:
            img_np = np.asarray(img)
            print(f"First image stats: shape={img_np.shape}, range=[{img_np.min():.4f}, {img_np.max():.4f}], mean={img_np.mean():.4f}")
            print(f"Unique values (rounded): {len(np.unique(img_np.round(decimals=2)))}")
            if img_np.mean() > 0.9 or img_np.mean() < 0.1:
                print("WARNING: Image appears blank/uniform - mesh may not be visible!")
        
        image_name = f"view_{idx:03d}.jpg"
        o3d.io.write_image(str(output_dir / "images" / image_name), img, quality=97)

        camera_intrinsics.append(
            CameraIntrinsics(
                camera_id=idx,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

        qvec = rotmat2qvec(R_w2c)
        samples.append(CameraSample(image_id=idx, camera_id=idx, image_name=image_name, qvec=qvec, tvec=tvec))

    return samples, camera_intrinsics


def _write_cameras_txt(path: Path, cameras: Sequence[CameraIntrinsics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam in cameras:
            f.write(
                f"{cam.camera_id} PINHOLE {cam.width} {cam.height} "
                f"{cam.fx:.6f} {cam.fy:.6f} {cam.cx:.6f} {cam.cy:.6f}\n"
            )


def _write_cameras_bin(path: Path, cameras: Sequence[CameraIntrinsics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_id = 1  # PINHOLE
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<iiQQ", cam.camera_id, model_id, cam.width, cam.height))
            params = [float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)]
            f.write(struct.pack("<" + "d" * len(params), *params))


def _write_images_txt(path: Path, samples: Sequence[CameraSample]) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for s in samples:
            f.write(
                f"{s.image_id} {s.qvec[0]:.8f} {s.qvec[1]:.8f} {s.qvec[2]:.8f} {s.qvec[3]:.8f} "
                f"{s.tvec[0]:.8f} {s.tvec[1]:.8f} {s.tvec[2]:.8f} {s.camera_id} {s.image_name}\n"
            )
            f.write("\n")


def _write_images_bin(path: Path, samples: Sequence[CameraSample]) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(samples)))
        for s in samples:
            f.write(
                struct.pack(
                    "<idddddddi",
                    s.image_id,
                    float(s.qvec[0]),
                    float(s.qvec[1]),
                    float(s.qvec[2]),
                    float(s.qvec[3]),
                    float(s.tvec[0]),
                    float(s.tvec[1]),
                    float(s.tvec[2]),
                    s.camera_id,
                )
            )
            f.write(s.image_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num points2D


def _sample_points(mesh: o3d.geometry.TriangleMesh, count: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if colors.max() <= 1.0:
        colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    rng = np.random.default_rng(seed)
    if len(vertices) > count:
        ids = rng.choice(len(vertices), size=count, replace=False)
        vertices = vertices[ids]
        colors = colors[ids]
    return vertices, colors


def _write_points3d_txt(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for idx, (v, c) in enumerate(zip(vertices, colors), start=1):
            f.write(f"{idx} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")


def _write_points3d_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        "comment synthetic polynomial dataset",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _ensure_dirs(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)


def _validate_dataset(dataset_dir: Path) -> None:
    sparse_dir = dataset_dir / "sparse" / "0"
    try:
        cams = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        xyzs, rgbs, _ = read_points3D_binary(str(sparse_dir / "points3D.bin"))

        print(
            "Validation via colmap_loader:",
            f"{len(cams)} camera(s), {len(images)} image(s), {len(xyzs)} points"
        )
    except Exception as exc:
        print("Validation warning (colmap_loader parsing failed):", exc)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    # Use mesh_level for loading the triangulated mesh (default: level 0)
    mesh_level = getattr(args, "mesh_level", args.level)
    mesh = _load_mesh(data_root, args.surface, mesh_level, args.color_scheme)

    # Load analytical normals if available, otherwise calculate from mesh
    _load_normals_if_available(data_root, args.surface, mesh_level, mesh)

    dataset_dir = Path(args.output_root) / args.surface / f"level_{args.level:02d}"
    _ensure_dirs(dataset_dir)

    centers, targets = _compute_camera_centers(mesh, args.num_views, args, args.seed)
    samples, camera_intrinsics = _render_images(mesh, centers, targets, args, dataset_dir)

    # Separate resolutions for COLMAP 
    colmap_count = args.points3d_colmap_count or args.points3d_count

    # Points for COLMAP points3D.*
    points_xyz_colmap, points_rgb_colmap = _sample_points(mesh, colmap_count, args.seed)

    sparse_dir = dataset_dir / "sparse" / "0"
    _write_cameras_txt(sparse_dir / "cameras.txt", camera_intrinsics)
    _write_cameras_bin(sparse_dir / "cameras.bin", camera_intrinsics)
    _write_images_txt(sparse_dir / "images.txt", samples)
    _write_images_bin(sparse_dir / "images.bin", samples)
    _write_points3d_txt(sparse_dir / "points3D.txt", points_xyz_colmap, points_rgb_colmap)
    _write_points3d_bin(sparse_dir / "points3D.bin", points_xyz_colmap, points_rgb_colmap)
    _write_points3d_ply(sparse_dir / "points3D.ply", points_xyz_colmap, points_rgb_colmap)

    print("Synthetic dataset created at", dataset_dir)
    print("Next step: run train.py with -s", dataset_dir)
    _validate_dataset(dataset_dir)


def _write_points3d_bin(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(vertices)))
        for idx, (v, c) in enumerate(zip(vertices, colors), start=1):
            f.write(struct.pack("<QdddBBBd", idx, v[0], v[1], v[2], int(c[0]), int(c[1]), int(c[2]), 1.0))
            f.write(struct.pack("<Q", 0))  # track length

if __name__ == "__main__":
    main()
