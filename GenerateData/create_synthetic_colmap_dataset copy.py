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
import numpy as np
import open3d as o3d
import struct
import trimesh

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
    rvec: np.ndarray  # (4,)
    tvec: np.ndarray  # (3,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render polynomial meshes into a synthetic COLMAP dataset")
    parser.add_argument("--surface", choices=SURFACES, help="Surface type to render")
    parser.add_argument("--level", type=int, help="Resolution level index to load (matches GenerateRawPolynomialMesh)")
    parser.add_argument("--data_root", type=str, default="TrainData/raw", help="Base directory containing generated surfaces")
    parser.add_argument("--output_root", type=str, default="SyntheticData", help="Destination root for the rendered dataset")
    parser.add_argument("--num_views", type=int, default=24, help="Number of rendered viewpoints along the orbit")
    parser.add_argument("--image_width", type=int, default=960, help="Rendered image width in pixels")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height in pixels")
    parser.add_argument("--vertical_fov", type=float, default=45.0, help="Camera vertical field of view in degrees")
    parser.add_argument("--orbit_radius_scale", type=float, default=1.4, help="Orbit radius as a multiple of the mesh bounding radius")
    parser.add_argument("--elevation_deg", type=float, default=25.0, help="Camera elevation angle in degrees")
    parser.add_argument("--light_intensity", type=float, default=3.5, help="Directional light intensity for pyrender")
    parser.add_argument("--points3d_count", type=int, default=20000, help="Number of vertices to dump into COLMAP points3D")
    parser.add_argument("--points3d_colmap_count", type=int, default=20000, help="Number of 3D points for COLMAP files (points3D.*)")
    parser.add_argument("--points3d_image_count", type=int, default=20000, help="Number of 3D points used to define image-space resolution / normals")
    parser.add_argument("--color_scheme", choices=["height", "surface"], default="height", help="Fallback color scheme when the mesh has no vertex colors")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for viewpoint shuffling and point sampling")
    return parser.parse_args()


def _load_mesh(data_root: Path, surface: str, level: int, color_scheme: str) -> trimesh.Trimesh:
    mesh_candidates = sorted((data_root / surface).glob(f"mesh_level{level}_*.ply"))
    if not mesh_candidates:
        raise FileNotFoundError(f"No mesh for surface={surface} level={level} under {data_root/surface}")
    mesh = trimesh.load(mesh_candidates[0], process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a trimesh.Trimesh, got {type(mesh)} from {mesh_candidates[0]}")
    if mesh.visual is None or not hasattr(mesh.visual, "vertex_colors") or len(mesh.visual.vertex_colors) == 0:
        mesh.visual.vertex_colors = _build_colors(np.asarray(mesh.vertices), surface, color_scheme)
    return mesh


def _load_normals_if_available(data_root: Path, surface: str, level: int) -> np.ndarray | None:
    """Load analytical normals saved by GenerateRawPolynomialMesh if present.

    Looks for normals_level{level}_*.ply next to the meshes.
    """
    surface_dir = data_root / surface
    candidates = sorted(surface_dir.glob(f"normals_level{level}_*.ply"))
    if not candidates:
        return None
    try:
        normals_pc = o3d.io.read_point_cloud(str(candidates[0]))
        normals = np.asarray(normals_pc.normals, dtype=np.float32)
        if normals.size == 0:
            return None
        return normals
    except Exception:
        return None


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


def _compute_orbit_centers(mesh: trimesh.Trimesh, num_views: int, radius_scale: float, elevation_deg: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    bbox_min, bbox_max = mesh.bounds
    center = (bbox_min + bbox_max) * 0.5
    extent = np.linalg.norm(bbox_max - bbox_min)
    radius = extent * radius_scale
    theta = np.linspace(0, 2 * math.pi, num_views, endpoint=False)
    rng = np.random.default_rng(seed)
    rng.shuffle(theta)
    elevation = math.radians(elevation_deg)
    centers = []
    targets = np.repeat(center[None, :], num_views, axis=0)
    for angle in theta:
        x = center[0] + radius * math.cos(angle) * math.cos(elevation)
        y = center[1] + radius * math.sin(angle) * math.cos(elevation)
        z = center[2] + radius * math.sin(elevation)
        centers.append([x, y, z])
    return np.asarray(centers), targets


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, forward)
    rot = np.stack([right, true_up, -forward], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = eye
    return pose


def _render_images(mesh: trimesh.Trimesh, centers: np.ndarray, targets: np.ndarray, args: argparse.Namespace, output_dir: Path) -> List[CameraSample]:
    # Convert trimesh to Open3D triangle mesh
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    # Use vertex colors if present
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Use normals from mesh if available, otherwise let Open3D estimate
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals).copy())
    else:
        o3d_mesh.compute_vertex_normals()

    extent = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.image_width, args.image_height)
    renderer.scene.set_background([0.18, 0.18, 0.18, 1.0])
    #renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    # Unlit shader makes vertex colors show up regardless of normals/lighting setup.
    mat.shader = "defaultUnlit"
    mat.base_color = [0.6, 0.6, 0.6, 1.0]  # NOT white

    renderer.scene.add_geometry("mesh", o3d_mesh, mat)

    # Controlled sun light
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light(
        direction=[0.3, 0.3, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=0.8,   # THIS is the key
    )

    yfov = math.radians(args.vertical_fov)
    aspect = args.image_width / args.image_height

    samples: List[CameraSample] = []
    for idx, (eye, target) in enumerate(zip(centers, targets), start=1):
        pose = _look_at(eye, target)

        # Open3D expects camera extrinsics as 4x4 world-to-camera
        R_c2w = pose[:3, :3]
        R_w2c = R_c2w.T
        tvec = -R_w2c @ pose[:3, 3]
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = R_w2c
        extr[:3, 3] = tvec

        # Get axis-aligned bounding box
        aabb = mesh.get_axis_aligned_bounding_box()

        # Get 8 corners
        corners = np.asarray(aabb.get_box_points())  # shape (8,3)
        # Transform corners to camera coordinates
        box_camera = R_w2c @ corners.T + tvec[:, None]  # shape (3,8)
        z_camera = box_camera[2, :]  # z values in camera coordinates
        assert np.all(z_camera > 0.2), f"Some bounding box corners have z_camera <= 0.2: {z_camera}"

        # Build intrinsics matrix
        fy = (args.image_height / 2.0) / math.tan(yfov / 2.0)
        fx = fy * aspect
        cx = args.image_width / 2.0
        cy = args.image_height / 2.0
        intr = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        renderer.setup_camera(intr, extr, args.image_width, args.image_height)
        img = renderer.render_to_image()
        color = np.asarray(img)
        if color.ndim == 3 and color.shape[2] == 4:
            color = color[:, :, :3]
        if color.dtype != np.uint8:
            color = np.clip(color * 255.0, 0, 255).astype(np.uint8)

        image_name = f"view_{idx:03d}.jpg"
        imageio.imwrite(output_dir / "images" / image_name, color, quality=95)

        qvec = rotmat2qvec(R_w2c)
        samples.append(CameraSample(image_id=idx, camera_id=1, image_name=image_name, qvec=qvec, tvec=tvec))

    # OffscreenRenderer will be cleaned up when it goes out of scope
    return samples


def _write_cameras_txt(path: Path, cam_id: int, width: int, height: int, yfov_rad: float) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    fy = (height / 2.0) / math.tan(yfov_rad / 2.0)
    fx = (width / 2.0) / math.tan(yfov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{cam_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def _write_cameras_bin(path: Path, cam_id: int, width: int, height: int, yfov_rad: float) -> None:
    # Make sure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    fy = (height / 2.0) / math.tan(yfov_rad / 2.0)
    fx = (width / 2.0) / math.tan(yfov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    params = [fx, fy, cx, cy]
    model_id = 1  # PINHOLE
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
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
                    float(s.rvec[0]),
                    float(s.rvec[1]),
                    float(s.rvec[2]),
                    float(s.rvec[3]),
                    float(s.tvec[0]),
                    float(s.tvec[1]),
                    float(s.tvec[2]),
                    s.camera_id,
                )
            )
            f.write(s.image_name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num points2D


def _sample_points(mesh: trimesh.Trimesh, count: int, seed: int, normals: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors[:, :3])
    if colors.max() <= 1.0:
        colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    rng = np.random.default_rng(seed)
    if len(vertices) > count:
        ids = rng.choice(len(vertices), size=count, replace=False)
        vertices = vertices[ids]
        colors = colors[ids]
        if normals is not None and len(normals) == len(mesh.vertices):
            normals = normals[ids]
    return vertices, colors, normals


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
    mesh = _load_mesh(data_root, args.surface, args.level, args.color_scheme)

    # Try to load analytical normals; if found, attach to mesh
    normals = _load_normals_if_available(data_root, args.surface, args.level)
    if normals is not None and len(normals) == len(mesh.vertices):
        mesh.vertex_normals = normals

    dataset_dir = Path(args.output_root) / args.surface / f"level_{args.level:02d}"
    _ensure_dirs(dataset_dir)

    centers, targets = _compute_orbit_centers(
        mesh,
        args.num_views,
        args.orbit_radius_scale,
        args.elevation_deg,
        args.seed,
    )
    samples = _render_images(mesh, centers, targets, args, dataset_dir)

    # Separate resolutions for COLMAP vs image-space point usage
    colmap_count = args.points3d_colmap_count or args.points3d_count
    image_count = args.points3d_image_count or args.points3d_count

    # Points for COLMAP points3D.*
    points_xyz_colmap, points_rgb_colmap, _ = _sample_points(mesh, colmap_count, args.seed, normals)

    sparse_dir = dataset_dir / "sparse" / "0"
    yfov = math.radians(args.vertical_fov)
    _write_cameras_txt(sparse_dir / "cameras.txt", 1, args.image_width, args.image_height, yfov)
    _write_cameras_bin(sparse_dir / "cameras.bin", 1, args.image_width, args.image_height, yfov)
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
