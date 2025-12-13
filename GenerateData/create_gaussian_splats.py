"""Convert generated polynomial surface data into a Gaussian Splat representation.

This helper loads the meshes/point clouds produced by GenerateRawPolynomialMesh.py,
selects a specific resolution level, and initializes a GaussianModel that can be
used directly with the RaDe-GS training/rendering stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import trimesh

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud

SURFACE_COLORS = {
    "Paraboloid": np.array([0.9, 0.4, 0.0], dtype=np.float32),
    "Saddle": np.array([0.2, 0.7, 0.9], dtype=np.float32),
    "HyperbolicParaboloid": np.array([0.4, 0.9, 0.3], dtype=np.float32),
}


def _load_level_file(surface_dir: Path, prefix: str, level: int) -> Path:
    matches = sorted(surface_dir.glob(f"{prefix}_level{level}_*.ply"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file with prefix '{prefix}_level{level}_' in {surface_dir}"
        )
    if len(matches) > 1:
        print(
            f"[Info] Multiple matches found for {prefix} level {level}. Using {matches[0].name}."
        )
    return matches[0]


def _load_vertices(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Trimesh):
        verts = np.asarray(mesh.vertices)
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(verts):
            normals = np.asarray(mesh.vertex_normals)
        else:
            normals = np.zeros_like(verts)
    elif isinstance(mesh, trimesh.PointCloud):
        verts = np.asarray(mesh.vertices)
        normals = np.zeros_like(verts)
    else:
        raise ValueError(f"Unsupported geometry type for {path}: {type(mesh)}")
    return verts, normals


def _maybe_subsample(points: np.ndarray, normals: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, normals
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], normals[idx]


def _build_colors(points: np.ndarray, scheme: str, surface: str) -> np.ndarray:
    if scheme == "surface":
        base = SURFACE_COLORS.get(surface, np.array([0.8, 0.8, 0.8]))
        return np.tile(base[None, :], (len(points), 1)).astype(np.float32, copy=False)
    # Default: encode height into a blue-red gradient
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.ptp() + 1e-9)
    colors = np.zeros((len(points), 3), dtype=np.float32)
    colors[:, 0] = z_norm  # red channel increases with height
    colors[:, 2] = 1.0 - z_norm  # blue channel decreases with height
    colors[:, 1] = 0.5 * (1.0 - np.abs(z_norm - 0.5))  # green around mid-range
    return np.clip(colors, 0.0, 1.0)


def _create_gaussians(points: np.ndarray, colors: np.ndarray, normals: np.ndarray, sh_degree: int, spatial_lr_scale: float) -> GaussianModel:
    cloud = BasicPointCloud(points=points, colors=colors, normals=normals)
    model = GaussianModel(sh_degree=sh_degree)
    model.create_from_pcd(cloud, spatial_lr_scale=spatial_lr_scale)
    model.reset_3D_filter()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Gaussian Splat representation from generated polynomial data")
    parser.add_argument("surface", choices=["Paraboloid", "Saddle", "HyperbolicParaboloid"], help="Surface type to convert")
    parser.add_argument("level", type=int, help="Resolution level index to load")
    parser.add_argument("--data_root", type=str, default="TrainData/raw", help="Base directory containing generated surfaces")
    parser.add_argument("--output", type=str, default=None, help="Destination path for the Gaussian PLY (defaults to <data_root>/<surface>/gaussians_level<level>.ply)")
    parser.add_argument("--use_mesh", action="store_true", help="Sample vertices from the mesh file instead of the stored point cloud")
    parser.add_argument("--max_points", type=int, default=0, help="Optional cap on the number of vertices used to seed Gaussians")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree for Gaussian colors")
    parser.add_argument("--spatial_lr_scale", type=float, default=1.0, help="Scale factor passed to GaussianModel.create_from_pcd")
    parser.add_argument("--color_scheme", choices=["height", "surface"], default="height", help="How to colorize the synthetic point cloud")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling when max_points > 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surface_dir = Path(args.data_root) / args.surface
    if not surface_dir.exists():
        raise FileNotFoundError(f"Surface directory not found: {surface_dir}")

    prefix = "mesh" if args.use_mesh else "pointcloud"
    data_path = _load_level_file(surface_dir, prefix, args.level)
    print(f"Loading {prefix} data from {data_path}")

    points, normals = _load_vertices(data_path)
    points, normals = _maybe_subsample(points, normals, args.max_points, args.seed)
    if len(points) == 0:
        raise ValueError("No points available after loading/subsampling. Try lowering --max_points or pick another level.")
    points = points.astype(np.float32, copy=False)
    normals = normals.astype(np.float32, copy=False)
    colors = _build_colors(points, args.color_scheme, args.surface)

    print(f"Seeding Gaussian model with {len(points)} points")
    model = _create_gaussians(points, colors, normals, args.sh_degree, args.spatial_lr_scale)

    default_output = surface_dir / f"gaussians_level{args.level}.ply"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_ply(str(output_path))
    print(f"Saved Gaussian splats to {output_path}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to initialize Gaussian splats in this script.")
    main()
