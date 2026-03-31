from dataclasses import dataclass
import numpy as np
import open3d as o3d
import math
from typing import Tuple
import argparse

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

    if getattr(args, 'auto_camera_radius', False):
        # Compute the smallest radius where the entire mesh bounding sphere fits in the FOV
        bounding_radius = extent / 2.0
        half_fov_v = math.radians(args.vertical_fov / 2.0)
        aspect = args.image_width / args.image_height
        half_fov_h = math.atan(math.tan(half_fov_v) * aspect)
        min_half_fov = min(half_fov_v, half_fov_h)
        radius = bounding_radius / math.tan(min_half_fov)
        print(f"[Auto radius] bounding_radius={bounding_radius:.3f}, "
              f"FOV_v={args.vertical_fov:.1f}\u00b0, aspect={aspect:.2f} "
              f"=> camera_radius={radius:.3f}")
    elif args.camera_radius is not None:
        radius = args.camera_radius
    else:
        radius = extent * args.orbit_radius_scale

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





