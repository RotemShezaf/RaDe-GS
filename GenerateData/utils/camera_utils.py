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


def align_mesh_principal_axes(mesh: o3d.geometry.TriangleMesh, return_matrix: bool = False, major_to: str = "z"):
    """Rotate mesh in-place so its principal axes align with specified world axes.

    By default maps the largest principal axis (max eigenvalue) to the world
    Z axis, the second to Y and the third to X. Returns (R, centroid) if
    `return_matrix` is True.

    Args:
        mesh: Open3D TriangleMesh to rotate in-place.
        return_matrix: If True, return (R, centroid) instead of None.
        major_to: target axis for the major principal component: 'x','y', or 'z'.
                  Currently only 'z' is used by callers.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    pts = verts - centroid
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    # Sort eigenvectors by descending eigenvalue (principal components)
    order = np.argsort(w)[::-1]
    V = v[:, order]  # columns: principal axes (major, mid, minor)

    # Build target basis T where columns are target world axes for (major, mid, minor)
    if major_to == "z":
        t_major = np.array([0.0, 0.0, 1.0])
        t_mid = np.array([1.0, 0.0, 0.0])
        t_minor = np.array([0.0, 1.0, 0.0])
    elif major_to == "y":
        t_major = np.array([0.0, 1.0, 0.0])
        t_mid = np.array([0.0, 0.0, 1.0])
        t_minor = np.array([1.0, 0.0, 0.0])
    else:
        t_major = np.array([1.0, 0.0, 0.0])
        t_mid = np.array([0.0, 0.0, 1.0])
        t_minor = np.array([0.0, 1.0, 0.0])

    T = np.column_stack([t_major, t_mid, t_minor])

    # Rotation R should satisfy: R @ V = T  =>  R = T @ V^T
    R = T @ V.T

    # Ensure right-handed rotation (determinant positive)
    if np.linalg.det(R) < 0:
        # Flip minor axis in T and recompute
        T[:, 2] *= -1
        R = T @ V.T

    # Apply rotation about centroid
    rotated = (R @ (verts - centroid).T).T + centroid
    mesh.vertices = o3d.utility.Vector3dVector(rotated)

    # Rotate normals if present
    try:
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            normals_rot = (R @ normals.T).T
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals_rot)
    except Exception:
        pass

    if return_matrix:
        return R, centroid
    return None


def apply_rotation_to_mesh(mesh: o3d.geometry.TriangleMesh, R: np.ndarray, centroid: np.ndarray):
    """Apply rotation R about centroid to mesh vertices and normals (in-place)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    rotated = (R @ (verts - centroid).T).T + centroid
    mesh.vertices = o3d.utility.Vector3dVector(rotated)
    try:
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            normals_rot = (R @ normals.T).T
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals_rot)
    except Exception:
        pass


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
        # Compute the smallest camera radius so that every mesh vertex projects
        # inside the image from every possible viewing direction.
        # The bounding sphere radius gives the worst-case angular extent;
        # we need  radius > bounding_radius / sin(half_fov_min)  so that
        # all vertices fit within the narrower FOV dimension.
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        bounding_radius = np.max(np.linalg.norm(vertices - center, axis=1))
        half_fov_v = math.radians(args.vertical_fov / 2.0)
        aspect = args.image_width / args.image_height
        half_fov_h = math.atan(math.tan(half_fov_v) * aspect)
        min_half_fov = min(half_fov_v, half_fov_h)
        radius = bounding_radius / math.sin(min_half_fov)
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
        if getattr(args, 'camera_distribution', 'orbit') == 'n_circles':
            # N concentric circular orbits at different elevation angles
            # Expected arg: args.circle_elevations as comma-separated degrees, e.g. "25,55,85"
            elevs_arg = getattr(args, 'circle_elevations', None)
            if elevs_arg is None:
                raise ValueError("camera_distribution='n_circles' requires --circle_elevations")
            parts = [p.strip() for p in str(elevs_arg).split(',') if p.strip()]
            if len(parts) < 2:
                raise ValueError("--circle_elevations expects at least two comma-separated elevation degrees, e.g. '25,55,85'")
            n_circles = len(parts)
            elev_rads = [math.radians(float(p)) for p in parts]

            # Split views as evenly as possible across circles
            base_n = num_views // n_circles
            remainder = num_views % n_circles
            views_per_circle = [base_n + (1 if i < remainder else 0) for i in range(n_circles)]

            centers_list = []
            for ci, (elev, nv) in enumerate(zip(elev_rads, views_per_circle)):
                theta = np.linspace(0, 2 * math.pi, nv, endpoint=False)
                for angle in theta:
                    cx = center[0] + radius * math.cos(angle) * math.sin(elev)
                    cy = center[1] + radius * math.sin(angle) * math.sin(elev)
                    cz = center[2] + radius * math.cos(elev)
                    centers_list.append([cx, cy, cz])

            centers = np.asarray(centers_list)
        else:
            theta = np.linspace(0, 2 * math.pi, num_views, endpoint=False)
            rng = np.random.default_rng(seed)
            rng.shuffle(theta)
            elevation = math.radians(args.elevation_deg)
            centers = []
            for angle in theta:
                x = center[0] + radius * math.cos(angle) * math.sin(elevation)
                y = center[1] + radius * math.sin(angle) * math.sin(elevation)
                z = center[2] + radius * math.cos(elevation)
                centers.append([x, y, z])
            centers = np.asarray(centers)

    # Ensure targets array length matches the actual number of centers
    n_centers = centers.shape[0]
    targets = np.repeat(center[None, :], n_centers, axis=0)
    return centers, targets





