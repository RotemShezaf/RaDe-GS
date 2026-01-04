"""Visualize COLMAP dataset cameras and 3D points using Open3D."""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    qvec2rotmat,
)


def create_camera_frustum(cam_intrinsic, cam_extrinsic, scale=0.3):
    """Create a line set representing a camera frustum."""
    
    # Camera intrinsics
    fx = cam_intrinsic.params[0]
    fy = cam_intrinsic.params[1]
    cx = cam_intrinsic.params[2]
    cy = cam_intrinsic.params[3]
    w = cam_intrinsic.width
    h = cam_intrinsic.height
    
    # Extrinsics: COLMAP uses world-to-camera convention
    R = qvec2rotmat(cam_extrinsic.qvec)
    t = cam_extrinsic.tvec
    
    # Camera center in world coordinates
    C = -R.T @ t
    
    # Four corners of the image plane in camera coordinates
    corners_cam = np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1],      # top-left
        [(w - cx) / fx, (0 - cy) / fy, 1],      # top-right
        [(w - cx) / fx, (h - cy) / fy, 1],      # bottom-right
        [(0 - cx) / fx, (h - cy) / fy, 1],      # bottom-left
    ]) * scale
    
    # Transform to world coordinates
    corners_world = (R.T @ corners_cam.T).T + C
    
    # Create frustum lines
    points = np.vstack([C.reshape(1, 3), corners_world])
    
    # Lines: from camera center to each corner, and around the image plane
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # around image plane
    ]
    
    colors = [[1, 0, 0] for _ in lines]  # Red frustums
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def visualize_colmap_dataset(sparse_dir: Path, max_points: int = 100000, frustum_scale: float = 0.3):
    """Visualize cameras and 3D points from a COLMAP dataset."""
    
    print(f"Loading COLMAP dataset from: {sparse_dir}")
    
    # Read data
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    points3d = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    
    print(f"Loaded: {len(cameras)} cameras, {len(images)} images, {len(points3d)} points3D")
    
    geometries = []
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Add camera frustums
    print("Creating camera frustums...")
    for img_id, img_data in images.items():
        cam_intrinsic = cameras[img_data.camera_id]
        frustum = create_camera_frustum(cam_intrinsic, img_data, scale=frustum_scale)
        geometries.append(frustum)
    
    # Add 3D points
    if len(points3d) > 0:
        print(f"Creating point cloud (showing up to {max_points} points)...")
        
        # Sample points if too many
        point_ids = list(points3d.keys())
        if len(point_ids) > max_points:
            np.random.seed(42)
            point_ids = np.random.choice(point_ids, max_points, replace=False)
        
        xyz = np.array([points3d[pid].xyz for pid in point_ids])
        rgb = np.array([points3d[pid].rgb for pid in point_ids]) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        geometries.append(pcd)
        
        # Print stats
        print(f"  Point cloud bounds:")
        print(f"    X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"    Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"    Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
    
    # Visualize
    print("\nLaunching visualizer...")
    print("Controls:")
    print("  - Mouse: Rotate/pan/zoom")
    print("  - Q/ESC: Quit")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="COLMAP Dataset Visualization",
        width=1280,
        height=720,
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize a COLMAP dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset root (should contain sparse/0/) or directly to sparse/0/"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=100000,
        help="Maximum number of 3D points to display"
    )
    parser.add_argument(
        "--frustum_scale",
        type=float,
        default=0.3,
        help="Scale factor for camera frustum size"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Handle both dataset root and sparse/0 paths
    if (dataset_path / "sparse" / "0").exists():
        sparse_dir = dataset_path / "sparse" / "0"
    elif dataset_path.name == "0" and (dataset_path.parent.name == "sparse"):
        sparse_dir = dataset_path
    else:
        print(f"ERROR: Could not find sparse/0 directory in {dataset_path}")
        sys.exit(1)
    
    visualize_colmap_dataset(sparse_dir, args.max_points, args.frustum_scale)


if __name__ == "__main__":
    main()
