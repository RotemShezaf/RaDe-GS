"""
Generate ground truth dataset for polynomial surfaces in TNT format.

Creates a ground truth dataset similar to Tanks and Temples format:
- {Surface}.ply: High-resolution ground truth mesh
- {Surface}.json: Bounding volume metadata
- {Surface}_COLMAP_SfM.log: Camera trajectory file
- {Surface}_trans.txt: Transformation matrix (identity for synthetic data)

The ground truth uses the highest resolution mesh from the raw data.
"""

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.convert_to_logfile import convert_COLMAP_to_log

SURFACES = ("Paraboloid", "Saddle", "HyperbolicParaboloid")


def compute_bounding_polygon(mesh, margin_factor=0.1):
    """Compute a bounding polygon from the mesh.
    
    For polynomial surfaces, we use a rectangular bounding box in XY plane.
    
    Args:
        mesh: trimesh.Trimesh object
        margin_factor: Add margin as fraction of bounds
    
    Returns:
        list of [x, y, 0] vertices forming a polygon
    """
    vertices = mesh.vertices
    x_min, y_min = vertices[:, 0].min(), vertices[:, 1].min()
    x_max, y_max = vertices[:, 0].max(), vertices[:, 1].max()
    
    # Add margin
    x_margin = (x_max - x_min) * margin_factor
    y_margin = (y_max - y_min) * margin_factor
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    # Create rectangular polygon (counter-clockwise)
    polygon = [
        [float(x_min), float(y_min), 0.0],
        [float(x_max), float(y_min), 0.0],
        [float(x_max), float(y_max), 0.0],
        [float(x_min), float(y_max), 0.0],
        [float(x_min), float(y_min), 0.0]  # Close the polygon
    ]
    
    return polygon


def create_json_metadata(mesh, output_path):
    """Create JSON metadata file in TNT format.
    
    Args:
        mesh: trimesh.Trimesh object
        output_path: Path to output .json file
    """
    vertices = mesh.vertices
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    
    polygon = compute_bounding_polygon(mesh)
    
    metadata = {
        "class_name": "SelectionPolygonVolume",
        "version_major": 1,
        "version_minor": 0,
        "orthogonal_axis": "Z",
        "axis_min": z_min,
        "axis_max": z_max,
        "bounding_polygon": polygon
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent="\t")
    
    print(f"  Created {output_path.name}")


def create_trans_file(output_path):
    """Create transformation matrix file (identity for synthetic data).
    
    For synthetic polynomial surfaces, we use an identity matrix because
    the training trajectory and ground truth mesh are in the same coordinate system.
    
    Format matches TNT dataset: 4x4 matrix with scientific notation (18 decimal places).
    
    Args:
        output_path: Path to output _trans.txt file
    """
    # Identity transformation (4x4 matrix)
    # This is correct for synthetic data where training and GT share coordinates
    trans_matrix = np.eye(4, dtype=np.float64)
    
    with open(output_path, 'w') as f:
        for row in trans_matrix:
            # Format: scientific notation with 18 decimals, space-separated
            line = ' '.join(f'{val:.18e}' for val in row)
            f.write(line + '\n')
    
    print(f"  Created {output_path.name}")


def generate_colmap_dataset(raw_data_dir, surface, level, output_dir, num_views=200):
    """Generate synthetic COLMAP dataset for ground truth.
    
    Args:
        raw_data_dir: Path to raw mesh data
        surface: Surface name
        level: Resolution level to use
        output_dir: Output directory for COLMAP data
        num_views: Number of camera views
    """
    # Import here to avoid circular dependencies
    import subprocess
    
    # Create temporary directory for COLMAP dataset
    temp_dir = output_dir / "temp_colmap"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Run create_synthetic_colmap_dataset_from_mesh.py
    cmd = [
        sys.executable,
        str(project_root / "GenerateData" / "create_synthetic_colmap_dataset_from_mesh.py"),
        "--surface", surface,
        "--level", str(level),
        "--mesh_level", str(level),
        "--data_root", str(raw_data_dir),
        "--output_root", str(temp_dir),
        "--num_views", str(num_views),
        "--points3d_count", "1000000",
        "--points3d_colmap_count", "1000000",
        "--camera_radius", "2.4",
        "--camera_distribution", "uniform_sphere",
        "--seed", "42"
    ]
    
    print(f"  Generating COLMAP dataset with {num_views} views...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Warning: COLMAP generation failed:")
        print(result.stderr)
        return None
    
    # Find the generated log file
    colmap_dir = temp_dir / surface / f"level_{level:02d}"
    log_file = colmap_dir / f"{surface}_COLMAP_SfM.log"
    
    if log_file.exists():
        return log_file
    else:
        print(f"  Warning: Log file not found at {log_file}")
        return None


def create_ground_truth_for_surface(surface, raw_data_dir, output_dir, num_views=400):
    """Create complete ground truth dataset for one surface.
    
    Args:
        surface: Surface name (Paraboloid, Saddle, HyperbolicParaboloid)
        raw_data_dir: Path to raw mesh data directory
        output_dir: Path to ground truth output directory
        num_views: Number of camera views for COLMAP
    """
    print(f"\nProcessing {surface}...")
    
    surface_raw_dir = Path(raw_data_dir) / surface
    if not surface_raw_dir.exists():
        print(f"  Error: Raw data directory not found: {surface_raw_dir}")
        return
    
    # Find highest resolution mesh (level 0)
    mesh_files = sorted(surface_raw_dir.glob("mesh_level0_*.ply"))
    if not mesh_files:
        print(f"  Error: No level 0 mesh found in {surface_raw_dir}")
        return
    
    mesh_file = mesh_files[0]
    print(f"  Loading mesh: {mesh_file.name}")
    
    # Load mesh
    mesh = trimesh.load(mesh_file, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  Error: Failed to load mesh from {mesh_file}")
        return
    
    # Create output directory
    surface_output_dir = Path(output_dir) / surface
    surface_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy/save high-resolution PLY
    ply_output = surface_output_dir / f"{surface}.ply"
    mesh.export(ply_output)
    print(f"  Created {ply_output.name}")
    
    # 2. Create JSON metadata
    json_output = surface_output_dir / f"{surface}.json"
    create_json_metadata(mesh, json_output)
    
    # 3. Create transformation file (identity)
    trans_output = surface_output_dir / f"{surface}_trans.txt"
    create_trans_file(trans_output)
    
    # 4. Generate COLMAP dataset and log file
    log_file = generate_colmap_dataset(
        raw_data_dir=raw_data_dir,
        surface=surface,
        level=0,  # Use highest resolution
        output_dir=Path(output_dir),
        num_views=num_views
    )
    
    if log_file and log_file.exists():
        # Copy log file to ground truth directory
        log_output = surface_output_dir / f"{surface}_COLMAP_SfM.log"
        import shutil
        shutil.copy(log_file, log_output)
        print(f"  Created {log_output.name}")
        
        # Clean up temporary COLMAP directory
        import shutil
        temp_dir = Path(output_dir) / "temp_colmap"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print(f"  ✓ Ground truth for {surface} complete")


def main():
    """Main function to generate ground truth dataset."""
    parser = argparse.ArgumentParser(
        description="Generate ground truth dataset for polynomial surfaces"
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="TrainData/Polynomial/raw",
        help="Directory containing raw mesh data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Polynomial/ground_truth",
        help="Output directory for ground truth dataset"
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=SURFACES,
        default=SURFACES,
        help="Surfaces to process"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=400,
        help="Number of camera views for COLMAP dataset"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Ground Truth Dataset Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Raw data directory: {args.raw_data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Surfaces: {', '.join(args.surfaces)}")
    print(f"  Camera views: {args.num_views}")
    
    # Verify raw data exists
    raw_dir = Path(args.raw_data_dir)
    if not raw_dir.exists():
        print(f"\nError: Raw data directory not found: {raw_dir}")
        print("Please run GenerateRawPolynomialMesh.py first to generate raw data.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each surface
    for surface in args.surfaces:
        try:
            create_ground_truth_for_surface(
                surface=surface,
                raw_data_dir=args.raw_data_dir,
                output_dir=args.output_dir,
                num_views=args.num_views
            )
        except Exception as e:
            print(f"\nError processing {surface}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Ground truth generation complete!")
    print("=" * 80)
    
    # Print summary
    print("\nGenerated ground truth structure:")
    print(f"  {args.output_dir}/")
    for surface in args.surfaces:
        surface_dir = Path(args.output_dir) / surface
        if surface_dir.exists():
            print(f"    ├── {surface}/")
            files = sorted(surface_dir.glob("*"))
            for f in files:
                print(f"    │   ├── {f.name}")
    
    print("\nFiles per surface:")
    print("  - {Surface}.ply: High-resolution ground truth mesh")
    print("  - {Surface}.json: Bounding volume metadata")
    print("  - {Surface}_COLMAP_SfM.log: Camera trajectory (200 views)")
    print("  - {Surface}_trans.txt: Transformation matrix (identity)")


if __name__ == "__main__":
    main()
