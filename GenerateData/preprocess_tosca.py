#!/usr/bin/env python3
"""
Preprocess TOSCA dataset from MATLAB format to organized PLY files.

This script processes TOSCA meshes from matlab/ directory and creates:
- Organized folders for each shape
- PLY files for meshes with normals
- PLY files for point clouds
- Normals as separate PLY files
- Both high and low resolution versions
- Copies of mesh visualization images

Structure created:
data/TOSCA/processed/
    ├── cat0/
    │   ├── mesh_high_res.ply
    │   ├── pointcloud_high_res.ply
    │   ├── normals_high_res.ply
    │   ├── mesh_low_res.ply
    │   ├── pointcloud_low_res.ply
    │   ├── normals_low_res.ply
    │   ├── image_high_res.png
    │   └── image_low_res.png
    └── ...
"""

# Ensure project root is in sys.path for module imports
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
import argparse
import shutil
from tqdm import tqdm


def load_matlab_mesh(mat_file):
    """Load mesh from MATLAB .mat file.
    
    Args:
        mat_file: Path to .mat file
        
    Returns:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
    """
    try:
        import scipy.io
        data = scipy.io.loadmat(mat_file)
        
        # TOSCA .mat files typically contain 'surface' struct with 'X', 'Y', 'Z' and 'TRIV'
        # or directly 'X', 'Y', 'Z' and 'TRIV'
        if 'surface' in data:
            surface = data['surface']
            # Handle structured array
            X = surface['X'][0, 0].flatten()
            Y = surface['Y'][0, 0].flatten()
            Z = surface['Z'][0, 0].flatten()
            faces = surface['TRIV'][0, 0] - 1  # MATLAB is 1-indexed
        else:
            # Direct arrays
            X = data.get('X', data.get('VERT', None))
            Y = data.get('Y', None)
            Z = data.get('Z', None)
            faces = data.get('TRIV', data.get('TRIV', None))
            
            if X is not None and X.ndim > 1 and X.shape[1] == 3:
                # Already in Nx3 format
                vertices = X
            elif X is not None and Y is not None and Z is not None:
                X = X.flatten()
                Y = Y.flatten()
                Z = Z.flatten()
                vertices = np.column_stack([X, Y, Z])
            else:
                raise ValueError(f"Cannot parse vertex data from {mat_file}")
            
            if faces is not None:
                faces = faces - 1  # Convert from MATLAB 1-indexed to 0-indexed
        
        if 'vertices' not in locals():
            vertices = np.column_stack([X, Y, Z])
            
        return vertices, faces
        
    except ImportError:
        print("scipy not available, trying alternative loading method...")
        # Fallback: try loading from .vert and .tri files if they exist
        base_path = mat_file.parent / mat_file.stem
        vert_file = base_path.with_suffix('.vert')
        tri_file = base_path.with_suffix('.tri')
        
        if vert_file.exists() and tri_file.exists():
            vertices = np.loadtxt(vert_file)
            faces = np.loadtxt(tri_file, dtype=int) - 1  # Convert to 0-indexed
            return vertices, faces
        else:
            raise RuntimeError(f"Cannot load {mat_file}: scipy not available and .vert/.tri files not found")


def compute_vertex_normals(vertices, faces):
    """Compute vertex normals from mesh.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        
    Returns:
        normals: Nx3 array of vertex normals
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.vertex_normals


def compute_mean_arc_length(mesh):
    """Compute mean arc length (average edge length) for a mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        
    Returns:
        mean_arc_length: average length of edges in the mesh
    """
    # Get all edges from the mesh
    edges = mesh.edges_unique
    
    # Compute edge lengths
    edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Return mean edge length
    mean_arc_length = np.mean(edge_lengths)
    
    return mean_arc_length


def process_mesh_file(mat_file, output_dir, resolution, copy_image=True):
    """Process a single MATLAB mesh file.
    
    Args:
        mat_file: Path to .mat file
        output_dir: Output directory for this shape
        resolution: 'high_res' or 'low_res'
        copy_image: Whether to copy the corresponding PNG image
    """
    try:
        # Load mesh
        vertices, faces = load_matlab_mesh(mat_file)
        
        # Create trimesh mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # Compute normals
        normals = mesh.vertex_normals
        
        # Compute mean arc length
        mean_arc_length = compute_mean_arc_length(mesh)
        
        # Save mesh with normals (include arc length in filename)
        mesh_filename = output_dir / f"mesh_{resolution}_arc{mean_arc_length:.6f}.ply"
        mesh.export(mesh_filename)
        
        # Save point cloud (just vertices)
        point_cloud = trimesh.PointCloud(vertices=vertices)
        pc_filename = output_dir / f"pointcloud_{resolution}_arc{mean_arc_length:.6f}.ply"
        point_cloud.export(pc_filename)
        
        # Save normals as point cloud with normals
        normals_pc = trimesh.PointCloud(vertices=vertices, normals=normals)
        normals_filename = output_dir / f"normals_{resolution}_arc{mean_arc_length:.6f}.ply"
        normals_pc.export(normals_filename)
        
        # Copy image if it exists
        if copy_image:
            image_file = mat_file.with_suffix('.png')
            if image_file.exists():
                image_out = output_dir / f"image_{resolution}.png"
                shutil.copy2(image_file, image_out)
        
        # Save metadata
        metadata_filename = output_dir / f"metadata_{resolution}.txt"
        with open(metadata_filename, 'w') as f:
            f.write(f"Shape: {mat_file.stem}\n")
            f.write(f"Resolution: {resolution}\n")
            f.write(f"Total Vertices: {len(vertices)}\n")
            f.write(f"Total Faces: {len(faces)}\n")
            f.write(f"Mean Arc Length: {mean_arc_length:.6f}\n")
            f.write(f"Mesh File: {mesh_filename.name}\n")
            f.write(f"Point Cloud File: {pc_filename.name}\n")
            f.write(f"Normals File: {normals_filename.name}\n")
        
        return True, len(vertices), len(faces), mean_arc_length
        
    except Exception as e:
        print(f"Error processing {mat_file}: {e}")
        return False, 0, 0, 0.0


def preprocess_tosca_dataset(matlab_dir, output_dir, resolutions=['high_res', 'low_res']):
    """Preprocess entire TOSCA dataset.
    
    Args:
        matlab_dir: Path to TOSCA/matlab directory
        output_dir: Output directory for processed data
        resolutions: List of resolution folders to process
    """
    matlab_path = Path(matlab_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TOSCA Dataset Preprocessor")
    print("=" * 80)
    print(f"\nInput directory: {matlab_path}")
    print(f"Output directory: {output_path}")
    print(f"Resolutions: {', '.join(resolutions)}")
    
    # Collect all .mat files by shape name
    shape_files = {}
    
    for resolution in resolutions:
        res_dir = matlab_path / resolution
        if not res_dir.exists():
            print(f"\nWarning: Resolution directory not found: {res_dir}")
            continue
        
        mat_files = list(res_dir.glob("*.mat"))
        print(f"\nFound {len(mat_files)} .mat files in {resolution}/")
        
        for mat_file in mat_files:
            shape_name = mat_file.stem  # e.g., 'cat0', 'centaur1'
            
            if shape_name not in shape_files:
                shape_files[shape_name] = {}
            
            shape_files[shape_name][resolution] = mat_file
    
    print(f"\nTotal unique shapes: {len(shape_files)}")
    
    # Process each shape
    stats = {
        'total_shapes': len(shape_files),
        'successful': 0,
        'failed': 0,
        'total_vertices': {},
        'total_faces': {},
        'mean_arc_lengths': {}
    }
    
    print("\nProcessing shapes...")
    for shape_name in tqdm(sorted(shape_files.keys())):
        shape_output_dir = output_path / shape_name
        shape_output_dir.mkdir(parents=True, exist_ok=True)
        
        shape_success = True
        
        # Process each resolution for this shape
        for resolution in resolutions:
            if resolution in shape_files[shape_name]:
                mat_file = shape_files[shape_name][resolution]
                success, num_verts, num_faces, arc_length = process_mesh_file(
                    mat_file, shape_output_dir, resolution, copy_image=True
                )
                
                if success:
                    if resolution not in stats['total_vertices']:
                        stats['total_vertices'][resolution] = []
                        stats['total_faces'][resolution] = []
                        stats['mean_arc_lengths'][resolution] = []
                    stats['total_vertices'][resolution].append(num_verts)
                    stats['total_faces'][resolution].append(num_faces)
                    stats['mean_arc_lengths'][resolution].append(arc_length)
                else:
                    shape_success = False
        
        if shape_success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nTotal shapes: {stats['total_shapes']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    for resolution in resolutions:
        if resolution in stats['total_vertices']:
            verts = stats['total_vertices'][resolution]
            faces = stats['total_faces'][resolution]
            arcs = stats['mean_arc_lengths'][resolution]
            print(f"\n{resolution}:")
            print(f"  Average vertices: {np.mean(verts):.0f} (min: {np.min(verts)}, max: {np.max(verts)})")
            print(f"  Average faces: {np.mean(faces):.0f} (min: {np.min(faces)}, max: {np.max(faces)})")
            print(f"  Average arc length: {np.mean(arcs):.6f} (min: {np.min(arcs):.6f}, max: {np.max(arcs):.6f})")
    
    # Print directory structure example
    print("\nOutput structure:")
    print(f"{output_path}/")
    example_shapes = sorted(shape_files.keys())[:3]
    for i, shape in enumerate(example_shapes):
        prefix = "├──" if i < len(example_shapes) - 1 else "└──"
        print(f"  {prefix} {shape}/")
        files = [
            "mesh_high_res_arc*.ply",
            "pointcloud_high_res_arc*.ply",
            "normals_high_res_arc*.ply",
            "mesh_low_res_arc*.ply",
            "pointcloud_low_res_arc*.ply",
            "normals_low_res_arc*.ply",
            "metadata_high_res.txt",
            "metadata_low_res.txt",
            "image_high_res.png",
            "image_low_res.png"
        ]
        for j, f in enumerate(files):
            file_prefix = "│   ├──" if i < len(example_shapes) - 1 else "    ├──"
            if j == len(files) - 1:
                file_prefix = "│   └──" if i < len(example_shapes) - 1 else "    └──"
            print(f"  {file_prefix} {f}")
    if len(shape_files) > 3:
        print(f"  └── ... ({len(shape_files) - 3} more shapes)")
    
    print(f"\nProcessed data saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess TOSCA dataset from MATLAB format to organized PLY files"
    )
    parser.add_argument(
        '--matlab_dir',
        type=str,
        default='TrainData/TOSCA/matlab',
        help='Path to TOSCA matlab directory (default: TrainData/TOSCA/matlab)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='TrainData/TOSCA/processed',
        help='Output directory for processed data (default: data/TOSCA/processed)'
    )
    parser.add_argument(
        '--resolutions',
        type=str,
        nargs='+',
        default=['high_res', 'low_res'],
        help='Resolutions to process (default: high_res low_res)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    matlab_dir = Path(args.matlab_dir)
    if not matlab_dir.is_absolute():
        matlab_dir = project_root / matlab_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    # Check input directory exists
    if not matlab_dir.exists():
        print(f"Error: MATLAB directory not found: {matlab_dir}")
        print("\nPlease ensure the TOSCA dataset is downloaded and extracted.")
        sys.exit(1)
    
    # Process dataset
    preprocess_tosca_dataset(
        matlab_dir=matlab_dir,
        output_dir=output_dir,
        resolutions=args.resolutions
    )


if __name__ == "__main__":
    main()
