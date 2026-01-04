"""
Generate triangulated meshes and point clouds for polynomial surfaces.

Creates data for three surfaces:
1. Paraboloid: z = x^2 + y^2
2. Saddle: z = x^2 - y^2
3. Hyperbolic Paraboloid: z = x^2 - y^2 + xy

Each surface is generated at multiple resolutions with corresponding point clouds.
"""


import numpy as np
import trimesh
import os
from pathlib import Path
import argparse


def evaluate_polynomial_normal(x, y, surface_type):
    """Analytical surface normal for the polynomial surfaces.

    Uses the implicit surface F(x, y, z) = z - f(x, y) and normal n ~ (-df/dx, -df/dy, 1).
    """
    if surface_type == 'Paraboloid':
        # z = x^2 + y^2 -> df/dx = 2x, df/dy = 2y
        dfdx = 2.0 * x
        dfdy = 2.0 * y
    elif surface_type == 'Saddle':
        # z = x^2 - y^2 -> df/dx = 2x, df/dy = -2y
        dfdx = 2.0 * x
        dfdy = -2.0 * y
    elif surface_type == 'HyperbolicParaboloid':
        # z = x^2 - y^2 + x*y -> df/dx = 2x + y, df/dy = -2y + x
        dfdx = 2.0 * x + y
        dfdy = -2.0 * y + x
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")

    nx = -dfdx
    ny = -dfdy
    nz = np.ones_like(x)
    n = np.stack([nx, ny, nz], axis=-1)
    # Normalize
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    return n / n_norm


def evaluate_polynomial(x, y, surface_type):
    """Evaluate polynomial surface at given coordinates.
    
    Args:
        x: x coordinates
        y: y coordinates
        surface_type: 'Paraboloid', 'Saddle', or 'HyperbolicParaboloid'
    
    Returns:
        z: height values
    """
    if surface_type == 'Paraboloid':
        # z = x^2 + y^2
        return x**2 + y**2
    elif surface_type == 'Saddle':
        # z = x^2 - y^2
        return x**2 - y**2
    elif surface_type == 'HyperbolicParaboloid':
        # z = x^2 - y^2 + xy
        return x**2 - y**2 + x*y
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")


def compute_arc_length_resolution(x_range, y_range, nx, ny, surface_type):
    """Compute approximate arc length resolution for the mesh.
    
    Args:
        x_range: (min, max) for x
        y_range: (min, max) for y
        nx: number of points in x direction
        ny: number of points in y direction
        surface_type: type of surface
    
    Returns:
        arc_length: average arc length between adjacent vertices
    """
    # Sample a few points to estimate arc length
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Sample middle row for arc length calculation
    mid_idx = ny // 2
    x_sample = x[:min(10, len(x))]
    y_sample = y[mid_idx]
    
    z_sample = evaluate_polynomial(x_sample, np.full_like(x_sample, y_sample), surface_type)
    
    # Compute arc length along x direction
    arc_lengths = []
    for i in range(len(x_sample) - 1):
        dz = z_sample[i+1] - z_sample[i]
        arc = np.sqrt(dx**2 + dz**2)
        arc_lengths.append(arc)
    
    avg_arc_length = np.mean(arc_lengths) if arc_lengths else dx
    
    return avg_arc_length


def _adaptive_sample_1d(param_range, num_points, surface_type, axis='x', y_fixed=0.0, x_fixed=0.0):
    """Sample parameter range adaptively to achieve uniform arc length spacing.
    
    Uses cumulative arc length to distribute points uniformly along the curve.
    
    Args:
        param_range: (min, max) for the parameter
        num_points: number of points to sample
        surface_type: type of surface
        axis: 'x' or 'y' - which axis to sample
        y_fixed: fixed y value when sampling x
        x_fixed: fixed x value when sampling y
    
    Returns:
        samples: array of parameter values with uniform arc length spacing
    """
    # First, sample densely to estimate arc length
    dense_samples = 1000
    t = np.linspace(param_range[0], param_range[1], dense_samples)
    
    if axis == 'x':
        x = t
        y = np.full_like(t, y_fixed)
    else:  # axis == 'y'
        x = np.full_like(t, x_fixed)
        y = t
    
    z = evaluate_polynomial(x, y, surface_type)
    
    # Compute arc length at each point
    if axis == 'x':
        dx = np.diff(x)
        dz = np.diff(z)
        ds = np.sqrt(dx**2 + dz**2)
    else:
        dy = np.diff(y)
        dz = np.diff(z)
        ds = np.sqrt(dy**2 + dz**2)
    
    # Cumulative arc length
    arc_lengths = np.concatenate([[0], np.cumsum(ds)])
    total_length = arc_lengths[-1]
    
    # Sample uniformly in arc length space
    target_arc_lengths = np.linspace(0, total_length, num_points)
    
    # Interpolate back to parameter space
    uniform_samples = np.interp(target_arc_lengths, arc_lengths, t)
    
    return uniform_samples


def generate_surface_mesh(surface_type, nx, ny, x_range, y_range, adaptive=False):
    """Generate triangulated mesh for polynomial surface.
    
    Args:
        surface_type: 'Paraboloid', 'Saddle', or 'HyperbolicParaboloid'
        nx: number of points in x direction
        ny: number of points in y direction
        x_range: (min, max) for x
        y_range: (min, max) for y
        adaptive: if True, use adaptive sampling based on arc length
    
    Returns:
        mesh: trimesh.Trimesh object
        arc_length: resolution metric
    """
    if adaptive:
        # Adaptive sampling: distribute points uniformly in arc length
        x = _adaptive_sample_1d(x_range, nx, surface_type, axis='x', y_fixed=0.0)
        y = _adaptive_sample_1d(y_range, ny, surface_type, axis='y', x_fixed=0.0)
    else:
        # Uniform parameter space sampling
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
    
    X, Y = np.meshgrid(x, y)
    
    # Evaluate surface
    Z = evaluate_polynomial(X, Y, surface_type)
    
    # Flatten to vertices
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    # Analytical normals
    normals = evaluate_polynomial_normal(X, Y, surface_type).reshape(-1, 3)
    
    # Create triangular faces (two triangles per grid cell)
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Vertex indices for current quad
            v0 = i * nx + j
            v1 = i * nx + (j + 1)
            v2 = (i + 1) * nx + j
            v3 = (i + 1) * nx + (j + 1)
            
            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
    
    # Compute arc length resolution
    arc_length = compute_arc_length_resolution(x_range, y_range, nx, ny, surface_type)
    
    return mesh, normals, arc_length


def downsample_mesh(mesh, target_vertices):
    """Downsample mesh to target number of vertices.
    
    Args:
        mesh: trimesh.Trimesh object
        target_vertices: approximate target number of vertices
    
    Returns:
        downsampled_mesh: new mesh with fewer vertices
    """
    if len(mesh.vertices) <= target_vertices:
        return mesh
    
    # Use simplify_quadric_decimation for quality downsampling
    try:
        downsampled = mesh.simplify_quadric_decimation(target_vertices)
        return downsampled
    except:
        # Fallback to uniform sampling if decimation fails
        print(f"  Warning: Quadric decimation failed, using uniform sampling")
        step = int(np.ceil(np.sqrt(len(mesh.vertices) / target_vertices)))
        return mesh


def mesh_to_point_cloud(mesh):
    """Convert mesh to point cloud by sampling vertices.
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        point_cloud: trimesh.PointCloud object
    """
    return trimesh.PointCloud(vertices=mesh.vertices)


def generate_multiresolution_data(surface_type, base_resolution=200, num_levels=5, 
                                   output_dir='TrainData/raw', x_range=(-0.5, 0.5), y_range=(-1, 1),
                                   adaptive_sampling=False):
    """Generate multi-resolution meshes and point clouds for a surface.
    
    Args:
        surface_type: 'Paraboloid', 'Saddle', or 'HyperbolicParaboloid'
        base_resolution: highest resolution (number of points per dimension)
        num_levels: number of resolution levels
        output_dir: base output directory
        adaptive_sampling: if True, use arc-length adaptive sampling for uniform mesh
    """
    # Create output directory
    surface_dir = Path(output_dir) / surface_type
    surface_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating data for {surface_type}...")
    print(f"Output directory: {surface_dir}")
    print(f"Adaptive sampling: {'enabled' if adaptive_sampling else 'disabled'}")
    
    # Generate resolution levels
    resolutions = []
    for level in range(num_levels):
        # Exponentially decrease resolution
        factor = 2 ** level
        nx = ny = max(10, base_resolution // factor)
        resolutions.append((nx, ny))
    
    # Generate meshes at each resolution
    for level, (nx, ny) in enumerate(resolutions):
        print(f"\n  Level {level}: Resolution {nx}x{ny}")
        # Generate mesh
        mesh, normals, arc_length = generate_surface_mesh(surface_type, nx, ny, x_range, y_range, adaptive=adaptive_sampling)
        print(f"    Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"    Arc length resolution: {arc_length:.6f}")
        # Save mesh
        mesh_filename = surface_dir / f"mesh_level{level}_res{nx}x{ny}_arc{arc_length:.6f}.ply"
        mesh.export(mesh_filename)
        print(f"    Saved mesh: {mesh_filename.name}")
        # Generate point cloud from mesh vertices
        point_cloud = mesh_to_point_cloud(mesh)
        # Save point cloud
        pc_filename = surface_dir / f"pointcloud_level{level}_res{nx}x{ny}_arc{arc_length:.6f}.ply"
        point_cloud.export(pc_filename)
        print(f"    Saved point cloud: {pc_filename.name}")

        # Save analytical vertex normals as a separate PLY point cloud
        normals_pc = trimesh.PointCloud(vertices=mesh.vertices, normals=normals)
        normals_filename = surface_dir / f"normals_level{level}_res{nx}x{ny}_arc{arc_length:.6f}.ply"
        normals_pc.export(normals_filename)
        print(f"    Saved normals: {normals_filename.name}")
        # Save metadata
        metadata_filename = surface_dir / f"metadata_level{level}.txt"
        with open(metadata_filename, 'w') as f:
            f.write(f"Surface Type: {surface_type}\n")
            f.write(f"Resolution Level: {level}\n")
            f.write(f"Grid Size: {nx} x {ny}\n")
            f.write(f"Total Vertices: {len(mesh.vertices)}\n")
            f.write(f"Total Faces: {len(mesh.faces)}\n")
            f.write(f"Arc Length Resolution: {arc_length:.6f}\n")
            f.write(f"X Range: {x_range}\n")
            f.write(f"Y Range: {y_range}\n")
            f.write(f"Mesh File: {mesh_filename.name}\n")
            f.write(f"Point Cloud File: {pc_filename.name}\n")
            f.write(f"Normals File: {normals_filename.name}\n")




def main():
    """Main function to generate all datasets."""
    parser = argparse.ArgumentParser(description="Polynomial Surface Mesh and Point Cloud Generator")
    parser.add_argument('--base_resolution', type=int, default=1000, help='Highest resolution grid size (default: 1000)')
    parser.add_argument('--num_levels', type=int, default=6, help='Number of resolution levels (default: 6)')
    parser.add_argument('--output_dir', type=str, default='TrainData/Polynomial/raw', help='Output directory (default: TrainData/raw)')
    parser.add_argument('--x_min', type=float, default=-0.8, help='Minimum x value (default: -1.0)')
    parser.add_argument('--x_max', type=float, default=0.8, help='Maximum x value (default: 1.0)')
    parser.add_argument('--y_min', type=float, default=-0.8, help='Minimum y value (default: -1.0)')
    parser.add_argument('--y_max', type=float, default=0.8, help='Maximum y value (default: 1.0)')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive sampling for uniform arc length spacing')
    args = parser.parse_args()

    surfaces = ['Paraboloid', 'Saddle', 'HyperbolicParaboloid']
    base_resolution = args.base_resolution
    num_levels = args.num_levels
    output_dir = args.output_dir
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    print("=" * 80)
    print("Polynomial Surface Mesh and Point Cloud Generator")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Surfaces: {', '.join(surfaces)}")
    print(f"  Base Resolution: {base_resolution}x{base_resolution}")
    print(f"  Number of Levels: {num_levels}")
    print(f"  Output Directory: {output_dir}")
    print(f"  X Range: {x_range}")
    print(f"  Y Range: {y_range}")
    print(f"  Adaptive Sampling: {'enabled' if args.adaptive else 'disabled'}")

    # Generate data for each surface
    for surface_type in surfaces:
        try:
            generate_multiresolution_data(
                surface_type=surface_type,
                base_resolution=base_resolution,
                num_levels=num_levels,
                output_dir=output_dir,
                x_range=x_range,
                y_range=y_range,
                adaptive_sampling=args.adaptive
            )
        except Exception as e:
            print(f"\nError generating {surface_type}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Data generation complete!")
    print("=" * 80)

    # Print summary
    print("\nGenerated datasets:")
    for surface_type in surfaces:
        surface_dir = Path(output_dir) / surface_type
        if surface_dir.exists():
            mesh_files = list(surface_dir.glob("mesh_*.ply"))
            pc_files = list(surface_dir.glob("pointcloud_*.ply"))
            print(f"\n  {surface_type}:")
            print(f"    Location: {surface_dir}")
            print(f"    Meshes: {len(mesh_files)}")
            print(f"    Point Clouds: {len(pc_files)}")

    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── Paraboloid/")
    print(f"    │   ├── mesh_level{args.num_levels}_res{args.base_resolution}x{args.base_resolution}_arc*.ply")
    print(f"    │   ├── pointcloud_level{args.num_levels}_res{args.base_resolution}x{args.base_resolution}_arc*.ply")
    print(f"    │   ├── metadata_level{args.num_levels}.txt")
    print(f"    │   └── ... ({args.num_levels} levels total)")
    print(f"    ├── Saddle/")
    print(f"    │   └── ... ({args.num_levels} levels total)")
    print(f"    └── HyperbolicParaboloid/")
    print(f"        └── ... ({args.num_levels} levels total)")

if __name__ == "__main__":
    main()
