#!/usr/bin/env python3
"""
Simple mesh rendering script using matplotlib for headless rendering.
Works without open3d dependency.

Usage:
    python render_mesh_simple.py <input_ply> <output_prefix> [max_points]
    
Arguments:
    input_ply: Path to input PLY file
    output_prefix: Prefix for output image files
    max_points: Maximum number of points to render (default: 100000)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import sys
import argparse

def read_ply_points(filepath, max_points=100000):
    """Read PLY file and extract vertex positions and colors."""
    print(f"Reading PLY file: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        
        # Parse header to get vertex count
        vertex_count = 0
        for line in header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                break
        
        print(f"Total vertices: {vertex_count:,}")
        
        # Read binary vertex data (downsampled)
        step = max(1, vertex_count // max_points)
        print(f"Downsampling: reading every {step} points")
        
        vertices = []
        colors = []
        
        # Each vertex: 3 doubles (xyz) + 3 uchars (rgb) = 24 + 3 = 27 bytes
        vertex_format = '<dddBBB'  # little endian: 3 doubles, 3 unsigned chars
        vertex_size = struct.calcsize(vertex_format)
        
        for i in range(vertex_count):
            data = f.read(vertex_size)
            if i % step == 0:
                x, y, z, r, g, b = struct.unpack(vertex_format, data)
                vertices.append([x, y, z])
                colors.append([r/255.0, g/255.0, b/255.0])
        
        vertices = np.array(vertices)
        colors = np.array(colors)
        
        print(f"Loaded {len(vertices):,} points for rendering")
        
        return vertices, colors

def render_mesh_images(vertices, colors, output_prefix):
    """Render mesh from multiple viewpoints."""
    
    # Calculate bounds
    center = vertices.mean(axis=0)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = extent.max()
    
    print(f"Mesh center: {center}")
    print(f"Mesh extent: {extent}")
    
    # Define viewpoints (azimuth, elevation)
    viewpoints = [
        (45, 30, "view_1"),
        (135, 30, "view_2"),
        (225, 30, "view_3"),
        (315, 30, "view_4"),
        (0, 90, "view_top"),
        (0, 0, "view_front"),
    ]
    
    for azim, elev, name in viewpoints:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points with colors
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c=colors, s=1, alpha=0.5)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Barn Mesh - {name} (azim={azim}°, elev={elev}°)')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Save figure
        output_path = f"{output_prefix}_{name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render PLY mesh to images from multiple viewpoints'
    )
    parser.add_argument('input_ply', type=str, 
                        help='Path to input PLY file')
    parser.add_argument('output_prefix', type=str,
                        help='Prefix for output image files')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Maximum number of points to render (default: 100000)')
    
    args = parser.parse_args()
    
    # Read and render
    vertices, colors = read_ply_points(args.input_ply, max_points=args.max_points)
    render_mesh_images(vertices, colors, args.output_prefix)
    
    print("\nDone! Rendered images saved.")
