#!/usr/bin/env python3
"""
Compute geodesic distances on Gaussian splats from polynomial surfaces.

This script calculates ground truth geodesic distances on a Gaussian splatting
reconstruction.  Gaussians are projected onto the nearest face of a high-resolution
ground-truth mesh and their distances are obtained by **barycentric interpolation** of the
three face-vertex geodesic distances (O(h²) accuracy).  This supersedes the former
vertex-snapping approach (O(h) accuracy) which erroneously returned distance = 0 for
Gaussian pairs closer than the mesh edge length (~70 % of nearest-neighbour pairs on
typical datasets).

WORKFLOW:
---------
1. Load Gaussian splat from output folder (point_cloud/iteration_X/point_cloud.ply)
2. Load ground truth mesh at level 0 (highest resolution) from raw data
3. Generate a low-resolution source mesh using the same parametric surface definition
4. Map source mesh vertices to nearest ground truth mesh vertices
5. Compute exact geodesic distances on GT mesh using MMP algorithm (gdist)
6. Find closest mesh vertex for each Gaussian splat
7. Transfer geodesic distances from mesh vertices to Gaussians
8. Save results in a structured format for further analysis

The script supports:
- Batch processing of multiple source vertices
- Partial computation with source range specification
- Automatic merging of partial results into complete ground truth
- Caching of expensive geodesic computations

OUTPUT STRUCTURE:
-----------------
{output_folder}/
    geodesic_distance/
        gt_partial/
            sources_range_{start}_{end}.npz    # Partial results for source range
        gt_geodesic.npz                         # Complete merged ground truth

Each .npz file contains:
    - gaussian_positions: (N_gaussians, 3) Gaussian center positions
    - source_indices: (N_sources,) mesh vertex indices of sources
    - source_positions: (N_sources, 3) positions of source vertices
    - geodesic_distances: (N_sources, N_gaussians) geodesic distances
    - closest_mesh_indices: (N_gaussians,) nearest mesh vertex for each Gaussian
    - closest_mesh_distances: (N_gaussians,) Euclidean distance to nearest vertex
    - source_gaussian_indices: (N_sources,) Gaussian indices corresponding to each source

USAGE:
------
# Compute geodesic distances using a 20x20 source mesh:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 20

# Use a different source mesh resolution:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 50

# Compute partial results for a source range (recommended for parallelization):
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --data_root TrainData/Polynomial/raw \\
    --surface Paraboloid \\
    --source_mesh_resolution 20 \\
    --source_start 0 \\
    --source_end 200

# Merge partial results into complete ground truth:
python compute_gaussian_geodesic_distances.py \\
    --gaussian_output output/polynomial/Paraboloid \\
    --merge_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
import pickle
import json
from datetime import datetime
from tqdm import tqdm
# Ensure project root is in sys.path for module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from distance.dist import exact_geodesic_via_vtp_vertex_distance, vertex_dist, exact_geodesic_via_gdist_vertex_distance
from plyfile import PlyData
from utils.geodesic_utils import load_ply
from GenerateData.GenerateRawPolynomialMesh import generate_surface_mesh
from utils.load_utils import load_ground_truth_mesh, find_available_iterations, load_gaussian_data_cpu
from utils.geodesic_utils import compute_exact_geodesic

# Import helper functions from separate module
from GenerateData.utils.compute_gaussian_geodesic_distances_helper import (
    generate_source_mesh_and_map,
    compute_geodesic_distances_for_sources,
    compute_and_save_geodesic_pipeline,
    save_mesh_geodesic_gt,
    load_mesh_geodesic_gt,
    find_missing_sources,
    merge_geodesic_data,
    map_indexes_between_gaussian_and_surfaces,
    find_closest_mesh_vertices,
    find_closest_mesh_faces_barycentric,
    transfer_geodesic_to_gaussians,
    save_partial_results,
    merge_partial_results,
    save_computation_metadata
)
from GenerateData.utils.geodesic_mesh_utils import load_geodesic_mesh

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute geodesic distances on Gaussian splats from polynomial surfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments (unless merge_only is specified)
    parser.add_argument(
        '--gaussian_output',
        type=str,
        required=True,
        help='Path to Gaussian splatting output folder (containing point_cloud/iteration_*/)'
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="TrainData/Polynomial/raw",
        help="Base directory containing raw polynomial mesh data"
    )
    parser.add_argument(
        "--surface",
        type=str,
        choices=["Paraboloid", "Saddle", "HyperbolicParaboloid"],
        help="Surface type (required unless --merge_only is specified)"
    )
    
    # Source vertex selection
    parser.add_argument(
        "--num_sources",
        type=int,
        default=None,
        help="Total number of source vertices to use. If not specified, uses all mesh vertices as sources."
    )
    parser.add_argument(
        "--source_mesh_resolution",
        type=int,
        default=20,
        help="Resolution for generating source mesh (NxN grid). Sources will be vertices of this mesh."
    )
    parser.add_argument(
        "--source_start",
        type=int,
        default=None,
        help="Start index of source vertex range (for partial computation)"
    )
    parser.add_argument(
        "--source_end",
        type=int,
        default=None,
        help="End index of source vertex range (exclusive, for partial computation)"
    )
    parser.add_argument(
        "--source_selection",
        type=str,
        choices=["uniform", "random", "all"],
        default="uniform",
        help="Strategy for selecting source vertices: uniform spacing, random sampling, or all vertices"
    )
    
    # Mesh and iteration selection
    parser.add_argument(
        "--mesh_level",
        type=int,
        default=0,
        help="Mesh resolution level to use for ground truth (0 = highest resolution)"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Gaussian splatting training iteration to use. If not specified, uses the highest available iteration."
    )
    
    # Merge mode
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Only merge existing partial results, skip computation"
    )
    
    # Optional settings
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing mesh geodesic data and compute only for missing sources"
    )
    parser.add_argument(
        "--use_mahalanobis",
        action="store_true",
        help="Use Mahalanobis distance based on Gaussian covariance for distance computations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (used in random source selection)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for geodesic computation. None=sequential, -1=use all CPUs, >0=specific number"
    )
    parser.add_argument(
        "--use_gaussian_mesh",
        action="store_true",
        help=(
            "Use a pre-built geodesic mesh whose vertices include every "
            "Gaussian (built by compute_geodesic_mesh_for_gaussians.py).  "
            "Eliminates barycentric interpolation: Gaussians ARE mesh "
            "vertices, so geodesic distances are read directly."
        ),
    )
    parser.add_argument(
        "--geodesic_method", type=str, choices=["vtp", "mmp", "fmm"], default="mmp",
        help="Geodesic computation method: 'vtp' (exact, requires manifold mesh), "
             "'mmp' (MMP via pygeodesic, works on non-manifold meshes), "
             "or 'fmm' (fast marching). Default: mmp.",
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.merge_only and args.surface is None:
        parser.error("--surface is required unless --merge_only is specified")
    
    if args.source_start is not None and args.source_end is None:
        parser.error("--source_end must be specified when --source_start is provided")
    
    if args.source_end is not None and args.source_start is None:
        parser.error("--source_start must be specified when --source_end is provided")
    
    if args.source_start is not None and args.source_end is not None:
        if args.source_start >= args.source_end:
            parser.error("--source_start must be less than --source_end")
    
    return args


def _run_gaussian_mesh_path(
    args: argparse.Namespace,
    output_folder: Path,
    gaussian_positions: np.ndarray,
) -> None:
    """Geodesic computation using the pre-built Gaussian mesh.

    Because every Gaussian IS a mesh vertex (vertices 0 … N_gauss - 1),
    geodesic distances are read straight from the mesh computation—no
    barycentric interpolation or vertex-snapping needed.
    """
    mesh_dir = output_folder / "geodesic_mesh"
    print(f"\n  Loading pre-built Gaussian mesh from {mesh_dir} …")

    vertices, faces, gaussian_vertex_indices = load_geodesic_mesh(mesh_dir)
    n_gauss = len(gaussian_vertex_indices)
    print(f"    → {len(vertices)} vertices, {len(faces)} faces, "
          f"{n_gauss} Gaussians in mesh")

    # ── Source selection ──────────────────────────────────────────────
    # Map source mesh vertices to nearest Gaussians first, then get the
    # corresponding mesh vertex index for each Gaussian.  This guarantees
    # that source_indices[i] == gaussian_vertex_indices[source_gaussian_indices[i]].
    x_range = (vertices[:, 0].min(), vertices[:, 0].max())
    y_range = (vertices[:, 1].min(), vertices[:, 1].max())

    # gaussian_vertex_indices: (N_gauss,) — mesh vertex for each Gaussian.
    # Pass the positions of those vertices so each source maps to a Gaussian.
    all_source_gaussian_indices, all_source_positions = generate_source_mesh_and_map(
        surface_type=args.surface,
        source_mesh_resolution=args.source_mesh_resolution,
        gaussian_positions=vertices[gaussian_vertex_indices],
        x_range=x_range,
        y_range=y_range,
    )
    # Translate Gaussian indices → mesh vertex indices
    all_source_indices = gaussian_vertex_indices[all_source_gaussian_indices]

    # Determine source range for this run
    if args.source_start is not None and args.source_end is not None:
        source_start = args.source_start
        source_end = min(args.source_end, len(all_source_indices))
    else:
        source_start = 0
        source_end = len(all_source_indices)

    source_indices = all_source_indices[source_start:source_end]
    source_positions = all_source_positions[source_start:source_end]
    source_gaussian_indices = all_source_gaussian_indices[source_start:source_end]

    print(f"\n  Computing geodesics for {len(source_indices)} sources "
          f"on a mesh with {len(vertices)} vertices …")

    # closest_mesh_indices/distances: each Gaussian IS its own vertex
    closest_mesh_indices = gaussian_vertex_indices.copy()
    closest_mesh_distances = np.zeros(n_gauss, dtype=np.float64)

    # ── Pipeline: compute, transfer, and save in parallel batches ─────
    compute_and_save_geodesic_pipeline(
        vertices=vertices,
        faces=faces,
        source_indices=source_indices,
        source_positions=source_positions,
        source_gaussian_indices=source_gaussian_indices,
        gaussian_positions=gaussian_positions,
        gaussian_to_mesh_indices=closest_mesh_indices,
        gaussian_to_mesh_distances=closest_mesh_distances,
        geodesic_method=args.geodesic_method,
        n_jobs=args.n_jobs,
        partial_save_dir=str(output_folder / "geodesic_distance" / "mesh_batch_cache"),
        output_dir=str(output_folder / "geodesic_distance" / "gt_partial"),
        gaussian_vertex_indices=gaussian_vertex_indices,
        verbose=args.verbose,
    )

    save_computation_metadata(
        output_folder=output_folder,
        args=args,
        num_gaussians=n_gauss,
        num_sources=len(all_source_indices),
    )

    print(f"\n  Results saved to {output_folder / 'geodesic_distance' / 'gt_partial'}")
    print(f"  (Gaussian-mesh mode: zero interpolation error)")


def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# Gaussian Splat Geodesic Distance Computation")
    print(f"{'#'*80}")
    
    output_folder = Path(args.gaussian_output)
    
    # Handle merge-only mode
    if args.merge_only:
        merge_partial_results(output_folder, verbose=args.verbose)
        return
    
    
    # Load Gaussian splat (CPU-compatible version)
    gaussian_data = load_gaussian_data_cpu(output_folder, args.iteration)
    
    gaussian_positions = gaussian_data.get_xyz()
    gaussian_scales = gaussian_data.get_scaling()
    gaussian_rotations = gaussian_data.get_rotation()
    gaussian_opacities = gaussian_data.get_opacity()

    # ── Gaussian-mesh path (no interpolation) ────────────────────────────
    if args.use_gaussian_mesh:
        _run_gaussian_mesh_path(args, output_folder, gaussian_positions)
        return

    # ── Standard path (GT mesh + barycentric interpolation) ──────────────
    data_root = Path(args.data_root)
    
    # Step 2: Load ground truth mesh
    mesh_vertices, mesh_faces = load_ground_truth_mesh(data_root, args.surface, args.mesh_level)
    
    # Step 3: Compute mesh range from ground truth vertices
    x_min, x_max = mesh_vertices[:, 0].min(), mesh_vertices[:, 0].max()
    y_min, y_max = mesh_vertices[:, 1].min(), mesh_vertices[:, 1].max()
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    
    print(f"\n  Ground truth mesh range:")
    print(f"    X: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
    print(f"    Y: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
    
    # Step 4: Generate source mesh, map vertices to nearest Gaussians, then get their
    # closest mesh vertex indices.  This ensures source_indices[i] is the mesh vertex
    # of Gaussian source_gaussian_indices[i] — guaranteeing geodesic distance ≈ 0 for
    # each source/Gaussian pair.
    all_source_gaussian_indices, all_source_positions = generate_source_mesh_and_map(
        surface_type=args.surface,
        source_mesh_resolution=args.source_mesh_resolution,
        gaussian_positions=gaussian_positions,
        x_range=x_range,
        y_range=y_range,
    )
    # Get the GT mesh vertex closest to each source Gaussian
    all_source_indices, _ = find_closest_mesh_vertices(
        gaussian_centers=gaussian_positions[all_source_gaussian_indices],
        mesh_vertices=mesh_vertices,
    )
    
    # Determine source range for this run
    if args.source_start is not None and args.source_end is not None:
        source_start = args.source_start
        source_end = min(args.source_end, len(all_source_indices))
        source_indices = all_source_indices[source_start:source_end]
        source_positions_subset = all_source_positions[source_start:source_end]
    else:
        source_start = 0
        source_end = len(all_source_indices) 
        source_indices = all_source_indices
        source_positions_subset = all_source_positions

    # Step 5: Find closest mesh vertices for Gaussians
    # (must run BEFORE source→Gaussian mapping so we can use the mesh footprint)
    gaussian_to_mesh_indices, gaussian_to_mesh_distances = find_closest_mesh_vertices(
        gaussian_centers=gaussian_positions,
        mesh_vertices=mesh_vertices,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=gaussian_scales if args.use_mahalanobis else None,
        gaussian_rotations=gaussian_rotations if args.use_mahalanobis else None
    )

    # Step 5b: For each Gaussian find its closest face and barycentric coordinates.
    # This allows interpolating geodesic distances at arbitrary Gaussian positions
    # rather than snapping to the nearest vertex (O(h²) vs O(h) error).
    barycentric_face_vertices, barycentric_weights = find_closest_mesh_faces_barycentric(
        gaussian_centers=gaussian_positions,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        closest_vertex_indices=gaussian_to_mesh_indices,
    )

    # Recover source_gaussian_indices for the final source set using the lookup
    # built from all_source_indices → all_source_gaussian_indices.
    _src_mesh_to_gauss = dict(zip(all_source_indices.tolist(),
                                  all_source_gaussian_indices.tolist()))
    source_gaussian_indices = np.array(
        [_src_mesh_to_gauss[idx] for idx in source_indices.tolist()],
        dtype=np.intp
    )

    # Step 6–8: Compute geodesics, transfer to Gaussians, and save —
    # all done in parallel batches via the unified pipeline.
    compute_and_save_geodesic_pipeline(
        vertices=mesh_vertices,
        faces=mesh_faces,
        source_indices=source_indices,
        source_positions=source_positions_subset,
        source_gaussian_indices=source_gaussian_indices,
        gaussian_positions=gaussian_positions,
        gaussian_to_mesh_indices=gaussian_to_mesh_indices,
        gaussian_to_mesh_distances=gaussian_to_mesh_distances,
        geodesic_method=args.geodesic_method,
        n_jobs=args.n_jobs,
        partial_save_dir=str(output_folder / "geodesic_distance" / "mesh_batch_cache"),
        output_dir=str(output_folder / "geodesic_distance" / "gt_partial"),
        barycentric_face_vertices=barycentric_face_vertices,
        barycentric_weights=barycentric_weights,
        verbose=args.verbose,
    )
    
    # Save computation metadata
    save_computation_metadata(
        output_folder=output_folder,
        args=args,
        num_gaussians=len(gaussian_positions),
        num_sources=len(all_source_indices)
    )
    
    print(f"\n{'#'*80}")
    print(f"# Computation Complete")
    print(f"{'#'*80}")
    print(f"\nPartial results saved to:")
    print(f"  {output_folder / 'geodesic_distance' / 'gt_partial'}")
    print(f"\nTo merge all partial results, run:")
    print(f"  python {Path(__file__).name} --gaussian_output {args.gaussian_output} --merge_only")


if __name__ == "__main__":
    main()
