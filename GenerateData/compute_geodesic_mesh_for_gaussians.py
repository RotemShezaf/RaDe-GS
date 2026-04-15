#!/usr/bin/env python3
"""
Build a geodesic-quality mesh whose vertices include every Gaussian splat.

Instead of the ``compute_gaussian_geodesic_distances.py`` approach—which
loads a separate high-resolution ground-truth mesh, then transfers
geodesic distances to Gaussians via barycentric interpolation—this script
places the Gaussians **directly on the mesh** by projecting their
``(x, y)`` positions onto the analytical polynomial surface.

This means:
  * Each Gaussian IS a mesh vertex → zero transfer error.
  * Geodesic distances between Gaussians can be read straight off the
    mesh (no interpolation, no snapping).

The optional *uniform surface sampling* fills in empty regions of the
domain so that the Delaunay triangulation has good aspect ratios
everywhere, not just where Gaussians happen to exist.

WORKFLOW
--------
1. Load Gaussian positions from the output folder.
2. Project every Gaussian onto the polynomial surface → mesh vertices
   0 … N_gauss - 1.
3. (Optional) Sample additional surface points via Poisson-disk /
   min-radius → mesh vertices N_gauss … N_gauss + K - 1.
4. Delaunay-triangulate the combined point set in the ``(x, y)``
   parameter plane and lift to 3-D.
5. Save the mesh (``geodesic_mesh.ply``) and the Gaussian-to-vertex
   mapping (``geodesic_mesh_data.npz``) under
   ``<gaussian_output>/geodesic_mesh/``.

OUTPUT
------
::

    <gaussian_output>/
        geodesic_mesh/
            geodesic_mesh.ply             # Trimesh PLY
            geodesic_mesh_data.npz        # vertices, faces, gaussian_vertex_indices, metadata

USAGE
-----
::

    # Minimal invocation (no extra samples):
    python compute_geodesic_mesh_for_gaussians.py \\
        --gaussian_output output/polynomial/Paraboloid \\
        --surface Paraboloid

    # Add ~500 extra samples via Poisson-disk:
    python compute_geodesic_mesh_for_gaussians.py \\
        --gaussian_output output/polynomial/Paraboloid \\
        --surface Paraboloid \\
        --n_points 500

    # Or use a minimum spacing instead:
    python compute_geodesic_mesh_for_gaussians.py \\
        --gaussian_output output/polynomial/Paraboloid \\
        --surface Paraboloid \\
        --min_radius 0.05 \\
        --x_range -1.0 1.0 --y_range -1.0 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.load_utils import load_gaussian_data_cpu
from GenerateData.utils.geodesic_mesh_utils import (
    project_gaussians_to_surface,
    compute_surface_curvature,
    sample_surface_uniform,
    build_surface_delaunay,
    build_surface_delaunay_3d,
    build_surface_ball_pivoting,
    build_surface_grid,
    insert_gaussians_into_grid_mesh,
    insert_steiner_points,
    fix_orphaned_gaussians,
    repair_orphan_gaussians,
    fix_face_winding,
    prepare_mesh_for_vtp,
    refine_bad_triangles,
    save_geodesic_mesh,
    _filter_boundary_long_edges,
    _count_mesh_holes,
    _detect_non_manifold_edges,
    _detect_duplicate_faces,
    fix_non_manifold_mesh,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mesh on a polynomial surface whose vertices "
            "include every Gaussian splat."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--gaussian_output",
        type=str,
        required=True,
        help="Path to Gaussian splatting output folder.",
    )
    parser.add_argument(
        "--surface",
        type=str,
        required=True,
        choices=["Paraboloid", "Saddle", "HyperbolicParaboloid"],
        help="Polynomial surface type.",
    )

    # Iteration
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Training iteration to load (default: highest available).",
    )

    # Uniform sampling (mutually exclusive)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--n_points",
        type=int,
        default=None,
        help="Number of additional uniform surface points to add.",
    )
    sampling.add_argument(
        "--min_radius",
        type=float,
        default=None,
        help="Minimum (x, y) spacing between any two mesh vertices.",
    )

    # Domain bounds (used for uniform sampling; auto-detected from Gaussians if omitted)
    parser.add_argument(
        "--x_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Domain bounds in x (default: detected from Gaussians ± 5%%).",
    )
    parser.add_argument(
        "--y_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Domain bounds in y (default: detected from Gaussians ± 5%%).",
    )

    parser.add_argument(
        "--max_edge_length",
        type=float,
        default=None,
        help=(
            "Remove triangles whose longest edge exceeds this value. "
            "Default: auto (10× median edge length). Use 0 to disable."
        ),
    )

    # Mesh method
    parser.add_argument(
        "--mesh_method",
        type=str,
        default="delaunay",
        choices=["delaunay", "delaunay_3d", "ball_pivoting", "grid"],
        help=(
            "Meshing algorithm. 'delaunay' uses 2-D Delaunay in (x,y) "
            "parameter space. 'delaunay_3d' performs 3-D Delaunay "
            "tetrahedralization (GPU via tetranerf if available, else "
            "scipy CPU) and extracts surface faces. 'ball_pivoting' "
            "uses Open3D Ball Pivoting on the 3-D point cloud. 'grid' "
            "builds a structured arc-length adapted grid, then inserts "
            "Gaussians by triangle splitting with Lawson edge-flipping. "
            "Default: delaunay."
        ),
    )

    # Local per-triangle refinement (grid method only)
    parser.add_argument(
        "--local_refinement",
        action="store_true",
        help=(
            "Use local per-triangle Gaussian insertion instead of global "
            "Delaunay when mesh_method='grid'.  Keeps grid topology intact "
            "and sub-triangulates only within each grid face."
        ),
    )

    # Steiner-point insertion
    parser.add_argument(
        "--steiner",
        action="store_true",
        help=(
            "Insert Steiner points (circumcentres of bad triangles) "
            "before the refinement pass.  Points are projected onto "
            "the polynomial surface."
        ),
    )
    parser.add_argument(
        "--steiner_max_aspect_ratio",
        type=float,
        default=3.0,
        help="Aspect-ratio threshold for Steiner insertion (default: 3.0).",
    )
    parser.add_argument(
        "--steiner_min_angle",
        type=float,
        default=20.0,
        help="Min-angle threshold (deg) for Steiner insertion (default: 20.0).",
    )
    parser.add_argument(
        "--steiner_iterations",
        type=int,
        default=3,
        help="Max Steiner insertion rounds (default: 3).",
    )

    # Curvature-adaptive sampling
    parser.add_argument(
        "--curvature_adaptive",
        action="store_true",
        help=(
            "Enable curvature-aware sampling: denser in high-curvature "
            "regions, sparser in flat regions.  Only effective when "
            "--n_points or --min_radius is given."
        ),
    )
    parser.add_argument(
        "--curvature_alpha",
        type=float,
        default=2.0,
        help=(
            "Strength of curvature adaptation (default: 2.0). "
            "Local radius = r_base / sqrt(1 + alpha * |K| / K_ref). "
            "Higher values concentrate more points near curvature."
        ),
    )
    parser.add_argument(
        "--gaussian_density_alpha",
        type=float,
        default=0.0,
        help=(
            "Strength of Gaussian-density grid adaptation (default: 0 = off). "
            "When > 0, the grid is denser near Gaussian clusters and "
            "coarser where Gaussians are sparse. Values 2-5 typical."
        ),
    )

    # Bad-triangle refinement
    parser.add_argument(
        "--refine",
        action="store_true",
        help=(
            "Enable iterative refinement of bad triangles. "
            "Inserts edge-midpoints on the surface and locally "
            "retriangulates until quality thresholds are met."
        ),
    )
    parser.add_argument(
        "--refine_max_aspect_ratio",
        type=float,
        default=1e6,
        help="Aspect-ratio threshold for bad triangles (default: 1e6 = disabled).",
    )
    parser.add_argument(
        "--refine_min_angle",
        type=float,
        default=20.0,
        help="Min-angle threshold (degrees) for bad triangles (default: 20.0).",
    )
    parser.add_argument(
        "--refine_max_area_factor",
        type=float,
        default=3.0,
        help="Area-factor threshold (multiple of median area) for non-Gaussian triangles (default: 3.0).",
    )
    parser.add_argument(
        "--refine_gauss_max_aspect_ratio",
        type=float,
        default=1e6,
        help=(
            "AR threshold for triangles touching a Gaussian vertex. "
            "Default: 1e6 (disabled — angle threshold is sufficient)."
        ),
    )
    parser.add_argument(
        "--refine_gauss_min_angle",
        type=float,
        default=30.0,
        help=(
            "Min-angle threshold for Gaussian-touching triangles. "
            "Default: 30.0° (matches general threshold for balanced quality)."
        ),
    )
    parser.add_argument(
        "--refine_gauss_max_edge_length",
        type=float,
        default=None,
        help=(
            "Max longest-edge threshold applied as a quality criterion for "
            "triangles touching a Gaussian vertex.  Triangles whose longest "
            "edge exceeds this value are treated as bad and refined. "
            "Default: None (no edge-length criterion for Gaussian triangles)."
        ),
    )
    parser.add_argument(
        "--refine_gauss_max_area_factor",
        type=float,
        default=None,
        help=(
            "Area-factor threshold (multiple of median area) for Gaussian-touching "
            "triangles.  Default: None (uses --refine_max_area_factor)."
        ),
    )
    parser.add_argument(
        "--refine_iterations",
        type=int,
        default=3,
        help="Maximum refinement passes (default: 1 — ring+flip is sufficient).",
    )
    parser.add_argument(
        "--refine_warmup_iterations",
        type=int,
        default=10,
        help=(
            "Number of initial refinement iterations in which longest-edge "
            "splitting is applied only to Gaussian-touching bad triangles. "
            "Non-Gaussian bad triangles receive smoothing only during warmup. "
            "0 = no warmup (default)."
        ),
    )

    parser.add_argument(
        "--refine_ring_fix",
        action="store_true",
        default=False,
        help=(
            "If set, run ring-based support-point insertion around every "
            "Gaussian in a bad triangle before the main refinement loop "
            "(Stage 0). Off by default."
        ),
    )
    parser.add_argument(
        "--refine_ring_fix_iterations",
        type=int,
        default=1,
        help=(
            "Number of ring insertion passes. Each pass halves the ring "
            "radius and targets Gaussians still in bad triangles. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--refine_surface_aware",
        action="store_true",
        default=False,
        help=(
            "If set, use the first fundamental form of the polynomial "
            "surface for metric-weighted Laplacian smoothing and geodesic "
            "midpoint splitting instead of flat parameter-space operations. "
            "Off by default."
        ),
    )

    parser.add_argument(
        "--refine_patience",
        type=int,
        default=3,
        help=(
            "Number of consecutive refinement iterations without >=5%% "
            "improvement before early stopping. Default: 3."
        ),
    )

    parser.add_argument(
        "--refine_max_splits",
        type=int,
        default=30000,
        help=(
            "Maximum number of edge midpoints to insert per refinement "
            "iteration. Worst triangles are prioritised. Default: 10000."
        ),
    )

    parser.add_argument(
        "--refine_smoothing_passes",
        type=int,
        default=60,
        help=(
            "Number of Laplacian smoothing passes per refinement iteration. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--refine_delaunay_flip",
        action="store_true",
        default=False,
        help=(
            "If set, apply Delaunay edge-flipping as a post-refinement "
            "quality polish pass. Improves angles without adding vertices."
        ),
    )

    # Extreme triangle cleanup
    parser.add_argument(
        "--refine_extreme_cleanup_passes",
        type=int,
        default=50,
        help=(
            "Number of extreme-triangle cleanup passes after the main "
            "refinement loop. Collapses extreme-AR slivers, ring-fixes "
            "Gaussian-touching extremes, splits remaining. 0 = off. Default: 3."
        ),
    )
    parser.add_argument(
        "--refine_extreme_ar_threshold",
        type=float,
        default= 1e6,
        help=(
            "Aspect-ratio threshold for extreme slivers. Triangles with "
            "AR above this get edge-collapsed (non-Gaussian) or ring-fixed "
            "(Gaussian-touching). Default: 20.0."
        ),
    )
    parser.add_argument(
        "--refine_extreme_min_angle",
        type=float,
        default=10.0,
        help=(
            "Min-angle threshold (degrees) for extreme triangles. "
            "Triangles with min angle below this are targeted by the "
            "extreme cleanup pass. Default: 9.0."
        ),
    )
    parser.add_argument(
        "--refine_extreme_max_splits",
        type=int,
        default=5000,
        help=(
            "Maximum number of midpoint insertions per extreme-cleanup pass. "
            "Worst (longest-edge) triangles are prioritised. "
            "0 = unlimited (default)."
        ),
    )
    parser.add_argument(
        "--refine_extreme_smoothing_passes",
        type=int,
        default=5,
        help=(
            "Number of Laplacian smoothing passes for non-Gaussian vertices "
            "of extreme triangles at the end of each extreme cleanup pass. "
            "0 = disabled (default)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info.",
    )

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    output_folder = Path(args.gaussian_output)

    print(f"\n{'#' * 72}")
    print(f"# Build Geodesic Mesh for Gaussians")
    print(f"{'#' * 72}")

    # 1. Load Gaussians -------------------------------------------------
    print(f"\n  [1/6] Loading Gaussians from {output_folder} ...")
    gaussian_data = load_gaussian_data_cpu(output_folder, args.iteration)
    positions = gaussian_data.get_xyz()  # (N, 3)
    n_gauss = len(positions)
    print(f"        → {n_gauss} Gaussians loaded")

    # 2. Project onto surface -------------------------------------------
    print(f"\n  [2/6] Projecting onto '{args.surface}' surface ...")
    projected = project_gaussians_to_surface(positions, args.surface)
    max_dz = np.abs(projected[:, 2] - positions[:, 2]).max()
    print(f"        → max |Δz| = {max_dz:.6f}")

    # 3. (Optional) uniform surface sampling ----------------------------
    # Determine domain bounds (needed for grid or sampling)
    margin = 0.05  # 5 % margin
    if args.x_range is not None:
        x_range = tuple(args.x_range)
    else:
        dx = np.ptp(projected[:, 0])
        x_range = (projected[:, 0].min() - margin * dx,
                   projected[:, 0].max() + margin * dx)

    if args.y_range is not None:
        y_range = tuple(args.y_range)
    else:
        dy = np.ptp(projected[:, 1])
        y_range = (projected[:, 1].min() - margin * dy,
                   projected[:, 1].max() + margin * dy)

    extra_points = np.empty((0, 3), dtype=np.float64)

    if args.mesh_method == "grid":
        # Grid method: the grid provides the background vertices,
        # no separate Poisson-disk sampling is needed.
        print(f"\n  [3/6] Building arc-length adapted grid "
              f"(curvature_adaptive={args.curvature_adaptive}) ...")
        grid_edge = (
            args.max_edge_length
            if args.max_edge_length and args.max_edge_length > 0
            else 0.05
        )
        grid_verts, grid_faces = build_surface_grid(
            surface_type=args.surface,
            x_range=x_range,
            y_range=y_range,
            target_edge_length=grid_edge,
            curvature_adaptive=args.curvature_adaptive,
            curvature_alpha=args.curvature_alpha,
            gaussian_xy=projected[:, :2] if args.gaussian_density_alpha > 0 else None,
            gaussian_density_alpha=args.gaussian_density_alpha,
        )
        mode_str = "curvature-adaptive" if args.curvature_adaptive else "arc-length uniform"
        print(f"        → grid: {len(grid_verts)} vertices, "
              f"{len(grid_faces)} faces ({mode_str})")
        print(f"        → domain x={x_range}, y={y_range}")

        # 4. Insert Gaussians into the grid mesh
        print(f"\n  [4/6] Inserting {n_gauss} Gaussians into grid mesh ...")
        vertices, faces, gaussian_vertex_indices = insert_gaussians_into_grid_mesh(
            grid_verts, grid_faces, projected, args.surface,
            local_refinement=args.local_refinement,
            verbose=args.verbose,
        )
        print(f"        → {len(vertices)} vertices, {len(faces)} faces "
              f"after insertion + edge-flipping")

        # max_edge_threshold for later filtering / refinement
        max_edge_threshold = grid_edge * 3.0

    elif args.n_points is not None or args.min_radius is not None:
        print(f"\n  [3/6] Sampling additional surface points ...")

        extra_points = sample_surface_uniform(
            surface_type=args.surface,
            x_range=x_range,
            y_range=y_range,
            n_points=args.n_points,
            min_radius=args.min_radius,
            existing_xy=projected[:, :2],
            seed=args.seed,
            curvature_adaptive=args.curvature_adaptive,
            curvature_alpha=args.curvature_alpha,
        )
        mode_str = "curvature-adaptive" if args.curvature_adaptive else "uniform"
        print(f"        → {len(extra_points)} extra points sampled ({mode_str})")
        if args.curvature_adaptive:
            print(f"        → curvature alpha = {args.curvature_alpha}")
        print(f"        → domain x={x_range}, y={y_range}")
    else:
        print(f"\n  [3/6] No extra sampling requested — skipped")

    # 4. Mesh construction (for delaunay / ball_pivoting) ──────────────
    if args.mesh_method != "grid":
        print(f"\n  [4/6] Building mesh (method={args.mesh_method}) ...")
        # Gaussian vertices first, then extras
        all_vertices = (
            np.vstack([projected, extra_points])
            if len(extra_points) > 0
            else projected
        )
        if args.mesh_method == "ball_pivoting":
            vertices, faces = build_surface_ball_pivoting(
                all_vertices, surface_type=args.surface,
            )
            print(f"        → {len(vertices)} vertices, {len(faces)} faces (Ball Pivoting)")
        elif args.mesh_method == "delaunay_3d":
            vertices, faces = build_surface_delaunay_3d(all_vertices, surface_type=args.surface)
            print(f"        → {len(vertices)} vertices, {len(faces)} faces (3-D Delaunay)")
        else:
            vertices, faces = build_surface_delaunay(all_vertices, surface_type=args.surface)
            print(f"        → {len(vertices)} vertices, {len(faces)} faces (raw Delaunay)")
        # For delaunay/ball_pivoting, Gaussians are the first n_gauss vertices
        gaussian_vertex_indices = np.arange(n_gauss, dtype=np.int32)

    # Filter out triangles with excessively long edges (convex-hull artifacts)
    if args.mesh_method != "grid":
        # For grid, max_edge_threshold is already set above
        v0_raw = vertices[faces[:, 0]]
        v1_raw = vertices[faces[:, 1]]
        v2_raw = vertices[faces[:, 2]]
        edge_lens = np.column_stack([
            np.linalg.norm(v1_raw - v0_raw, axis=1),
            np.linalg.norm(v2_raw - v1_raw, axis=1),
            np.linalg.norm(v0_raw - v2_raw, axis=1),
        ])
        longest_edge = edge_lens.max(axis=1)

        if args.max_edge_length == 0:
            max_edge_threshold = None
        elif args.max_edge_length is not None:
            max_edge_threshold = args.max_edge_length
        else:
            max_edge_threshold = float(10.0 * np.median(edge_lens))

    n_removed = 0
    if max_edge_threshold is not None:
        is_3d = (args.mesh_method == "delaunay_3d")
        faces, n_removed = _filter_boundary_long_edges(
            vertices, faces, max_edge_threshold,
            surface_type=args.surface if is_3d else None,
            is_3d=is_3d,
        )
        if n_removed > 0:
            print(f"        → removed {n_removed} convex-hull artifact faces with edge > {max_edge_threshold:.6f}")
            print(f"        → {len(faces)} faces after filtering")
        else:
            print(f"        → no convex-hull artifacts exceed max_edge_length={max_edge_threshold:.6f}")

    # Fix orphaned Gaussian vertices (only for delaunay/ball_pivoting
    # where Gaussians are at indices 0..n_gauss-1; for grid, Gaussians
    # were inserted explicitly and are always connected).
    if args.mesh_method != "grid" and max_edge_threshold is not None and max_edge_threshold > 0:
        vertices, faces = fix_orphaned_gaussians(
            vertices, faces, n_gauss,
            max_edge_length=max_edge_threshold,
            surface_type=args.surface,
            verbose=True,
        )

    # Compact: remove orphan vertices and update indices
    used_verts = np.unique(faces.ravel())
    if len(used_verts) < len(vertices):
        n_orphan = len(vertices) - len(used_verts)
        new_idx = np.full(len(vertices), -1, dtype=np.int32)
        new_idx[used_verts] = np.arange(len(used_verts), dtype=np.int32)
        vertices = vertices[used_verts]
        faces = new_idx[faces]
        # Update Gaussian vertex indices
        gaussian_vertex_indices = new_idx[gaussian_vertex_indices]
        valid_gauss = gaussian_vertex_indices >= 0
        if not valid_gauss.all():
            n_lost = int((~valid_gauss).sum())
            print(f"        ⚠ {n_lost} Gaussian vertices still orphaned after repair — dropped")
            gaussian_vertex_indices = gaussian_vertex_indices[valid_gauss]
        if n_orphan > 0 and args.verbose:
            print(f"        → removed {n_orphan} orphan vertices")
            print(f"        → {len(vertices)} vertices after cleanup")

    # 4a. (Optional) Steiner-point insertion ─────────────────────────────
    if args.steiner:
        print(f"\n  [4a/6] Inserting Steiner points ...")
        print(f"         thresholds: AR > {args.steiner_max_aspect_ratio}, "
              f"angle < {args.steiner_min_angle}°, "
              f"max {args.steiner_iterations} rounds")
        n_verts_before = len(vertices)
        vertices, faces, gaussian_vertex_indices = insert_steiner_points(
            vertices,
            faces,
            surface_type=args.surface,
            max_aspect_ratio=args.steiner_max_aspect_ratio,
            min_angle_deg=args.steiner_min_angle,
            max_edge_length=max_edge_threshold,
            max_iterations=args.steiner_iterations,
            gaussian_vertex_indices=gaussian_vertex_indices,
            use_3d_delaunay=(args.mesh_method == "delaunay_3d"),
            verbose=True,
        )
        n_steiner = len(vertices) - n_verts_before
        print(f"         → {n_steiner} Steiner points added")
        print(f"         → {len(vertices)} vertices, {len(faces)} faces")
    else:
        print(f"\n  [4a/6] Steiner insertion not requested — skipped")

    # 4b. (Optional) Refine bad triangles ──────────────────────────────
    if args.refine:
        gauss_ar = args.refine_gauss_max_aspect_ratio
        gauss_ma = args.refine_gauss_min_angle
        print(f"\n  [4b/6] Refining bad triangles ...")
        print(f"         thresholds: AR > {args.refine_max_aspect_ratio}, "
              f"angle < {args.refine_min_angle}°, "
              f"area_factor={args.refine_max_area_factor}, "
              f"max {args.refine_iterations} iterations, "
              f"warmup={args.refine_warmup_iterations} iters")
        if gauss_ar is not None or gauss_ma is not None \
                or args.refine_gauss_max_edge_length is not None \
                or args.refine_gauss_max_area_factor is not None:
            print(f"         Gaussian-touching: AR > {gauss_ar}, "
                  f"angle < {gauss_ma}°, "
                  f"max_edge_length={args.refine_gauss_max_edge_length}, "
                  f"area_factor={args.refine_gauss_max_area_factor}")
        n_verts_before = len(vertices)
        n_faces_before = len(faces)
        vertices, faces, gaussian_vertex_indices = refine_bad_triangles(
            vertices,
            faces,
            surface_type=args.surface,
            max_aspect_ratio=args.refine_max_aspect_ratio,
            min_angle_deg=args.refine_min_angle,
            max_area_factor=args.refine_max_area_factor,
            gauss_max_aspect_ratio=gauss_ar,
            gauss_min_angle_deg=gauss_ma,
            gauss_max_edge_length=args.refine_gauss_max_edge_length,
            gauss_max_area_factor=args.refine_gauss_max_area_factor,
            warmup_iterations=args.refine_warmup_iterations,
            ring_fix_invalid_gaussians=args.refine_ring_fix,
            ring_fix_iterations=args.refine_ring_fix_iterations,
            surface_aware=args.refine_surface_aware,
            patience=args.refine_patience,
            max_iterations=args.refine_iterations,
            max_edge_length=max_edge_threshold,
            gaussian_vertex_indices=gaussian_vertex_indices,
            use_3d_delaunay=(args.mesh_method == "delaunay_3d"),
            delaunay_flip_polish=args.refine_delaunay_flip,
            extreme_cleanup_passes=args.refine_extreme_cleanup_passes,
            extreme_ar_threshold=args.refine_extreme_ar_threshold,
            extreme_min_angle_deg=args.refine_extreme_min_angle,
            extreme_max_splits_per_pass=args.refine_extreme_max_splits,
            extreme_smoothing_passes=args.refine_extreme_smoothing_passes,
            verbose=True,
        )
        n_delta_verts = len(vertices) - n_verts_before
        n_delta_faces = len(faces) - n_faces_before
        print(f"         → net vertex change: {n_delta_verts:+d}, "
              f"net face change: {n_delta_faces:+d}")
        print(f"         → {len(vertices)} vertices, {len(faces)} faces, "
              f"{len(gaussian_vertex_indices)} Gaussian verts retained")
        # gaussian_vertex_indices returned by refine_bad_triangles reflects
        # any vertex removals; do NOT recompute from n_gauss here.
    else:
        print(f"\n  [4b/6] Refinement not requested — skipped")

    # ── Remove valence-1 non-Gaussian vertices (boundary ear triangles) ──
    gauss_idx_set = set(gaussian_vertex_indices.tolist())
    _val = np.zeros(len(vertices), dtype=np.int32)
    for _c in range(3):
        np.add.at(_val, faces[:, _c], 1)
    v1_mask = _val == 1
    v1_nongauss = np.array([i for i in np.where(v1_mask)[0] if i not in gauss_idx_set])
    if len(v1_nongauss) > 0:
        # Remove faces that contain any valence-1 non-Gaussian vertex
        v1_set = set(v1_nongauss.tolist())
        keep = np.array(
            [not any(int(v) in v1_set for v in f) for f in faces], dtype=bool
        )
        n_removed_v1 = int((~keep).sum())
        faces = faces[keep]
        # Compact unreferenced vertices
        used_v = np.unique(faces.ravel())
        if len(used_v) < len(vertices):
            remap = np.full(len(vertices), -1, dtype=np.int32)
            remap[used_v] = np.arange(len(used_v), dtype=np.int32)
            vertices = vertices[used_v]
            faces = remap[faces]
            gaussian_vertex_indices = remap[gaussian_vertex_indices]
            assert (gaussian_vertex_indices >= 0).all(), "valence-1 cleanup removed a Gaussian"
        print(f"        → removed {n_removed_v1} boundary ear faces "
              f"({len(v1_nongauss)} valence-1 non-Gaussian vertices)")

    # ── Topology cleanup: duplicate faces / non-manifold edges ───────
    n_dup_before, _ = _detect_duplicate_faces(faces)
    n_nm_before, _ = _detect_non_manifold_edges(faces)
    if n_dup_before > 0 or n_nm_before > 0:
        print(f"\n  [4c/6] Fixing mesh topology "
              f"({n_dup_before} duplicate faces, "
              f"{n_nm_before} non-manifold edges) ...")
        faces, vertices = fix_non_manifold_mesh(
            faces, gaussian_vertex_indices, verbose=True,
            vertices=vertices, surface_type=args.surface,
        )
    else:
        print(f"\n  [4c/6] Mesh topology clean — no duplicates or non-manifold edges")

    # ── Final repair: fix any Gaussian vertices left with valence 0 ──
    faces = repair_orphan_gaussians(
        vertices, faces, gaussian_vertex_indices, verbose=True,
    )

    # ── Prepare mesh for VTP geodesic (connectivity + winding) ──
    print(f"\n  [4d/6] Preparing mesh for VTP geodesic ...")
    faces = prepare_mesh_for_vtp(
        vertices, faces, gaussian_vertex_indices, verbose=True,
    )

    # ── Compact unreferenced vertices after VTP preparation ──
    used_verts = np.unique(faces.ravel())
    if len(used_verts) < len(vertices):
        n_orphan = len(vertices) - len(used_verts)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)
        vertices = vertices[used_verts]
        faces = remap[faces]
        gaussian_vertex_indices = remap[gaussian_vertex_indices]
        assert (gaussian_vertex_indices >= 0).all(), \
            "VTP preparation orphaned a Gaussian vertex"
        if args.verbose:
            print(f"        → compacted {n_orphan} unreferenced vertices "
                  f"after VTP preparation")

    # ── Mesh quality statistics ────────────────────────────────────────
    print(f"\n  [5/6] Computing mesh quality statistics ...")
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    shortest = np.minimum(np.minimum(e0, e1), e2)
    longest = np.maximum(np.maximum(e0, e1), e2)
    aspect_ratio = longest / np.maximum(shortest, 1e-15)

    cos_A = np.clip((e1**2 + e2**2 - e0**2) / (2 * e1 * e2 + 1e-30), -1, 1)
    cos_B = np.clip((e0**2 + e2**2 - e1**2) / (2 * e0 * e2 + 1e-30), -1, 1)
    cos_C = np.clip((e0**2 + e1**2 - e2**2) / (2 * e0 * e1 + 1e-30), -1, 1)
    min_angle = np.minimum(
        np.minimum(np.degrees(np.arccos(cos_A)), np.degrees(np.arccos(cos_B))),
        np.degrees(np.arccos(cos_C)),
    )

    all_edges = np.concatenate([e0, e1, e2])
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Radius-ratio quality metric: Q = 2 * inradius / circumradius
    # Q = 1 for equilateral, 0 for degenerate
    perimeter = e0 + e1 + e2
    inradius = 2.0 * areas / np.maximum(perimeter, 1e-30)
    circumradius = (e0 * e1 * e2) / np.maximum(4.0 * areas, 1e-30)
    radius_ratio = 2.0 * inradius / np.maximum(circumradius, 1e-30)
    radius_ratio = np.clip(radius_ratio, 0.0, 1.0)

    # Surface curvature at vertices
    K_vert, H_vert = compute_surface_curvature(
        vertices[:, 0], vertices[:, 1], args.surface,
    )

    # Vertex valence (number of incident faces per vertex)
    valence = np.zeros(len(vertices), dtype=np.int32)
    for col in range(3):
        np.add.at(valence, faces[:, col], 1)

    # Hole count
    n_holes = _count_mesh_holes(faces)

    # Non-manifold / duplicate detection (post-fix — should be 0)
    n_nm_edges, nm_edge_details = _detect_non_manifold_edges(faces)
    n_dup_faces, _ = _detect_duplicate_faces(faces)

    # Area outlier detection (watered-down regions)
    median_area = float(np.median(areas))
    n_area_10x = int((areas > 10.0 * median_area).sum()) if median_area > 0 else 0
    n_area_50x = int((areas > 50.0 * median_area).sum()) if median_area > 0 else 0
    pct_area_10x = 100.0 * n_area_10x / len(faces) if len(faces) > 0 else 0.0

    # Connected component detection (watered regions)
    from collections import defaultdict, deque
    vert_adj: dict = defaultdict(set)
    for fi in range(len(faces)):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        vert_adj[a].update([b, c])
        vert_adj[b].update([a, c])
        vert_adj[c].update([a, b])
    visited_cc: set = set()
    n_components = 0
    component_sizes = []
    for node in vert_adj:
        if node not in visited_cc:
            n_components += 1
            size = 0
            queue = deque([node])
            while queue:
                vv = queue.popleft()
                if vv in visited_cc:
                    continue
                visited_cc.add(vv)
                size += 1
                for nb in vert_adj[vv]:
                    if nb not in visited_cc:
                        queue.append(nb)
            component_sizes.append(size)

    # Projected-to-mesh-vertex distance (Gaussians only)
    gauss_mesh_verts = vertices[gaussian_vertex_indices]
    projected_to_mesh_dist = np.linalg.norm(
        projected - gauss_mesh_verts, axis=1,
    )

    # Gaussians to projected Gaussians distance
    gauss_to_proj_dist = np.linalg.norm(
        projected - positions, axis=1,
    )

    # Check ALL mesh vertices lie on the surface (not just Gaussians)
    from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial
    all_xy = vertices[:, :2]
    all_z_surface = evaluate_polynomial(all_xy[:, 0], all_xy[:, 1], args.surface)
    all_z_residual = np.abs(vertices[:, 2] - all_z_surface)
    vertex_proj_mismatch_mask = all_z_residual > 1e-6
    n_vertex_proj_mismatch = int(vertex_proj_mismatch_mask.sum())

    # Gaussian-specific z residual (for backward compat)
    gauss_xy = gauss_mesh_verts[:, :2]
    gauss_z_surface = evaluate_polynomial(gauss_xy[:, 0], gauss_xy[:, 1], args.surface)
    gauss_z_residual = np.abs(gauss_mesh_verts[:, 2] - gauss_z_surface)

    # Per-triangle classification: touches Gaussian vertex?
    gauss_set = set(gaussian_vertex_indices.tolist())
    gauss_touch = np.array([
        any(int(v) in gauss_set for v in f) for f in faces
    ], dtype=bool)
    gt_only_mask = ~gauss_touch  # triangles with only extra-sample vertices

    def _scalar_stats(arr: np.ndarray, name: str = "") -> dict:
        """Min / median / mean / max summary for a 1-D array."""
        if len(arr) == 0:
            return {}
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }

    def _bucket_stats(
        ar: np.ndarray,
        ma: np.ndarray,
        ed: np.ndarray | None = None,
        qr: np.ndarray | None = None,
    ):
        """Return a dict of summary statistics for a subset of triangles."""
        if len(ar) == 0:
            return {}
        stats: dict = {
            "count": int(len(ar)),
            "aspect_ratio": {
                "min": float(np.min(ar)),
                "median": float(np.median(ar)),
                "mean": float(np.mean(ar)),
                "max": float(np.max(ar)),
                "pct_lt_2": float(100 * np.mean(ar < 2)),
                "pct_lt_3": float(100 * np.mean(ar < 3)),
                "pct_lt_5": float(100 * np.mean(ar < 5)),
            },
            "min_angle_deg": {
                "min": float(np.min(ma)),
                "median": float(np.median(ma)),
                "mean": float(np.mean(ma)),
                "max": float(np.max(ma)),
                "pct_gt_30": float(100 * np.mean(ma > 30)),
                "pct_gt_20": float(100 * np.mean(ma > 20)),
                "pct_gt_10": float(100 * np.mean(ma > 10)),
                "pct_lt_5": float(100 * np.mean(ma < 5)),
            },
        }
        if ed is not None:
            stats["edge_length"] = _scalar_stats(ed)
        if qr is not None:
            stats["radius_ratio"] = {
                **_scalar_stats(qr),
                "pct_gt_0.5": float(100 * np.mean(qr > 0.5)),
                "pct_gt_0.3": float(100 * np.mean(qr > 0.3)),
                "pct_gt_0.1": float(100 * np.mean(qr > 0.1)),
            }
        return stats

    # Edges per category
    gauss_edges = np.concatenate([
        e0[gauss_touch], e1[gauss_touch], e2[gauss_touch],
    ]) if gauss_touch.any() else np.array([])
    extra_edges = np.concatenate([
        e0[gt_only_mask], e1[gt_only_mask], e2[gt_only_mask],
    ]) if gt_only_mask.any() else np.array([])

    mesh_stats: dict = {
        "surface": args.surface,
        "n_gaussians": int(len(gaussian_vertex_indices)),
        "n_extra_samples": int(len(extra_points)),
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "seed": args.seed,
        "sampling": {
            "curvature_adaptive": args.curvature_adaptive,
            "curvature_alpha": args.curvature_alpha,
        },
        "edge_filter": {
            "max_edge_length": float(max_edge_threshold) if max_edge_threshold is not None else None,
            "faces_removed": n_removed,
        },
        "vertex_bounds": {
            "x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
            "y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
            "z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())],
        },
        "curvature": {
            "gaussian_K": _scalar_stats(np.abs(K_vert[gaussian_vertex_indices])),
            "gaussian_H": _scalar_stats(np.abs(H_vert[gaussian_vertex_indices])),
            "all_K": _scalar_stats(np.abs(K_vert)),
            "all_H": _scalar_stats(np.abs(H_vert)),
        },
        "area": {
            "total": float(areas.sum()),
            **_scalar_stats(areas),
        },
        "overall": _bucket_stats(aspect_ratio, min_angle, all_edges, radius_ratio),
        "gaussian_triangles": _bucket_stats(
            aspect_ratio[gauss_touch], min_angle[gauss_touch],
            gauss_edges,
            radius_ratio[gauss_touch] if gauss_touch.any() else np.array([]),
        ),
        "extra_sample_triangles": _bucket_stats(
            aspect_ratio[gt_only_mask], min_angle[gt_only_mask],
            extra_edges,
            radius_ratio[gt_only_mask] if gt_only_mask.any() else np.array([]),
        ),
        "mesh_quality": {
            "radius_ratio": {
                **_scalar_stats(radius_ratio),
                "pct_gt_0.5": float(100 * np.mean(radius_ratio > 0.5)),
                "pct_gt_0.3": float(100 * np.mean(radius_ratio > 0.3)),
                "pct_gt_0.1": float(100 * np.mean(radius_ratio > 0.1)),
            },
            "degenerate_triangles": {
                "min_angle_lt_1deg": int(np.sum(min_angle < 1)),
                "min_angle_lt_5deg": int(np.sum(min_angle < 5)),
                "min_angle_lt_10deg": int(np.sum(min_angle < 10)),
            },
            "valence": {
                **_scalar_stats(valence.astype(float)),
                "n_valence_1": int(np.sum(valence == 1)),
                "n_valence_2": int(np.sum(valence == 2)),
                "n_valence_3": int(np.sum(valence == 3)),
            },
            "holes": n_holes,
            "non_manifold_edges": n_nm_edges,
            "duplicate_faces": n_dup_faces,
            "connected_components": n_components,
            "component_sizes": sorted(component_sizes, reverse=True)[:5],
            "area_outliers": {
                "median_area": float(median_area),
                "n_gt_10x_median": n_area_10x,
                "n_gt_50x_median": n_area_50x,
                "pct_gt_10x_median": round(pct_area_10x, 3),
            },
        },
        "projected_to_mesh_distance": {
            **_scalar_stats(projected_to_mesh_dist),
            "pct_exact_match": float(
                100.0 * (projected_to_mesh_dist < 1e-10).sum()
                / max(len(projected_to_mesh_dist), 1)
            ),
            "n_mismatch_gt_1e6": int((projected_to_mesh_dist > 1e-6).sum()),
        },
        "gauss_to_proj_distance": {
            **_scalar_stats(gauss_to_proj_dist),
        },
        "vertex_projection": {
            "z_residual": _scalar_stats(gauss_z_residual),
            "all_vertex_z_residual": _scalar_stats(all_z_residual),
            "n_vertex_proj_mismatch": n_vertex_proj_mismatch,
            "mismatch_max_z_residual": float(
                all_z_residual[vertex_proj_mismatch_mask].max()
                if n_vertex_proj_mismatch > 0
                else 0.0
            ),
        },
    }

    # Print summary
    s = mesh_stats["overall"]
    g = mesh_stats.get("gaussian_triangles", {})
    mq = mesh_stats["mesh_quality"]
    kc = mesh_stats["curvature"]
    degen = mq["degenerate_triangles"]
    val = mq["valence"]
    ao = mq["area_outliers"]

    print(f"        ── Overall ({mesh_stats['n_vertices']:,} verts, "
          f"{mesh_stats['n_faces']:,} faces, "
          f"{mesh_stats['n_gaussians']:,} Gaussians) ──")
    print(f"        Aspect ratio : median={s['aspect_ratio']['median']:.3f}, "
          f"max={s['aspect_ratio']['max']:.1f}, "
          f"<2: {s['aspect_ratio']['pct_lt_2']:.1f}%, "
          f"<3: {s['aspect_ratio']['pct_lt_3']:.1f}%, "
          f"<5: {s['aspect_ratio']['pct_lt_5']:.1f}%")
    print(f"        Min angle    : median={s['min_angle_deg']['median']:.2f}°, "
          f"min={s['min_angle_deg']['min']:.2f}°, "
          f">30°: {s['min_angle_deg']['pct_gt_30']:.1f}%, "
          f">20°: {s['min_angle_deg']['pct_gt_20']:.1f}%, "
          f"<5°: {s['min_angle_deg']['pct_lt_5']:.2f}%")
    print(f"        Edge lengths : min={s['edge_length']['min']:.6f}, "
          f"median={s['edge_length']['median']:.6f}, "
          f"max={s['edge_length']['max']:.6f}")
    print(f"        Radius ratio : median={s['radius_ratio']['median']:.4f}, "
          f"min={s['radius_ratio']['min']:.4f}, "
          f">0.5: {s['radius_ratio']['pct_gt_0.5']:.1f}%, "
          f">0.3: {s['radius_ratio']['pct_gt_0.3']:.1f}%")
    if g:
        print(f"        ── Gaussian-touching triangles ({g['count']:,}) ──")
        print(f"        Gauss AR     : median={g['aspect_ratio']['median']:.3f}, "
              f"max={g['aspect_ratio']['max']:.1f}, "
              f"<2: {g['aspect_ratio']['pct_lt_2']:.1f}%")
        print(f"        Gauss angle  : median={g['min_angle_deg']['median']:.2f}°, "
              f"min={g['min_angle_deg']['min']:.2f}°, "
              f">20°: {g['min_angle_deg']['pct_gt_20']:.1f}%")
    print(f"        ── Quality checks ──")
    print(f"        Holes        : {mq['holes']}")
    print(f"        Non-manifold : {mq['non_manifold_edges']} edges, "
          f"{mq['duplicate_faces']} duplicate faces")
    print(f"        Components   : {mq['connected_components']}")
    print(f"        Degenerate   : <1°: {degen['min_angle_lt_1deg']}, "
          f"<5°: {degen['min_angle_lt_5deg']}, "
          f"<10°: {degen['min_angle_lt_10deg']}")
    print(f"        Valence      : min={val['min']:.0f}, median={val['median']:.1f}, "
          f"max={val['max']:.0f}, "
          f"v1={val['n_valence_1']}, v2={val['n_valence_2']}, v3={val['n_valence_3']}")
    print(f"        Area outliers: >10× median: {ao['n_gt_10x_median']} ({ao['pct_gt_10x_median']:.2f}%), "
          f">50×: {ao['n_gt_50x_median']}")
    print(f"        |K| (Gauss)  : median={kc['gaussian_K']['median']:.6f}, "
          f"max={kc['gaussian_K']['max']:.6f}")

    pmd = mesh_stats["projected_to_mesh_distance"]
    vp = mesh_stats["vertex_projection"]
    g2p = mesh_stats["gauss_to_proj_distance"]
    print(f"        ── Distance statistics ──")
    print(f"        Proj→mesh    : mean={pmd['mean']:.2e}, max={pmd['max']:.2e}, "
          f"exact: {pmd['pct_exact_match']:.1f}%")
    print(f"        Gauss→proj   : mean={g2p['mean']:.2e}, max={g2p['max']:.2e}")
    print(f"        Z residual   : mean={vp['z_residual']['mean']:.2e}, "
          f"max={vp['z_residual']['max']:.2e}")
    if vp['n_vertex_proj_mismatch'] > 0:
        print(f"        ⚠ {vp['n_vertex_proj_mismatch']} vertices off-surface "
              f"(max z residual={vp['mismatch_max_z_residual']:.2e})")

    # 6. Save -----------------------------------------------------------
    mesh_dir = output_folder / "geodesic_mesh"
    print(f"\n  [6/6] Saving mesh to {mesh_dir} ...")

    metadata = dict(
        surface=np.array(args.surface),
        n_gaussians=np.array(n_gauss),
        n_extra_samples=np.array(len(extra_points)),
        seed=np.array(args.seed),
    )
    npz_path = save_geodesic_mesh(
        output_dir=mesh_dir,
        vertices=vertices,
        faces=faces,
        gaussian_vertex_indices=gaussian_vertex_indices,
        metadata=metadata,
    )

    # Save statistics JSON
    stats_path = mesh_dir / "mesh_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(mesh_stats, f, indent=2)

    elapsed = time.time() - t_start
    mesh_stats["elapsed_seconds"] = round(elapsed, 2)
    with open(stats_path, "w") as f:
        json.dump(mesh_stats, f, indent=2)

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Mesh PLY  : {mesh_dir / 'geodesic_mesh.ply'}")
    print(f"  Data NPZ  : {npz_path}")
    print(f"  Stats JSON: {stats_path}")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    main()
