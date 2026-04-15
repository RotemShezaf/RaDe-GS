#!/usr/bin/env python3
"""
Test geodesic mesh quality after building.

Checks five aspects for every built mesh:
  1. Mesh triangle quality: aspect ratio, min angle, % bad triangles
  2. Gaussian-to-closest-mesh-vertex distance (should be zero for snapped)
  3. Projected Gaussian (on polynom) vs. mesh vertex distance (should be ~0)
  4. Hole detection: interior boundary loops (should be 0)
  5. Watered-down detection: large-area triangle outliers that indicate
     sparse / under-sampled regions

Usage:
    python scripts/polynomial/geodesic/test_mesh_quality.py [--base_dir DIR] [--surfaces S1,S2,...] [--verbose]
    python scripts/polynomial/geodesic/test_mesh_quality.py --single_mesh <path/to/geodesic_mesh_data.npz> --surface Saddle [--verbose]

    By default scans all surfaces × levels × lights under
    TrainData/Polynomial/SyntheticColmapData/blue_texture/.
    When --single_mesh is given the result is also written as
    quality_report.json next to the .npz file.

Example:
    # Test all meshes
    python scripts/polynomial/geodesic/test_mesh_quality.py --verbose

    # Test only Paraboloid
    python scripts/polynomial/geodesic/test_mesh_quality.py --surfaces Paraboloid --verbose

    # Test one specific mesh
    python scripts/polynomial/geodesic/test_mesh_quality.py \
        --single_mesh TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_04/light_0/output/geodesic_mesh/geodesic_mesh_data.npz \
        --surface Saddle --verbose
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial
from GenerateData.utils.geodesic_mesh_utils import (
    _count_mesh_holes,
    _triangle_quality,
    project_gaussians_to_surface,
)


def _load_gaussian_positions(output_folder: Path) -> np.ndarray:
    """Load Gaussian xyz from the highest-iteration PLY (no open3d needed).

    Returns float32 to match load_gaussian_data_cpu (which the mesh builder uses).
    """
    from plyfile import PlyData

    pc_dir = output_folder / "point_cloud"
    iters = sorted(
        int(p.name.split("_")[1])
        for p in pc_dir.iterdir()
        if p.is_dir() and p.name.startswith("iteration_")
    )
    if not iters:
        raise FileNotFoundError(f"No iterations in {pc_dir}")
    ply_path = pc_dir / f"iteration_{iters[-1]}" / "point_cloud.ply"
    vertex = PlyData.read(str(ply_path)).elements[0]
    return np.column_stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ]).astype(np.float32)


def analyse_mesh(
    mesh_npz_path: Path,
    gaussian_output: Path,
    surface_type: str,
    verbose: bool = False,
) -> dict:
    """Analyse a single geodesic mesh.

    Returns a dict with quality metrics.
    """
    data = np.load(mesh_npz_path)
    vertices = data["vertices"]       # (V, 3)
    faces = data["faces"]             # (F, 3)
    gauss_idx = data["gaussian_vertex_indices"]  # (G,)

    n_verts = len(vertices)
    n_faces = len(faces)
    n_gauss = len(gauss_idx)

    # ── 0. Hole detection ────────────────────────────────────────────
    n_holes = _count_mesh_holes(faces)

    # ── 1. Triangle quality ─────────────────────────────────────────
    ar, min_angle, longest_edge = _triangle_quality(vertices, faces)
    bad_ar = ar > 2.0
    bad_angle = min_angle < 20.0
    bad = bad_ar | bad_angle
    pct_bad = 100.0 * bad.sum() / n_faces
    pct_angle_lt20 = 100.0 * (min_angle < 20.0).sum() / n_faces
    pct_angle_lt5 = 100.0 * (min_angle < 5.0).sum() / n_faces
    worst_ar = float(ar.max())
    zero_angle = float((min_angle < 0.01).sum())

    # Gaussian-touching triangle quality
    touches_gauss_mask = np.zeros(len(vertices), dtype=bool)
    touches_gauss_mask[gauss_idx] = True
    gauss_face_mask = (
        touches_gauss_mask[faces[:, 0]]
        | touches_gauss_mask[faces[:, 1]]
        | touches_gauss_mask[faces[:, 2]]
    )
    n_gauss_faces = int(gauss_face_mask.sum())
    if n_gauss_faces > 0:
        gauss_bad = bad[gauss_face_mask]
        pct_gauss_bad = 100.0 * gauss_bad.sum() / n_gauss_faces
        gauss_min_angle = min_angle[gauss_face_mask]
        gauss_ar = ar[gauss_face_mask]
    else:
        pct_gauss_bad = 0.0
        gauss_min_angle = np.array([])
        gauss_ar = np.array([])

    # ── 2. Gaussian-to-closest-mesh-vertex distance ─────────────────
    #    Gaussians were projected onto the surface before insertion.
    #    Load the raw Gaussian positions and project them.
    gdata = _load_gaussian_positions(gaussian_output)
    raw_positions = gdata  # (N, 3) original Gaussian positions
    projected = project_gaussians_to_surface(raw_positions, surface_type)

    # Mesh vertices at Gaussian indices
    mesh_gauss_verts = vertices[gauss_idx]  # (G, 3)

    # Distance from projected Gaussian to its mesh vertex
    gauss_to_mesh = np.linalg.norm(projected - mesh_gauss_verts, axis=1)
    mean_gauss_to_mesh = float(gauss_to_mesh.mean())
    max_gauss_to_mesh = float(gauss_to_mesh.max())
    pct_exact = 100.0 * (gauss_to_mesh < 1e-10).sum() / len(gauss_to_mesh)

    # ── 2b. Watered-down detection ───────────────────────────────────
    #    A "watered-down" mesh has isolated large triangles that cover
    #    sparse regions.  We flag them as area outliers relative to the
    #    median triangle area.
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)          # (F,)
    median_area = float(np.median(tri_areas))
    area_max = float(tri_areas.max())
    area_max_median_ratio = area_max / median_area if median_area > 0 else 0.0
    area_p99_median_ratio = float(
        np.percentile(tri_areas, 99) / median_area if median_area > 0 else 0.0
    )
    n_large_10x = int((tri_areas > 10.0 * median_area).sum())
    n_large_50x = int((tri_areas > 50.0 * median_area).sum())
    pct_large_10x = 100.0 * n_large_10x / n_faces

    # All edge lengths
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    all_edge_lens = np.concatenate([e0, e1, e2])
    median_edge = float(np.median(all_edge_lens))
    edge_max = float(all_edge_lens.max())
    edge_max_median_ratio = edge_max / median_edge if median_edge > 0 else 0.0

    # ── 3. Mesh vertex z vs. polynomial z (projection residual) ─────
    #    Every mesh vertex should lie exactly on the polynomial surface.
    #    Check that z_mesh == evaluate_polynomial(x_mesh, y_mesh).
    x_mesh = vertices[:, 0]
    y_mesh = vertices[:, 1]
    z_mesh = vertices[:, 2]
    z_surface = evaluate_polynomial(x_mesh, y_mesh, surface_type)
    z_residual = np.abs(z_mesh - z_surface)
    mean_z_residual = float(z_residual.mean())
    max_z_residual = float(z_residual.max())

    # Specifically for Gaussian mesh vertices
    z_gauss_residual = np.abs(
        mesh_gauss_verts[:, 2]
        - evaluate_polynomial(mesh_gauss_verts[:, 0], mesh_gauss_verts[:, 1], surface_type)
    )
    mean_gauss_z_residual = float(z_gauss_residual.mean())
    max_gauss_z_residual = float(z_gauss_residual.max())

    result = {
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "n_gaussians": n_gauss,
        # Holes
        "n_holes": n_holes,
        # Quality
        "pct_bad_triangles": round(pct_bad, 2),
        "pct_angle_lt20": round(pct_angle_lt20, 2),
        "pct_angle_lt5": round(pct_angle_lt5, 2),
        "zero_angle_faces": int(zero_angle),
        "worst_aspect_ratio": round(worst_ar, 2),
        "median_aspect_ratio": round(float(np.median(ar)), 3),
        "p95_aspect_ratio": round(float(np.percentile(ar, 95)), 3),
        "median_min_angle": round(float(np.median(min_angle)), 2),
        # Gaussian triangle quality
        "n_gauss_faces": n_gauss_faces,
        "pct_gauss_bad": round(pct_gauss_bad, 2),
        # Gaussian distance
        "mean_gauss_to_mesh_dist": mean_gauss_to_mesh,
        "max_gauss_to_mesh_dist": max_gauss_to_mesh,
        "pct_gauss_exact_match": round(pct_exact, 2),
        # Surface residual (all vertices)
        "mean_z_residual": mean_z_residual,
        "max_z_residual": max_z_residual,
        # Surface residual (Gaussian vertices only)
        "mean_gauss_z_residual": mean_gauss_z_residual,
        "max_gauss_z_residual": max_gauss_z_residual,
        # Watered-down / sparse-region metrics
        "median_triangle_area": round(median_area, 10),
        "area_max_median_ratio": round(area_max_median_ratio, 1),
        "area_p99_median_ratio": round(area_p99_median_ratio, 2),
        "n_large_area_10x": n_large_10x,
        "n_large_area_50x": n_large_50x,
        "pct_large_area_10x": round(pct_large_10x, 3),
        "median_edge_length": round(median_edge, 6),
        "max_edge_length": round(edge_max, 6),
        "edge_max_median_ratio": round(edge_max_median_ratio, 1),
    }

    if verbose:
        print(f"    Vertices: {n_verts:,}  Faces: {n_faces:,}  Gaussians: {n_gauss:,}")
        print(f"    [Holes]    {n_holes} interior hole(s)")
        print(f"    [Quality]  bad: {pct_bad:.1f}%  angle<20°: {pct_angle_lt20:.1f}%  "
              f"angle<5°: {pct_angle_lt5:.1f}%  zero-angle: {int(zero_angle)}")
        print(f"               worst AR: {worst_ar:.1f}  median AR: {np.median(ar):.3f}  "
              f"p95 AR: {np.percentile(ar, 95):.3f}")
        print(f"    [Gauss Q]  {n_gauss_faces} Gaussian faces, {pct_gauss_bad:.1f}% bad")
        print(f"    [Gauss→Mesh] mean dist: {mean_gauss_to_mesh:.2e}  "
              f"max dist: {max_gauss_to_mesh:.2e}  exact match: {pct_exact:.1f}%")
        print(f"    [Z resid]  all verts: mean={mean_z_residual:.2e} max={max_z_residual:.2e}")
        print(f"               Gauss verts: mean={mean_gauss_z_residual:.2e} "
              f"max={max_gauss_z_residual:.2e}")
        print(f"    [Coverage] area max/median={area_max_median_ratio:.1f}x  "
              f"p99/median={area_p99_median_ratio:.2f}x  "
              f">10x median: {n_large_10x} ({pct_large_10x:.2f}%)  "
              f">50x median: {n_large_50x}")
        print(f"               edge max/median={edge_max_median_ratio:.1f}x  "
              f"max edge={edge_max:.4f}  median edge={median_edge:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test geodesic mesh quality")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="TrainData/Polynomial/SyntheticColmapData/blue_texture",
        help="Base directory containing surface folders",
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        default="Paraboloid,Saddle,HyperbolicParaboloid",
        help="Comma-separated surface types to test",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="02,03,04",
        help="Comma-separated level IDs to test",
    )
    parser.add_argument(
        "--lights",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated light IDs to test",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Save results to JSON file (default: print summary only)",
    )
    # ── Single-mesh shortcut ─────────────────────────────────────────────
    parser.add_argument(
        "--single_mesh",
        type=str,
        default=None,
        help=(
            "Path to a geodesic_mesh_data.npz (or the geodesic_mesh/ directory). "
            "When provided, only that mesh is tested and the result is written as "
            "quality_report.json next to the .npz file. --surface is required."
        ),
    )
    parser.add_argument(
        "--surface",
        type=str,
        default=None,
        help="Surface type for --single_mesh mode (e.g. Saddle, Paraboloid).",
    )
    args = parser.parse_args()

    # ── Single-mesh mode ─────────────────────────────────────────────────
    if args.single_mesh is not None:
        if args.surface is None:
            parser.error("--surface is required when --single_mesh is given")
        mesh_npz = Path(args.single_mesh)
        if mesh_npz.is_dir():
            mesh_npz = mesh_npz / "geodesic_mesh_data.npz"
        if not mesh_npz.exists():
            print(f"ERROR: mesh file not found: {mesh_npz}")
            sys.exit(1)
        # gaussian output folder is two levels up from the .npz: geodesic_mesh/ → output/
        gaussian_output = mesh_npz.parent.parent
        label = str(mesh_npz.parent)
        print(f"  TEST  {label}")
        result = analyse_mesh(mesh_npz, gaussian_output, args.surface, verbose=args.verbose)
        result["label"] = label
        result["surface"] = args.surface

        issues = []
        if result["n_holes"] > 0:
            issues.append(f"{result['n_holes']} interior hole(s)")
        if result["zero_angle_faces"] > 0:
            issues.append(f"{result['zero_angle_faces']} zero-angle faces")
        if result["max_gauss_to_mesh_dist"] > 1e-6:
            issues.append(f"max gauss→mesh dist={result['max_gauss_to_mesh_dist']:.2e}")
        if result["max_gauss_z_residual"] > 1e-6:
            issues.append(f"max gauss z residual={result['max_gauss_z_residual']:.2e}")
        if result["pct_bad_triangles"] > 10.0:
            issues.append(f"bad triangles={result['pct_bad_triangles']:.1f}%")
        if result["pct_large_area_10x"] > 2.0:
            issues.append(
                f"watered-down: {result['pct_large_area_10x']:.2f}% triangles "
                f">10x median area ({result['n_large_area_10x']} faces)"
            )

        if issues:
            print(f"  FAIL  {label}: {'; '.join(issues)}")
            result["status"] = "FAIL"
            result["issues"] = issues
        else:
            print(f"  PASS  {label}")
            result["status"] = "PASS"

        # Always auto-save next to the .npz
        out_path = mesh_npz.parent / "quality_report.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {out_path}")
        sys.exit(1 if issues else 0)

    base = PROJECT_ROOT / args.base_dir
    surfaces = args.surfaces.split(",")
    levels = args.levels.split(",")
    lights = args.lights.split(",")

    all_results: list[dict] = []
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for surface in surfaces:
        for level in levels:
            for light in lights:
                gaussian_output = (
                    base / surface / f"level_{level}" / f"light_{light}" / "output"
                )
                mesh_npz = gaussian_output / "geodesic_mesh" / "geodesic_mesh_data.npz"

                label = f"{surface}/level_{level}/light_{light}"
                if not mesh_npz.exists():
                    if args.verbose:
                        print(f"  SKIP  {label} — no mesh found")
                    n_skip += 1
                    continue

                print(f"  TEST  {label}")
                try:
                    result = analyse_mesh(
                        mesh_npz, gaussian_output, surface, verbose=args.verbose,
                    )
                    result["label"] = label
                    result["surface"] = surface
                    result["level"] = level
                    result["light"] = light

                    # PASS/FAIL criteria
                    issues = []
                    if result["n_holes"] > 0:
                        issues.append(f"{result['n_holes']} interior hole(s)")
                    if result["zero_angle_faces"] > 0:
                        issues.append(f"{result['zero_angle_faces']} zero-angle faces")
                    if result["max_gauss_to_mesh_dist"] > 1e-6:
                        issues.append(f"max gauss→mesh dist={result['max_gauss_to_mesh_dist']:.2e}")
                    if result["max_gauss_z_residual"] > 1e-6:
                        issues.append(f"max gauss z residual={result['max_gauss_z_residual']:.2e}")
                    if result["pct_bad_triangles"] > 10.0:
                        issues.append(f"bad triangles={result['pct_bad_triangles']:.1f}%")
                    if result["pct_large_area_10x"] > 2.0:
                        issues.append(
                            f"watered-down: {result['pct_large_area_10x']:.2f}% triangles "
                            f">10x median area ({result['n_large_area_10x']} faces)"
                        )

                    if issues:
                        print(f"  FAIL  {label}: {'; '.join(issues)}")
                        result["status"] = "FAIL"
                        result["issues"] = issues
                        n_fail += 1
                    else:
                        print(f"  PASS  {label}")
                        result["status"] = "PASS"
                        n_pass += 1

                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR {label}: {e}")
                    n_fail += 1

    # ── Summary ─────────────────────────────────────────────────────
    total = n_pass + n_fail + n_skip
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP (total {total})")
    print(f"{'='*60}")

    if all_results:
        pct_bads = [r["pct_bad_triangles"] for r in all_results]
        gauss_dists = [r["max_gauss_to_mesh_dist"] for r in all_results]
        z_residuals = [r["max_gauss_z_residual"] for r in all_results]
        zero_angles = [r["zero_angle_faces"] for r in all_results]
        total_holes = [r["n_holes"] for r in all_results]
        pct_large = [r["pct_large_area_10x"] for r in all_results]

        print(f"  Bad triangles:   min={min(pct_bads):.1f}%  max={max(pct_bads):.1f}%  "
              f"mean={np.mean(pct_bads):.1f}%")
        print(f"  Gauss→mesh dist: max across all={max(gauss_dists):.2e}")
        print(f"  Gauss z residual: max across all={max(z_residuals):.2e}")
        print(f"  Zero-angle faces: total={sum(zero_angles)}")
        print(f"  Holes:           total={sum(total_holes)} (max in one mesh={max(total_holes)})")
        print(f"  Watered-down:    max >10x area fraction={max(pct_large):.2f}%  "
              f"mean={np.mean(pct_large):.2f}%")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
