#!/usr/bin/env python3
"""
Test geodesic mesh quality after building.

Checks three aspects for every built mesh:
  1. Mesh triangle quality: aspect ratio, min angle, % bad triangles
  2. Gaussian-to-closest-mesh-vertex distance (should be zero for snapped)
  3. Projected Gaussian (on polynom) vs. mesh vertex distance (should be ~0)

Usage:
    python scripts/polynomial/geodesic/test_mesh_quality.py [--base_dir DIR] [--surfaces S1,S2,...] [--verbose]

    By default scans all surfaces × levels × lights under
    TrainData/Polynomial/SyntheticColmapData/blue_texture/.

Example:
    # Test all meshes
    python scripts/polynomial/geodesic/test_mesh_quality.py --verbose

    # Test only Paraboloid
    python scripts/polynomial/geodesic/test_mesh_quality.py --surfaces Paraboloid --verbose
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
    }

    if verbose:
        print(f"    Vertices: {n_verts:,}  Faces: {n_faces:,}  Gaussians: {n_gauss:,}")
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
    args = parser.parse_args()

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
                    if result["zero_angle_faces"] > 0:
                        issues.append(f"{result['zero_angle_faces']} zero-angle faces")
                    if result["max_gauss_to_mesh_dist"] > 1e-6:
                        issues.append(f"max gauss→mesh dist={result['max_gauss_to_mesh_dist']:.2e}")
                    if result["max_gauss_z_residual"] > 1e-6:
                        issues.append(f"max gauss z residual={result['max_gauss_z_residual']:.2e}")
                    if result["pct_bad_triangles"] > 10.0:
                        issues.append(f"bad triangles={result['pct_bad_triangles']:.1f}%")

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

        print(f"  Bad triangles:   min={min(pct_bads):.1f}%  max={max(pct_bads):.1f}%  "
              f"mean={np.mean(pct_bads):.1f}%")
        print(f"  Gauss→mesh dist: max across all={max(gauss_dists):.2e}")
        print(f"  Gauss z residual: max across all={max(z_residuals):.2e}")
        print(f"  Zero-angle faces: total={sum(zero_angles)}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
