"""Evaluate Gaussian Splatting quality and Gaussian-to-mesh distance statistics.

Usage:
    python scripts/evaluate_gaussian_mesh_quality.py \
        --output_dir TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output

Metrics reported:
    1. Gaussian center -> nearest reconstructed mesh surface distance statistics
    2. Mesh vertex -> nearest Gaussian center distance (coverage)
    3. Threshold analysis at multiple distance levels
    4. Optional: Chamfer distance against a ground-truth mesh

A JSON report (quality_report.json) is written to the output directory.
"""

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from plyfile import PlyData


def load_gaussian_centers(ply_path: Path) -> np.ndarray:
    """Load Gaussian centers (xyz) from a trained point_cloud.ply."""
    plydata = PlyData.read(str(ply_path))
    vertex = plydata['vertex']
    return np.stack([np.array(vertex['x']),
                     np.array(vertex['y']),
                     np.array(vertex['z'])], axis=-1)


def load_mesh(ply_path: Path):
    """Load mesh from a .ply file, return trimesh object."""
    mesh = trimesh.load(str(ply_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def dist_stats(distances: np.ndarray) -> dict:
    """Return a compact statistics dict for a distance array."""
    return {
        "count": int(len(distances)),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "std": float(distances.std()),
        "min": float(distances.min()),
        "max": float(distances.max()),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
    }


def print_dist_stats(label: str, stats: dict):
    print(f"\n  {label}:")
    print(f"    Count:   {stats['count']}")
    print(f"    Mean:    {stats['mean']:.6f}")
    print(f"    Median:  {stats['median']:.6f}")
    print(f"    Std:     {stats['std']:.6f}")
    print(f"    Min:     {stats['min']:.6f}")
    print(f"    Max:     {stats['max']:.6f}")
    for pct in [90, 95, 99]:
        print(f"    P{pct}:     {stats[f'p{pct}']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian-mesh quality")
    parser.add_argument("--output_dir", type=str, default="/home/rotem.shezaf/RaDe-GS/TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_0/output",
                        help="Path to training output directory")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Iteration of point cloud to evaluate")
    parser.add_argument("--gt_mesh", type=str, default=None,
                        help="Path to ground-truth mesh for Chamfer distance evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    recon_ply = output_dir / "recon.ply"
    gauss_ply = output_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"

    # Validate files exist
    if not recon_ply.exists():
        print(f"ERROR: Reconstructed mesh not found at {recon_ply}")
        return
    if not gauss_ply.exists():
        print(f"ERROR: Point cloud not found at {gauss_ply}")
        return

    # ── Report dict that will be saved to JSON ──
    report: dict = {"iteration": args.iteration, "output_dir": str(output_dir)}

    print("=" * 70)
    print(f"Evaluating: {output_dir}")
    print(f"Iteration: {args.iteration}")
    print("=" * 70)

    # Load data
    print("\nLoading Gaussian centers...")
    gaussians = load_gaussian_centers(gauss_ply)
    print(f"  Gaussian count: {len(gaussians)}")

    print("Loading reconstructed mesh...")
    mesh = load_mesh(recon_ply)
    recon_verts = np.asarray(mesh.vertices)
    print(f"  Mesh vertices: {len(recon_verts)}")
    print(f"  Mesh faces: {len(mesh.faces)}")

    report["gaussian_count"] = int(len(gaussians))
    report["mesh_vertices"] = int(len(recon_verts))
    report["mesh_faces"] = int(len(mesh.faces))

    # Bounding box info
    gauss_bb = gaussians.max(axis=0) - gaussians.min(axis=0)
    mesh_bb = recon_verts.max(axis=0) - recon_verts.min(axis=0)
    gauss_diag = float(np.linalg.norm(gauss_bb))
    mesh_diag = float(np.linalg.norm(mesh_bb))
    print(f"\n  Gaussian bounding box diagonal: {gauss_diag:.4f}")
    print(f"  Mesh bounding box diagonal:     {mesh_diag:.4f}")

    report["gaussian_bbox_diagonal"] = gauss_diag
    report["mesh_bbox_diagonal"] = mesh_diag

    # === Gaussian -> Nearest Mesh Vertex ===
    print("\n" + "=" * 70)
    print("GAUSSIAN -> MESH DISTANCE")
    print("=" * 70)
    tree_mesh = cKDTree(recon_verts)
    g2m_dists, _ = tree_mesh.query(gaussians)
    g2m_stats = dist_stats(g2m_dists)
    print_dist_stats("Gaussian center -> nearest mesh vertex", g2m_stats)
    report["gaussian_to_mesh_vertex"] = g2m_stats

    # Point-to-surface distance
    print("\n  Computing point-to-surface (closest point on mesh face)...")
    closest_pts, g2s_dists, face_ids = trimesh.proximity.closest_point(mesh, gaussians)
    g2s_stats = dist_stats(g2s_dists)
    print_dist_stats("Gaussian center -> nearest mesh surface point", g2s_stats)
    report["gaussian_to_mesh_surface"] = g2s_stats

    # Threshold analysis
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
    threshold_report = {}
    print(f"\n  Gaussians within distance threshold of mesh surface:")
    for thresh in thresholds:
        abs_count = int((g2s_dists < thresh).sum())
        pct = abs_count / len(g2s_dists) * 100
        rel = thresh / mesh_diag * 100
        print(f"    < {thresh:.3f} ({rel:.2f}% of bbox diag): {pct:.1f}% ({abs_count}/{len(g2s_dists)})")
        threshold_report[str(thresh)] = {"count": abs_count, "percent": round(pct, 2)}
    report["gaussian_to_surface_thresholds"] = threshold_report

    # === Mesh Vertex -> Nearest Gaussian (coverage) ===
    print("\n" + "=" * 70)
    print("MESH -> GAUSSIAN DISTANCE (coverage)")
    print("=" * 70)
    tree_gauss = cKDTree(gaussians)
    m2g_dists, _ = tree_gauss.query(recon_verts)
    m2g_stats = dist_stats(m2g_dists)
    print_dist_stats("Mesh vertex -> nearest Gaussian center", m2g_stats)
    report["mesh_to_gaussian"] = m2g_stats

    # === GT Mesh Comparison (if provided) ===
    if args.gt_mesh:
        gt_mesh_path = Path(args.gt_mesh)
        if gt_mesh_path.exists():
            print("\n" + "=" * 70)
            print("RECONSTRUCTED MESH vs GROUND-TRUTH MESH")
            print("=" * 70)
            gt_mesh = load_mesh(gt_mesh_path)
            gt_verts = np.asarray(gt_mesh.vertices)
            gt_diag = float(np.linalg.norm(gt_verts.max(axis=0) - gt_verts.min(axis=0)))
            print(f"  GT mesh vertices: {len(gt_verts)}, faces: {len(gt_mesh.faces)}")
            print(f"  GT mesh bounding box diagonal: {gt_diag:.4f}")

            gt_section: dict = {
                "gt_mesh_path": str(gt_mesh_path),
                "gt_vertices": int(len(gt_verts)),
                "gt_faces": int(len(gt_mesh.faces)),
                "gt_bbox_diagonal": gt_diag,
            }

            # Recon -> GT (accuracy)
            _, recon2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, recon_verts)
            r2gt = dist_stats(recon2gt_dists)
            print_dist_stats("Recon vertex -> GT surface (accuracy)", r2gt)
            gt_section["recon_to_gt_accuracy"] = r2gt

            # GT -> Recon (completeness)
            _, gt2recon_dists, _ = trimesh.proximity.closest_point(mesh, gt_verts)
            gt2r = dist_stats(gt2recon_dists)
            print_dist_stats("GT vertex -> Recon surface (completeness)", gt2r)
            gt_section["gt_to_recon_completeness"] = gt2r

            # Chamfer distance
            chamfer = float((recon2gt_dists.mean() + gt2recon_dists.mean()) / 2)
            print(f"\n  Chamfer distance: {chamfer:.6f} ({chamfer / gt_diag * 100:.3f}% of GT bbox diagonal)")
            gt_section["chamfer_distance"] = chamfer
            gt_section["chamfer_pct_bbox"] = round(chamfer / gt_diag * 100, 4)

            # Gaussian -> GT mesh
            _, g2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, gaussians)
            g2gt = dist_stats(g2gt_dists)
            print_dist_stats("Gaussian center -> GT surface", g2gt)
            gt_section["gaussian_to_gt_surface"] = g2gt

            gt_thresh = {}
            for thresh in [0.01, 0.05, 0.1, 0.5, 1.0]:
                pct = float((g2gt_dists < thresh).sum() / len(g2gt_dists) * 100)
                print(f"  Gaussians within {thresh} of GT surface: {pct:.1f}%")
                gt_thresh[str(thresh)] = round(pct, 2)
            gt_section["gaussian_to_gt_thresholds_pct"] = gt_thresh

            report["gt_mesh_comparison"] = gt_section
        else:
            print(f"\n  WARNING: GT mesh not found at {gt_mesh_path}")

    # === Summary / Quality Assessment ===
    mean_g2s = g2s_stats["mean"]
    median_g2s = g2s_stats["median"]
    p95_g2s = g2s_stats["p95"]
    max_g2s = g2s_stats["max"]
    pct_close = threshold_report["0.01"]["percent"]

    rel_mean = mean_g2s / mesh_diag * 100
    rel_p95 = p95_g2s / mesh_diag * 100
    rel_max = max_g2s / mesh_diag * 100

    if rel_mean < 1.0 and pct_close > 80:
        quality = "GOOD"
    elif rel_mean < 2.0 and pct_close > 60:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    summary = {
        "quality": quality,
        "gaussian_to_surface_mean": round(mean_g2s, 6),
        "gaussian_to_surface_median": round(median_g2s, 6),
        "gaussian_to_surface_p95": round(p95_g2s, 6),
        "gaussian_to_surface_max": round(max_g2s, 6),
        "gaussian_to_surface_mean_pct_bbox": round(rel_mean, 4),
        "gaussian_to_surface_p95_pct_bbox": round(rel_p95, 4),
        "gaussian_to_surface_max_pct_bbox": round(rel_max, 4),
        "gaussians_within_0.01_pct": pct_close,
    }
    report["summary"] = summary

    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    print(f"  Mean Gaussian-to-surface distance: {mean_g2s:.6f} ({rel_mean:.2f}% of bbox diagonal)")
    print(f"  Median Gaussian-to-surface distance: {median_g2s:.6f}")
    print(f"  P95 Gaussian-to-surface distance: {p95_g2s:.6f} ({rel_p95:.2f}% of bbox diagonal)")
    print(f"  Max Gaussian-to-surface distance: {max_g2s:.6f} ({rel_max:.2f}% of bbox diagonal)")
    print(f"  Gaussians within 0.01 of surface: {pct_close:.1f}%")
    print(f"\n  >>> {quality}")

    # ── Write report to file ──
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
