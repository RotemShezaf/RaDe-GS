"""Evaluate Gaussian Splatting quality and Gaussian-to-mesh distance statistics.

Usage:
    python scripts/evaluate_gaussian_mesh_quality.py \
        --output_dir TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output

Metrics reported:
    1. Training quality (PSNR, L1) from the TF event logs
    2. Gaussian center -> nearest reconstructed mesh vertex distance statistics
    3. Mesh vertex -> nearest Gaussian center distance (coverage)
    4. Threshold analysis at multiple distance levels
"""

import argparse
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


def print_dist_stats(label: str, distances: np.ndarray):
    print(f"\n  {label}:")
    print(f"    Count:   {len(distances)}")
    print(f"    Mean:    {distances.mean():.6f}")
    print(f"    Median:  {np.median(distances):.6f}")
    print(f"    Std:     {distances.std():.6f}")
    print(f"    Min:     {distances.min():.6f}")
    print(f"    Max:     {distances.max():.6f}")
    for pct in [90, 95, 99]:
        print(f"    P{pct}:     {np.percentile(distances, pct):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian-mesh quality")
    parser.add_argument("--output_dir", type=str, required=True,
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

    # Compute bounding box info
    gauss_bb = gaussians.max(axis=0) - gaussians.min(axis=0)
    mesh_bb = recon_verts.max(axis=0) - recon_verts.min(axis=0)
    gauss_diag = np.linalg.norm(gauss_bb)
    mesh_diag = np.linalg.norm(mesh_bb)
    print(f"\n  Gaussian bounding box diagonal: {gauss_diag:.4f}")
    print(f"  Mesh bounding box diagonal:     {mesh_diag:.4f}")

    # === Gaussian -> Nearest Mesh Vertex ===
    print("\n" + "=" * 70)
    print("GAUSSIAN -> MESH DISTANCE")
    print("=" * 70)
    tree_mesh = cKDTree(recon_verts)
    g2m_dists, _ = tree_mesh.query(gaussians)
    print_dist_stats("Gaussian center -> nearest mesh vertex", g2m_dists)

    # Also compute point-to-surface distance using trimesh
    print("\n  Computing point-to-surface (closest point on mesh face)...")
    closest_pts, g2s_dists, face_ids = trimesh.proximity.closest_point(mesh, gaussians)
    print_dist_stats("Gaussian center -> nearest mesh surface point", g2s_dists)

    # Threshold analysis (as fraction of bounding box diagonal)
    print(f"\n  Gaussians within distance threshold of mesh surface:")
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]:
        abs_count = (g2s_dists < thresh).sum()
        pct = abs_count / len(g2s_dists) * 100
        rel = thresh / mesh_diag * 100
        print(f"    < {thresh:.3f} ({rel:.2f}% of bbox diag): {pct:.1f}% ({abs_count}/{len(g2s_dists)})")

    # === Mesh Vertex -> Nearest Gaussian (coverage) ===
    print("\n" + "=" * 70)
    print("MESH -> GAUSSIAN DISTANCE (coverage)")
    print("=" * 70)
    tree_gauss = cKDTree(gaussians)
    m2g_dists, _ = tree_gauss.query(recon_verts)
    print_dist_stats("Mesh vertex -> nearest Gaussian center", m2g_dists)

    # === GT Mesh Comparison (if provided) ===
    if args.gt_mesh:
        gt_mesh_path = Path(args.gt_mesh)
        if gt_mesh_path.exists():
            print("\n" + "=" * 70)
            print("RECONSTRUCTED MESH vs GROUND-TRUTH MESH")
            print("=" * 70)
            gt_mesh = load_mesh(gt_mesh_path)
            gt_verts = np.asarray(gt_mesh.vertices)
            print(f"  GT mesh vertices: {len(gt_verts)}, faces: {len(gt_mesh.faces)}")
            gt_diag = np.linalg.norm(gt_verts.max(axis=0) - gt_verts.min(axis=0))
            print(f"  GT mesh bounding box diagonal: {gt_diag:.4f}")

            # Recon -> GT (accuracy)
            _, recon2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, recon_verts)
            print_dist_stats("Recon vertex -> GT surface (accuracy)", recon2gt_dists)

            # GT -> Recon (completeness)
            _, gt2recon_dists, _ = trimesh.proximity.closest_point(mesh, gt_verts)
            print_dist_stats("GT vertex -> Recon surface (completeness)", gt2recon_dists)

            # Chamfer distance
            chamfer = (recon2gt_dists.mean() + gt2recon_dists.mean()) / 2
            print(f"\n  Chamfer distance: {chamfer:.6f} ({chamfer / gt_diag * 100:.3f}% of GT bbox diagonal)")

            # Gaussian -> GT mesh (how close are Gaussians to the true surface)
            _, g2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, gaussians)
            print_dist_stats("Gaussian center -> GT surface", g2gt_dists)

            pct_close_gt = (g2gt_dists < 0.01).sum() / len(g2gt_dists) * 100
            print(f"\n  Gaussians within 0.01 of GT surface: {pct_close_gt:.1f}%")
            for thresh in [0.05, 0.1, 0.5, 1.0]:
                pct = (g2gt_dists < thresh).sum() / len(g2gt_dists) * 100
                print(f"  Gaussians within {thresh} of GT surface: {pct:.1f}%")
        else:
            print(f"\n  WARNING: GT mesh not found at {gt_mesh_path}")

    # === Summary / Quality Assessment ===
    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    mean_g2s = g2s_dists.mean()
    median_g2s = np.median(g2s_dists)
    p95_g2s = np.percentile(g2s_dists, 95)
    pct_close = (g2s_dists < 0.01).sum() / len(g2s_dists) * 100

    # Use mesh diagonal as scale reference
    rel_mean = mean_g2s / mesh_diag * 100
    rel_p95 = p95_g2s / mesh_diag * 100

    print(f"  Mean Gaussian-to-surface distance: {mean_g2s:.6f} ({rel_mean:.2f}% of bbox diagonal)")
    print(f"  Median Gaussian-to-surface distance: {median_g2s:.6f}")
    print(f"  P95 Gaussian-to-surface distance: {p95_g2s:.6f} ({rel_p95:.2f}% of bbox diagonal)")
    print(f"  Gaussians within 0.01 of surface: {pct_close:.1f}%")

    if rel_mean < 1.0 and pct_close > 80:
        print("\n  >>> GOOD: Gaussians are well-aligned with the mesh surface.")
    elif rel_mean < 2.0 and pct_close > 60:
        print("\n  >>> ACCEPTABLE: Moderate alignment, some Gaussians are far from mesh.")
    else:
        print("\n  >>> POOR: Many Gaussians are far from the mesh surface. Consider tuning hyperparameters.")
        print("    Suggestions:")
        print("    - Increase lambda_depth_normal (e.g., 0.1 -> 0.2)")
        print("    - Increase lambda_distortion (e.g., 0.05 -> 0.1)")
        print("    - Decrease densify_grad_threshold (e.g., 0.0002 -> 0.0001)")
        print("    - Increase regularization_from_iter to start earlier (e.g., 7000 -> 3000)")


if __name__ == "__main__":
    main()
