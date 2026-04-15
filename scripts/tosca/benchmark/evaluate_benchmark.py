"""Evaluate a single benchmark run: mesh quality, Gaussian distances, connected components.

Usage:
    python scripts/tosca/benchmark/evaluate_benchmark.py \
        --output_dir <gaussian_output_dir> \
        --gt_mesh <ground_truth_mesh.ply> \
        --iteration 30000

Produces a JSON report saved to <output_dir>/benchmark_report.json
"""

import argparse
import json
import numpy as np
import re
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
    """Return statistics dict for a distance array."""
    return {
        "count": int(len(distances)),
        "mean": float(distances.mean()),
        "mean_squared": float((distances ** 2).mean()),
        "median": float(np.median(distances)),
        "std": float(distances.std()),
        "min": float(distances.min()),
        "max": float(distances.max()),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
    }


def count_connected_components(mesh) -> dict:
    """Count connected components of a trimesh mesh."""
    components = mesh.split(only_watertight=False)
    sizes = [len(c.vertices) for c in components]
    return {
        "num_components": len(components),
        "largest_component_vertices": int(max(sizes)) if sizes else 0,
        "smallest_component_vertices": int(min(sizes)) if sizes else 0,
        "component_sizes": sorted(sizes, reverse=True),
    }


def extract_psnr(output_dir: Path, iteration: int) -> dict:
    """Extract PSNR from train.log at the given iteration."""
    train_log = output_dir / "train.log"
    result = {}
    if not train_log.exists():
        return result
    text = train_log.read_text()
    # Match: [ITER 30000] Evaluating test: L1 0.000790 PSNR 46.871
    pattern = rf'\[ITER {iteration}\] Evaluating test: L1 ([0-9.e+-]+) PSNR ([0-9.e+-]+)'
    m = re.search(pattern, text)
    if m:
        result["test_l1"] = float(m.group(1))
        result["test_psnr"] = float(m.group(2))
    pattern_train = rf'\[ITER {iteration}\] Evaluating train: L1 ([0-9.e+-]+) PSNR ([0-9.e+-]+)'
    m2 = re.search(pattern_train, text)
    if m2:
        result["train_l1"] = float(m2.group(1))
        result["train_psnr"] = float(m2.group(2))
    # Also try the final iteration if not found at given iteration
    if not result:
        # Find the last PSNR line
        for m in re.finditer(r'Evaluating test: L1 ([0-9.e+-]+) PSNR ([0-9.e+-]+)', text):
            result["test_l1"] = float(m.group(1))
            result["test_psnr"] = float(m.group(2))
        for m in re.finditer(r'Evaluating train: L1 ([0-9.e+-]+) PSNR ([0-9.e+-]+)', text):
            result["train_l1"] = float(m.group(1))
            result["train_psnr"] = float(m.group(2))
    return result


def evaluate(output_dir: str, gt_mesh_path: str, iteration: int) -> dict:
    """Run full benchmark evaluation and return report dict."""
    output_dir = Path(output_dir)
    recon_ply = output_dir / "recon.ply"
    gauss_ply = output_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    report = {"iteration": iteration, "output_dir": str(output_dir)}

    # Extract PSNR from training log
    psnr_info = extract_psnr(output_dir, iteration)
    if psnr_info:
        report["psnr"] = psnr_info

    # Check files
    if not recon_ply.exists():
        report["error"] = f"Reconstructed mesh not found at {recon_ply}"
        return report
    if not gauss_ply.exists():
        report["error"] = f"Point cloud not found at {gauss_ply}"
        return report

    # Load data
    gaussians = load_gaussian_centers(gauss_ply)
    report["num_gaussians"] = int(len(gaussians))

    mesh = load_mesh(recon_ply)
    recon_verts = np.asarray(mesh.vertices)
    report["mesh_vertices"] = int(len(recon_verts))
    report["mesh_faces"] = int(len(mesh.faces))

    # Connected components of reconstructed mesh
    cc_info = count_connected_components(mesh)
    report["connected_components"] = cc_info

    # Bounding box diagonal
    mesh_bb = recon_verts.max(axis=0) - recon_verts.min(axis=0)
    mesh_diag = float(np.linalg.norm(mesh_bb))
    report["mesh_bbox_diagonal"] = mesh_diag

    # Gaussian -> nearest reconstructed mesh vertex
    tree_mesh_verts = cKDTree(recon_verts)
    g2mv_dists, _ = tree_mesh_verts.query(gaussians)
    report["gaussian_to_recon_mesh_vertex"] = dist_stats(g2mv_dists)

    # Gaussian -> nearest reconstructed mesh surface point
    _, g2ms_dists, _ = trimesh.proximity.closest_point(mesh, gaussians)
    report["gaussian_to_recon_mesh_surface"] = dist_stats(g2ms_dists)

    # Ground-truth mesh evaluation
    gt_path = Path(gt_mesh_path) if gt_mesh_path else None
    if gt_path and gt_path.exists():
        gt_mesh = load_mesh(gt_path)
        gt_verts = np.asarray(gt_mesh.vertices)
        gt_diag = float(np.linalg.norm(gt_verts.max(axis=0) - gt_verts.min(axis=0)))
        report["gt_mesh_path"] = str(gt_path)
        report["gt_vertices"] = int(len(gt_verts))
        report["gt_bbox_diagonal"] = gt_diag

        # Recon mesh -> GT surface (accuracy)
        _, recon2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, recon_verts)
        report["recon_to_gt_accuracy"] = dist_stats(recon2gt_dists)

        # GT -> Recon surface (completeness)
        _, gt2recon_dists, _ = trimesh.proximity.closest_point(mesh, gt_verts)
        report["gt_to_recon_completeness"] = dist_stats(gt2recon_dists)

        # Chamfer distance
        chamfer = float((recon2gt_dists.mean() + gt2recon_dists.mean()) / 2)
        report["chamfer_distance"] = chamfer
        report["chamfer_pct_bbox"] = round(chamfer / gt_diag * 100, 4) if gt_diag > 0 else 0

        # Gaussian point cloud -> GT mesh surface
        _, g2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, gaussians)
        report["gaussian_to_gt_surface"] = dist_stats(g2gt_dists)

        # Gaussian -> closest reconstructed mesh vertex (then that vertex -> GT)
        _, g2mv_idx = tree_mesh_verts.query(gaussians)
        nearest_recon_verts = recon_verts[g2mv_idx]
        _, nrv2gt_dists, _ = trimesh.proximity.closest_point(gt_mesh, nearest_recon_verts)
        report["gaussian_nearest_recon_vertex_to_gt"] = dist_stats(nrv2gt_dists)
    elif gt_path:
        report["gt_mesh_warning"] = f"GT mesh not found at {gt_path}"

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for a single run")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gt_mesh", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=45000)
    args = parser.parse_args()

    report = evaluate(args.output_dir, args.gt_mesh, args.iteration)

    # Save report
    report_path = Path(args.output_dir) / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    # Print summary
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Benchmark Report: {args.output_dir}")
    print(f"{'='*60}")
    print(f"  Gaussians:              {report['num_gaussians']}")
    print(f"  Mesh vertices/faces:    {report['mesh_vertices']}/{report['mesh_faces']}")
    print(f"  Connected components:   {report['connected_components']['num_components']}")
    if "chamfer_distance" in report:
        print(f"  Chamfer distance:       {report['chamfer_distance']:.6f} ({report['chamfer_pct_bbox']:.3f}% bbox)")
        print(f"  Gauss->GT mean:         {report['gaussian_to_gt_surface']['mean']:.6f}")
        print(f"  Gauss->GT max:          {report['gaussian_to_gt_surface']['max']:.6f}")
        print(f"  Recon->GT accuracy:     {report['recon_to_gt_accuracy']['mean']:.6f}")
        print(f"  GT->Recon completeness: {report['gt_to_recon_completeness']['mean']:.6f}")
    if "psnr" in report:
        psnr = report["psnr"]
        if "test_psnr" in psnr:
            print(f"  Test PSNR:              {psnr['test_psnr']:.4f}")
        if "train_psnr" in psnr:
            print(f"  Train PSNR:             {psnr['train_psnr']:.4f}")


if __name__ == "__main__":
    main()
