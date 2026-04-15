#!/usr/bin/env python3
"""
Compare mesh quality for different build parameters on HyperbolicParaboloid.

This script builds the mesh with various parameter combinations and reports
quality metrics (without running the expensive MMP solver). We focus on metrics
that correlate with MMP inf failures: min-angle distribution, aspect ratio,
vertex degree, and number of near-degenerate triangles.
"""
import subprocess
import sys
import json
import time
import shutil
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
BASE_OUTPUT = "TrainData/Polynomial/SyntheticColmapData/blue_texture/HyperbolicParaboloid/level_02/light_0/output_old"
COMPUTE_SCRIPT = "GenerateData/compute_geodesic_mesh_for_gaussians.py"
SURFACE = "HyperbolicParaboloid"

# Parameters to vary
CONFIGS = {
    "baseline": {
        "desc": "Current params (alpha=2, edge=0.008, refine=15, warmup=10, angle=20/30)",
        "args": [
            "--max_edge_length", "0.008",
            "--curvature_adaptive", "--curvature_alpha", "2.0",
            "--refine", "--refine_iterations", "15", "--refine_warmup_iterations", "10",
            "--refine_gauss_max_edge_length", "0.005",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--local_refinement",
        ],
    },
    "more_refine_ring2": {
        "desc": "25 iters + 2 ring fix passes + angle 20/30 (reduces d3 verts)",
        "args": [
            "--max_edge_length", "0.008",
            "--curvature_adaptive", "--curvature_alpha", "2.0",
            "--refine", "--refine_iterations", "25", "--refine_warmup_iterations", "15",
            "--refine_gauss_max_edge_length", "0.005",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--refine_patience", "6",
            "--local_refinement",
        ],
    },
    "finer_more_refine": {
        "desc": "Finer grid 0.006 + 25 iters + angle 20/30 (best quality combo)",
        "args": [
            "--max_edge_length", "0.006",
            "--curvature_adaptive", "--curvature_alpha", "2.0",
            "--refine", "--refine_iterations", "25", "--refine_warmup_iterations", "15",
            "--refine_gauss_max_edge_length", "0.004",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--refine_patience", "6",
            "--local_refinement",
        ],
    },
    "finer007_more_ref": {
        "desc": "Grid 0.007 + 25 iters + angle 20/30 (balanced verts vs quality)",
        "args": [
            "--max_edge_length", "0.007",
            "--curvature_adaptive", "--curvature_alpha", "2.0",
            "--refine", "--refine_iterations", "25", "--refine_warmup_iterations", "15",
            "--refine_gauss_max_edge_length", "0.0045",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--refine_patience", "6",
            "--local_refinement",
        ],
    },
    "alpha3_more_refine": {
        "desc": "Alpha=3 + 25 iters + angle 20/30 (moderate curvature density)",
        "args": [
            "--max_edge_length", "0.008",
            "--curvature_adaptive", "--curvature_alpha", "3.0",
            "--refine", "--refine_iterations", "25", "--refine_warmup_iterations", "15",
            "--refine_gauss_max_edge_length", "0.005",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--refine_patience", "6",
            "--local_refinement",
        ],
    },
    "steiner_then_refine": {
        "desc": "Steiner insertion + 15 iter refinement + angle 20/30",
        "args": [
            "--max_edge_length", "0.008",
            "--curvature_adaptive", "--curvature_alpha", "2.0",
            "--steiner", "--steiner_max_aspect_ratio", "3.0",
            "--steiner_min_angle", "20", "--steiner_iterations", "3",
            "--refine", "--refine_iterations", "15", "--refine_warmup_iterations", "10",
            "--refine_gauss_max_edge_length", "0.005",
            "--refine_max_area_factor", "2", "--refine_gauss_max_area_factor", "1.5",
            "--refine_min_angle", "20", "--refine_gauss_min_angle", "30",
            "--refine_ring_fix", "--refine_ring_fix_iterations", "2",
            "--refine_delaunay_flip",
            "--local_refinement",
        ],
    },
}


def build_mesh(config_name, config, backup_dir):
    """Build mesh with given params, save to a temporary output, return stats."""
    # We'll build in-place then move the mesh to a comparison directory
    mesh_out = Path(BASE_OUTPUT) / "geodesic_mesh"
    
    # Backup existing mesh if first run
    if mesh_out.exists() and not (backup_dir / "geodesic_mesh_backup").exists():
        shutil.copytree(str(mesh_out), str(backup_dir / "geodesic_mesh_backup"))
    
    cmd = [
        sys.executable, COMPUTE_SCRIPT,
        "--gaussian_output", BASE_OUTPUT,
        "--surface", SURFACE,
        "--seed", "42",
        "--mesh_method", "grid",
    ] + config["args"]
    
    print(f"\n{'='*70}")
    print(f"  CONFIG: {config_name}")
    print(f"  DESC:   {config['desc']}")
    print(f"  CMD:    {' '.join(cmd)}")
    print(f"{'='*70}")
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  FAILED! stderr:\n{result.stderr[-500:]}")
        return None
    
    # Print last portion of stdout (has quality stats)
    lines = result.stdout.strip().split('\n')
    for line in lines[-30:]:
        print(f"  {line}")
    
    # Move mesh to comparison dir
    dest = backup_dir / f"mesh_{config_name}"
    if dest.exists():
        shutil.rmtree(str(dest))
    shutil.copytree(str(mesh_out), str(dest))
    
    # Read stats
    stats_file = dest / "mesh_statistics.json"
    stats = None
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
    
    return {
        "name": config_name,
        "desc": config["desc"],
        "elapsed": elapsed,
        "stats": stats,
    }


def analyze_mesh_quality(mesh_dir):
    """Detailed quality analysis of a built mesh."""
    import numpy as np
    sys.path.insert(0, '.')
    from GenerateData.utils.geodesic_mesh_utils import _triangle_quality
    
    data = np.load(str(mesh_dir / "geodesic_mesh_data.npz"))
    verts = data['vertices']
    faces = data['faces']
    gauss_idx = data['gaussian_vertex_indices']
    
    ar, ma, le = _triangle_quality(verts, faces)
    
    # Vertex degree
    deg = np.zeros(verts.shape[0], dtype=int)
    for v in faces.ravel():
        deg[v] += 1
    
    gauss_set = set(int(x) for x in gauss_idx)
    
    # Metrics
    bad_mask = (ar > 2) | (ma < 20)
    very_bad = (ar > 5) | (ma < 10)
    
    # Gaussian-touching faces
    gauss_faces = np.zeros(len(faces), dtype=bool)
    for fi, f in enumerate(faces):
        if any(int(v) in gauss_set for v in f):
            gauss_faces[fi] = True
    
    return {
        "n_vertices": len(verts),
        "n_faces": len(faces),
        "n_gaussians": len(gauss_idx),
        "pct_bad": float(100 * bad_mask.sum() / len(faces)),
        "pct_very_bad": float(100 * very_bad.sum() / len(faces)),
        "worst_ar": float(ar.max()),
        "median_ar": float(np.median(ar)),
        "p99_ar": float(np.percentile(ar, 99)),
        "worst_min_angle": float(ma.min()),
        "median_min_angle": float(np.median(ma)),
        "p1_min_angle": float(np.percentile(ma, 1)),
        "mean_degree": float(deg.mean()),
        "deg_le_4": int((deg <= 4).sum()),
        "deg_le_3": int((deg <= 3).sum()),
        "pct_deg_le_4": float(100 * (deg <= 4).sum() / len(deg)),
        "gauss_bad": float(100 * (bad_mask & gauss_faces).sum() / max(gauss_faces.sum(), 1)),
        "gauss_very_bad": float(100 * (very_bad & gauss_faces).sum() / max(gauss_faces.sum(), 1)),
    }


def main():
    backup_dir = Path("scripts/polynomial/geodesic/mesh_comparison")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for name, config in CONFIGS.items():
        try:
            res = build_mesh(name, config, backup_dir)
            if res:
                results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Detailed analysis
    print(f"\n\n{'='*90}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*90}")
    
    all_quality = {}
    for res in results:
        mesh_dir = backup_dir / f"mesh_{res['name']}"
        if (mesh_dir / "geodesic_mesh_data.npz").exists():
            q = analyze_mesh_quality(mesh_dir)
            all_quality[res['name']] = q
            q['elapsed'] = res['elapsed']
    
    # Print comparison table
    header = f"{'Config':<22} {'Verts':>8} {'Faces':>10} {'%Bad':>6} {'%VBad':>6} {'WrstAR':>7} {'P99AR':>6} {'WrstAng':>7} {'P1Ang':>6} {'%D≤4':>6} {'D≤3':>5} {'G%Bad':>6} {'Time':>6}"
    print(header)
    print("-" * len(header))
    
    for name in CONFIGS:
        if name in all_quality:
            q = all_quality[name]
            print(f"{name:<22} {q['n_vertices']:>8} {q['n_faces']:>10} "
                  f"{q['pct_bad']:>6.1f} {q['pct_very_bad']:>6.1f} "
                  f"{q['worst_ar']:>7.1f} {q['p99_ar']:>6.1f} "
                  f"{q['worst_min_angle']:>7.1f} {q['p1_min_angle']:>6.1f} "
                  f"{q['pct_deg_le_4']:>6.1f} {q['deg_le_3']:>5} "
                  f"{q['gauss_bad']:>6.1f} {q['elapsed']:>6.0f}s")
    
    # Save full results
    out_file = backup_dir / "comparison_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_quality, f, indent=2)
    print(f"\nFull results saved to {out_file}")
    
    # Restore original mesh
    orig = backup_dir / "geodesic_mesh_backup"
    mesh_out = Path(BASE_OUTPUT) / "geodesic_mesh"
    if orig.exists():
        if mesh_out.exists():
            shutil.rmtree(str(mesh_out))
        shutil.copytree(str(orig), str(mesh_out))
        print(f"\nOriginal mesh restored.")


if __name__ == "__main__":
    main()
