#!/usr/bin/env python3
"""
Parallel mesh parameter sweep for HyperbolicParaboloid.

Generates a grid of curvature_alpha × edge_length × refinement configs,
builds each mesh in parallel (one process per CPU), and collects quality
metrics into a single comparison table.

Usage:
    python scripts/polynomial/geodesic/compare_mesh_params_parallel.py \
        --output_dir  <gaussian_output_old> \
        --surface     HyperbolicParaboloid \
        --workers     36

Results are saved to <output_dir>/geodesic_mesh_sweep/sweep_results.json
Each mesh variant is kept under <output_dir>/geodesic_mesh_sweep/mesh_<config>/
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------
# Alpha axis: focus on LOW values (higher alpha = worse d3 from round 1)
ALPHA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Edge length axis
EDGE_VALUES = [0.006, 0.007, 0.008]

# Refinement presets
REFINE_PRESETS = {
    "light": {  # current baseline
        "iter": "15", "warmup": "10", "ring_iters": "1", "patience": "4",
    },
    "ring2": {  # light + double ring fix (benchmark showed ring=2 catches more bad triangles)
        "iter": "15", "warmup": "10", "ring_iters": "2", "patience": "4",
    },
    "heavy": {  # more refinement (best d3 in round 1)
        "iter": "25", "warmup": "15", "ring_iters": "2", "patience": "6",
    },
    "nowarm": {  # no warmup = split all bad tris from start
        "iter": "25", "warmup": "0", "ring_iters": "2", "patience": "6",
    },
}

# Additionally test no-curvature-adaptive (uniform grid in param space)
NO_CURV_CONFIGS = True  # adds 9 configs (3 edges × 3 refine presets)


def make_config_name(alpha, edge, refine_key, no_curv=False):
    """Generate a unique config name."""
    if no_curv:
        return f"nocurv_e{edge:.3f}_{refine_key}"
    return f"a{alpha:.2f}_e{edge:.3f}_{refine_key}"


def make_args(alpha, edge, refine_key, no_curv=False):
    """Build CLI arguments list for compute_geodesic_mesh_for_gaussians.py."""
    rp = REFINE_PRESETS[refine_key]
    # Edge-length for Gaussian-touching triangles scales with grid edge
    gauss_edge = f"{edge * 0.625:.4f}"  # 0.005/0.008 ratio from baseline

    args = [
        "--max_edge_length", f"{edge}",
        "--refine",
        "--refine_iterations", rp["iter"],
        "--refine_warmup_iterations", rp["warmup"],
        "--refine_gauss_max_edge_length", gauss_edge,
        "--refine_max_area_factor", "2",
        "--refine_gauss_max_area_factor", "1.5",
        "--refine_ring_fix",
        "--refine_ring_fix_iterations", rp["ring_iters"],
        "--refine_min_angle", "20",
        "--refine_gauss_min_angle", "30",
        "--refine_delaunay_flip",
        "--refine_patience", rp["patience"],
        "--local_refinement",
    ]

    if no_curv:
        # Do NOT pass --curvature_adaptive
        pass
    else:
        args += ["--curvature_adaptive", "--curvature_alpha", f"{alpha}"]

    return args


def generate_all_configs():
    """Return dict of {config_name: {"args": [...], "desc": "..."}}."""
    configs = {}

    # Curvature-adaptive grid
    for alpha, edge, rkey in itertools.product(ALPHA_VALUES, EDGE_VALUES, REFINE_PRESETS):
        name = make_config_name(alpha, edge, rkey)
        configs[name] = {
            "desc": f"alpha={alpha} edge={edge} {rkey}",
            "args": make_args(alpha, edge, rkey),
        }

    # Non-curvature-adaptive grid
    if NO_CURV_CONFIGS:
        for edge, rkey in itertools.product(EDGE_VALUES, REFINE_PRESETS):
            name = make_config_name(0, edge, rkey, no_curv=True)
            configs[name] = {
                "desc": f"no_curv edge={edge} {rkey}",
                "args": make_args(0, edge, rkey, no_curv=True),
            }

    return configs


# ---------------------------------------------------------------------------
# Build + analyse one config
# ---------------------------------------------------------------------------
def build_one(config_name, config_args, gaussian_output, surface, sweep_dir,
              compute_script):
    """Build mesh for one config. Returns quality dict or None on failure."""
    # Each config writes to its own subdirectory inside sweep_dir
    mesh_dest = Path(sweep_dir) / f"mesh_{config_name}"
    mesh_dest.mkdir(parents=True, exist_ok=True)

    # We need to build in the real output location (the script expects
    # gaussian_output/geodesic_mesh/).  Use a temp symlink so each worker
    # writes to its own place.
    # Actually, the script writes to <gaussian_output>/geodesic_mesh/ always.
    # To parallelise we create per-config shadow output dirs that symlink
    # point_cloud back to the real one.
    shadow_output = Path(sweep_dir) / f"shadow_{config_name}"
    shadow_output.mkdir(parents=True, exist_ok=True)

    # Symlink everything from the real output except geodesic_mesh
    real_output = Path(gaussian_output)
    for item in real_output.iterdir():
        link = shadow_output / item.name
        if not link.exists():
            os.symlink(str(item.resolve()), str(link))

    # Remove old geodesic_mesh in shadow if present
    shadow_mesh = shadow_output / "geodesic_mesh"
    if shadow_mesh.is_symlink():
        shadow_mesh.unlink()
    elif shadow_mesh.is_dir():
        shutil.rmtree(str(shadow_mesh))

    cmd = [
        sys.executable, str(compute_script),
        "--gaussian_output", str(shadow_output),
        "--surface", surface,
        "--seed", "42",
        "--mesh_method", "grid",
    ] + config_args

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"name": config_name, "error": "timeout"}
    elapsed = time.time() - t0

    if result.returncode != 0:
        return {"name": config_name, "error": result.stderr[-300:], "elapsed": elapsed}

    # Move built mesh to permanent location
    if shadow_mesh.exists():
        if mesh_dest.exists():
            shutil.rmtree(str(mesh_dest))
        shutil.copytree(str(shadow_mesh), str(mesh_dest))
        shutil.rmtree(str(shadow_mesh))

    # Cleanup shadow
    shutil.rmtree(str(shadow_output), ignore_errors=True)

    # Analyse
    quality = analyse_mesh(mesh_dest)
    if quality:
        quality["elapsed"] = elapsed
        quality["name"] = config_name
    return quality


def analyse_mesh(mesh_dir):
    """Compute quality metrics from a built mesh."""
    import numpy as np
    npz_path = Path(mesh_dir) / "geodesic_mesh_data.npz"
    if not npz_path.exists():
        return None

    # Lazy import to avoid loading in every subprocess at startup
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from GenerateData.utils.geodesic_mesh_utils import _triangle_quality

    data = np.load(str(npz_path))
    verts = data["vertices"]
    faces = data["faces"]
    gauss_idx = data["gaussian_vertex_indices"]

    ar, ma, _le = _triangle_quality(verts, faces)

    # Vertex degree (vectorised)
    deg = np.bincount(faces.ravel(), minlength=len(verts))

    gauss_set = set(gauss_idx.astype(int).tolist())

    bad = (ar > 2) | (ma < 20)
    vbad = (ar > 5) | (ma < 10)

    # Gaussian-touching faces (vectorised)
    gauss_mask = np.zeros(len(verts), dtype=bool)
    gauss_mask[list(gauss_set)] = True
    gf = gauss_mask[faces].any(axis=1)

    return {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "n_gaussians": int(len(gauss_idx)),
        "pct_bad": round(100 * bad.sum() / len(faces), 2),
        "pct_very_bad": round(100 * vbad.sum() / len(faces), 3),
        "worst_ar": round(float(ar.max()), 2),
        "median_ar": round(float(np.median(ar)), 3),
        "p99_ar": round(float(np.percentile(ar, 99)), 2),
        "worst_min_angle": round(float(ma.min()), 4),
        "median_min_angle": round(float(np.median(ma)), 2),
        "p1_min_angle": round(float(np.percentile(ma, 1)), 2),
        "mean_degree": round(float(deg.mean()), 3),
        "deg_le_4": int((deg <= 4).sum()),
        "deg_le_3": int((deg == 3).sum()),
        "pct_deg_le_4": round(100 * (deg <= 4).sum() / len(deg), 2),
        "gauss_bad_pct": round(100 * (bad & gf).sum() / max(gf.sum(), 1), 2),
        "gauss_vbad_pct": round(100 * (vbad & gf).sum() / max(gf.sum(), 1), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallel mesh param sweep")
    parser.add_argument("--output_dir", required=True,
                        help="Gaussian output directory (e.g. .../output_old)")
    parser.add_argument("--surface", default="HyperbolicParaboloid")
    parser.add_argument("--workers", type=int, default=36,
                        help="Number of parallel builds")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configs without building")
    args = parser.parse_args()

    compute_script = Path(__file__).resolve().parents[3] / \
        "GenerateData" / "compute_geodesic_mesh_for_gaussians.py"
    assert compute_script.exists(), f"Not found: {compute_script}"

    configs = generate_all_configs()
    print(f"Generated {len(configs)} configurations")

    sweep_dir = Path(args.output_dir) / "geodesic_mesh_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for name, cfg in sorted(configs.items()):
            print(f"  {name:40s}  {cfg['desc']}")
        print(f"\n{len(configs)} configs total")
        return

    # Run in parallel
    results = {}
    n_done = 0
    t0_all = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for name, cfg in configs.items():
            f = pool.submit(
                build_one, name, cfg["args"], args.output_dir, args.surface,
                str(sweep_dir), str(compute_script),
            )
            futures[f] = name

        for f in as_completed(futures):
            name = futures[f]
            n_done += 1
            try:
                res = f.result()
                if res and "error" not in res:
                    results[name] = res
                    print(f"  [{n_done}/{len(configs)}] {name:40s} "
                          f"V={res['n_vertices']:>8}  bad={res['pct_bad']:5.1f}%  "
                          f"d3={res['deg_le_3']:>4}  "
                          f"elapsed={res['elapsed']:.0f}s")
                else:
                    err = res.get("error", "unknown") if res else "null result"
                    print(f"  [{n_done}/{len(configs)}] {name:40s} FAILED: "
                          f"{str(err)[:80]}")
            except Exception as e:
                print(f"  [{n_done}/{len(configs)}] {name:40s} EXCEPTION: {e}")

    total_time = time.time() - t0_all
    print(f"\nCompleted {len(results)}/{len(configs)} configs in {total_time:.0f}s")

    # Save full results
    results_file = sweep_dir / "sweep_results.json"
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    # Print sorted comparison table
    print(f"\n{'='*130}")
    print(f"  RESULTS SORTED BY deg_le_3 (MMP inf proxy)")
    print(f"{'='*130}")
    header = (f"{'Config':<42} {'Verts':>8} {'Faces':>10} "
              f"{'%Bad':>6} {'%VBad':>6} {'WrstAR':>8} {'P99AR':>6} "
              f"{'P1Ang':>6} {'%D≤4':>6} {'D≤3':>5} {'G%Bad':>6} "
              f"{'Time':>5}")
    print(header)
    print("-" * len(header))

    sorted_results = sorted(results.values(), key=lambda r: r.get("deg_le_3", 99999))
    for r in sorted_results:
        print(f"{r['name']:<42} {r['n_vertices']:>8} {r['n_faces']:>10} "
              f"{r['pct_bad']:>6.1f} {r['pct_very_bad']:>6.2f} "
              f"{r['worst_ar']:>8.1f} {r['p99_ar']:>6.1f} "
              f"{r['p1_min_angle']:>6.1f} {r['pct_deg_le_4']:>6.1f} "
              f"{r['deg_le_3']:>5} {r['gauss_bad_pct']:>6.1f} "
              f"{r.get('elapsed', 0):>5.0f}")

    # Also sort by composite score (lower is better)
    # Score = pct_bad * 0.3 + pct_very_bad * 2 + deg_le_3 * 0.01 + (n_vertices/100000) * 0.5
    print(f"\n{'='*130}")
    print(f"  RESULTS SORTED BY COMPOSITE SCORE (lower = better)")
    print(f"  Score = 0.3*%bad + 2*%vbad + 0.01*d3 + 0.5*(verts/100K)")
    print(f"{'='*130}")
    for r in sorted_results:
        r["score"] = (r["pct_bad"] * 0.3
                      + r["pct_very_bad"] * 2
                      + r["deg_le_3"] * 0.01
                      + (r["n_vertices"] / 100000) * 0.5)

    scored = sorted(results.values(), key=lambda r: r.get("score", 99999))
    print(f"{'Config':<42} {'Score':>6} {'Verts':>8} "
          f"{'%Bad':>6} {'%VBad':>6} {'D≤3':>5} {'G%Bad':>6}")
    print("-" * 90)
    for r in scored[:20]:
        print(f"{r['name']:<42} {r['score']:>6.2f} {r['n_vertices']:>8} "
              f"{r['pct_bad']:>6.1f} {r['pct_very_bad']:>6.2f} "
              f"{r['deg_le_3']:>5} {r['gauss_bad_pct']:>6.1f}")

    print(f"\nFull results: {results_file}")


if __name__ == "__main__":
    main()
