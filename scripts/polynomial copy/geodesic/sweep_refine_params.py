#!/usr/bin/env python3
"""
Sweep grid edge length + refinement angle/area parameters for
geodesic mesh quality on HyperbolicParaboloid.

Fixes the best base from the previous sweep:
    curvature_alpha=2.0,
    refine_iterations=15, warmup=10, ring_fix=2, patience=4

and varies 3 axes (4 × 4 × 4 = 64 configs):
    1. max_edge_length (grid spacing): [0.004, 0.005, 0.006, 0.008]
       gauss_edge = edge × 0.625 (proportional)
    2. refine_max_area_factor:         [1.5, 2.0, 3.0, 5.0]
       gauss_area_factor = area_factor × 0.75
    3. refine_min_angle:               [15, 20, 25, 30]
       gauss_min_angle = min(min_angle + 10, 30)

Usage:
    python scripts/polynomial/geodesic/sweep_refine_params.py \\
        --output_dir  <gaussian_output> \\
        --surface     HyperbolicParaboloid \\
        --workers     36
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
# Fixed base config (winner of previous curvature/edge sweep)
# ---------------------------------------------------------------------------
BASE_ALPHA = 2.0
REFINE_ITER = "15"
REFINE_WARMUP = "10"
REFINE_RING = "2"
REFINE_PATIENCE = "4"
GAUSS_EDGE_RATIO = 0.625  # gauss_edge = grid_edge × ratio

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------
EDGE_VALUES = [0.004, 0.005, 0.006, 0.008]
AREA_FACTOR_VALUES = [1.5, 2.0, 3.0, 5.0]
MIN_ANGLE_VALUES = [15, 20, 25, 30]

# Derived: gauss_edge = edge × GAUSS_EDGE_RATIO
# Derived: gauss_area_factor = area_factor × 0.75
# Derived: gauss_min_angle = min(min_angle + 10, 30)


def make_config_name(edge, area_factor, min_angle):
    return f"e{edge:.3f}_af{area_factor:.1f}_ma{min_angle:02d}"


def make_args(edge, area_factor, min_angle):
    gauss_edge = round(edge * GAUSS_EDGE_RATIO, 4)
    gauss_area_factor = round(area_factor * 0.75, 2)
    gauss_min_angle = min(min_angle + 10, 30)

    return [
        "--max_edge_length", f"{edge}",
        "--mesh_method", "grid",
        "--curvature_adaptive",
        "--curvature_alpha", f"{BASE_ALPHA}",
        "--refine",
        "--refine_iterations", REFINE_ITER,
        "--refine_warmup_iterations", REFINE_WARMUP,
        "--refine_gauss_max_edge_length", f"{gauss_edge}",
        "--refine_max_area_factor", f"{area_factor}",
        "--refine_gauss_max_area_factor", f"{gauss_area_factor}",
        "--refine_min_angle", f"{min_angle}",
        "--refine_gauss_min_angle", f"{gauss_min_angle}",
        "--refine_ring_fix",
        "--refine_ring_fix_iterations", REFINE_RING,
        "--refine_delaunay_flip",
        "--refine_patience", REFINE_PATIENCE,
        "--local_refinement",
    ]


def generate_all_configs():
    configs = {}
    for edge, af, ma in itertools.product(
        EDGE_VALUES, AREA_FACTOR_VALUES, MIN_ANGLE_VALUES
    ):
        name = make_config_name(edge, af, ma)
        configs[name] = {
            "desc": f"edge={edge} area_factor={af} min_angle={ma}",
            "args": make_args(edge, af, ma),
            "edge": edge,
            "area_factor": af,
            "min_angle": ma,
        }
    return configs


# ---------------------------------------------------------------------------
# Build + analyse one config  (same pattern as compare_mesh_params_parallel.py)
# ---------------------------------------------------------------------------
def build_one(config_name, config_args, gaussian_output, surface, sweep_dir,
              compute_script):
    mesh_dest = Path(sweep_dir) / f"mesh_{config_name}"
    mesh_dest.mkdir(parents=True, exist_ok=True)

    shadow_output = Path(sweep_dir) / f"shadow_{config_name}"
    shadow_output.mkdir(parents=True, exist_ok=True)

    real_output = Path(gaussian_output)
    for item in real_output.iterdir():
        if item.name in ("geodesic_mesh", "geodesic_mesh_backups",
                         "geodesic_mesh_sweep", "geodesic_refine_sweep"):
            continue
        link = shadow_output / item.name
        if not link.exists():
            os.symlink(str(item.resolve()), str(link))

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
    ] + config_args

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        shutil.rmtree(str(shadow_output), ignore_errors=True)
        return {"name": config_name, "error": "timeout"}
    elapsed = time.time() - t0

    if result.returncode != 0:
        shutil.rmtree(str(shadow_output), ignore_errors=True)
        return {"name": config_name, "error": result.stderr[-500:],
                "elapsed": elapsed}

    if shadow_mesh.exists():
        if mesh_dest.exists():
            shutil.rmtree(str(mesh_dest))
        shutil.copytree(str(shadow_mesh), str(mesh_dest))
        shutil.rmtree(str(shadow_mesh))

    shutil.rmtree(str(shadow_output), ignore_errors=True)

    quality = analyse_mesh(mesh_dest)
    if quality:
        quality["elapsed"] = elapsed
        quality["name"] = config_name
    return quality


def analyse_mesh(mesh_dir):
    import numpy as np

    npz_path = Path(mesh_dir) / "geodesic_mesh_data.npz"
    if not npz_path.exists():
        return None

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from GenerateData.utils.geodesic_mesh_utils import _triangle_quality

    data = np.load(str(npz_path))
    verts = data["vertices"]
    faces = data["faces"]
    gauss_idx = data["gaussian_vertex_indices"]

    ar, ma, _le = _triangle_quality(verts, faces)
    deg = np.bincount(faces.ravel(), minlength=len(verts))

    bad = (ar > 2) | (ma < 20)
    vbad = (ar > 5) | (ma < 10)

    gauss_mask = np.zeros(len(verts), dtype=bool)
    gauss_mask[gauss_idx.astype(int)] = True
    gf = gauss_mask[faces].any(axis=1)

    return {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "n_gaussians": int(len(gauss_idx)),
        "pct_bad": round(100 * bad.sum() / len(faces), 2),
        "pct_very_bad": round(100 * vbad.sum() / len(faces), 3),
        "worst_ar": round(float(ar.max()), 2),
        "p99_ar": round(float(np.percentile(ar, 99)), 2),
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
    parser = argparse.ArgumentParser(
        description="Sweep refinement params (edge + angle thresholds)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--surface", default="HyperbolicParaboloid")
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    compute_script = Path(__file__).resolve().parents[3] / \
        "GenerateData" / "compute_geodesic_mesh_for_gaussians.py"
    assert compute_script.exists(), f"Not found: {compute_script}"

    configs = generate_all_configs()
    print(f"Generated {len(configs)} configurations")
    print(f"  Grid edge:    {EDGE_VALUES}")
    print(f"  Area factor:  {AREA_FACTOR_VALUES}")
    print(f"  Min angle:    {MIN_ANGLE_VALUES}")

    sweep_dir = Path(args.output_dir) / "geodesic_refine_sweep"
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
    results_file = sweep_dir / "refine_sweep_results.json"
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    # Print sorted by d3
    print(f"\n{'='*140}")
    print(f"  RESULTS SORTED BY deg_le_3 (MMP inf proxy)")
    print(f"{'='*140}")
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

    # Composite score
    print(f"\n{'='*100}")
    print(f"  RESULTS SORTED BY COMPOSITE SCORE (lower = better)")
    print(f"  Score = 0.3*%bad + 2*%vbad + 0.01*d3 + 0.5*(verts/100K)")
    print(f"{'='*100}")
    for r in sorted_results:
        r["score"] = (r["pct_bad"] * 0.3
                      + r["pct_very_bad"] * 2
                      + r["deg_le_3"] * 0.01
                      + (r["n_vertices"] / 100000) * 0.5)

    scored = sorted(results.values(), key=lambda r: r.get("score", 99999))
    print(f"{'Config':<42} {'Score':>6} {'Verts':>8} "
          f"{'%Bad':>6} {'%VBad':>6} {'D≤3':>5} {'G%Bad':>6}")
    print("-" * 80)
    for r in scored[:20]:
        print(f"{r['name']:<42} {r['score']:>6.2f} {r['n_vertices']:>8} "
              f"{r['pct_bad']:>6.1f} {r['pct_very_bad']:>6.2f} "
              f"{r['deg_le_3']:>5} {r['gauss_bad_pct']:>6.1f}")

    # Summary by axis
    print(f"\n{'='*80}")
    print(f"  AXIS SUMMARY: mean d3 by each parameter value")
    print(f"{'='*80}")

    for axis_name, axis_vals, key_fn in [
        ("grid_edge", EDGE_VALUES,
         lambda n: float(n.split("_")[0].replace("e", ""))),
        ("area_factor", AREA_FACTOR_VALUES,
         lambda n: float(n.split("_")[1].replace("af", ""))),
        ("min_angle", MIN_ANGLE_VALUES,
         lambda n: int(n.split("_")[2].replace("ma", ""))),
    ]:
        print(f"\n  {axis_name}:")
        for val in axis_vals:
            matching = [r for r in results.values()
                        if abs(key_fn(r["name"]) - val) < 0.0001]
            if matching:
                mean_d3 = sum(r["deg_le_3"] for r in matching) / len(matching)
                mean_bad = sum(r["pct_bad"] for r in matching) / len(matching)
                mean_verts = sum(r["n_vertices"] for r in matching) / len(matching)
                print(f"    {val:>8} -> mean d3={mean_d3:>7.1f}  "
                      f"mean %bad={mean_bad:>5.1f}  "
                      f"mean V={mean_verts:>10.0f}  (n={len(matching)})")

    print(f"\nFull results: {results_file}")


if __name__ == "__main__":
    main()
