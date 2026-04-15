#!/usr/bin/env python3
"""
Autotune geodesic mesh parameters for a single Gaussian output.

Tries multiple parameter configurations in parallel, evaluates mesh quality,
selects the best config, and manages backups for easy switching.

Based on sweeps of 63 + 64 parameter combinations, the key findings are:
  - "light" refinement (15 iter, warmup=10, ring=1) dominates
  - Edge=0.006 gives lowest d3 (MMP inf proxy) at cost of more vertices
  - Higher curvature_alpha slightly helps d3 but costs more vertices
  - refine_min_angle=20 + gauss_min_angle=30 improves d3 (88 vs 103)
  - Area factor has negligible effect on d3
  - nocurv and "heavy"/"nowarm" refinement are strictly worse

Usage:
    # Build & pick best mesh (6 configs tried in parallel):
    python GenerateData/autotune_geodesic_mesh.py \\
        --output_dir <gaussian_output> --surface Paraboloid --workers 6

    # List available backups:
    python GenerateData/autotune_geodesic_mesh.py \\
        --output_dir <gaussian_output> --list_backups

    # Switch active mesh to a specific backup:
    python GenerateData/autotune_geodesic_mesh.py \\
        --output_dir <gaussian_output> --switch a2.00_e0.006_light

Backup structure:
    <output_dir>/geodesic_mesh/          # active mesh
    <output_dir>/geodesic_mesh_backups/  # all tried configs
        <config_name>/                   # one per config tried
            geodesic_mesh.ply
            geodesic_mesh_data.npz
            ...
        autotune_results.json            # metrics + active config name
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _nfs_safe_rmtree(path):
    """Remove a directory tree, tolerating NFS stale handles (.nfs* files).

    On NFS, open files get renamed to .nfs* and cannot be deleted until the
    handle is closed.  shutil.rmtree raises OSError on these.  We first try
    the normal rmtree; if it fails we rename the dir to a random temp name
    (same filesystem) and do a best-effort cleanup — any leftover .nfs*
    files will be garbage-collected by NFS eventually.
    """
    path = str(path)
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        # Rename to temp dir on the same filesystem, then best-effort remove
        try:
            parent = os.path.dirname(path)
            tmp = tempfile.mkdtemp(prefix=".nfs_cleanup_", dir=parent)
            os.rename(path, os.path.join(tmp, "old"))
            shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            shutil.rmtree(path, ignore_errors=True)

# ---------------------------------------------------------------------------
# Autotune parameter grid (from 63+64 sweep combinations)
#
# All use "light" refinement: iter=15, warmup=10, ring_fix=1, patience=4.
# refine_min_angle=20, gauss_min_angle=30 (best from sweep 2).
# Configs span edge_length × curvature_alpha for a Pareto front of
# {d3 (MMP-inf proxy), vertex count, %bad triangles}.
# ---------------------------------------------------------------------------
AUTOTUNE_CONFIGS = [
    # (name,                 alpha,  edge,   curvature_adaptive)
    ("a0.75_e0.008",         0.75,   0.008,  True),   # compact
    ("a1.50_e0.008",         1.50,   0.008,  True),   # moderate
    ("a0.75_e0.006",         0.75,   0.006,  True),   # balanced
    ("a1.50_e0.006",         1.50,   0.006,  True),   # quality
    ("a2.00_e0.006",         2.00,   0.006,  True),   # best d3
    ("a2.00_e0.005",         2.00,   0.005,  True),   # finest grid
]

# Refinement constants (light preset + best angle from sweeps)
REFINE_ITER = "15"
REFINE_WARMUP = "10"
REFINE_RING = "2"
REFINE_PATIENCE = "4"
GAUSS_EDGE_RATIO = 0.625   # gauss_edge = edge × ratio
REFINE_MIN_ANGLE = "20"
REFINE_GAUSS_MIN_ANGLE = "30"


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def make_build_args(alpha, edge, curvature):
    """Build CLI arguments for compute_geodesic_mesh_for_gaussians.py."""
    gauss_edge = f"{edge * GAUSS_EDGE_RATIO:.4f}"
    args = [
        "--max_edge_length", f"{edge}",
        "--mesh_method", "grid",
        "--refine",
        "--refine_iterations", REFINE_ITER,
        "--refine_warmup_iterations", REFINE_WARMUP,
        "--refine_gauss_max_edge_length", gauss_edge,
        "--refine_max_area_factor", "2",
        "--refine_gauss_max_area_factor", "1.5",
        "--refine_min_angle", REFINE_MIN_ANGLE,
        "--refine_gauss_min_angle", REFINE_GAUSS_MIN_ANGLE,
        "--refine_ring_fix",
        "--refine_ring_fix_iterations", REFINE_RING,
        "--refine_delaunay_flip",
        "--refine_patience", REFINE_PATIENCE,
        "--local_refinement",
    ]
    if curvature:
        args += ["--curvature_adaptive", "--curvature_alpha", f"{alpha}"]
    return args


def build_one(config_name, build_args, gaussian_output, surface,
              backups_dir, compute_script):
    """Build mesh for one config.  Returns quality dict or error dict."""
    mesh_dest = Path(backups_dir) / config_name
    shadow_output = Path(backups_dir) / f".shadow_{config_name}"

    try:
        return _build_one_inner(config_name, build_args, gaussian_output,
                                surface, mesh_dest, shadow_output,
                                compute_script)
    except Exception as exc:
        _nfs_safe_rmtree(shadow_output)
        return {"name": config_name, "error": str(exc)}


def _build_one_inner(config_name, build_args, gaussian_output, surface,
                     mesh_dest, shadow_output, compute_script):
    """Inner build logic — separated so build_one can catch all exceptions."""

    # Clean up any leftover shadow dir from a previous failed run
    _nfs_safe_rmtree(shadow_output)
    shadow_output.mkdir(parents=True, exist_ok=True)

    real_output = Path(gaussian_output)
    for item in real_output.iterdir():
        if item.name in ("geodesic_mesh", "geodesic_mesh_backups",
                         "geodesic_mesh_sweep"):
            continue
        if item.name.startswith(".nfs"):
            continue
        link = shadow_output / item.name
        # Remove stale symlinks (exists() follows symlinks; broken link → False)
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except OSError:
                continue
        try:
            os.symlink(str(item.resolve()), str(link))
        except OSError:
            pass  # source vanished or other transient NFS issue

    shadow_mesh = shadow_output / "geodesic_mesh"
    if shadow_mesh.is_symlink():
        shadow_mesh.unlink()
    elif shadow_mesh.is_dir():
        _nfs_safe_rmtree(shadow_mesh)

    cmd = [
        sys.executable, str(compute_script),
        "--gaussian_output", str(shadow_output),
        "--surface", surface,
        "--seed", "42",
    ] + build_args

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        _nfs_safe_rmtree(shadow_output)
        return {"name": config_name, "error": "timeout"}
    elapsed = time.time() - t0

    if result.returncode != 0:
        _nfs_safe_rmtree(shadow_output)
        return {"name": config_name, "error": result.stderr[-500:],
                "elapsed": round(elapsed, 1)}

    # Move built mesh to backup location
    if shadow_mesh.exists():
        _nfs_safe_rmtree(mesh_dest)
        shutil.copytree(str(shadow_mesh), str(mesh_dest))
        _nfs_safe_rmtree(shadow_mesh)

    _nfs_safe_rmtree(shadow_output)

    quality = analyse_mesh(mesh_dest)
    if quality:
        quality["elapsed"] = round(elapsed, 1)
        quality["name"] = config_name
    return quality


def analyse_mesh(mesh_dir):
    """Compute quality metrics from a built mesh."""
    import numpy as np

    npz_path = Path(mesh_dir) / "geodesic_mesh_data.npz"
    if not npz_path.exists():
        return None

    # Ensure GenerateData/ is importable
    gen_dir = str(Path(__file__).resolve().parent)
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)
    from utils.geodesic_mesh_utils import _triangle_quality

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
        "pct_bad": round(100 * bad.sum() / len(faces), 2),
        "pct_very_bad": round(100 * vbad.sum() / len(faces), 3),
        "deg_le_3": int((deg == 3).sum()),
        "deg_le_4": int((deg <= 4).sum()),
        "pct_deg_le_4": round(100 * (deg <= 4).sum() / len(deg), 2),
        "gauss_bad_pct": round(100 * (bad & gf).sum() / max(gf.sum(), 1), 2),
        "worst_ar": round(float(ar.max()), 2),
        "p99_ar": round(float(np.percentile(ar, 99)), 2),
        "p1_min_angle": round(float(np.percentile(ma, 1)), 2),
    }


# ---------------------------------------------------------------------------
# Selection — rank-based across all factors
# ---------------------------------------------------------------------------

def select_best(results, metric="rank"):
    """Select best config considering all factors via rank-based scoring.

    For each metric, configs are ranked 0..N-1 (lower=better).
    Final score = weighted sum of ranks. Weights:
        d3 (deg_le_3):    3  — MMP inf proxy, critical for geodesic quality
        n_vertices:       1  — fewer = faster downstream computation
        pct_bad:          2  — overall triangle quality
        gauss_bad_pct:    2  — Gaussian-touching triangle quality

    Falls back to pure d3 if metric="d3" or pure composite if metric="composite".
    """
    valid = [r for r in results.values() if "error" not in r]
    if not valid:
        return None

    if len(valid) == 1:
        return valid[0]

    if metric == "d3":
        return min(valid, key=lambda r: r["deg_le_3"])

    if metric == "composite":
        for r in valid:
            r["score"] = (r["pct_bad"] * 0.3
                          + r["pct_very_bad"] * 2
                          + r["deg_le_3"] * 0.01
                          + (r["n_vertices"] / 100000) * 0.5)
        return min(valid, key=lambda r: r["score"])

    # --- rank-based (default) ---
    rank_metrics = [
        ("deg_le_3",      3),   # MMP inf proxy — most important
        ("pct_bad",       2),   # triangle quality
        ("gauss_bad_pct", 2),   # Gaussian-triangle quality
        ("n_vertices",    1),   # fewer vertices = faster
    ]

    for r in valid:
        r["rank_score"] = 0.0

    for key, weight in rank_metrics:
        ordered = sorted(valid, key=lambda r: r.get(key, 1e9))
        for rank, r in enumerate(ordered):
            r["rank_score"] += rank * weight

    best = min(valid, key=lambda r: r["rank_score"])
    return best


# ---------------------------------------------------------------------------
# Backup management
# ---------------------------------------------------------------------------

def activate_config(output_dir, backups_dir, config_name):
    """Copy a backup config as the active geodesic_mesh."""
    src = Path(backups_dir) / config_name
    dst = Path(output_dir) / "geodesic_mesh"

    if not src.exists():
        print(f"  Error: backup not found: {src}")
        return False

    if dst.exists():
        _nfs_safe_rmtree(dst)
    shutil.copytree(str(src), str(dst))

    # Update active marker in results JSON
    results_file = Path(backups_dir) / "autotune_results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        data["active"] = config_name
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
    return True


def list_backups(output_dir):
    """Print available mesh backups."""
    backups_dir = Path(output_dir) / "geodesic_mesh_backups"
    if not backups_dir.exists():
        print("No backups found.")
        return

    results_file = backups_dir / "autotune_results.json"
    active = None
    results = {}
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        active = data.get("active")
        results = data.get("results", {})

    configs = sorted([d.name for d in backups_dir.iterdir()
                      if d.is_dir() and not d.name.startswith(".")])
    if not configs:
        print("No backups found.")
        return

    print(f"\nBackups in {backups_dir}:")
    print(f"  {'Config':<35} {'Active':>7} {'Verts':>8} {'%Bad':>6} "
          f"{'D≤3':>5} {'G%Bad':>6}")
    print(f"  {'-'*70}")
    for name in configs:
        marker = "  >>>" if name == active else ""
        r = results.get(name, {})
        v = r.get("n_vertices", "?")
        b = r.get("pct_bad", "?")
        d = r.get("deg_le_3", "?")
        g = r.get("gauss_bad_pct", "?")
        print(f"  {name:<35} {marker:>7} {v:>8} {b:>6} {d:>5} {g:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autotune geodesic mesh parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True,
                        help="Gaussian output directory")
    parser.add_argument("--surface", default=None,
                        help="Surface type (required for build mode)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel workers (default: 6 = one per config)")
    parser.add_argument("--metric", choices=["rank", "d3", "composite"],
                        default="rank",
                        help="Selection metric: rank (default, balanced), d3 (min deg_le_3), composite")
    parser.add_argument("--list_backups", action="store_true",
                        help="List available backups and exit")
    parser.add_argument("--switch", metavar="CONFIG",
                        help="Switch active mesh to named backup")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if autotune_results.json already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    backups_dir = output_dir / "geodesic_mesh_backups"

    # --- List mode ---
    if args.list_backups:
        list_backups(str(output_dir))
        return

    # --- Switch mode ---
    if args.switch:
        if activate_config(str(output_dir), str(backups_dir), args.switch):
            print(f"  Switched active mesh to: {args.switch}")
        return

    # --- Build mode ---
    if not args.surface:
        parser.error("--surface is required for build mode")

    if args.skip_existing and (backups_dir / "autotune_results.json").exists():
        print(f"  Skipping (already autotuned): {output_dir}")
        return
    elif args.skip_existing:
        print(f"  DEBUG: NOT skipping: {backups_dir / 'autotune_results.json'} exists={os.path.exists(str(backups_dir / 'autotune_results.json'))}")

    compute_script = Path(__file__).resolve().parent / \
        "compute_geodesic_mesh_for_gaussians.py"
    if not compute_script.exists():
        print(f"Error: compute script not found: {compute_script}")
        sys.exit(1)

    backups_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Autotuning mesh for: {output_dir}")
    print(f"  Surface: {args.surface}  |  Configs: {len(AUTOTUNE_CONFIGS)}"
          f"  |  Workers: {args.workers}  |  Metric: {args.metric}")

    if args.dry_run:
        for name, alpha, edge, curv in AUTOTUNE_CONFIGS:
            print(f"    {name}")
        return

    # Build all configs in parallel
    results = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for name, alpha, edge, curv in AUTOTUNE_CONFIGS:
            build_args = make_build_args(alpha, edge, curv)
            f = pool.submit(
                build_one, name, build_args, str(output_dir), args.surface,
                str(backups_dir), str(compute_script),
            )
            futures[f] = name

        for f in as_completed(futures):
            name = futures[f]
            try:
                res = f.result()
                if res and "error" not in res:
                    results[name] = res
                    print(f"    {name:35s} V={res['n_vertices']:>8} "
                          f"bad={res['pct_bad']:5.1f}% "
                          f"d3={res['deg_le_3']:>4}  "
                          f"({res['elapsed']:.0f}s)")
                else:
                    err = res.get("error", "?") if res else "null"
                    print(f"    {name:35s} FAILED: {str(err)[:60]}")
            except Exception as e:
                print(f"    {name:35s} EXCEPTION: {e}")

    total_time = time.time() - t0
    print(f"\n  Completed {len(results)}/{len(AUTOTUNE_CONFIGS)} "
          f"in {total_time:.0f}s")

    if not results:
        print("  ERROR: All configs failed!")
        sys.exit(1)

    # Select and activate best
    best = select_best(results, args.metric)
    rank_info = f" rank_score={best.get('rank_score', '?')}" if 'rank_score' in best else ""
    print(f"\n  >>> Best: {best['name']}  "
          f"(d3={best['deg_le_3']}, V={best['n_vertices']}, "
          f"bad={best['pct_bad']}%, gbad={best['gauss_bad_pct']}%{rank_info})")

    activate_config(str(output_dir), str(backups_dir), best["name"])

    # Save results
    save_data = {
        "active": best["name"],
        "metric": args.metric,
        "surface": args.surface,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 1),
        "results": results,
    }
    results_file = backups_dir / "autotune_results.json"
    with open(results_file, "w") as fh:
        json.dump(save_data, fh, indent=2)

    print(f"  Active mesh: {output_dir}/geodesic_mesh/")
    print(f"  Backups:     {backups_dir}/")
    print(f"  Results:     {results_file}")


if __name__ == "__main__":
    main()
