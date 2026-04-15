#!/usr/bin/env python3
"""
Benchmark geodesic mesh building on real Gaussian splatting outputs.

Discovers all Gaussian outputs under SyntheticColmapData (same structure
as build_geodesic_mesh_polynomial_all.sh), builds meshes with several
max_edge_length values using the full pipeline from
compute_geodesic_mesh_for_gaussians.py, and reports vertex/face counts
+ quality metrics.

Goal: find max_edge_length that produces meshes with ~700K-1M vertices.

Usage:
    # Default: sweep all outputs with edge_lengths [0.005..0.010]
    python scripts/polynomial/geodesic/benchmark_refine.py

    # Specify workers:
    python scripts/polynomial/geodesic/benchmark_refine.py --workers 60

    # Custom edge lengths:
    python scripts/polynomial/geodesic/benchmark_refine.py --edge_lengths 0.006,0.008

    # Only one surface and level:
    python scripts/polynomial/geodesic/benchmark_refine.py --surfaces Paraboloid --levels 02
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
_generate_data_dir = str(project_root / "GenerateData")
if _generate_data_dir not in sys.path:
    sys.path.insert(0, _generate_data_dir)

# ---------------------------------------------------------------------------
# Default edge lengths to sweep (target: 700K-1M vertices)
# ---------------------------------------------------------------------------
_DEFAULT_EDGE_LENGTHS = [0.005, 0.006, 0.007, 0.008, 0.010]
_DEFAULT_RING_STEPS = [1, 2]

# Fixed refine parameters -- same as build_geodesic_mesh_polynomial_all.sh
_REFINE_PARAMS = dict(
    max_aspect_ratio=1e6,
    min_angle_deg=20.0,
    max_area_factor=3.0,
    gauss_max_aspect_ratio=1e6,
    gauss_min_angle_deg=30.0,
    warmup_iterations=10,
    ring_fix_invalid_gaussians=True,
    patience=4,
    max_iterations=15,
    delaunay_flip_polish=True,
    surface_aware=False,
    # Extreme cleanup params
    extreme_cleanup_passes=3,
    extreme_ar_threshold=20.0,
    extreme_min_angle_deg=5.0,
)


# ---------------------------------------------------------------------------
# Discovery: find all Gaussian outputs
# ---------------------------------------------------------------------------

def discover_outputs(synth_data_base, surfaces, textures, levels, light_ids):
    """Find all Gaussian outputs matching the directory structure."""
    outputs = []
    base = Path(synth_data_base)
    for texture in textures:
        for surface in surfaces:
            for level in levels:
                for light_id in light_ids:
                    path = (base / f"{texture}_texture" / surface
                            / f"level_{level}" / f"light_{light_id}" / "output")
                    if (path / "point_cloud").is_dir():
                        outputs.append(dict(
                            path=str(path),
                            surface=surface,
                            level=level,
                            light_id=light_id,
                            label=f"{texture}/{surface}/L{level}/light_{light_id}",
                        ))
    return outputs


# ---------------------------------------------------------------------------
# Worker: build full mesh for one (output, edge_length) pair
# ---------------------------------------------------------------------------

def _run_one(args):
    """Full pipeline: load Gaussians -> grid -> insert -> refine -> metrics."""
    label, gaussian_output, surface, edge_length, ring_fix_iters, refine_params = args

    import time as _time
    import traceback as _tb
    import sys as _sys
    import numpy as _np
    from pathlib import Path as _Path

    # Ensure project root + GenerateData/ are on sys.path in worker (fork-safe)
    _proj = str(_Path(__file__).resolve().parent.parent.parent.parent)
    _gen = str(_Path(_proj) / "GenerateData")
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)
    if _gen not in _sys.path:
        _sys.path.insert(0, _gen)

    from utils.load_utils import load_gaussian_data_cpu
    from GenerateData.utils.geodesic_mesh_utils import (
        project_gaussians_to_surface,
        build_surface_grid,
        insert_gaussians_into_grid_mesh,
        refine_bad_triangles,
        _triangle_quality,
        _filter_boundary_long_edges,
    )

    config_name = f"e{edge_length:.3f}_r{ring_fix_iters}"
    try:
        t_start = _time.time()

        # 1. Load Gaussians
        t0 = _time.time()
        gd = load_gaussian_data_cpu(Path(gaussian_output))
        positions = gd.get_xyz()
        n_gauss = len(positions)
        dt_load = _time.time() - t0

        # 2. Project onto surface
        projected = project_gaussians_to_surface(positions, surface)

        # 3. Domain bounds (same as compute_geodesic_mesh_for_gaussians.py)
        margin = 0.05
        dx = _np.ptp(projected[:, 0])
        x_range = (float(projected[:, 0].min() - margin * dx),
                   float(projected[:, 0].max() + margin * dx))
        dy = _np.ptp(projected[:, 1])
        y_range = (float(projected[:, 1].min() - margin * dy),
                   float(projected[:, 1].max() + margin * dy))

        # 4. Build grid
        t0 = _time.time()
        grid_verts, grid_faces = build_surface_grid(
            surface_type=surface,
            x_range=x_range,
            y_range=y_range,
            target_edge_length=edge_length,
            curvature_adaptive=True,
            curvature_alpha=2.0,
        )
        dt_grid = _time.time() - t0
        n_grid_verts = len(grid_verts)

        # 5. Insert Gaussians
        t0 = _time.time()
        vertices, faces, gauss_idx = insert_gaussians_into_grid_mesh(
            grid_verts, grid_faces, projected, surface,
            local_refinement=True,
        )
        dt_insert = _time.time() - t0
        n_pre_verts = len(vertices)

        # 6. Filter boundary long edges
        max_edge_threshold = edge_length * 3.0
        faces, n_removed = _filter_boundary_long_edges(
            vertices, faces, max_edge_threshold,
        )

        # 7. Compact orphan vertices
        used_verts = _np.unique(faces.ravel())
        if len(used_verts) < len(vertices):
            new_idx = _np.full(len(vertices), -1, dtype=_np.int32)
            new_idx[used_verts] = _np.arange(len(used_verts), dtype=_np.int32)
            vertices = vertices[used_verts]
            faces = new_idx[faces]
            gauss_idx = new_idx[gauss_idx.astype(_np.int64)]
            gauss_idx = gauss_idx[gauss_idx >= 0]

        # 8. Refine bad triangles
        t0 = _time.time()
        gauss_edge = edge_length * 0.75
        gauss_area_factor = 1.5
        vertices, faces, gauss_idx = refine_bad_triangles(
            vertices, faces, surface,
            gaussian_vertex_indices=gauss_idx,
            max_edge_length=max_edge_threshold,
            gauss_max_edge_length=gauss_edge,
            gauss_max_area_factor=gauss_area_factor,
            ring_fix_iterations=ring_fix_iters,
            verbose=False,
            **refine_params,
        )
        dt_refine = _time.time() - t0
        dt_total = _time.time() - t_start

        # 9. Quality metrics
        ar, ma, le = _triangle_quality(vertices, faces)
        gauss_mask = _np.zeros(len(vertices), dtype=bool)
        gauss_mask[gauss_idx.astype(int)] = True
        gf = gauss_mask[faces].any(axis=1)
        bad = (ar > 2) | (ma < 20)
        vbad = (ar > 5) | (ma < 10)

        return dict(
            config=config_name,
            label=label,
            gaussian_output=gaussian_output,
            surface=surface,
            edge_length=edge_length,
            ring_fix_iterations=ring_fix_iters,
            status="ok",
            # Sizes
            n_gaussians=n_gauss,
            n_grid_verts=int(n_grid_verts),
            n_pre_refine_verts=int(n_pre_verts),
            n_vertices=int(len(vertices)),
            n_faces=int(len(faces)),
            verts_added_by_refine=int(len(vertices) - n_pre_verts),
            # Input params
            gauss_max_edge_length=gauss_edge,
            gauss_max_area_factor=gauss_area_factor,
            # Timing
            time_load_s=round(dt_load, 2),
            time_grid_s=round(dt_grid, 2),
            time_insert_s=round(dt_insert, 2),
            time_refine_s=round(dt_refine, 2),
            time_total_s=round(dt_total, 2),
            # Triangle quality
            pct_bad=round(100 * float(bad.sum()) / len(faces), 2),
            pct_very_bad=round(100 * float(vbad.sum()) / len(faces), 2),
            worst_ar=round(float(ar.max()), 2),
            p99_ar=round(float(_np.percentile(ar, 99)), 3),
            p95_ar=round(float(_np.percentile(ar, 95)), 3),
            median_ar=round(float(_np.median(ar)), 3),
            min_angle=round(float(ma.min()), 2),
            p1_min_angle=round(float(_np.percentile(ma, 1)), 2),
            p5_min_angle=round(float(_np.percentile(ma, 5)), 2),
            median_angle=round(float(_np.median(ma)), 2),
            # Edge lengths
            min_edge=round(float(le.min()), 6),
            median_edge=round(float(_np.median(le)), 6),
            p95_edge=round(float(_np.percentile(le, 95)), 6),
            max_edge=round(float(le.max()), 6),
            # Gaussian-triangle quality
            gauss_bad_pct=round(
                100 * float((bad & gf).sum()) / max(int(gf.sum()), 1), 2),
            gauss_vbad_pct=round(
                100 * float((vbad & gf).sum()) / max(int(gf.sum()), 1), 2),
            n_gauss_faces=int(gf.sum()),
        )
    except Exception as exc:
        return dict(
            config=config_name,
            label=label,
            gaussian_output=gaussian_output,
            surface=surface,
            edge_length=edge_length,
            ring_fix_iterations=ring_fix_iters,
            status="error",
            error=str(exc),
            traceback=_tb.format_exc(),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mesh building on real Gaussian data",
    )
    parser.add_argument("--synth_data_base",
                        default="TrainData/Polynomial/SyntheticColmapData")
    parser.add_argument("--surfaces",
                        default="Paraboloid,Saddle,HyperbolicParaboloid")
    parser.add_argument("--textures", default="blue")
    parser.add_argument("--levels", default="02,03,04")
    parser.add_argument("--light_ids", default="0,1,2,3,4")
    parser.add_argument("--edge_lengths",
                        default=",".join(str(e) for e in _DEFAULT_EDGE_LENGTHS),
                        help="Comma-separated edge lengths to sweep")
    parser.add_argument("--ring_steps",
                        default=",".join(str(r) for r in _DEFAULT_RING_STEPS),
                        help="Comma-separated ring_fix_iterations to sweep")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = nproc)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory for JSON report (default: script dir)")
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else os.cpu_count() or 4
    edge_lengths = [float(x) for x in args.edge_lengths.split(",")]
    ring_steps = [int(x) for x in args.ring_steps.split(",")]
    surfaces = args.surfaces.split(",")
    textures = args.textures.split(",")
    levels = args.levels.split(",")
    light_ids = args.light_ids.split(",")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover outputs
    outputs = discover_outputs(args.synth_data_base, surfaces, textures,
                               levels, light_ids)
    if not outputs:
        print("No Gaussian outputs found!")
        return

    total_jobs = len(outputs) * len(edge_lengths) * len(ring_steps)
    print(f"{'=' * 70}")
    print(f"  Benchmark: Real Gaussian Mesh Building")
    print(f"  Outputs:        {len(outputs)}")
    print(f"  Edge lengths:   {edge_lengths}")
    print(f"  Ring steps:     {ring_steps}")
    print(f"  Configs/output: {len(edge_lengths) * len(ring_steps)}")
    print(f"  Total jobs:     {total_jobs}")
    print(f"  Workers:        {workers}")
    print(f"  Target verts:   700,000 - 1,000,000")
    print(f"{'=' * 70}\n")

    # Build work list
    work = []
    for out in outputs:
        for edge in edge_lengths:
            for ring in ring_steps:
                work.append((
                    out["label"], out["path"], out["surface"],
                    edge, ring, _REFINE_PARAMS,
                ))

    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, w): w for w in work}
        done = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            results.append(r)
            if r["status"] == "ok":
                in_range = (
                    "OK" if 700_000 <= r["n_vertices"] <= 1_000_000
                    else "LOW" if r["n_vertices"] < 700_000
                    else "HIGH"
                )
                print(
                    f"  [{done}/{total_jobs}] {r['label']} e={r['edge_length']:.3f} "
                    f"r={r['ring_fix_iterations']}: "
                    f"verts={r['n_vertices']:>10,d} [{in_range:>4s}]  "
                    f"bad={r['pct_bad']:.1f}%  worstAR={r['worst_ar']:.1f}  "
                    f"minAngle={r['min_angle']:.2f}°  {r['time_total_s']:.0f}s"
                )
            else:
                print(
                    f"  [{done}/{total_jobs}] {r['label']} e={r['edge_length']:.3f} "
                    f"r={r.get('ring_fix_iterations', '?')}: "
                    f"ERROR -- {r.get('error', '?')}"
                )

    dt_total = time.time() - t_start

    # Sort results
    results.sort(key=lambda r: (r["surface"], r.get("label", ""), r["edge_length"], r.get("ring_fix_iterations", 0)))

    # Save JSON report
    report = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_outputs=len(outputs),
        edge_lengths=edge_lengths,
        ring_steps=ring_steps,
        refine_params={k: v for k, v in _REFINE_PARAMS.items()},
        total_jobs=total_jobs,
        workers=workers,
        total_time_s=round(dt_total, 1),
        results=results,
    )
    report_path = output_dir / "benchmark_refine_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # -- Console summary: per (surface, edge_length) aggregate ---------
    ok_results = [r for r in results if r["status"] == "ok"]
    n_errors = sum(1 for r in results if r["status"] != "ok")

    print(f"\n{'=' * 120}")
    print(f"  Summary: Average vertices per (surface, edge_length, ring_steps)")
    print(f"  Target range: 700,000 - 1,000,000 vertices")
    if n_errors:
        print(f"  WARNING: {n_errors} errors")
    print(f"{'=' * 120}")

    for surface in surfaces:
        print(f"\n  --- {surface} ---")
        hdr = (
            f"  {'Edge':>6s}  {'Ring':>4s}  {'AvgVerts':>10s}  {'MinVerts':>10s}  {'MaxVerts':>10s}  "
            f"{'AvgBad%':>7s}  {'AvgWorstAR':>10s}  {'AvgMinAng':>9s}  "
            f"{'AvgTime':>7s}  {'InRange':>8s}"
        )
        print(hdr)
        for edge in edge_lengths:
            for ring in ring_steps:
                rs = [r for r in ok_results
                      if r["surface"] == surface
                      and r["edge_length"] == edge
                      and r.get("ring_fix_iterations", 0) == ring]
                if not rs:
                    continue
                avg_v = np.mean([r["n_vertices"] for r in rs])
                min_v = min(r["n_vertices"] for r in rs)
                max_v = max(r["n_vertices"] for r in rs)
                avg_bad = np.mean([r["pct_bad"] for r in rs])
                avg_worst_ar = np.mean([r["worst_ar"] for r in rs])
                avg_min_angle = np.mean([r["min_angle"] for r in rs])
                avg_t = np.mean([r["time_total_s"] for r in rs])
                in_range = sum(1 for r in rs if 700_000 <= r["n_vertices"] <= 1_000_000)
                mark = "ALL" if in_range == len(rs) else f"{in_range}/{len(rs)}"
                print(
                    f"  {edge:>6.3f}  {ring:>4d}  {avg_v:>10,.0f}  {min_v:>10,d}  {max_v:>10,d}  "
                    f"{avg_bad:>6.1f}%  {avg_worst_ar:>10.1f}  {avg_min_angle:>8.2f}°  "
                    f"{avg_t:>6.1f}s  {mark:>8s}"
                )

    # Per-level breakdown
    print(f"\n  --- Per-level breakdown ---")
    all_levels = sorted(set(o["level"] for o in outputs))
    for surface in surfaces:
        for level in all_levels:
            rs_level = [r for r in ok_results
                        if r["surface"] == surface
                        and f"L{level}" in r.get("label", "")]
            if not rs_level:
                continue
            print(f"\n  {surface} level_{level}:")
            for edge in edge_lengths:
                for ring in ring_steps:
                    rs = [r for r in rs_level
                          if r["edge_length"] == edge
                          and r.get("ring_fix_iterations", 0) == ring]
                    if not rs:
                        continue
                    avg_v = np.mean([r["n_vertices"] for r in rs])
                    in_r = sum(1 for r in rs if 700_000 <= r["n_vertices"] <= 1_000_000)
                    avg_bad = np.mean([r["pct_bad"] for r in rs])
                    avg_worst = np.mean([r["worst_ar"] for r in rs])
                    avg_mina = np.mean([r["min_angle"] for r in rs])
                    print(
                        f"    e={edge:.3f} r={ring}  avg_verts={avg_v:>10,.0f}  "
                        f"bad={avg_bad:.1f}%  worstAR={avg_worst:.1f}  "
                        f"minAngle={avg_mina:.2f}°  in_range={in_r}/{len(rs)}"
                    )

    print(f"\nTotal wall time: {dt_total:.1f}s")


if __name__ == "__main__":
    main()
