#!/usr/bin/env python3
"""
Summarize mesh_statistics.json files across all polynomial surfaces/levels/lights
into a single CSV table + a printed console summary.

Usage:
    python scripts/polynomial/geodesic/summarize_mesh_statistics.py [--data_root PATH] [--output PATH]

Defaults:
    --data_root  TrainData/Polynomial/SyntheticColmapData/blue_texture
    --output     <data_root>/mesh_summary.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _safe(d, *keys, default=""):
    """Safely traverse nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def collect_rows(data_root):
    rows = []
    root = Path(data_root)
    for stats_path in sorted(root.rglob("mesh_statistics.json")):
        # Only include files directly inside geodesic_mesh/, skip backups
        if stats_path.parent.name != "geodesic_mesh":
            continue
        try:
            with open(stats_path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Parse path to extract surface/level/light
        rel = stats_path.relative_to(root)
        parts = rel.parts  # e.g. Paraboloid/level_04/light_0/output/geodesic_mesh/mesh_statistics.json
        surface = parts[0] if len(parts) > 0 else "?"
        level = parts[1].replace("level_", "") if len(parts) > 1 else "?"
        light = parts[2].replace("light_", "") if len(parts) > 2 else "?"

        ov = d.get("overall", {})
        mq = d.get("mesh_quality", {})
        gt = d.get("gaussian_triangles", {})

        row = {
            "surface": surface,
            "level": level,
            "light": light,
            "n_vertices": d.get("n_vertices", ""),
            "n_faces": d.get("n_faces", ""),
            "n_gaussians": d.get("n_gaussians", ""),
            # Aspect ratio
            "ar_median": _safe(ov, "aspect_ratio", "median"),
            "ar_max": _safe(ov, "aspect_ratio", "max"),
            "ar_pct_lt3": _safe(ov, "aspect_ratio", "pct_lt_3"),
            # Min angle
            "angle_median": _safe(ov, "min_angle_deg", "median"),
            "angle_min": _safe(ov, "min_angle_deg", "min"),
            "angle_pct_gt20": _safe(ov, "min_angle_deg", "pct_gt_20"),
            "angle_pct_lt5": _safe(ov, "min_angle_deg", "pct_lt_5"),
            # Edge length
            "edge_min": _safe(ov, "edge_length", "min"),
            "edge_median": _safe(ov, "edge_length", "median"),
            "edge_max": _safe(ov, "edge_length", "max"),
            # Radius ratio
            "rr_median": _safe(mq, "radius_ratio", "median"),
            "rr_pct_gt05": _safe(mq, "radius_ratio", "pct_gt_0.5"),
            # Valence
            "val_min": _safe(mq, "valence", "min"),
            "val_median": _safe(mq, "valence", "median"),
            "val_max": _safe(mq, "valence", "max"),
            "val_n1": _safe(mq, "valence", "n_valence_1"),
            "val_n2": _safe(mq, "valence", "n_valence_2"),
            # Holes & area outliers
            "holes": _safe(mq, "holes"),
            "area_10x": _safe(mq, "area_outliers", "n_gt_10x_median"),
            "area_10x_pct": _safe(mq, "area_outliers", "pct_gt_10x_median"),
            "area_50x": _safe(mq, "area_outliers", "n_gt_50x_median"),
            # Degenerate
            "degen_lt1": _safe(mq, "degenerate_triangles", "min_angle_lt_1deg"),
            "degen_lt5": _safe(mq, "degenerate_triangles", "min_angle_lt_5deg"),
            "degen_lt10": _safe(mq, "degenerate_triangles", "min_angle_lt_10deg"),
            # Topology (new)
            "non_manifold": _safe(mq, "non_manifold_edges"),
            "dup_faces": _safe(mq, "duplicate_faces"),
            "components": _safe(mq, "connected_components"),
            # Gaussian triangles
            "gauss_tri_count": _safe(gt, "count"),
            "gauss_ar_median": _safe(gt, "aspect_ratio", "median"),
            "gauss_angle_min": _safe(gt, "min_angle_deg", "min"),
            # Projected-to-mesh distance (renamed from gaussian_mesh_distance)
            "pmd_mean": _safe(d, "projected_to_mesh_distance", "mean"),
            "pmd_max": _safe(d, "projected_to_mesh_distance", "max"),
            "pmd_pct_exact": _safe(d, "projected_to_mesh_distance", "pct_exact_match"),
            # Gauss-to-projected distance (new)
            "g2p_mean": _safe(d, "gauss_to_proj_distance", "mean"),
            "g2p_max": _safe(d, "gauss_to_proj_distance", "max"),
            # Vertex projection (renamed from gaussian_projection)
            "vp_z_max": _safe(d, "vertex_projection", "all_vertex_z_residual", "max"),
            "vp_n_mismatch": _safe(d, "vertex_projection", "n_vertex_proj_mismatch"),
            # Curvature
            "K_median": _safe(d, "curvature", "gaussian_K", "median"),
            "K_max": _safe(d, "curvature", "gaussian_K", "max"),
            # Timing
            "elapsed_s": d.get("elapsed_seconds", ""),
        }
        rows.append(row)
    return rows


def print_table(rows):
    """Print a compact console table."""
    if not rows:
        print("  No mesh_statistics.json files found.")
        return

    # Group by surface
    surfaces = sorted(set(r["surface"] for r in rows))
    header = (
        f"{'Surface':<25} {'Lvl':>3} {'Lt':>2} | "
        f"{'Verts':>8} {'Faces':>8} {'Gauss':>7} | "
        f"{'AR med':>6} {'AR max':>6} | "
        f"{'Ang med':>7} {'Ang min':>7} | "
        f"{'Val min':>7} {'V1':>3} {'V2':>3} | "
        f"{'Holes':>5} {'A>10x':>5} | "
        f"{'NM':>3} {'Dup':>3} {'Comp':>4} | "
        f"{'GMD mean':>8} {'GMD%':>5} {'G2P':>6} | "
        f"{'Time':>6}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  MESH STATISTICS SUMMARY")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for surf in surfaces:
        surface_rows = sorted(
            [r for r in rows if r["surface"] == surf],
            key=lambda r: (r["level"], r["light"]),
        )
        for r in surface_rows:
            elapsed = r["elapsed_s"]
            time_str = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
            print(
                f"{r['surface']:<25} {r['level']:>3} {r['light']:>2} | "
                f"{_fmt(r['n_vertices']):>8} {_fmt(r['n_faces']):>8} {_fmt(r['n_gaussians']):>7} | "
                f"{_fmtf(r['ar_median'], 3):>6} {_fmtf(r['ar_max'], 1):>6} | "
                f"{_fmtf(r['angle_median'], 2):>7} {_fmtf(r['angle_min'], 2):>7} | "
                f"{_fmtf(r['val_min'], 0):>7} {_fmt(r['val_n1']):>3} {_fmt(r['val_n2']):>3} | "
                f"{_fmt(r['holes']):>5} {_fmt(r['area_10x']):>5} | "
                f"{_fmt(r['non_manifold']):>3} {_fmt(r['dup_faces']):>3} {_fmt(r['components']):>4} | "
                f"{_fmtf(r.get('pmd_mean', r.get('gmd_mean', '')), 2):>8} "
                f"{_fmtf(r.get('pmd_pct_exact', r.get('gmd_pct_exact', '')), 1):>5} "
                f"{_fmtf(r.get('g2p_max', ''), 4):>6} | "
                f"{time_str:>6}"
            )
        print(sep)

    # Aggregate warnings
    issues = []
    for r in rows:
        tag = f"{r['surface']}/level_{r['level']}/light_{r['light']}"
        v1 = r.get("val_n1", 0) or 0
        holes = r.get("holes", 0) or 0
        a10x = r.get("area_10x_pct", 0) or 0
        nm = r.get("non_manifold", 0) or 0
        dup = r.get("dup_faces", 0) or 0
        comp = r.get("components", 1) or 1
        if isinstance(nm, (int, float)) and nm > 0:
            issues.append(f"  WARNING: {tag} has {nm} non-manifold edges")
        if isinstance(dup, (int, float)) and dup > 0:
            issues.append(f"  WARNING: {tag} has {dup} duplicate faces")
        if isinstance(comp, (int, float)) and comp > 1:
            issues.append(f"  WARNING: {tag} has {comp} connected components")
        if isinstance(v1, (int, float)) and v1 > 0:
            issues.append(f"  WARNING: {tag} has {v1} valence-1 vertices")
        if isinstance(holes, (int, float)) and holes > 0:
            issues.append(f"  WARNING: {tag} has {holes} holes")
        if isinstance(a10x, (int, float)) and a10x > 2.0:
            issues.append(f"  WARNING: {tag} has {a10x:.1f}% area outliers (>10x median)")
        vp_mm = r.get("vp_n_mismatch", 0) or 0
        if isinstance(vp_mm, (int, float)) and vp_mm > 0:
            issues.append(f"  WARNING: {tag} has {vp_mm} vertices off-surface")

    if issues:
        print("\n  WARNINGS:")
        for i in issues:
            print(i)
    else:
        print("\n  All meshes PASS quality checks.")


def _fmt(v):
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.0f}" if v == int(v) else f"{v}"
    return str(v)


def _fmtf(v, decimals):
    if v == "" or v is None:
        return "-"
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def save_png(rows, png_path):
    """Save summary table as a PNG image using matplotlib."""
    import io
    from contextlib import redirect_stdout

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Capture console output
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_table(rows)
    text = buf.getvalue()

    lines = text.split("\n")
    n_lines = len(lines)
    max_len = max((len(l) for l in lines), default=80)

    fig_w = max(14, max_len * 0.072)
    fig_h = max(4, n_lines * 0.22)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="left",
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  PNG saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize mesh statistics")
    parser.add_argument(
        "--data_root",
        default="TrainData/Polynomial/SyntheticColmapData/blue_texture",
        help="Root directory containing surface subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <data_root>/mesh_summary.csv)",
    )
    parser.add_argument(
        "--png",
        default=None,
        help="Output PNG path (default: <data_root>/mesh_summary.png)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"Error: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    rows = collect_rows(data_root)

    # Print to console
    print_table(rows)

    # Write CSV
    output = args.output or os.path.join(data_root, "mesh_summary.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV saved to: {output}")
        print(f"  Total meshes: {len(rows)}")

    # Write PNG
    png_path = args.png or os.path.join(data_root, "mesh_summary.png")
    if rows:
        save_png(rows, png_path)


if __name__ == "__main__":
    main()
