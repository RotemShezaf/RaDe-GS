#!/usr/bin/env python3
"""Generate PNG benchmark report images with formatted tables from benchmark results.

Usage:
    python scripts/tosca/benchmark/generate_report_png.py \
        --benchmark_dir output/benchmarks/tosca_params/cat0_blue_high_res

    # Or from a sweep source path:
    python scripts/tosca/benchmark/generate_report_png.py \
        --benchmark_dir TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_0

Produces: <benchmark_dir>/report_*.png files
"""

import argparse
import json
import re
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Columns to display (order matters) ──────────────────────────────────────
# "G→Mesh" = distance from each Gaussian to the nearest point on the
#            reconstructed mesh surface (gaussian_to_recon_mesh_surface).
DISPLAY_COLS = [
    ("run_name",              "Run",           "s"),
    ("chamfer_distance",      "Chamfer",       ".5f"),
    ("test_psnr",             "PSNR",          ".2f"),
    ("g2s_mean_sq",           "G→Surf\nmean(d²)",".4f"),
    ("g2v_mean_sq",           "G→V\nmean(d²)", ".4f"),
    ("g2mesh_mean",           "G→Mesh\nmean",  ".4f"),
    ("g2mesh_p95",            "G→Mesh\np95",   ".4f"),
    ("g2mesh_max",            "G→Mesh\nmax",   ".2f"),
    ("recon_to_gt_accuracy",  "Acc\n(R→GT)",   ".5f"),
    ("gt_to_recon_completeness", "Comp\n(GT→R)", ".5f"),
    ("num_gaussians",         "#Gauss",        "d"),
    ("connected_components",  "CC",            "d"),
]


def load_rows(benchmark_dir: Path):
    """Load benchmark data from individual JSON reports + train logs.
    
    Supports both benchmark output dirs (r4_xxx/) and sweep dirs (sweep_xxx/).
    """
    rows = []
    for run_dir in sorted(benchmark_dir.iterdir()):
        report_path = run_dir / "benchmark_report.json"
        if not report_path.exists() or not run_dir.is_dir():
            continue
        with open(report_path) as f:
            report = json.load(f)

        row = {"run_name": run_dir.name}
        row["num_gaussians"] = report.get("num_gaussians", "")
        row["connected_components"] = report.get("connected_components", {}).get("num_components", "")

        if "chamfer_distance" in report:
            row["chamfer_distance"] = report["chamfer_distance"]
        # G→V mean squared distance (Gaussian to nearest mesh vertex)
        g2v = report.get("gaussian_to_recon_mesh_vertex", {})
        if g2v:
            row["g2v_mean_sq"] = g2v.get("mean_squared", "")
        # G→Mesh = distance from each Gaussian to nearest recon mesh surface
        g2ms = report.get("gaussian_to_recon_mesh_surface", {})
        if g2ms:
            row["g2s_mean_sq"] = g2ms.get("mean_squared", "")
            row["g2mesh_mean"] = g2ms.get("mean", "")
            row["g2mesh_p95"] = g2ms.get("p95", "")
            row["g2mesh_max"] = g2ms.get("max", "")
        r2gt = report.get("recon_to_gt_accuracy", {})
        if r2gt:
            row["recon_to_gt_accuracy"] = r2gt.get("mean", "")
        gt2r = report.get("gt_to_recon_completeness", {})
        if gt2r:
            row["gt_to_recon_completeness"] = gt2r.get("mean", "")

        # PSNR — from report if available, else from train.log
        psnr_info = report.get("psnr", {})
        if psnr_info and "test_psnr" in psnr_info:
            row["test_psnr"] = psnr_info["test_psnr"]
        else:
            # Fallback: parse train.log
            train_log = run_dir / "train.log"
            if train_log.exists():
                text = train_log.read_text()
                for m in re.finditer(r"Evaluating test:.*?PSNR ([0-9.e+-]+)", text):
                    row["test_psnr"] = float(m.group(1))

        rows.append(row)
    return rows


def fmt_val(val, fmt_spec):
    if val == "" or val is None:
        return ""
    try:
        return format(val, fmt_spec)
    except (ValueError, TypeError):
        return str(val)


def make_table_figure(rows, title, sort_key="chamfer_distance", ascending=True):
    """Create a matplotlib figure with a nicely formatted table."""
    # Sort rows
    def sort_fn(r):
        v = r.get(sort_key, float("inf") if ascending else float("-inf"))
        if v == "" or v is None:
            return float("inf") if ascending else float("-inf")
        return float(v) if ascending else -float(v)

    sorted_rows = sorted(rows, key=sort_fn)

    headers = [c[1] for c in DISPLAY_COLS]
    cell_text = []
    for r in sorted_rows:
        row_data = []
        for col_key, _, fmt_spec in DISPLAY_COLS:
            row_data.append(fmt_val(r.get(col_key, ""), fmt_spec))
        cell_text.append(row_data)

    n_rows = len(cell_text)
    n_cols = len(headers)

    fig_height = max(3, 1.0 + 0.35 * n_rows)
    fig_width = max(10, 1.3 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Alternating row colors
    for i in range(1, n_rows + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(color)

    # Highlight best value in each metric column (skip run_name)
    metric_cols_ascending = {"chamfer_distance", "g2s_mean_sq", "g2v_mean_sq", "g2mesh_mean", "g2mesh_p95",
                              "g2mesh_max", "recon_to_gt_accuracy", "gt_to_recon_completeness"}
    metric_cols_descending = {"test_psnr"}

    for j, (col_key, _, _) in enumerate(DISPLAY_COLS):
        if col_key in metric_cols_ascending or col_key in metric_cols_descending:
            vals = []
            for r in sorted_rows:
                v = r.get(col_key, "")
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(None)
            valid = [v for v in vals if v is not None]
            if valid:
                if col_key in metric_cols_descending:
                    best = max(valid)
                else:
                    best = min(valid)
                for i, v in enumerate(vals):
                    if v is not None and abs(v - best) < 1e-10:
                        table[i + 1, j].set_text_props(fontweight="bold", color="#2E7D32")

    fig.tight_layout()
    return fig


def make_per_param_figures(rows):
    """Create one figure per parameter showing metric trends."""
    # Detect which parameter is varied per run by prefix
    param_groups = {}
    for r in rows:
        name = r.get("run_name", "")
        prefix = name.split("_")[0] if "_" in name else ""
        if prefix not in param_groups:
            param_groups[prefix] = []
        param_groups[prefix].append(r)

    prefix_to_label = {
        "mop": "min_opacity_prune",
        "lmvg": "lambda_multi_view_geo",
        "ldist": "lambda_distortion",
        "dgt": "densify_grad_threshold",
        "bpsf": "big_point_scale_factor",
        "ldn": "lambda_depth_normal",
        "ldssim": "lambda_dssim",
        "pd": "percent_dense",
        "mvncc": "lambda_multi_view_ncc",
    }

    figures = []
    for prefix, group in sorted(param_groups.items()):
        if len(group) < 2:
            continue
        label = prefix_to_label.get(prefix, prefix)

        # Extract numeric param value from run name
        vals = []
        for r in group:
            name = r["run_name"]
            parts = name.split("_", 1)
            try:
                vals.append(float(parts[1]))
            except (ValueError, IndexError):
                vals.append(0)
        sorted_idx = np.argsort(vals)
        x_vals = [vals[i] for i in sorted_idx]
        group_sorted = [group[i] for i in sorted_idx]

        metrics_to_plot = [
            ("chamfer_distance", "Chamfer Distance", True),
            ("test_psnr", "Test PSNR", False),
            ("g2s_mean_sq", "G→Surf mean(d²)", True),
            ("g2v_mean_sq", "G→V mean(d²)", True),
            ("g2mesh_mean", "G→Mesh mean", True),
            ("g2mesh_max", "G→Mesh max", True),
        ]

        # Only plot metrics that have data
        available = []
        for col, name, lower_better in metrics_to_plot:
            data = [g.get(col, None) for g in group_sorted]
            if any(d is not None and d != "" for d in data):
                available.append((col, name, lower_better, data))

        if not available:
            continue

        n_plots = len(available)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5))
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(f"Parameter: {label}", fontsize=12, fontweight="bold")
        for ax, (col, name, lower_better, data) in zip(axes, available):
            y_vals = []
            for d in data:
                try:
                    y_vals.append(float(d))
                except (ValueError, TypeError):
                    y_vals.append(np.nan)
            ax.plot(x_vals, y_vals, "o-", linewidth=2, markersize=6)
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(name, fontsize=9)
            ax.set_title(name, fontsize=10)
            ax.tick_params(labelsize=8)
            # Highlight best
            valid_y = [(i, y) for i, y in enumerate(y_vals) if not np.isnan(y)]
            if valid_y:
                best_i = min(valid_y, key=lambda t: t[1] if lower_better else -t[1])[0]
                ax.plot(x_vals[best_i], y_vals[best_i], "*", color="green", markersize=14, zorder=5)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        figures.append(fig)

    return figures


def main():
    parser = argparse.ArgumentParser(description="Generate PNG benchmark report")
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for PNGs (default: <benchmark_dir>)")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"ERROR: {benchmark_dir} does not exist")
        return

    rows = load_rows(benchmark_dir)
    if not rows:
        print("No benchmark results found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else benchmark_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating PNG reports with {len(rows)} runs...")

    saved = []

    # Table sorted by Chamfer distance
    fig = make_table_figure(rows, "All Runs — Sorted by Chamfer Distance",
                            sort_key="chamfer_distance", ascending=True)
    p = output_dir / "report_chamfer.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Table sorted by PSNR (descending)
    fig = make_table_figure(rows, "All Runs — Sorted by PSNR (higher is better)",
                            sort_key="test_psnr", ascending=False)
    p = output_dir / "report_psnr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Table sorted by G→Surf mean(d²)
    fig = make_table_figure(rows, "All Runs — Sorted by G→Surf mean(d²)",
                            sort_key="g2s_mean_sq", ascending=True)
    p = output_dir / "report_g2s_msq.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Per-parameter trend plots
    figures = make_per_param_figures(rows)
    for i, fig in enumerate(figures):
        p = output_dir / f"report_param_{i:02d}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    print(f"Saved {len(saved)} PNGs to: {output_dir}")
    for s in saved:
        print(f"  {s.name}")


if __name__ == "__main__":
    main()
