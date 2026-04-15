#!/usr/bin/env python3
"""Deep analysis of cat0 R4-R6 + sweep data, then per-shape sweep analysis."""

import json, glob, os, csv, sys
from collections import defaultdict
import math

BASE = "TrainData/TOSCA/SyntheticColmapData/blue_texture"
ROUND_BASE = "output/benchmarks/tosca_params/cat0_blue_high_res"
SHAPES = ["cat0", "cat2", "centaur0", "centaur1", "centaur5", "david0", "dog0", "gorilla5", "horse0"]
CONFIGS = ["sw1", "sw2", "sw3", "sw4", "sw5", "sw6", "sw7", "sw8"]

SWEEP_PARAMS = {
    "sw1": {"bpsf": 0.005, "mop": 0.35, "lmvg": 1.0, "ldist": 0.05, "ldn": 0.04, "pd": 0.05, "psa": 0, "pmst": 0},
    "sw2": {"bpsf": 0.005, "mop": 0.35, "lmvg": 0.5, "ldist": 0.05, "ldn": 0.04, "pd": 0.05, "psa": 0, "pmst": 0},
    "sw3": {"bpsf": 0.005, "mop": 0.30, "lmvg": 1.0, "ldist": 0.05, "ldn": 0.04, "pd": 0.05, "psa": 0, "pmst": 0},
    "sw4": {"bpsf": 0.005, "mop": 0.25, "lmvg": 1.0, "ldist": 0.05, "ldn": 0.04, "pd": 0.10, "psa": 0, "pmst": 0},
    "sw5": {"bpsf": 0.005, "mop": 0.35, "lmvg": 2.0, "ldist": 0.05, "ldn": 0.04, "pd": 0.10, "psa": 0, "pmst": 0},
    "sw6": {"bpsf": 0.005, "mop": 0.35, "lmvg": 1.0, "ldist": 0.05, "ldn": 0.01, "pd": 0.05, "psa": 0, "pmst": 0},
    "sw7": {"bpsf": 0.005, "mop": 0.35, "lmvg": 1.5, "ldist": 0.05, "ldn": 0.015, "pd": 0.15, "psa": 0, "pmst": 0},
    "sw8": {"bpsf": 0.005, "mop": 0.40, "lmvg": 1.0, "ldist": 0.05, "ldn": 0.02, "pd": 0.05, "psa": 0, "pmst": 0},
}


def load_round_csv(path):
    """Load R5 or R6 summary CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_sweep_report(shape, cfg):
    """Load a sweep benchmark_report.json."""
    path = f"{BASE}/{shape}/high_res/light_0/sweep_{cfg}/benchmark_report.json"
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_metrics(report):
    """Extract the 4 key metrics + num_gaussians from a benchmark report."""
    return {
        "psnr": report.get("psnr", {}).get("test_psnr", 0) if isinstance(report.get("psnr"), dict) else report.get("test_psnr", 0),
        "chamfer": report.get("chamfer_distance", 0),
        "g2s_max": report.get("gaussian_to_gt_surface", {}).get("max", 0),
        "g2s_msd": report.get("gaussian_to_gt_surface", {}).get("mean_squared", 0),
        "num_gauss": report.get("num_gaussians", 0),
    }


def extract_metrics_from_csv_row(row):
    """Extract metrics from R5/R6 CSV row."""
    return {
        "psnr": float(row.get("test_psnr", 0)),
        "chamfer": float(row.get("chamfer_distance", 0)),
        "g2s_max": float(row.get("g2gt_max", 0)),
        "g2s_msd": float(row.get("g2gt_mean_squared", 0)),
        "num_gauss": int(row.get("num_gaussians", 0)),
    }


def extract_params_from_csv_row(row):
    """Extract parameters from R5/R6 CSV row."""
    p = {}
    for key in ["big_point_scale_factor", "min_opacity_prune", "percent_dense",
                 "lambda_depth_normal", "prune_scale_anisotropy", "lambda_multi_view_geo"]:
        if key in row and row[key]:
            p[key] = float(row[key])
    # Map to short names
    return {
        "bpsf": p.get("big_point_scale_factor", 0.005),
        "mop": p.get("min_opacity_prune", 0.1),
        "pd": p.get("percent_dense", 0.1),
        "ldn": p.get("lambda_depth_normal", 0.08),
        "psa": p.get("prune_scale_anisotropy", 0),
        "lmvg": p.get("lambda_multi_view_geo", 0.02),
    }


# ============================================================================
# PART 1: DEEP CAT0 ANALYSIS
# ============================================================================
def analyze_cat0():
    print("=" * 120)
    print("  PART 1: DEEP CAT0 ANALYSIS (R5 + R6 + Sweep = ~47 runs)")
    print("=" * 120)

    all_runs = []  # list of (name, params, metrics)

    # Load R5
    r5_rows = load_round_csv(f"{ROUND_BASE}/r5_summary.csv")
    for row in r5_rows:
        params = extract_params_from_csv_row(row)
        metrics = extract_metrics_from_csv_row(row)
        all_runs.append((row["run_name"], params, metrics))

    # Load R6
    r6_rows = load_round_csv(f"{ROUND_BASE}/r6_summary.csv")
    for row in r6_rows:
        params = extract_params_from_csv_row(row)
        # R6 has lambda_multi_view_geo column
        if "lambda_multi_view_geo" in row and row["lambda_multi_view_geo"]:
            params["lmvg"] = float(row["lambda_multi_view_geo"])
        metrics = extract_metrics_from_csv_row(row)
        all_runs.append((row["run_name"], params, metrics))

    # Load sweep
    for cfg in CONFIGS:
        report = load_sweep_report("cat0", cfg)
        if report:
            metrics = extract_metrics(report)
            all_runs.append((f"sweep_{cfg}", SWEEP_PARAMS[cfg], metrics))

    print(f"\nLoaded {len(all_runs)} cat0 runs")

    # --- Parameter ranges ---
    param_names = ["ldn", "mop", "lmvg", "pd", "bpsf", "psa"]
    print(f"\nParameter ranges across all runs:")
    for pname in param_names:
        vals = sorted(set(r[1].get(pname, 0) for r in all_runs))
        print(f"  {pname:>6}: {vals}")

    # --- Correlation analysis ---
    print(f"\n{'─' * 120}")
    print(f"  PARAMETER → METRIC CORRELATIONS (grouped by parameter value)")
    print(f"{'─' * 120}")

    metric_names = ["psnr", "chamfer", "g2s_max", "g2s_msd", "num_gauss"]

    for pname in param_names:
        vals = sorted(set(r[1].get(pname, 0) for r in all_runs))
        if len(vals) < 2:
            continue

        print(f"\n  ── {pname} ──")
        header = f"  {'value':>8}  {'count':>5}" + "".join(f"  {m:>12}" for m in metric_names)
        print(header)

        for v in vals:
            matching = [r for r in all_runs if r[1].get(pname, 0) == v]
            if not matching:
                continue
            avgs = {}
            for m in metric_names:
                values = [r[2][m] for r in matching if r[2][m] != 0]
                avgs[m] = sum(values) / len(values) if values else 0

            row = f"  {v:>8.4f}  {len(matching):>5}"
            for m in metric_names:
                if m == "psnr":
                    row += f"  {avgs[m]:>12.2f}"
                elif m == "num_gauss":
                    row += f"  {avgs[m]:>12.0f}"
                else:
                    row += f"  {avgs[m]:>12.4f}"
            print(row)

    # --- Top runs by each metric ---
    print(f"\n{'─' * 120}")
    print(f"  TOP 10 CAT0 RUNS BY EACH METRIC")
    print(f"{'─' * 120}")

    for metric, direction in [("psnr", "max"), ("chamfer", "min"), ("g2s_max", "min"), ("g2s_msd", "min"), ("num_gauss", "min")]:
        rev = direction == "max"
        sorted_runs = sorted(all_runs, key=lambda r: r[2][metric], reverse=rev)
        # Filter out zero values for num_gauss
        if metric == "num_gauss":
            sorted_runs = [r for r in sorted_runs if r[2][metric] > 0]

        print(f"\n  Top 10 by {metric} ({'↑' if rev else '↓'}):")
        header = f"  {'run':>15}  {'ldn':>6}  {'mop':>5}  {'lmvg':>5}  {'pd':>5}  {'psa':>5}  {'bpsf':>6}  |  {'PSNR':>7}  {'Chamfer':>8}  {'g2s_max':>8}  {'g2s_msd':>8}  {'N_gauss':>8}"
        print(header)
        for name, params, metrics in sorted_runs[:10]:
            print(f"  {name:>15}  {params.get('ldn', 0):>6.3f}  {params.get('mop', 0):>5.2f}  {params.get('lmvg', 0):>5.1f}  {params.get('pd', 0):>5.3f}  {params.get('psa', 0):>5.1f}  {params.get('bpsf', 0):>6.3f}  |  {metrics['psnr']:>7.2f}  {metrics['chamfer']:>8.4f}  {metrics['g2s_max']:>8.2f}  {metrics['g2s_msd']:>8.4f}  {metrics['num_gauss']:>8.0f}")

    # --- Key findings summary ---
    print(f"\n{'─' * 120}")
    print(f"  CAT0 KEY FINDINGS")
    print(f"{'─' * 120}")

    # Best balanced run
    # Normalize metrics and find best combo
    all_psnr = [r[2]["psnr"] for r in all_runs]
    all_chamfer = [r[2]["chamfer"] for r in all_runs]
    all_g2s_max = [r[2]["g2s_max"] for r in all_runs]
    all_g2s_msd = [r[2]["g2s_msd"] for r in all_runs]

    def rank_score(run):
        _, p, m = run
        # Rank 0-1 for each metric (1=best)
        psnr_rank = (m["psnr"] - min(all_psnr)) / (max(all_psnr) - min(all_psnr) + 1e-9)
        chamfer_rank = 1 - (m["chamfer"] - min(all_chamfer)) / (max(all_chamfer) - min(all_chamfer) + 1e-9)
        g2s_max_rank = 1 - (m["g2s_max"] - min(all_g2s_max)) / (max(all_g2s_max) - min(all_g2s_max) + 1e-9)
        g2s_msd_rank = 1 - (m["g2s_msd"] - min(all_g2s_msd)) / (max(all_g2s_msd) - min(all_g2s_msd) + 1e-9)
        return psnr_rank + chamfer_rank + g2s_max_rank + g2s_msd_rank

    ranked = sorted(all_runs, key=rank_score, reverse=True)
    print(f"\n  Top 10 balanced runs (normalized rank across all 4 metrics):")
    header = f"  {'#':>2}  {'run':>15}  {'ldn':>6}  {'mop':>5}  {'lmvg':>5}  {'pd':>5}  {'psa':>5}  {'bpsf':>6}  |  {'PSNR':>7}  {'Chamfer':>8}  {'g2s_max':>8}  {'g2s_msd':>8}  {'N_gauss':>8}  {'score':>6}"
    print(header)
    for i, (name, params, metrics) in enumerate(ranked[:10]):
        score = rank_score((name, params, metrics))
        print(f"  {i+1:>2}  {name:>15}  {params.get('ldn', 0):>6.3f}  {params.get('mop', 0):>5.2f}  {params.get('lmvg', 0):>5.1f}  {params.get('pd', 0):>5.3f}  {params.get('psa', 0):>5.1f}  {params.get('bpsf', 0):>6.3f}  |  {metrics['psnr']:>7.2f}  {metrics['chamfer']:>8.4f}  {metrics['g2s_max']:>8.2f}  {metrics['g2s_msd']:>8.4f}  {metrics['num_gauss']:>8.0f}  {score:>6.3f}")


# ============================================================================
# PART 2: PER-SHAPE SWEEP ANALYSIS
# ============================================================================
def analyze_per_shape():
    print(f"\n\n{'=' * 120}")
    print(f"  PART 2: PER-SHAPE SWEEP ANALYSIS")
    print(f"{'=' * 120}")

    for shape in SHAPES:
        runs = []
        for cfg in CONFIGS:
            report = load_sweep_report(shape, cfg)
            if report:
                metrics = extract_metrics(report)
                runs.append((cfg, SWEEP_PARAMS[cfg], metrics))

        if not runs:
            continue

        print(f"\n{'─' * 120}")
        print(f"  {shape} ({len(runs)} configs)")
        print(f"{'─' * 120}")

        header = f"  {'cfg':>5}  {'ldn':>6}  {'mop':>5}  {'lmvg':>5}  {'pd':>5}  |  {'PSNR':>7}  {'Chamfer':>8}  {'g2s_max':>8}  {'g2s_msd':>8}  {'N_gauss':>8}"
        print(header)

        for cfg, params, metrics in sorted(runs, key=lambda r: r[2]["chamfer"]):
            print(f"  {cfg:>5}  {params['ldn']:>6.3f}  {params['mop']:>5.2f}  {params['lmvg']:>5.1f}  {params['pd']:>5.3f}  |  {metrics['psnr']:>7.2f}  {metrics['chamfer']:>8.4f}  {metrics['g2s_max']:>8.2f}  {metrics['g2s_msd']:>8.4f}  {metrics['num_gauss']:>8.0f}")

        # Shape-specific issues
        avg_gauss = sum(r[2]["num_gauss"] for r in runs) / len(runs)
        avg_g2s_max = sum(r[2]["g2s_max"] for r in runs) / len(runs)
        avg_g2s_msd = sum(r[2]["g2s_msd"] for r in runs) / len(runs)
        max_gauss = max(r[2]["num_gauss"] for r in runs)
        min_gauss = min(r[2]["num_gauss"] for r in runs)
        best_chamfer_cfg = min(runs, key=lambda r: r[2]["chamfer"])[0]
        best_psnr_cfg = max(runs, key=lambda r: r[2]["psnr"])[0]
        best_g2s_max_cfg = min(runs, key=lambda r: r[2]["g2s_max"])[0]
        best_g2s_msd_cfg = min(runs, key=lambda r: r[2]["g2s_msd"])[0]

        print(f"\n  Summary: avg_gauss={avg_gauss:.0f} (range {min_gauss:.0f}-{max_gauss:.0f}), avg_g2s_max={avg_g2s_max:.1f}, avg_g2s_msd={avg_g2s_msd:.4f}")
        print(f"  Winners: Chamfer→{best_chamfer_cfg}, PSNR→{best_psnr_cfg}, g2s_max→{best_g2s_max_cfg}, g2s_msd→{best_g2s_msd_cfg}")

        # Does it need gaussian reduction? (>150K)
        if max_gauss > 150000:
            print(f"  ⚠ NEEDS GAUSSIAN REDUCTION: {max_gauss:.0f} > 150K limit")
        else:
            print(f"  ✓ Gaussian count OK (max {max_gauss:.0f} < 150K)")

        # Outlier severity
        if avg_g2s_max > 15:
            print(f"  ⚠ SEVERE OUTLIERS: avg g2s_max={avg_g2s_max:.1f}")
        elif avg_g2s_max > 10:
            print(f"  ⚠ MODERATE OUTLIERS: avg g2s_max={avg_g2s_max:.1f}")
        else:
            print(f"  ✓ Outliers manageable: avg g2s_max={avg_g2s_max:.1f}")


# ============================================================================
# PART 3: PARAMETER EFFECT SUMMARY
# ============================================================================
def parameter_effects_summary():
    print(f"\n\n{'=' * 120}")
    print(f"  PART 3: CROSS-SHAPE PARAMETER EFFECT SUMMARY")
    print(f"{'=' * 120}")

    # Compare low ldn vs high ldn, high mop vs low mop, etc.
    comparisons = [
        ("ldn effect", ["sw1", "sw2", "sw3"], ["sw6", "sw7", "sw8"], "ldn=0.04 (Block A)", "ldn≤0.02 (Block B)"),
        ("mop effect", ["sw3"], ["sw1", "sw2"], "mop=0.30", "mop=0.35"),
        ("mop=0.40 effect", ["sw6"], ["sw8"], "mop=0.35 (sw6)", "mop=0.40 (sw8)"),
        ("pd effect", ["sw1"], ["sw7"], "pd=0.05 (sw1)", "pd=0.15 (sw7)"),
        ("lmvg effect", ["sw1"], ["sw5"], "lmvg=1.0", "lmvg=2.0"),
    ]

    for label, group_a, group_b, name_a, name_b in comparisons:
        print(f"\n  ── {label}: {name_a} vs {name_b} ──")

        for metric in ["psnr", "chamfer", "g2s_max", "g2s_msd", "num_gauss"]:
            vals_a, vals_b = [], []
            for shape in SHAPES:
                for cfg in group_a:
                    r = load_sweep_report(shape, cfg)
                    if r:
                        m = extract_metrics(r)
                        vals_a.append(m[metric])
                for cfg in group_b:
                    r = load_sweep_report(shape, cfg)
                    if r:
                        m = extract_metrics(r)
                        vals_b.append(m[metric])

            if vals_a and vals_b:
                avg_a = sum(vals_a) / len(vals_a)
                avg_b = sum(vals_b) / len(vals_b)
                diff_pct = (avg_b - avg_a) / abs(avg_a) * 100 if avg_a != 0 else 0
                direction = "↑" if diff_pct > 0 else "↓"
                if metric == "psnr":
                    print(f"    {metric:>10}: {avg_a:>10.2f} → {avg_b:>10.2f}  ({direction}{abs(diff_pct):>5.1f}%)")
                elif metric == "num_gauss":
                    print(f"    {metric:>10}: {avg_a:>10.0f} → {avg_b:>10.0f}  ({direction}{abs(diff_pct):>5.1f}%)")
                else:
                    print(f"    {metric:>10}: {avg_a:>10.4f} → {avg_b:>10.4f}  ({direction}{abs(diff_pct):>5.1f}%)")


if __name__ == "__main__":
    analyze_cat0()
    analyze_per_shape()
    parameter_effects_summary()
