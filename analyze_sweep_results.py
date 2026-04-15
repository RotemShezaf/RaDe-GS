#!/usr/bin/env python3
"""Analyze benchmark sweep results across all TOSCA shapes and configs."""

import json, glob, os, sys
from collections import defaultdict

BASE = "TrainData/TOSCA/SyntheticColmapData/blue_texture"
SHAPES = ["cat0", "cat2", "centaur0", "centaur1", "centaur5", "david0", "dog0", "gorilla5", "horse0", "michael0"]
CONFIGS = ["sw1", "sw2", "sw3", "sw4", "sw5", "sw6", "sw7", "sw8"]

CONFIG_PARAMS = {
    "sw1": "ldn=0.04 mop=0.35 lmvg=1.0 pd=0.05",
    "sw2": "ldn=0.04 mop=0.35 lmvg=0.5 pd=0.05",
    "sw3": "ldn=0.04 mop=0.30 lmvg=1.0 pd=0.05",
    "sw4": "ldn=0.04 mop=0.25 lmvg=1.0 pd=0.10",
    "sw5": "ldn=0.04 mop=0.35 lmvg=2.0 pd=0.10",
    "sw6": "ldn=0.01 mop=0.35 lmvg=1.0 pd=0.05",
    "sw7": "ldn=0.015 mop=0.35 lmvg=1.5 pd=0.15",
    "sw8": "ldn=0.02 mop=0.40 lmvg=1.0 pd=0.05",
}

METRICS = {
    "PSNR": lambda r: r["psnr"]["test_psnr"],
    "Chamfer": lambda r: r["chamfer_distance"],
    "g2s_max": lambda r: r["gaussian_to_gt_surface"]["max"],
    "g2s_msd": lambda r: r["gaussian_to_gt_surface"]["mean_squared"],
}

# Higher is better for PSNR, lower is better for rest
HIGHER_BETTER = {"PSNR"}


def load_reports():
    data = {}  # (shape, config) -> report
    for shape in SHAPES:
        for cfg in CONFIGS:
            path = f"{BASE}/{shape}/high_res/light_0/sweep_{cfg}/benchmark_report.json"
            if os.path.isfile(path):
                with open(path) as f:
                    data[(shape, cfg)] = json.load(f)
    return data


def print_per_shape_tables(data):
    for metric_name, extract in METRICS.items():
        higher = metric_name in HIGHER_BETTER
        print(f"\n{'='*100}")
        print(f"  {metric_name}  ({'higher=better' if higher else 'lower=better'})")
        print(f"{'='*100}")

        # Header
        header = f"{'Shape':<12}" + "".join(f"{cfg:>10}" for cfg in CONFIGS) + f"  {'Best':>6}"
        print(header)
        print("-" * len(header))

        shape_winners = defaultdict(int)

        for shape in SHAPES:
            vals = {}
            for cfg in CONFIGS:
                if (shape, cfg) in data:
                    vals[cfg] = extract(data[(shape, cfg)])

            if not vals:
                continue

            if higher:
                best_cfg = max(vals, key=vals.get)
            else:
                best_cfg = min(vals, key=vals.get)

            shape_winners[best_cfg] += 1

            row = f"{shape:<12}"
            for cfg in CONFIGS:
                if cfg in vals:
                    v = vals[cfg]
                    marker = " *" if cfg == best_cfg else "  "
                    if metric_name == "PSNR":
                        row += f"{v:>8.2f}{marker}"
                    elif metric_name == "Chamfer":
                        row += f"{v:>8.4f}{marker}"
                    else:
                        row += f"{v:>8.4f}{marker}"
                else:
                    row += f"{'---':>10}"
            row += f"  {best_cfg:>6}"
            print(row)

        # Average row
        print("-" * len(header))
        avg_row = f"{'AVERAGE':<12}"
        avg_vals = {}
        for cfg in CONFIGS:
            values = [extract(data[(s, cfg)]) for s in SHAPES if (s, cfg) in data]
            if values:
                avg_vals[cfg] = sum(values) / len(values)
                if metric_name == "PSNR":
                    avg_row += f"{avg_vals[cfg]:>8.2f}  "
                else:
                    avg_row += f"{avg_vals[cfg]:>8.4f}  "
            else:
                avg_row += f"{'---':>10}"

        if avg_vals:
            if higher:
                best_avg = max(avg_vals, key=avg_vals.get)
            else:
                best_avg = min(avg_vals, key=avg_vals.get)
            avg_row += f"  {best_avg:>6}"
        print(avg_row)

        # Win count
        print(f"\nWin count: ", end="")
        for cfg in CONFIGS:
            print(f"{cfg}={shape_winners.get(cfg, 0)}  ", end="")
        print()


def print_config_ranking(data):
    print(f"\n{'='*100}")
    print(f"  OVERALL CONFIG RANKING (average across shapes)")
    print(f"{'='*100}")

    scores = {}
    for cfg in CONFIGS:
        cfg_scores = {}
        for metric_name, extract in METRICS.items():
            values = [extract(data[(s, cfg)]) for s in SHAPES if (s, cfg) in data]
            if values:
                cfg_scores[metric_name] = sum(values) / len(values)
        if cfg_scores:
            scores[cfg] = cfg_scores

    header = f"{'Config':<8}" + "".join(f"{m:>12}" for m in METRICS) + f"  {'Params'}"
    print(header)
    print("-" * len(header))

    # Rank by Chamfer (primary indicator)
    ranked = sorted(scores.items(), key=lambda x: x[1].get("Chamfer", float('inf')))
    for cfg, s in ranked:
        row = f"{cfg:<8}"
        for m in METRICS:
            v = s.get(m)
            if v is not None:
                if m == "PSNR":
                    row += f"{v:>12.2f}"
                else:
                    row += f"{v:>12.4f}"
            else:
                row += f"{'---':>12}"
        row += f"  {CONFIG_PARAMS[cfg]}"
        print(row)


def print_coverage(data):
    print(f"\n{'='*100}")
    print(f"  COVERAGE: {len(data)} reports")
    print(f"{'='*100}")

    header = f"{'Shape':<12}" + "".join(f"{cfg:>6}" for cfg in CONFIGS) + f"  {'Count':>6}"
    print(header)
    print("-" * len(header))

    for shape in SHAPES:
        row = f"{shape:<12}"
        count = 0
        for cfg in CONFIGS:
            if (shape, cfg) in data:
                row += f"{'  OK':>6}"
                count += 1
            else:
                row += f"{'  --':>6}"
        row += f"  {count:>6}"
        print(row)


def print_win_summary(data):
    """Print a compact table showing which config wins on each shape for each metric."""
    print(f"\n{'='*100}")
    print(f"  WINNER TABLE (best config per shape per metric)")
    print(f"{'='*100}")

    header = f"{'Shape':<12}" + "".join(f"{m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))

    total_wins = defaultdict(int)

    for shape in SHAPES:
        has_data = any((shape, cfg) in data for cfg in CONFIGS)
        if not has_data:
            continue

        row = f"{shape:<12}"
        for metric_name, extract in METRICS.items():
            higher = metric_name in HIGHER_BETTER
            vals = {}
            for cfg in CONFIGS:
                if (shape, cfg) in data:
                    vals[cfg] = extract(data[(shape, cfg)])
            if vals:
                if higher:
                    best = max(vals, key=vals.get)
                else:
                    best = min(vals, key=vals.get)
                total_wins[best] += 1
                row += f"{best:>12}"
            else:
                row += f"{'---':>12}"
        print(row)

    print("-" * len(header))
    print(f"\nTotal wins across all shapes×metrics:")
    for cfg in CONFIGS:
        if total_wins[cfg] > 0:
            print(f"  {cfg}: {total_wins[cfg]}")


def main():
    data = load_reports()
    print(f"Loaded {len(data)} benchmark reports")

    print_coverage(data)
    print_per_shape_tables(data)
    print_win_summary(data)
    print_config_ranking(data)


if __name__ == "__main__":
    main()
