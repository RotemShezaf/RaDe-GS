#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze nearest-neighbor distance statistics for a Gaussian PLY file.

Computes k-NN distances for all Gaussians and prints summary statistics,
percentiles, ASCII histograms, and optionally compares against a known
mesh resolution.

Usage:
    # Polynomial surface
    python geodesic_propagation/analyze_nn_stats.py \
        --gaussian_dir TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_02/light_0/output

    # TOSCA shape with mesh resolution comparison
    python geodesic_propagation/analyze_nn_stats.py \
        --gaussian_dir TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance/output \
        --mesh_resolution 0.005 --n_neighbors 16

    # Custom iteration and more histogram bins
    python geodesic_propagation/analyze_nn_stats.py \
        --gaussian_dir <path> --iteration 7000 --hist_bins 30

All parameters are configurable via CLI — run with --help for details.
"""
import argparse
import os
import sys

import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / 'GenerateData'))
os.chdir(PROJECT_ROOT)

from GenerateData.utils.load_utils import load_gaussian_data_cpu
from scipy.spatial import KDTree


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def ascii_histogram(data, bins=20, width=50, label='', vline=None, vline_label=''):
    """Print an ASCII histogram. *vline* draws a vertical marker at a value."""
    counts, edges = np.histogram(data, bins=bins)
    max_count = counts.max()
    print(f"\n  {label}")
    print(f"  {'─'*(width+28)}")
    for i, (c, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:])):
        bar_len = int(round(c / max_count * width)) if max_count > 0 else 0
        bar = '█' * bar_len
        marker = ''
        if vline is not None and lo <= vline < hi:
            bar = bar[:bar_len] + '|'
            marker = f' <- {vline_label}'
        pct = c / len(data) * 100
        print(f"  [{lo:9.5f}, {hi:9.5f}) | {bar:<{width+1}} {c:6d} ({pct:5.1f}%){marker}")
    print(f"  {'─'*(width+28)}")
    print(f"  n={len(data)}  mean={data.mean():.6f}  median={np.median(data):.6f}  "
          f"std={data.std():.6f}  min={data.min():.6f}  max={data.max():.6f}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze nearest-neighbor distance statistics for a Gaussian PLY file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gaussian_dir', type=str, required=True,
                        help='Path to Gaussian output dir (containing point_cloud/iteration_*/point_cloud.ply)')
    parser.add_argument('--iteration', type=int, default=None,
                        help='Gaussian iteration to load (default: highest available)')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of nearest neighbors to compute (k)')
    parser.add_argument('--mesh_resolution', type=float, default=None,
                        help='Mesh resolution to compare against (draws vertical marker on histograms)')
    parser.add_argument('--hist_bins', type=int, default=20,
                        help='Number of histogram bins')
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    gdir = Path(args.gaussian_dir)
    print(f"\nGaussian dir : {gdir}")
    print(f"N neighbors  : {args.n_neighbors}")
    if args.iteration is not None:
        print(f"Iteration    : {args.iteration}")

    gdata = load_gaussian_data_cpu(gdir, iteration=args.iteration, load_sh_features=False)
    pos = gdata.get_xyz()
    N = len(pos)
    print(f"Total Gaussians: {N}")
    print(f"Position range : x=[{pos[:,0].min():.4f}, {pos[:,0].max():.4f}]  "
          f"y=[{pos[:,1].min():.4f}, {pos[:,1].max():.4f}]  "
          f"z=[{pos[:,2].min():.4f}, {pos[:,2].max():.4f}]")

    tree = KDTree(pos)
    all_dists_full, _ = tree.query(pos, k=args.n_neighbors + 1)
    all_dists_full = all_dists_full[:, 1:]  # drop self (dist=0)
    nn_dists = all_dists_full[:, 0]
    mean_nn_dist = nn_dists.mean()

    # ── Summary statistics ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  NEAREST NEIGHBOR DISTANCE STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean        : {nn_dists.mean():.8f}")
    print(f"  Median      : {np.median(nn_dists):.8f}")
    print(f"  Std         : {nn_dists.std():.8f}")
    print(f"  Min         : {nn_dists.min():.8f}")
    print(f"  Max         : {nn_dists.max():.8f}")
    print(f"  CV (std/mean): {nn_dists.std()/nn_dists.mean():.3f}")
    print(f"\n  Percentiles:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = np.percentile(nn_dists, pct)
        suffix = f"  ({v/args.mesh_resolution:.2f}x mesh_res)" if args.mesh_resolution else ""
        print(f"    P{pct:2d} = {v:.8f}{suffix}")

    # ── Histogram of NN distances ─────────────────────────────────────
    ascii_histogram(nn_dists, bins=args.hist_bins, width=45,
                    label='Histogram: Nearest Neighbor Distance',
                    vline=args.mesh_resolution,
                    vline_label=f'mesh_res={args.mesh_resolution}' if args.mesh_resolution else '')

    # ── Mesh resolution breakdown ─────────────────────────────────────
    if args.mesh_resolution is not None:
        h = args.mesh_resolution
        print(f"\n{'='*60}")
        print(f"  DISTRIBUTION RELATIVE TO MESH RESOLUTION ({h})")
        print(f"{'='*60}")
        fracs = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        prev = 0
        for frac in fracs:
            thresh = h * frac
            cnt = (nn_dists < thresh).sum()
            new = cnt - prev
            bar = '█' * int(cnt / N * 40)
            print(f"  < {frac:4.2f}x ({thresh:.5f}): {cnt:6d}/{N} ({cnt/N*100:5.1f}%)  +{new:5d} in this band  {bar}")
            prev = cnt

    # ── K-th neighbor statistics ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  K-TH NEIGHBOR DISTANCE STATISTICS  (k=1..{args.n_neighbors})")
    print(f"{'='*60}")
    print(f"  {'k':>3}   {'mean':>10}   {'median':>10}   {'std':>10}   {'p5':>10}   {'p95':>10}   {'<mesh_res':>10}")
    print(f"  {'-'*3}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}")
    for k in range(args.n_neighbors):
        d = all_dists_full[:, k]
        below = (d < args.mesh_resolution).sum() / N * 100 if args.mesh_resolution else float('nan')
        below_str = f"{below:8.1f}%" if args.mesh_resolution else "       N/A"
        print(f"  {k+1:>3}   {d.mean():>10.6f}   {np.median(d):>10.6f}   {d.std():>10.6f}   "
              f"{np.percentile(d,5):>10.6f}   {np.percentile(d,95):>10.6f}   {below_str}")

    # ── Histogram of k=1 vs k=5 vs k=10 ──────────────────────────────
    for k in [1, 5, args.n_neighbors]:
        if k <= args.n_neighbors:
            ascii_histogram(all_dists_full[:, k-1], bins=args.hist_bins, width=40,
                            label=f'Histogram: k={k} neighbor distance',
                            vline=args.mesh_resolution,
                            vline_label=f'mesh_res' if args.mesh_resolution else '')


if __name__ == "__main__":
    main()
