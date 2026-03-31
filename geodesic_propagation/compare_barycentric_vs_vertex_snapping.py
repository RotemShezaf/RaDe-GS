#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare barycentric-interpolated geodesic distances vs. old vertex-snapping
for the same source point.

Loads two GT geodesic .npz files (one produced with vertex snapping, one with
barycentric interpolation) and prints detailed comparison statistics:
absolute/signed differences, near-zero analysis, distance-bin breakdowns,
and ASCII histograms.

Usage:
    python geodesic_propagation/compare_barycentric_vs_vertex_snapping.py \
        --old_file  path/to/gt_geodesic_vertex_snap.npz \
        --new_file  path/to/gt_geodesic_barycentric.npz

    # Specify which source row to compare (when multiple sources exist)
    python geodesic_propagation/compare_barycentric_vs_vertex_snapping.py \
        --old_file old.npz --old_source_row 2 \
        --new_file new.npz --new_source_row 2

    # More histogram bins
    python geodesic_propagation/compare_barycentric_vs_vertex_snapping.py \
        --old_file old.npz --new_file new.npz --hist_bins 40

All parameters are configurable via CLI — run with --help for details.
"""
import argparse
import sys

import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def ascii_hist(data, bins=25, width=50, label='', vline=None, vline_label=''):
    """Print an ASCII histogram."""
    counts, edges = np.histogram(data, bins=bins)
    max_c = counts.max() or 1
    print(f"\n  {label}")
    print(f"  {'─'*(width+26)}")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar_len = int(round(c / max_c * width))
        bar = '█' * bar_len
        marker = ''
        if vline is not None and lo <= vline < hi:
            bar += '|'
            marker = f' <- {vline_label}'
        pct = c / len(data) * 100
        print(f"  [{lo:10.6f}, {hi:10.6f}) | {bar:<{width+1}} {c:6d} ({pct:5.1f}%){marker}")
    print(f"  {'─'*(width+26)}")
    print(f"  n={len(data)}  mean={data.mean():.6f}  median={np.median(data):.6f}  "
          f"std={data.std():.6f}  min={data.min():.6f}  max={data.max():.6f}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare barycentric vs. vertex-snapping geodesic ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--old_file', type=str, required=True,
                        help='NPZ file with vertex-snapping GT result')
    parser.add_argument('--old_source_row', type=int, default=0,
                        help='Source row index in the old file')
    parser.add_argument('--new_file', type=str, required=True,
                        help='NPZ file with barycentric GT result')
    parser.add_argument('--new_source_row', type=int, default=0,
                        help='Source row index in the new file')
    parser.add_argument('--hist_bins', type=int, default=25,
                        help='Number of histogram bins')
    parser.add_argument('--near_zero_threshold', type=float, default=1e-4,
                        help='Threshold for near-zero distance analysis')
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    old = np.load(args.old_file, allow_pickle=False)
    new = np.load(args.new_file, allow_pickle=False)

    d_old = old['geodesic_distances'][args.old_source_row]   # (N_gauss,)
    d_new = new['geodesic_distances'][args.new_source_row]   # (N_gauss,)

    assert len(d_old) == len(d_new), \
        f"Gaussian count mismatch: old={len(d_old)}, new={len(d_new)}"

    G = len(d_old)
    diff = d_new - d_old          # signed: positive = new predicted higher distance
    absdiff = np.abs(diff)
    rel_diff = absdiff / (d_old + 1e-8)  # relative difference w.r.t. old

    src_old_gauss = int(old['source_gaussian_indices'][args.old_source_row])
    src_new_gauss = int(new['source_gaussian_indices'][args.new_source_row])

    print(f"\n{'='*70}")
    print(f"  BARYCENTRIC vs VERTEX-SNAPPING COMPARISON")
    print(f"{'='*70}")
    print(f"  Old file : {args.old_file}")
    print(f"  New file : {args.new_file}")
    print(f"  Old source row {args.old_source_row} → Gaussian {src_old_gauss}")
    print(f"  New source row {args.new_source_row} → Gaussian {src_new_gauss}")
    print(f"  Total Gaussians compared: {G}")

    print(f"\n{'='*70}")
    print(f"  ABSOLUTE DIFFERENCE  |new - old|")
    print(f"{'='*70}")
    print(f"  Mean        : {absdiff.mean():.8f}")
    print(f"  Median      : {np.median(absdiff):.8f}")
    print(f"  Std         : {absdiff.std():.8f}")
    print(f"  Max         : {absdiff.max():.8f}")
    print(f"  Changed > 1e-4 : {(absdiff > 1e-4).sum():6d} / {G}  ({(absdiff > 1e-4).mean()*100:.1f}%)")
    print(f"  Changed > 1e-3 : {(absdiff > 1e-3).sum():6d} / {G}  ({(absdiff > 1e-3).mean()*100:.1f}%)")
    print(f"  Changed > 1e-2 : {(absdiff > 1e-2).sum():6d} / {G}  ({(absdiff > 1e-2).mean()*100:.1f}%)")
    print(f"  Changed > 1e-1 : {(absdiff > 1e-1).sum():6d} / {G}  ({(absdiff > 1e-1).mean()*100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"  SIGNED DIFFERENCE   new - old   (positive = new is larger)")
    print(f"{'='*70}")
    print(f"  Mean   : {diff.mean():.8f}")
    print(f"  Median : {np.median(diff):.8f}")
    print(f"  Std    : {diff.std():.8f}")
    print(f"  # new > old : {(diff > 1e-6).sum():6d} / {G}  ({(diff > 1e-6).mean()*100:.1f}%)")
    print(f"  # new < old : {(diff < -1e-6).sum():6d} / {G}  ({(diff < -1e-6).mean()*100:.1f}%)")
    print(f"  # same      : {(np.abs(diff) <= 1e-6).sum():6d} / {G}  ({(np.abs(diff) <= 1e-6).mean()*100:.1f}%)")

    # ------ Zero-distance aliasing fix analysis ------
    NEAR_ZERO = args.near_zero_threshold
    zero_old = d_old < NEAR_ZERO
    zero_new = d_new < NEAR_ZERO
    print(f"\n{'='*70}")
    print(f"  NEAR-ZERO DISTANCE ANALYSIS  (threshold = {NEAR_ZERO})")
    print(f"{'='*70}")
    print(f"  Old: {zero_old.sum():6d} Gaussians with dist < {NEAR_ZERO}  ({zero_old.mean()*100:.2f}%)")
    print(f"  New: {zero_new.sum():6d} Gaussians with dist < {NEAR_ZERO}  ({zero_new.mean()*100:.2f}%)")
    fixed = zero_old & ~zero_new
    print(f"  Fixed (old≈0, new>0): {fixed.sum():6d} Gaussians")
    if fixed.any():
        print(f"    New distances for these (mean={d_new[fixed].mean():.6f}, "
              f"max={d_new[fixed].max():.6f})")
    created = ~zero_old & zero_new
    print(f"  Created (old>0, new≈0): {created.sum():6d} Gaussians")

    # ------ OLD distance distribution ------
    print(f"\n{'='*70}")
    print(f"  OLD DISTANCE DISTRIBUTION  (vertex snapping)")
    print(f"{'='*70}")
    print(f"  Min={d_old.min():.6f}  Max={d_old.max():.6f}  Mean={d_old.mean():.6f}  Median={np.median(d_old):.6f}")

    # ------ NEW distance distribution ------
    print(f"\n{'='*70}")
    print(f"  NEW DISTANCE DISTRIBUTION  (barycentric)")
    print(f"{'='*70}")
    print(f"  Min={d_new.min():.6f}  Max={d_new.max():.6f}  Mean={d_new.mean():.6f}  Median={np.median(d_new):.6f}")

    # ------ Absolute difference by distance bin ------
    print(f"\n{'='*70}")
    print(f"  MEAN |DIFF| BY DISTANCE BIN  (based on old distance)")
    print(f"{'='*70}")
    bin_edges = np.percentile(d_old, np.arange(0, 101, 10))
    # deduplicate edges
    bin_edges = np.unique(bin_edges)
    for i in range(len(bin_edges) - 1):
        mask = (d_old >= bin_edges[i]) & (d_old < bin_edges[i+1])
        if mask.any():
            print(f"  [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}) "
                  f"  n={mask.sum():6d}  mean|diff|={absdiff[mask].mean():.6f}  "
                  f"rel={rel_diff[mask].mean()*100:.1f}%")

    # ------ Histograms ------
    ascii_hist(absdiff, bins=args.hist_bins, width=45,
               label='Histogram: |new - old|  (absolute difference)')
    ascii_hist(d_old, bins=args.hist_bins, width=45,
               label='Histogram: OLD distances  (vertex snapping)')
    ascii_hist(d_new, bins=args.hist_bins, width=45,
               label='Histogram: NEW distances  (barycentric)')


if __name__ == "__main__":
    main()