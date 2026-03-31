#!/usr/bin/env python3
"""
Compare geodesic distances computed on different meshes against a ground truth.

Takes one **ground-truth** geodesic file and one or more files to
**compare** against it.  Produces per-source and aggregate statistics,
plus optional matplotlib visualisations.

Typical usage
-------------
::

    python GenerateData/compare_geodesic.py \\
        --gt    .../geodesic_distance_gt/gt_geodesic.npz \\
        --compare \\
            .../geodesic_uniform/gt_geodesic.npz \\
            .../geodesic_curvature/gt_geodesic.npz \\
        --labels "GT (Poisson mesh)" "uniform 50k" "curvature 50k" \\
        --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: str):
    """Load geodesic npz and return (distances, source_positions, gaussian_positions, source_gaussian_indices)."""
    d = np.load(path, allow_pickle=True)
    return (
        d["geodesic_distances"],        # (S, G)
        d["source_positions"],          # (S, 3)
        d["gaussian_positions"],        # (G, 3)
        d.get("source_gaussian_indices", None),  # (S,) or None
    )


def _match_sources(pos_a, pos_b, tol=1e-4):
    """Return index arrays (idx_a, idx_b) matching sources by position.

    Uses nearest-neighbour matching and enforces one-to-one mapping
    (each target matched at most once).  If the initial *tol* yields
    zero matches, the tolerance is automatically relaxed up to 0.05.
    """
    from scipy.spatial import KDTree
    tree_b = KDTree(pos_b)
    dists, idxs = tree_b.query(pos_a)

    # Auto-escalate tolerance if initial tol gives no matches
    for try_tol in [tol, 0.005, 0.01, 0.02, 0.05]:
        if try_tol < tol:
            continue
        mask = dists < try_tol
        if mask.sum() > 0:
            if try_tol > tol:
                print(f"    (auto-relaxed tolerance to {try_tol})")
            break
    else:
        mask = np.zeros(len(dists), dtype=bool)

    idx_a = np.where(mask)[0]
    idx_b = idxs[mask]

    # Ensure one-to-one: if multiple idx_a map to same idx_b, keep closest
    if len(idx_b) > 0:
        seen = {}
        keep = []
        for k in range(len(idx_a)):
            b = idx_b[k]
            if b not in seen or dists[idx_a[k]] < dists[idx_a[seen[b]]]:
                if b in seen:
                    # Remove previous, keep this one
                    keep[seen[b]] = False
                seen[b] = k
                keep.append(True)
            else:
                keep.append(False)
        keep = np.array(keep)
        idx_a = idx_a[keep]
        idx_b = idx_b[keep]

    return idx_a, idx_b


def _pairwise_stats(d_a: np.ndarray, d_b: np.ndarray, label_a: str, label_b: str):
    """Compute comparison statistics between two aligned distance arrays.

    Parameters
    ----------
    d_a, d_b : ndarray, shape ``(S, G)``
        Matched geodesic distances (same sources, same Gaussians).

    Returns
    -------
    stats : dict
    """
    diff = d_a - d_b
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.maximum(d_a, d_b), 1e-12)

    # Per-source statistics
    per_source = {}
    for s in range(d_a.shape[0]):
        sd = abs_diff[s]
        rd = rel_diff[s]
        per_source[int(s)] = {
            "abs_diff_mean": float(sd.mean()),
            "abs_diff_median": float(np.median(sd)),
            "abs_diff_max": float(sd.max()),
            "rel_diff_mean": float(rd.mean()),
            "rel_diff_median": float(np.median(rd)),
            "rel_diff_max": float(rd.max()),
        }

    # Correlation
    valid = np.isfinite(d_a.ravel()) & np.isfinite(d_b.ravel())
    corr = float(np.corrcoef(d_a.ravel()[valid], d_b.ravel()[valid])[0, 1])

    return {
        "compared": f"{label_a} vs {label_b}",
        "n_sources_matched": int(d_a.shape[0]),
        "n_gaussians": int(d_a.shape[1]),
        "correlation": corr,
        "absolute_difference": {
            "mean": float(abs_diff.mean()),
            "median": float(np.median(abs_diff)),
            "std": float(abs_diff.std()),
            "max": float(abs_diff.max()),
            "pct_lt_0.001": float(100 * np.mean(abs_diff < 0.001)),
            "pct_lt_0.01": float(100 * np.mean(abs_diff < 0.01)),
            "pct_lt_0.05": float(100 * np.mean(abs_diff < 0.05)),
        },
        "relative_difference": {
            "mean": float(rel_diff.mean()),
            "median": float(np.median(rel_diff)),
            "std": float(rel_diff.std()),
            "max": float(rel_diff.max()),
            "pct_lt_1pct": float(100 * np.mean(rel_diff < 0.01)),
            "pct_lt_5pct": float(100 * np.mean(rel_diff < 0.05)),
            "pct_lt_10pct": float(100 * np.mean(rel_diff < 0.10)),
        },
        "signed_bias": {
            "mean": float(diff.mean()),
            "std": float(diff.std()),
            "description": f"positive = {label_a} > {label_b}",
        },
        "distance_ranges": {
            label_a: {
                "min": float(d_a.min()),
                "median": float(np.median(d_a)),
                "max": float(d_a.max()),
            },
            label_b: {
                "min": float(d_b.min()),
                "median": float(np.median(d_b)),
                "max": float(d_b.max()),
            },
        },
        "per_source": per_source,
    }


def _print_summary(stats: dict):
    """Pretty-print comparison summary."""
    print(f"\n{'=' * 70}")
    print(f"  {stats['compared']}")
    print(f"{'=' * 70}")
    print(f"  Matched sources : {stats['n_sources_matched']}")
    print(f"  Gaussians       : {stats['n_gaussians']}")
    print(f"  Correlation     : {stats['correlation']:.6f}")

    ad = stats["absolute_difference"]
    print(f"\n  Absolute difference:")
    print(f"    mean={ad['mean']:.6f}  median={ad['median']:.6f}  "
          f"max={ad['max']:.6f}  std={ad['std']:.6f}")
    print(f"    <0.001: {ad['pct_lt_0.001']:.1f}%  "
          f"<0.01: {ad['pct_lt_0.01']:.1f}%  "
          f"<0.05: {ad['pct_lt_0.05']:.1f}%")

    rd = stats["relative_difference"]
    print(f"\n  Relative difference:")
    print(f"    mean={rd['mean']:.4f}  median={rd['median']:.4f}  "
          f"max={rd['max']:.4f}  std={rd['std']:.4f}")
    print(f"    <1%: {rd['pct_lt_1pct']:.1f}%  "
          f"<5%: {rd['pct_lt_5pct']:.1f}%  "
          f"<10%: {rd['pct_lt_10pct']:.1f}%")

    sb = stats["signed_bias"]
    print(f"\n  Signed bias: mean={sb['mean']:.6f}  std={sb['std']:.6f}")
    print(f"    ({sb['description']})")

    dr = stats["distance_ranges"]
    for name, r in dr.items():
        print(f"\n  Range [{name}]:")
        print(f"    min={r['min']:.6f}  median={r['median']:.6f}  max={r['max']:.6f}")
    print()


def _plot_comparison(
    d_a: np.ndarray,
    d_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: Optional[Path] = None,
):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter: d_a vs d_b (subsample for speed)
    ax = axes[0, 0]
    n_total = d_a.size
    max_pts = 100_000
    if n_total > max_pts:
        idx = np.random.RandomState(42).choice(n_total, max_pts, replace=False)
        xa, xb = d_a.ravel()[idx], d_b.ravel()[idx]
    else:
        xa, xb = d_a.ravel(), d_b.ravel()

    ax.scatter(xa, xb, s=0.5, alpha=0.3, rasterized=True)
    lim = max(xa.max(), xb.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title("Geodesic distance scatter")
    ax.legend()
    ax.set_aspect("equal")

    # 2. Histogram of absolute differences
    ax = axes[0, 1]
    abs_diff = np.abs(d_a - d_b).ravel()
    ax.hist(abs_diff, bins=100, density=True, alpha=0.7, color="steelblue")
    ax.axvline(np.median(abs_diff), color="red", linestyle="--", label=f"median={np.median(abs_diff):.4f}")
    ax.set_xlabel("Absolute difference")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of absolute differences")
    ax.legend()

    # 3. Histogram of relative differences
    ax = axes[1, 0]
    rel_diff = abs_diff / np.maximum(np.maximum(d_a.ravel(), d_b.ravel()), 1e-12)
    # Clip for better visualization
    clip_val = np.percentile(rel_diff, 99)
    ax.hist(rel_diff[rel_diff < clip_val], bins=100, density=True, alpha=0.7, color="coral")
    ax.axvline(np.median(rel_diff), color="red", linestyle="--", label=f"median={np.median(rel_diff):.4f}")
    ax.set_xlabel("Relative difference")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of relative differences (clipped at {clip_val:.3f})")
    ax.legend()

    # 4. Per-source median absolute difference
    ax = axes[1, 1]
    per_source_median = np.median(np.abs(d_a - d_b), axis=1)
    ax.bar(range(len(per_source_median)), per_source_median, color="teal", alpha=0.7)
    ax.axhline(np.median(per_source_median), color="red", linestyle="--",
               label=f"overall median={np.median(per_source_median):.4f}")
    ax.set_xlabel("Source index")
    ax.set_ylabel("Median absolute difference")
    ax.set_title("Per-source median absolute difference")
    ax.legend()

    plt.suptitle(f"{label_a} vs {label_b}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    else:
        plt.savefig("geodesic_comparison.png", dpi=150, bbox_inches="tight")
        print("  Plot saved to geodesic_comparison.png")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare geodesic distances against a ground-truth reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to the ground-truth geodesic npz file.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to geodesic npz file(s) to compare against --gt.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Labels for the datasets.  First label is for --gt, rest "
            "for --compare (in order).  Default: auto-generated from "
            "parent directory names."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save comparison results.  Default: same dir as --gt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-source statistics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_paths = [args.gt] + args.compare
    n_datasets = len(all_paths)

    # Labels
    if args.labels and len(args.labels) == n_datasets:
        labels = args.labels
    else:
        labels = []
        labels.append("GT")
        for p in args.compare:
            parent = Path(p).parent.name
            labels.append(parent)

    print(f"\n{'#' * 70}")
    print(f"# Geodesic Distance Comparison")
    print(f"{'#' * 70}")
    for i, (lbl, p) in enumerate(zip(labels, all_paths)):
        print(f"  [{i}] {lbl}: {p}")

    # Load all datasets
    datasets = []
    for lbl, p in zip(labels, all_paths):
        print(f"\n  Loading {lbl} ...")
        dists, src_pos, gauss_pos, src_gi = _load(p)
        print(f"    shape: {dists.shape}, range: [{dists.min():.6f}, {dists.max():.6f}]")
        datasets.append((dists, src_pos, gauss_pos, src_gi))

    # Compare each --compare dataset vs GT
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.gt).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []

    gt_dists, gt_src, gt_gauss, gt_gi = datasets[0]
    gt_label = labels[0]

    for i, (cmp_dists, cmp_src, cmp_gauss, cmp_gi) in enumerate(datasets[1:], 1):
        cmp_label = labels[i]

        print(f"\n  Comparing {cmp_label} vs {gt_label} ...")

        # Match sources
        idx_cmp, idx_gt = _match_sources(cmp_src, gt_src)
        print(f"    Matched {len(idx_cmp)} / {cmp_src.shape[0]} sources by position")

        if len(idx_cmp) == 0:
            print(f"    [!] No positional matches — trying by source_gaussian_indices ...")
            if cmp_gi is not None and gt_gi is not None:
                cmp_set = {int(gi): s for s, gi in enumerate(cmp_gi)}
                gt_set = {int(gi): s for s, gi in enumerate(gt_gi)}
                common = sorted(set(cmp_set) & set(gt_set))
                idx_cmp = np.array([cmp_set[gi] for gi in common])
                idx_gt = np.array([gt_set[gi] for gi in common])
                print(f"    Matched {len(idx_cmp)} sources by gaussian index")

        if len(idx_cmp) == 0:
            print(f"    [!] Could not match any sources — skipping")
            continue

        # Check Gaussian positions match
        gauss_diffs = np.linalg.norm(cmp_gauss - gt_gauss, axis=1)
        if gauss_diffs.max() > 0.01:
            print(f"    [!] Gaussian positions differ (max diff={gauss_diffs.max():.6f})")

        d_cmp = cmp_dists[idx_cmp]
        d_gt = gt_dists[idx_gt]

        stats = _pairwise_stats(d_cmp, d_gt, cmp_label, gt_label)
        _print_summary(stats)
        all_stats.append(stats)

        if args.verbose:
            print(f"\n  Per-source details (top 10 by max abs diff):")
            per_src = stats["per_source"]
            sorted_src = sorted(per_src.items(),
                                key=lambda x: x[1]["abs_diff_max"], reverse=True)
            for rank, (src_idx, s) in enumerate(sorted_src[:10]):
                print(f"    src {src_idx:3d}: abs_max={s['abs_diff_max']:.6f}  "
                      f"abs_mean={s['abs_diff_mean']:.6f}  "
                      f"rel_mean={s['rel_diff_mean']:.4f}")

        if args.plot:
            safe_label = cmp_label.replace(" ", "_").replace("/", "_")
            plot_path = output_dir / f"comparison_{safe_label}_vs_GT.png"
            _plot_comparison(d_cmp, d_gt, cmp_label, gt_label, plot_path)

    # Save all stats to JSON
    if all_stats:
        # Remove per_source from JSON to keep it readable
        stats_for_json = []
        for s in all_stats:
            sj = {k: v for k, v in s.items() if k != "per_source"}
            stats_for_json.append(sj)

        json_path = output_dir / "geodesic_comparison.json"
        with open(json_path, "w") as f:
            json.dump(stats_for_json, f, indent=2)
        print(f"\n  Comparison JSON saved to {json_path}")

    print(f"\n{'#' * 70}\n")


if __name__ == "__main__":
    main()
