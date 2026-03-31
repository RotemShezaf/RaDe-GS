#!/usr/bin/env python3
"""
Analyze ring neighbor statistics for TOSCA and Polynomial datasets.

Computes ring-1/2/3/4 neighbor counts, geo/euclidean ratio distributions,
recommends ring_size_mapping values and outlier thresholds, and performs
adaptive kNN over-cap analysis.

Usage:
    python DataSets/analyze_ring_statistics.py [--dataset tosca|polynomial|both]
    python DataSets/analyze_ring_statistics.py --dataset tosca --shapes cat0,dog0
    python DataSets/analyze_ring_statistics.py --dataset polynomial --surfaces Paraboloid
    python DataSets/analyze_ring_statistics.py --mode adaptive --dataset both
"""
import os
import sys
import argparse
import multiprocessing
import numpy as np
from numpy import linalg as LA
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
# Remove DataSets/ from sys.path to prevent DataSets/utils/ from shadowing
# the project-root utils/ package (needed by data_generation_utils.py).
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.data_generation_utils import (
    get_all_points_nbrs_all_rings,
    ring1_neighbors_gaussians,
    get_neighborhood_by_ring,
    adaptive_ring1_neighbors,
    ring_counts_sparse,
)
from GenerateData.utils.load_utils import load_gaussian_data_cpu


# ============================================================================
# Dataset layout helpers
# ============================================================================
TOSCA_BASE = "TrainData/TOSCA/SyntheticColmapData/blue_texture"
POLY_BASE = "TrainData/Polynomial/SyntheticColmapData/blue_texture"

ALL_TOSCA_SHAPES = [
    "cat0", "cat2", "centaur0", "centaur1", "centaur5",
    "david0", "dog0", "gorilla5", "horse0",
    "michael0", "michael2", "victoria0", "victoria2", "wolf0",
]

ALL_POLY_SURFACES = ["Paraboloid", "Saddle", "HyperbolicParaboloid"]
POLY_LEVELS = ["level_02", "level_03", "level_04"]
POLY_LIGHTS = ["light_0", "light_1", "light_2", "light_3", "light_4"]


def find_tosca_output(shape: str) -> str:
    """Return path to a single TOSCA Gaussian output for analysis."""
    base = project_root / TOSCA_BASE / shape / "high_res" / "decoupled_appearance" / "output"
    if base.exists():
        return str(base)
    # fallback: try light_0 pattern
    base2 = project_root / TOSCA_BASE / shape / "high_res" / "light_0" / "output"
    if base2.exists():
        return str(base2)
    raise FileNotFoundError(f"No Gaussian output found for TOSCA shape {shape}")


def find_poly_output(surface: str, level: str = "level_03", light: str = "light_0") -> str:
    """Return path to a polynomial Gaussian output for analysis."""
    base = project_root / POLY_BASE / surface / level / light / "output"
    if base.exists():
        return str(base)
    raise FileNotFoundError(f"No Gaussian output found for {surface}/{level}/{light}")


def find_geodesic_data(output_path: str) -> str:
    """Find geodesic .npz in the output or nearby."""
    geo_dir = Path(output_path) / "geodesic_distance"
    npz = geo_dir / "gt_geodesic.npz"
    if npz.exists():
        return str(npz)
    raise FileNotFoundError(f"No geodesic data in {geo_dir}")


# ============================================================================
# Analysis functions
# ============================================================================
def compute_ring_stats(positions, ring_nbrs_dict, geodesic_distances,
                       ring_label="ring-2", max_sample=5000):
    """Compute detailed statistics for a given ring neighborhood."""
    num_pts = len(ring_nbrs_dict)
    sample_indices = np.random.choice(num_pts, min(max_sample, num_pts), replace=False)
    
    neighbor_counts = []
    all_geo_euc_ratios = []
    all_euc_distances = []
    all_geo_distances = []
    
    for idx in sample_indices:
        nbrs = ring_nbrs_dict[idx]
        if len(nbrs) == 0:
            neighbor_counts.append(0)
            continue
        neighbor_counts.append(len(nbrs))
        
        nbrs_xyz = positions[nbrs] - positions[idx]
        euc_dists = LA.norm(nbrs_xyz, axis=1)
        geo_dists = geodesic_distances[nbrs]
        
        eps = 1e-8
        ratios = geo_dists / np.maximum(euc_dists, eps)
        
        all_geo_euc_ratios.extend(ratios.tolist())
        all_euc_distances.extend(euc_dists.tolist())
        all_geo_distances.extend(geo_dists.tolist())
    
    neighbor_counts = np.array(neighbor_counts)
    all_geo_euc_ratios = np.array(all_geo_euc_ratios) if all_geo_euc_ratios else np.array([0.0])
    
    stats = {
        "ring": ring_label,
        "num_samples": len(sample_indices),
        "count_mean": np.mean(neighbor_counts),
        "count_std": np.std(neighbor_counts),
        "count_min": np.min(neighbor_counts),
        "count_p50": np.percentile(neighbor_counts, 50),
        "count_p90": np.percentile(neighbor_counts, 90),
        "count_p95": np.percentile(neighbor_counts, 95),
        "count_p99": np.percentile(neighbor_counts, 99),
        "count_max": np.max(neighbor_counts),
        "ratio_mean": np.mean(all_geo_euc_ratios),
        "ratio_std": np.std(all_geo_euc_ratios),
        "ratio_median": np.median(all_geo_euc_ratios),
        "ratio_p90": np.percentile(all_geo_euc_ratios, 90),
        "ratio_p95": np.percentile(all_geo_euc_ratios, 95),
        "ratio_p99": np.percentile(all_geo_euc_ratios, 99),
        "ratio_max": np.max(all_geo_euc_ratios),
    }
    return stats


def compute_outlier_analysis(positions, ring_nbrs_dict, geodesic_distances,
                             ring_label="ring-3", max_sample=5000,
                             multipliers=[2.0, 3.0, 5.0, 7.0, 10.0]):
    """Analyze what fraction of neighbors would be removed at each multiplier."""
    num_pts = len(ring_nbrs_dict)
    sample_indices = np.random.choice(num_pts, min(max_sample, num_pts), replace=False)
    
    results = {m: {"removed": 0, "total": 0, "points_affected": 0} for m in multipliers}
    
    for idx in sample_indices:
        nbrs = ring_nbrs_dict[idx]
        if len(nbrs) == 0:
            continue
        
        nbrs_xyz = positions[nbrs] - positions[idx]
        euc_dists = LA.norm(nbrs_xyz, axis=1)
        geo_dists = geodesic_distances[nbrs]
        
        eps = 1e-8
        ratios = geo_dists / np.maximum(euc_dists, eps)
        median_ratio = np.median(ratios)
        
        for m in multipliers:
            threshold = max(median_ratio * m, 2.0)
            outliers = ratios > threshold
            n_removed = np.sum(outliers)
            results[m]["removed"] += n_removed
            results[m]["total"] += len(nbrs)
            if n_removed > 0:
                results[m]["points_affected"] += 1
    
    return results, len(sample_indices)


def compute_post_filter_stats(positions, ring_nbrs_dict, geodesic_distances,
                              ring_label="ring-3", multiplier=5.0, max_sample=5000):
    """Compute neighbor count stats AFTER outlier filtering + Euclidean trim."""
    num_pts = len(ring_nbrs_dict)
    sample_indices = np.random.choice(num_pts, min(max_sample, num_pts), replace=False)
    
    post_filter_counts = []
    
    for idx in sample_indices:
        nbrs = ring_nbrs_dict[idx]
        if len(nbrs) == 0:
            post_filter_counts.append(0)
            continue
        
        nbrs_xyz = positions[nbrs] - positions[idx]
        euc_dists = LA.norm(nbrs_xyz, axis=1)
        geo_dists = geodesic_distances[nbrs]
        
        eps = 1e-8
        ratios = geo_dists / np.maximum(euc_dists, eps)
        median_ratio = np.median(ratios)
        threshold = max(median_ratio * multiplier, 2.0)
        inlier_mask = ratios <= threshold
        
        post_filter_counts.append(np.sum(inlier_mask))
    
    post_filter_counts = np.array(post_filter_counts)
    return {
        "ring": ring_label,
        "multiplier": multiplier,
        "count_mean": np.mean(post_filter_counts),
        "count_std": np.std(post_filter_counts),
        "count_min": np.min(post_filter_counts),
        "count_p50": np.percentile(post_filter_counts, 50),
        "count_p90": np.percentile(post_filter_counts, 90),
        "count_p95": np.percentile(post_filter_counts, 95),
        "count_p99": np.percentile(post_filter_counts, 99),
        "count_max": np.max(post_filter_counts),
    }


def analyze_single_output(output_path, name, max_sample=5000, rings=[2, 3],
                          n_neighbors=10, adaptive_target_ring=None,
                          adaptive_target_neighbors=None, adaptive_k_boost=20):
    """Analyze ring statistics for a single Gaussian output."""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {name}")
    print(f"  Path: {output_path}")
    print(f"{'='*70}")
    
    # Load Gaussian data
    try:
        data = load_gaussian_data_cpu(Path(output_path))
        positions = data.get_xyz()
        print(f"  Num Gaussians: {len(positions)}")
    except Exception as e:
        print(f"  ERROR loading Gaussian data: {e}")
        return None
    
    # Load geodesic data
    try:
        geo_path = find_geodesic_data(output_path)
        geo_data = np.load(geo_path)
        # Use first source's geodesic distances
        all_geo = geo_data["geodesic_distances"]  # (num_sources, num_gaussians)
        source_idx = 0
        geodesic_distances = all_geo[source_idx]
        print(f"  Num sources in geodesic data: {all_geo.shape[0]}")
        print(f"  Using source {source_idx} for ratio analysis")
    except Exception as e:
        print(f"  WARNING: No geodesic data found ({e}), skipping ratio analysis")
        geodesic_distances = None
    
    # Compute ring neighborhoods
    adaptive_str = ""
    if adaptive_target_ring is not None and adaptive_target_neighbors is not None:
        adaptive_str = f", adaptive ring-{adaptive_target_ring}≥{adaptive_target_neighbors} k_boost={adaptive_k_boost}"
    print(f"  Computing ring neighborhoods (k={n_neighbors}{adaptive_str})...")

    if adaptive_target_ring is not None and adaptive_target_neighbors is not None:
        from GenerateData.utils.data_generation_utils import adaptive_ring1_neighbors
        ring1_nbrs, mean_dist, per_point_dist = adaptive_ring1_neighbors(
            positions, target_ring=adaptive_target_ring,
            target_ring_neighbors=adaptive_target_neighbors,
            n_neighbors_base=n_neighbors, k_boost=adaptive_k_boost,
            use_mahalanobis=False,
        )
    else:
        ring1_nbrs, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            positions, n_neighbors=n_neighbors, use_mahalanobis=False
        )
    
    ring_nbrs_dict = {}
    max_ring = max(rings)
    for r in range(2, max_ring + 1):
        print(f"  Computing ring-{r} neighborhoods...")
        ring_nbrs_dict[r] = {idx: get_neighborhood_by_ring(idx, r, ring1_nbrs) 
                             for idx in range(len(positions))}
    
    print(f"  Mean NN distance: {mean_dist:.6f}")
    
    all_stats = {}
    all_outlier = {}
    all_post_filter = {}
    
    for r in rings:
        ring_label = f"ring-{r}"
        
        # Basic ring stats
        if geodesic_distances is not None:
            stats = compute_ring_stats(
                positions, ring_nbrs_dict[r], geodesic_distances,
                ring_label=ring_label, max_sample=max_sample
            )
        else:
            # Only count stats without geo
            num_pts = len(ring_nbrs_dict[r])
            sample_indices = np.random.choice(num_pts, min(max_sample, num_pts), replace=False)
            counts = np.array([len(ring_nbrs_dict[r][idx]) for idx in sample_indices])
            stats = {
                "ring": ring_label,
                "num_samples": len(sample_indices),
                "count_mean": np.mean(counts),
                "count_std": np.std(counts),
                "count_min": np.min(counts),
                "count_p50": np.percentile(counts, 50),
                "count_p90": np.percentile(counts, 90),
                "count_p95": np.percentile(counts, 95),
                "count_p99": np.percentile(counts, 99),
                "count_max": np.max(counts),
            }
        
        all_stats[r] = stats
        
        print(f"\n  --- {ring_label} Neighbor Count Stats ---")
        print(f"    mean={stats['count_mean']:.1f}  std={stats['count_std']:.1f}")
        print(f"    min={stats['count_min']}  p50={stats['count_p50']:.0f}  "
              f"p90={stats['count_p90']:.0f}  p95={stats['count_p95']:.0f}  "
              f"p99={stats['count_p99']:.0f}  max={stats['count_max']}")
        
        if geodesic_distances is not None and 'ratio_mean' in stats:
            print(f"  --- {ring_label} Geo/Euc Ratio Stats ---")
            print(f"    mean={stats['ratio_mean']:.2f}  median={stats['ratio_median']:.2f}  "
                  f"std={stats['ratio_std']:.2f}")
            print(f"    p90={stats['ratio_p90']:.2f}  p95={stats['ratio_p95']:.2f}  "
                  f"p99={stats['ratio_p99']:.2f}  max={stats['ratio_max']:.2f}")
            
            # Outlier analysis
            outlier_results, n_sample = compute_outlier_analysis(
                positions, ring_nbrs_dict[r], geodesic_distances,
                ring_label=ring_label, max_sample=max_sample
            )
            all_outlier[r] = (outlier_results, n_sample)
            
            print(f"  --- {ring_label} Outlier Removal by Multiplier ---")
            for m in sorted(outlier_results.keys()):
                res = outlier_results[m]
                pct = 100 * res["removed"] / max(res["total"], 1)
                pts_pct = 100 * res["points_affected"] / max(n_sample, 1)
                print(f"    {m:5.1f}× median:  {res['removed']:6d}/{res['total']:6d} "
                      f"nbrs removed ({pct:5.2f}%), "
                      f"{res['points_affected']:5d}/{n_sample:5d} points affected ({pts_pct:.1f}%)")
            
            # Post-filter count stats (at 5.0× multiplier)
            pf_stats = compute_post_filter_stats(
                positions, ring_nbrs_dict[r], geodesic_distances,
                ring_label=ring_label, multiplier=5.0, max_sample=max_sample
            )
            all_post_filter[r] = pf_stats
            
            print(f"  --- {ring_label} Post-Filter Count (5.0× multiplier) ---")
            print(f"    mean={pf_stats['count_mean']:.1f}  std={pf_stats['count_std']:.1f}")
            print(f"    min={pf_stats['count_min']}  p50={pf_stats['count_p50']:.0f}  "
                  f"p90={pf_stats['count_p90']:.0f}  p95={pf_stats['count_p95']:.0f}  "
                  f"p99={pf_stats['count_p99']:.0f}  max={pf_stats['count_max']}")
    
    return {
        "name": name,
        "num_gaussians": len(positions),
        "mean_nn_dist": mean_dist,
        "ring_stats": all_stats,
        "outlier_analysis": all_outlier,
        "post_filter_stats": all_post_filter,
    }


# ============================================================================
# Adaptive kNN over-cap analysis (parallelised ring computation)
# ============================================================================


def _load_positions(output_path):
    """Load Gaussian positions from an output directory."""
    data = load_gaussian_data_cpu(Path(output_path))
    return data.get_xyz()


def _adaptive_single_worker(args_tuple):
    """Picklable wrapper for :func:`analyze_adaptive_single` (multiprocessing)."""
    return analyze_adaptive_single(*args_tuple)


def analyze_adaptive_single(output_path, name, n_neighbors_base, k_boost,
                            target_ring, target_neighbors,
                            adaptive_max_mean_cut, adaptive_max_steps):
    """Run base vs adaptive comparison for one Gaussian output.

    Returns a dict with all relevant statistics.
    """
    positions = _load_positions(output_path)
    N = len(positions)

    # ── Base ring-1 and ring counts ──────────────────────────────
    ring1_base, _, _ = ring1_neighbors_gaussians(
        positions, n_neighbors=n_neighbors_base, use_mahalanobis=False,
    )
    base_counts = ring_counts_sparse(ring1_base, N, target_ring)

    # ── Adaptive ring-1 and ring counts ──────────────────────────
    ring1_adaptive, _, _ = adaptive_ring1_neighbors(
        positions,
        target_ring=target_ring,
        target_ring_neighbors=target_neighbors,
        n_neighbors_base=n_neighbors_base,
        k_boost=k_boost,
        adaptive_max_mean_cut=adaptive_max_mean_cut,
        adaptive_max_steps=adaptive_max_steps,
        use_mahalanobis=False,
    )
    adaptive_counts = ring_counts_sparse(ring1_adaptive, N, target_ring)

    # ── Per-point ring-1 k values ────────────────────────────────
    adaptive_k_values = np.array([len(ring1_adaptive[i]) for i in range(N)])

    # ── Statistics ───────────────────────────────────────────────
    def _stats(counts, label):
        over = int((counts > target_neighbors).sum())
        under = int((counts < target_neighbors).sum())
        exact = int((counts == target_neighbors).sum())
        over_vals = counts[counts > target_neighbors]
        under_vals = counts[counts < target_neighbors]
        cuts = np.maximum(counts.astype(np.int64) - target_neighbors, 0)
        return {
            "label": label,
            "N": N,
            "over": over,
            "over_pct": 100 * over / N,
            "under": under,
            "under_pct": 100 * under / N,
            "exact": exact,
            "mean_count": float(counts.mean()),
            "median_count": float(np.median(counts)),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "p5": float(np.percentile(counts, 5)),
            "p50": float(np.percentile(counts, 50)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
            "mean_cut": float(cuts.mean()),
            "over_mean": float(over_vals.mean()) if len(over_vals) else 0,
            "over_median": float(np.median(over_vals)) if len(over_vals) else 0,
            "over_max": int(over_vals.max()) if len(over_vals) else 0,
        }

    base_stats = _stats(base_counts, f"base k={n_neighbors_base}")
    adaptive_stats = _stats(adaptive_counts, f"adaptive k_boost={k_boost}")

    return {
        "name": name,
        "N": N,
        "base": base_stats,
        "adaptive": adaptive_stats,
        "adaptive_k_mean": float(adaptive_k_values.mean()),
        "adaptive_k_min": int(adaptive_k_values.min()),
        "adaptive_k_max": int(adaptive_k_values.max()),
    }


def _full_single_worker(args_tuple):
    """Picklable wrapper for :func:`analyze_single_output` (multiprocessing)."""
    return analyze_single_output(*args_tuple)


def _print_adaptive_table_row(name, stats, target):
    """Print a single row of the adaptive comparison table."""
    b = stats["base"]
    a = stats["adaptive"]
    print(f"  {name:<30s}  N={stats['N']:>6,d}")
    print(f"    {'':30s}  {'over':>7s}  {'pct':>6s}  {'under':>7s}  "
          f"{'mean':>6s}  {'p95':>6s}  {'max':>6s}  {'mean_cut':>8s}")
    print(f"    {'BASE  k=' + str(b['label'].split('=')[1]):30s}  "
          f"{b['over']:7,d}  {b['over_pct']:5.1f}%  {b['under']:7,d}  "
          f"{b['mean_count']:6.1f}  {b['p95']:6.0f}  {b['max_count']:6d}  "
          f"{b['mean_cut']:8.2f}")
    print(f"    {'ADAPTIVE k_boost=' + str(a['label'].split('=')[1]):30s}  "
          f"{a['over']:7,d}  {a['over_pct']:5.1f}%  {a['under']:7,d}  "
          f"{a['mean_count']:6.1f}  {a['p95']:6.0f}  {a['max_count']:6d}  "
          f"{a['mean_cut']:8.2f}")
    if a["over"] > 0:
        print(f"      overshoot: mean={a['over_mean']:.1f}  "
              f"median={a['over_median']:.0f}  max={a['over_max']}")
    print(f"      per-point k: mean={stats['adaptive_k_mean']:.1f}  "
          f"min={stats['adaptive_k_min']}  max={stats['adaptive_k_max']}")
    print()


def print_adaptive_report(results, dataset_label, args):
    """Print a formatted adaptive analysis report for a dataset."""
    print()
    print("=" * 86)
    print(f"  ADAPTIVE kNN ANALYSIS — {dataset_label}")
    print(f"  base k={args.n_neighbors}  k_boost={args.adaptive_k_boost}  "
          f"target ring-{args.adaptive_target_ring} ≤ {args.adaptive_target_neighbors}  "
          f"max_mean_cut={args.adaptive_max_mean_cut}  max_steps={args.adaptive_max_steps}")
    print("=" * 86)

    for r in results:
        _print_adaptive_table_row(r["name"], r, args.adaptive_target_neighbors)

    # Summary
    all_over_pct = [r["adaptive"]["over_pct"] for r in results]
    all_mean_cut = [r["adaptive"]["mean_cut"] for r in results]
    all_max = [r["adaptive"]["max_count"] for r in results]
    print("-" * 86)
    print(f"  SUMMARY across {len(results)} outputs:")
    print(f"    over-{args.adaptive_target_neighbors} pct:  "
          f"mean={np.mean(all_over_pct):.2f}%  max={np.max(all_over_pct):.2f}%")
    print(f"    mean cut:       mean={np.mean(all_mean_cut):.2f}  "
          f"max={np.max(all_mean_cut):.2f}")
    print(f"    absolute max ring-{args.adaptive_target_ring} count: "
          f"{np.max(all_max)}")
    print("=" * 86)
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze ring statistics")
    parser.add_argument("--mode", default="full", choices=["full", "adaptive"],
                        help="'full' = legacy ring statistics, 'adaptive' = over-cap comparison")
    parser.add_argument("--dataset", default="both", choices=["tosca", "polynomial", "both"],
                        help="Which dataset to analyze")
    parser.add_argument("--shapes", default="", help="Comma-separated TOSCA shapes (default: all)")
    parser.add_argument("--surfaces", default="", help="Comma-separated polynomial surfaces (default: all)")
    parser.add_argument("--max_sample", type=int, default=5000,
                        help="Max points to sample per shape for analysis")
    parser.add_argument("--rings", default="2,3", help="Comma-separated ring levels to analyze")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="Base ring-1 neighbor count (default: 10)")
    parser.add_argument("--adaptive_target_ring", type=int, default=None,
                        help="Ring level to optimize with adaptive kNN (e.g. 3)")
    parser.add_argument("--adaptive_target_neighbors", type=int, default=None,
                        help="Desired ring-k count for adaptive kNN (e.g. 128)")
    parser.add_argument("--tosca_adaptive_target_neighbors", type=int, default=None,
                        help="Override adaptive_target_neighbors for TOSCA (default: same as --adaptive_target_neighbors)")
    parser.add_argument("--adaptive_k_boost", type=int, default=20,
                        help="Max boosted k for adaptive kNN (default: 20)")
    parser.add_argument("--adaptive_max_mean_cut", type=float, default=2.0,
                        help="Stop binary search when mean cut <= this (default: 2.0)")
    parser.add_argument("--adaptive_max_steps", type=int, default=5,
                        help="Max binary search iterations (default: 5)")
    args = parser.parse_args()
    
    rings = [int(r) for r in args.rings.split(",")]
    np.random.seed(42)

    # ================================================================
    # Adaptive mode — base vs adaptive comparison
    # ================================================================
    if args.mode == "adaptive":
        if args.adaptive_target_ring is None or args.adaptive_target_neighbors is None:
            parser.error("--adaptive_target_ring and --adaptive_target_neighbors are required "
                         "for --mode adaptive")

        all_adaptive_results = {}

        n_workers = min(multiprocessing.cpu_count() - 1, 70)

        if args.dataset in ("tosca", "both"):
            tosca_target = args.tosca_adaptive_target_neighbors or args.adaptive_target_neighbors
            shapes = args.shapes.split(",") if args.shapes else ALL_TOSCA_SHAPES
            shapes = [s.strip() for s in shapes if s.strip()]
            tasks = []
            for shape in shapes:
                try:
                    output_path = find_tosca_output(shape)
                    tasks.append((
                        output_path, f"TOSCA/{shape}",
                        args.n_neighbors, args.adaptive_k_boost,
                        args.adaptive_target_ring, tosca_target,
                        args.adaptive_max_mean_cut, args.adaptive_max_steps,
                    ))
                except FileNotFoundError as e:
                    print(f"  SKIP {shape}: {e}")
            tosca_results = []
            if tasks:
                with multiprocessing.Pool(min(n_workers, len(tasks))) as pool:
                    tosca_results = pool.map(_adaptive_single_worker, tasks)
            if tosca_results:
                # Override target_neighbors for display
                saved = args.adaptive_target_neighbors
                args.adaptive_target_neighbors = tosca_target
                print_adaptive_report(tosca_results, "TOSCA", args)
                args.adaptive_target_neighbors = saved
                all_adaptive_results["tosca"] = tosca_results

        if args.dataset in ("polynomial", "both"):
            surfaces = args.surfaces.split(",") if args.surfaces else ALL_POLY_SURFACES
            surfaces = [s.strip() for s in surfaces if s.strip()]
            tasks = []
            for surface in surfaces:
                for level in POLY_LEVELS:
                    for light in POLY_LIGHTS:
                        try:
                            output_path = find_poly_output(surface, level, light)
                            tasks.append((
                                output_path, f"Poly/{surface}/{level}/{light}",
                                args.n_neighbors, args.adaptive_k_boost,
                                args.adaptive_target_ring, args.adaptive_target_neighbors,
                                args.adaptive_max_mean_cut, args.adaptive_max_steps,
                            ))
                        except FileNotFoundError as e:
                            print(f"  SKIP {surface}/{level}/{light}: {e}")
            poly_results = []
            if tasks:
                with multiprocessing.Pool(min(n_workers, len(tasks))) as pool:
                    poly_results = pool.map(_adaptive_single_worker, tasks)
            if poly_results:
                print_adaptive_report(poly_results, "Polynomial", args)
                all_adaptive_results["polynomial"] = poly_results

        return

    # ================================================================
    # Full mode (legacy) — detailed ring statistics
    # ================================================================
    n_workers = min(multiprocessing.cpu_count() - 1, 70)
    all_results = {}
    
    # ---- TOSCA analysis ----
    if args.dataset in ("tosca", "both"):
        shapes = args.shapes.split(",") if args.shapes else ALL_TOSCA_SHAPES
        shapes = [s.strip() for s in shapes if s.strip()]
        
        print("\n" + "#"*70)
        print("#  TOSCA Ring Neighbor Analysis")
        print("#"*70)
        
        tasks = []
        for shape in shapes:
            try:
                output_path = find_tosca_output(shape)
                tasks.append((
                    output_path, f"TOSCA/{shape}",
                    args.max_sample, rings,
                    args.n_neighbors, args.adaptive_target_ring,
                    args.adaptive_target_neighbors, args.adaptive_k_boost,
                ))
            except FileNotFoundError as e:
                print(f"\n  SKIP {shape}: {e}")
        
        if tasks:
            pool_size = min(n_workers, len(tasks))
            if pool_size > 1:
                with multiprocessing.Pool(pool_size) as pool:
                    tosca_results = [r for r in pool.map(_full_single_worker, tasks) if r]
            else:
                tosca_results = [r for r in map(_full_single_worker, tasks) if r]
        else:
            tosca_results = []
        
        if tosca_results:
            print("\n" + "="*70)
            print("  TOSCA SUMMARY (across all shapes)")
            print("="*70)
            for r in rings:
                ring_label = f"ring-{r}"
                raw_counts = [res["ring_stats"][r]["count_p99"] for res in tosca_results 
                              if r in res["ring_stats"]]
                raw_max = [res["ring_stats"][r]["count_max"] for res in tosca_results
                           if r in res["ring_stats"]]
                raw_mean = [res["ring_stats"][r]["count_mean"] for res in tosca_results
                            if r in res["ring_stats"]]
                
                print(f"\n  {ring_label} (raw, before outlier filter):")
                print(f"    Across shapes: mean of means = {np.mean(raw_mean):.1f}")
                print(f"    Across shapes: mean of p99   = {np.mean(raw_counts):.1f} "
                      f"(range: {np.min(raw_counts):.0f}-{np.max(raw_counts):.0f})")
                print(f"    Across shapes: mean of max   = {np.mean(raw_max):.1f} "
                      f"(range: {np.min(raw_max):.0f}-{np.max(raw_max):.0f})")
                
                pf_counts = [res["post_filter_stats"][r]["count_p99"] 
                             for res in tosca_results if r in res.get("post_filter_stats", {})]
                pf_max = [res["post_filter_stats"][r]["count_max"] 
                          for res in tosca_results if r in res.get("post_filter_stats", {})]
                pf_mean = [res["post_filter_stats"][r]["count_mean"] 
                           for res in tosca_results if r in res.get("post_filter_stats", {})]
                
                if pf_counts:
                    print(f"\n  {ring_label} (after 5.0× outlier filter):")
                    print(f"    Across shapes: mean of means = {np.mean(pf_mean):.1f}")
                    print(f"    Across shapes: mean of p99   = {np.mean(pf_counts):.1f} "
                          f"(range: {np.min(pf_counts):.0f}-{np.max(pf_counts):.0f})")
                    print(f"    Across shapes: mean of max   = {np.mean(pf_max):.1f} "
                          f"(range: {np.min(pf_max):.0f}-{np.max(pf_max):.0f})")
                
                # Recommendation
                if pf_counts:
                    recommended = int(np.ceil(np.max(pf_counts) / 16) * 16)  # round up to 16
                    recommended = max(recommended, 32)
                    print(f"\n    >>> RECOMMENDED ring_size_mapping[{r}] = {recommended}")
                    print(f"        (based on max p99 across shapes = {np.max(pf_counts):.0f})")
            
            all_results["tosca"] = tosca_results
    
    # ---- Polynomial analysis ----
    if args.dataset in ("polynomial", "both"):
        surfaces = args.surfaces.split(",") if args.surfaces else ALL_POLY_SURFACES
        surfaces = [s.strip() for s in surfaces if s.strip()]
        
        print("\n" + "#"*70)
        print("#  Polynomial Ring Neighbor Analysis")
        print("#"*70)
        
        tasks = []
        for surface in surfaces:
            # Analyze at level_03 as representative
            for level in ["level_03"]:
                for light in ["light_0"]:
                    try:
                        output_path = find_poly_output(surface, level, light)
                        tasks.append((
                            output_path, f"Polynomial/{surface}/{level}/{light}",
                            args.max_sample, rings,
                            args.n_neighbors, args.adaptive_target_ring,
                            args.adaptive_target_neighbors, args.adaptive_k_boost,
                        ))
                    except FileNotFoundError as e:
                        print(f"\n  SKIP {surface}/{level}/{light}: {e}")
        
        if tasks:
            pool_size = min(n_workers, len(tasks))
            if pool_size > 1:
                with multiprocessing.Pool(pool_size) as pool:
                    poly_results = [r for r in pool.map(_full_single_worker, tasks) if r]
            else:
                poly_results = [r for r in map(_full_single_worker, tasks) if r]
        else:
            poly_results = []
        
        if poly_results:
            print("\n" + "="*70)
            print("  POLYNOMIAL SUMMARY (across all surfaces)")
            print("="*70)
            for r in rings:
                ring_label = f"ring-{r}"
                raw_counts = [res["ring_stats"][r]["count_p99"] for res in poly_results
                              if r in res["ring_stats"]]
                raw_max = [res["ring_stats"][r]["count_max"] for res in poly_results
                           if r in res["ring_stats"]]
                raw_mean = [res["ring_stats"][r]["count_mean"] for res in poly_results
                            if r in res["ring_stats"]]
                
                print(f"\n  {ring_label} (raw, before outlier filter):")
                print(f"    Across surfaces: mean of means = {np.mean(raw_mean):.1f}")
                print(f"    Across surfaces: mean of p99   = {np.mean(raw_counts):.1f} "
                      f"(range: {np.min(raw_counts):.0f}-{np.max(raw_counts):.0f})")
                print(f"    Across surfaces: mean of max   = {np.mean(raw_max):.1f} "
                      f"(range: {np.min(raw_max):.0f}-{np.max(raw_max):.0f})")
                
                pf_counts = [res["post_filter_stats"][r]["count_p99"]
                             for res in poly_results if r in res.get("post_filter_stats", {})]
                pf_max = [res["post_filter_stats"][r]["count_max"]
                          for res in poly_results if r in res.get("post_filter_stats", {})]
                pf_mean = [res["post_filter_stats"][r]["count_mean"]
                           for res in poly_results if r in res.get("post_filter_stats", {})]
                
                if pf_counts:
                    print(f"\n  {ring_label} (after 5.0× outlier filter):")
                    print(f"    Across surfaces: mean of means = {np.mean(pf_mean):.1f}")
                    print(f"    Across surfaces: mean of p99   = {np.mean(pf_counts):.1f} "
                          f"(range: {np.min(pf_counts):.0f}-{np.max(pf_counts):.0f})")
                    print(f"    Across surfaces: mean of max   = {np.mean(pf_max):.1f} "
                          f"(range: {np.min(pf_max):.0f}-{np.max(pf_max):.0f})")
                
                # Recommendation
                if pf_counts:
                    recommended = int(np.ceil(np.max(pf_counts) / 16) * 16)
                    recommended = max(recommended, 32)
                    print(f"\n    >>> RECOMMENDED ring_size_mapping[{r}] = {recommended}")
                    print(f"        (based on max p99 across surfaces = {np.max(pf_counts):.0f})")
            
            all_results["polynomial"] = poly_results
    
    # ---- Final recommendation summary ----
    print("\n" + "#"*70)
    print("#  FINAL RECOMMENDATIONS")
    print("#"*70)
    
    for dataset_name, results in all_results.items():
        print(f"\n  {dataset_name.upper()}:")
        for r in rings:
            pf_p99 = [res["post_filter_stats"][r]["count_p99"]
                      for res in results if r in res.get("post_filter_stats", {})]
            pf_max_vals = [res["post_filter_stats"][r]["count_max"]
                           for res in results if r in res.get("post_filter_stats", {})]
            if pf_p99:
                rec = int(np.ceil(np.max(pf_p99) / 16) * 16)
                rec = max(rec, 32)
                print(f"    ring-{r}: recommended = {rec}")
                print(f"             (max p99 = {np.max(pf_p99):.0f}, "
                      f"absolute max = {np.max(pf_max_vals):.0f})")
    
    print("\n  Outlier threshold: 5.0× median (with min bound 2.0)")
    print("  This is the currently configured value in training_patches_helpers.py")
    print()


if __name__ == "__main__":
    main()
