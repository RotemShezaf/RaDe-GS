#!/usr/bin/env python3
"""
Analyze the number of valid ring neighbors after outlier filtering.

For each shape/source and ring level, samples random center points and
applies the same outlier filtering logic used by
``DataSets/utils/training_patches_helpers.py::create_train_example``.
Reports how many neighbors survive filtering and how many are removed.

Supports reading outlier parameters from a dataset config YAML
(the same configs used by ``create_gaussian_training_patches.py``).

Usage:
    # TOSCA shapes, default outlier params:
    python DataSets/analyze_outlier_filtering.py --dataset tosca

    # Read outlier params from a config file:
    python DataSets/analyze_outlier_filtering.py --config DataSets/configs/tosca/outlier_filtering/tosca_cat.yaml

    # Override specific params on CLI:
    python DataSets/analyze_outlier_filtering.py --dataset tosca --shapes cat2 --outlier_median_multiplier 5.0
"""
import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as LA

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.data_generation_utils import (
    get_all_points_nbrs_all_rings,
)
from GenerateData.utils.load_utils import load_gaussian_data_cpu
from DataSets.utils.config_utils import (
    load_config,
    resolve_gaussian_outputs,
)
from DataSets.utils.training_patches_helpers import _filter_outliers_without_pu

logger = logging.getLogger(__name__)

# ============================================================================
# Dataset Layouts
# ============================================================================
TOSCA_BASE = "TrainData/TOSCA/SyntheticColmapData/blue_texture"
POLY_BASE = "TrainData/Polynomial/SyntheticColmapData/blue_texture"


def discover_tosca_shapes():
    """List TOSCA shapes that have geodesic data."""
    base = Path(TOSCA_BASE)
    shapes = []
    if not base.exists():
        return shapes
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        geo = d / "high_res" / "decoupled_appearance" / "output" / "geodesic_distance" / "gt_geodesic.npz"
        if geo.exists():
            shapes.append(d.name)
    return shapes


def discover_polynomial_shapes(surfaces=None):
    """List polynomial shapes that have geodesic data."""
    base = Path(POLY_BASE)
    shapes = []
    if not base.exists():
        return shapes
    for surface_dir in sorted(base.iterdir()):
        if not surface_dir.is_dir():
            continue
        if surfaces and surface_dir.name not in surfaces:
            continue
        for level_dir in sorted(surface_dir.iterdir()):
            if not level_dir.is_dir():
                continue
            for light_dir in sorted(level_dir.iterdir()):
                if not light_dir.is_dir():
                    continue
                geo = light_dir / "output" / "geodesic_distance" / "gt_geodesic.npz"
                if geo.exists():
                    shapes.append(f"{surface_dir.name}/{level_dir.name}/{light_dir.name}")
    return shapes


# ============================================================================
# Vectorized outlier filtering (mirrors training_patches_helpers.py logic)
# ============================================================================

def _pad_neighbor_indices(nbrs_dict, point_indices, pad_value=-1):
    """Build a padded (P, max_K) int array from a ragged neighbor dict.

    ``pad_value`` fills slots beyond each point's actual neighbor count.
    Also returns ``counts`` (P,) with the true neighbor count per row and
    ``valid`` (P, max_K) bool mask.
    """
    rows = [nbrs_dict.get(int(pt), np.array([], dtype=int)) for pt in point_indices]
    max_k = max((len(r) for r in rows), default=0)
    if max_k == 0:
        P = len(point_indices)
        return (np.full((P, 1), pad_value, dtype=np.int64),
                np.zeros(P, dtype=np.int64),
                np.zeros((P, 1), dtype=bool))
    padded = np.full((len(rows), max_k), pad_value, dtype=np.int64)
    counts = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        n = len(r)
        counts[i] = n
        if n > 0:
            padded[i, :n] = r
    valid = np.arange(max_k)[np.newaxis, :] < counts[:, np.newaxis]
    return padded, counts, valid


def compute_outlier_stats_vectorized(
    analysis_pts,
    geodesic_distances,
    ring_nbrs,
    ring1_nbrs,
    positions,
    outlier_median_multiplier,
    outlier_threshold_floor,
    outlier_hard_cap,
    outlier_fallback_multiplier,
    outlier_fallback_floor,
):
    """Vectorized outlier-filtering statistics for many points at once.

    Mirrors the training-path filter from ``create_train_example`` but
    operates on all *analysis_pts* simultaneously using padded arrays,
    avoiding any Python-level per-point loop.

    Returns dict with arrays of length P (one entry per analysis point):
        all_n_before, all_n_after, all_n_removed,
        all_thresholds, all_median_discs, all_fallback, all_zero_after
    """
    eps = 1e-8
    pts = np.asarray(analysis_pts, dtype=np.int64)
    P = len(pts)

    # --- Build padded neighbor index arrays ---
    rk_idx, rk_counts, rk_valid = _pad_neighbor_indices(ring_nbrs, pts)
    r1_idx, r1_counts, r1_valid = _pad_neighbor_indices(ring1_nbrs, pts)

    # Filter out points with zero ring-k neighbors
    has_nbrs = rk_counts > 0
    # Keep only valid analysis points
    if not has_nbrs.all():
        keep = has_nbrs
        pts = pts[keep]
        rk_idx = rk_idx[keep]; rk_counts = rk_counts[keep]; rk_valid = rk_valid[keep]
        r1_idx = r1_idx[keep]; r1_counts = r1_counts[keep]; r1_valid = r1_valid[keep]
        P = len(pts)

    if P == 0:
        empty = np.array([], dtype=np.float64)
        return {
            'all_n_before': empty, 'all_n_after': empty,
            'all_n_removed': empty, 'all_thresholds': empty,
            'all_median_discs': empty, 'n_fallback': 0, 'n_zero_after': 0,
        }

    # Center positions & geodesics: (P, 3), (P,)
    p_xyz = positions[pts]                          # (P, 3)
    p_u = geodesic_distances[pts]                   # (P,)

    # --- Ring-k neighbor data (padded) ---
    # Clamp pad indices to 0 for safe gather; masked out later
    safe_rk = np.where(rk_valid, rk_idx, 0)
    nbrs_xyz = positions[safe_rk] - p_xyz[:, np.newaxis, :]  # (P, Kmax, 3)
    nbrs_euc = LA.norm(nbrs_xyz, axis=2)                     # (P, Kmax)
    nbrs_u = geodesic_distances[safe_rk]                      # (P, Kmax)

    # --- Ring-1 neighbor data (padded) ---
    safe_r1 = np.where(r1_valid, r1_idx, 0)
    r1_xyz = positions[safe_r1] - p_xyz[:, np.newaxis, :]    # (P, K1max, 3)
    r1_euc = LA.norm(r1_xyz, axis=2)                          # (P, K1max)
    r1_u = geodesic_distances[safe_r1]                        # (P, K1max)

    # --- Geo discrepancy ---
    geo_disc = np.abs(nbrs_u - p_u[:, np.newaxis]) / np.maximum(nbrs_euc, eps)  # (P, Kmax)
    r1_disc = np.abs(r1_u - p_u[:, np.newaxis]) / np.maximum(r1_euc, eps)       # (P, K1max)

    # Mean of ring-1 discrepancies (only over valid slots)
    r1_disc_masked = np.where(r1_valid, r1_disc, 0.0)
    r1_sum = r1_disc_masked.sum(axis=1)          # (P,)
    r1_cnt = r1_counts.astype(np.float64)
    median_discs = np.where(r1_cnt > 0, r1_sum / r1_cnt, 0.0)  # (P,)

    # Adaptive threshold per point
    adaptive_thresh = np.maximum(median_discs * outlier_median_multiplier,
                                 outlier_threshold_floor)
    outlier_thresh = np.minimum(adaptive_thresh, outlier_hard_cap)  # (P,)

    # Inlier mask: (P, Kmax) – True where neighbor passes AND slot is valid
    inlier = rk_valid & (geo_disc <= outlier_thresh[:, np.newaxis])
    n_inlier = inlier.sum(axis=1)         # (P,)
    n_removed = rk_counts - n_inlier      # (P,)

    # --- Fallback for points where primary filter removed everything ---
    needs_fallback = (n_inlier == 0) & (rk_counts > 0)
    if needs_fallback.any():
        fallback_thresh = np.maximum(
            median_discs * outlier_fallback_multiplier,
            outlier_fallback_floor,
        )
        fallback_inlier = rk_valid & (geo_disc <= fallback_thresh[:, np.newaxis])
        fb_n_inlier = fallback_inlier.sum(axis=1)
        # Update only fallback rows
        n_inlier = np.where(needs_fallback, fb_n_inlier, n_inlier)
        n_removed = rk_counts - n_inlier

    all_n_before = rk_counts.astype(np.float64)
    all_n_after = n_inlier.astype(np.float64)
    all_n_removed = n_removed.astype(np.float64)

    return {
        'all_n_before': all_n_before,
        'all_n_after': all_n_after,
        'all_n_removed': all_n_removed,
        'all_thresholds': outlier_thresh,
        'all_median_discs': median_discs,
        'n_fallback': int(needs_fallback.sum()),
        'n_zero_after': int((all_n_after == 0).sum()),
    }


def compute_outlier_stats_without_pu(
    analysis_pts,
    geodesic_distances,
    ring_nbrs,
    ring1_nbrs,
    positions,
    sentinel=1e10,
    max_removal_fraction=None,
):
    """Outlier-filtering statistics using the inference-time filter
    (``_filter_outliers_without_pu``) which does NOT use p_u.

    To simulate realistic inference conditions, only neighbors whose
    geodesic distance is **<= p_u** (the center point's own geodesic) are
    considered to have valid geodesic data.  Neighbors with geodesic > p_u
    are assigned the *sentinel* value so that
    ``_filter_outliers_without_pu`` treats them as unvisited.

    Returns dict with arrays of length P:
        all_n_before, all_n_after, all_n_removed, n_zero_after
    """
    pts = np.asarray(analysis_pts, dtype=np.int64)
    P = len(pts)

    all_n_before = np.empty(P, dtype=np.float64)
    all_n_after = np.empty(P, dtype=np.float64)
    valid_count = 0

    for i, pt in enumerate(pts):
        nbrs = ring_nbrs.get(int(pt), np.array([], dtype=int))
        r1 = ring1_nbrs.get(int(pt), np.array([], dtype=int))
        n = len(nbrs)
        if n == 0:
            continue
        p_xyz = positions[pt]
        p_u = geodesic_distances[pt]

        nbrs_xyz = positions[nbrs] - p_xyz
        nbrs_u = geodesic_distances[nbrs].copy()
        nbrs_u[nbrs_u > p_u] = sentinel  # mark as unvisited

        r1_xyz = positions[r1] - p_xyz
        r1_u = geodesic_distances[r1].copy()
        r1_u[r1_u > p_u] = sentinel  # mark as unvisited

        inlier_mask = _filter_outliers_without_pu(
            nbrs_xyz, nbrs_u, r1_xyz, r1_u, sentinel=sentinel,
            max_removal_fraction=max_removal_fraction,
        )
        all_n_before[valid_count] = n
        all_n_after[valid_count] = int(inlier_mask.sum())
        valid_count += 1

    all_n_before = all_n_before[:valid_count]
    all_n_after = all_n_after[:valid_count]
    all_n_removed = all_n_before - all_n_after

    return {
        'all_n_before': all_n_before,
        'all_n_after': all_n_after,
        'all_n_removed': all_n_removed,
        'n_zero_after': int((all_n_after == 0).sum()),
    }


# ============================================================================
# Core Analysis
# ============================================================================

def analyze_shape(
    shape_name,
    gaussian_output,
    ring,
    n_neighbors,
    use_mahalanobis,
    num_sources,
    num_sample_points,
    seed,
    outlier_median_multiplier,
    outlier_threshold_floor,
    outlier_hard_cap,
    outlier_fallback_multiplier,
    outlier_fallback_floor,
    adaptive_target_ring=None,
    adaptive_target_neighbors=None,
    adaptive_k_boost=20,
    adaptive_max_mean_cut=2.0,
    adaptive_max_steps=5,
    outlier_max_removal_fraction=None,
):
    """Analyze outlier filtering statistics for a single shape."""
    gdata = load_gaussian_data_cpu(Path(gaussian_output))
    positions = gdata.get_xyz()
    scales = gdata.get_scaling()
    rotations = gdata.get_rotation()
    N = len(positions)

    # Load geodesic data
    geo_path = Path(gaussian_output) / "geodesic_distance" / "gt_geodesic.npz"
    if not geo_path.exists():
        logger.warning("[SKIP] %s: no geodesic data at %s", shape_name, geo_path)
        return None
    geo_data = np.load(str(geo_path))
    all_geodesic_distances = geo_data['geodesic_distances']  # (S, N)
    source_gaussian_indices = geo_data['source_gaussian_indices']  # (S,)
    num_available_sources = len(source_gaussian_indices)

    # Build ring neighborhoods using get_all_points_nbrs_all_rings
    logger.info("%s: %d gaussians, %d sources, ring=%d", shape_name, N,
                num_available_sources, ring)

    ring1_nbrs_arr, ring2_nbrs_arr, ring3_nbrs_arr, ring4_nbrs_arr, \
        mean_dist, per_point_dist = get_all_points_nbrs_all_rings(
            positions,
            use_mahalanobis=use_mahalanobis,
            gaussian_scales=scales if use_mahalanobis else None,
            gaussian_rotations=rotations if use_mahalanobis else None,
            n_neighbors_ring1=n_neighbors,
            adaptive_target_ring=adaptive_target_ring,
            adaptive_target_neighbors=adaptive_target_neighbors,
            adaptive_k_boost=adaptive_k_boost,
            adaptive_max_mean_cut=adaptive_max_mean_cut,
            adaptive_max_steps=adaptive_max_steps,
        )

    ring_nbrs_dict = {1: ring1_nbrs_arr, 2: ring2_nbrs_arr,
                      3: ring3_nbrs_arr, 4: ring4_nbrs_arr}

    # Select which ring to analyze
    target_ring_nbrs = ring_nbrs_dict[ring]

    # Exclude source gaussians from analysis
    all_source_set = set(source_gaussian_indices.tolist())
    analysis_pts = [pt for pt in range(N) if pt not in all_source_set
                    and len(target_ring_nbrs.get(pt, np.array([], dtype=int))) > 0]
    rng = np.random.RandomState(seed)

    # Limit analysis points if requested
    if num_sample_points > 0 and num_sample_points < len(analysis_pts):
        analysis_pts = rng.choice(analysis_pts, num_sample_points, replace=False).tolist()

    n_analysis = len(analysis_pts)
    n_srcs = min(num_sources, num_available_sources)

    # Pick sources for multi-source analysis
    if n_srcs >= num_available_sources:
        selected_srcs = np.arange(num_available_sources)
    else:
        selected_srcs = rng.choice(num_available_sources, n_srcs, replace=False)

    # Compute min geodesic distance across selected sources
    min_distances = all_geodesic_distances[selected_srcs].min(axis=0)

    # --- Training-path filter (uses p_u) ---
    vstats = compute_outlier_stats_vectorized(
        analysis_pts, min_distances, target_ring_nbrs, ring1_nbrs_arr,
        positions, outlier_median_multiplier, outlier_threshold_floor,
        outlier_hard_cap, outlier_fallback_multiplier, outlier_fallback_floor,
    )
    all_n_before = vstats['all_n_before']
    all_n_after = vstats['all_n_after']
    all_n_removed = vstats['all_n_removed']
    all_thresholds = vstats['all_thresholds']
    all_median_discs = vstats['all_median_discs']
    n_fallback = vstats['n_fallback']
    n_zero_after = vstats['n_zero_after']

    # --- Inference-path filter (without p_u) ---
    vstats_no_pu = compute_outlier_stats_without_pu(
        analysis_pts, min_distances, target_ring_nbrs, ring1_nbrs_arr,
        positions, max_removal_fraction=outlier_max_removal_fraction,
    )

    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    def _summarize(arr):
        return {
            'percentiles': {p: float(np.percentile(arr, p)) for p in pcts},
            'mean': float(arr.mean()),
        }

    # Without-pu results
    no_pu_before = vstats_no_pu['all_n_before']
    no_pu_after = vstats_no_pu['all_n_after']
    no_pu_removed = vstats_no_pu['all_n_removed']

    return {
        'shape': shape_name,
        'num_gaussians': N,
        'ring': ring,
        'num_sources_used': n_srcs,
        'n_analysis_points': n_analysis,
        'n_evaluated': len(all_n_before),
        'outlier_params': {
            'median_multiplier': outlier_median_multiplier,
            'threshold_floor': outlier_threshold_floor,
            'hard_cap': outlier_hard_cap,
            'fallback_multiplier': outlier_fallback_multiplier,
            'fallback_floor': outlier_fallback_floor,
        },
        # Training-path (with p_u)
        'n_before': _summarize(all_n_before),
        'n_after': _summarize(all_n_after),
        'n_removed': _summarize(all_n_removed),
        'removal_fraction': _summarize(all_n_removed / np.maximum(all_n_before, 1)),
        'thresholds': _summarize(all_thresholds),
        'median_discs': _summarize(all_median_discs),
        'n_fallback': n_fallback,
        'n_zero_after': n_zero_after,
        # Inference-path (without p_u)
        'no_pu_n_evaluated': len(no_pu_before),
        'no_pu_n_before': _summarize(no_pu_before) if len(no_pu_before) else None,
        'no_pu_n_after': _summarize(no_pu_after) if len(no_pu_after) else None,
        'no_pu_n_removed': _summarize(no_pu_removed) if len(no_pu_removed) else None,
        'no_pu_removal_fraction': _summarize(no_pu_removed / np.maximum(no_pu_before, 1)) if len(no_pu_before) else None,
        'no_pu_n_zero_after': vstats_no_pu['n_zero_after'],
    }


def _worker(args):
    """Wrapper for multiprocessing."""
    return analyze_shape(*args)


# ============================================================================
# Reporting
# ============================================================================

def _pct_row(label, stats, pcts_to_show=(0, 5, 25, 50, 75, 95, 100)):
    """Format a row of percentile stats."""
    vals = "  ".join(f"{stats['percentiles'][p]:7.1f}" for p in pcts_to_show)
    return f"    {label:<20s}  mean={stats['mean']:7.1f}  {vals}"


def print_report(results):
    pcts_header = [0, 5, 25, 50, 75, 95, 100]
    pct_labels = "  ".join(f"{'p'+str(p):>7s}" for p in pcts_header)

    print("\n" + "=" * 100)
    print("  OUTLIER FILTERING — VALID NEIGHBORS ANALYSIS")
    print("=" * 100)

    for res in results:
        if res is None:
            continue
        print(f"\n{'─' * 90}")
        print(f"Shape: {res['shape']}  ({res['num_gaussians']} gaussians, "
              f"ring={res['ring']}, {res['num_sources_used']} sources)")
        op = res['outlier_params']
        print(f"  Outlier params: median_mult={op['median_multiplier']}, "
              f"floor={op['threshold_floor']}, hard_cap={op['hard_cap']}, "
              f"fallback_mult={op['fallback_multiplier']}, fallback_floor={op['fallback_floor']}")
        print(f"  Points analyzed: {res['n_evaluated']}/{res['n_analysis_points']}")

        # --- Training-path (with p_u) ---
        print(f"\n  ── Training-path filter (with p_u) ──")
        print(f"    {'':20s}  {'mean':>7s}  {pct_labels}")
        print(f"    {'─' * 80}")
        print(_pct_row("Neighbors BEFORE", res['n_before'], pcts_header))
        print(_pct_row("Neighbors AFTER", res['n_after'], pcts_header))
        print(_pct_row("Removed", res['n_removed'], pcts_header))
        print(_pct_row("Removal fraction", res['removal_fraction'], pcts_header))
        print(_pct_row("Adaptive threshold", res['thresholds'], pcts_header))
        print(_pct_row("Median discrepancy", res['median_discs'], pcts_header))
        print()
        print(f"    Fallback used:         {res['n_fallback']}/{res['n_evaluated']}")
        print(f"    Zero neighbors after:  {res['n_zero_after']}/{res['n_evaluated']}")

        # --- Inference-path (without p_u) ---
        print(f"\n  ── Inference-path filter (without p_u) ──")
        if res['no_pu_n_before'] is not None:
            print(f"    Points analyzed: {res['no_pu_n_evaluated']}/{res['n_analysis_points']}")
            print(f"    {'':20s}  {'mean':>7s}  {pct_labels}")
            print(f"    {'─' * 80}")
            print(_pct_row("Neighbors BEFORE", res['no_pu_n_before'], pcts_header))
            print(_pct_row("Neighbors AFTER", res['no_pu_n_after'], pcts_header))
            print(_pct_row("Removed", res['no_pu_n_removed'], pcts_header))
            print(_pct_row("Removal fraction", res['no_pu_removal_fraction'], pcts_header))
            print()
            print(f"    Zero neighbors after:  {res['no_pu_n_zero_after']}/{res['no_pu_n_evaluated']}")
        else:
            print(f"    (no data)")

    # Summary table
    valid = [r for r in results if r is not None]
    if len(valid) > 1:
        print(f"\n{'=' * 120}")
        print(f"  SUMMARY — Training-path filter (with p_u)")
        print(f"{'=' * 120}")
        print(f"  {'Shape':<20s} {'#Gauss':>7s} {'Ring':>4s}"
              f"  {'Nbrs Before':>11s} {'Nbrs After':>11s} {'Removed':>8s}"
              f"  {'Rm%':>6s} {'Fallback':>8s} {'Zero':>6s}")
        print(f"  {'─' * 100}")
        for r in valid:
            print(f"  {r['shape']:<20s} {r['num_gaussians']:7d} {r['ring']:4d}"
                  f"  {r['n_before']['mean']:11.1f} {r['n_after']['mean']:11.1f}"
                  f"  {r['n_removed']['mean']:8.1f}"
                  f"  {r['removal_fraction']['mean']*100:5.1f}%"
                  f"  {r['n_fallback']:8d} {r['n_zero_after']:6d}")

        print(f"\n{'=' * 120}")
        print(f"  SUMMARY — Inference-path filter (without p_u)")
        print(f"{'=' * 120}")
        print(f"  {'Shape':<20s} {'#Gauss':>7s} {'Ring':>4s}"
              f"  {'Nbrs Before':>11s} {'Nbrs After':>11s} {'Removed':>8s}"
              f"  {'Rm%':>6s} {'Zero':>6s}")
        print(f"  {'─' * 90}")
        for r in valid:
            if r['no_pu_n_before'] is not None:
                print(f"  {r['shape']:<20s} {r['num_gaussians']:7d} {r['ring']:4d}"
                      f"  {r['no_pu_n_before']['mean']:11.1f} {r['no_pu_n_after']['mean']:11.1f}"
                      f"  {r['no_pu_n_removed']['mean']:8.1f}"
                      f"  {r['no_pu_removal_fraction']['mean']*100:5.1f}%"
                      f"  {r['no_pu_n_zero_after']:6d}")
            else:
                print(f"  {r['shape']:<20s} {r['num_gaussians']:7d} {r['ring']:4d}"
                      f"  {'(no data)':>11s}")

    print("\n" + "=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze valid ring neighbors after outlier filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset config YAML. Outlier params, ring, "
                             "n_neighbors, and gaussian sources are read from it. "
                             "CLI args override config values.")
    # Dataset discovery (used when --config is not provided)
    parser.add_argument("--dataset", choices=["tosca", "polynomial", "both"],
                        default="tosca",
                        help="Dataset to analyze (ignored when --config is given)")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Comma-separated shape names (TOSCA)")
    parser.add_argument("--surfaces", type=str, default=None,
                        help="Comma-separated surface names (Polynomial)")
    # Ring / neighborhood
    parser.add_argument("--ring", type=int, default=None,
                        help="Ring level to analyze (default: 3, or from config)")
    parser.add_argument("--n_neighbors", type=int, default=None,
                        help="Ring-1 kNN k (default: 6, or from config)")
    parser.add_argument("--use_mahalanobis", action="store_true")
    # Adaptive kNN
    parser.add_argument("--adaptive_target_ring", type=int, default=None)
    parser.add_argument("--adaptive_target_neighbors", type=int, default=None)
    parser.add_argument("--adaptive_k_boost", type=int, default=None)
    parser.add_argument("--adaptive_max_mean_cut", type=float, default=None)
    parser.add_argument("--adaptive_max_steps", type=int, default=None)
    # Analysis params
    parser.add_argument("--num_sources", type=int, default=2,
                        help="Number of geodesic sources to use (multi-source min)")
    parser.add_argument("--num_sample_points", type=int, default=0,
                        help="Max points to sample per shape (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = sequential)")
    # Outlier filtering params (CLI overrides config)
    parser.add_argument("--outlier_median_multiplier", type=float, default=None)
    parser.add_argument("--outlier_threshold_floor", type=float, default=None)
    parser.add_argument("--outlier_hard_cap", type=float, default=None)
    parser.add_argument("--outlier_fallback_multiplier", type=float, default=None)
    parser.add_argument("--outlier_fallback_floor", type=float, default=None)
    parser.add_argument("--outlier_max_removal_fraction", type=float, default=None,
                        help="Max fraction of visited neighbors the without-pu "
                             "filter may remove (e.g. 0.5 for 50%%). None=no cap.")

    args = parser.parse_args()

    # ── Load config if provided ──
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

    # Resolve parameters: CLI > config > defaults
    def _resolve(cli_val, cfg_key, default):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, default)

    ring = _resolve(args.ring, 'rings', 3)
    # If config has a list of rings, take the max
    if isinstance(ring, list):
        ring = max(ring)
    n_neighbors = _resolve(args.n_neighbors, 'n_neighbors', 6)
    use_mahalanobis = args.use_mahalanobis or cfg.get('use_mahalanobis', False)
    num_sources = _resolve(args.num_sources, 'num_sources', 2)
    seed = _resolve(args.seed, 'seed', 42)

    # Adaptive kNN
    adaptive_target_ring = _resolve(args.adaptive_target_ring, 'adaptive_target_ring', None)
    adaptive_target_neighbors = _resolve(args.adaptive_target_neighbors, 'adaptive_target_neighbors', None)
    adaptive_k_boost = _resolve(args.adaptive_k_boost, 'adaptive_k_boost', 20)
    adaptive_max_mean_cut = _resolve(args.adaptive_max_mean_cut, 'adaptive_max_mean_cut', 2.0)
    adaptive_max_steps = _resolve(args.adaptive_max_steps, 'adaptive_max_steps', 5)

    # Outlier params
    outlier_median_multiplier = _resolve(args.outlier_median_multiplier, 'outlier_median_multiplier', 3.0)
    outlier_threshold_floor = _resolve(args.outlier_threshold_floor, 'outlier_threshold_floor', 2.0)
    outlier_hard_cap = _resolve(args.outlier_hard_cap, 'outlier_hard_cap', 500.0)
    outlier_fallback_multiplier = _resolve(args.outlier_fallback_multiplier, 'outlier_fallback_multiplier', 5.0)
    outlier_fallback_floor = _resolve(args.outlier_fallback_floor, 'outlier_fallback_floor', 3.0)
    outlier_max_removal_fraction = _resolve(args.outlier_max_removal_fraction, 'outlier_max_removal_fraction', None)

    # ── Discover shapes ──
    tasks = []

    adaptive_kwargs = (
        adaptive_target_ring, adaptive_target_neighbors,
        adaptive_k_boost, adaptive_max_mean_cut, adaptive_max_steps,
        outlier_max_removal_fraction,
    )

    if args.config and 'gaussian_output' in cfg:
        # Use gaussian outputs from config
        gout_paths = resolve_gaussian_outputs(cfg['gaussian_output'])
        for gout in gout_paths:
            # Derive shape name from path
            shape_name = Path(gout).parent.name or Path(gout).name
            # Resolve to actual output dir (handle txt-based sources)
            gauss_dir = gout
            # If it's a relative path pointing to an output dir, check for /output suffix
            if not Path(gauss_dir).is_dir():
                logger.warning("Gaussian output dir not found: %s", gauss_dir)
                continue
            tasks.append((
                shape_name, gauss_dir, ring, n_neighbors, use_mahalanobis,
                num_sources, args.num_sample_points, seed,
                outlier_median_multiplier, outlier_threshold_floor,
                outlier_hard_cap, outlier_fallback_multiplier,
                outlier_fallback_floor,
                *adaptive_kwargs,
            ))
    else:
        # Discovery mode
        dataset = args.dataset or "tosca"
        if dataset in ("tosca", "both"):
            shapes = discover_tosca_shapes()
            if args.shapes:
                filter_set = set(args.shapes.split(","))
                shapes = [s for s in shapes if s in filter_set]
            for shape in shapes:
                gauss_out = f"{TOSCA_BASE}/{shape}/high_res/decoupled_appearance/output"
                tasks.append((
                    shape, gauss_out, ring, n_neighbors, use_mahalanobis,
                    num_sources, args.num_sample_points, seed,
                    outlier_median_multiplier, outlier_threshold_floor,
                    outlier_hard_cap, outlier_fallback_multiplier,
                    outlier_fallback_floor,
                    *adaptive_kwargs,
                ))

        if dataset in ("polynomial", "both"):
            surfaces_filter = args.surfaces.split(",") if args.surfaces else None
            shapes = discover_polynomial_shapes(surfaces_filter)
            for shape in shapes:
                gauss_out = f"{POLY_BASE}/{shape}/output"
                tasks.append((
                    shape, gauss_out, ring, n_neighbors, use_mahalanobis,
                    num_sources, args.num_sample_points, seed,
                    outlier_median_multiplier, outlier_threshold_floor,
                    outlier_hard_cap, outlier_fallback_multiplier,
                    outlier_fallback_floor,
                    *adaptive_kwargs,
                ))

    if not tasks:
        logger.error("No shapes found. Check --dataset/--shapes/--config arguments.")
        return

    n_workers = args.workers
    n_workers = min(n_workers, len(tasks))

    logger.info("Analyzing %d shapes (ring=%d, k=%d, outlier_mult=%.1f, "
                "floor=%.1f, hard_cap=%.1f)",
                len(tasks), ring, n_neighbors,
                outlier_median_multiplier, outlier_threshold_floor,
                outlier_hard_cap)

    if n_workers > 0:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]

    print_report([r for r in results if r is not None])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
