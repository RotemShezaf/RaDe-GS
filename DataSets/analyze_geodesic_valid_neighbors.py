#!/usr/bin/env python3
"""
Analyze valid-neighbor statistics using geodesic distances for TOSCA shapes.

For each shape and each source, computes how many ring-k neighbors of each
point have geodesic distance ≤ the center point's distance (i.e. are "valid"
during Fast Marching propagation).  Reports percentile statistics so you can
judge whether the model will have enough valid context at inference time.

Multi-source mode picks several random sources (like training) and reports
statistics on the per-point minimum distance across sources.

Usage:
    python DataSets/analyze_geodesic_valid_neighbors.py --dataset tosca
    python DataSets/analyze_geodesic_valid_neighbors.py --dataset tosca --shapes cat2,gorilla8
    python DataSets/analyze_geodesic_valid_neighbors.py --dataset polynomial --surfaces Paraboloid
    python DataSets/analyze_geodesic_valid_neighbors.py --num_sources 3 --num_trials 10
"""
import argparse
import logging
import multiprocessing
import os
import sys
from collections import defaultdict
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
    get_neighborhood_by_ring,
    ring1_neighbors_gaussians,
    adaptive_ring1_neighbors,
)
from GenerateData.utils.load_utils import load_gaussian_data_cpu

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
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        geo = d / "high_res" / "decoupled_appearance" / "output" / "geodesic_distance" / "gt_geodesic.npz"
        if geo.exists():
            shapes.append(d.name)
    return shapes


def discover_polynomial_shapes(surfaces=None):
    """List polynomial shapes (surface/level/light) that have geodesic data."""
    base = Path(POLY_BASE)
    shapes = []
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
# Core Analysis
# ============================================================================

def analyze_shape(
    shape_name: str,
    gaussian_output: str,
    ring: int,
    n_neighbors: int,
    use_mahalanobis: bool,
    num_sources: int,
    num_trials: int,
    seed: int,
    adaptive_target_ring: int = None,
    adaptive_target_neighbors: int = None,
    adaptive_k_boost: int = 20,
    adaptive_max_mean_cut: float = 2.0,
    adaptive_max_steps: int = 5,
):
    """
    Analyze valid-neighbor statistics for a single shape.

    Returns a dict with:
        shape, num_gaussians,
        per_source_stats: list of dicts with source_idx, valid_counts, percentiles
        multi_source_stats: dict with aggregated stats across random source combos
    """
    # Load gaussian data
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

    # Build ring neighborhoods
    use_adaptive = (adaptive_target_ring is not None and
                    adaptive_target_neighbors is not None)
    if use_adaptive:
        logger.info("%s: %d gaussians, %d sources, ring=%d, adaptive ring-%d≥%d "
                    "k_boost=%d", shape_name, N,
                    num_available_sources, ring, adaptive_target_ring,
                    adaptive_target_neighbors, adaptive_k_boost)
        ring1_nbrs, _, _ = adaptive_ring1_neighbors(
            positions,
            target_ring=adaptive_target_ring,
            target_ring_neighbors=adaptive_target_neighbors,
            n_neighbors_base=n_neighbors,
            k_boost=adaptive_k_boost,
            adaptive_max_mean_cut=adaptive_max_mean_cut,
            adaptive_max_steps=adaptive_max_steps,
            use_mahalanobis=use_mahalanobis,
        )
    else:
        logger.info("%s: %d gaussians, %d sources, ring=%d, k=%d",
                    shape_name, N, num_available_sources, ring, n_neighbors)
        ring1_nbrs, _, _ = ring1_neighbors_gaussians(
            positions,
            n_neighbors=n_neighbors,
            use_mahalanobis=use_mahalanobis,
            gaussian_scales=scales,
            gaussian_rotations=rotations,
        )
    ring_nbrs_dict = {idx: get_neighborhood_by_ring(idx, ring, ring1_nbrs)
                       for idx in range(N)}

    # Build inverse ring: for each point p, which points q have p in their ring
    inverse_ring_dict = defaultdict(list)
    for q in range(N):
        for nbr in ring_nbrs_dict.get(q, []):
            inverse_ring_dict[int(nbr)].append(q)
    # Convert to numpy arrays for fast indexing
    for k in inverse_ring_dict:
        inverse_ring_dict[k] = np.array(inverse_ring_dict[k], dtype=int)

    # --- Per-source analysis: use ALL sources ---
    rng = np.random.RandomState(seed)

    # Exclude all source gaussians from analysis points
    all_source_set = set(source_gaussian_indices.tolist())
    analysis_pts = [pt for pt in range(N) if pt not in all_source_set
                    and len(ring_nbrs_dict.get(pt, np.array([], dtype=int))) > 0]
    n_analysis = len(analysis_pts)

    # Track per-point valid counts across all sources: (num_sources, n_analysis)
    per_point_valid_all = np.zeros((num_available_sources, n_analysis), dtype=int)

    # Track per-point inverse-ring reachability: (num_sources, n_analysis)
    per_point_inv_reachable = np.zeros((num_available_sources, n_analysis), dtype=bool)

    per_source_stats = []
    for src_idx in range(num_available_sources):
        geo_dist = all_geodesic_distances[src_idx]

        for i, pt in enumerate(analysis_pts):
            nbrs = ring_nbrs_dict[pt]
            n_valid = int(np.sum(geo_dist[nbrs] <= geo_dist[pt]))
            per_point_valid_all[src_idx, i] = n_valid

            # Inverse ring check: is there any q that has pt in its ring
            # AND has smaller distance? If so, q will push an update to pt.
            inv_nbrs = inverse_ring_dict.get(pt, np.array([], dtype=int))
            if len(inv_nbrs) > 0:
                per_point_inv_reachable[src_idx, i] = bool(
                    np.any(geo_dist[inv_nbrs] < geo_dist[pt])
                )

        valid_counts = per_point_valid_all[src_idx]
        inv_reach = per_point_inv_reachable[src_idx]
        pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        per_source_stats.append({
            'source_idx': int(src_idx),
            'valid_percentiles': {p: float(np.percentile(valid_counts, p)) for p in pcts},
            'mean_valid': float(valid_counts.mean()),
            'zero_valid_count': int(np.sum(valid_counts == 0)),
            'not_inv_reachable': int(np.sum(~inv_reach)),
            'n_points': n_analysis,
        })

    # Points that have 0 valid neighbors for at least one source
    min_valid_across_sources = per_point_valid_all.min(axis=0)  # (n_analysis,)
    zero_valid_any_source = int(np.sum(min_valid_across_sources == 0))

    # Inverse-ring: points not reachable for ALL sources (worst case)
    not_inv_reachable_all = int(np.sum(~per_point_inv_reachable.any(axis=0)))
    # Inverse-ring: points not reachable for at least one source
    not_inv_reachable_any = int(np.sum(~per_point_inv_reachable.all(axis=0)))

    # Max valid neighbors across all sources (best single-source per point)
    max_valid_across_sources = per_point_valid_all.max(axis=0)  # (n_analysis,)
    pcts_all = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    max_across_sources_stats = {
        'valid_percentiles': {p: float(np.percentile(max_valid_across_sources, p)) for p in pcts_all},
        'mean_valid': float(max_valid_across_sources.mean()),
        'zero_valid_count': int(np.sum(max_valid_across_sources == 0)),
        'n_points': n_analysis,
    }

    # --- Multi-source analysis: max valid neighbors across trials ---
    actual_trials = min(num_trials, 1) if num_sources >= num_available_sources else num_trials
    all_trial_valid = []
    all_src_gaussians = set()
    trial_selections = []
    for trial in range(actual_trials):
        if num_sources >= num_available_sources:
            sel = np.arange(num_available_sources)
        else:
            sel = rng.choice(num_available_sources, num_sources, replace=False)
        trial_selections.append(sel)
        all_src_gaussians.update(source_gaussian_indices[sel].tolist())

    available = np.array(sorted(set(range(N)) - all_src_gaussians))
    pt_nbrs = {pt: ring_nbrs_dict.get(pt, np.array([], dtype=int)) for pt in available}
    has_nbrs = np.array([len(pt_nbrs[pt]) > 0 for pt in available])
    available = available[has_nbrs]

    pt_inv_nbrs = {pt: inverse_ring_dict.get(pt, np.array([], dtype=int)) for pt in available}

    all_trial_inv_reachable = []
    for sel in trial_selections:
        min_distances = all_geodesic_distances[sel].min(axis=0)
        trial_valid = np.zeros(len(available), dtype=int)
        trial_inv_reach = np.zeros(len(available), dtype=bool)
        for i, pt in enumerate(available):
            nbrs = pt_nbrs[pt]
            trial_valid[i] = int(np.sum(min_distances[nbrs] <= min_distances[pt]))
            inv_nbrs = pt_inv_nbrs[pt]
            if len(inv_nbrs) > 0:
                trial_inv_reach[i] = bool(np.any(min_distances[inv_nbrs] < min_distances[pt]))
        all_trial_valid.append(trial_valid)
        all_trial_inv_reachable.append(trial_inv_reach)

    if all_trial_valid:
        stacked = np.stack(all_trial_valid, axis=0)
        max_valid = stacked.max(axis=0)
        min_valid = stacked.min(axis=0)
        inv_stacked = np.stack(all_trial_inv_reachable, axis=0)
        pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        multi_source_stats = {
            'num_sources': num_sources,
            'num_trials': actual_trials,
            'valid_percentiles': {p: float(np.percentile(max_valid, p)) for p in pcts},
            'mean_valid': float(max_valid.mean()),
            'zero_valid_count': int(np.sum(max_valid == 0)),
            'zero_valid_any_trial': int(np.sum(min_valid == 0)),
            'not_inv_reachable_all_trials': int(np.sum(~inv_stacked.any(axis=0))),
            'not_inv_reachable_any_trial': int(np.sum(~inv_stacked.all(axis=0))),
            'n_points': len(available),
        }
    else:
        multi_source_stats = None

    return {
        'shape': shape_name,
        'num_gaussians': N,
        'num_sources_available': num_available_sources,
        'per_source': per_source_stats,
        'multi_source': multi_source_stats,
        'zero_valid_any_source': zero_valid_any_source,
        'not_inv_reachable_all': not_inv_reachable_all,
        'not_inv_reachable_any': not_inv_reachable_any,
        'max_across_sources': max_across_sources_stats,
        'n_analysis_points': n_analysis,
    }


def _worker(args):
    """Wrapper for multiprocessing."""
    return analyze_shape(*args)


# ============================================================================
# Reporting
# ============================================================================

def print_report(results, num_sources):
    print("\n" + "=" * 80)
    print("  GEODESIC VALID-NEIGHBOR ANALYSIS REPORT")
    print("=" * 80)

    for res in results:
        if res is None:
            continue
        print(f"\n{'─' * 70}")
        print(f"Shape: {res['shape']}  ({res['num_gaussians']} gaussians, "
              f"{res['num_sources_available']} GT sources)")

        # --- Per-source summary (all sources) ---
        print(f"\n  Per-source valid-neighbor percentiles (all {len(res['per_source'])} sources):")
        print(f"  {'Src':>4s}  {'Mean':>6s}  {'p0':>4s}  {'p1':>4s}  {'p5':>4s}  "
              f"{'p10':>4s}  {'p25':>4s}  {'p50':>4s}  {'p75':>4s}  {'p90':>4s}  "
              f"{'p95':>4s}  {'p99':>4s} {'p100':>4s}  {'0-valid':>8s}  {'!inv':>8s}")
        for s in res['per_source']:
            vp = s['valid_percentiles']
            print(f"  {s['source_idx']:4d}  {s['mean_valid']:6.1f}  "
                  f"{vp[0]:4.0f}  {vp[1]:4.0f}  {vp[5]:4.0f}  {vp[10]:4.0f}  "
                  f"{vp[25]:4.0f}  {vp[50]:4.0f}  {vp[75]:4.0f}  {vp[90]:4.0f}  "
                  f"{vp[95]:4.0f}  {vp[99]:4.0f}  {vp[100]:4.0f}  "
                  f"{s['zero_valid_count']:5d}/{s['n_points']}  "
                  f"{s['not_inv_reachable']:5d}/{s['n_points']}")

        # --- Points with 0 valid at any source ---
        print(f"\n  Points with 0 valid neighbors for at least one source: "
              f"{res['zero_valid_any_source']}/{res['n_analysis_points']}")

        # --- Inverse-ring reachability ---
        print(f"  Not inverse-reachable for ALL sources (unreachable by FM): "
              f"{res['not_inv_reachable_all']}/{res['n_analysis_points']}")
        print(f"  Not inverse-reachable for at least one source: "
              f"{res['not_inv_reachable_any']}/{res['n_analysis_points']}")

        # --- Max valid across all sources (best single-source per point) ---
        mx = res['max_across_sources']
        print(f"\n  Max valid across all {res['num_sources_available']} sources (best single-source per point):")
        vp = mx['valid_percentiles']
        print(f"    Mean={mx['mean_valid']:.1f}  "
              f"p0={vp[0]:.0f}  p1={vp[1]:.0f}  p5={vp[5]:.0f}  p10={vp[10]:.0f}  "
              f"p25={vp[25]:.0f}  p50={vp[50]:.0f}  p75={vp[75]:.0f}  p90={vp[90]:.0f}  "
              f"p95={vp[95]:.0f}  p99={vp[99]:.0f}  p100={vp[100]:.0f}  "
              f"0-valid={mx['zero_valid_count']}/{mx['n_points']}")

        # --- Multi-source summary (max across trials) ---
        m = res['multi_source']
        if m:
            print(f"\n  Multi-source analysis ({m['num_sources']} src/trial, "
                  f"{m['num_trials']} trials, max valid across trials):")
            vp = m['valid_percentiles']
            print(f"  {'Mean':>6s}  {'p0':>4s}  {'p1':>4s}  {'p5':>4s}  "
                  f"{'p10':>4s}  {'p25':>4s}  {'p50':>4s}  {'p75':>4s}  "
                  f"{'p90':>4s}  {'p95':>4s}  {'p99':>4s} {'p100':>4s}  {'0-valid':>8s}")
            print(f"  {m['mean_valid']:6.1f}  "
                  f"{vp[0]:4.0f}  {vp[1]:4.0f}  {vp[5]:4.0f}  "
                  f"{vp[10]:4.0f}  {vp[25]:4.0f}  {vp[50]:4.0f}  "
                  f"{vp[75]:4.0f}  {vp[90]:4.0f}  {vp[95]:4.0f}  "
                  f"{vp[99]:4.0f}  {vp[100]:4.0f}  "
                  f"{m['zero_valid_count']:5d}/{m['n_points']}")
            print(f"  Points with 0 valid in at least one trial: "
                  f"{m['zero_valid_any_trial']}/{m['n_points']}")
            print(f"  Not inverse-reachable in ALL trials: "
                  f"{m['not_inv_reachable_all_trials']}/{m['n_points']}")
            print(f"  Not inverse-reachable in at least one trial: "
                  f"{m['not_inv_reachable_any_trial']}/{m['n_points']}")

    # ── Summary tables across all shapes ──
    print_summary_tables(results)

    print("\n" + "=" * 80)


def print_summary_tables(results):
    """Print compact summary tables across all shapes."""
    valid = [r for r in results if r is not None]
    if not valid:
        return

    # --- Table 1: Per-source aggregate (mean/worst across sources) ---
    print(f"\n{'='*100}")
    print(f"  SUMMARY — Per-source aggregate (across all sources)")
    print(f"{'='*100}")
    print(f"  {'Shape':<16s} {'#Gauss':>7s} {'#Src':>5s} {'#Pts':>7s}"
          f"  {'MeanValid':>9s} {'p0':>5s} {'p5':>5s} {'p25':>5s} {'p50':>5s}"
          f"  {'0-valid%':>8s} {'!inv%':>8s}")
    print(f"  {'─'*95}")
    for r in valid:
        n_src = len(r['per_source'])
        n_pts = r['n_analysis_points']
        # Average stats across all sources
        mean_vals = [s['mean_valid'] for s in r['per_source']]
        zero_vals = [s['zero_valid_count'] for s in r['per_source']]
        inv_vals = [s['not_inv_reachable'] for s in r['per_source']]
        p0_vals = [s['valid_percentiles'][0] for s in r['per_source']]
        p5_vals = [s['valid_percentiles'][5] for s in r['per_source']]
        p25_vals = [s['valid_percentiles'][25] for s in r['per_source']]
        p50_vals = [s['valid_percentiles'][50] for s in r['per_source']]
        avg_mean = np.mean(mean_vals)
        avg_p0 = np.mean(p0_vals)
        avg_p5 = np.mean(p5_vals)
        avg_p25 = np.mean(p25_vals)
        avg_p50 = np.mean(p50_vals)
        avg_zero_pct = 100.0 * np.mean(zero_vals) / n_pts if n_pts else 0
        avg_inv_pct = 100.0 * np.mean(inv_vals) / n_pts if n_pts else 0
        print(f"  {r['shape']:<16s} {r['num_gaussians']:7d} {n_src:5d} {n_pts:7d}"
              f"  {avg_mean:9.1f} {avg_p0:5.1f} {avg_p5:5.1f} {avg_p25:5.1f} {avg_p50:5.1f}"
              f"  {avg_zero_pct:7.2f}% {avg_inv_pct:7.2f}%")

    # --- Table 2: Best-across-sources (max valid per point) ---
    print(f"\n{'='*100}")
    print(f"  SUMMARY — Max valid across ALL sources (best single-source per point)")
    print(f"{'='*100}")
    print(f"  {'Shape':<16s} {'#Pts':>7s}"
          f"  {'Mean':>6s} {'p0':>5s} {'p5':>5s} {'p25':>5s} {'p50':>5s} {'p95':>5s} {'p100':>5s}"
          f"  {'0-val':>7s}  {'!invAll':>7s} {'!invAny':>7s}")
    print(f"  {'─'*95}")
    for r in valid:
        mx = r['max_across_sources']
        vp = mx['valid_percentiles']
        n_pts = r['n_analysis_points']
        print(f"  {r['shape']:<16s} {n_pts:7d}"
              f"  {mx['mean_valid']:6.1f} {vp[0]:5.0f} {vp[5]:5.0f} {vp[25]:5.0f}"
              f"  {vp[50]:5.0f} {vp[95]:5.0f} {vp[100]:5.0f}"
              f"  {mx['zero_valid_count']:5d}"
              f"  {r['not_inv_reachable_all']:7d} {r['not_inv_reachable_any']:7d}")

    # --- Table 3: Multi-source trials ---
    has_multi = [r for r in valid if r['multi_source'] is not None]
    if has_multi:
        print(f"\n{'='*100}")
        print(f"  SUMMARY — Multi-source trials (max valid across trials)")
        print(f"{'='*100}")
        m0 = has_multi[0]['multi_source']
        print(f"  ({m0['num_sources']} src/trial, {m0['num_trials']} trials)")
        print(f"  {'Shape':<16s} {'#Pts':>7s}"
              f"  {'Mean':>6s} {'p0':>5s} {'p5':>5s} {'p25':>5s} {'p50':>5s} {'p95':>5s} {'p100':>5s}"
              f"  {'0-val':>7s} {'0-any':>7s}  {'!invAll':>7s} {'!invAny':>7s}")
        print(f"  {'─'*105}")
        for r in has_multi:
            m = r['multi_source']
            vp = m['valid_percentiles']
            print(f"  {r['shape']:<16s} {m['n_points']:7d}"
                  f"  {m['mean_valid']:6.1f} {vp[0]:5.0f} {vp[5]:5.0f} {vp[25]:5.0f}"
                  f"  {vp[50]:5.0f} {vp[95]:5.0f} {vp[100]:5.0f}"
                  f"  {m['zero_valid_count']:5d} {m['zero_valid_any_trial']:7d}"
                  f"  {m['not_inv_reachable_all_trials']:7d} {m['not_inv_reachable_any_trial']:7d}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze valid-neighbor counts using GT geodesic distances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="full", choices=["full", "adaptive"],
                        help="'full' = per-source valid-neighbor analysis, "
                             "'adaptive' = use adaptive kNN ring building")
    parser.add_argument("--dataset", choices=["tosca", "polynomial", "both"],
                        default="tosca")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Comma-separated shape names (TOSCA)")
    parser.add_argument("--surfaces", type=str, default=None,
                        help="Comma-separated surface names (Polynomial)")
    parser.add_argument("--ring", type=int, default=3)
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="Ring-1 kNN k")
    parser.add_argument("--use_mahalanobis", action="store_true")
    parser.add_argument("--num_sources", type=int, default=1,
                        help="Number of sources for multi-source analysis")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of random source selections for multi-source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=-1,
                        help="Parallel workers (-1 = auto, 0 = sequential)")
    parser.add_argument("--adaptive_target_ring", type=int, default=None,
                        help="Ring level to optimize with adaptive kNN (e.g. 3)")
    parser.add_argument("--adaptive_target_neighbors", type=int, default=None,
                        help="Desired ring-k count for adaptive kNN (e.g. 128)")
    parser.add_argument("--adaptive_k_boost", type=int, default=20,
                        help="Max boosted k for adaptive kNN")
    parser.add_argument("--adaptive_max_mean_cut", type=float, default=2.0,
                        help="Stop binary search when mean cut <= this")
    parser.add_argument("--adaptive_max_steps", type=int, default=5,
                        help="Max binary search iterations")
    args = parser.parse_args()

    # In adaptive mode, inherit adaptive_target_ring as ring if not set separately
    if args.mode == "adaptive":
        if args.adaptive_target_ring is None or args.adaptive_target_neighbors is None:
            parser.error("--adaptive_target_ring and --adaptive_target_neighbors "
                         "are required for --mode adaptive")
        if args.ring == 3 and args.adaptive_target_ring is not None:
            args.ring = args.adaptive_target_ring

    # Discover shapes
    tasks = []
    # Adaptive args to forward
    adaptive_kwargs = (
        args.adaptive_target_ring,
        args.adaptive_target_neighbors,
        args.adaptive_k_boost,
        args.adaptive_max_mean_cut,
        args.adaptive_max_steps,
    ) if args.mode == "adaptive" else (None, None, 20, 2.0, 5)

    if args.dataset in ("tosca", "both"):
        shapes = discover_tosca_shapes()
        if args.shapes:
            filter_set = set(args.shapes.split(","))
            shapes = [s for s in shapes if s in filter_set]
        for shape in shapes:
            gauss_out = f"{TOSCA_BASE}/{shape}/high_res/decoupled_appearance/output"
            tasks.append((
                shape, gauss_out, args.ring, args.n_neighbors,
                args.use_mahalanobis, args.num_sources, args.num_trials, args.seed,
                *adaptive_kwargs,
            ))

    if args.dataset in ("polynomial", "both"):
        surfaces_filter = args.surfaces.split(",") if args.surfaces else None
        shapes = discover_polynomial_shapes(surfaces_filter)
        for shape in shapes:
            gauss_out = f"{POLY_BASE}/{shape}/output"
            tasks.append((
                shape, gauss_out, args.ring, args.n_neighbors,
                args.use_mahalanobis, args.num_sources, args.num_trials, args.seed,
                *adaptive_kwargs,
            ))

    if not tasks:
        logger.error("No shapes found – check --dataset and --shapes/--surfaces arguments.")
        return

    n_workers = args.workers
    if n_workers < 0:
        n_workers = min(multiprocessing.cpu_count() - 1, 70)
    n_workers = min(n_workers, len(tasks))

    logger.info("Analyzing %d shapes (ring=%d, k=%d, num_sources=%d, "
                "mahalanobis=%s, workers=%d)",
                len(tasks), args.ring, args.n_neighbors, args.num_sources,
                args.use_mahalanobis, n_workers)

    if n_workers > 0:
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]

    print_report([r for r in results if r is not None], args.num_sources)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
