#!/usr/bin/env python3
"""
Sweep kNN k values to find the best k for polynomial surfaces that:
  1. Eliminates "no-closer-neighbor" fallback points (ring-3 local minima)
  2. Keeps ring-3 neighbor counts within the 128 cap

Usage:
    python geodesic_propagation/sweep_knn_for_polynomial.py
    python geodesic_propagation/sweep_knn_for_polynomial.py --k_values 8,10,12,14,16
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as LA
import yaml

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

from GenerateData.utils.data_generation_utils import (
    ring1_neighbors_gaussians,
    get_neighborhood_by_ring,
)
from GenerateData.utils.load_utils import load_gaussian_data_cpu


POLY_BASE = Path(PROJECT_ROOT) / "TrainData/Polynomial/SyntheticColmapData/blue_texture"
ALL_SURFACES = ["Paraboloid", "Saddle", "HyperbolicParaboloid"]
ALL_LEVELS = ["level_02", "level_03", "level_04"]
ALL_LIGHTS = ["light_0", "light_1", "light_2", "light_3", "light_4"]


def find_poly_outputs(surfaces, levels, lights):
    """Yield (name, output_path, gt_path) for each existing output."""
    for surface in surfaces:
        for level in levels:
            for light in lights:
                out = POLY_BASE / surface / level / light / "output"
                gt = out / "geodesic_distance" / "gt_geodesic.npz"
                if out.exists() and gt.exists():
                    yield f"{surface}/{level}/{light}", str(out), str(gt)


def analyze_k(output_path, gt_path, k, ring, seed, num_sources):
    """
    For a given kNN k, compute:
      - ring-{ring} neighbor count stats (p95, p99, max)
      - number of 'fallback points' (no ring-k neighbor with GT dist <= point's)
    Returns dict with results.
    """
    np.random.seed(seed)

    data = load_gaussian_data_cpu(Path(output_path), iteration=None, load_sh_features=False)
    positions = data.get_xyz()
    scales = data.get_scaling()
    rotations = data.get_rotation()
    N = len(positions)

    # Load GT geodesic
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]
    gt_source_indices = gt_data["source_gaussian_indices"]

    ns = min(num_sources, len(gt_source_indices))
    selected_rows = np.random.choice(len(gt_source_indices), ns, replace=False).tolist()
    selected_dists = gt_all_dists[selected_rows].astype(np.float64)
    gt_dists = selected_dists[0] if ns == 1 else selected_dists.min(axis=0)
    source_gaussian_idxs = gt_source_indices[selected_rows]

    # Build ring-1 neighbors
    ring1_nbrs, mean_nn_dist, _ = ring1_neighbors_gaussians(
        vertices=positions, n_neighbors=k,
        use_mahalanobis=False,
    )

    # Build ring-k neighborhoods
    ring_neighbors = {}
    for i in range(N):
        ring_neighbors[i] = get_neighborhood_by_ring(i, ring, ring1_nbrs)

    # Eval mask
    eval_mask = np.ones(N, dtype=bool)
    for sg_idx in source_gaussian_idxs:
        eval_mask[int(sg_idx)] = False
    eval_mask[~np.isfinite(gt_dists)] = False
    eval_mask[gt_dists <= 0] = False
    eval_indices = np.where(eval_mask)[0]

    # Count ring neighbors and fallback points
    ring_counts = []
    n_fallback = 0
    n_empty = 0
    for pid in eval_indices:
        nbrs = ring_neighbors.get(pid, np.array([], dtype=np.int64))
        ring_counts.append(len(nbrs))
        if len(nbrs) == 0:
            n_empty += 1
            continue
        has_closer = np.any(gt_dists[nbrs] <= gt_dists[pid])
        if not has_closer:
            n_fallback += 1

    ring_counts = np.array(ring_counts)

    return {
        "k": k,
        "N": N,
        "n_eval": len(eval_indices),
        "n_fallback": n_fallback,
        "n_empty": n_empty,
        "ring_count_mean": ring_counts.mean(),
        "ring_count_p95": np.percentile(ring_counts, 95),
        "ring_count_p99": np.percentile(ring_counts, 99),
        "ring_count_max": ring_counts.max(),
        "pct_over_128": 100 * np.sum(ring_counts > 128) / len(ring_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep kNN k to find best value for polynomial evaluation")
    parser.add_argument("--k_values", type=str, default="8,10,11,12,13,14,15,16,18,20",
                        help="Comma-separated k values to test")
    parser.add_argument("--ring", type=int, default=3)
    parser.add_argument("--surfaces", type=str, default="",
                        help="Comma-separated surfaces (default: all)")
    parser.add_argument("--levels", type=str, default="level_03",
                        help="Comma-separated levels (default: level_03)")
    parser.add_argument("--lights", type=str, default="light_0",
                        help="Comma-separated lights (default: light_0)")
    parser.add_argument("--num_sources", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=128,
                        help="Ring-3 neighbor cap to stay within")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()] or ALL_SURFACES
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    lights = [l.strip() for l in args.lights.split(",") if l.strip()]

    outputs = list(find_poly_outputs(surfaces, levels, lights))
    if not outputs:
        print("No polynomial outputs found!")
        return

    print(f"Testing k values: {k_values}")
    print(f"Ring: {args.ring}, Cap: {args.cap}")
    print(f"Scenes: {len(outputs)}")
    for name, _, _ in outputs:
        print(f"  - {name}")

    # header
    print(f"\n{'k':>4s}  {'scene':>40s}  {'N':>6s}  {'fallback':>8s}  "
          f"{'empty':>5s}  {'mean':>6s}  {'p95':>6s}  {'p99':>6s}  "
          f"{'max':>6s}  {'%>cap':>6s}")
    print("-" * 110)

    # Aggregate results per k
    k_summary = {}

    for k in k_values:
        total_fallback = 0
        total_eval = 0
        max_p99 = 0
        max_max = 0
        max_pct_over = 0

        for name, out_path, gt_path in outputs:
            res = analyze_k(out_path, gt_path, k, args.ring, args.seed, args.num_sources)
            total_fallback += res["n_fallback"]
            total_eval += res["n_eval"]
            max_p99 = max(max_p99, res["ring_count_p99"])
            max_max = max(max_max, res["ring_count_max"])
            max_pct_over = max(max_pct_over, res["pct_over_128"])

            print(f"{k:>4d}  {name:>40s}  {res['N']:>6d}  {res['n_fallback']:>8d}  "
                  f"{res['n_empty']:>5d}  {res['ring_count_mean']:>6.1f}  "
                  f"{res['ring_count_p95']:>6.0f}  {res['ring_count_p99']:>6.0f}  "
                  f"{res['ring_count_max']:>6.0f}  {res['pct_over_128']:>5.1f}%")

        k_summary[k] = {
            "total_fallback": total_fallback,
            "total_eval": total_eval,
            "max_p99": max_p99,
            "max_max": max_max,
            "max_pct_over": max_pct_over,
        }

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: ring-{args.ring} neighborhood, cap={args.cap}")
    print(f"{'='*80}")
    print(f"{'k':>4s}  {'fallback':>10s}  {'max_p99':>8s}  {'max_max':>8s}  "
          f"{'%>cap':>7s}  {'fits_cap?':>10s}  {'verdict':>20s}")
    print("-" * 80)

    best_k = None
    for k in k_values:
        s = k_summary[k]
        fits = s["max_p99"] <= args.cap
        no_fallback = s["total_fallback"] == 0
        verdict = ""
        if no_fallback and fits:
            verdict = "GOOD"
            if best_k is None:
                best_k = k
                verdict = "<<< BEST"
        elif no_fallback and not fits:
            verdict = "no fallback, EXCEEDS cap"
        elif not no_fallback and fits:
            verdict = f"{s['total_fallback']} fallback pts"
        else:
            verdict = f"{s['total_fallback']} fallback, EXCEEDS cap"

        print(f"{k:>4d}  {s['total_fallback']:>10d}  {s['max_p99']:>8.0f}  "
              f"{s['max_max']:>8.0f}  {s['max_pct_over']:>6.1f}%  "
              f"{'YES' if fits else 'NO':>10s}  {verdict:>20s}")

    if best_k:
        print(f"\n>>> RECOMMENDED: k={best_k}")
        print(f"    Eliminates all fallback points while keeping ring-{args.ring} p99 <= {args.cap}")
    else:
        # Find smallest k with no fallback
        no_fb = [k for k in k_values if k_summary[k]["total_fallback"] == 0]
        if no_fb:
            k_nf = min(no_fb)
            print(f"\n>>> Smallest k with no fallback: k={k_nf}")
            print(f"    But max p99={k_summary[k_nf]['max_p99']:.0f} exceeds cap={args.cap}")
            print(f"    Consider increasing ring_size_mapping to {int(np.ceil(k_summary[k_nf]['max_p99']/16)*16)}")
        else:
            print(f"\n>>> No k value eliminates all fallback points in tested range.")
            print(f"    Try larger k values.")


if __name__ == "__main__":
    main()
