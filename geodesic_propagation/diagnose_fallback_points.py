#!/usr/bin/env python3
"""
Diagnose the 8 fallback points that have no ring-3 neighbor with GT dist <= their own.
Check whether using ALL ring-3 neighbors (uncapped) vs capped (128) matters,
and whether higher rings or more kNN neighbors would resolve the issue.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian_dir", type=str,
                        default="TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_04/light_4/output")
    parser.add_argument("--gt_subpath", type=str, default="geodesic_distance/gt_geodesic.npz")
    parser.add_argument("--dataset_config", type=str,
                        default="DataSets/configs/polynomial/combined_polynomial_all_one_source.yaml")
    parser.add_argument("--ring", type=int, default=3)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--source_row", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    gaussian_dir = Path(args.gaussian_dir)
    gt_path = gaussian_dir / args.gt_subpath
    dataset_config_path = Path(args.dataset_config)

    # Load data
    gdata = load_gaussian_data_cpu(gaussian_dir, iteration=None, load_sh_features=False)
    positions = gdata.get_xyz()
    scales = gdata.get_scaling()
    rotations = gdata.get_rotation()
    N = len(positions)
    print(f"Loaded {N} Gaussians")

    # Load GT
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]
    gt_source_indices = gt_data["source_gaussian_indices"]

    config = yaml.safe_load(open(dataset_config_path))
    num_sources = int(config.get("num_sources", 1))
    use_mahalanobis = config.get("use_mahalanobis", False)

    if args.source_row is not None:
        selected_rows = [args.source_row]
    else:
        selected_rows = np.random.choice(len(gt_source_indices), num_sources, replace=False).tolist()

    selected_dists = gt_all_dists[selected_rows].astype(np.float64)
    gt_dists = selected_dists[0] if len(selected_rows) == 1 else selected_dists.min(axis=0)
    source_gaussian_idxs = gt_source_indices[selected_rows]

    print(f"Source row(s): {selected_rows}, source Gaussian idx(s): {source_gaussian_idxs.tolist()}")
    print(f"GT dist range: [{gt_dists.min():.8f}, {gt_dists.max():.8f}]")

    # Build ring-1 neighbors (standard n_neighbors)
    ring1_nbrs, mean_nn_dist, per_point_nn_dist = ring1_neighbors_gaussians(
        vertices=positions,
        n_neighbors=args.n_neighbors,
        use_mahalanobis=use_mahalanobis,
        gaussian_scales=scales if use_mahalanobis else None,
        gaussian_rotations=rotations if use_mahalanobis else None,
    )

    # Build ring-k neighborhoods — these are UNCAPPED (all neighbors at ring distance k)
    ring_neighbors = {}
    for i in range(N):
        ring_neighbors[i] = get_neighborhood_by_ring(i, args.ring, ring1_nbrs)

    # Eval mask (same as evaluate script)
    eval_mask = np.ones(N, dtype=bool)
    for sg_idx in source_gaussian_idxs:
        eval_mask[int(sg_idx)] = False
    eval_mask[~np.isfinite(gt_dists)] = False
    eval_mask[gt_dists <= 0] = False
    eval_indices = np.where(eval_mask)[0]

    # Find fallback points
    fallback_pids = []
    for pid in eval_indices:
        nbrs = ring_neighbors.get(pid, np.array([], dtype=np.int64))
        if len(nbrs) == 0:
            continue
        visited_mask = gt_dists[nbrs] <= gt_dists[pid]
        if not np.any(visited_mask):
            fallback_pids.append(pid)

    print(f"\n{'='*80}")
    print(f"Found {len(fallback_pids)} fallback points (no ring-{args.ring} neighbor with GT dist <= point's)")
    print(f"{'='*80}")

    if not fallback_pids:
        print("No fallback points found — the issue may not reproduce with this seed/source_row.")
        return

    # Detailed analysis of each fallback point
    for pid in fallback_pids:
        nbrs = ring_neighbors[pid]
        r1_nbrs = ring1_nbrs[pid]
        gt_pid = gt_dists[pid]

        print(f"\n--- Point {pid} ---")
        print(f"  GT dist:              {gt_pid:.10f}")
        print(f"  Position:             {positions[pid]}")
        print(f"  #ring-{args.ring} neighbors:  {len(nbrs)}  (all, uncapped)")
        print(f"  #ring-1 neighbors:    {len(r1_nbrs)}")

        # Ring-k neighbor GT distances
        nbr_gt = gt_dists[nbrs]
        print(f"  Ring-{args.ring} nbr GT dist:  min={nbr_gt.min():.10f}  max={nbr_gt.max():.10f}")
        print(f"  Deficit (min_nbr - point): {nbr_gt.min() - gt_pid:.10f}")

        # Ring-1 neighbor GT distances
        r1_gt = gt_dists[r1_nbrs]
        print(f"  Ring-1 nbr GT dist:   min={r1_gt.min():.10f}  max={r1_gt.max():.10f}")
        print(f"  Ring-1 deficit:       {r1_gt.min() - gt_pid:.10f}")

        # How many neighbors are VERY close in GT dist?
        close_thresh = gt_pid * 1.01  # within 1%
        n_close = int(np.sum(nbr_gt < close_thresh))
        print(f"  #neighbors within 1%: {n_close}")

        # Euclidean distance to source
        src_idx = int(source_gaussian_idxs[0])
        euc_to_src = np.linalg.norm(positions[pid] - positions[src_idx])
        print(f"  Euc dist to source ({src_idx}): {euc_to_src:.10f}")

        # Is the source in ring-k neighborhood?
        src_in_ring = src_idx in nbrs
        print(f"  Source in ring-{args.ring}?    {src_in_ring}")

        # Check: would higher ring or more kNN help?
        # Build ring-4 for this point
        ring4_nbrs = get_neighborhood_by_ring(pid, 4, ring1_nbrs)
        r4_gt = gt_dists[ring4_nbrs]
        r4_has_closer = np.any(r4_gt <= gt_pid) if len(ring4_nbrs) > 0 else False
        print(f"  Ring-4 neighbors:     {len(ring4_nbrs)}")
        if len(ring4_nbrs) > 0:
            print(f"  Ring-4 nbr GT dist:   min={r4_gt.min():.10f}")
            print(f"  Ring-4 has closer?    {r4_has_closer}")

    # Summary: would any cap (128 vs uncapped) matter?
    print(f"\n{'='*80}")
    print("SUMMARY: Does the 128 cap matter?")
    print(f"{'='*80}")
    cap = config.get("ring_size_mapping", {}).get("euclidean", {}).get(args.ring, 200)
    print(f"  Config cap (max_num_nbrs): {cap}")
    for pid in fallback_pids:
        nbrs = ring_neighbors[pid]
        total = len(nbrs)
        capped = min(total, cap)
        nbr_gt = gt_dists[nbrs]
        n_closer = int(np.sum(nbr_gt <= gt_dists[pid]))
        print(f"  pid={pid}: {total} total ring-{args.ring} neighbors, "
              f"{n_closer} have GT dist <= point's GT dist. "
              f"{'CAP IRRELEVANT' if total <= cap else f'Would cap from {total} to {cap}'}")

    # Check: would increasing n_neighbors (kNN k) help?
    print(f"\n{'='*80}")
    print("EXPERIMENT: Rebuilding with n_neighbors=20 (double kNN)")
    print(f"{'='*80}")
    ring1_nbrs_20, _, _ = ring1_neighbors_gaussians(
        vertices=positions,
        n_neighbors=20,
        use_mahalanobis=use_mahalanobis,
        gaussian_scales=scales if use_mahalanobis else None,
        gaussian_rotations=rotations if use_mahalanobis else None,
    )
    for pid in fallback_pids:
        nbrs_20 = get_neighborhood_by_ring(pid, args.ring, ring1_nbrs_20)
        nbr_gt = gt_dists[nbrs_20]
        n_closer = int(np.sum(nbr_gt <= gt_dists[pid])) if len(nbrs_20) > 0 else 0
        print(f"  pid={pid}: {len(nbrs_20)} ring-{args.ring} neighbors (k=20), "
              f"{n_closer} have GT dist <= point's GT dist → "
              f"{'SOLVED' if n_closer > 0 else 'STILL NO CLOSER NEIGHBOR'}")


if __name__ == "__main__":
    main()
