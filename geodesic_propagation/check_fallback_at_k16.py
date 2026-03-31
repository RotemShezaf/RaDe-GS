#!/usr/bin/env python3
"""
For the fallback points identified at k=10, check their ring-3 neighbor
count when rebuilt with k=16. Do they stay under 128/230?
"""
import sys
from pathlib import Path
import numpy as np

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
SCENES = [
    ("Paraboloid/level_04/light_4", "Paraboloid", "level_04", "light_4"),
    ("Saddle/level_04/light_4", "Saddle", "level_04", "light_4"),
    ("HyperbolicParaboloid/level_04/light_4", "HyperbolicParaboloid", "level_04", "light_4"),
]

RING = 3
SEED = 42
NUM_SOURCES = 1

np.random.seed(SEED)

for name, surface, level, light in SCENES:
    out_path = str(POLY_BASE / surface / level / light / "output")
    gt_path = str(POLY_BASE / surface / level / light / "output" / "geodesic_distance" / "gt_geodesic.npz")

    data = load_gaussian_data_cpu(Path(out_path), iteration=None, load_sh_features=False)
    positions = data.get_xyz()
    N = len(positions)

    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]
    gt_source_indices = gt_data["source_gaussian_indices"]
    np.random.seed(SEED)
    selected_rows = np.random.choice(len(gt_source_indices), NUM_SOURCES, replace=False).tolist()
    gt_dists = gt_all_dists[selected_rows[0]].astype(np.float64)
    source_gaussian_idxs = gt_source_indices[selected_rows]

    eval_mask = np.ones(N, dtype=bool)
    for sg_idx in source_gaussian_idxs:
        eval_mask[int(sg_idx)] = False
    eval_mask[~np.isfinite(gt_dists)] = False
    eval_mask[gt_dists <= 0] = False
    eval_indices = np.where(eval_mask)[0]

    # Build ring-1 with k=10
    ring1_k10, _, _ = ring1_neighbors_gaussians(positions, n_neighbors=10, use_mahalanobis=False)
    # Build ring-1 with k=16
    ring1_k16, _, _ = ring1_neighbors_gaussians(positions, n_neighbors=16, use_mahalanobis=False)

    # Find fallback points at k=10
    fallback_pids = []
    for pid in eval_indices:
        nbrs = get_neighborhood_by_ring(pid, RING, ring1_k10)
        if len(nbrs) == 0:
            continue
        if not np.any(gt_dists[nbrs] <= gt_dists[pid]):
            fallback_pids.append(pid)

    print(f"\n{'='*70}")
    print(f"  {name}: {len(fallback_pids)} fallback points at k=10")
    print(f"{'='*70}")
    print(f"  {'pid':>7s}  {'gt_dist':>12s}  {'r3_k10':>7s}  {'r3_k16':>7s}  {'k16_fix?':>8s}  {'k16<=128?':>9s}")
    print(f"  {'-'*60}")

    n_fixed = 0
    n_under_128 = 0
    counts_k16 = []
    for pid in fallback_pids:
        nbrs_k10 = get_neighborhood_by_ring(pid, RING, ring1_k10)
        nbrs_k16 = get_neighborhood_by_ring(pid, RING, ring1_k16)
        cnt_k10 = len(nbrs_k10)
        cnt_k16 = len(nbrs_k16)
        counts_k16.append(cnt_k16)

        has_closer_k16 = np.any(gt_dists[nbrs_k16] <= gt_dists[pid]) if len(nbrs_k16) > 0 else False
        if has_closer_k16:
            n_fixed += 1
        under_128 = cnt_k16 <= 128
        if under_128:
            n_under_128 += 1

        print(f"  {pid:>7d}  {gt_dists[pid]:>12.8f}  {cnt_k10:>7d}  {cnt_k16:>7d}  "
              f"{'YES' if has_closer_k16 else 'NO':>8s}  {'YES' if under_128 else 'NO':>9s}")

    counts_k16 = np.array(counts_k16) if counts_k16 else np.array([0])
    print(f"\n  Summary:")
    print(f"    Fixed by k=16:   {n_fixed}/{len(fallback_pids)}")
    print(f"    k16 count <= 128: {n_under_128}/{len(fallback_pids)}")
    print(f"    k16 ring-3 counts: min={counts_k16.min()}  mean={counts_k16.mean():.1f}  "
          f"max={counts_k16.max()}  p95={np.percentile(counts_k16, 95):.0f}")
