#!/usr/bin/env python3
"""Check ring neighbor statistics for all TOSCA shapes."""
import numpy as np
from plyfile import PlyData
from pathlib import Path
from GenerateData.utils.data_generation_utils import get_all_points_nbrs_single_ring, ring1_neighbors_gaussians

base = Path('TrainData/TOSCA/SyntheticColmapData/blue_texture')
shapes = sorted([d.name for d in base.iterdir() if d.is_dir()])

header = (f"{'Shape':>15} | {'N_Gauss':>8} | "
          f"{'R1 mean':>8} {'R1 max':>7} | "
          f"{'R2 mean':>8} {'R2 max':>7} {'R2 p99':>7} | "
          f"{'R3 mean':>8} {'R3 max':>7} {'R3 p99':>7} {'R3 p95':>7} | "
          f"{'MeanNN':>8}")
print(header)
print('-' * 130)

all_r2 = []
all_r3 = []

for shape in shapes:
    plys = list((base / shape).glob('**/point_cloud/iteration_*/point_cloud.ply'))
    if not plys:
        continue
    ply = PlyData.read(str(plys[0]))
    v = ply['vertex']
    positions = np.column_stack([v['x'], v['y'], v['z']])
    n = len(positions)

    # Only compute ring-1, then individually ring-2 and ring-3 (skip ring-4)
    ring1, ring2 = get_all_points_nbrs_single_ring(positions, ring=2, n_neighbors_ring1=10)
    _, ring3 = get_all_points_nbrs_single_ring(positions, ring=3, n_neighbors_ring1=10)
    _, mean_dist, _ = ring1_neighbors_gaussians(positions, n_neighbors=10, use_mahalanobis=False)

    r1_sizes = np.array([len(ring1[i]) for i in range(n)])
    r2_sizes = np.array([len(ring2[i]) for i in range(n)])
    r3_sizes = np.array([len(ring3[i]) for i in range(n)])

    all_r2.extend(r2_sizes.tolist())
    all_r3.extend(r3_sizes.tolist())

    print(f"{shape:>15} | {n:>8} | "
          f"{r1_sizes.mean():>8.1f} {r1_sizes.max():>7} | "
          f"{r2_sizes.mean():>8.1f} {r2_sizes.max():>7} {np.percentile(r2_sizes, 99):>7.0f} | "
          f"{r3_sizes.mean():>8.1f} {r3_sizes.max():>7} {np.percentile(r3_sizes, 99):>7.0f} {np.percentile(r3_sizes, 95):>7.0f} | "
          f"{mean_dist:>8.4f}")

print()
all_r2 = np.array(all_r2)
all_r3 = np.array(all_r3)
print(f"Overall ring-2: mean={all_r2.mean():.1f}, p50={np.median(all_r2):.0f}, p95={np.percentile(all_r2,95):.0f}, p99={np.percentile(all_r2,99):.0f}, max={all_r2.max()}")
print(f"Overall ring-3: mean={all_r3.mean():.1f}, p50={np.median(all_r3):.0f}, p95={np.percentile(all_r3,95):.0f}, p99={np.percentile(all_r3,99):.0f}, max={all_r3.max()}")
print()
# Show how many points would be trimmed at various ring-3 thresholds
for threshold in [64, 128, 192, 256, 384, 512]:
    pct_over = (all_r3 > threshold).mean() * 100
    print(f"  Ring-3 > {threshold}: {pct_over:.2f}% of points would be trimmed")
