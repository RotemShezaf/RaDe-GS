"""Deeper analysis: find the best discriminating features for floaters."""
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree

PC_PATH = "TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_4/output/point_cloud/iteration_30000/point_cloud.ply"
MESH_PATH = "TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_4/output/recon.ply"

pc = PlyData.read(PC_PATH)
v = pc['vertex']
xyz = np.column_stack([v['x'], v['y'], v['z']])
opacity_raw = v['opacity']
opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
scales_raw = np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']])
scales = np.exp(scales_raw)
f_dc = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']])

mesh = PlyData.read(MESH_PATH)
mesh_verts = np.column_stack([mesh['vertex']['x'], mesh['vertex']['y'], mesh['vertex']['z']])
tree = KDTree(mesh_verts)
dists, _ = tree.query(xyz)

max_scale = scales.max(axis=1)
min_scale = scales.min(axis=1)
mean_scale = scales.mean(axis=1)
volume = scales.prod(axis=1)
scale_ratio = max_scale / (min_scale + 1e-10)

# Weighted opacity*volume: a floater "impact" metric
impact = opacity * volume

print("=" * 80)
print("CORRELATION WITH DISTANCE (what predicts being far from mesh?)")
print("=" * 80)
features = {
    'opacity': opacity,
    'max_scale': max_scale,
    'min_scale': min_scale,
    'mean_scale': mean_scale,
    'volume': volume,
    'scale_ratio': scale_ratio,
    'log_volume': np.log(volume + 1e-15),
    'log_max_scale': np.log(max_scale + 1e-15),
    'impact (opacity*volume)': impact,
    'log_impact': np.log(impact + 1e-15),
    'opacity*max_scale': opacity * max_scale,
}
for name, feat in features.items():
    r = np.corrcoef(dists, feat)[0, 1]
    print(f"  {name:30s}  r = {r:+.4f}")

# Try various pruning criteria and see how well they separate
print("\n" + "=" * 80)
print("PRUNING CRITERION ANALYSIS")
print("Evaluating how well each criterion removes far Gaussians while keeping close ones")
print("Using dist > 1.0 as 'far' definition (meaningful geometric distance)")
print("=" * 80)

far_mask = dists > 1.0
close_mask = dists < 0.1
mid_mask = ~far_mask & ~close_mask
n_total = len(dists)
n_far = far_mask.sum()
n_close = close_mask.sum()
print(f"Close (<0.1): {n_close}, Mid (0.1-1.0): {mid_mask.sum()}, Far (>1.0): {n_far}")

print(f"\n{'Criterion':40s} | {'Pruned':>7} | {'Far pruned':>10} | {'Close pruned':>12} | {'Precision':>9} | {'Recall':>7}")
print("-" * 100)

def eval_criterion(name, mask):
    pruned = mask.sum()
    far_pruned = (mask & far_mask).sum()
    close_pruned = (mask & close_mask).sum()
    precision = far_pruned / (pruned + 1e-10)  # of pruned, how many were far?
    recall = far_pruned / (n_far + 1e-10)  # of far, how many got pruned?
    print(f"{name:40s} | {pruned:7d} | {far_pruned:10d} | {close_pruned:12d} | {precision:8.1%} | {recall:6.1%}")

# Volume-based
for t in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    eval_criterion(f"volume > {t}", volume > t)

print()
# Max scale absolute
for t in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    eval_criterion(f"max_scale > {t}", max_scale > t)

print()
# Impact (opacity * volume)
for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
    eval_criterion(f"impact (op*vol) > {t}", impact > t)

print()
# Scale factor relative to extent (configuring the 0.1 factor)
extent = 297.36
for factor in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
    thresh = factor * extent
    eval_criterion(f"max_scale > {factor}*extent (={thresh:.2f})", max_scale > thresh)

print()
# Min scale (flatness filter - surface Gaussians should be flat/pancake)
for t in [0.01, 0.02, 0.05, 0.1, 0.2]:
    eval_criterion(f"min_scale > {t}", min_scale > t)

print()
# Combined: high opacity + large volume
for vt in [0.05, 0.1, 0.2]:
    for ot in [0.5, 0.8]:
        mask = (volume > vt) & (opacity > ot)
        eval_criterion(f"vol>{vt} AND op>{ot}", mask)

print()
# Opacity-only (existing mechanism)
for t in [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]:
    eval_criterion(f"opacity < {t} (EXISTING)", opacity < t)

print()
# What about screen-space size = 20? Check max_radii2D distribution
# We can't check that here, but we can check scale_ratio
for t in [5, 10, 20, 50]:
    eval_criterion(f"scale_ratio < {t} (blobby/round)", scale_ratio < t)

print("\n" + "=" * 80)
print("BEST APPROACH: Combined volume + scale threshold for this scene")
print("=" * 80)

# Find the sweet spot
best = None
for vt in np.arange(0.02, 0.5, 0.01):
    mask = volume > vt
    pruned = mask.sum()
    far_pruned = (mask & far_mask).sum()
    close_pruned = (mask & close_mask).sum()
    if pruned == 0:
        continue
    precision = far_pruned / pruned
    recall = far_pruned / n_far
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    if best is None or f1 > best[0]:
        best = (f1, vt, precision, recall, pruned, far_pruned, close_pruned)

if best:
    f1, vt, prec, rec, pruned, fp, cp = best
    print(f"Best volume threshold: {vt:.2f}")
    print(f"  F1={f1:.3f}, Precision={prec:.1%}, Recall={rec:.1%}")
    print(f"  Pruned: {pruned}, Far pruned: {fp}, Close pruned: {cp}")

# Also check combined with max_scale
best2 = None
for st in np.arange(1.0, 5.0, 0.1):
    mask = max_scale > st
    pruned = mask.sum()
    far_pruned = (mask & far_mask).sum()
    close_pruned = (mask & close_mask).sum()
    if pruned == 0:
        continue
    precision = far_pruned / pruned
    recall = far_pruned / n_far
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    if best2 is None or f1 > best2[0]:
        best2 = (f1, st, precision, recall, pruned, far_pruned, close_pruned)

if best2:
    f1, st, prec, rec, pruned, fp, cp = best2
    print(f"\nBest max_scale threshold: {st:.1f}")
    print(f"  F1={f1:.3f}, Precision={prec:.1%}, Recall={rec:.1%}")
    print(f"  Pruned: {pruned}, Far pruned: {fp}, Close pruned: {cp}")
