#!/usr/bin/env python3
"""Quick data quality analysis for TOSCA outlier filtering datasets."""
import numpy as np
import os

animals = ['tosca_cat', 'tosca_centaur', 'tosca_david', 'tosca_dog', 'tosca_gorilla', 'tosca_horse', 'tosca_michael', 'tosca_victoria', 'tosca_wolf']
base = 'TrainData/datasets/gaussian_patches/outlier_filtering'

file_map = {
    'tosca_cat': 'n100000', 'tosca_centaur': 'n1500000', 'tosca_david': 'n50000',
    'tosca_dog': 'n50000', 'tosca_gorilla': 'n50000', 'tosca_horse': 'n50000',
    'tosca_michael': 'n100000', 'tosca_victoria': 'n100000', 'tosca_wolf': 'n50000'
}

total_neg = 0
total_zero_r1 = 0
total_samples = 0

for animal in animals:
    n = file_map[animal]
    path = os.path.join(base, animal, f'gaussian_examples_ring3_{n}.npy')
    data = np.load(path)
    target = data[:, -2]
    r1_min = data[:, -1]
    neg_targets = int((target < 0).sum())
    zero_r1 = int((r1_min < 1e-6).sum())
    total_neg += neg_targets
    total_zero_r1 += zero_r1
    total_samples += len(target)
    pct_bad = 100.0 * (neg_targets + zero_r1) / len(target)
    print(f"{animal}: shape={data.shape}, neg_tgt={neg_targets}, r1_zero={zero_r1}, pct_bad={pct_bad:.1f}%, tgt=[{target.min():.2f}, {target.max():.2f}], r1=[{r1_min.min():.6f}, {r1_min.max():.2f}]")

print(f"\nTOTAL: {total_samples} samples, {total_neg} negative targets ({100*total_neg/total_samples:.1f}%), {total_zero_r1} zero r1_min ({100*total_zero_r1/total_samples:.1f}%)")

# After-normalization analysis: simulate what happens during training
print("\n--- Post-normalization target distribution (tosca_cat) ---")
data = np.load(os.path.join(base, 'tosca_cat', 'gaussian_examples_ring3_n100000.npy'))
target = data[:, -2]
r1_min = data[:, -1]

# Simulate xyz normalization: max_dist = max euclidean distance to neighbors
# Each neighbor: [x, y, z, geo] = 4 cols, 192 neighbors = 768 cols, then point xyz (3) = 771 features, then target, r1_min
n_nbrs = 192
xyz_cols = []
for i in range(n_nbrs):
    xyz_cols.extend([4*i, 4*i+1, 4*i+2])

mask_constant = -10.0
good_count = 0
bad_count = 0 
max_normalized_targets = []

for idx in range(min(5000, len(target))):
    row = data[idx]
    # Get neighbor xyz
    nbr_xyz = row[:n_nbrs*4].reshape(n_nbrs, 4)[:, :3]
    # Point xyz is at cols 768, 769, 770
    pt_xyz = row[n_nbrs*4:n_nbrs*4+3]
    
    # Get valid neighbors (not masked)
    valid = nbr_xyz[:, 0] != mask_constant
    if not valid.any():
        continue
    
    # Center
    centered = nbr_xyz[valid] - pt_xyz
    dists = np.linalg.norm(centered, axis=1)
    max_dist = max(dists.max(), 1e-8)
    
    # Normalized target
    norm_target = target[idx] / max_dist
    max_normalized_targets.append(norm_target)
    
    if abs(norm_target) > 10:
        bad_count += 1
    else:
        good_count += 1

norm_tgts = np.array(max_normalized_targets)
print(f"Samples analyzed: {len(norm_tgts)}")
print(f"Normalized target stats: min={norm_tgts.min():.4f}, max={norm_tgts.max():.4f}, mean={norm_tgts.mean():.4f}, std={norm_tgts.std():.4f}")
print(f"  |norm_tgt| > 10: {(np.abs(norm_tgts) > 10).sum()}")
print(f"  |norm_tgt| > 5: {(np.abs(norm_tgts) > 5).sum()}")
print(f"  |norm_tgt| > 2: {(np.abs(norm_tgts) > 2).sum()}")
print(f"  norm_tgt < 0: {(norm_tgts < 0).sum()}")
print(f"  Percentiles: 0.1%={np.percentile(norm_tgts, 0.1):.4f}, 1%={np.percentile(norm_tgts, 1):.4f}, 99%={np.percentile(norm_tgts, 99):.4f}, 99.9%={np.percentile(norm_tgts, 99.9):.4f}")
