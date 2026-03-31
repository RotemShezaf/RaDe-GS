#!/usr/bin/env python3
"""Analyze normalized target distribution across TOSCA datasets."""
import numpy as np
import os
import glob

base = 'TrainData/datasets/gaussian_patches'
animals = sorted([
    'tosca_cat', 'tosca_centaur', 'tosca_david', 'tosca_dog',
    'tosca_gorilla', 'tosca_horse', 'tosca_michael', 'tosca_victoria', 'tosca_wolf'
])

# 773 cols = source_xyz(3) + r1_min_val(1) + 192*[xyz(3)+geo(1)] + target(1)
max_nbrs = 192

print('=' * 90)
print('COMPREHENSIVE NORMALIZED TARGET ANALYSIS')
print('=' * 90)

all_norm_targets = []

for name in animals:
    fpath = glob.glob(os.path.join(base, name, '*ring3*.npy'))[0]
    data = np.load(fpath)
    n_examples = data.shape[0]

    sample_n = min(10000, n_examples)
    idx = np.random.RandomState(42).choice(n_examples, sample_n, replace=False)
    sample = data[idx]

    source_xyz = sample[:, 0:3]
    target = sample[:, -1]  # col 772

    # Extract neighbor xyz (starting col 4, stride 4)
    nbr_xyz = np.zeros((sample_n, max_nbrs, 3))
    for j in range(max_nbrs):
        start = 4 + j * 4
        nbr_xyz[:, j, :] = sample[:, start:start + 3]

    centered = nbr_xyz - source_xyz[:, np.newaxis, :]
    dists = np.linalg.norm(centered, axis=2)
    max_euc = dists.max(axis=1)

    norm_target = target / np.clip(max_euc, 1e-8, None)
    all_norm_targets.extend(norm_target)

    pcts = [50, 90, 99, 99.9, 100]
    pv = np.percentile(norm_target, pcts)
    n5 = (norm_target > 5).sum()
    n10 = (norm_target > 10).sum()
    print(f'{name:20s}: mean={norm_target.mean():.3f}, p99={pv[2]:.3f}, p99.9={pv[3]:.3f}, max={pv[4]:.1f}, >5:{n5}, >10:{n10}')

all_norm = np.array(all_norm_targets)
N = len(all_norm)

print()
print(f'COMBINED: {N} examples')
pcts = [50, 90, 95, 99, 99.5, 99.9, 99.95, 99.99, 100]
pv = np.percentile(all_norm, pcts)
print(f'  Percentiles: ' + ', '.join(f'p{p}={v:.4f}' for p, v in zip(pcts, pv)))

print()
print('--- OUTLIER COUNTS AND MSE IMPACT ---')
for t in [1, 2, 3, 5, 10, 20, 50, 100]:
    mask = all_norm > t
    cnt = mask.sum()
    pct = 100 * cnt / N
    if cnt > 0:
        mean_good = all_norm[all_norm <= t].mean() if (all_norm <= t).sum() > 0 else 0
        sq_errors = (all_norm[mask] - mean_good) ** 2
        mse_from_outliers = sq_errors.sum() / N
        print(f'  > {t:>3d}: {cnt:>5d} ({pct:>6.3f}%), MSE contrib if model predicts mean: {mse_from_outliers:.4f}')
    else:
        print(f'  > {t:>3d}: {cnt:>5d}')

print()
print('--- NEAR-ZERO TARGETS (rel error explosion) ---')
for tv in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
    cnt = (all_norm < tv).sum()
    pct = 100 * cnt / N
    rel_per = 0.065 / tv * 100
    print(f'  < {tv:.3f}: {cnt:>6d} ({pct:>5.2f}%), at MAE=0.065 -> rel_error: {rel_per:.0f}%')

print()
print('--- EFFECT OF CLIPPING TARGETS ---')
for clip in [2, 3, 5, 10]:
    clipped = np.clip(all_norm, 0, clip)
    print(f'  Clip at {clip:>2d}: std={clipped.std():.4f}, var={clipped.var():.4f}, mean={clipped.mean():.4f}')

print()
print('--- EXTREME EXAMPLES ---')
extreme_idx = np.where(all_norm > 50)[0]
if len(extreme_idx) > 0:
    print(f'  {len(extreme_idx)} examples with norm_target > 50')

print()
print('DONE')
