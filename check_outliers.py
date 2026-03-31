#!/usr/bin/env python3
"""Check outlier statistics in the newly generated TOSCA training patches."""
import numpy as np
from pathlib import Path
import yaml

# Load config to get ring_size_mapping
config_path = Path('DataSets/configs/tosca/combined_tosca_all.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

ring = 3
max_nbrs = config['ring_size_mapping']['euclidean'][ring]
print(f"Ring: {ring}, max_neighbors: {max_nbrs}")

# Attribute dimension: xyz=3, +1 for geodesic = 4 features per neighbor
attr_dim = 4  # xyz(3) + geodesic(1)
# Example layout: [neighborhood(max_nbrs * attr_dim), point_features(3), r1_min_val, target]
example_size = max_nbrs * attr_dim + 3 + 1 + 1  # neighborhood + point_feats + r1_min + target

base = Path('TrainData/datasets/gaussian_patches')
shapes = sorted([d.name for d in base.iterdir() if d.name.startswith('tosca_') and d.name.endswith('_blue')])

print(f"\n{'Shape':>25} | {'N_examples':>10} | {'target_mean':>11} {'target_max':>11} {'target_p99':>11} | {'ratio>5':>8} {'ratio>10':>8} {'max_ratio':>10}")
print('-' * 120)

all_targets = []
all_max_xyz = []

for shape_dir_name in shapes:
    npy_path = base / shape_dir_name / f'gaussian_examples_ring3_n50000.npy'
    if not npy_path.exists():
        continue
    
    data = np.load(str(npy_path))
    n_examples = len(data)
    
    # Extract target (last element)
    targets = data[:, -1]
    
    # Extract neighborhood xyz and find max Euclidean distance
    neighborhood = data[:, :max_nbrs * attr_dim].reshape(n_examples, max_nbrs, attr_dim)
    xyz = neighborhood[:, :, :3]  # (N, max_nbrs, 3)
    
    # Find max Euclidean distance per example (excluding masked neighbors with xyz=mask_constant)
    mask_constant = -10.0
    valid_mask = ~np.all(xyz == mask_constant, axis=2)  # (N, max_nbrs)
    
    eucl_dists = np.linalg.norm(xyz, axis=2)  # (N, max_nbrs) — xyz is already relative to center
    eucl_dists[~valid_mask] = 0
    max_eucl = eucl_dists.max(axis=1)  # (N,)
    
    # Compute ratio = target / max_euclidean_distance
    safe_max_eucl = np.where(max_eucl > 0, max_eucl, 1.0)
    ratios = np.abs(targets) / safe_max_eucl
    
    pct_over5 = (ratios > 5).mean() * 100
    pct_over10 = (ratios > 10).mean() * 100
    max_r = ratios.max()
    
    shape_name = shape_dir_name.replace('tosca_', '').replace('_blue', '')
    print(f"{shape_name:>25} | {n_examples:>10} | {targets.mean():>11.4f} {targets.max():>11.4f} {np.percentile(targets, 99):>11.4f} | {pct_over5:>7.3f}% {pct_over10:>7.3f}% {max_r:>10.2f}")
    
    all_targets.extend(targets.tolist())
    all_max_xyz.extend(max_eucl.tolist())

all_targets = np.array(all_targets)
all_max_xyz = np.array(all_max_xyz)
safe_all_max = np.where(all_max_xyz > 0, all_max_xyz, 1.0)
all_ratios = np.abs(all_targets) / safe_all_max

print()
print(f"Overall: {len(all_targets)} examples")
print(f"  Target:  mean={all_targets.mean():.4f}, p50={np.median(all_targets):.4f}, p99={np.percentile(all_targets, 99):.4f}, max={all_targets.max():.4f}")
print(f"  Ratio:   mean={all_ratios.mean():.4f}, p50={np.median(all_ratios):.4f}, p99={np.percentile(all_ratios, 99):.4f}, max={all_ratios.max():.4f}")
print(f"  Ratio>5: {(all_ratios > 5).sum()} ({(all_ratios > 5).mean()*100:.4f}%)")
print(f"  Ratio>10: {(all_ratios > 10).sum()} ({(all_ratios > 10).mean()*100:.4f}%)")
print(f"  Ratio>50: {(all_ratios > 50).sum()} ({(all_ratios > 50).mean()*100:.4f}%)")

# Check MSE vs MAE
print()
mse = (all_targets ** 2).mean()
mae = np.abs(all_targets).mean()
print(f"  If predicting zero:")
print(f"    MSE = {mse:.4f}")
print(f"    MAE = {mae:.4f}")
print(f"    MSE/MAE ratio = {mse/mae:.2f}")

# Check what top outliers contribute
sorted_sq = np.sort(all_targets ** 2)[::-1]
total_sq = sorted_sq.sum()
top10_pct = sorted_sq[:10].sum() / total_sq * 100
top100_pct = sorted_sq[:100].sum() / total_sq * 100
print(f"    Top 10 outliers contribute: {top10_pct:.2f}% of total MSE")
print(f"    Top 100 outliers contribute: {top100_pct:.2f}% of total MSE")
