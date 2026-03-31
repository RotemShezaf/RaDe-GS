#!/usr/bin/env python3
"""Analyze target ranges across ALL datasets in TrainData/datasets/gaussian_patches/.
Compares TOSCA vs polynomial datasets to check if extreme targets exist in both.
Uses memory-mapped loading to handle multi-GB files efficiently."""

import numpy as np
import os
import glob
import sys
from collections import defaultdict

BASE = "/home/rotem.shezaf/RaDe-GS/TrainData/datasets/gaussian_patches"

def analyze_dataset(path, label):
    """Analyze all .npy files in a dataset directory using memory-mapped loading."""
    npy_files = sorted(glob.glob(os.path.join(path, "*.npy")))
    if not npy_files:
        return None
    
    all_targets = []
    all_r1_min = []
    total_patches = 0
    
    for f in npy_files:
        print(f"  Loading {os.path.basename(f)}...", end=" ", flush=True)
        # Use memory-mapped mode to avoid loading entire file into RAM
        data = np.load(f, mmap_mode='r')
        if data.ndim != 2 or data.shape[1] < 2:
            print("SKIP (bad shape)")
            continue
        # Only copy the last 2 columns into memory (target + r1_min_val)
        targets = np.array(data[:, -1])   # last column = target
        r1_min = np.array(data[:, -2])    # second-to-last = r1_min_val
        all_targets.append(targets)
        all_r1_min.append(r1_min)
        total_patches += len(targets)
        print(f"{len(targets)} patches, shape={data.shape}")
        del data  # release mmap
    
    if total_patches == 0:
        return None
    
    targets = np.concatenate(all_targets)
    r1_min = np.concatenate(all_r1_min)
    
    neg_r1 = np.sum(r1_min < 0)
    zero_r1 = np.sum(r1_min == 0)
    
    # Extreme targets: abs > 3, abs > 10, abs > 50
    ext3 = np.sum(np.abs(targets) > 3)
    ext10 = np.sum(np.abs(targets) > 10)
    ext50 = np.sum(np.abs(targets) > 50)
    
    # Negative targets
    neg_targets = np.sum(targets < 0)
    
    return {
        "label": label,
        "n_files": len(npy_files),
        "n_patches": total_patches,
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "target_median": float(np.median(targets)),
        "target_p1": float(np.percentile(targets, 1)),
        "target_p5": float(np.percentile(targets, 5)),
        "target_p95": float(np.percentile(targets, 95)),
        "target_p99": float(np.percentile(targets, 99)),
        "target_p999": float(np.percentile(targets, 99.9)),
        "neg_targets": neg_targets,
        "ext3": ext3,
        "ext10": ext10,
        "ext50": ext50,
        "r1_min_min": float(np.min(r1_min)),
        "r1_min_max": float(np.max(r1_min)),
        "neg_r1": neg_r1,
        "zero_r1": zero_r1,
        "neg_r1_pct": 100.0 * neg_r1 / total_patches,
    }

def print_result(r):
    print(f"\n{'='*80}")
    print(f"  {r['label']}")
    print(f"{'='*80}")
    print(f"  Files: {r['n_files']}, Patches: {r['n_patches']}")
    print(f"  TARGET  min={r['target_min']:.4f}  max={r['target_max']:.4f}  mean={r['target_mean']:.4f}  std={r['target_std']:.4f}  median={r['target_median']:.4f}")
    print(f"  TARGET percentiles  p1={r['target_p1']:.4f}  p5={r['target_p5']:.4f}  p95={r['target_p95']:.4f}  p99={r['target_p99']:.4f}  p99.9={r['target_p999']:.4f}")
    print(f"  Negative targets: {r['neg_targets']} ({100*r['neg_targets']/r['n_patches']:.2f}%)")
    print(f"  |target|>3: {r['ext3']}  |target|>10: {r['ext10']}  |target|>50: {r['ext50']}")
    print(f"  R1_MIN  min={r['r1_min_min']:.4f}  max={r['r1_min_max']:.4f}")
    print(f"  Negative r1_min: {r['neg_r1']} ({r['neg_r1_pct']:.2f}%)  Zero r1_min: {r['zero_r1']}")

# Collect all datasets
datasets = []

# Top-level datasets
for name in sorted(os.listdir(BASE)):
    path = os.path.join(BASE, name)
    if os.path.isdir(path) and name not in ("outlier_filtering", "split_enc_dec", "new_dataset"):
        datasets.append((path, name))

# outlier_filtering subfolder
of_path = os.path.join(BASE, "outlier_filtering")
if os.path.isdir(of_path):
    for name in sorted(os.listdir(of_path)):
        path = os.path.join(of_path, name)
        if os.path.isdir(path):
            datasets.append((path, f"outlier_filtering/{name}"))

# split_enc_dec subfolder
sed_path = os.path.join(BASE, "split_enc_dec")
if os.path.isdir(sed_path):
    for name in sorted(os.listdir(sed_path)):
        path = os.path.join(sed_path, name)
        if os.path.isdir(path):
            datasets.append((path, f"split_enc_dec/{name}"))

# Analyze all
results_tosca = []
results_poly = []
results_other = []

for path, label in datasets:
    r = analyze_dataset(path, label)
    if r is None:
        print(f"  SKIP {label} (no data)")
        continue
    
    if "tosca" in label.lower():
        results_tosca.append(r)
    elif any(p in label.lower() for p in ("paraboloid", "saddle", "hyperbolic")):
        results_poly.append(r)
    else:
        results_other.append(r)

# Print results grouped
print("\n" + "#"*80)
print("#  POLYNOMIAL DATASETS")
print("#"*80)
for r in results_poly:
    print_result(r)

print("\n" + "#"*80)
print("#  TOSCA DATASETS")
print("#"*80)
for r in results_tosca:
    print_result(r)

if results_other:
    print("\n" + "#"*80)
    print("#  OTHER DATASETS")
    print("#"*80)
    for r in results_other:
        print_result(r)

# Summary comparison
print("\n\n" + "#"*80)
print("#  SUMMARY COMPARISON")
print("#"*80)

print(f"\n{'Dataset':<55} {'Patches':>8} {'TargetMax':>10} {'|t|>3':>6} {'|t|>10':>7} {'|t|>50':>7} {'NegR1%':>7}")
print("-"*100)

for group_name, group in [("POLYNOMIAL", results_poly), ("TOSCA", results_tosca), ("OTHER", results_other)]:
    if not group:
        continue
    for r in group:
        print(f"{r['label']:<55} {r['n_patches']:>8} {r['target_max']:>10.2f} {r['ext3']:>6} {r['ext10']:>7} {r['ext50']:>7} {r['neg_r1_pct']:>7.2f}")
    print()
