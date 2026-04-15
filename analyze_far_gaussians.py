"""Analyze Gaussians that are far from the reconstructed mesh."""
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
import sys

PC_PATH = "TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_4/output/point_cloud/iteration_30000/point_cloud.ply"
MESH_PATH = "TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/light_4/output/recon.ply"

# Load data
print("Loading point cloud...")
pc = PlyData.read(PC_PATH)
v = pc['vertex']
xyz = np.column_stack([v['x'], v['y'], v['z']])
opacity_raw = v['opacity']  # raw logit (pre-sigmoid)
opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid
scales_raw = np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']])
scales = np.exp(scales_raw)  # scales stored as log
normals = np.column_stack([v['nx'], v['ny'], v['nz']])
f_dc = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']])
filter_3d = v['filter_3D']

print(f"  {len(xyz)} Gaussians loaded")

print("Loading mesh...")
mesh = PlyData.read(MESH_PATH)
mesh_verts = np.column_stack([mesh['vertex']['x'], mesh['vertex']['y'], mesh['vertex']['z']])
print(f"  {len(mesh_verts)} mesh vertices loaded")

# Build KDTree on mesh vertices and compute distances
print("Computing distances from Gaussians to mesh...")
tree = KDTree(mesh_verts)
dists, _ = tree.query(xyz)

# Define distance thresholds
scene_diag = np.linalg.norm(mesh_verts.max(axis=0) - mesh_verts.min(axis=0))
print(f"\nScene diagonal: {scene_diag:.4f}")
print(f"Distance stats: min={dists.min():.6f}, median={np.median(dists):.6f}, "
      f"mean={dists.mean():.6f}, 95th={np.percentile(dists, 95):.6f}, "
      f"99th={np.percentile(dists, 99):.6f}, max={dists.max():.6f}")

# Thresholds
thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
print(f"\n{'Threshold':>10} | {'# Far':>7} | {'% Far':>6} | {'Mean Opacity':>13} | {'Median Scale':>13} | {'Max Scale':>10}")
print("-" * 80)
for t in thresholds:
    far = dists > t
    n_far = far.sum()
    pct = 100.0 * n_far / len(dists)
    if n_far > 0:
        mo = opacity[far].mean()
        ms = np.median(scales[far].max(axis=1))
        mxs = scales[far].max()
    else:
        mo = ms = mxs = 0
    print(f"{t:10.4f} | {n_far:7d} | {pct:5.1f}% | {mo:13.6f} | {ms:13.8f} | {mxs:10.6f}")

# Detailed analysis: close vs far
threshold = 0.01  # pick a reasonable threshold
far_mask = dists > threshold
close_mask = ~far_mask
print(f"\n{'='*80}")
print(f"DETAILED COMPARISON (threshold = {threshold})")
print(f"  Close: {close_mask.sum()} Gaussians, Far: {far_mask.sum()} Gaussians")
print(f"{'='*80}")

def describe(name, close_vals, far_vals):
    print(f"\n--- {name} ---")
    print(f"  {'':15} {'Close':>12} {'Far':>12}")
    print(f"  {'Mean':15} {close_vals.mean():12.6f} {far_vals.mean():12.6f}")
    print(f"  {'Median':15} {np.median(close_vals):12.6f} {np.median(far_vals):12.6f}")
    print(f"  {'Std':15} {close_vals.std():12.6f} {far_vals.std():12.6f}")
    print(f"  {'Min':15} {close_vals.min():12.6f} {far_vals.min():12.6f}")
    print(f"  {'Max':15} {close_vals.max():12.6f} {far_vals.max():12.6f}")
    print(f"  {'5th pct':15} {np.percentile(close_vals,5):12.6f} {np.percentile(far_vals,5):12.6f}")
    print(f"  {'95th pct':15} {np.percentile(close_vals,95):12.6f} {np.percentile(far_vals,95):12.6f}")

if far_mask.sum() > 0:
    describe("Opacity (sigmoid)", opacity[close_mask], opacity[far_mask])
    describe("Opacity (raw logit)", opacity_raw[close_mask], opacity_raw[far_mask])
    describe("Max Scale (exp)", scales[close_mask].max(axis=1), scales[far_mask].max(axis=1))
    describe("Mean Scale (exp)", scales[close_mask].mean(axis=1), scales[far_mask].mean(axis=1))
    describe("Min Scale (exp)", scales[close_mask].min(axis=1), scales[far_mask].min(axis=1))
    describe("Scale_0 (raw log)", scales_raw[close_mask, 0], scales_raw[far_mask, 0])
    describe("Scale_1 (raw log)", scales_raw[close_mask, 1], scales_raw[far_mask, 1])
    describe("Scale_2 (raw log)", scales_raw[close_mask, 2], scales_raw[far_mask, 2])
    
    # Scale ratio (max/min) - measure of anisotropy
    scale_ratio_close = scales[close_mask].max(axis=1) / (scales[close_mask].min(axis=1) + 1e-10)
    scale_ratio_far = scales[far_mask].max(axis=1) / (scales[far_mask].min(axis=1) + 1e-10)
    describe("Scale Ratio (max/min)", scale_ratio_close, scale_ratio_far)
    
    # Volume proxy = product of scales
    vol_close = scales[close_mask].prod(axis=1)
    vol_far = scales[far_mask].prod(axis=1)
    describe("Volume (scale product)", vol_close, vol_far)
    
    describe("filter_3D", filter_3d[close_mask], filter_3d[far_mask])
    
    # Normal magnitude
    norm_mag_close = np.linalg.norm(normals[close_mask], axis=1)
    norm_mag_far = np.linalg.norm(normals[far_mask], axis=1)
    describe("Normal magnitude", norm_mag_close, norm_mag_far)
    
    # SH DC coefficients (color)
    describe("f_dc_0 (R)", f_dc[close_mask, 0], f_dc[far_mask, 0])
    describe("f_dc_1 (G)", f_dc[close_mask, 1], f_dc[far_mask, 1])
    describe("f_dc_2 (B)", f_dc[close_mask, 2], f_dc[far_mask, 2])

    # What fraction of far Gaussians have very low opacity?
    print(f"\n{'='*80}")
    print("FAR GAUSSIAN BREAKDOWN")
    print(f"{'='*80}")
    for op_thresh in [0.01, 0.05, 0.1, 0.5, 0.9]:
        n = (opacity[far_mask] < op_thresh).sum()
        print(f"  Opacity < {op_thresh}: {n}/{far_mask.sum()} ({100*n/far_mask.sum():.1f}%)")
    
    # Scale distribution of far Gaussians
    print("\nFar Gaussians max-scale percentiles:")
    far_max_scale = scales[far_mask].max(axis=1)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(far_max_scale, p):.8f}")
    
    # Correlation: distance vs properties
    print(f"\n{'='*80}")
    print("CORRELATIONS (all Gaussians)")
    print(f"{'='*80}")
    print(f"  dist vs opacity:    r={np.corrcoef(dists, opacity)[0,1]:.4f}")
    print(f"  dist vs max_scale:  r={np.corrcoef(dists, scales.max(axis=1))[0,1]:.4f}")
    print(f"  dist vs volume:     r={np.corrcoef(dists, scales.prod(axis=1))[0,1]:.4f}")
    print(f"  dist vs filter_3D:  r={np.corrcoef(dists, filter_3d)[0,1]:.4f}")
    
    # Look at the VERY far Gaussians (top 1% by distance)
    top1_thresh = np.percentile(dists, 99)
    top1_mask = dists > top1_thresh
    print(f"\n{'='*80}")
    print(f"TOP 1% FARTHEST GAUSSIANS (dist > {top1_thresh:.6f}, n={top1_mask.sum()})")
    print(f"{'='*80}")
    if top1_mask.sum() > 0:
        print(f"  Opacity mean: {opacity[top1_mask].mean():.6f}, median: {np.median(opacity[top1_mask]):.6f}")
        print(f"  Max scale mean: {scales[top1_mask].max(axis=1).mean():.8f}")
        print(f"  Volume mean: {scales[top1_mask].prod(axis=1).mean():.10f}")
        print(f"  Distance range: {dists[top1_mask].min():.6f} - {dists[top1_mask].max():.6f}")
        # What percentage have opacity > 0.5?
        hi_op = (opacity[top1_mask] > 0.5).sum()
        print(f"  With opacity > 0.5: {hi_op}/{top1_mask.sum()} ({100*hi_op/top1_mask.sum():.1f}%)")
        hi_op2 = (opacity[top1_mask] > 0.1).sum()
        print(f"  With opacity > 0.1: {hi_op2}/{top1_mask.sum()} ({100*hi_op2/top1_mask.sum():.1f}%)")
else:
    print("No far Gaussians found!")
