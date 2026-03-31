#!/usr/bin/env python3
"""
Comprehensive diagnostic for blue_texture TOSCA data.
Checks Gaussian splat positions vs GT mesh positions to identify
why points are far from the mesh.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import glob

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import trimesh


def load_ply_mesh(path):
    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def load_gaussian_ply(path):
    """Load Gaussian splat positions from PLY."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    x = np.asarray(vertex['x'], dtype=np.float64)
    y = np.asarray(vertex['y'], dtype=np.float64)
    z = np.asarray(vertex['z'], dtype=np.float64)
    positions = np.column_stack([x, y, z])
    
    # Also try to get opacity and scales if available
    extra = {}
    for prop_name in ['opacity', 'scale_0', 'scale_1', 'scale_2']:
        if prop_name in [p.name for p in vertex.properties]:
            extra[prop_name] = np.asarray(vertex[prop_name], dtype=np.float64)
    
    return positions, extra


def sigmoid(x):
    x = np.clip(x, -88, 88)
    return 1.0 / (1.0 + np.exp(-x))


def bbox_str(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return f"  min=({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})  max=({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})  range=({mx[0]-mn[0]:.4f}, {mx[1]-mn[1]:.4f}, {mx[2]-mn[2]:.4f})"


def analyze_shape(shape_name, data_root, processed_root, resolution="high_res"):
    print(f"\n{'='*80}")
    print(f"  SHAPE: {shape_name} / {resolution}")
    print(f"{'='*80}")
    
    # --- 1. Load GT mesh ---
    shape_dir = processed_root / shape_name
    mesh_candidates = sorted(shape_dir.glob(f"mesh_{resolution}_*.ply"))
    if not mesh_candidates:
        print(f"  [SKIP] No GT mesh found in {shape_dir}")
        return
    mesh_path = mesh_candidates[0]
    print(f"\n  GT Mesh: {mesh_path.name}")
    mesh_verts, mesh_faces = load_ply_mesh(str(mesh_path))
    print(f"    Vertices: {len(mesh_verts):,}  Faces: {len(mesh_faces):,}")
    print(f"    Bounding box:")
    print(f"    {bbox_str(mesh_verts)}")
    mesh_center = mesh_verts.mean(axis=0)
    print(f"    Centroid: ({mesh_center[0]:.4f}, {mesh_center[1]:.4f}, {mesh_center[2]:.4f})")
    mesh_extent = np.linalg.norm(mesh_verts.max(axis=0) - mesh_verts.min(axis=0))
    print(f"    Diagonal extent: {mesh_extent:.4f}")
    
    # --- 2. Load Gaussian splats ---
    # Find the output directory
    shape_data_dir = data_root / shape_name / resolution
    
    # Search for point_cloud.ply recursively
    pc_candidates = sorted(shape_data_dir.rglob("point_cloud/iteration_*/point_cloud.ply"))
    if not pc_candidates:
        print(f"  [SKIP] No Gaussian point cloud found under {shape_data_dir}")
        return
    
    # Use highest iteration
    pc_path = pc_candidates[-1]
    print(f"\n  Gaussian PLY: {pc_path.relative_to(data_root)}")
    gauss_pos, gauss_extra = load_gaussian_ply(str(pc_path))
    print(f"    Gaussians: {len(gauss_pos):,}")
    print(f"    Bounding box:")
    print(f"    {bbox_str(gauss_pos)}")
    gauss_center = gauss_pos.mean(axis=0)
    print(f"    Centroid: ({gauss_center[0]:.4f}, {gauss_center[1]:.4f}, {gauss_center[2]:.4f})")
    gauss_extent = np.linalg.norm(gauss_pos.max(axis=0) - gauss_pos.min(axis=0))
    print(f"    Diagonal extent: {gauss_extent:.4f}")
    
    # Opacity stats
    if 'opacity' in gauss_extra:
        raw_opacity = gauss_extra['opacity']
        activated_opacity = sigmoid(raw_opacity)
        print(f"\n    Opacity (raw):       min={raw_opacity.min():.4f}  max={raw_opacity.max():.4f}  mean={raw_opacity.mean():.4f}")
        print(f"    Opacity (sigmoid):   min={activated_opacity.min():.4f}  max={activated_opacity.max():.4f}  mean={activated_opacity.mean():.4f}")
        n_low_opacity = (activated_opacity < 0.01).sum()
        n_mid_opacity = ((activated_opacity >= 0.01) & (activated_opacity < 0.5)).sum()
        n_high_opacity = (activated_opacity >= 0.5).sum()
        print(f"    Opacity distribution: <0.01: {n_low_opacity:,}  [0.01,0.5): {n_mid_opacity:,}  >=0.5: {n_high_opacity:,}")
    
    # Scale stats
    if 'scale_0' in gauss_extra:
        scales = np.column_stack([
            np.exp(gauss_extra['scale_0']),
            np.exp(gauss_extra['scale_1']),
            np.exp(gauss_extra['scale_2']),
        ])
        print(f"\n    Gaussian scales (exp-activated):")
        print(f"      scale_0: min={scales[:,0].min():.6f}  max={scales[:,0].max():.6f}  mean={scales[:,0].mean():.6f}")
        print(f"      scale_1: min={scales[:,1].min():.6f}  max={scales[:,1].max():.6f}  mean={scales[:,1].mean():.6f}")
        print(f"      scale_2: min={scales[:,2].min():.6f}  max={scales[:,2].max():.6f}  mean={scales[:,2].mean():.6f}")
        max_scale = scales.max(axis=1)
        print(f"      max scale per Gaussian: mean={max_scale.mean():.6f}  max={max_scale.max():.6f}")
    
    # --- 3. Centroid offset ---
    centroid_offset = np.linalg.norm(gauss_center - mesh_center)
    print(f"\n  Centroid offset (Gaussians vs Mesh): {centroid_offset:.4f}")
    print(f"  Scale ratio (Gaussian extent / Mesh extent): {gauss_extent / mesh_extent:.4f}")
    
    # --- 4. Distance analysis: Gaussian -> nearest mesh vertex ---
    print(f"\n  Distance analysis: Gaussian -> nearest mesh vertex")
    tree_mesh = KDTree(mesh_verts)
    dists_to_mesh, idx_to_mesh = tree_mesh.query(gauss_pos)
    
    print(f"    min:    {dists_to_mesh.min():.6f}")
    print(f"    max:    {dists_to_mesh.max():.6f}")
    print(f"    mean:   {dists_to_mesh.mean():.6f}")
    print(f"    median: {np.median(dists_to_mesh):.6f}")
    print(f"    std:    {dists_to_mesh.std():.6f}")
    
    # Percentiles
    for p in [90, 95, 99, 99.5, 99.9]:
        print(f"    P{p}: {np.percentile(dists_to_mesh, p):.6f}")
    
    # Count Gaussians at various distance thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 18.0]
    print(f"\n    Gaussians beyond distance threshold:")
    for t in thresholds:
        n_beyond = (dists_to_mesh > t).sum()
        if n_beyond > 0:
            print(f"      > {t:8.3f}: {n_beyond:6,} ({100*n_beyond/len(gauss_pos):.2f}%)")
    
    # --- 5. Distance analysis: Mesh vertex -> nearest Gaussian ---
    print(f"\n  Distance analysis: Mesh vertex -> nearest Gaussian")
    tree_gauss = KDTree(gauss_pos)
    dists_to_gauss, _ = tree_gauss.query(mesh_verts)
    
    print(f"    min:    {dists_to_gauss.min():.6f}")
    print(f"    max:    {dists_to_gauss.max():.6f}")
    print(f"    mean:   {dists_to_gauss.mean():.6f}")
    print(f"    median: {np.median(dists_to_gauss):.6f}")
    
    # --- 6. Outlier Gaussians analysis ---
    # Use a data-driven threshold: e.g. 10x median distance
    median_dist = np.median(dists_to_mesh)
    outlier_threshold = max(10 * median_dist, 0.1)
    outlier_mask = dists_to_mesh > outlier_threshold
    n_outliers = outlier_mask.sum()
    
    print(f"\n  OUTLIER ANALYSIS (threshold = {outlier_threshold:.4f} = max(10*median, 0.1)):")
    print(f"    Outlier Gaussians: {n_outliers:,} / {len(gauss_pos):,} ({100*n_outliers/len(gauss_pos):.2f}%)")
    
    if n_outliers > 0 and n_outliers <= 50:
        print(f"\n    Outlier details:")
        outlier_indices = np.where(outlier_mask)[0]
        for i in outlier_indices[:20]:
            pos = gauss_pos[i]
            d = dists_to_mesh[i]
            nearest_mesh_pos = mesh_verts[idx_to_mesh[i]]
            print(f"      Gauss[{i:5d}]: pos=({pos[0]:8.4f},{pos[1]:8.4f},{pos[2]:8.4f})  "
                  f"dist={d:.4f}  nearest_mesh=({nearest_mesh_pos[0]:8.4f},{nearest_mesh_pos[1]:8.4f},{nearest_mesh_pos[2]:8.4f})")
            if 'opacity' in gauss_extra:
                print(f"                   opacity(sigmoid)={sigmoid(gauss_extra['opacity'][i]):.4f}")
            if 'scale_0' in gauss_extra:
                s = [np.exp(gauss_extra[f'scale_{j}'][i]) for j in range(3)]
                print(f"                   scales=({s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f})")
    elif n_outliers > 50:
        print(f"\n    Top 20 worst outlier details:")
        outlier_indices = np.where(outlier_mask)[0]
        worst_order = np.argsort(dists_to_mesh[outlier_indices])[::-1]
        for rank, oidx in enumerate(worst_order[:20]):
            i = outlier_indices[oidx]
            pos = gauss_pos[i]
            d = dists_to_mesh[i]
            nearest_mesh_pos = mesh_verts[idx_to_mesh[i]]
            line = (f"      #{rank+1} Gauss[{i:5d}]: pos=({pos[0]:8.4f},{pos[1]:8.4f},{pos[2]:8.4f})  "
                    f"dist={d:.4f}  nearest_mesh=({nearest_mesh_pos[0]:8.4f},{nearest_mesh_pos[1]:8.4f},{nearest_mesh_pos[2]:8.4f})")
            if 'opacity' in gauss_extra:
                line += f"  opacity={sigmoid(gauss_extra['opacity'][i]):.4f}"
            print(line)
    
    # --- 7. Check if there's a coordinate system issue ---
    print(f"\n  COORDINATE SYSTEM CHECK:")
    # Check if axes might be swapped
    for perm_label, perm in [("XYZ (identity)", [0,1,2]),
                              ("XZY", [0,2,1]),
                              ("YXZ", [1,0,2]),
                              ("YZX", [1,2,0]),
                              ("ZXY", [2,0,1]),
                              ("ZYX", [2,1,0])]:
        permuted = gauss_pos[:, perm]
        tree_tmp = KDTree(mesh_verts)
        d_tmp, _ = tree_tmp.query(permuted)
        print(f"    Gauss[{perm_label:15s}] -> mesh: mean_dist={d_tmp.mean():.4f}  median={np.median(d_tmp):.4f}")
    
    # Check if mesh needs scaling
    print(f"\n  SCALE CHECK (what if mesh is in different units?):")
    for scale_factor in [0.001, 0.01, 0.1, 10, 100, 1000]:
        scaled_mesh = mesh_verts * scale_factor
        tree_tmp = KDTree(scaled_mesh)
        d_tmp, _ = tree_tmp.query(gauss_pos)
        print(f"    mesh*{scale_factor:7.3f}: mean_dist={d_tmp.mean():.4f}  median={np.median(d_tmp):.4f}")
    
    # Check if it's a sign flip issue
    print(f"\n  SIGN FLIP CHECK:")
    for label, transform in [("negate X", [-1,1,1]),
                              ("negate Y", [1,-1,1]),
                              ("negate Z", [1,1,-1]),
                              ("negate XY", [-1,-1,1]),
                              ("negate XZ", [-1,1,-1]),
                              ("negate YZ", [1,-1,-1]),
                              ("negate XYZ", [-1,-1,-1])]:
        t = np.array(transform)
        flipped = gauss_pos * t[None, :]
        tree_tmp = KDTree(mesh_verts)
        d_tmp, _ = tree_tmp.query(flipped)
        if d_tmp.mean() < dists_to_mesh.mean() * 0.9:  # only print if better
            print(f"    Gauss[{label:12s}] -> mesh: mean_dist={d_tmp.mean():.4f}  median={np.median(d_tmp):.4f}  *** BETTER ***")
    
    # --- 8. Check the input.ply (the SfM point cloud) ---
    input_ply = pc_path.parent.parent.parent / "input.ply"
    if input_ply.exists():
        print(f"\n  INPUT POINT CLOUD (SfM): {input_ply.name}")
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(input_ply))
            vertex = plydata['vertex']
            ix = np.asarray(vertex['x'], dtype=np.float64)
            iy = np.asarray(vertex['y'], dtype=np.float64)
            iz = np.asarray(vertex['z'], dtype=np.float64)
            input_pos = np.column_stack([ix, iy, iz])
            print(f"    Points: {len(input_pos):,}")
            print(f"    Bounding box:")
            print(f"    {bbox_str(input_pos)}")
            input_center = input_pos.mean(axis=0)
            print(f"    Centroid: ({input_center[0]:.4f}, {input_center[1]:.4f}, {input_center[2]:.4f})")
            
            # Distance from input points to mesh
            d_input, _ = tree_mesh.query(input_pos)
            print(f"    Distance input -> mesh: mean={d_input.mean():.4f}  median={np.median(d_input):.4f}  max={d_input.max():.4f}")
        except Exception as e:
            print(f"    Error loading: {e}")
    
    # --- 9. Check cameras.json for scale info ---
    cameras_json = pc_path.parent.parent.parent / "cameras.json"
    if cameras_json.exists():
        import json
        with open(cameras_json) as f:
            cameras = json.load(f)
        if isinstance(cameras, list) and len(cameras) > 0:
            cam = cameras[0]
            print(f"\n  CAMERA INFO (first camera):")
            if 'position' in cam:
                pos = cam['position']
                print(f"    Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                cam_pos = np.array(pos)
                cam_to_mesh_center = np.linalg.norm(cam_pos - mesh_center)
                print(f"    Distance to mesh center: {cam_to_mesh_center:.4f}")
    
    # --- 10. Check for existing geodesic results ---
    geo_dir = pc_path.parent.parent.parent / "geodesic_distance"
    if geo_dir.exists():
        print(f"\n  EXISTING GEODESIC RESULTS:")
        for f in sorted(geo_dir.rglob("*.npz")):
            print(f"    {f.relative_to(geo_dir)}")
            try:
                data = np.load(str(f))
                print(f"      Keys: {list(data.keys())}")
                if 'closest_mesh_distances' in data:
                    cmd = data['closest_mesh_distances']
                    print(f"      closest_mesh_distances: min={cmd.min():.6f}  max={cmd.max():.6f}  mean={cmd.mean():.6f}")
                    n_far = (cmd > 1.0).sum()
                    print(f"      Gaussians with closest_mesh_dist > 1.0: {n_far}")
                if 'geodesic_distances' in data:
                    gd = data['geodesic_distances']
                    print(f"      geodesic_distances shape: {gd.shape}")
                    n_inf = np.isinf(gd).sum()
                    n_nan = np.isnan(gd).sum()
                    print(f"      inf: {n_inf}  nan: {n_nan}")
                    finite = gd[np.isfinite(gd)]
                    if len(finite) > 0:
                        print(f"      finite: min={finite.min():.4f}  max={finite.max():.4f}  mean={finite.mean():.4f}")
            except Exception as e:
                print(f"      Error: {e}")
    
    return dists_to_mesh


# ===========================================================================
#  MAIN
# ===========================================================================

if __name__ == "__main__":
    data_root = Path("/home/rotem.shezaf/RaDe-GS/TrainData/TOSCA/SyntheticColmapData/blue_texture")
    processed_root = Path("/home/rotem.shezaf/RaDe-GS/TrainData/TOSCA/processed")
    
    # Analyze a few shapes to see if the problem is universal or shape-specific
    shapes_to_check = ["cat0", "dog0", "horse0", "david0", "victoria0"]
    
    print("=" * 80)
    print("  TOSCA blue_texture DATA QUALITY DIAGNOSTIC")
    print("=" * 80)
    
    for shape in shapes_to_check:
        shape_dir = data_root / shape
        if shape_dir.exists():
            try:
                analyze_shape(shape, data_root, processed_root)
            except Exception as e:
                print(f"\n  ERROR analyzing {shape}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")
