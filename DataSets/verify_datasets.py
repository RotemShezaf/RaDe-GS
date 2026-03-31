#!/usr/bin/env python3
"""
Verify generated training patch datasets and report detailed statistics.

Reports per-file:
- Shape, dtype, NaN/Inf counts
- Target geodesic (p_u) distribution with full percentiles and outlier flags
- r1_min_val distribution and ratio to target
- Neighbor count stats (real vs padded)
- Neighbor Euclidean distance distribution (||xyz|| from center)
- Neighbor geodesic distance distribution (for real, non-masked neighbors)
- Geodesic / Euclidean ratio per neighbor — key outlier metric
- Neighbor-to-target relationship (|geo_nbr − target|, geo_nbr / target)
- Scale and opacity distributions (when present)
- Point feature (center) verification
- Old sentinel padding detection
- Mask constant coverage

Usage:
    python DataSets/verify_datasets.py [--dataset tosca|polynomial|both]
    python DataSets/verify_datasets.py --dataset polynomial --ring 3
    python DataSets/verify_datasets.py --verbose
    python DataSets/verify_datasets.py --skip-blue   # skip tosca_*_blue dirs
"""
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================================
# Constants
# ============================================================================
DATASETS_BASE = project_root / "TrainData" / "datasets" / "gaussian_patches"
MASK_CONSTANT = -10.0

# TOSCA per-animal dataset directories (without shape-number suffix)
TOSCA_ANIMALS = ["cat", "centaur", "david", "dog", "gorilla", "horse",
                 "michael", "victoria", "wolf"]

# Polynomial surface dataset directories
POLY_SURFACES = ["paraboloid_all", "saddle_all", "hyperbolic_paraboloid_all"]
POLY_SURFACES_ONE_SOURCE = [f"{s}_one_source" for s in POLY_SURFACES]
POLY_SURFACES_SCALE_OPACITY = [f"{s}_scale_opacity" for s in POLY_SURFACES]
POLY_SURFACES_SCALE_OPACITY_ONE_SOURCE = [f"{s}_scale_opacity_one_source" for s in POLY_SURFACES]

# ring_size_mapping (from configs)
RING_SIZE_MAPPING = {
    "tosca": {2: 64, 3: 192},
    "polynomial": {2: 48, 3: 128},
}

# Known attribute configurations and their per-neighbor feature widths.
# key = feat_per_nbr → column layout within each neighbor entry.
ATTRIBUTE_CONFIGS = {
    4: {"point_feat": 3, "attrs": ["xyz"],
        "neighbor_cols": {"xyz": (0, 3), "geodesic": (3, 4)},
        "point_cols": {"xyz": (0, 3)}},
    8: {"point_feat": 7, "attrs": ["xyz", "scale", "opacity"],
        "neighbor_cols": {"xyz": (0, 3), "scale": (3, 6), "opacity": (6, 7),
                          "geodesic": (7, 8)},
        "point_cols": {"xyz": (0, 3), "scale": (3, 6), "opacity": (6, 7)}},
}


# ============================================================================
# Helpers
# ============================================================================
def detect_outliers(arr, name, multiplier=5.0):
    """Flag values beyond mean + multiplier × std."""
    if len(arr) == 0:
        return []
    issues = []
    mean, std = np.mean(arr), np.std(arr)
    threshold = mean + multiplier * std
    n_extreme = int(np.sum(arr > threshold))
    if n_extreme > 0:
        pct = n_extreme / len(arr) * 100
        issues.append(
            f"{name}: {n_extreme} values ({pct:.4f}%) > mean+{multiplier}×std "
            f"(threshold={threshold:.4f}, max={np.max(arr):.4f})")
    return issues


def detect_fold_outliers_per_example(
    nbr_xyz: np.ndarray,
    nbr_geo: np.ndarray,
    is_masked: np.ndarray,
    target: float,
    sentinel: float = -10.0,
) -> dict:
    """
    Post-hoc fold-outlier detection mimicking ``_filter_outliers_without_pu``.

    For each training example, split real neighbors into a reference set
    (bottom quartile by geodesic — these are closest to the center and most
    likely on the same surface sheet) and the rest.  Then compute geodesic
    gradients from each non-reference neighbor to the reference set.  A
    fold-outlier will have a much higher minimum gradient than same-sheet
    neighbors.

    Returns a dict with:
      - ``n_suspect``: number of neighbors flagged as fold-likely.
      - ``n_real``: number of real (non-masked) neighbors.
      - ``suspect_max_grad``: max gradient of any suspect neighbor.
      - ``baseline_median``: median pairwise gradient in reference set.
      - ``threshold``: adaptive threshold used.
    """
    eps = 1e-8
    real = ~is_masked
    n_real = int(real.sum())
    result = {"n_real": n_real, "n_suspect": 0, "suspect_max_grad": 0.0,
              "baseline_median": 0.0, "threshold": 0.0}
    if n_real < 4:
        return result

    xyz_r = nbr_xyz[real]
    geo_r = nbr_geo[real]

    # Use bottom-quartile by geodesic as reference (like ring-1 in the filter)
    q25 = np.percentile(geo_r, 25)
    ref_mask = geo_r <= max(q25, eps)
    if ref_mask.sum() < 2:
        ref_idx = np.argsort(geo_r)[:max(2, n_real // 4)]
        ref_mask = np.zeros(n_real, dtype=bool)
        ref_mask[ref_idx] = True

    ref_xyz = xyz_r[ref_mask]
    ref_geo = geo_r[ref_mask]
    n_ref = len(ref_geo)

    # Baseline from reference pairwise
    ref_eucl = np.linalg.norm(
        ref_xyz[:, np.newaxis, :] - ref_xyz[np.newaxis, :, :], axis=2)
    ref_gdiff = np.abs(ref_geo[:, np.newaxis] - ref_geo[np.newaxis, :])
    ref_grad = ref_gdiff / np.maximum(ref_eucl, eps)
    triu = np.triu_indices(n_ref, k=1)
    if len(triu[0]) == 0:
        return result
    baseline = float(np.median(ref_grad[triu]))
    threshold = min(max(baseline * 5.0, 3.0), 50.0)

    # Check non-reference neighbors
    check_mask = real.copy()
    check_idx_in_real = np.where(~ref_mask)[0]
    if len(check_idx_in_real) == 0:
        result["baseline_median"] = baseline
        result["threshold"] = threshold
        return result

    check_xyz = xyz_r[check_idx_in_real]
    check_geo = geo_r[check_idx_in_real]

    dists = np.linalg.norm(
        check_xyz[:, np.newaxis, :] - ref_xyz[np.newaxis, :, :], axis=2)
    gdiffs = np.abs(check_geo[:, np.newaxis] - ref_geo[np.newaxis, :])
    grads = gdiffs / np.maximum(dists, eps)
    min_grads = grads.min(axis=1)

    n_suspect = int((min_grads > threshold).sum())
    result["n_suspect"] = n_suspect
    result["suspect_max_grad"] = float(min_grads.max()) if len(min_grads) > 0 else 0.0
    result["baseline_median"] = baseline
    result["threshold"] = threshold
    return result


# ============================================================================
# Core file analysis
# ============================================================================
def analyze_file(npy_file, file_ring, ring_size_map, dataset_type, verbose):
    """Analyse a single .npy file and return a rich stats dict."""
    info = {"file": npy_file.name, "warnings": [], "outlier_flags": []}

    # Expected example count from filename
    try:
        n_str = npy_file.stem.split("_n")[1]
        expected_n = int(n_str)
    except (IndexError, ValueError):
        expected_n = None

    try:
        data = np.load(str(npy_file))
    except Exception as e:
        info["error"] = str(e)
        return info

    n_examples, total_cols = data.shape
    info["shape"] = data.shape
    info["dtype"] = str(data.dtype)
    info["size_mb"] = data.nbytes / 1024 / 1024

    if expected_n is not None and n_examples != expected_n:
        info["warnings"].append(
            f"shape[0]={n_examples} != filename expected {expected_n}")

    # NaN / Inf
    info["nan_count"] = int(np.count_nonzero(np.isnan(data)))
    info["inf_count"] = int(np.count_nonzero(np.isinf(data)))

    # ---- Identify layout ----
    max_num_nbrs = None
    feat_per_nbr = None
    attr_config = None

    if file_ring is not None and file_ring in ring_size_map:
        max_num_nbrs = ring_size_map[file_ring]
        for fpn in sorted(ATTRIBUTE_CONFIGS.keys(), reverse=True):
            cfg = ATTRIBUTE_CONFIGS[fpn]
            pfw = cfg["point_feat"]
            if max_num_nbrs * fpn + pfw + 2 == total_cols:
                feat_per_nbr = fpn
                attr_config = cfg
                break
            if max_num_nbrs * fpn + pfw + 1 == total_cols:
                feat_per_nbr = fpn
                attr_config = cfg
                info["has_r1_min_val"] = False
                break

    if attr_config is None:
        info["warnings"].append("Could not identify attribute layout")
        targets = data[:, -1]
        info["target_stats"] = _target_stats(targets)
        info["mask_constant_count"] = int(np.sum(data == MASK_CONSTANT))
        info["mask_constant_fraction"] = float(info["mask_constant_count"]) / data.size
        return info

    has_r1 = info.get("has_r1_min_val", True)
    info["has_r1_min_val"] = has_r1
    info["feat_per_nbr"] = feat_per_nbr
    info["point_feat_width"] = attr_config["point_feat"]
    info["max_num_nbrs"] = max_num_nbrs
    info["attributes"] = attr_config["attrs"]

    nbr_block_width = max_num_nbrs * feat_per_nbr
    pfw = attr_config["point_feat"]
    ncols = attr_config["neighbor_cols"]
    xyz_s, xyz_e = ncols["xyz"]
    geo_s, geo_e = ncols["geodesic"]

    # ---- Extract blocks ----
    nbr_block = data[:, :nbr_block_width].reshape(n_examples, max_num_nbrs, feat_per_nbr)

    point_start = nbr_block_width
    point_end = point_start + pfw
    point_features = data[:, point_start:point_end]

    if has_r1:
        r1_min_vals = data[:, point_end]
        targets = data[:, point_end + 1]
    else:
        r1_min_vals = None
        targets = data[:, point_end]

    # ---- Mask analysis ----
    nbr_geo = nbr_block[:, :, geo_s:geo_e].squeeze(-1)
    is_masked = (nbr_geo == MASK_CONSTANT)
    real_per_example = max_num_nbrs - np.sum(is_masked, axis=1)
    padded_per_example = np.sum(is_masked, axis=1)

    info["real_nbrs_stats"] = {
        "mean": float(np.mean(real_per_example)),
        "std": float(np.std(real_per_example)),
        "min": int(np.min(real_per_example)),
        "p25": float(np.percentile(real_per_example, 25)),
        "p50": float(np.percentile(real_per_example, 50)),
        "p75": float(np.percentile(real_per_example, 75)),
        "p95": float(np.percentile(real_per_example, 95)),
        "p99": float(np.percentile(real_per_example, 99)),
        "max": int(np.max(real_per_example)),
    }
    info["padded_nbrs_stats"] = {
        "mean": float(np.mean(padded_per_example)),
        "pct_with_padding": float(np.mean(padded_per_example > 0) * 100),
        "pct_fully_padded": float(np.mean(real_per_example == 0) * 100),
    }

    # ---- Old sentinel padding ----
    nbr_xyz = nbr_block[:, :, xyz_s:xyz_e]
    xyz_is_sentinel = np.all(nbr_xyz == MASK_CONSTANT, axis=2)
    sentinel_count = int(np.sum(xyz_is_sentinel))
    info["sentinel_xyz_padding_count"] = sentinel_count
    if sentinel_count > 0:
        info["warnings"].append(
            f"OLD SENTINEL PADDING: {sentinel_count} entries with xyz=={MASK_CONSTANT}")

    # ---- Global mask constant ----
    info["mask_constant_count"] = int(np.sum(data == MASK_CONSTANT))
    info["mask_constant_fraction"] = float(info["mask_constant_count"]) / data.size

    # ================================================================
    # TARGET (p_u) — full distribution
    # ================================================================
    info["target_stats"] = _target_stats(targets)
    info["outlier_flags"].extend(detect_outliers(targets, "target"))

    # ================================================================
    # r1_min_val
    # ================================================================
    if r1_min_vals is not None:
        info["r1_min_stats"] = _basic_stats(r1_min_vals, "r1_min_val")
        valid_t = targets > 0
        if np.any(valid_t):
            ratio = r1_min_vals[valid_t] / targets[valid_t]
            info["r1_min_over_target"] = _basic_stats(ratio, "r1_min/target")

    # ================================================================
    # NEIGHBOR STATISTICS  (sample for speed)
    # ================================================================
    sample_n = min(n_examples, 200_000)
    if sample_n < n_examples:
        rng = np.random.RandomState(42)
        si = rng.choice(n_examples, sample_n, replace=False)
        s_nbr, s_mask, s_targets = nbr_block[si], is_masked[si], targets[si]
        info["sampled_n"] = sample_n
    else:
        s_nbr, s_mask, s_targets = nbr_block, is_masked, targets
        info["sampled_n"] = n_examples

    real_mask = ~s_mask
    ri = np.where(real_mask)

    if len(ri[0]) > 0:
        real_nbrs = s_nbr[ri]                         # (M, feat_per_nbr)
        real_targets_exp = s_targets[ri[0]]            # (M,)

        # ---- Euclidean distance ----
        nbr_xyz_r = real_nbrs[:, xyz_s:xyz_e]
        euc_dist = np.linalg.norm(nbr_xyz_r, axis=1)
        info["nbr_euclidean_dist"] = _full_dist_stats(euc_dist)
        info["outlier_flags"].extend(detect_outliers(euc_dist, "nbr_euc_dist"))

        # ---- Geodesic at neighbors ----
        nbr_geo_r = real_nbrs[:, geo_s]
        info["nbr_geodesic"] = _full_dist_stats(nbr_geo_r)
        info["outlier_flags"].extend(detect_outliers(nbr_geo_r, "nbr_geodesic"))
        info["nbr_beyond_target_pct"] = float(
            np.sum(nbr_geo_r > real_targets_exp) / len(nbr_geo_r) * 100)
        info["nbr_negative_geodesic_count"] = int(np.sum(nbr_geo_r < 0))

        # ---- Geo / Euc ratio ----
        eps = 1e-8
        geo_euc_ratio = nbr_geo_r / np.maximum(euc_dist, eps)
        info["geo_euc_ratio"] = _full_dist_stats(geo_euc_ratio)
        for thresh in [10, 50, 100, 500]:
            info[f"geo_euc_ratio_gt{thresh}"] = int(np.sum(geo_euc_ratio > thresh))
        info["outlier_flags"].extend(
            detect_outliers(geo_euc_ratio, "geo/euc_ratio", multiplier=5.0))

        # ---- Neighbor-to-target ----
        geo_diff = np.abs(nbr_geo_r - real_targets_exp)
        info["nbr_target_abs_diff"] = _full_dist_stats(geo_diff)
        valid_t2 = real_targets_exp > 0
        if np.any(valid_t2):
            info["nbr_geo_over_target"] = _full_dist_stats(
                nbr_geo_r[valid_t2] / real_targets_exp[valid_t2])

        # ---- Scale / Opacity per neighbor ----
        if "scale" in ncols:
            sc_s, sc_e = ncols["scale"]
            sv = real_nbrs[:, sc_s:sc_e]
            info["nbr_scale"] = {
                "per_dim": [_basic_stats(sv[:, d], f"scale_{d}")
                            for d in range(sv.shape[1])],
                "norm": _basic_stats(np.linalg.norm(sv, axis=1), "scale_norm"),
            }
        if "opacity" in ncols:
            op_s, op_e = ncols["opacity"]
            info["nbr_opacity"] = _basic_stats(
                real_nbrs[:, op_s:op_e].ravel(), "opacity")

    # ---- Padded neighbor Euclidean ----
    pi = np.where(s_mask)
    if len(pi[0]) > 0:
        pad_nbrs = s_nbr[pi]
        pad_euc = np.linalg.norm(pad_nbrs[:, xyz_s:xyz_e], axis=1)
        info["padded_euclidean_dist"] = _basic_stats(pad_euc, "padded_euc_dist")

    # ================================================================
    # FOLD-OUTLIER ANALYSIS (mirrors _filter_outliers_without_pu)
    # ================================================================
    fold_sample_n = min(info["sampled_n"], 10_000)
    if fold_sample_n < info["sampled_n"]:
        fold_rng = np.random.RandomState(99)
        fold_si = fold_rng.choice(info["sampled_n"], fold_sample_n, replace=False)
        fold_nbr = s_nbr[fold_si]
        fold_mask = s_mask[fold_si]
        fold_targets = s_targets[fold_si]
    else:
        fold_nbr = s_nbr
        fold_mask = s_mask
        fold_targets = s_targets

    fold_results = []
    for ex_i in range(len(fold_nbr)):
        fr = detect_fold_outliers_per_example(
            fold_nbr[ex_i, :, xyz_s:xyz_e],
            fold_nbr[ex_i, :, geo_s],
            fold_mask[ex_i],
            fold_targets[ex_i],
        )
        fold_results.append(fr)

    total_suspects = sum(r["n_suspect"] for r in fold_results)
    total_real_nbrs = sum(r["n_real"] for r in fold_results)
    examples_with_suspects = sum(1 for r in fold_results if r["n_suspect"] > 0)
    suspect_counts = [r["n_suspect"] for r in fold_results if r["n_suspect"] > 0]
    baselines = [r["baseline_median"] for r in fold_results if r["n_real"] >= 4]
    thresholds = [r["threshold"] for r in fold_results if r["n_real"] >= 4]

    info["fold_outlier_analysis"] = {
        "sampled_examples": len(fold_results),
        "total_suspect_neighbors": total_suspects,
        "total_real_neighbors_checked": total_real_nbrs,
        "suspect_pct": float(total_suspects / max(total_real_nbrs, 1) * 100),
        "examples_with_suspects": examples_with_suspects,
        "examples_with_suspects_pct": float(
            examples_with_suspects / max(len(fold_results), 1) * 100),
    }
    if suspect_counts:
        info["fold_outlier_analysis"]["suspect_per_example"] = {
            "mean": float(np.mean(suspect_counts)),
            "max": int(np.max(suspect_counts)),
            "p50": float(np.median(suspect_counts)),
            "p95": float(np.percentile(suspect_counts, 95)),
        }
    if baselines:
        info["fold_outlier_analysis"]["baseline_gradient"] = {
            "mean": float(np.mean(baselines)),
            "p50": float(np.median(baselines)),
            "p95": float(np.percentile(baselines, 95)),
        }
        info["fold_outlier_analysis"]["adaptive_threshold"] = {
            "mean": float(np.mean(thresholds)),
            "p50": float(np.median(thresholds)),
            "p95": float(np.percentile(thresholds, 95)),
        }
    if total_suspects > 0:
        max_grads = [r["suspect_max_grad"] for r in fold_results
                     if r["n_suspect"] > 0]
        info["fold_outlier_analysis"]["max_suspect_gradient"] = float(
            np.max(max_grads))
        info["outlier_flags"].append(
            f"fold_outliers: {total_suspects} suspect neighbors "
            f"({info['fold_outlier_analysis']['suspect_pct']:.4f}%) in "
            f"{examples_with_suspects} examples "
            f"({info['fold_outlier_analysis']['examples_with_suspects_pct']:.2f}%) "
            f"[sampled {len(fold_results)} examples]")

    # ================================================================
    # POINT FEATURES
    # ================================================================
    pf = {}
    for attr_name, (cs, ce) in attr_config["point_cols"].items():
        col_vals = point_features[:, cs:ce]
        if attr_name == "xyz":
            all_zero = bool(np.all(col_vals == 0))
            pf["xyz_all_zero"] = all_zero
            if not all_zero:
                pf["xyz_nonzero_count"] = int(np.count_nonzero(col_vals))
                info["warnings"].append(
                    f"Center xyz not all zero! {pf['xyz_nonzero_count']} nonzero")
        else:
            for d in range(col_vals.shape[1]):
                pf[f"{attr_name}_{d}"] = _basic_stats(
                    col_vals[:, d], f"center_{attr_name}_{d}")
    info["point_features"] = pf

    # ================================================================
    # PER-EXAMPLE aggregate Euclidean stats (full, not sampled)
    # ================================================================
    euc_all = np.linalg.norm(nbr_block[:, :, xyz_s:xyz_e], axis=2)
    euc_all_masked = np.where(is_masked, np.nan, euc_all)
    with np.errstate(all='ignore'):
        per_ex_mean = np.nanmean(euc_all_masked, axis=1)
        per_ex_max = np.nanmax(euc_all_masked, axis=1)
        per_ex_min = np.nanmin(euc_all_masked, axis=1)
    has_real = real_per_example > 0
    if np.any(has_real):
        info["per_example_euc"] = {
            "mean_of_means": float(np.nanmean(per_ex_mean[has_real])),
            "mean_of_maxs": float(np.nanmean(per_ex_max[has_real])),
            "max_of_maxs": float(np.nanmax(per_ex_max[has_real])),
            "mean_of_mins": float(np.nanmean(per_ex_min[has_real])),
            "min_of_mins": float(np.nanmin(per_ex_min[has_real])),
        }

    return info


# ============================================================================
# Stat helpers
# ============================================================================
def _target_stats(targets):
    return {
        "count": len(targets),
        "min": float(np.min(targets)),
        "p1": float(np.percentile(targets, 1)),
        "p5": float(np.percentile(targets, 5)),
        "p10": float(np.percentile(targets, 10)),
        "p25": float(np.percentile(targets, 25)),
        "p50": float(np.percentile(targets, 50)),
        "p75": float(np.percentile(targets, 75)),
        "p90": float(np.percentile(targets, 90)),
        "p95": float(np.percentile(targets, 95)),
        "p99": float(np.percentile(targets, 99)),
        "max": float(np.max(targets)),
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
        "pct_zero": float(np.mean(targets == 0) * 100),
        "pct_negative": float(np.mean(targets < 0) * 100),
        "pct_gt_100": float(np.mean(targets > 100) * 100),
    }


def _basic_stats(arr, name=""):
    if len(arr) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name, "count": len(arr),
        "min": float(np.min(arr)),  "max": float(np.max(arr)),
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _full_dist_stats(arr, name=""):
    if len(arr) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name, "count": len(arr),
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


# ============================================================================
# Per-dataset verification
# ============================================================================
def verify_single_dataset(dataset_dir, ring=None, dataset_type="tosca",
                          verbose=False):
    dataset_dir = Path(dataset_dir)
    result = {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "files": [], "errors": [], "warnings": [],
    }

    if not dataset_dir.exists():
        result["errors"].append(f"Directory does not exist: {dataset_dir}")
        return result

    npy_files = sorted(dataset_dir.glob("gaussian_examples_ring*_n*.npy"))
    if not npy_files:
        result["errors"].append("No .npy dataset files found")
        return result

    if ring is not None:
        npy_files = [f for f in npy_files if f"ring{ring}" in f.name]

    ring_size_map = RING_SIZE_MAPPING.get(dataset_type,
                                          RING_SIZE_MAPPING["tosca"])

    for npy_file in npy_files:
        try:
            ring_str = npy_file.stem.split("ring")[1].split("_")[0]
            file_ring = int(ring_str)
        except (IndexError, ValueError):
            file_ring = None
            result["warnings"].append(
                f"Could not parse ring from: {npy_file.name}")

        info = analyze_file(npy_file, file_ring, ring_size_map,
                            dataset_type, verbose)
        if "error" in info:
            result["errors"].append(
                f"Cannot load {npy_file.name}: {info['error']}")
            continue

        if info.get("nan_count", 0) > 0:
            result["errors"].append(
                f"{npy_file.name}: {info['nan_count']} NaN values")
        if info.get("inf_count", 0) > 0:
            result["errors"].append(
                f"{npy_file.name}: {info['inf_count']} Inf values")

        result["warnings"].extend(info.get("warnings", []))
        result["files"].append(info)

    if result["files"]:
        result["stats"] = {
            "total_examples": sum(f["shape"][0] for f in result["files"]
                                  if "shape" in f),
            "total_size_mb": round(
                sum(f.get("size_mb", 0) for f in result["files"]), 1),
            "num_files": len(result["files"]),
        }
    else:
        result["stats"] = {}

    return result


# ============================================================================
# Pretty-printing
# ============================================================================
def _print_dist(label, stats, indent=6):
    sp = " " * indent
    if stats.get("count", 0) == 0:
        print(f"{sp}{label}: (empty)"); return
    s = stats
    print(f"{sp}{label}:")
    print(f"{sp}  mean={s['mean']:.6f}  std={s['std']:.6f}  "
          f"min={s['min']:.6f}  max={s['max']:.6f}")
    pcts = [f"p{p}={s[f'p{p}']:.4f}" for p in [1,5,10,25,50,75,90,95,99]
            if f"p{p}" in s]
    if pcts:
        print(f"{sp}  {', '.join(pcts)}")


def _print_basic(label, stats, indent=6):
    sp = " " * indent
    if stats.get("count", 0) == 0:
        print(f"{sp}{label}: (empty)"); return
    s = stats
    print(f"{sp}{label}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
          f"[{s['min']:.4f} .. {s['max']:.4f}], "
          f"p50={s['p50']:.4f}, p95={s['p95']:.4f}, p99={s['p99']:.4f}")


def print_result(result, verbose=False):
    status = "OK" if not result["errors"] else "ERRORS"
    nw = len(result.get("warnings", []))
    ws = f" ({nw} warnings)" if nw > 0 else ""

    print(f"\n{'='*70}")
    print(f"  {result['name']}  [{status}]{ws}")
    print(f"{'='*70}")

    if not result["exists"]:
        print(f"  MISSING: {result['path']}"); return

    st = result.get("stats", {})
    if st:
        print(f"  Files: {st.get('num_files', 0)}")
        print(f"  Total examples: {st.get('total_examples', 0):,}")
        print(f"  Total size: {st.get('total_size_mb', 0):.1f} MB")

    for f in result["files"]:
        print(f"\n  --- {f['file']} ---")
        print(f"    Shape: {f.get('shape','?')}, dtype: {f.get('dtype','?')}, "
              f"size: {f.get('size_mb',0):.1f} MB")

        if f.get("nan_count", 0):
            print(f"    *** NaN: {f['nan_count']} ***")
        if f.get("inf_count", 0):
            print(f"    *** Inf: {f['inf_count']} ***")

        attrs = f.get("attributes", ["?"])
        print(f"    Attributes: {attrs}, feat_per_nbr={f.get('feat_per_nbr','?')}, "
              f"max_nbrs={f.get('max_num_nbrs','?')}")

        # ---- TARGET ----
        ts = f.get("target_stats", {})
        if ts:
            print(f"\n    TARGET (geodesic p_u):")
            print(f"      mean={ts['mean']:.4f}, std={ts['std']:.4f}, "
                  f"min={ts['min']:.4f}, max={ts['max']:.4f}")
            pcts = [f"p{p}={ts[f'p{p}']:.4f}"
                    for p in [1,5,10,25,50,75,90,95,99] if f"p{p}" in ts]
            print(f"      {', '.join(pcts)}")
            flags = []
            if ts.get("pct_zero", 0) > 0:
                flags.append(f"zero={ts['pct_zero']:.2f}%")
            if ts.get("pct_negative", 0) > 0:
                flags.append(f"negative={ts['pct_negative']:.2f}%")
            if ts.get("pct_gt_100", 0) > 0:
                flags.append(f">100={ts['pct_gt_100']:.3f}%")
            if flags:
                print(f"      Flags: {', '.join(flags)}")

        # ---- r1_min ----
        if "r1_min_stats" in f:
            _print_basic("r1_min_val", f["r1_min_stats"])
        if "r1_min_over_target" in f:
            _print_basic("r1_min / target", f["r1_min_over_target"])

        # ---- NEIGHBORS ----
        if "real_nbrs_stats" in f:
            rns = f["real_nbrs_stats"]
            pns = f["padded_nbrs_stats"]
            print(f"\n    NEIGHBOR COUNTS:")
            print(f"      Real: mean={rns['mean']:.1f}, "
                  f"min={rns['min']}, p25={rns['p25']:.0f}, p50={rns['p50']:.0f}, "
                  f"p75={rns['p75']:.0f}, p95={rns['p95']:.0f}, "
                  f"p99={rns['p99']:.0f}, max={rns['max']}")
            print(f"      Padded: mean={pns['mean']:.1f}, "
                  f"{pns['pct_with_padding']:.1f}% have padding, "
                  f"{pns.get('pct_fully_padded',0):.2f}% fully padded")

        if f.get("sentinel_xyz_padding_count", 0) > 0:
            print(f"      *** OLD SENTINEL PADDING: "
                  f"{f['sentinel_xyz_padding_count']} entries ***")

        # ---- Point features ----
        pf = f.get("point_features", {})
        if pf:
            xyz_ok = pf.get("xyz_all_zero")
            if xyz_ok is True:
                ch = "\u2713"
                print(f"\n    CENTER POINT: xyz all zeros {ch}")
            elif xyz_ok is False:
                print(f"\n    CENTER POINT: *** xyz NOT all zeros! "
                      f"{pf.get('xyz_nonzero_count','?')} nonzero ***")
            for key in sorted(pf.keys()):
                if key.startswith(("scale_", "opacity_")):
                    _print_basic(f"center {key}", pf[key], indent=6)

        # ---- Euclidean distance ----
        if "nbr_euclidean_dist" in f:
            print(f"\n    NEIGHBOR EUCLIDEAN DISTANCE (||xyz|| from center):")
            _print_dist("distribution", f["nbr_euclidean_dist"])
            pe = f.get("per_example_euc", {})
            if pe:
                print(f"      Per-example: "
                      f"avg_mean={pe['mean_of_means']:.4f}, "
                      f"avg_max={pe['mean_of_maxs']:.4f}, "
                      f"global_max={pe['max_of_maxs']:.4f}, "
                      f"avg_min={pe['mean_of_mins']:.4f}, "
                      f"global_min={pe['min_of_mins']:.6f}")

        # ---- Geodesic at neighbors ----
        if "nbr_geodesic" in f:
            print(f"\n    NEIGHBOR GEODESIC DISTANCE:")
            _print_dist("distribution", f["nbr_geodesic"])
            print(f"      Beyond-target (geo>target): "
                  f"{f.get('nbr_beyond_target_pct', 0):.2f}%")
            neg = f.get("nbr_negative_geodesic_count", 0)
            if neg > 0:
                print(f"      *** {neg} with NEGATIVE geodesic ***")

        # ---- Geo / Euc ratio ----
        if "geo_euc_ratio" in f:
            print(f"\n    GEODESIC / EUCLIDEAN RATIO:")
            _print_dist("distribution", f["geo_euc_ratio"])
            extremes = []
            for t in [10, 50, 100, 500]:
                n = f.get(f"geo_euc_ratio_gt{t}", 0)
                if n > 0:
                    extremes.append(f">{t}: {n}")
            if extremes:
                print(f"      Extreme counts: {', '.join(extremes)}")

        # ---- Neighbor vs target ----
        if "nbr_target_abs_diff" in f:
            print(f"\n    NEIGHBOR vs TARGET:")
            _print_dist("|geo_nbr - target|", f["nbr_target_abs_diff"])
        if "nbr_geo_over_target" in f:
            _print_dist("geo_nbr / target", f["nbr_geo_over_target"])

        # ---- Scale / Opacity ----
        if "nbr_scale" in f:
            print(f"\n    NEIGHBOR SCALE:")
            for ds in f["nbr_scale"]["per_dim"]:
                _print_basic(ds["name"], ds)
            _print_basic("scale_norm", f["nbr_scale"]["norm"])
        if "nbr_opacity" in f:
            print(f"\n    NEIGHBOR OPACITY:")
            _print_basic("opacity", f["nbr_opacity"])

        # ---- Padded Euclidean ----
        if "padded_euclidean_dist" in f:
            print(f"\n    PADDED NEIGHBOR EUCLIDEAN (should match real dist):")
            _print_basic("distribution", f["padded_euclidean_dist"])

        # ---- Fold-outlier analysis ----
        foa = f.get("fold_outlier_analysis")
        if foa:
            print(f"\n    FOLD-OUTLIER ANALYSIS "
                  f"(mimics _filter_outliers_without_pu, "
                  f"{foa['sampled_examples']:,} examples):")
            print(f"      Suspect neighbors: "
                  f"{foa['total_suspect_neighbors']:,} / "
                  f"{foa['total_real_neighbors_checked']:,} "
                  f"({foa['suspect_pct']:.4f}%)")
            print(f"      Examples with suspects: "
                  f"{foa['examples_with_suspects']:,} "
                  f"({foa['examples_with_suspects_pct']:.2f}%)")
            if "suspect_per_example" in foa:
                spe = foa["suspect_per_example"]
                print(f"      Suspects per affected example: "
                      f"mean={spe['mean']:.1f}, max={spe['max']}, "
                      f"p50={spe['p50']:.0f}, p95={spe['p95']:.0f}")
            if "baseline_gradient" in foa:
                bg = foa["baseline_gradient"]
                print(f"      Baseline gradient: "
                      f"mean={bg['mean']:.4f}, p50={bg['p50']:.4f}, "
                      f"p95={bg['p95']:.4f}")
            if "adaptive_threshold" in foa:
                at = foa["adaptive_threshold"]
                print(f"      Adaptive threshold: "
                      f"mean={at['mean']:.4f}, p50={at['p50']:.4f}, "
                      f"p95={at['p95']:.4f}")
            if "max_suspect_gradient" in foa:
                print(f"      Max suspect gradient: "
                      f"{foa['max_suspect_gradient']:.4f}")

        # ---- Mask constant ----
        print(f"\n    MASK CONSTANT ({MASK_CONSTANT}): "
              f"{f.get('mask_constant_count',0):,} values "
              f"({f.get('mask_constant_fraction',0)*100:.3f}%)")

        # ---- Outlier flags ----
        oflags = f.get("outlier_flags", [])
        if oflags:
            print(f"\n    *** OUTLIER FLAGS ***")
            for flag in oflags:
                print(f"      ! {flag}")

        if f.get("sampled_n") and f["sampled_n"] < f["shape"][0]:
            print(f"\n    (neighbor stats from {f['sampled_n']:,} / "
                  f"{f['shape'][0]:,} sampled examples)")

    if result["errors"]:
        print(f"\n  ERRORS:")
        for e in result["errors"]:
            print(f"    * {e}")
    if result["warnings"] and verbose:
        print(f"\n  WARNINGS:")
        for w in result["warnings"]:
            print(f"    * {w}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Verify generated datasets")
    parser.add_argument("--dataset", default="both",
                        choices=["tosca", "polynomial", "both"])
    parser.add_argument("--ring", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--variant", default="one_source",
                        choices=["all", "one_source", "scale_opacity",
                                 "scale_opacity_one_source"])
    parser.add_argument("--skip-blue", action="store_true",
                        help="Skip tosca_*_blue per-shape dirs")
    args = parser.parse_args()

    all_results = []

    print(f"\n{'#'*70}")
    print(f"#  Dataset Verification Report (Detailed)")
    print(f"#  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Base: {DATASETS_BASE}")
    print(f"{'#'*70}")

    # ---- TOSCA ----
    if args.dataset in ("tosca", "both"):
        print(f"\n{'#'*70}")
        print(f"#  TOSCA Datasets")
        print(f"{'#'*70}")
        for animal in TOSCA_ANIMALS:
            r = verify_single_dataset(DATASETS_BASE / f"tosca_{animal}",
                                      ring=args.ring, dataset_type="tosca",
                                      verbose=args.verbose)
            all_results.append(r)
            print_result(r, verbose=args.verbose)

        if not args.skip_blue:
            for d in sorted(DATASETS_BASE.glob("tosca_*_blue")):
                r = verify_single_dataset(d, ring=args.ring,
                                          dataset_type="tosca",
                                          verbose=args.verbose)
                all_results.append(r)
                print_result(r, verbose=args.verbose)

    # ---- Polynomial ----
    if args.dataset in ("polynomial", "both"):
        print(f"\n{'#'*70}")
        print(f"#  Polynomial Datasets")
        print(f"{'#'*70}")
        surfs = {"one_source": POLY_SURFACES_ONE_SOURCE,
                 "scale_opacity": POLY_SURFACES_SCALE_OPACITY,
                 "scale_opacity_one_source": POLY_SURFACES_SCALE_OPACITY_ONE_SOURCE,
                 "all": POLY_SURFACES}[args.variant]
        for s in surfs:
            r = verify_single_dataset(DATASETS_BASE / s, ring=args.ring,
                                      dataset_type="polynomial",
                                      verbose=args.verbose)
            all_results.append(r)
            print_result(r, verbose=args.verbose)

    # ---- Summary ----
    print(f"\n{'#'*70}")
    print(f"#  SUMMARY")
    print(f"{'#'*70}")

    total_files = sum(r["stats"].get("num_files", 0)
                      for r in all_results if r["exists"])
    total_examples = sum(r["stats"].get("total_examples", 0)
                         for r in all_results if r["exists"])
    total_size = sum(r["stats"].get("total_size_mb", 0)
                     for r in all_results if r["exists"])
    errors = sum(len(r["errors"]) for r in all_results)
    warnings = sum(len(r.get("warnings", [])) for r in all_results)
    missing = sum(1 for r in all_results if not r["exists"])

    all_oflags = []
    for r in all_results:
        for fi in r["files"]:
            for flag in fi.get("outlier_flags", []):
                all_oflags.append(f"{r['name']}/{fi['file']}: {flag}")

    print(f"\n  Datasets checked:  {len(all_results)}")
    print(f"  Missing:           {missing}")
    print(f"  Total .npy files:  {total_files}")
    print(f"  Total examples:    {total_examples:,}")
    print(f"  Total size:        {total_size:.1f} MB")
    print(f"  Errors:            {errors}")
    print(f"  Warnings:          {warnings}")
    print(f"  Outlier flags:     {len(all_oflags)}")

    if all_oflags:
        print(f"\n  --- Outlier Flags ---")
        for flag in all_oflags:
            print(f"  ! {flag}")

    if errors:
        print(f"\n  *** {errors} ERRORS FOUND ***")
    elif all_oflags:
        print(f"\n  All datasets OK with {len(all_oflags)} outlier flag(s).")
    else:
        print(f"\n  All datasets verified OK — no outliers detected.")

    print()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
