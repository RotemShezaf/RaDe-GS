#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug evaluation: run Fast Marching propagation on the FULL scene (no FPS)
and stop after the first N points are visited.  Compare predicted distances
against ground-truth geodesic distances.

This script is designed for debugging the propagation algorithm by inspecting
the first few thousand visited points in detail, without FPS downsampling
artifacts.

Usage (from project root, geo_splat env):
    python geodesic_propagation/run_eval_propagation.py --max_visited 2000
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Project root on sys.path ────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
os.chdir(PROJECT_ROOT)

from GenerateData.utils.load_utils import load_gaussian_data_cpu
from geodesic_propagation.input_builder import GaussianInputBuilder
from geodesic_propagation.utils.model_handler import ModelHandler
from geodesic_propagation.fast_marching import FastMarchingPropagator
from geodesic_propagation.utils.output_path import build_eval_output_dir


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def p(msg=""):
    print(msg, flush=True)


def banner(title: str):
    p(f"\n{'=' * 74}")
    p(f"  {title}")
    p("=" * 74)


def stat(label, arr, indent=4):
    prefix = " " * indent
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr).flatten()
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        p(f"{prefix}{label}: ALL non-finite ({len(arr)} elts)")
        return
    p(f"{prefix}{label}: "
      f"min={finite.min():.6f}  max={finite.max():.6f}  "
      f"mean={finite.mean():.6f}  median={np.median(finite):.6f}  "
      f"std={finite.std():.6f}  [{len(finite)}/{len(arr)}]")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug: run FM propagation on full scene, stop after N visited",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- Paths -----
    parser.add_argument(
        "--model_path", type=str,
        default=None,
        help="Path to model checkpoint (.pth). If omitted, resolved from train_config.",
    )
    parser.add_argument(
        "--train_config", type=str,
        default="models/configs/one_source/combined_polynomial_ring3.yaml",
        help="Path to the training config YAML",
    )
    parser.add_argument(
        "--dataset_config", type=str,
        default=None,
        help="Path to the dataset config YAML. If omitted, resolved from train_config.",
    )
    parser.add_argument(
        "--gaussian_dir", type=str,
        default="TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_04/light_4/output",
        help="Path to the Gaussian splat output directory",
    )
    parser.add_argument(
        "--gt_subpath", type=str,
        default="geodesic_distance/gt_geodesic.npz",
        help="Relative path (inside gaussian_dir) to the GT geodesic .npz",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="geodesic_propagation/eval_propagation_debug",
        help="Directory for CSV and plot output",
    )

    # ----- Neighborhood -----
    parser.add_argument("--ring", type=int, default=3, help="Ring level for neighborhoods")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Ring-1 neighbor count (kNN k)")
    parser.add_argument("--use_mahalanobis", action="store_true", help="Use Mahalanobis distance for kNN")

    # ----- Propagation -----
    parser.add_argument("--max_visited", type=int, default=2000,
                        help="Stop propagation after this many points are visited")
    parser.add_argument("--source_row", type=int, default=1,
                        help="Which GT source row to use")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for FM batch propagation")
    parser.add_argument("--refine_passes", type=int, default=0,
                        help="Number of post-FM iterative refinement passes (0 = disabled)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detect if omitted.")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="Number of bins for distance-based error analysis")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gaussian_dir = Path(args.gaussian_dir)
    gt_path = gaussian_dir / args.gt_subpath
    train_config_path = Path(args.train_config)

    # ── Auto-resolve model_path and dataset_config from train_config ──
    import yaml as _yaml
    with open(train_config_path) as _f:
        _train_cfg_raw = _yaml.safe_load(_f)

    if args.model_path is None:
        save_dir = _train_cfg_raw.get("infrastructure", {}).get("save_dir", "")
        args.model_path = str(Path(save_dir) / "best_model.pth")
        p(f"    [auto-resolve] model_path = {args.model_path}")
    if args.dataset_config is None:
        args.dataset_config = _train_cfg_raw.get("dataset", {}).get("dataset_config", "")
        p(f"    [auto-resolve] dataset_config = {args.dataset_config}")

    # ── Build structured output directory from model + gaussian paths ──
    output_dir = build_eval_output_dir(
        args.output_dir, args.model_path, args.gaussian_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    dataset_config_path = Path(args.dataset_config)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── 1. Config echo ─────────────────────────────────────────────
    banner("1. Configuration")
    for k, v in vars(args).items():
        p(f"    {k:20s} = {v}")
    p(f"    {'device':20s} = {device}")

    # ── 2. Load Gaussian data ──────────────────────────────────────
    banner("2. Load Gaussian data")
    t0 = time.perf_counter()
    gdata = load_gaussian_data_cpu(gaussian_dir, iteration=None, load_sh_features=False)
    positions = gdata.get_xyz()
    scales = gdata.get_scaling()
    rotations = gdata.get_rotation()
    opacities = gdata.get_opacity()
    N = len(positions)
    t_load = time.perf_counter() - t0
    p(f"    {N:,} Gaussians loaded [{t_load:.2f}s]")
    stat("positions", positions)

    # ── 3. Load GT geodesic ────────────────────────────────────────
    banner("3. Load GT geodesic distances")
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]
    gt_source_indices = gt_data["source_gaussian_indices"]
    p(f"    {len(gt_source_indices)} GT sources available")

    gt_row = args.source_row
    source_idx = int(gt_source_indices[gt_row])
    gt_dists = gt_all_dists[gt_row].astype(np.float64)
    p(f"    GT source row={gt_row}, global idx={source_idx}")
    stat("GT geodesic", gt_dists)

    # ── 4. Load model ──────────────────────────────────────────────
    banner("4. Load model")
    train_cfg = _train_cfg_raw

    model_handler = ModelHandler(model_path=str(model_path), device=str(device))
    transforms_cfg = train_cfg.get("dataset", {}).get("transforms", None)
    p(f"    Model loaded on {device}")

    # ── 5. Build input builder + propagator ────────────────────────
    banner("5. Build GaussianInputBuilder + FastMarchingPropagator")
    t0 = time.perf_counter()

    config = _yaml.safe_load(open(dataset_config_path))
    args.use_mahalanobis = config.get("use_mahalanobis", args.use_mahalanobis or False)

    ib = GaussianInputBuilder(
        positions=positions,
        dataset_config=str(dataset_config_path),
        ring=args.ring,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        device=str(device),
        n_neighbors=args.n_neighbors,
        use_mahalanobis=args.use_mahalanobis,
        transforms_config=transforms_cfg,
    )

    ring1_nbrs = ib.ring1_neighbors
    ring_neighbors = ib.ring_neighbors

    p(f"    mean_nn_dist   = {ib.normalization_factor:.8f}")
    p(f"    attributes     = {ib.attributes}")
    p(f"    max_neighbors  = {ib.max_neighbors}")
    p(f"    mask_constant  = {ib.mask_constant}")
    p(f"    nn_mean        = {ib.nn_mean}")

    sizes = np.array([len(ring_neighbors[i]) for i in range(N)])
    stat(f"ring-{args.ring} neighborhood sizes", sizes)

    propagator = FastMarchingPropagator(
        model_handler=model_handler,
        input_builder=ib,
        ring1_neighbors=ring1_nbrs,
        ring_neighbors=ring_neighbors,
        ring=args.ring,
        verbose=True,
    )
    t_build = time.perf_counter() - t0
    p(f"    Propagator built [{t_build:.2f}s]")

    # ── 6. Run FM propagation (stop at max_visited) ────────────────
    banner(f"6. Fast Marching propagation (max_visited={args.max_visited}, refine_passes={args.refine_passes})")
    t0 = time.perf_counter()
    distances = propagator.propagate_batch(
        [source_idx],
        batch_size=args.batch_size,
        max_visited=args.max_visited,
        refine_passes=args.refine_passes,
    )
    t_prop = time.perf_counter() - t0

    visited_mask = np.isfinite(distances)
    visited_count = int(visited_mask.sum())
    finite = distances[visited_mask]
    p(f"    Visited: {visited_count:,}/{N:,} ({100 * visited_count / N:.1f}%)")
    if len(finite) > 0:
        p(f"    Predicted distance range: [{finite.min():.6f}, {finite.max():.6f}]")
    p(f"    Propagation time: {t_prop:.2f}s")

    mf = propagator.get_model_floor_stats()
    p(f"    model_wins={mf['model_wins']}, floor_wins={mf['floor_wins']}, "
      f"euclidean_fallback={mf['euclidean_fallback']}")

    # ── 7. Normalization diagnostic ──────────────────────────────────
    banner("7. Normalization diagnostic: FM-input vs GT-input for sample points")

    # Pick 5 sample points spread across the GT range
    valid_diag = visited_mask & np.isfinite(gt_dists) & (gt_dists > 0)
    valid_diag[source_idx] = False
    valid_diag_idx = np.where(valid_diag)[0]
    sorted_idx = valid_diag_idx[np.argsort(gt_dists[valid_diag_idx])]
    sample_positions = [0, len(sorted_idx)//4, len(sorted_idx)//2,
                        3*len(sorted_idx)//4, len(sorted_idx)-1]
    sample_pids = [int(sorted_idx[sp]) for sp in sample_positions]

    for pid in sample_pids:
        p(f"\n  --- Point {pid} (GT={gt_dists[pid]:.6f}, FM={distances[pid]:.6f}) ---")
        nbrs = ring_neighbors.get(pid, np.array([], dtype=np.int64))

        # ---- Build input using FM distances (as propagation would) ----
        fm_vis = propagator.visited_mask.copy()
        fm_dists_arr = distances.copy()
        fm_visited_nbrs = nbrs[fm_vis[nbrs]]
        fm_visited_dists = fm_dists_arr[fm_visited_nbrs]
        p(f"    FM input: {len(fm_visited_nbrs)}/{len(nbrs)} visited neighbors")
        if len(fm_visited_dists) > 0:
            p(f"    FM visited dists: min={fm_visited_dists.min():.8f}, "
              f"max={fm_visited_dists.max():.8f}, mean={fm_visited_dists.mean():.8f}")

        result_fm = ib.build_input(
            point_idx=pid,
            all_neighbor_indices=nbrs,
            neighbor_distances=fm_dists_arr,
            ring1_neighbor_indices=ring1_nbrs.get(pid, None),
            visited_mask=fm_vis,
        )
        if result_fm is not None:
            nb_fm, pf_fm, vm_fm, bi_fm = result_fm
            p(f"    FM build_info: min_input={bi_fm['min_input']:.8f}, "
              f"cur_norm={bi_fm['current_normalization']:.8f}, "
              f"max_dist={bi_fm['max_dist']:.8f}")
            p(f"    FM valid entries: {vm_fm.sum().item()}/{nb_fm.shape[0]}")
            # Show geodesic column stats for valid entries
            geo_col = nb_fm[:, -1].cpu().numpy()
            valid_geo = geo_col[vm_fm.cpu().numpy().astype(bool)]
            if len(valid_geo) > 0:
                p(f"    FM geo feature (valid): min={valid_geo.min():.4f}, "
                  f"max={valid_geo.max():.4f}, mean={valid_geo.mean():.4f}")
            # Model prediction
            with torch.no_grad():
                pred_fm = model_handler.predict(
                    nb_fm.unsqueeze(0), pf_fm.unsqueeze(0), vm_fm.unsqueeze(0))
            raw_fm = pred_fm[0, 0].item()
            denorm_fm = ib.denormalize_result(raw_fm, bi_fm)
            p(f"    FM raw_pred={raw_fm:.6f}, denorm={denorm_fm:.8f}")

        # ---- Build input using GT distances (as evaluate_model_vs_gt would) ----
        gt_vis = gt_dists <= gt_dists[pid]
        gt_visited_nbrs = nbrs[gt_vis[nbrs]]
        gt_visited_dists = gt_dists[gt_visited_nbrs]
        p(f"    GT input: {len(gt_visited_nbrs)}/{len(nbrs)} visited neighbors")
        if len(gt_visited_dists) > 0:
            p(f"    GT visited dists: min={gt_visited_dists.min():.8f}, "
              f"max={gt_visited_dists.max():.8f}, mean={gt_visited_dists.mean():.8f}")

        result_gt = ib.build_input(
            point_idx=pid,
            all_neighbor_indices=nbrs,
            neighbor_distances=gt_dists,
            ring1_neighbor_indices=ring1_nbrs.get(pid, None),
            visited_mask=gt_vis,
        )
        if result_gt is not None:
            nb_gt, pf_gt, vm_gt, bi_gt = result_gt
            p(f"    GT build_info: min_input={bi_gt['min_input']:.8f}, "
              f"cur_norm={bi_gt['current_normalization']:.8f}, "
              f"max_dist={bi_gt['max_dist']:.8f}")
            p(f"    GT valid entries: {vm_gt.sum().item()}/{nb_gt.shape[0]}")
            geo_col_gt = nb_gt[:, -1].cpu().numpy()
            valid_geo_gt = geo_col_gt[vm_gt.cpu().numpy().astype(bool)]
            if len(valid_geo_gt) > 0:
                p(f"    GT geo feature (valid): min={valid_geo_gt.min():.4f}, "
                  f"max={valid_geo_gt.max():.4f}, mean={valid_geo_gt.mean():.4f}")
            with torch.no_grad():
                pred_gt = model_handler.predict(
                    nb_gt.unsqueeze(0), pf_gt.unsqueeze(0), vm_gt.unsqueeze(0))
            raw_gt = pred_gt[0, 0].item()
            denorm_gt = ib.denormalize_result(raw_gt, bi_gt)
            p(f"    GT raw_pred={raw_gt:.6f}, denorm={denorm_gt:.8f}")
            p(f"    GT target (actual GT dist)={gt_dists[pid]:.8f}")

    # ── 8. Compare with GT ─────────────────────────────────────────
    banner("8. Compare with GT (continued)")
    valid = visited_mask & np.isfinite(gt_dists) & (gt_dists > 0)
    valid[source_idx] = False  # exclude source
    valid_idx = np.where(valid)[0]

    if len(valid_idx) == 0:
        p("    No valid comparison points!")
        return

    pred_v = distances[valid_idx].astype(np.float64)
    gt_v = gt_dists[valid_idx]
    abs_err = np.abs(pred_v - gt_v)
    rel_err = abs_err / np.maximum(gt_v, 1e-10)
    signed_err = pred_v - gt_v

    p(f"    {len(valid_idx)} valid comparison points")

    # ── 8. Per-point diagnostics: visited neighbors ────────────────
    banner("9. Visited-neighbor diagnostics for evaluated points")

    n_visited_nbrs_arr = np.zeros(len(valid_idx), dtype=np.int32)
    n_ring_nbrs_arr = np.zeros(len(valid_idx), dtype=np.int32)
    for i, pid in enumerate(valid_idx):
        nbrs = ring_neighbors.get(pid, np.array([], dtype=np.int64))
        n_ring_nbrs_arr[i] = len(nbrs)
        if len(nbrs) > 0:
            n_visited_nbrs_arr[i] = int(propagator.visited_mask[nbrs].sum())

    stat("visited neighbors per point", n_visited_nbrs_arr)
    stat("total ring-k neighbors per point", n_ring_nbrs_arr)
    frac = n_visited_nbrs_arr / np.maximum(n_ring_nbrs_arr, 1)
    stat("fraction visited", frac)

    # ── 9. Save CSV ───────────────────────────────────────────────
    banner("10. Save CSV")
    csv_path = output_dir / "propagation_debug.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_idx", "gt_dist", "pred_dist", "abs_err", "rel_err",
            "signed_err", "n_visited_nbrs", "n_ring_nbrs",
        ])
        for i, pid in enumerate(valid_idx):
            writer.writerow([
                pid, gt_v[i], pred_v[i], abs_err[i], rel_err[i],
                signed_err[i], n_visited_nbrs_arr[i], n_ring_nbrs_arr[i],
            ])
    p(f"    Saved {len(valid_idx)} rows -> {csv_path}")

    # ── 10. Binned by GT distance ─────────────────────────────────
    banner("11. Accuracy binned by GT distance")
    bin_edges = np.linspace(gt_v.min(), gt_v.max() + 1e-10, args.n_bins + 1)
    hdr = (f"{'bin_range':>24s}  {'count':>6s}  {'MAE':>10s}  {'medAE':>10s}  "
           f"{'meanRE':>10s}  {'mean_pred':>10s}  {'mean_GT':>10s}  "
           f"{'bias':>10s}  {'avg_vis_nbrs':>12s}")
    p(f"  {hdr}")
    p(f"  {'-' * len(hdr)}")

    for b in range(args.n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (gt_v >= lo) & (gt_v < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mae_b = abs_err[mask].mean()
        med_ae = np.median(abs_err[mask])
        mean_re = rel_err[mask].mean()
        mean_pred = pred_v[mask].mean()
        mean_gt = gt_v[mask].mean()
        bias = signed_err[mask].mean()
        avg_vis = n_visited_nbrs_arr[mask].mean()
        p(f"  [{lo:10.4f}, {hi:10.4f})  {cnt:>6}  {mae_b:>10.6f}  {med_ae:>10.6f}  "
          f"{mean_re:>9.4%}  {mean_pred:>10.6f}  {mean_gt:>10.6f}  "
          f"{bias:>+10.6f}  {avg_vis:>12.1f}")

    # ── 11. Binned by #visited neighbors ───────────────────────────
    banner("12. Accuracy binned by number of visited neighbors")
    nbr_bins = sorted(set(n_visited_nbrs_arr))
    hdr2 = (f"{'n_vis_nbrs':>10s}  {'count':>6s}  {'MAE':>10s}  {'meanRE':>10s}  "
            f"{'mean_pred':>10s}  {'mean_GT':>10s}  {'bias':>10s}")
    p(f"  {hdr2}")
    p(f"  {'-' * len(hdr2)}")

    for nv in nbr_bins:
        mask = n_visited_nbrs_arr == nv
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mae_b = abs_err[mask].mean()
        mean_re = rel_err[mask].mean()
        mean_pred = pred_v[mask].mean()
        mean_gt = gt_v[mask].mean()
        bias = signed_err[mask].mean()
        p(f"  {int(nv):>10}  {cnt:>6}  {mae_b:>10.6f}  {mean_re:>9.4%}  "
          f"{mean_pred:>10.6f}  {mean_gt:>10.6f}  {bias:>+10.6f}")

    # ── 12. Overall summary ───────────────────────────────────────
    banner("13. Overall summary")
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
    corr = float(np.corrcoef(gt_v, pred_v)[0, 1]) if len(gt_v) > 2 else float("nan")
    ss_res = np.sum((pred_v - gt_v) ** 2)
    ss_tot = np.sum((gt_v - gt_v.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    p(f"    Points evaluated   = {len(valid_idx)}")
    p(f"    MAE                = {mae:.8f}")
    p(f"    RMSE               = {rmse:.8f}")
    p(f"    Median AE          = {np.median(abs_err):.8f}")
    p(f"    Mean relative err  = {rel_err.mean():.4%}")
    p(f"    Max  absolute err  = {abs_err.max():.8f}")
    p(f"    P90  absolute err  = {np.percentile(abs_err, 90):.8f}")
    p(f"    P95  absolute err  = {np.percentile(abs_err, 95):.8f}")
    p(f"    P99  absolute err  = {np.percentile(abs_err, 99):.8f}")
    p(f"    Mean bias (pred-GT)= {signed_err.mean():+.8f}")
    p(f"    Mean pred          = {pred_v.mean():.8f}")
    p(f"    Mean GT            = {gt_v.mean():.8f}")
    p(f"    Pearson corr       = {corr:.6f}")
    p(f"    R² score           = {r2:.6f}")
    n_zero = int(np.sum(pred_v == 0.0))
    p(f"    Preds == 0         = {n_zero}/{len(valid_idx)}")
    n_neg = int(np.sum(pred_v < 0.0))
    p(f"    Preds < 0          = {n_neg}")

    # ── 13. GT distance range for visited points ───────────────────
    banner("14. GT distance coverage")
    gt_visited = gt_dists[visited_mask & (gt_dists > 0)]
    gt_all_positive = gt_dists[gt_dists > 0]
    p(f"    GT range (visited):     [{gt_visited.min():.6f}, {gt_visited.max():.6f}]")
    p(f"    GT range (all points):  [{gt_all_positive.min():.6f}, {gt_all_positive.max():.6f}]")
    p(f"    Coverage: visited covers {100 * gt_visited.max() / gt_all_positive.max():.1f}% "
      f"of max GT distance")

    # ── 14. Plots ─────────────────────────────────────────────────
    banner("15. Save plots")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_name = Path(args.model_path).parent.name
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(
            f"FM Propagation Debug: {model_name}\n"
            f"Scene: {gaussian_dir.name} | N={N:,} | visited={visited_count} | "
            f"MAE={mae:.6f} | R²={r2:.4f}",
            fontsize=13,
        )

        # (0,0) pred vs GT scatter
        ax = axes[0, 0]
        sc = ax.scatter(gt_v, pred_v, s=4, alpha=0.3, c=n_visited_nbrs_arr,
                        cmap="viridis", vmin=0, vmax=max(n_visited_nbrs_arr.max(), 1))
        lim = [min(gt_v.min(), pred_v.min()) * 0.95, max(gt_v.max(), pred_v.max()) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=0.8, label="y=x")
        ax.set_title("Predicted vs GT (color = #visited nbrs)")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("FM predicted distance")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="#vis nbrs")

        # (0,1) abs error vs #visited neighbors
        ax = axes[0, 1]
        ax.scatter(n_visited_nbrs_arr, abs_err, s=4, alpha=0.3, c="coral")
        ax.set_title("Absolute error vs #visited neighbors")
        ax.set_xlabel("#visited neighbors")
        ax.set_ylabel("|pred - GT|")

        # (1,0) signed error vs GT distance
        ax = axes[1, 0]
        ax.scatter(gt_v, signed_err, s=4, alpha=0.3, c="darkred")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title("Signed error (pred - GT) vs GT distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("pred - GT")

        # (1,1) abs error histogram
        ax = axes[1, 1]
        ax.hist(abs_err, bins=80, color="coral", edgecolor="black", alpha=0.8)
        ax.axvline(mae, color="red", linestyle="--", label=f"MAE={mae:.4f}")
        ax.axvline(np.median(abs_err), color="green", linestyle=":",
                   label=f"med={np.median(abs_err):.4f}")
        ax.set_title("Absolute error distribution")
        ax.set_xlabel("|pred - GT|")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = output_dir / "propagation_debug_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        p(f"    Saved plots -> {plot_path}")
    except ImportError:
        p("    matplotlib not available - skipping plots")

    # ── 15. Save NPZ ──────────────────────────────────────────────
    npz_path = output_dir / "propagation_debug.npz"
    np.savez(
        npz_path,
        distances=distances,
        gt_dists=gt_dists,
        visited_mask=visited_mask,
        source_idx=source_idx,
    )
    p(f"    Results saved to {npz_path}")

    # ── 16. Timing summary ────────────────────────────────────────
    banner("17. Timing summary")
    p(f"    Load data:          {t_load:7.2f}s")
    p(f"    Build propagator:   {t_build:7.2f}s")
    p(f"    FM Propagation:     {t_prop:7.2f}s")
    p(f"    Throughput: {visited_count / max(t_prop, 0.01):,.0f} points/s")

    p(f"\n{'=' * 74}")
    p("  Evaluation complete.")
    p(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
