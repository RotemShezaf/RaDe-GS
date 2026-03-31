#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geodesic propagation evaluation with optional FPS downsampling and GPU support.

Runs Fast Marching propagation from a GT source point, then compares
predicted distances against ground-truth geodesic distances.

All paths and parameters are configurable via CLI arguments.

Usage examples:
    # Full scene on CPU (no FPS):
    python geodesic_propagation/run_eval_fps.py --device cpu --fps_target 0

    # Downsampled on GPU via srun:
    srun --nodelist=gipdeep7 --gres=gpu:1 --cpus-per-task=6 --time=2:00:00 --pty bash -c \\
        'cd /home/rotem.shezaf/RaDe-GS && conda activate geo_splat && \\
         python geodesic_propagation/run_eval_fps.py --fps_target 10000'
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
from utils.misc import fps_gs


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
        description="Evaluate geodesic propagation (Fast Marching) vs GT distances",
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
        default="models/configs/combined_polynomial_ring3.yaml",
        help="Path to training config YAML (model arch + transforms)",
    )
    parser.add_argument(
        "--dataset_config", type=str,
        default=None,
        help="Path to dataset config YAML. If omitted, resolved from train_config.",
    )
    parser.add_argument(
        "--gaussian_dir", type=str,
        default="TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/level_04/light_4/output",
        help="Gaussian splat output directory",
    )
    parser.add_argument(
        "--gt_subpath", type=str,
        default="geodesic_distance/gt_geodesic.npz",
        help="Relative path inside gaussian_dir to GT geodesic .npz",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="geodesic_propagation/eval_propagation_output",
        help="Directory for CSV / plot output",
    )

    # ----- Neighborhood -----
    parser.add_argument("--ring", type=int, default=3, help="Ring level")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Ring-1 kNN k")
    parser.add_argument("--use_mahalanobis", action="store_true", help="Mahalanobis kNN")

    # ----- FPS / evaluation -----
    parser.add_argument("--fps_target", type=int, default=0,
                        help="FPS downsample target (0 = no downsampling)")
    parser.add_argument("--source_row", type=int, default=0,
                        help="GT source row index")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for FM batch propagation")
    parser.add_argument("--refine_passes", type=int, default=3,
                        help="Number of post-FM refinement passes (0 to skip)")
    parser.add_argument("--dampening", type=float, default=1.0,
                        help="Refinement dampening factor (0-1). 1.0=full replacement, 0.3=gentle")
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
    import yaml
    with open(train_config_path) as _f:
        _train_cfg_raw = yaml.safe_load(_f)

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
    positions_full = gdata.get_xyz()
    scales_full = gdata.get_scaling()
    rotations_full = gdata.get_rotation()
    opacities_full = gdata.get_opacity()
    N_full = len(positions_full)
    t_load = time.perf_counter() - t0
    p(f"    {N_full:,} Gaussians loaded [{t_load:.2f}s]")

    # ── 3. FPS downsample ──────────────────────────────────────────
    if args.fps_target > 0 and args.fps_target < N_full:
        banner("3. FPS downsampling")
        t0 = time.perf_counter()
        fps_indices = fps_gs(positions_full, args.fps_target, device=str(device))
        t_fps = time.perf_counter() - t0
        positions = positions_full[fps_indices]
        scales = scales_full[fps_indices]
        rotations = rotations_full[fps_indices]
        opacities = opacities_full[fps_indices]
        N = len(positions)
        p(f"    {N_full:,} → {N:,} points [{t_fps:.2f}s]")
    else:
        banner("3. No downsampling")
        fps_indices = np.arange(N_full)
        positions = positions_full
        scales = scales_full
        rotations = rotations_full
        opacities = opacities_full
        N = N_full
        t_fps = 0.0
        p(f"    Using all {N:,} points")

    # ── 4. Load GT geodesic ────────────────────────────────────────
    banner("4. Load GT geodesic")
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]  # (num_sources, N_full)
    gt_source_indices = gt_data["source_gaussian_indices"]
    p(f"    {len(gt_source_indices)} GT sources")

    # Find a GT source that survived FPS
    fps_set = set(fps_indices.tolist())
    best_gt_row = None
    best_local_idx = None

    # First try the requested source_row
    src_global = int(gt_source_indices[args.source_row])
    if src_global in fps_set:
        best_gt_row = args.source_row
        best_local_idx = int(np.where(fps_indices == src_global)[0][0])
    else:
        # Search for any GT source that survived FPS
        for gt_row, gt_src in enumerate(gt_source_indices):
            if int(gt_src) in fps_set:
                best_gt_row = gt_row
                best_local_idx = int(np.where(fps_indices == int(gt_src))[0][0])
                break

    if best_gt_row is None:
        p("    WARNING: No GT source survived FPS — using closest to centroid")
        center = positions.mean(axis=0)
        best_local_idx = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
        best_gt_row = args.source_row  # fallback

    source_idx_local = best_local_idx
    source_idx_global = int(fps_indices[source_idx_local])
    gt_dists_full = gt_all_dists[best_gt_row].astype(np.float64)
    gt_dists = gt_dists_full[fps_indices]
    p(f"    GT source row={best_gt_row}, global idx={source_idx_global}, local idx={source_idx_local}")
    stat("GT geodesic (this source, downsampled)", gt_dists)

    # ── 5. Load model ──────────────────────────────────────────────
    banner("5. Load model")
    t0 = time.perf_counter()

    train_cfg = _train_cfg_raw

    model_handler = ModelHandler(model_path=str(model_path), device=str(device))
    transforms_cfg = train_cfg.get("dataset", {}).get("transforms", None)
    t_model = time.perf_counter() - t0
    p(f"    Model loaded on {device} [{t_model:.2f}s]")

    # ── 6. Build input builder + propagator ────────────────────────
    banner("6. Build GaussianInputBuilder + FastMarchingPropagator")
    t0 = time.perf_counter()

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
    mean_nn_dist = ib.normalization_factor
    per_point_nn_dist = ib.per_point_nn_distances

    p(f"    mean_nn_dist   = {mean_nn_dist:.8f}")
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

    # ── 7. Debug: inspect model I/O for one source neighbor ────────
    banner("7. Debug: model I/O for source's first neighbor")
    src_nbrs = ring1_nbrs.get(source_idx_local, np.array([], dtype=np.int64))
    p(f"    Source {source_idx_local} has {len(src_nbrs)} ring-1 neighbors")

    if len(src_nbrs) > 0:
        test_point = int(src_nbrs[0])
        test_all_nbrs = ring_neighbors.get(test_point, np.array([], dtype=np.int64))

        # Simulate: only source is visited at dist=0
        visited_mask_debug = np.zeros(N, dtype=bool)
        visited_mask_debug[source_idx_local] = True
        distances_debug = np.full(N, np.inf, dtype=np.float64)
        distances_debug[source_idx_local] = 0.0

        result = ib.build_input(
            point_idx=test_point,
            all_neighbor_indices=test_all_nbrs,
            neighbor_distances=distances_debug,
            ring1_neighbor_indices=ring1_nbrs[test_point],
            visited_mask=visited_mask_debug,
        )
        if result is not None:
            nb, pf, vm, build_info = result
            p(f"    Test point: {test_point}")
            p(f"    Neighborhood shape: {nb.shape}, valid: {vm.sum().item()}/{nb.shape[0]}")
            p(f"    Point features: {pf.cpu().numpy()}")
            p(f"    build_info keys: {list(build_info.keys())}")
            p(f"    min_input={build_info['min_input']:.8f}, "
              f"cur_norm={build_info['current_normalization']:.8f}, "
              f"max_dist={build_info['max_dist']:.8f}")

            # Show a few valid rows
            n_show = min(3, int(vm.sum().item()))
            valid_rows = torch.where(vm)[0][:n_show]
            for ri in valid_rows:
                p(f"    nbhd[{ri.item()}]: {nb[ri].cpu().numpy()}")

            # Model prediction
            with torch.no_grad():
                pred = model_handler.predict(
                    nb.unsqueeze(0), pf.unsqueeze(0), vm.unsqueeze(0)
                )
            raw_pred = pred[0, 0].item()
            denorm = ib.denormalize_result(raw_pred, build_info)

            gt_test = gt_dists[test_point]
            p(f"    Raw model prediction: {raw_pred:.8f}")
            p(f"    Denormalized dist:    {denorm:.8f}")
            p(f"    GT distance:          {gt_test:.8f}")
            p(f"    Error:                {abs(denorm - gt_test):.8f}")
        else:
            p(f"    build_input returned None for test point {test_point}")

    # ── 8. Run FM propagation ──────────────────────────────────────
    banner("8. Fast Marching propagation")
    t0 = time.perf_counter()
    distances = propagator.propagate_batch(
        [source_idx_local],
        batch_size=args.batch_size,
    )
    t_prop = time.perf_counter() - t0

    visited = int(np.sum(np.isfinite(distances)))
    finite = distances[np.isfinite(distances)]
    p(f"    Visited: {visited:,}/{N:,} ({100 * visited / N:.1f}%)")
    if len(finite) > 0:
        p(f"    Distance range: [{finite.min():.6f}, {finite.max():.6f}]")
    p(f"    Propagation time: {t_prop:.2f}s")

    mf = propagator.get_model_floor_stats()
    p(f"    model_wins={mf['model_wins']}, floor_wins={mf['floor_wins']}, "
      f"euclidean_fallback={mf['euclidean_fallback']}")
    if mf["total_predictions"] > 0:
        pct = 100.0 * mf["model_wins"] / mf["total_predictions"]
        p(f"    Model beat floor in {pct:.1f}% of predictions")

    # ── 9. Compare with GT ─────────────────────────────────────────
    banner("9. Compare with GT")
    valid = np.isfinite(distances) & np.isfinite(gt_dists)
    valid[source_idx_local] = False  # exclude source
    valid[gt_dists <= 0] = False
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

    # ── 10. Save CSV ──────────────────────────────────────────────
    banner("10. Save CSV")
    csv_path = output_dir / "propagation_vs_gt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_idx", "gt_dist", "pred_dist", "abs_err", "rel_err", "signed_err"])
        for i, pid in enumerate(valid_idx):
            writer.writerow([pid, gt_v[i], pred_v[i], abs_err[i], rel_err[i], signed_err[i]])
    p(f"    Saved {len(valid_idx)} rows → {csv_path}")

    # ── 11. Binned by GT distance ─────────────────────────────────
    banner("11. Accuracy binned by GT distance")
    bin_edges = np.linspace(gt_v.min(), gt_v.max() + 1e-10, args.n_bins + 1)
    hdr = (f"{'bin_range':>24s}  {'count':>6s}  {'MAE':>10s}  {'medAE':>10s}  "
           f"{'meanRE':>10s}  {'mean_pred':>10s}  {'mean_GT':>10s}  {'bias':>10s}")
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
        p(f"  [{lo:10.4f}, {hi:10.4f})  {cnt:>6}  {mae_b:>10.6f}  {med_ae:>10.6f}  "
          f"{mean_re:>9.4%}  {mean_pred:>10.6f}  {mean_gt:>10.6f}  {bias:>+10.6f}")

    # ── 12. Overall summary ───────────────────────────────────────
    banner("12. Overall summary")
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

    # ── 13. Plots ─────────────────────────────────────────────────
    banner("13. Save plots")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_name = Path(args.model_path).parent.name
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(
            f"FM Propagation: {model_name}\n"
            f"Scene: {gaussian_dir.name} | N={N:,} | MAE={mae:.6f} | R²={r2:.4f}",
            fontsize=13,
        )

        # (0,0) pred vs GT scatter
        ax = axes[0, 0]
        ax.scatter(gt_v, pred_v, s=2, alpha=0.2, c="steelblue")
        lim = [min(gt_v.min(), pred_v.min()) * 0.95, max(gt_v.max(), pred_v.max()) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=0.8, label="y=x")
        ax.set_title("Predicted vs GT geodesic distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("FM predicted distance")
        ax.legend(fontsize=8)

        # (0,1) abs error histogram
        ax = axes[0, 1]
        ax.hist(abs_err, bins=80, color="coral", edgecolor="black", alpha=0.8)
        ax.axvline(mae, color="red", linestyle="--", label=f"MAE={mae:.4f}")
        ax.axvline(np.median(abs_err), color="green", linestyle=":", label=f"med={np.median(abs_err):.4f}")
        ax.set_title("Absolute error distribution")
        ax.set_xlabel("|pred − GT|")
        ax.legend(fontsize=8)

        # (1,0) abs error vs GT distance
        ax = axes[1, 0]
        ax.scatter(gt_v, abs_err, s=2, alpha=0.2, c="navy")
        ax.set_title("Absolute error vs GT distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("|pred − GT|")

        # (1,1) signed error vs GT distance (bias)
        ax = axes[1, 1]
        ax.scatter(gt_v, signed_err, s=2, alpha=0.2, c="darkred")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title("Signed error (pred − GT) vs GT distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("pred − GT")

        plt.tight_layout()
        plot_path = output_dir / "propagation_vs_gt_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        p(f"    Saved plots → {plot_path}")
    except ImportError:
        p("    matplotlib not available — skipping plots")

    # ── 14. Timing summary ────────────────────────────────────────
    banner("14. Timing summary")
    total = t_load + t_fps + t_model + t_build + t_prop
    rows = [
        ("Load data", t_load),
        ("FPS downsample", t_fps),
        ("Load model", t_model),
        ("Build propagator+KNN", t_build),
        ("FM Propagation", t_prop),
        ("TOTAL", total),
    ]
    for label, s in rows:
        p(f"    {label:<22s}  {s:7.2f}s  ({100 * s / max(total, 0.01):5.1f}%)")
    p(f"    Throughput: {visited / max(t_prop, 0.01):,.0f} points/s")

    # ── Save results ───────────────────────────────────────────────
    npz_path = output_dir / "propagation_results.npz"
    np.savez(
        npz_path,
        distances=distances,
        gt_dists=gt_dists,
        fps_indices=fps_indices,
        source_idx_local=source_idx_local,
        source_idx_global=source_idx_global,
    )
    p(f"\n    Results saved to {npz_path}")

    p(f"\n  Output files:")
    p(f"    CSV:   {csv_path}")
    p(f"    NPZ:   {npz_path}")
    p(f"    Plots: {output_dir}/*.png")
    p(f"\n{'=' * 74}")
    p("  Evaluation complete.")
    p(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
