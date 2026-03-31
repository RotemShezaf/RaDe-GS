#!/usr/bin/env python3
"""
Evaluate a trained GaussianPatchTransformer model against ground-truth
geodesic distances on a specific Gaussian splat scene.

For every point (except the source), builds a model input using
GaussianInputBuilder with GT neighbour distances (all neighbours treated
as visited), runs the model, denormalizes the prediction, and compares
to the ground-truth absolute geodesic distance.

Outputs:
  - Per-point CSV  (point_idx, gt_dist, pred_dist, abs_err, rel_err, …)
  - Console tables: accuracy binned by GT distance, by #neighbors, overall summary
  - Histogram PNG  (pred vs GT scatter, error distribution, etc.)

Usage (from project root, geo_splat env):
    python geodesic_propagation/evaluate_model_vs_gt.py

All paths are configurable via CLI arguments — run with --help for details.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

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
        description="Evaluate GaussianPatchTransformer model accuracy vs GT geodesic distances",
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
        help="Path to the training config YAML (for model architecture & transforms)",
    )
    parser.add_argument(
        "--dataset_config", type=str,
        default=None,
        help="Path to the dataset config YAML. If omitted, resolved from train_config.",
    )
    parser.add_argument(
        "--gaussian_dir", type=str,
        default="TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_04/light_4/output",
        help="Path to the Gaussian splat output directory",
    )
    parser.add_argument(
        "--gt_subpath", type=str,
        default="geodesic_distance/gt_geodesic.npz",
        help="Relative path (inside gaussian_dir) to the GT geodesic .npz",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="geodesic_propagation/eval_output",
        help="Directory for CSV and plot output",
    )

    # ----- Neighborhood -----
    parser.add_argument("--ring", type=int, default=3, help="Ring level for neighborhoods")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Ring-1 neighbor count (kNN k)")
    parser.add_argument("--use_mahalanobis", action="store_true", help="Use Mahalanobis distance for kNN")

    # ----- Evaluation -----
    parser.add_argument("--source_row", type=int, default=None,
                        help="Which GT source row to use (single-source mode). "
                             "If omitted, uses num_sources from --dataset_config")
    parser.add_argument("--num_sources", type=int, default=None,
                        help="Override num_sources from dataset config")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for model inference")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu). Auto-detect if omitted.")
    parser.add_argument("--n_bins", type=int, default=15, help="Number of distance bins for binned statistics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

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

    # ── 1. Echo configuration ──────────────────────────────────────
    banner("1. Configuration")
    for k, v in vars(args).items():
        p(f"    {k:20s} = {v}")
    p(f"    {'device':20s} = {device}")

    # ── 2. Load Gaussian data ──────────────────────────────────────
    banner("2. Load Gaussian data")
    gdata = load_gaussian_data_cpu(gaussian_dir, iteration=None, load_sh_features=False)
    positions = gdata.get_xyz()
    scales = gdata.get_scaling()
    rotations = gdata.get_rotation()
    opacities = gdata.get_opacity()
    N = len(positions)
    p(f"    {N} Gaussians loaded")
    stat("positions", positions)

    # ── 3. Load GT geodesic ────────────────────────────────────────
    banner("3. Load GT geodesic distances")
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data["geodesic_distances"]
    gt_source_indices = gt_data["source_gaussian_indices"]
    p(f"    {len(gt_source_indices)} GT sources available")

    # Read num_sources from dataset config (matches training patch generation)
    _ds_cfg = yaml.safe_load(open(dataset_config_path)) if not isinstance(dataset_config_path, dict) else dataset_config_path
    num_sources_cfg = int(_ds_cfg.get("num_sources", 1))
    
    if args.source_row is not None:
        # Legacy single-source mode
        num_sources = 1
        selected_rows = [args.source_row]
        p(f"    Single-source mode: source_row={args.source_row}")
    else:
        num_sources = args.num_sources if args.num_sources is not None else num_sources_cfg
        if num_sources > len(gt_source_indices):
            num_sources = len(gt_source_indices)
        selected_rows = np.random.choice(
            len(gt_source_indices), num_sources, replace=False
        ).tolist()
        p(f"    Multi-source mode: num_sources={num_sources} (from {'--num_sources' if args.num_sources else 'dataset config'})")
        p(f"    Selected source rows: {selected_rows}")

    print(f"    num_sources: {len(selected_rows)}")
    # Extract geodesic field for the selected source(s).
    # With a single source (the normal case) this is just the raw field for
    # that source.  With multiple sources it is the elementwise min, but that
    # mode is provided only for experimentation and does not match training.
    selected_dists = gt_all_dists[selected_rows].astype(np.float64)  # (num_sources, N)
    gt_dists = selected_dists[0] if len(selected_rows) == 1 else selected_dists.min(axis=0)

    source_gaussian_idxs = gt_source_indices[selected_rows]
    p(f"    Source Gaussian indices: {source_gaussian_idxs.tolist()}")
    stat("GT geodesic", gt_dists)

    # ── 4. Load model ────────────────────────────────────────────────
    banner("4. Load model")
    train_cfg = _train_cfg_raw

    model_handler = ModelHandler(model_path=str(model_path), device=str(device))
    transforms_cfg = train_cfg.get("dataset", {}).get("transforms", None)
    p(f"    Model loaded: {model_path}")

    # ── 5. Build GaussianInputBuilder (ring computation + transforms) ─
    banner("5. Build GaussianInputBuilder")
    config = yaml.safe_load(open(dataset_config_path)) if not isinstance(dataset_config_path, dict) else dataset_config_path
    args.n_neighbors = config.get("n_neighbors", args.n_neighbors)
    args.use_mahalanobis = config.get("use_mahalanobis", args.use_mahalanobis or False)
    adaptive_target_ring = config.get("adaptive_target_ring", None)
    adaptive_k_boost = config.get("adaptive_k_boost", None)
    adaptive_target_neighbors = config.get("adaptive_target_neighbors", None)
    adaptive_max_mean_cut = config.get("adaptive_max_mean_cut", 0)
    adaptive_max_steps = config.get("adaptive_max_steps", 12)

    builder = GaussianInputBuilder(
        positions=positions,
        dataset_config=str(dataset_config_path),
        ring=args.ring,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        device=str(device),
        n_neighbors=args.n_neighbors,
        use_mahalanobis=args.use_mahalanobis,
        adaptive_target_ring=adaptive_target_ring,
        adaptive_target_neighbors=adaptive_target_neighbors,
        adaptive_k_boost=adaptive_k_boost if adaptive_k_boost is not None else 20,
        adaptive_max_mean_cut=adaptive_max_mean_cut,
        adaptive_max_steps=adaptive_max_steps,
        transforms_config=transforms_cfg,
    )

    ring1_nbrs = builder.ring1_neighbors
    ring_neighbors = builder.ring_neighbors
    mean_nn_dist = builder.normalization_factor

    p(f"    mean_nn_dist     = {mean_nn_dist:.8f}")
    sizes = np.array([len(ring_neighbors[i]) for i in range(N)])
    stat(f"ring-{args.ring} neighborhood sizes", sizes)
    p(f"    attributes       = {builder.attributes}")
    p(f"    max_neighbors    = {builder.max_neighbors}")
    p(f"    mask_constant    = {builder.mask_constant}")
    p(f"    nn_mean          = {builder.nn_mean}")
    p(f"    entry_size       = {builder.get_neighbor_feature_dim()}")
    p(f"    point_feat_dim   = {builder.get_point_feature_dim()}")

    # ── 6. Evaluate all points ─────────────────────────────────────
    banner("6. Evaluate model on all points with GT distances")

    # All-visited mask: every point has known GT distance
    visited_mask_all = np.ones(N, dtype=bool)
    # Use GT distances as the global distance array
    distances_global = gt_dists.copy()

    # Exclude all source points + points with non-finite GT
    eval_mask = np.ones(N, dtype=bool)
    for sg_idx in source_gaussian_idxs:
        eval_mask[int(sg_idx)] = False
    eval_mask[~np.isfinite(gt_dists)] = False
    eval_mask[gt_dists <= 0] = False
    eval_indices = np.where(eval_mask)[0]
    p(f"    Evaluating {len(eval_indices)} points (excluding {len(source_gaussian_idxs)} sources + non-finite)")

    # Result arrays
    pred_dists = np.full(N, np.nan, dtype=np.float64)
    raw_preds_arr = np.full(N, np.nan, dtype=np.float64)
    n_valid_nbrs_arr = np.zeros(N, dtype=np.int32)
    n_ring_nbrs_arr = np.zeros(N, dtype=np.int32)

    t0 = time.time()
    n_skipped = 0
    n_fallback = 0       # points that fell back to all-visited
    fallback_pids = []   # track which points used the fallback

    for batch_start in range(0, len(eval_indices), args.batch_size):
        batch_idx = eval_indices[batch_start : batch_start + args.batch_size]

        batch_neighborhoods = []
        batch_point_features = []
        batch_valid_masks = []
        batch_build_infos = []
        valid_in_batch = []

        for pid in batch_idx:
            nbrs = ring_neighbors.get(pid, np.array([], dtype=np.int64))
            if len(nbrs) == 0:
                n_skipped += 1
                continue

            # FM-like visited mask: only neighbors with GT distance <= current
            visited_mask = gt_dists <= gt_dists[pid]

            # Fallback: if no ring-k neighbor qualifies, treat ALL neighbors
            # as visited so the model still receives a valid input.
            if not np.any(visited_mask[nbrs]):
                visited_mask = np.ones(N, dtype=bool)
                n_fallback += 1
                fallback_pids.append(pid)

            result = builder.build_input(
                point_idx=pid,
                all_neighbor_indices=nbrs,
                neighbor_distances=distances_global,
                ring1_neighbor_indices=ring1_nbrs[pid],
                visited_mask=visited_mask,
            )

            if result is None:
                n_skipped += 1
                continue

            neighborhood, point_features, valid_mask, build_info = result

            batch_neighborhoods.append(neighborhood)
            batch_point_features.append(point_features)
            batch_valid_masks.append(valid_mask)
            batch_build_infos.append(build_info)
            valid_in_batch.append(pid)
            n_ring_nbrs_arr[pid] = len(nbrs)
            n_valid_nbrs_arr[pid] = int(valid_mask.sum())

        if not valid_in_batch:
            continue

        neighborhoods_t = torch.stack(batch_neighborhoods, dim=0)
        point_features_t = torch.stack(batch_point_features, dim=0)
        valid_masks_t = torch.stack(batch_valid_masks, dim=0)

        predictions = model_handler.predict(neighborhoods_t, point_features_t, valid_masks_t)

        for i, pid in enumerate(valid_in_batch):
            rp = predictions[i, 0].item()
            raw_preds_arr[pid] = rp
            recovered = builder.denormalize_result(rp, batch_build_infos[i])
            pred_dists[pid] = recovered

        # Progress
        done = batch_start + len(batch_idx)
        if done % (args.batch_size * 10) == 0 or done >= len(eval_indices):
            p(f"    {done}/{len(eval_indices)} done ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    p(f"    Finished in {elapsed:.1f}s  (skipped {n_skipped}, fallback-to-all-visited {n_fallback})")
    if fallback_pids:
        p(f"    Fallback points (no ring-{args.ring} neighbor with GT dist <= point's GT dist):")
        for fp in fallback_pids:
            fp_nbrs = ring_neighbors.get(fp, np.array([], dtype=np.int64))
            nbr_min_gt = gt_dists[fp_nbrs].min() if len(fp_nbrs) > 0 else float('nan')
            nbr_max_gt = gt_dists[fp_nbrs].max() if len(fp_nbrs) > 0 else float('nan')
            p(f"      pid={fp}  gt_dist={gt_dists[fp]:.8f}  "
              f"#ring_nbrs={len(fp_nbrs)}  "
              f"nbr_gt_range=[{nbr_min_gt:.8f}, {nbr_max_gt:.8f}]")

    # ── 8. Compute errors ──────────────────────────────────────────
    banner("7. Compute errors")
    valid = ~np.isnan(pred_dists) & eval_mask
    valid_idx = np.where(valid)[0]

    gt_v = gt_dists[valid_idx]
    pred_v = pred_dists[valid_idx]
    raw_v = raw_preds_arr[valid_idx]
    nvn = n_valid_nbrs_arr[valid_idx]
    nrn = n_ring_nbrs_arr[valid_idx]

    abs_err = np.abs(pred_v - gt_v)
    rel_err = abs_err / np.maximum(gt_v, 1e-10)

    p(f"    {len(valid_idx)} valid predictions")

    # ── 9. Save CSV ────────────────────────────────────────────────
    banner("8. Save CSV")
    csv_path = output_dir / "model_vs_gt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_idx", "gt_dist", "pred_dist", "raw_pred",
            "abs_err", "rel_err", "n_valid_nbrs", "n_ring_nbrs",
        ])
        for i, pid in enumerate(valid_idx):
            writer.writerow([
                pid, gt_v[i], pred_v[i], raw_v[i],
                abs_err[i], rel_err[i], nvn[i], nrn[i],
            ])
    p(f"    Saved {len(valid_idx)} rows → {csv_path}")

    # ── 10. Binned statistics by GT distance ───────────────────────
    banner("9. Accuracy binned by GT distance")
    bin_edges = np.linspace(gt_v.min(), gt_v.max() + 1e-10, args.n_bins + 1)
    hdr = (f"{'bin_range':>24s}  {'count':>6s}  {'MAE':>10s}  {'medAE':>10s}  "
           f"{'meanRE':>10s}  {'mean_pred':>10s}  {'mean_GT':>10s}  "
           f"{'mean_raw':>10s}  {'avg_nbrs':>9s}")
    p(f"  {hdr}")
    p(f"  {'-' * len(hdr)}")

    for b in range(args.n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (gt_v >= lo) & (gt_v < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mae = abs_err[mask].mean()
        med_ae = np.median(abs_err[mask])
        mean_re = rel_err[mask].mean()
        mean_pred = pred_v[mask].mean()
        mean_gt = gt_v[mask].mean()
        mean_raw = raw_v[mask].mean()
        avg_n = nvn[mask].mean()
        p(f"  [{lo:10.4f}, {hi:10.4f})  {cnt:>6}  {mae:>10.6f}  {med_ae:>10.6f}  "
          f"{mean_re:>9.4%}  {mean_pred:>10.6f}  {mean_gt:>10.6f}  "
          f"{mean_raw:>10.6f}  {avg_n:>9.1f}")

    # ── 11. Binned by #valid neighbors ──────────────────────────────
    banner("10. Accuracy binned by number of valid neighbors")
    hdr2 = f"{'n_nbrs':>6s}  {'count':>6s}  {'MAE':>10s}  {'meanRE':>10s}  {'mean_pred':>10s}  {'mean_GT':>10s}"
    p(f"  {hdr2}")
    p(f"  {'-' * len(hdr2)}")

    for nv in sorted(set(nvn)):
        mask = nvn == nv
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mae = abs_err[mask].mean()
        mean_re = rel_err[mask].mean()
        mean_pred = pred_v[mask].mean()
        mean_gt = gt_v[mask].mean()
        p(f"  {int(nv):>6}  {cnt:>6}  {mae:>10.6f}  {mean_re:>9.4%}  "
          f"{mean_pred:>10.6f}  {mean_gt:>10.6f}")

    # ── 12. Overall summary ─────────────────────────────────────────
    banner("11. Overall summary")
    p(f"    Total evaluated    = {len(valid_idx)}")
    p(f"    MAE                = {abs_err.mean():.8f}")
    p(f"    Median AE          = {np.median(abs_err):.8f}")
    p(f"    Mean relative err  = {rel_err.mean():.4%}")
    p(f"    Max  absolute err  = {abs_err.max():.8f}")
    p(f"    P90  absolute err  = {np.percentile(abs_err, 90):.8f}")
    p(f"    P95  absolute err  = {np.percentile(abs_err, 95):.8f}")
    p(f"    P99  absolute err  = {np.percentile(abs_err, 99):.8f}")
    p(f"    Mean pred          = {pred_v.mean():.8f}")
    p(f"    Mean GT            = {gt_v.mean():.8f}")
    p(f"    Mean raw pred      = {raw_v.mean():.8f}")
    n_zero = int(np.sum(pred_v == 0.0))
    p(f"    Preds == 0         = {n_zero}/{len(valid_idx)} ({100 * n_zero / max(len(valid_idx), 1):.1f}%)")
    n_negative = int(np.sum(pred_v < 0.0))
    p(f"    Preds < 0          = {n_negative}")

    corr = np.corrcoef(gt_v, pred_v)[0, 1] if len(gt_v) > 2 else float("nan")
    p(f"    Pearson corr       = {corr:.6f}")

    # R² score
    ss_res = np.sum((pred_v - gt_v) ** 2)
    ss_tot = np.sum((gt_v - gt_v.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    p(f"    R² score           = {r2:.6f}")

    # ── 13. Histograms / scatter plot ───────────────────────────────
    banner("12. Save plots")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        model_name = Path(args.model_path).parent.name
        fig.suptitle(f"Model evaluation: {model_name}\n"
                     f"Scene: {gaussian_dir.name}  |  MAE={abs_err.mean():.6f}  R²={r2:.4f}",
                     fontsize=13)

        # (0,0) Pred vs GT scatter
        ax = axes[0, 0]
        ax.scatter(gt_v, pred_v, s=3, alpha=0.3, c="steelblue")
        lim = [min(gt_v.min(), pred_v.min()) * 0.95, max(gt_v.max(), pred_v.max()) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=0.8, label="y = x")
        ax.set_title("Predicted vs GT geodesic distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("Predicted distance")
        ax.legend(fontsize=8)

        # (0,1) Absolute error histogram
        ax = axes[0, 1]
        ax.hist(abs_err, bins=80, color="coral", edgecolor="black", alpha=0.8)
        ax.axvline(abs_err.mean(), color="red", linestyle="--", label=f"MAE={abs_err.mean():.4f}")
        ax.axvline(np.median(abs_err), color="green", linestyle=":", label=f"med={np.median(abs_err):.4f}")
        ax.set_title("Absolute error distribution")
        ax.set_xlabel("|pred − GT|")
        ax.legend(fontsize=8)

        # (1,0) Abs error vs GT distance
        ax = axes[1, 0]
        ax.scatter(gt_v, abs_err, s=3, alpha=0.3, c="navy")
        ax.set_title("Absolute error vs GT distance")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("|pred − GT|")

        # (1,1) Relative error vs GT distance
        ax = axes[1, 1]
        # Clip rel_err for plotting to avoid outlier stretching
        re_clipped = np.clip(rel_err, 0, np.percentile(rel_err, 99))
        ax.scatter(gt_v, re_clipped, s=3, alpha=0.3, c="darkred")
        ax.set_title("Relative error vs GT distance (clipped at P99)")
        ax.set_xlabel("GT distance")
        ax.set_ylabel("|pred − GT| / GT")

        plt.tight_layout()
        plot_path = output_dir / "model_vs_gt_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        p(f"    Saved plots → {plot_path}")

    except ImportError:
        p("    matplotlib not available — skipping plots")

    # ── Done ────────────────────────────────────────────────────────
    p(f"\n  Output files:")
    p(f"    CSV:   {csv_path}")
    p(f"    Plots: {output_dir}/*.png")
    p(f"\n{'=' * 74}")
    p("  Evaluation complete.")
    p(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
