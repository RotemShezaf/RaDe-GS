#!/usr/bin/env python3
"""
Analyze outlier predictions from a trained GaussianPatchTransformer on training data.

Loads the best checkpoint and evaluates on the FULL dataset (train+val splits),
then reports which examples have the highest errors, grouped by source shape.

Usage (via srun):
    srun --gres=gpu:1 --cpus-per-task=4 --time=01:00:00 --pty bash -c \
        'cd /home/rotem.shezaf/RaDe-GS && \
         source $(conda info --base)/etc/profile.d/conda.sh && \
         conda activate geo_splat && \
         python scripts/tosca/analyze_outliers.py'

Outputs:
    - Console: per-shape and global statistics, top-N worst examples
    - NPZ file: checkpoints/combined_tosca_ring3/outlier_analysis.npz
      with per-example predictions, targets, errors, source indices, etc.
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.GaussianPatchTransformer import create_gaussian_patch_transformer
from DataSets.gaussian_dataset import CombinedGaussianPatchDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze outlier predictions")
    parser.add_argument(
        "--train_config",
        type=str,
        default=str(PROJECT_ROOT / "models/configs/tosca/combined_tosca_ring3.yaml"),
        help="Training config YAML path",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=50, help="Number of worst examples to print")
    parser.add_argument(
        "--percentile_thresholds",
        type=float,
        nargs="+",
        default=[90, 95, 99, 99.5, 99.9],
        help="Error percentiles to report",
    )
    return parser.parse_args()


def load_model_and_config(train_config_path: str, device: torch.device):
    """Load the trained model from its config and best checkpoint."""
    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    dataset_cfg_path = PROJECT_ROOT / train_cfg["dataset"]["dataset_config"]
    with open(dataset_cfg_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    ring = train_cfg["dataset"]["ring"]
    attributes = dataset_cfg.get("attributes", ["xyz"])
    use_mahalanobis = dataset_cfg.get("use_mahalanobis", False)
    method = "mahalanobis" if use_mahalanobis else "euclidean"
    max_neighbors = dataset_cfg["ring_size_mapping"][method][ring]

    model_cfg = train_cfg["model"]
    model = create_gaussian_patch_transformer(
        attributes=attributes,
        point_attributes=model_cfg.get("point_attributes"),
        max_neighbors=max_neighbors,
        embed_dim=model_cfg.get("embed_dim", 128),
        encoder_depth=model_cfg.get("encoder_depth", 4),
        num_heads=model_cfg.get("num_heads", 8),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        pool=model_cfg.get("pool", "max"),
        pos_encoding_type=model_cfg.get("pos_encoding_type", "index"),
        encoder_type=model_cfg.get("encoder_type", "linear"),
    )

    # Load checkpoint
    ckpt_dir = Path(train_cfg["infrastructure"]["save_dir"])
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    best_mae = ckpt.get("best_val_mae", ckpt.get("metrics", {}).get("mae", "?"))
    print(f"Loaded checkpoint from epoch {epoch}  (best val MAE = {best_mae})")

    return model, train_cfg, dataset_cfg, attributes, ring, max_neighbors


def build_dataset(dataset_cfg, train_cfg, attributes, ring):
    """Build the full dataset (no transforms, no augmentation) for clean analysis."""
    dataset_cfg_path = PROJECT_ROOT / train_cfg["dataset"]["dataset_config"]
    use_r1_min = train_cfg["dataset"].get("use_r1_min", False)

    dataset = CombinedGaussianPatchDataset(
        config=str(dataset_cfg_path),
        attributes=attributes,
        ring=ring,
        use_r1_min=use_r1_min,
        transform=None,          # NO augmentations
        mask_constant=-10.0,
    )
    return dataset


@torch.no_grad()
def run_inference(model, dataset, batch_size, num_workers, device):
    """Run model on every example; return predictions, targets, and per-example metadata."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []
    all_valid_counts = []     # how many valid (non-masked) neighbors per example
    all_max_geodesic = []     # max geodesic distance among valid neighbors
    all_mean_geodesic = []    # mean geodesic distance among valid neighbors
    all_target_scale = []     # raw target value (= p_u, the query-point geodesic)

    for neighborhood, point_features, targets, valid_mask in tqdm(loader, desc="Inference"):
        neighborhood = neighborhood.to(device)
        point_features = point_features.to(device)
        valid_mask = valid_mask.to(device)

        preds = model(neighborhood, point_features, valid_mask).squeeze(-1)

        # Compute per-example statistics on the input patches
        geodesic_col = neighborhood[:, :, -1]                    # (B, N) — last feature is geodesic
        valid_geodesic = geodesic_col.clone()
        valid_geodesic[~valid_mask] = float("nan")

        valid_count = valid_mask.sum(dim=1).cpu()                # (B,)
        max_geo = torch.nanmax(valid_geodesic, dim=1).values.cpu()
        mean_geo = torch.nanmean(valid_geodesic, dim=1).cpu()

        all_preds.append(preds.cpu())
        all_targets.append(targets)
        all_valid_counts.append(valid_count)
        all_max_geodesic.append(max_geo)
        all_mean_geodesic.append(mean_geo)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    valid_counts = torch.cat(all_valid_counts).numpy()
    max_geodesic = torch.cat(all_max_geodesic).numpy()
    mean_geodesic = torch.cat(all_mean_geodesic).numpy()

    return preds, targets, valid_counts, max_geodesic, mean_geodesic


def compute_source_indices(dataset):
    """Return (source_idx_per_example, source_names)."""
    source_indices = np.empty(len(dataset), dtype=np.int32)
    for i, (start, end) in enumerate(
        zip(dataset.cumulative_lengths[:-1], dataset.cumulative_lengths[1:])
    ):
        source_indices[start:end] = i
    return source_indices, dataset.dataset_names


def analyze_and_report(
    preds, targets, valid_counts, max_geodesic, mean_geodesic,
    source_indices, source_names, top_n, percentile_thresholds, save_path,
):
    """Print detailed analysis and save results."""
    errors = np.abs(preds - targets)
    sq_errors = (preds - targets) ** 2
    rel_errors = errors / np.clip(np.abs(targets), 1e-6, None)

    n = len(preds)
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    print(f"  Total examples:       {n:,}")
    print(f"  MAE:                  {errors.mean():.6f}")
    print(f"  RMSE:                 {np.sqrt(sq_errors.mean()):.6f}")
    print(f"  Median AE:            {np.median(errors):.6f}")
    print(f"  Max AE:               {errors.max():.6f}")
    print(f"  Mean Rel Error:       {rel_errors.mean():.4f}  ({rel_errors.mean()*100:.2f}%)")
    print(f"  Target range:         [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"  Prediction range:     [{preds.min():.4f}, {preds.max():.4f}]")

    print(f"\n  Error percentiles:")
    for p in percentile_thresholds:
        val = np.percentile(errors, p)
        print(f"    P{p:5.1f}:  {val:.6f}")

    # ── Per-source breakdown ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-SHAPE BREAKDOWN")
    print("=" * 80)
    print(f"{'Shape':<20s} {'N':>8s} {'MAE':>10s} {'RMSE':>10s} {'MedAE':>10s} "
          f"{'MaxAE':>10s} {'MeanRel%':>10s} {'P99':>10s} {'AvgNeigh':>10s}")
    print("-" * 110)
    for si in range(len(source_names)):
        mask = source_indices == si
        e = errors[mask]
        se = sq_errors[mask]
        re = rel_errors[mask]
        vc = valid_counts[mask]
        print(
            f"{source_names[si]:<20s} {mask.sum():>8,d} {e.mean():>10.6f} "
            f"{np.sqrt(se.mean()):>10.6f} {np.median(e):>10.6f} "
            f"{e.max():>10.6f} {re.mean()*100:>9.2f}% "
            f"{np.percentile(e, 99):>10.6f} {vc.mean():>10.1f}"
        )

    # ── Correlation between patch characteristics and error ───────────
    print("\n" + "=" * 80)
    print("ERROR CORRELATIONS (Pearson r)")
    print("=" * 80)
    correlations = {
        "valid_neighbor_count": valid_counts,
        "target_geodesic":      targets,
        "max_neighbor_geodesic": max_geodesic,
        "mean_neighbor_geodesic": mean_geodesic,
    }
    for name, values in correlations.items():
        finite = np.isfinite(values)
        if finite.sum() > 10:
            r = np.corrcoef(errors[finite], values[finite])[0, 1]
            print(f"  {name:<30s}  r = {r:+.4f}")

    # ── Bucket analysis: error vs valid neighbor count ────────────────
    print("\n" + "=" * 80)
    print("ERROR BY VALID NEIGHBOR COUNT (buckets)")
    print("=" * 80)
    count_bins = [0, 10, 30, 50, 80, 100, 128, 200]
    print(f"{'Bucket':<15s} {'N':>8s} {'MAE':>10s} {'P95':>10s} {'P99':>10s}")
    print("-" * 60)
    for lo, hi in zip(count_bins[:-1], count_bins[1:]):
        mask = (valid_counts >= lo) & (valid_counts < hi)
        if mask.sum() == 0:
            continue
        e = errors[mask]
        print(f"[{lo:>3d}, {hi:>3d}){' ':>6s} {mask.sum():>8,d} {e.mean():>10.6f} "
              f"{np.percentile(e, 95):>10.6f} {np.percentile(e, 99):>10.6f}")

    # ── Bucket analysis: error vs target geodesic value ───────────────
    print("\n" + "=" * 80)
    print("ERROR BY TARGET GEODESIC VALUE (buckets)")
    print("=" * 80)
    target_pcts = np.percentile(targets, [0, 10, 25, 50, 75, 90, 100])
    print(f"{'Bucket':<20s} {'N':>8s} {'MAE':>10s} {'MeanRel%':>10s} {'P99':>10s}")
    print("-" * 65)
    for lo, hi in zip(target_pcts[:-1], target_pcts[1:]):
        mask = (targets >= lo) & (targets < hi + 1e-8)
        if mask.sum() == 0:
            continue
        e = errors[mask]
        re = rel_errors[mask]
        print(f"[{lo:>6.3f}, {hi:>6.3f}){' ':>3s} {mask.sum():>8,d} {e.mean():>10.6f} "
              f"{re.mean()*100:>9.2f}% {np.percentile(e, 99):>10.6f}")

    # ── Signed error analysis ─────────────────────────────────────────
    signed = preds - targets
    print("\n" + "=" * 80)
    print("SIGNED ERROR (pred - target) ANALYSIS")
    print("=" * 80)
    print(f"  Mean signed error:  {signed.mean():+.6f}  (>0 = model overestimates)")
    print(f"  Std signed error:   {signed.std():.6f}")
    print(f"  Skewness:           {float(np.mean(((signed - signed.mean()) / signed.std()) ** 3)):+.3f}")
    over = (signed > 0).sum()
    under = (signed < 0).sum()
    print(f"  Overestimates:      {over:,d} ({100*over/n:.1f}%)")
    print(f"  Underestimates:     {under:,d} ({100*under/n:.1f}%)")

    # ── Top-N worst examples ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"TOP-{top_n} WORST EXAMPLES")
    print("=" * 80)
    worst_idx = np.argsort(errors)[-top_n:][::-1]
    print(f"{'Rank':>4s} {'Idx':>8s} {'Shape':<18s} {'Target':>8s} {'Pred':>8s} "
          f"{'AbsErr':>8s} {'Rel%':>8s} {'#Neigh':>8s} {'MaxGeo':>8s} {'MeanGeo':>8s}")
    print("-" * 110)
    for rank, idx in enumerate(worst_idx, 1):
        si = source_indices[idx]
        print(
            f"{rank:>4d} {idx:>8d} {source_names[si]:<18s} "
            f"{targets[idx]:>8.4f} {preds[idx]:>8.4f} "
            f"{errors[idx]:>8.4f} {rel_errors[idx]*100:>7.1f}% "
            f"{valid_counts[idx]:>8d} "
            f"{max_geodesic[idx]:>8.4f} {mean_geodesic[idx]:>8.4f}"
        )

    # ── Save NPZ ─────────────────────────────────────────────────────
    np.savez_compressed(
        save_path,
        predictions=preds,
        targets=targets,
        errors=errors,
        rel_errors=rel_errors,
        valid_counts=valid_counts,
        max_geodesic=max_geodesic,
        mean_geodesic=mean_geodesic,
        source_indices=source_indices,
        source_names=np.array(source_names),
    )
    print(f"\nSaved detailed results to: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model
    model, train_cfg, dataset_cfg, attributes, ring, max_neighbors = load_model_and_config(
        args.train_config, device
    )

    # 2. Build full dataset (no augmentations)
    dataset = build_dataset(dataset_cfg, train_cfg, attributes, ring)

    # 3. Run inference
    preds, targets, valid_counts, max_geodesic, mean_geodesic = run_inference(
        model, dataset, args.batch_size, args.num_workers, device
    )

    # 4. Source mapping
    source_indices, source_names = compute_source_indices(dataset)

    # 5. Analyze & report
    save_path = Path(train_cfg["infrastructure"]["save_dir"]) / "outlier_analysis.npz"
    analyze_and_report(
        preds, targets, valid_counts, max_geodesic, mean_geodesic,
        source_indices, source_names,
        args.top_n, args.percentile_thresholds,
        save_path,
    )


if __name__ == "__main__":
    main()
