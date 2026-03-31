#!/usr/bin/env python3
"""
Geodesic propagation evaluation - tests different cloud densities.

Runs FM propagation at several FPS densities (and optionally the full cloud)
to find how density affects accuracy.

Usage:
    python geodesic_propagation/run_eval_density_test.py
    python geodesic_propagation/run_eval_density_test.py --ring 3 \
        --fps_targets 20000 10000 5000 2000
    python geodesic_propagation/run_eval_density_test.py --include_full
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
import os; os.chdir(PROJECT_ROOT)

import yaml
from GenerateData.utils.load_utils import load_gaussian_data_cpu
from utils.misc import fps_gs
from geodesic_propagation.utils.model_handler import ModelHandler
from geodesic_propagation.input_builder import GaussianInputBuilder
from geodesic_propagation.fast_marching import FastMarchingPropagator


def p(msg=''):
    print(msg, flush=True)





def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate geodesic propagation at multiple cloud densities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/combined_polynomial_ring3/best_model.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--train_config', type=str,
                        default='models/configs/combined_polynomial_ring3.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--dataset_config', type=str,
                        default='DataSets/configs/polynomial/combined_polynomial_all.yaml',
                        help='Path to dataset config YAML (attributes, mask_constant, …)')
    parser.add_argument('--gaussian_dir', type=str,
                        default='TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_04/light_0/output',
                        help='Gaussian splat output directory')
    parser.add_argument('--gt_subpath', type=str,
                        default='geodesic_distance/gt_geodesic.npz',
                        help='Relative path inside gaussian_dir to GT .npz')

    # Neighborhood
    parser.add_argument('--ring', type=int, default=3, help='Ring level')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Ring-1 kNN k')

    # Density testing
    parser.add_argument('--fps_targets', type=int, nargs='+',
                        default=[20000, 10000, 5000, 2000],
                        help='FPS target densities to test')
    parser.add_argument('--include_full', action='store_true',
                        help='Also test on the full (no FPS) cloud')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for FM propagation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detect if omitted.')
    return parser.parse_args()


def run_propagation(positions, scales, rotations, opacities,
                    fps_indices, gt_all_dists, gt_source_indices,
                    mh, ib_factory, ring, n_neighbors, batch_size, device, label=""):
    """Run propagation on the given point cloud and return metrics."""
    N = len(positions)

    # Build input builder (ring computation included)
    t0 = time.perf_counter()
    ib = ib_factory(positions, scales, rotations, opacities, device)
    ring1_nbrs = ib.ring1_neighbors
    mean_nn_dist = ib.normalization_factor
    t_knn = time.perf_counter() - t0

    # Build propagator
    prop = FastMarchingPropagator(
        model_handler=mh, input_builder=ib,
        ring1_neighbors=ring1_nbrs,
        ring_neighbors=ib.ring_neighbors,
        ring=ring, verbose=False,
    )

    # Pick source that exists in both GT and our point set
    fps_set = set(fps_indices.tolist())
    best_gt_row = None
    best_local_idx = None
    for gt_row, gt_src in enumerate(gt_source_indices):
        if int(gt_src) in fps_set:
            best_gt_row = gt_row
            best_local_idx = int(np.where(fps_indices == int(gt_src))[0][0])
            break

    if best_gt_row is None:
        p(f"  [{label}] WARNING: no GT source in point set")
        return None

    source_idx = best_local_idx

    # Propagate
    t0 = time.perf_counter()
    distances = prop.propagate_batch([source_idx], batch_size=batch_size)
    t_prop = time.perf_counter() - t0

    # Map GT to our indices
    gt_dists = gt_all_dists[best_gt_row][fps_indices]

    # Metrics
    valid = np.isfinite(distances) & np.isfinite(gt_dists)
    valid[source_idx] = False
    valid[gt_dists <= 0] = False
    n_valid = int(np.sum(valid))
    pred = distances[valid]
    gt = gt_dists[valid]

    if n_valid == 0:
        p(f"  [{label}] No valid comparison points!")
        return None

    errors = np.abs(pred - gt)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    rel_err = float(np.mean(errors / (gt + 1e-8))) * 100
    corr = float(np.corrcoef(pred, gt)[0, 1]) if n_valid > 1 else 0.0

    gt_pos = gt[gt > 0]
    pred_pos = pred[pred > 0]

    mf = prop.get_model_floor_stats()

    result = {
        'n_points': N,
        'n_valid': n_valid,
        'mae': mae,
        'rmse': rmse,
        'rel_err': rel_err,
        'corr': corr,
        'gt_range': (float(gt_pos.min()) if len(gt_pos) > 0 else 0, float(gt.max())),
        'pred_range': (float(pred_pos.min()) if len(pred_pos) > 0 else 0, float(pred.max())),
        'mean_nn_dist': mean_nn_dist,
        't_knn': t_knn,
        't_prop': t_prop,
        'throughput': N / t_prop if t_prop > 0 else 0,
        'median_err': float(np.median(errors)),
        'p90_err': float(np.percentile(errors, 90)),
        'model_wins': mf['model_wins'],
        'floor_wins': mf['floor_wins'],
    }

    p(f"  [{label}] N={N:,}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
      f"RelErr={rel_err:.1f}%, Corr={corr:.4f}, "
      f"GT=[{result['gt_range'][0]:.3f},{result['gt_range'][1]:.3f}], "
      f"Pred=[{result['pred_range'][0]:.3f},{result['pred_range'][1]:.3f}], "
      f"nn_dist={mean_nn_dist:.5f}, prop={t_prop:.1f}s")

    return result


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    p(f"Device: {device}")

    gaussian_dir = Path(args.gaussian_dir)
    gt_path = gaussian_dir / args.gt_subpath

    # Print config
    p(f"\n{'='*60}")
    p("Configuration")
    p(f"{'='*60}")
    for k, v in vars(args).items():
        p(f"  {k:20s} = {v}")
    p(f"  {'device':20s} = {device}")

    # Load model
    p(f"\n{'='*60}")
    p("Loading model")
    p(f"{'='*60}")
    mh = ModelHandler(model_path=str(args.model_path), device=str(device))
    p(f"  Model config: {mh.config}")

    # Inference transforms config
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    transforms_cfg = train_cfg.get('dataset', {}).get('transforms', None)
    p(f"  Transforms config: {transforms_cfg is not None}")

    # Factory for building GaussianInputBuilder per density
    def ib_factory(positions, scales, rotations, opacities, dev):
        return GaussianInputBuilder(
            positions=positions,
            dataset_config=str(args.dataset_config),
            ring=args.ring,
            scales=scales, rotations=rotations,
            opacities=opacities,
            device=str(dev),
            n_neighbors=args.n_neighbors,
            transforms_config=transforms_cfg,
        )

    # Load full data
    p(f"\n{'='*60}")
    p("Loading Gaussian data")
    p(f"{'='*60}")
    gdata = load_gaussian_data_cpu(gaussian_dir, iteration=None, load_sh_features=False)
    positions_full = gdata.get_xyz()
    scales_full = gdata.get_scaling()
    rotations_full = gdata.get_rotation()
    opacities_full = gdata.get_opacity()
    N_full = len(positions_full)
    p(f"  Full cloud: {N_full:,} Gaussians")

    # Load GT
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_all_dists = gt_data['geodesic_distances']
    gt_source_indices = gt_data['source_gaussian_indices']
    p(f"  GT sources: {len(gt_source_indices)}")

    # Build test configurations
    fps_targets = []
    if args.include_full:
        fps_targets.append(None)
    fps_targets.extend(sorted(args.fps_targets, reverse=True))

    results = {}

    for fps_target in fps_targets:
        label = f"FPS={fps_target}" if fps_target else "FULL"
        p(f"\n{'='*60}")
        p(f"Testing: {label}")
        p(f"{'='*60}")

        if fps_target is not None and fps_target < N_full:
            t0 = time.perf_counter()
            fps_indices = fps_gs(positions_full, fps_target, device=str(device))
            t_fps = time.perf_counter() - t0
            p(f"  FPS to {fps_target}: {t_fps:.2f}s")

            positions = positions_full[fps_indices]
            scales = scales_full[fps_indices]
            rotations = rotations_full[fps_indices]
            opacities = opacities_full[fps_indices]
        else:
            fps_indices = np.arange(N_full)
            positions = positions_full
            scales = scales_full
            rotations = rotations_full
            opacities = opacities_full

        result = run_propagation(
            positions, scales, rotations, opacities,
            fps_indices, gt_all_dists, gt_source_indices,
            mh, ib_factory, args.ring, args.n_neighbors, args.batch_size, device, label=label,
        )

        if result:
            results[label] = result

    # Summary table
    p(f"\n{'='*100}")
    p("SUMMARY")
    p(f"{'='*100}")
    p(f"{'Config':<12} {'N':>7} {'MAE':>8} {'RMSE':>8} {'MedAE':>8} {'P90':>8} "
      f"{'RelErr%':>8} {'Corr':>7} {'nn_dist':>8} {'Prop(s)':>8} {'Tput':>8}")
    p("-" * 100)
    for label, r in results.items():
        p(f"{label:<12} {r['n_points']:>7,} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
          f"{r['median_err']:>8.4f} {r['p90_err']:>8.4f} "
          f"{r['rel_err']:>7.1f}% {r['corr']:>7.4f} "
          f"{r['mean_nn_dist']:>8.5f} {r['t_prop']:>8.1f} {r['throughput']:>8.0f}")


if __name__ == '__main__':
    main()
