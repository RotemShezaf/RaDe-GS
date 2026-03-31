#!/usr/bin/env python3
"""
Main script for evaluating geodesic distance propagation on Gaussian splats.

This script:
1. Loads a trained geodesic distance prediction model
2. Loads Gaussian splat data
3. Runs Fast Marching propagation from specified source points
4. Optionally compares with ground truth geodesic distances
5. Saves results and metrics

Usage:
    python evaluate_geodesic.py --model_path <path> --gaussian_output <path> --source_indices 0 1 2
    python evaluate_geodesic.py --config <config.yaml>
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
import sys
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.fast_marching import FastMarchingPropagator, create_propagator
from geodesic_propagation.utils.model_handler import ModelHandler
from geodesic_propagation.utils.results_saver import ResultsSaver
from geodesic_propagation.utils.output_path import build_eval_output_dir
from geodesic_propagation.input_builder import GaussianInputBuilder


def load_gaussian_data(
    gaussian_output: Path,
    iteration: Optional[int] = None,
    load_sh: bool = False
) -> Dict:
    """
    Load Gaussian splat data from output folder.
    
    Args:
        gaussian_output: Path to Gaussian output folder
        iteration: Iteration number (None = highest)
        load_sh: Whether to load SH features
        
    Returns:
        Dictionary with positions, scales, rotations, opacities, sh_features
        
    Note on SH features:
        The input_builder and dataset code treat 'sh' as a 3-dimensional
        attribute (DC component only).  ``get_features_dc()`` returns exactly
        those 3 values per Gaussian, reshaped to (N, 3).  If the model was
        trained with higher-order SH bands, you must align sh_features to the
        training convention.  See ``GaussianInputBuilder.sh_dim``.
    """
    from GenerateData.utils.load_utils import load_gaussian_data_cpu
    
    gaussian_data = load_gaussian_data_cpu(
        gaussian_output, 
        iteration, 
        load_sh_features=load_sh
    )
    
    sh_features = None
    if load_sh:
        # get_features_dc() returns the DC component with shape (N, 1, 3);
        # reshape to (N, 3) to match the sh_dim=3 assumption in GaussianInputBuilder.
        dc = gaussian_data.get_features_dc()
        if dc is not None:
            sh_features = dc.reshape(dc.shape[0], -1)  # (N, 3)
    
    return {
        'positions': gaussian_data.get_xyz(),
        'scales': gaussian_data.get_scaling(),
        'rotations': gaussian_data.get_rotation(),
        'opacities': gaussian_data.get_opacity(),
        'sh_features': sh_features,
    }


def load_ground_truth(
    geodesic_data_path: Path,
    source_indices: List[int]
) -> Optional[np.ndarray]:
    """
    Load ground truth geodesic distances if available.
    
    Args:
        geodesic_data_path: Path to geodesic data NPZ file
        source_indices: Indices of source points used
        
    Returns:
        Ground truth distances array or None if not available
    """
    if not geodesic_data_path.exists():
        return None
    
    data = np.load(geodesic_data_path, allow_pickle=True)
    
    if 'geodesic_distances' not in data:
        return None
    
    all_distances = data['geodesic_distances']  # (num_sources, num_points)
    all_source_indices = data.get('source_gaussian_indices', None)
    
    if all_source_indices is None:
        return None
    
    # Find matching source configurations
    # For now, take minimum distance from all matching sources
    source_mask = np.isin(all_source_indices, source_indices)
    if not np.any(source_mask):
        print("Warning: No matching source indices found in ground truth")
        return None
    
    matched_distances = all_distances[source_mask]
    min_distances = np.min(matched_distances, axis=0)
    
    return min_distances


def compute_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics between predicted and ground truth distances.
    
    Args:
        predicted: Predicted geodesic distances
        ground_truth: Ground truth geodesic distances
        
    Returns:
        Dictionary of metrics
    """
    # Valid mask (both finite)
    valid_mask = np.isfinite(predicted) & np.isfinite(ground_truth)
    
    if not np.any(valid_mask):
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'relative_error_pct': float('nan'),
            'max_error': float('nan'),
            'num_valid': 0,
        }
    
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    errors = pred_valid - gt_valid
    abs_errors = np.abs(errors)
    
    # Metrics
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    relative_error = float(np.mean(abs_errors / (gt_valid + 1e-8))) * 100
    max_error = float(np.max(abs_errors))
    
    # Percentile errors
    p50_error = float(np.percentile(abs_errors, 50))
    p90_error = float(np.percentile(abs_errors, 90))
    p99_error = float(np.percentile(abs_errors, 99))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'relative_error_pct': relative_error,
        'max_error': max_error,
        'p50_error': p50_error,
        'p90_error': p90_error,
        'p99_error': p99_error,
        'num_valid': int(np.sum(valid_mask)),
    }


def run_evaluation(
    model_path: Path,
    gaussian_output: Path,
    source_indices: List[int],
    output_dir: Path,
    iteration: Optional[int] = None,
    geodesic_data_path: Optional[Path] = None,
    n_neighbors: int = 16,
    use_mahalanobis: bool = True,
    ring: int = 2,
    batch_size: int = 32,
    device: Optional[str] = None,
    export_ply: bool = True,
    verbose: bool = True,
    refine_passes: int = 0,
    dataset_config: Optional[str] = None,
) -> Dict:
    """
    Run complete geodesic propagation evaluation.
    
    Args:
        model_path: Path to trained model checkpoint
        gaussian_output: Path to Gaussian output folder
        source_indices: Indices of source points
        output_dir: Directory for saving results
        iteration: Gaussian iteration number
        geodesic_data_path: Path to ground truth geodesic data
        n_neighbors: Number of ring-1 neighbors
        use_mahalanobis: Whether to use Mahalanobis distance
        ring: Ring level for neighbor expansion
        batch_size: Batch size for predictions
        device: Device for model
        export_ply: Whether to export PLY visualization
        verbose: Whether to print progress
        
    Returns:
        Dictionary with results and metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Gaussian data
    if verbose:
        print("\n" + "=" * 60)
        print("Loading Gaussian data...")
    
    gaussian_data = load_gaussian_data(gaussian_output, iteration, load_sh=True)
    positions = gaussian_data['positions']
    
    if verbose:
        print(f"Loaded {len(positions)} Gaussians")
    
    # Validate source indices
    num_gaussians = len(positions)
    invalid = [i for i in source_indices if i < 0 or i >= num_gaussians]
    if invalid:
        raise ValueError(
            f"source_indices {invalid} are out of range [0, {num_gaussians}). "
            f"Gaussian data has {num_gaussians} points."
        )
    
    # Create propagator
    if verbose:
        print("\n" + "=" * 60)
        print("Creating propagator...")
    
    # Read KNN parameters from dataset config if available
    ds_knn = {}
    if dataset_config is not None:
        ds_path = Path(dataset_config)
        if ds_path.exists():
            with open(ds_path) as _f:
                ds_knn = yaml.safe_load(_f) or {}
    
    _n_neighbors = ds_knn.get('n_neighbors', n_neighbors)
    _use_mahalanobis = ds_knn.get('use_mahalanobis', use_mahalanobis)
    _adaptive_target_ring = ds_knn.get('adaptive_target_ring', None)
    _adaptive_target_neighbors = ds_knn.get('adaptive_target_neighbors', None)
    _adaptive_k_boost = ds_knn.get('adaptive_k_boost', 20)
    _adaptive_max_mean_cut = ds_knn.get('adaptive_max_mean_cut', 0)
    _adaptive_max_steps = ds_knn.get('adaptive_max_steps', 12)

    propagator = create_propagator(
        model_path=str(model_path),
        gaussian_data=gaussian_data,
        dataset_config=dataset_config,
        n_neighbors=_n_neighbors,
        use_mahalanobis=_use_mahalanobis,
        ring=ring,
        device=device,
        verbose=verbose,
        adaptive_target_ring=_adaptive_target_ring,
        adaptive_target_neighbors=_adaptive_target_neighbors,
        adaptive_k_boost=_adaptive_k_boost,
        adaptive_max_mean_cut=_adaptive_max_mean_cut,
        adaptive_max_steps=_adaptive_max_steps,
    )
    
    # Run propagation
    if verbose:
        print("\n" + "=" * 60)
        print(f"Running Fast Marching propagation from {len(source_indices)} source(s)...")
    
    start_time = time.time()
    
    if batch_size > 1:
        distances = propagator.propagate_batch(
            source_indices, batch_size=batch_size,
            refine_passes=refine_passes,
        )
    else:
        distances = propagator.propagate(source_indices)
    
    propagation_time = time.time() - start_time
    
    if verbose:
        print(f"\nPropagation completed in {propagation_time:.2f} seconds")
    
    # Get propagation stats
    stats = propagator.get_propagation_stats()
    
    # Load ground truth and compute metrics
    metrics = None
    ground_truth = None
    
    if geodesic_data_path is not None:
        if verbose:
            print("\n" + "=" * 60)
            print("Loading ground truth and computing metrics...")
        
        ground_truth = load_ground_truth(Path(geodesic_data_path), source_indices)
        
        if ground_truth is not None:
            metrics = compute_metrics(distances, ground_truth)
            
            if verbose:
                print("\nEvaluation Metrics:")
                print("-" * 40)
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
    
    # Save results
    if verbose:
        print("\n" + "=" * 60)
        print("Saving results...")
    
    saver = ResultsSaver(output_dir)
    
    # Prepare metadata
    metadata = {
        'model_path': str(model_path),
        'gaussian_output': str(gaussian_output),
        'iteration': iteration,
        'n_neighbors': n_neighbors,
        'use_mahalanobis': use_mahalanobis,
        'ring': ring,
        'batch_size': batch_size,
        'propagation_time_seconds': propagation_time,
        'propagation_stats': stats,
    }
    
    if metrics is not None:
        # Save evaluation results
        saver.save_evaluation_results(
            predicted_distances=distances,
            ground_truth_distances=ground_truth,
            source_indices=source_indices,
            metrics=metrics,
            metadata=metadata,
            name='evaluation_results'
        )
        
        # Generate summary report
        saver.generate_summary_report('evaluation_results')
    else:
        # Save propagation results only
        saver.save_propagation_results(
            distances=distances,
            source_indices=source_indices,
            positions=positions,
            metadata=metadata,
            name='propagation_results'
        )
    
    # Export PLY for visualization
    if export_ply:
        saver.export_to_ply(
            positions=positions,
            distances=distances,
            output_name='geodesic_visualization'
        )
    
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print(f"Results saved to: {output_dir}")
    
    return {
        'distances': distances,
        'ground_truth': ground_truth,
        'metrics': metrics,
        'stats': stats,
        'propagation_time': propagation_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate geodesic distance propagation on Gaussian splats'
    )
    
    # Core arguments (required unless supplied via --config or --train_config)
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='Path to trained model checkpoint. If omitted, resolved from --train_config.'
    )
    parser.add_argument(
        '--train_config', type=str, default=None,
        help='Path to training config YAML (auto-resolves model_path and ring)'
    )
    parser.add_argument(
        '--gaussian_output', type=str,
        default='TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_04/light_4/output',
        help='Path to Gaussian output folder'
    )
    parser.add_argument(
        '--source_indices', type=int, nargs='+', default=None,
        help='Indices of source points (auto-resolved from GT data if omitted)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir', type=str, default='./geodesic_results',
        help='Directory for saving results'
    )
    parser.add_argument(
        '--iteration', type=int, default=None,
        help='Gaussian iteration number (default: highest)'
    )
    parser.add_argument(
        '--geodesic_data', type=str, default=None,
        help='Path to ground truth geodesic data NPZ'
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=16,
        help='Number of ring-1 neighbors'
    )
    parser.add_argument(
        '--use_mahalanobis', action='store_true',
        help='Use Mahalanobis distance for neighbors (default: True; use --no_mahalanobis to disable)'
    )
    parser.add_argument(
        '--no_mahalanobis', action='store_true',
        help='Use Euclidean distance instead of Mahalanobis'
    )
    parser.add_argument(
        '--ring', type=int, default=2, choices=[1, 2, 3, 4],
        help='Ring level for neighbor expansion'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for predictions'
    )
    parser.add_argument(
        '--refine_passes', type=int, default=0,
        help='Number of post-FM iterative refinement passes (0 = disabled)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--no_ply', action='store_true',
        help='Skip PLY export'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override args with config values (command line takes precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Auto-resolve from train_config
    _dataset_config = None
    if args.train_config:
        with open(args.train_config, 'r') as f:
            train_cfg = yaml.safe_load(f)
        if args.model_path is None:
            save_dir = train_cfg.get('infrastructure', {}).get('save_dir', '')
            args.model_path = str(Path(save_dir) / 'best_model.pth')
            print(f'    [auto-resolve] model_path = {args.model_path}')
        ds_cfg = train_cfg.get('dataset', {})
        if args.ring == 2 and ds_cfg.get('ring'):
            args.ring = ds_cfg['ring']
            print(f'    [auto-resolve] ring = {args.ring}')
        if ds_cfg.get('dataset_config'):
            _dataset_config = ds_cfg['dataset_config']
            print(f'    [auto-resolve] dataset_config = {_dataset_config}')
    
    # Auto-resolve source_indices from GT geodesic data
    if args.source_indices is None:
        gt_dir = Path(args.gaussian_output) / 'geodesic_distance'
        gt_npz = gt_dir / 'gt_geodesic.npz'
        if gt_npz.exists():
            gt_data = np.load(gt_npz, allow_pickle=True)
            src_idxs = gt_data.get('source_gaussian_indices', None)
            if src_idxs is not None and len(src_idxs) > 0:
                args.source_indices = [int(src_idxs[0])]
                args.geodesic_data = str(gt_npz)
                print(f'    [auto-resolve] source_indices = {args.source_indices} (from {gt_npz})')
    
    # Validate that essential arguments are present
    missing = []
    if not args.model_path:
        missing.append('--model_path or --train_config')
    if not args.gaussian_output:
        missing.append('--gaussian_output')
    if not args.source_indices:
        missing.append('--source_indices')
    if missing:
        parser.error(
            f"The following arguments are required (via CLI, --config, or --train_config): "
            f"{', '.join(missing)}"
        )
    
    # Handle Mahalanobis flag
    use_mahalanobis = not args.no_mahalanobis
    
    # Build structured output directory from model + gaussian paths
    output_dir = build_eval_output_dir(
        args.output_dir, args.model_path, args.gaussian_output
    )
    
    # Run evaluation
    results = run_evaluation(
        model_path=Path(args.model_path),
        gaussian_output=Path(args.gaussian_output),
        source_indices=args.source_indices,
        output_dir=output_dir,
        iteration=args.iteration,
        geodesic_data_path=Path(args.geodesic_data) if args.geodesic_data else None,
        n_neighbors=args.n_neighbors,
        use_mahalanobis=use_mahalanobis,
        ring=args.ring,
        batch_size=args.batch_size,
        device=args.device,
        export_ply=not args.no_ply,
        verbose=not args.quiet,
        refine_passes=args.refine_passes,
        dataset_config=_dataset_config,
    )
    
    # Print final metrics
    if results['metrics']:
        print("\nFinal Metrics:")
        print("=" * 40)
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
