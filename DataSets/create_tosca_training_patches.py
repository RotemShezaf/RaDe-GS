#!/usr/bin/env python3
"""
Generate training examples from Gaussian splats for TOSCA shapes.

This script is the TOSCA-specific counterpart of create_gaussian_training_patches.py.
It shares the same overall pipeline and uses the same uniform source sampling
strategy via :func:`generate_training_examples` from
``DataSets.utils.training_patches_helpers``.

The only TOSCA-specific behaviour is the shape/texture extraction from the path
convention produced by ``create_synthetic_colmap_dataset_from_mesh_tosca.py``.

TOSCA PATH CONVENTION
---------------------
After running create_synthetic_colmap_dataset_from_mesh_tosca.py, Gaussian
outputs live under:
  {synth_data_base}/{texture}_texture/{shape}/{colmap_resolution}/light_{id}/

The shape name (e.g. ``cat0``, ``centaur1``) is extracted automatically from
each output path, or you may specify ``--shape`` explicitly for single-source
mode.

USAGE
-----
  # Single shape via command line
  python DataSets/create_tosca_training_patches.py \\
      --gaussian_output TrainData/TOSCA/SyntheticColmapData/colors_texture/cat0/high_res/light_0/output \\
      --output_dir TrainData/datasets/gaussian_patches/tosca_cat0_colors \\
      --num_iterations 1000

  # Multi-output via config (shape auto-detected from each path)
  python DataSets/create_tosca_training_patches.py \\
      --config DataSets/configs/tosca/tosca_cat0_colors.yaml

  # All shapes via combined config
  python DataSets/create_tosca_training_patches.py \\
      --config DataSets/configs/tosca/combined_tosca_all.yaml
"""

import os
import sys
import argparse
import json
import multiprocessing
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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

from GenerateData.utils.load_utils import load_gaussian_data_cpu
from utils.misc import fps_gs
from DataSets.utils.config_utils import (
    load_config,
    get_data_sources,
    get_ring_size_mapping,
    resolve_geodesic_data_path,
    load_geodesic_distances,
    is_multi_source_config,
    is_gaussian_outputs_file,
    resolve_gaussian_outputs,
)
from DataSets.utils.training_patches_helpers import (
    compute_gaussian_normals,
    create_train_example,
    generate_training_examples,
    merge_config_with_args,
    compute_scale_stats,
    GaussianData,
    PatchConfig,
)
from GenerateData.utils.data_generation_utils import get_all_points_nbrs_all_rings


# ===========================================================================
# TOSCA path utilities
# ===========================================================================

def extract_tosca_shape_from_path(gaussian_output_path: Path) -> Optional[str]:
    """Infer the TOSCA shape name from a Gaussian output path.

    Looks for a path segment that is a known TOSCA shape directory name.
    The expected directory layout is::

        …/{texture}_texture/{shape}/{resolution}/light_{id}/output

    so the shape appears **two levels above** ``light_*/`` (or three above
    ``output/``).

    Falls back to the first segment that starts with a lowercase letter and
    does NOT end with ``_texture``.

    Args:
        gaussian_output_path: The Gaussian output directory.

    Returns:
        The shape name (e.g. ``cat0``), or *None* if it cannot be inferred.
    """
    parts = list(Path(gaussian_output_path).parts)

    # Heuristic 1: look for 'light_*' and take two levels up
    for i, part in enumerate(parts):
        if part.startswith("light_") and i >= 2:
            candidate = parts[i - 1]          # resolution (e.g. high_res)
            shape_candidate = parts[i - 2]    # shape (e.g. cat0)
            if not shape_candidate.endswith("_texture"):
                return shape_candidate

    # Heuristic 2: first lowercase segment before 'output' that is not a
    # resolution tag and not *_texture
    for i, part in enumerate(parts):
        if (part[0:1].islower()
            and not part.endswith("_texture")
            and part not in {"output", "high_res", "low_res", "med_res"}
            and not part.startswith("light_")):
            return part

    return None


# ===========================================================================
# parse_args
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TOSCA training patches from Gaussian splats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override config values.",
    )
    parser.add_argument(
        "--gaussian_output", type=str, default=None,
        help="Path to Gaussian output folder, or .txt listing multiple paths.",
    )
    parser.add_argument(
        "--shape", type=str, default=None,
        help="TOSCA shape name (e.g. cat0). Auto-inferred from path if not given.",
    )
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Training iteration to use (default: highest available).",
    )
    parser.add_argument(
        "--geodesic_data", type=str, default=None,
        help="Explicit path to precomputed geodesic NPZ. "
             "If not set, auto-detected from "
             "{gaussian_output}/geodesic_distance/gt_geodesic.npz",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save training examples.",
    )
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--num_sources", type=int, default=1)
    parser.add_argument("--num_train_points", type=int, default=100)
    parser.add_argument("--use_mahalanobis", action="store_true")
    parser.add_argument("--use_r1_min_val", action="store_true")
    parser.add_argument("--n_neighbors", type=int, default=22)
    parser.add_argument("--adaptive_target_ring", type=int, default=None,
                        help="Ring level to optimize with adaptive kNN")
    parser.add_argument("--adaptive_target_neighbors", type=int, default=None,
                        help="Desired ring-k neighbor count for adaptive kNN")
    parser.add_argument("--adaptive_k_boost", type=int, default=20,
                        help="Maximum boosted k for deficient points in adaptive kNN mode")
    parser.add_argument("--adaptive_max_mean_cut", type=float, default=5.0,
                        help="Stop adaptive binary search when mean cut <= this value")
    parser.add_argument("--adaptive_max_steps", type=int, default=5,
                        help="Maximum binary-search steps in adaptive kNN mode")
    parser.add_argument("--rings", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--attributes", type=str, nargs="+", default=["xyz"],
        choices=["xyz", "scale", "opacity", "rotation", "sh",
                 "normals", "euclidean_distances"],
    )
    parser.add_argument(
        "--add_normals", action="store_true",
        help="[DEPRECATED] Use --attributes normals instead.",
    )
    parser.add_argument(
        "--add_euclidean_distance", action="store_true",
        help="[DEPRECATED] Use --attributes euclidean_distances instead.",
    )
    parser.add_argument("--nn_mean", type=float, default=1.0)
    parser.add_argument("--normalize_per_patch", action="store_true")
    parser.add_argument(
        "--mask_attributes", type=str, nargs="+", default=[],
        choices=["xyz", "opacity", "rotation", "scale", "sh",
                 "normals", "euclidean_distances", "geodesic_distance"],
    )
    parser.add_argument("--mask_constant", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_output_workers", type=int, default=1,
        help="Parallel workers for processing multiple Gaussian outputs.",
    )
    parser.add_argument(
        "--fps_target", type=int, default=0,
        help="FPS downsample the Gaussian cloud to this many points before "
             "computing neighborhoods and generating patches. 0 = no downsampling.",
    )
    parser.add_argument(
        "--fps_attributes", type=str, nargs="+", default=["xyz"],
        choices=["xyz", "scale", "opacity", "rotation", "sh", "normals"],
        help="Attributes to use for FPS distance metric (default: xyz only).",
    )

    return parser.parse_args()


# ===========================================================================
# save_metadata
# ===========================================================================

def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_gaussians: int,
    scale_stats: Dict,
    examples_info: Dict,
    surface_name: Optional[str] = None,
    texture: Optional[str] = None,
) -> None:
    """Save JSON + README metadata for a TOSCA training-patch run."""
    metadata = {
        'creation_info': {
            'timestamp': datetime.now().isoformat(),
            'script': 'create_tosca_training_patches.py',
            'description': 'TOSCA training examples for geodesic distance '
                           'learning on Gaussian splats',
        },
        'surface': {
            'name': surface_name if surface_name else 'Unknown',
            'type': 'TOSCA Gaussian Splatting Reconstruction',
            'texture': texture if texture else 'Not specified',
        },
        'gaussian_data': {
            'source_folder': str(getattr(args, 'gaussian_output', 'Unknown')),
            'iteration': (args.iteration
                          if hasattr(args, 'iteration') and args.iteration
                          else 'auto (highest)'),
            'num_gaussians': num_gaussians,
            'scale_statistics': scale_stats,
        },
        'generation_parameters': {
            'num_iterations': args.num_iterations,
            'num_sources_per_iteration': args.num_sources,
            'num_train_points_per_iteration': args.num_train_points,
            'rings': args.rings,
            'seed': args.seed,
        },
        'neighborhood': {
            'method': 'Mahalanobis' if args.use_mahalanobis else 'Euclidean',
            'n_neighbors_ring1': args.n_neighbors,
            'adaptive_target_ring': getattr(args, 'adaptive_target_ring', None),
            'adaptive_target_neighbors': getattr(args, 'adaptive_target_neighbors', None),
            'adaptive_k_boost': getattr(args, 'adaptive_k_boost', 20),
            'adaptive_max_mean_cut': getattr(args, 'adaptive_max_mean_cut', 5.0),
            'adaptive_max_steps': getattr(args, 'adaptive_max_steps', 5),
        },
        'features': {
            'attributes': args.attributes,
            'use_r1_min_val': args.use_r1_min_val,
            'nn_mean': getattr(args, 'nn_mean', 1.0),
            'mask_constant': args.mask_constant,
            'normalize_per_patch': getattr(args, 'normalize_per_patch', False),
        },
        'fps': {
            'fps_target': getattr(args, 'fps_target', 0),
            'fps_attributes': getattr(args, 'fps_attributes', ['xyz']),
        },
        'output_files': examples_info,
    }

    metadata_path = output_dir / 'generation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOSCA GAUSSIAN TRAINING EXAMPLES METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {metadata['creation_info']['timestamp']}\n")
        f.write(f"Script:    {metadata['creation_info']['script']}\n\n")
        f.write("SURFACE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Name: {metadata['surface']['name']}\n")
        f.write(f"  Type: {metadata['surface']['type']}\n")
        f.write(f"  Texture: {metadata['surface']['texture']}\n\n")
        f.write("GAUSSIAN DATA:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Source: {metadata['gaussian_data']['source_folder']}\n")
        f.write(f"  Iteration: {metadata['gaussian_data']['iteration']}\n")
        f.write(f"  Number of Gaussians: "
                f"{metadata['gaussian_data']['num_gaussians']:,}\n")
        f.write(f"  Mean scale: {scale_stats['mean']:.6f}\n")
        f.write(f"  Scale range: "
                f"[{scale_stats['min']:.6f}, {scale_stats['max']:.6f}]\n\n")
        f.write("GENERATION PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Iterations: {args.num_iterations}\n")
        f.write(f"  Sources per iteration: {args.num_sources}\n")
        f.write(f"  Train points per iteration: {args.num_train_points}\n")
        f.write(f"  Rings: {args.rings}\n")
        f.write(f"  Random seed: {args.seed}\n\n")
        f.write("NEIGHBORHOOD COMPUTATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Method: "
                f"{'Mahalanobis' if args.use_mahalanobis else 'Euclidean'}\n")
        f.write(f"  Ring-1 neighbors: {args.n_neighbors}\n\n")
        f.write("FEATURES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Attributes: {args.attributes}\n")
        f.write(f"  Use R1 min val: {args.use_r1_min_val}\n")
        f.write(f"  NN mean: {getattr(args, 'nn_mean', 1.0)}\n\n")
        fps_target = getattr(args, 'fps_target', 0)
        if fps_target and fps_target > 0:
            f.write("FPS DOWNSAMPLING:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Target points: {fps_target}\n")
            f.write(f"  FPS attributes: "
                    f"{getattr(args, 'fps_attributes', ['xyz'])}\n\n")
        f.write("OUTPUT FILES:\n")
        f.write("-" * 40 + "\n")
        for ring_key, info in examples_info.items():
            f.write(f"  {ring_key}:\n")
            f.write(f"    File:     {info['filename']}\n")
            f.write(f"    Examples: {info['num_examples']:,}\n")
            f.write(f"    Shape:    {info['shape']}\n")
            f.write(f"    Size:     {info['size_mb']:.2f} MB\n")
            ofs = info.get('outlier_filtering', {})
            if ofs and ofs.get('total_patches', 0) > 0:
                pct = ofs.get('outlier_patch_fraction', 0) * 100
                f.write(f"    Outlier filtering:\n")
                f.write(f"      Patches with outliers: "
                        f"{ofs['patches_with_outliers']:,} / {ofs['total_patches']:,}"
                        f" ({pct:.1f}%)\n")
                f.write(f"      Neighbors removed:     {ofs['total_neighbors_removed']:,}\n")
            f.write("\n")
    print(f"README saved to: {readme_path}")


# ===========================================================================
# process_single_gaussian_output
# ===========================================================================

def process_single_gaussian_output(
    gaussian_output: Path,
    args: argparse.Namespace,
    geodesic_data_path: Optional[str] = None,
    iteration: Optional[int] = None,
) -> Tuple:
    """Load data from a single Gaussian output and compute neighborhoods.

    Returns all the data needed for generating training examples.

    Args:
        gaussian_output: Path to Gaussian splatting output folder.
        args:            Parsed CLI arguments.
        geodesic_data_path: Optional explicit geodesic data path.
        iteration:       Specific iteration to load.

    Returns:
        Tuple of (geodesic_data, positions, scales, rotations, opacities,
                  normals, sh_features, ring_nbrs_dict, ring1_nbrs,
                  normalization_factor, per_point_dist,
                  shape, texture, scale_stats)
    """
    # Load Gaussian data (CPU mode – no CUDA required)
    print(f"\nLoading Gaussian data from: {gaussian_output}")
    load_sh = "sh" in args.attributes
    gaussian_data = load_gaussian_data_cpu(
        gaussian_output, iteration, load_sh_features=load_sh
    )

    positions  = gaussian_data.get_xyz()
    scales     = gaussian_data.get_scaling()
    rotations  = gaussian_data.get_rotation()
    opacities  = gaussian_data.get_opacity()

    # Infer shape name from path, then fall back to args.shape
    shape = extract_tosca_shape_from_path(gaussian_output)
    if shape is None:
        shape = getattr(args, 'shape', None)
    if shape is None:
        raise ValueError(
            f"Cannot infer TOSCA shape from path: {gaussian_output}. "
            f"Please specify --shape explicitly."
        )

    # Infer texture from path segment ending in '_texture'
    texture = None
    for part in Path(gaussian_output).parts:
        if part.endswith("_texture"):
            texture = part
            break

    print(f"  Shape:    {shape}")
    print(f"  Texture:  {texture or 'Not specified'}")
    print(f"  Gaussians: {len(positions)}")

    # Optional: normals
    normals = None
    if "normals" in args.attributes:
        print("  Computing normals...")
        normals = compute_gaussian_normals(scales, rotations)

    # Optional: SH features
    sh_features = None
    if "sh" in args.attributes:
        sh_features = gaussian_data.get_features()
        if sh_features is not None:
            sh_features = sh_features.reshape(sh_features.shape[0], -1)

    # Load geodesic distances
    geodesic_path = resolve_geodesic_data_path(gaussian_output, geodesic_data_path)
    geodesic_data = load_geodesic_distances(geodesic_path)

    # ── Optional FPS downsampling ───────────────────────────────────
    fps_target = getattr(args, 'fps_target', 0) or 0
    if fps_target > 0 and fps_target < len(positions):
        fps_attrs = getattr(args, 'fps_attributes', ['xyz'])
        # Protect source Gaussian indices so they are never dropped by FPS
        src_gauss_idx = geodesic_data.get('source_gaussian_indices')
        protected = src_gauss_idx if src_gauss_idx is not None else None
        n_protected = len(protected) if protected is not None else 0
        print(f"\nFPS downsampling: {len(positions)} -> {fps_target} "
              f"(attrs={fps_attrs}, {n_protected} source points protected)")
        fps_idx = fps_gs(
            positions, fps_target, attributes=fps_attrs,
            scales=scales, rotations=rotations, opacities=opacities,
            sh_features=sh_features, normals=normals,
            protected_indices=protected,
            device='cuda',
        )
        # Remap geodesic data to the FPS subset (all sources guaranteed present)
        old_to_new = {int(old): new for new, old in enumerate(fps_idx)}
        geo_dists = geodesic_data['geodesic_distances'][:, fps_idx]
        if src_gauss_idx is not None:
            new_src_gauss = np.array(
                [old_to_new[int(s)] for s in src_gauss_idx]
            )
        else:
            new_src_gauss = None
        geodesic_data = {
            'gaussian_positions': positions[fps_idx],
            'source_indices': geodesic_data.get('source_indices'),
            'source_positions': geodesic_data.get('source_positions'),
            'geodesic_distances': geo_dists,
            'source_gaussian_indices': new_src_gauss,
        }
        # Slice arrays
        positions  = positions[fps_idx]
        scales     = scales[fps_idx]
        rotations  = rotations[fps_idx]
        opacities  = opacities[fps_idx]
        if normals is not None:
            normals = normals[fps_idx]
        if sh_features is not None:
            sh_features = sh_features[fps_idx]
        print(f"  After FPS: {len(positions)} Gaussians, "
              f"{len(geo_dists)} GT sources (all kept)")
    elif fps_target > 0:
        print(f"  FPS target {fps_target} >= cloud size "
              f"{len(positions)}, skipping")

    scale_stats = {
        'mean': float(scales.mean()),
        'min':  float(scales.min()),
        'max':  float(scales.max()),
        'std':  float(scales.std()),
    }

    # Compute KNN neighborhoods
    print("\nComputing neighborhood rings...")
    print(f"  Method: {'Mahalanobis' if args.use_mahalanobis else 'Euclidean'}")
    if getattr(args, 'adaptive_target_ring', None) is not None:
        print(f"  Adaptive kNN: target ring-{args.adaptive_target_ring} ≥ {args.adaptive_target_neighbors}, k_boost={args.adaptive_k_boost}")
    ring1_nbrs, ring2_nbrs, ring3_nbrs, ring4_nbrs, \
    mean_dist, per_point_dist = get_all_points_nbrs_all_rings(
        positions,
        use_mahalanobis=args.use_mahalanobis,
        gaussian_scales=scales   if args.use_mahalanobis else None,
        gaussian_rotations=rotations if args.use_mahalanobis else None,
        n_neighbors_ring1=args.n_neighbors,
        adaptive_target_ring=getattr(args, 'adaptive_target_ring', None),
        adaptive_target_neighbors=getattr(args, 'adaptive_target_neighbors', None),
        adaptive_k_boost=getattr(args, 'adaptive_k_boost', 20),
        adaptive_max_mean_cut=getattr(args, 'adaptive_max_mean_cut', 5.0),
        adaptive_max_steps=getattr(args, 'adaptive_max_steps', 5),
    )
    ring_nbrs_dict = {
        1: ring1_nbrs, 2: ring2_nbrs,
        3: ring3_nbrs, 4: ring4_nbrs,
    }

    return (geodesic_data, positions, scales, rotations, opacities, normals,
            sh_features, ring_nbrs_dict, ring1_nbrs, mean_dist, per_point_dist,
            shape, texture, scale_stats)


# ===========================================================================
# Worker for parallel multi-output processing
# ===========================================================================

def _process_output_worker(worker_args):
    """Multiprocessing worker: process one Gaussian output end-to-end."""
    (gout_path_str, args, geodesic_data_path, iteration,
     rings, gout_idx, total_outputs) = worker_args

    gout_path = Path(gout_path_str)
    print(f"\n--- [Worker {gout_idx+1}/{total_outputs}] {gout_path} ---")

    (geodesic_data, positions, scales, rotations, opacities, normals,
     sh_features, ring_nbrs_dict, ring1_nbrs, normalization_factor,
     per_point_dist, shape, texture, scale_stats) = \
        process_single_gaussian_output(
            gout_path, args, geodesic_data_path, iteration
        )

    gdata = GaussianData(
        positions=positions, scales=scales, rotations=rotations,
        opacities=opacities, normals=normals, sh_features=sh_features,
        per_point_nn_distances=per_point_dist,
    )
    examples_per_ring = {}
    outlier_stats_per_ring = {}
    for ring in rings:
        print(f"\n  Ring {ring} (output {gout_idx+1}/{total_outputs}):")
        ring_stats: Dict = {}
        pcfg = PatchConfig(
            ring=ring, normalization_factor=normalization_factor,
            nn_mean=args.nn_mean, attributes=args.attributes,
            use_mahalanobis=args.use_mahalanobis, use_r1_min_val=args.use_r1_min_val,
            mask_attributes=args.mask_attributes, mask_constant=args.mask_constant,
            ring_size_mapping=getattr(args, 'ring_size_mapping', None),
            normalize_per_patch=args.normalize_per_patch,
        )
        examples = generate_training_examples(
            geodesic_data, ring_nbrs_dict, ring1_nbrs,
            args.num_iterations, args.num_sources, args.num_train_points,
            gaussian_data=gdata, patch_config=pcfg,
            seed=args.seed + ring + gout_idx * 1000,
            num_workers=1,  # sequential – avoid nested pools
            _stats_out=ring_stats,
        )
        examples_per_ring[ring] = examples
        outlier_stats_per_ring[ring] = ring_stats
        print(f"    Generated {len(examples)} examples")

    print(f"\n--- [Worker {gout_idx+1}/{total_outputs}] Done: {shape} ---")

    return {
        'gout_idx':      gout_idx,
        'shape':         shape,
        'texture':       texture,
        'num_positions': len(positions),
        'scale_stats':   scale_stats,
        'examples_per_ring': examples_per_ring,
        'outlier_stats_per_ring': outlier_stats_per_ring,
    }


# ===========================================================================
# process_single_source
# ===========================================================================

def process_single_source(
    source: Dict,
    args: argparse.Namespace,
    config: Optional[Dict] = None,
) -> Dict:
    """Process one data source and produce training patches.

    Args:
        source: Source dict (gaussian_output, output_dir, …).
        args:   Parsed command-line / config args.
        config: Full config dict (optional).

    Returns:
        Result summary dictionary.
    """
    print("\n" + "=" * 80)
    print(f"Processing source: {source['name']}")
    print("=" * 80)

    gaussian_output_raw = source['gaussian_output']
    output_dir          = Path(source['output_dir'])
    iteration           = source.get('iteration') or args.iteration
    geodesic_data_path  = source.get('geodesic_data')

    output_dir.mkdir(parents=True, exist_ok=True)

    gaussian_output_paths = resolve_gaussian_outputs(gaussian_output_raw)
    is_multi_output = len(gaussian_output_paths) > 1

    if is_multi_output:
        print(f"\nMulti-output mode: {len(gaussian_output_paths)} Gaussian outputs")
        for i, p in enumerate(gaussian_output_paths):
            print(f"  [{i+1}] {p}")

    all_examples_per_ring = {ring: [] for ring in args.rings}
    all_outlier_stats_per_ring: Dict = {ring: [] for ring in args.rings}
    all_shapes:   List[Optional[str]] = []
    all_textures: List[Optional[str]] = []
    total_gaussians = 0
    combined_scale_stats: Dict[str, List[float]] = {
        'mean': [], 'min': [], 'max': [], 'std': [],
    }

    num_output_workers = max(1, min(
        getattr(args, 'num_output_workers', 1) or 1,
        len(gaussian_output_paths),
    ))

    if num_output_workers > 1 and len(gaussian_output_paths) > 1:
        # ---- parallel -------------------------------------------------------
        print(f"\nParallel processing: {num_output_workers} workers, "
              f"{len(gaussian_output_paths)} outputs")
        worker_args = [
            (str(p), args, geodesic_data_path, iteration,
             args.rings, idx, len(gaussian_output_paths))
            for idx, p in enumerate(gaussian_output_paths)
        ]
        with multiprocessing.Pool(num_output_workers) as pool:
            for result in pool.imap_unordered(
                _process_output_worker, worker_args
            ):
                idx = result['gout_idx']
                all_shapes.append(result['shape'])
                all_textures.append(result['texture'])
                total_gaussians += result['num_positions']
                for k in combined_scale_stats:
                    combined_scale_stats[k].append(result['scale_stats'][k])
                for ring in args.rings:
                    all_examples_per_ring[ring].append(
                        result['examples_per_ring'][ring]
                    )
                    all_outlier_stats_per_ring[ring].append(
                        result.get('outlier_stats_per_ring', {}).get(ring, {})
                    )
                print(f"  Completed [{idx+1}/{len(gaussian_output_paths)}]")
    else:
        # ---- sequential -----------------------------------------------------
        for gout_idx, gout_path_str in enumerate(gaussian_output_paths):
            gout_path = Path(gout_path_str)
            if is_multi_output:
                print(f"\n--- Output [{gout_idx+1}/"
                      f"{len(gaussian_output_paths)}]: {gout_path} ---")

            (geodesic_data, positions, scales, rotations, opacities, normals,
             sh_features, ring_nbrs_dict, ring1_nbrs, normalization_factor,
             per_point_dist, shape, texture, scale_stats) = \
                process_single_gaussian_output(
                    gout_path, args, geodesic_data_path, iteration
                )

            all_shapes.append(shape)
            all_textures.append(texture)
            total_gaussians += len(positions)
            for k in combined_scale_stats:
                combined_scale_stats[k].append(scale_stats[k])

            gdata = GaussianData(
                positions=positions, scales=scales, rotations=rotations,
                opacities=opacities, normals=normals, sh_features=sh_features,
                per_point_nn_distances=per_point_dist,
            )
            for ring in args.rings:
                print(f"\n  Ring {ring}:")
                ring_stats: Dict = {}
                pcfg = PatchConfig(
                    ring=ring, normalization_factor=normalization_factor,
                    nn_mean=args.nn_mean, attributes=args.attributes,
                    use_mahalanobis=args.use_mahalanobis, use_r1_min_val=args.use_r1_min_val,
                    mask_attributes=args.mask_attributes, mask_constant=args.mask_constant,
                    ring_size_mapping=getattr(args, 'ring_size_mapping', None),
                    normalize_per_patch=args.normalize_per_patch,
                )
                examples = generate_training_examples(
                    geodesic_data, ring_nbrs_dict, ring1_nbrs,
                    args.num_iterations, args.num_sources, args.num_train_points,
                    gaussian_data=gdata, patch_config=pcfg,
                    seed=args.seed + ring + gout_idx * 1000,
                    _stats_out=ring_stats,
                )
                all_examples_per_ring[ring].append(examples)
                all_outlier_stats_per_ring[ring].append(ring_stats)
                print(f"    Generated {len(examples)} examples")

    # Aggregate scale statistics
    final_scale_stats = {
        'mean': float(np.mean(combined_scale_stats['mean'])),
        'min':  float(np.min(combined_scale_stats['min'])),
        'max':  float(np.max(combined_scale_stats['max'])),
        'std':  float(np.mean(combined_scale_stats['std'])),
    }

    # Save combined examples for each ring
    examples_info: Dict = {}
    for ring in args.rings:
        print(f"\n--- Ring {ring} (combined) ---")
        combined = np.vstack(all_examples_per_ring[ring])
        out_name = f"gaussian_examples_ring{ring}_n{len(combined)}.npy"
        out_path = output_dir / out_name
        np.save(out_path, combined)
        if is_multi_output:
            per_counts = [len(ex) for ex in all_examples_per_ring[ring]]
            print(f"  Combined {len(combined)} examples "
                  f"from {len(gaussian_output_paths)} outputs "
                  f"(per-output: {per_counts})")
        else:
            print(f"  Saved {len(combined)} examples")
        print(f"  Saved to: {out_path}")
        ring_outlier_stats = all_outlier_stats_per_ring.get(ring, [])
        agg_outlier: Dict = {
            'patches_with_outliers': sum(s.get('patches_with_outliers', 0) for s in ring_outlier_stats),
            'total_neighbors_removed': sum(s.get('total_neighbors_removed', 0) for s in ring_outlier_stats),
            'total_patches': sum(s.get('total_patches', 0) for s in ring_outlier_stats),
        }
        if agg_outlier['total_patches'] > 0:
            agg_outlier['outlier_patch_fraction'] = round(
                agg_outlier['patches_with_outliers'] / agg_outlier['total_patches'], 4
            )
        examples_info[f"ring_{ring}"] = {
            'filename':     out_name,
            'num_examples': len(combined),
            'shape':        str(combined.shape),
            'size_mb':      combined.nbytes / 1e6,
            'attributes':   args.attributes,
            'outlier_filtering': agg_outlier,
        }
        if is_multi_output:
            examples_info[f"ring_{ring}"]['per_output_counts'] = [
                len(ex) for ex in all_examples_per_ring[ring]
            ]

    # Determine final shape/texture labels for metadata
    unique_shapes = list(filter(None, set(all_shapes)))
    if is_multi_output:
        final_shape   = f"TOSCA Combined ({', '.join(unique_shapes)})"
        final_texture = ', '.join(filter(None, set(all_textures))) or None
    else:
        final_shape   = all_shapes[0] if all_shapes else None
        final_texture = all_textures[0] if all_textures else None

    save_metadata(
        output_dir, args, total_gaussians, final_scale_stats, examples_info,
        final_shape, final_texture,
    )

    # Save source config YAML for dataset loading
    source_config = {
        'output_dir':       str(output_dir),
        'gaussian_output':  str(gaussian_output_raw),
        'gaussian_output_paths': ([str(p) for p in gaussian_output_paths]
                                  if is_multi_output else None),
        'geodesic_data':    (str(geodesic_data_path)
                             if geodesic_data_path else None),
        'attributes':       args.attributes,
        'rings':            args.rings,
        'use_mahalanobis':  args.use_mahalanobis,
        'use_r1_min_val':   args.use_r1_min_val,
        'ring_size_mapping': getattr(args, 'ring_size_mapping', None),
        'mask_constant':    args.mask_constant,
        'shape':            final_shape,
        'texture':          final_texture,
        'num_gaussian_outputs': len(gaussian_output_paths),
    }
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(source_config, f, default_flow_style=False)
    print(f"\nSource config saved to: {config_path}")

    return {
        'name':                 source['name'],
        'output_dir':           str(output_dir),
        'num_gaussians':        total_gaussians,
        'num_gaussian_outputs': len(gaussian_output_paths),
        'examples_info':        examples_info,
        'shape':                final_shape,
        'texture':              final_texture,
    }


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    args = parse_args()

    # Load config
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config: {config_path}")
        config = load_config(config_path)
        args = merge_config_with_args(config, args)

    np.random.seed(args.seed)

    # Handle deprecated arguments
    if args.add_normals and "normals" not in args.attributes:
        print("Warning: --add_normals is deprecated. "
              "Use --attributes normals instead.")
        args.attributes.append("normals")
    if args.add_euclidean_distance and "euclidean_distances" not in args.attributes:
        print("Warning: --add_euclidean_distance is deprecated. "
              "Use --attributes euclidean_distances instead.")
        args.attributes.append("euclidean_distances")

    print("=" * 80)
    print("TOSCA Gaussian Training Patch Generation")
    print("=" * 80)
    print(f"Attributes: {', '.join(args.attributes)}")
    print(f"Rings:      {args.rings}")

    if config and is_multi_source_config(config):
        # ---- Multi-source mode ----------------------------------------------
        sources = get_data_sources(config)
        print(f"\nMulti-source mode: {len(sources)} sources")

        all_results = []
        for source in sources:
            result = process_single_source(source, args, config)
            all_results.append(result)

        base_output_dir = Path(
            config.get('output_dir', 'TrainData/datasets/gaussian_patches')
        )
        base_output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'multi_source': True,
            'num_sources':  len(sources),
            'sources':      all_results,
            'shared_config': {
                'attributes':     args.attributes,
                'rings':          args.rings,
                'use_mahalanobis': args.use_mahalanobis,
                'use_r1_min_val': args.use_r1_min_val,
            },
        }
        summary_path = base_output_dir / 'multi_source_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Multi-source generation complete!")
        print(f"Summary: {summary_path}")
        print(f"{'=' * 80}")

    else:
        # ---- Single-source mode ---------------------------------------------
        if args.gaussian_output is None:
            raise ValueError(
                "--gaussian_output is required (via CLI or config file)"
            )
        if args.output_dir is None:
            raise ValueError(
                "--output_dir is required (via CLI or config file)"
            )

        source = {
            'name':            'default',
            'gaussian_output': args.gaussian_output,
            'output_dir':      args.output_dir,
            'geodesic_data':   args.geodesic_data,
            'iteration':       args.iteration,
        }
        process_single_source(source, args, config)

        print(f"\n{'=' * 80}")
        print("Training patch generation complete!")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
