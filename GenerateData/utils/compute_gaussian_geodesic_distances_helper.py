#!/usr/bin/env python3
"""
Helper functions for computing geodesic distances on Gaussian splats.

This module contains all the helper functions used by compute_gaussian_geodesic_distances.py
to generate ground truth geodesic distances on Gaussian splatting reconstructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
import json
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Ensure project root is in sys.path for module imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from distance.dist import exact_geodesic_via_vtp_vertex_distance, geodesic_via_fmm_vertex_distance
from utils.geodesic_utils import compute_exact_geodesic
from GenerateData.GenerateRawPolynomialMesh import generate_surface_mesh


def generate_source_mesh_and_map(
    surface_type: str,
    source_mesh_resolution: int,
    gaussian_positions: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a low-resolution source mesh and map its vertices to nearest Gaussians.

    The source mesh is generated at the requested resolution and each source
    vertex is matched to the nearest Gaussian by Euclidean distance.  The
    caller is responsible for subsequently finding the GT mesh vertex that
    corresponds to each source Gaussian (e.g. via find_closest_mesh_vertices).

    Args:
        surface_type: Surface type (e.g., 'Paraboloid', 'Saddle', 'HyperbolicParaboloid')
        source_mesh_resolution: Resolution for source mesh (NxN grid)
        gaussian_positions: (G, 3) array of Gaussian center positions
        x_range: (min, max) x range for source mesh generation
        y_range: (min, max) y range for source mesh generation

    Returns:
        Tuple of:
        - source_gaussian_indices: (S,) array of Gaussian indices nearest to each source
        - source_positions: (S, 3) array of source mesh vertex positions
    """
    print(f"\n{'='*80}")
    print(f"Generating Source Mesh")
    print(f"{'='*80}")
    print(f"  Surface type: {surface_type}")
    print(f"  Source mesh resolution: {source_mesh_resolution}x{source_mesh_resolution}")
    print(f"  X range: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
    print(f"  Y range: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
    
    # Generate source mesh using the same parametric definition
    source_mesh, source_normals, arc_length = generate_surface_mesh(
        surface_type=surface_type,
        nx=source_mesh_resolution,
        ny=source_mesh_resolution,
        x_range=x_range,
        y_range=y_range,
        adaptive=False
    )
    
    source_positions = np.asarray(source_mesh.vertices)
    print(f"  Generated {len(source_positions)} source vertices")
    print(f"  Arc length resolution: {arc_length:.6f}")
    
    # Map source positions to nearest Gaussians
    print(f"\n  Mapping sources to nearest Gaussians...")
    tree = KDTree(gaussian_positions)
    distances, source_gaussian_indices = tree.query(source_positions)
    
    print(f"  Mapping statistics:")
    print(f"    Mean distance to nearest Gaussian: {distances.mean():.6f}")
    print(f"    Max distance to nearest Gaussian: {distances.max():.6f}")
    print(f"    Min distance to nearest Gaussian: {distances.min():.6f}")
    
    if distances.max() > 0.1:
        print(f"\n  Warning: Some sources are far from any Gaussian (max = {distances.max():.6f})")
        print(f"           Consider using a denser source mesh or checking alignment.")
    
    return source_gaussian_indices, source_positions


def _compute_batch_source_geodesic(
    batch_source_indices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    geodesic_method: str = 'vtp',
    partial_save_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Compute geodesic distances for a batch of sources to all vertices.
    Helper function for parallel processing.

    When *partial_save_dir* is provided, results are cached to disk so
    that a subsequent call with the same source indices can skip
    re-computation.  This provides crash-resilience for long-running
    geodesic computations.

    Args:
        batch_source_indices: Array of source vertex indices for this batch
        vertices: (N, 3) array of mesh vertex positions
        faces: (M, 3) array of triangle indices
        geodesic_method: 'vtp' for exact VTP, 'mmp' for MMP via pygeodesic,
                         'fmm' for fast marching method
        partial_save_dir: If set, directory to cache per-batch mesh-level
                          geodesic results.

    Returns:
        (batch_size, N) array of geodesic distances from batch sources to all vertices
    """
    # -- Try to load cached result ----------------------------------------
    save_path = None
    if partial_save_dir is not None:
        save_dir = Path(partial_save_dir)
        save_path = save_dir / f"mesh_batch_{int(batch_source_indices[0])}_{int(batch_source_indices[-1])}.npz"
        if save_path.exists():
            try:
                cached = np.load(save_path)
                if (cached['source_indices'].shape == batch_source_indices.shape
                        and np.array_equal(cached['source_indices'], batch_source_indices)):
                    return cached['distances']
            except Exception:
                pass  # corrupted cache; recompute

    try:
        if geodesic_method == 'mmp':
            distances = compute_exact_geodesic(
                vertices=vertices,
                faces=faces,
                sources_id=batch_source_indices,
                sources_are_disjoint=True
            )
        elif geodesic_method == 'fmm':
            distances = geodesic_via_fmm_vertex_distance(
                v=vertices,
                f=faces,
                src_vi=batch_source_indices.tolist(),
                sources_are_disjoint=True
            )
        else:
            # Use VTP for batch computation (efficient)
            distances = exact_geodesic_via_vtp_vertex_distance(
                v=vertices,
                f=faces,
                src_vi=batch_source_indices.tolist(),
                sources_are_disjoint=True
            )
            
            # Check for problematic results and fix with MMP
            valid_mask = distances.max(axis=1) > 10000
            if valid_mask.any():
                problematic_indices = batch_source_indices[valid_mask]
                distances_fix = compute_exact_geodesic(
                    vertices=vertices,
                    faces=faces,
                    sources_id=problematic_indices,
                    sources_are_disjoint=True
                )
                distances[valid_mask, :] = distances_fix

        # -- Save to cache ------------------------------------------------
        if save_path is not None:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    save_path,
                    source_indices=batch_source_indices,
                    distances=distances,
                )
            except Exception as e:
                print(f"Warning: Failed to save batch cache to {save_path}: {e}")

        return distances
    except Exception as e:
        print(f"Warning: Error computing geodesic for batch: {e}")
        # Return infinity as fallback
        return np.full((len(batch_source_indices), len(vertices)), np.inf, dtype=np.float32)


def compute_geodesic_distances_for_sources(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
    geodesic_method: str = 'vtp',
    partial_save_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Compute geodesic distances from each source to all vertices.
    Supports parallel processing by splitting sources into batches across multiple processes.
    
    Args:
        vertices: (N, 3) array of mesh vertex positions
        faces: (M, 3) array of triangle indices
        source_indices: (S,) array of source vertex indices
        verbose: Whether to show detailed progress
        n_jobs: Number of parallel jobs. None = sequential, -1 = use all CPUs, 
                1 = sequential, >1 = specific number of processes.
                Sources are split into batches for each process.
        geodesic_method: 'vtp' for exact VTP (requires manifold mesh),
                         'mmp' for MMP via pygeodesic (works on non-manifold meshes),
                         'fmm' for fast marching method.
        partial_save_dir: If set, directory to cache per-batch mesh-level
                          geodesic results for crash-resilience.  Batches
                          whose results already exist on disk are skipped.
    
    Returns:
        (S, N) array of geodesic distances from each source to all vertices
    """
    method_labels = {'fmm': 'FMM (Fast Marching)', 'mmp': 'MMP (pygeodesic)', 'vtp': 'MMP Algorithm (VTP)'}
    method_label = method_labels.get(geodesic_method, geodesic_method)
    print(f"\n{'='*80}")
    print(f"Computing Geodesic Distances ({method_label})")
    print(f"{'='*80}")
    print(f"  Geodesic method: {geodesic_method}")
    print(f"  Sources: {len(source_indices)}")
    print(f"  Target vertices: {len(vertices)}")
    print(f"  Total computations: {len(source_indices)} x {len(vertices)} = {len(source_indices) * len(vertices):,}")
    
    num_sources = len(source_indices)
    num_vertices = len(vertices)
    if num_sources == 0:
        print(f"\n  No new sources to compute. Skipping geodesic computation.")
        return np.array([]).reshape(0, num_vertices)
    
    # Determine parallelization mode
    use_parallel = n_jobs is not None and n_jobs != 1
    if use_parallel:
        n_processes = cpu_count() if n_jobs == -1 else min(n_jobs-1, cpu_count()-1)
        # Don't use more processes than sources
        n_processes = min(n_processes, num_sources)
        print(f"\n  Parallelization: ENABLED ({n_processes} processes)")
        print(f"  Strategy: Split {num_sources} sources into {n_processes} batches")
    else:
        print(f"\n  Parallelization: DISABLED (sequential processing)")
        print(f"  Note: Use --n_jobs to enable parallel processing for faster computation.")
        print(f"        Example: --n_jobs -1 (use all CPUs) or --n_jobs 4 (use 4 CPUs)")
    
    if use_parallel:
        # Split sources into batches for parallel processing
        batch_size = int(np.ceil(num_sources / n_processes))
        batches = []
        for i in range(n_processes):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sources)
            if start_idx < num_sources:
                batches.append(source_indices[start_idx:end_idx])
        
        print(f"  Batch sizes: {[len(b) for b in batches]}")
        print(f"  Processing batches in parallel...\n")
        
        # Process batches in parallel
        compute_func = partial(_compute_batch_source_geodesic, vertices=vertices, faces=faces, geodesic_method=geodesic_method, partial_save_dir=partial_save_dir)
        
        with Pool(processes=n_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(compute_func, batches),
                total=len(batches),
                desc="  Computing geodesics",
                unit="batch"
            ))
        
        # Concatenate results from all batches
        distances = np.concatenate(batch_results, axis=0)
        
    else:
        # Sequential processing: use single batch computation
        print(f"  Processing all sources in single batch...\n")
        distances = _compute_batch_source_geodesic(
            batch_source_indices=source_indices,
            vertices=vertices,
            faces=faces,
            geodesic_method=geodesic_method,
            partial_save_dir=partial_save_dir,
        )
    
    # Cap inf/nan values for non-manifold meshes with disconnected components
    n_inf = np.isinf(distances).sum()
    n_nan = np.isnan(distances).sum()
    if n_inf > 0 or n_nan > 0:
        finite_vals = distances[np.isfinite(distances)]
        if len(finite_vals) > 0:
            cap_value = finite_vals.max() * 2.0
        else:
            cap_value = 1e6
        distances = np.where(np.isfinite(distances), distances, cap_value)
        print(f"\n  Warning: Capped {n_inf} inf and {n_nan} nan values to {cap_value:.4f}")
        print(f"           (mesh likely has disconnected components)")

    print(f"\n  Distance statistics:")
    print(f"    Min: {distances.min():.6f}")
    print(f"    Max: {distances.max():.6f}")
    print(f"    Mean: {distances.mean():.6f}")
    print(f"    Std: {distances.std():.6f}")
    
    return distances


def _compute_batch_pipeline(
    batch_source_indices: np.ndarray,
    batch_source_positions: np.ndarray,
    batch_source_gaussian_indices: np.ndarray,
    batch_index: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    gaussian_positions: np.ndarray,
    gaussian_to_mesh_indices: np.ndarray,
    gaussian_to_mesh_distances: np.ndarray,
    geodesic_method: str = 'mmp',
    partial_save_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    barycentric_face_vertices: Optional[np.ndarray] = None,
    barycentric_weights: Optional[np.ndarray] = None,
    gaussian_vertex_indices: Optional[np.ndarray] = None,
) -> None:
    """Compute geodesics, transfer to Gaussians, and save for one batch.

    This is the per-worker function used by
    :func:`compute_and_save_geodesic_pipeline`.  It chains:

    1. ``_compute_batch_source_geodesic`` – mesh-level geodesic distances.
    2. Transfer to Gaussians (direct indexing or barycentric interpolation).
    3. ``save_partial_results`` – write the partial ``.npz`` file.

    Args:
        batch_source_indices:          (B,) mesh vertex indices for this batch.
        batch_source_positions:        (B, 3) positions of these sources.
        batch_source_gaussian_indices: (B,) Gaussian indices for each source.
        batch_index:                   Numeric index of this batch (for naming).
        vertices:                      (V, 3) mesh vertices.
        faces:                         (F, 3) mesh faces.
        gaussian_positions:            (G, 3) Gaussian centre positions.
        gaussian_to_mesh_indices:      (G,) nearest mesh vertex per Gaussian.
        gaussian_to_mesh_distances:    (G,) Euclidean distance to nearest vertex.
        geodesic_method:               ``'mmp'``, ``'vtp'``, or ``'fmm'``.
        partial_save_dir:              Directory for mesh-level batch cache.
        output_dir:                    Directory for the partial ``.npz`` file.
        barycentric_face_vertices:     (G, 3) – pass for barycentric interpolation.
        barycentric_weights:           (G, 3) – pass for barycentric interpolation.
        gaussian_vertex_indices:       (G,) – pass for direct-indexing transfer
                                       (embedded Gaussians mode).
    """
    if len(batch_source_indices) == 0:
        return

    # 1. Geodesic distances on mesh
    mesh_distances = _compute_batch_source_geodesic(
        batch_source_indices=batch_source_indices,
        vertices=vertices,
        faces=faces,
        geodesic_method=geodesic_method,
        partial_save_dir=partial_save_dir,
    )

    # Cap inf/nan
    n_inf = np.isinf(mesh_distances).sum()
    n_nan = np.isnan(mesh_distances).sum()
    if n_inf > 0 or n_nan > 0:
        finite_vals = mesh_distances[np.isfinite(mesh_distances)]
        cap_value = finite_vals.max() * 2.0 if len(finite_vals) > 0 else 1e6
        mesh_distances = np.where(np.isfinite(mesh_distances), mesh_distances, cap_value)

    # 2. Transfer to Gaussians
    if gaussian_vertex_indices is not None:
        gaussian_geodesic_distances = mesh_distances[:, gaussian_vertex_indices]
    else:
        gaussian_geodesic_distances = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances=mesh_distances,
            gaussian_to_mesh_indices=gaussian_to_mesh_indices,
            source_mesh_indices=batch_source_indices,
            source_gaussian_indices=batch_source_gaussian_indices,
            barycentric_face_vertices=barycentric_face_vertices,
            barycentric_weights=barycentric_weights,
        )

    # 3. Save partial results
    if output_dir is not None:
        src_start = int(batch_source_indices[0])
        src_end = int(batch_source_indices[-1]) + 1
        output_path = Path(output_dir) / f"sources_batch_{batch_index}_{src_start}_{src_end}.npz"
        save_partial_results(
            output_path=output_path,
            gaussian_positions=gaussian_positions,
            source_indices=batch_source_indices,
            source_positions=batch_source_positions,
            geodesic_distances=gaussian_geodesic_distances,
            closest_mesh_indices=gaussian_to_mesh_indices,
            closest_mesh_distances=gaussian_to_mesh_distances,
            source_gaussian_indices=batch_source_gaussian_indices,
        )


def compute_and_save_geodesic_pipeline(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    source_positions: np.ndarray,
    source_gaussian_indices: np.ndarray,
    gaussian_positions: np.ndarray,
    gaussian_to_mesh_indices: np.ndarray,
    gaussian_to_mesh_distances: np.ndarray,
    geodesic_method: str = 'mmp',
    n_jobs: Optional[int] = None,
    partial_save_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    barycentric_face_vertices: Optional[np.ndarray] = None,
    barycentric_weights: Optional[np.ndarray] = None,
    gaussian_vertex_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> None:
    """High-level pipeline: split sources → compute geodesics → transfer → save.

    Splits *source_indices* into batches, then dispatches each batch to
    :func:`_compute_batch_pipeline` either sequentially or via
    ``multiprocessing.Pool``.  Each worker independently produces its own
    partial ``.npz`` file inside *output_dir*, so a crash only loses the
    currently running batch.

    After all batches finish, the caller can invoke :func:`merge_partial_results`
    to combine the partial files into the final ``gt_geodesic.npz``.

    Args:
        vertices / faces:              Mesh geometry.
        source_indices:                (S,) mesh vertex indices of sources.
        source_positions:              (S, 3) source positions.
        source_gaussian_indices:       (S,) Gaussian index of each source.
        gaussian_positions:            (G, 3) Gaussian centre positions.
        gaussian_to_mesh_indices:      (G,) nearest mesh vertex per Gaussian.
        gaussian_to_mesh_distances:    (G,) distance to nearest mesh vertex.
        geodesic_method:               ``'mmp'``, ``'vtp'``, or ``'fmm'``.
        n_jobs:                        Parallelism (None/1 = sequential, -1 = all CPUs).
        partial_save_dir:              Dir for mesh-level per-batch cache.
        output_dir:                    Dir for partial ``.npz`` results.
        barycentric_face_vertices:     (G, 3) or *None*.
        barycentric_weights:           (G, 3) or *None*.
        gaussian_vertex_indices:       (G,) or *None* (embedded mode).
        verbose:                       Print progress info.
    """
    num_sources = len(source_indices)
    if num_sources == 0:
        print("  No sources to compute. Skipping pipeline.")
        return

    # Determine parallelism
    use_parallel = n_jobs is not None and n_jobs != 1
    if use_parallel:
        n_processes = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        n_processes = min(n_processes, num_sources)
    else:
        n_processes = 1

    # Split sources into batches
    batch_size = int(np.ceil(num_sources / n_processes))
    batches = []
    for i in range(n_processes):
        s = i * batch_size
        e = min((i + 1) * batch_size, num_sources)
        if s < num_sources:
            batches.append((
                i,
                source_indices[s:e],
                source_positions[s:e],
                source_gaussian_indices[s:e],
            ))

    method_labels = {'fmm': 'FMM', 'mmp': 'MMP', 'vtp': 'VTP'}
    print(f"\n{'='*80}")
    print(f"Geodesic Pipeline ({method_labels.get(geodesic_method, geodesic_method)})")
    print(f"{'='*80}")
    print(f"  Sources: {num_sources}")
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"  Gaussians: {len(gaussian_positions)}")
    print(f"  Batches: {len(batches)} ({n_processes} workers)")
    print(f"  Batch sizes: {[len(b[1]) for b in batches]}")

    # Build the partial function for workers
    worker = partial(
        _compute_batch_pipeline_wrapper,
        vertices=vertices,
        faces=faces,
        gaussian_positions=gaussian_positions,
        gaussian_to_mesh_indices=gaussian_to_mesh_indices,
        gaussian_to_mesh_distances=gaussian_to_mesh_distances,
        geodesic_method=geodesic_method,
        partial_save_dir=partial_save_dir,
        output_dir=output_dir,
        barycentric_face_vertices=barycentric_face_vertices,
        barycentric_weights=barycentric_weights,
        gaussian_vertex_indices=gaussian_vertex_indices,
    )

    if use_parallel and n_processes > 1:
        print(f"  Dispatching {len(batches)} batches across {n_processes} processes ...\n")
        with Pool(processes=n_processes) as pool:
            list(tqdm(
                pool.imap_unordered(worker, batches),
                total=len(batches),
                desc="  Pipeline batches",
                unit="batch",
            ))
    else:
        print(f"  Running {len(batches)} batches sequentially ...\n")
        for batch_args in tqdm(batches, desc="  Pipeline batches", unit="batch"):
            worker(batch_args)

    print(f"  Pipeline complete. Partial results in: {output_dir}")


def _compute_batch_pipeline_wrapper(
    batch_args: tuple,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    gaussian_positions: np.ndarray,
    gaussian_to_mesh_indices: np.ndarray,
    gaussian_to_mesh_distances: np.ndarray,
    geodesic_method: str,
    partial_save_dir: Optional[str],
    output_dir: Optional[str],
    barycentric_face_vertices: Optional[np.ndarray],
    barycentric_weights: Optional[np.ndarray],
    gaussian_vertex_indices: Optional[np.ndarray],
) -> None:
    """Unpack batch tuple and call :func:`_compute_batch_pipeline`.

    ``multiprocessing.Pool.imap`` passes a single positional arg, so we
    use ``functools.partial`` to bind the shared keyword arguments and this
    wrapper to unpack the per-batch tuple.
    """
    batch_index, src_idx, src_pos, src_gauss_idx = batch_args
    _compute_batch_pipeline(
        batch_source_indices=src_idx,
        batch_source_positions=src_pos,
        batch_source_gaussian_indices=src_gauss_idx,
        batch_index=batch_index,
        vertices=vertices,
        faces=faces,
        gaussian_positions=gaussian_positions,
        gaussian_to_mesh_indices=gaussian_to_mesh_indices,
        gaussian_to_mesh_distances=gaussian_to_mesh_distances,
        geodesic_method=geodesic_method,
        partial_save_dir=partial_save_dir,
        output_dir=output_dir,
        barycentric_face_vertices=barycentric_face_vertices,
        barycentric_weights=barycentric_weights,
        gaussian_vertex_indices=gaussian_vertex_indices,
    )


def save_mesh_geodesic_gt(
    data_root: Path,
    surface: str,
    mesh_level: int,
    source_indices: np.ndarray,
    source_positions: np.ndarray,
    geodesic_distances: np.ndarray,
    mesh_vertices: np.ndarray
) -> Path:
    """
    Save mesh geodesic ground truth to data_root/surface/level_{level}/geodesic/.
    
    Args:
        data_root: Base data directory
        surface: Surface name
        mesh_level: Mesh resolution level
        source_indices: (S,) source vertex indices
        source_positions: (S, 3) source positions
        geodesic_distances: (S, N) geodesic distances from sources to all vertices
        mesh_vertices: (N, 3) mesh vertex positions
    
    Returns:
        Path to saved file
    """
    output_dir = data_root / surface / f"level_{mesh_level}" / "geodesic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "mesh_geodesic_gt.npz"
    
    print(f"\n{'='*80}")
    print(f"Saving Mesh Geodesic Ground Truth")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")
    
    np.savez_compressed(
        output_path,
        source_indices=source_indices,
        source_positions=source_positions,
        geodesic_distances=geodesic_distances,
        mesh_vertices=mesh_vertices
    )
    
    print(f"  Saved data:")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    print(f"    mesh_vertices: {mesh_vertices.shape}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return output_path


def load_mesh_geodesic_gt(
    data_root: Path,
    surface: str,
    mesh_level: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load existing mesh geodesic ground truth if it exists.
    
    Args:
        data_root: Base data directory
        surface: Surface name
        mesh_level: Mesh resolution level
    
    Returns:
        Tuple of (source_indices, source_positions, geodesic_distances) or None if not found
    """
    geodesic_path = data_root / surface / f"level_{mesh_level}" / "geodesic" / "mesh_geodesic_gt.npz"
    
    if not geodesic_path.exists():
        print(f"\n  No existing mesh geodesic data found at: {geodesic_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Loading Existing Mesh Geodesic Ground Truth")
    print(f"{'='*80}")
    print(f"  Path: {geodesic_path}")
    
    data = np.load(geodesic_path)
    source_indices = data['source_indices']
    source_positions = data['source_positions']
    geodesic_distances = data['geodesic_distances']
    
    print(f"  Loaded data:")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    
    return source_indices, source_positions, geodesic_distances


def find_missing_sources(
    all_source_indices: np.ndarray,
    existing_source_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find which sources are missing from existing data.
    
    Args:
        all_source_indices: All source indices we want to compute
        existing_source_indices: Source indices already computed
    
    Returns:
        Tuple of (missing_indices, missing_mask) where missing_mask is boolean array
    """
    # Find which indices from all_source_indices are not in existing_source_indices
    missing_mask = ~np.isin(all_source_indices, existing_source_indices)
    missing_indices = all_source_indices[missing_mask]
    
    print(f"\n{'='*80}")
    print(f"Checking for Missing Sources")
    print(f"{'='*80}")
    print(f"  Total requested sources: {len(all_source_indices)}")
    print(f"  Existing sources: {len(existing_source_indices)}")
    print(f"  Missing sources: {len(missing_indices)}")
    
    return missing_indices, missing_mask


def merge_geodesic_data(
    existing_source_indices: np.ndarray,
    existing_source_positions: np.ndarray,
    existing_geodesic_distances: np.ndarray,
    new_source_indices: np.ndarray,
    new_source_positions: np.ndarray,
    new_geodesic_distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge existing and newly computed geodesic data.
    
    Args:
        existing_source_indices: Existing source indices
        existing_source_positions: Existing source positions
        existing_geodesic_distances: Existing geodesic distances
        new_source_indices: New source indices
        new_source_positions: New source positions
        new_geodesic_distances: New geodesic distances
    
    Returns:
        Tuple of merged (source_indices, source_positions, geodesic_distances)
    """
    print(f"\n{'='*80}")
    print(f"Merging Geodesic Data")
    print(f"{'='*80}")
    
    # Handle empty arrays
    existing_empty = len(existing_source_indices) == 0
    new_empty = len(new_source_indices) == 0
    
    if existing_empty and new_empty:
        # Both empty - return empty arrays with proper shapes
        print(f"  Warning: Both existing and new data are empty")
        num_vertices = 0
        return (np.array([]), 
                np.array([]).reshape(0, 3), 
                np.array([]).reshape(0, num_vertices))
    
    if existing_empty:
        # Only existing is empty - return new data
        print(f"  No existing data, using only new data ({len(new_source_indices)} sources)")
        return new_source_indices, new_source_positions, new_geodesic_distances
    
    if new_empty:
        # Only new is empty - return existing data
        print(f"  No new data, using only existing data ({len(existing_source_indices)} sources)")
        return existing_source_indices, existing_source_positions, existing_geodesic_distances
    
    # Both have data - concatenate
    merged_source_indices = np.concatenate([existing_source_indices, new_source_indices])
    merged_source_positions = np.concatenate([existing_source_positions, new_source_positions])
    merged_geodesic_distances = np.concatenate([existing_geodesic_distances, new_geodesic_distances], axis=0)
    
    # Sort by source index
    sort_order = np.argsort(merged_source_indices)
    merged_source_indices = merged_source_indices[sort_order]
    merged_source_positions = merged_source_positions[sort_order]
    merged_geodesic_distances = merged_geodesic_distances[sort_order]
    
    print(f"  Merged total sources: {len(merged_source_indices)}")
    print(f"    Existing: {len(existing_source_indices)}, New: {len(new_source_indices)}")
    
    return merged_source_indices, merged_source_positions, merged_geodesic_distances


def map_indexes_between_gaussian_and_surfaces(
    source_idx: np.ndarray,
    source_points: np.ndarray,
    dest_surface: np.ndarray,
    use_mahalanobis: bool = False,
    dest_scales: Optional[np.ndarray] = None,
    dest_rotations: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Map indexes from source surface to destination surface.
    
    Args:
        source_idx: Indices to map from source
        source_points: Source point positions
        dest_surface: Destination point positions
        use_mahalanobis: Whether to use Mahalanobis distance
        dest_scales: (N, 3) scales for destination Gaussians (required if use_mahalanobis=True)
        dest_rotations: (N, 4) rotations for destination Gaussians (required if use_mahalanobis=True)
    
    Returns:
        Array of closest destination indices
    """
    from GenerateData.utils.data_generation_utils import map_points_to_surface
    
    return map_points_to_surface(
        query_points=source_points[source_idx],
        target_points=dest_surface,
        use_mahalanobis=use_mahalanobis,
        target_scales=dest_scales,
        target_rotations=dest_rotations
    )


def project_points_to_triangles_vectorized(
    P: np.ndarray,  # (N, 3)
    A: np.ndarray,  # (N, 3)
    B: np.ndarray,  # (N, 3)
    C: np.ndarray,  # (N, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised closest-point-on-triangle projection (Ericson 2005).

    For each of the N (point, triangle) pairs, computes the barycentric
    coordinates of the closest point on the triangle to the query point and
    the Euclidean distance to that closest point.

    Args:
        P: (N, 3) query points
        A, B, C: (N, 3) triangle vertices (per-row triangle)

    Returns:
        bary:  (N, 3) barycentric weights  (non-negative, sum to 1)
        dists: (N,)   distance from P to closest point on triangle
    """
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = np.einsum('ij,ij->i', AB, AP)
    d2 = np.einsum('ij,ij->i', AC, AP)

    BP = P - B
    d3 = np.einsum('ij,ij->i', AB, BP)
    d4 = np.einsum('ij,ij->i', AC, BP)

    CP = P - C
    d5 = np.einsum('ij,ij->i', AB, CP)
    d6 = np.einsum('ij,ij->i', AC, CP)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    N = len(P)
    bary = np.empty((N, 3), dtype=np.float64)

    # Region masks (vertex regions take priority).
    rA  = (d1 <= 0) & (d2 <= 0)
    rB  = (d3 >= 0) & (d4 <= d3)
    rC  = (d6 >= 0) & (d5 <= d6)
    eAB = ~rA & ~rB & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    eAC = ~rA & ~rC & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    eBC = ~rB & ~rC & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    rInt = ~(rA | rB | rC | eAB | eAC | eBC)

    bary[rA] = [1.0, 0.0, 0.0]
    bary[rB] = [0.0, 1.0, 0.0]
    bary[rC] = [0.0, 0.0, 1.0]

    idxAB = np.where(eAB)[0]
    if len(idxAB):
        denom = d1[idxAB] - d3[idxAB]
        v = np.where(denom != 0, d1[idxAB] / denom, 0.5)
        bary[idxAB] = np.column_stack([1 - v, v, np.zeros(len(idxAB))])

    idxAC = np.where(eAC)[0]
    if len(idxAC):
        denom = d2[idxAC] - d6[idxAC]
        w = np.where(denom != 0, d2[idxAC] / denom, 0.5)
        bary[idxAC] = np.column_stack([1 - w, np.zeros(len(idxAC)), w])

    idxBC = np.where(eBC)[0]
    if len(idxBC):
        denom = (d4[idxBC] - d3[idxBC]) + (d5[idxBC] - d6[idxBC])
        w = np.where(denom != 0, (d4[idxBC] - d3[idxBC]) / denom, 0.5)
        bary[idxBC] = np.column_stack([np.zeros(len(idxBC)), 1 - w, w])

    idxInt = np.where(rInt)[0]
    if len(idxInt):
        denom = va[idxInt] + vb[idxInt] + vc[idxInt]
        v = np.where(denom != 0, vb[idxInt] / denom, 1.0 / 3)
        w = np.where(denom != 0, vc[idxInt] / denom, 1.0 / 3)
        bary[idxInt] = np.column_stack([1 - v - w, v, w])

    proj  = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    dists = np.linalg.norm(P - proj, axis=1)
    return bary, dists


def find_closest_mesh_faces_barycentric(
    gaussian_centers: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    closest_vertex_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each Gaussian, find the closest mesh face and the barycentric
    coordinates of its projection onto that face.

    Uses the 1-ring of each Gaussian's already-computed closest vertex to
    restrict candidate faces, giving O(G * avg_degree) cost.

    Args:
        gaussian_centers:       (G, 3) Gaussian center positions
        mesh_vertices:          (V, 3) mesh vertex positions
        mesh_faces:             (F, 3) face connectivity (integer indices)
        closest_vertex_indices: (G,)  nearest mesh vertex per Gaussian
                                      (pre-computed by find_closest_mesh_vertices)

    Returns:
        face_vertices: (G, 3) vertex indices of the closest face per Gaussian
        bary_weights:  (G, 3) barycentric weights (non-negative, sum ≈ 1)
    """
    print(f"\n  Computing barycentric face projections for {len(gaussian_centers)} Gaussians...")

    # Build vertex -> list-of-face-index adjacency
    V = len(mesh_vertices)
    vertex_to_faces: list = [[] for _ in range(V)]
    for f_idx, face in enumerate(mesh_faces):
        vertex_to_faces[face[0]].append(f_idx)
        vertex_to_faces[face[1]].append(f_idx)
        vertex_to_faces[face[2]].append(f_idx)

    # Enumerate all (Gaussian, candidate-face) pairs
    g_idx_list, f_idx_list = [], []
    for g_idx, v_idx in enumerate(closest_vertex_indices):
        for f_idx in vertex_to_faces[v_idx]:
            g_idx_list.append(g_idx)
            f_idx_list.append(f_idx)

    G_idx  = np.array(g_idx_list, dtype=np.int64)  # (P,)
    F_idx  = np.array(f_idx_list, dtype=np.int64)  # (P,)

    # Gather triangle vertex positions for all candidate pairs
    V_faces = mesh_faces[F_idx]               # (P, 3)
    P_pts   = gaussian_centers[G_idx]          # (P, 3)
    A       = mesh_vertices[V_faces[:, 0]]     # (P, 3)
    B       = mesh_vertices[V_faces[:, 1]]     # (P, 3)
    C       = mesh_vertices[V_faces[:, 2]]     # (P, 3)

    # Vectorised projection onto all candidate triangles
    bary, dists = project_points_to_triangles_vectorized(P_pts, A, B, C)

    # For each Gaussian keep the candidate with minimum projection distance.
    # lexsort with (dists, G_idx) → primary sort by G_idx, tie-break by dists;
    # np.unique with return_index gives the first (= min-dist) occurrence per G.
    order    = np.lexsort((dists, G_idx))
    _, first = np.unique(G_idx[order], return_index=True)
    best     = order[first]  # indices into the candidate arrays

    face_vertices = V_faces[best]   # (G, 3)
    bary_weights  = bary[best]      # (G, 3)

    best_dists = dists[best]
    print(f"    Projection-distance stats (Gaussian → closest face surface):")
    print(f"      Mean={best_dists.mean():.6f}  Median={np.median(best_dists):.6f}  "
          f"Max={best_dists.max():.6f}")
    return face_vertices, bary_weights


def find_closest_mesh_vertices(
    gaussian_centers: np.ndarray,
    mesh_vertices: np.ndarray,
    use_mahalanobis: bool = False,
    gaussian_scales: Optional[np.ndarray] = None,
    gaussian_rotations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the closest mesh vertex for each Gaussian center.
    
    Args:
        gaussian_centers: (G, 3) array of Gaussian center positions
        mesh_vertices: (N, 3) array of mesh vertex positions
        use_mahalanobis: Whether to use Mahalanobis distance
        gaussian_scales: (G, 3) Gaussian scales (required if use_mahalanobis=True)
        gaussian_rotations: (G, 4) Gaussian rotations (required if use_mahalanobis=True)
    
    Returns:
        Tuple of:
        - closest_indices: (G,) array of mesh vertex indices
        - closest_distances: (G,) array of distances to closest vertex
    """
    from GenerateData.utils.data_generation_utils import map_points_to_surface
    
    print(f"\n{'='*80}")
    print(f"Finding Closest Mesh Vertices")
    print(f"{'='*80}")
    print(f"  Method: {'Mahalanobis' if use_mahalanobis else 'Euclidean'} distance")
    print(f"  Querying nearest neighbors for {len(gaussian_centers)} Gaussians...")
    
    closest_indices, closest_distances = map_points_to_surface(
        query_points=gaussian_centers,
        target_points=mesh_vertices,
        use_mahalanobis=use_mahalanobis,
        query_scales=gaussian_scales,
        query_rotations=gaussian_rotations,
        return_distances=True
    )
    
    print(f"\n  Projection statistics:")
    print(f"    Mean distance to closest vertex: {closest_distances.mean():.6f}")
    print(f"    Max distance to closest vertex: {closest_distances.max():.6f}")
    print(f"    Min distance to closest vertex: {closest_distances.min():.6f}")
    print(f"    Std distance to closest vertex: {closest_distances.std():.6f}")
    
    # Check for potential issues
    if closest_distances.max() > 0.1:
        print(f"\n  Warning: Some Gaussians are far from the mesh (max distance = {closest_distances.max():.6f})")
        print(f"           This may indicate misalignment between Gaussian splat and ground truth mesh.")
    
    return closest_indices, closest_distances


def transfer_geodesic_to_gaussians(
    mesh_geodesic_distances: np.ndarray,
    gaussian_to_mesh_indices: np.ndarray,
    source_mesh_indices: np.ndarray,
    source_gaussian_indices: np.ndarray,
    barycentric_face_vertices: Optional[np.ndarray] = None,
    barycentric_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transfer geodesic distances from mesh vertices to Gaussian splats.

    Two modes:
    - **Barycentric interpolation** (preferred): each Gaussian's distance is
      computed as a weighted sum of the geodesic distances at the 3 vertices of
      its closest face, using barycentric weights.  This is O(h²) accurate
      (vs O(h) for vertex snapping) and handles Gaussians that lie off the mesh.
      Requires `barycentric_face_vertices` and `barycentric_weights`.
    - **Vertex snapping** (fallback): each Gaussian is assigned the geodesic
      distance of its nearest mesh vertex.

    Args:
        mesh_geodesic_distances:    (S, V) geodesic distances on mesh vertices
        gaussian_to_mesh_indices:   (G,)   nearest mesh vertex per Gaussian
                                           (used only in vertex-snapping mode)
        source_mesh_indices:        (S,)   source mesh vertex indices
        source_gaussian_indices:    (S,)   Gaussian indices of sources
        barycentric_face_vertices:  (G, 3) vertex indices of the closest face per
                                           Gaussian (from find_closest_mesh_faces_barycentric)
        barycentric_weights:        (G, 3) barycentric weights (sum ≈ 1)

    Returns:
        gaussian_geodesic_distances: (S, G) geodesic distances for Gaussians
    """
    print(f"\n{'='*80}")
    print(f"Transferring Geodesic Distances to Gaussians")
    print(f"{'='*80}")

    num_sources   = mesh_geodesic_distances.shape[0]
    num_gaussians = len(gaussian_to_mesh_indices)
    use_bary      = (barycentric_face_vertices is not None) and (barycentric_weights is not None)

    print(f"  Sources:   {num_sources}")
    print(f"  Gaussians: {num_gaussians}")
    print(f"  Mode:      {'Barycentric interpolation' if use_bary else 'Vertex snapping (fallback)'}")

    if use_bary:
        # dist[i, j] = w0[j]*D[i,v0[j]] + w1[j]*D[i,v1[j]] + w2[j]*D[i,v2[j]]
        d0 = mesh_geodesic_distances[:, barycentric_face_vertices[:, 0]]  # (S, G)
        d1 = mesh_geodesic_distances[:, barycentric_face_vertices[:, 1]]  # (S, G)
        d2 = mesh_geodesic_distances[:, barycentric_face_vertices[:, 2]]  # (S, G)
        w0 = barycentric_weights[:, 0]  # (G,)
        w1 = barycentric_weights[:, 1]
        w2 = barycentric_weights[:, 2]
        gaussian_geodesic_distances = (
            d0 * w0[None, :] + d1 * w1[None, :] + d2 * w2[None, :]
        )
    else:
        # Fallback: vertex snapping
        gaussian_geodesic_distances = mesh_geodesic_distances[:, gaussian_to_mesh_indices]

    print(f"  Source Gaussian index range: [{source_gaussian_indices.min()}, {source_gaussian_indices.max()}]")
    print(f"\n  Transferred distance statistics:")
    print(f"    Min:  {gaussian_geodesic_distances.min():.6f}")
    print(f"    Max:  {gaussian_geodesic_distances.max():.6f}")
    print(f"    Mean: {gaussian_geodesic_distances.mean():.6f}")
    print(f"    Std:  {gaussian_geodesic_distances.std():.6f}")

    return gaussian_geodesic_distances


def save_partial_results(
    output_path: Path,
    gaussian_positions: np.ndarray,
    source_indices: np.ndarray,
    source_positions: np.ndarray,
    geodesic_distances: np.ndarray,
    closest_mesh_indices: np.ndarray,
    closest_mesh_distances: np.ndarray,
    source_gaussian_indices: np.ndarray
) -> None:
    """
    Save partial geodesic distance results to NPZ file.
    
    Args:
        output_path: Path to output .npz file
        gaussian_positions: (N_gaussian, 3) Gaussian center positions
        source_indices: (S,) source vertex indices
        source_positions: (S, 3) source vertex positions
        geodesic_distances: (S, N_gaussian) geodesic distances
        closest_mesh_indices: (N_gaussian,) nearest mesh vertex for each Gaussian
        closest_mesh_distances: (N_gaussian,) Euclidean distance to nearest vertex
        source_gaussian_indices: (S,) Gaussian indices corresponding to sources
    """
    print(f"\n{'='*80}")
    print(f"Saving Partial Results")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        gaussian_positions=gaussian_positions,
        source_indices=source_indices,
        source_positions=source_positions,
        geodesic_distances=geodesic_distances,
        closest_mesh_indices=closest_mesh_indices,
        closest_mesh_distances=closest_mesh_distances,
        source_gaussian_indices=source_gaussian_indices
    )
    
    print(f"  Saved data:")
    print(f"    gaussian_positions: {gaussian_positions.shape}")
    print(f"    source_indices: {source_indices.shape}")
    print(f"    source_positions: {source_positions.shape}")
    print(f"    geodesic_distances: {geodesic_distances.shape}")
    print(f"    closest_mesh_indices: {closest_mesh_indices.shape}")
    print(f"    closest_mesh_distances: {closest_mesh_distances.shape}")
    print(f"    source_gaussian_indices: {source_gaussian_indices.shape}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")


def merge_partial_results(output_folder: Path, verbose: bool = False) -> None:
    """
    Merge all partial results into a single complete ground truth file.
    
    Args:
        output_folder: Base output folder containing gt_partial subdirectory
        verbose: Whether to show detailed progress
    """
    print(f"\n{'='*80}")
    print(f"Merging Partial Results")
    print(f"{'='*80}")
    
    partial_dir = output_folder / "geodesic_distance" / "gt_partial"
    if not partial_dir.exists():
        print(f"  Error: No partial results directory found at {partial_dir}")
        return
    
    # Find all partial result files (both old range and new batch formats)
    partial_files = sorted(partial_dir.glob("sources_range_*.npz")) + \
                    sorted(partial_dir.glob("sources_batch_*.npz"))
    if not partial_files:
        print(f"  Error: No partial result files found in {partial_dir}")
        return
    
    print(f"  Found {len(partial_files)} partial result files:")
    for f in partial_files:
        print(f"    - {f.name}")
    
    # Load all partial results
    all_source_indices = []
    all_source_positions = []
    all_geodesic_distances = []
    all_source_gaussian_indices = []
    gaussian_positions = None
    closest_mesh_indices = None
    closest_mesh_distances = None
    
    for partial_file in tqdm(partial_files, desc="  Loading partial files"):
        data = np.load(partial_file)
        
        # Verify consistency (should be same for all files)
        if gaussian_positions is None:
            gaussian_positions = data['gaussian_positions']
            closest_mesh_indices = data['closest_mesh_indices']
            closest_mesh_distances = data['closest_mesh_distances']
        else:
            # Sanity check
            assert np.allclose(gaussian_positions, data['gaussian_positions']), \
                f"Gaussian positions mismatch in {partial_file.name}"
        
        all_source_indices.append(data['source_indices'])
        all_source_positions.append(data['source_positions'])
        all_geodesic_distances.append(data['geodesic_distances'])
        all_source_gaussian_indices.append(data['source_gaussian_indices'])
        
        if verbose:
            print(f"    Loaded {len(data['source_indices'])} sources from {partial_file.name}")
    
    # Concatenate all sources
    merged_source_indices = np.concatenate(all_source_indices, axis=0)
    merged_source_positions = np.concatenate(all_source_positions, axis=0)
    merged_geodesic_distances = np.concatenate(all_geodesic_distances, axis=0)
    merged_source_gaussian_indices = np.concatenate(all_source_gaussian_indices, axis=0)
    
    # Deduplicate by source_gaussian_indices (keep last occurrence, which is
    # typically the more complete computation from a later run).
    n_before_dedup = len(merged_source_gaussian_indices)
    _, unique_idx = np.unique(merged_source_gaussian_indices, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve order from concatenation
    merged_source_indices = merged_source_indices[unique_idx]
    merged_source_positions = merged_source_positions[unique_idx]
    merged_geodesic_distances = merged_geodesic_distances[unique_idx]
    merged_source_gaussian_indices = merged_source_gaussian_indices[unique_idx]
    n_removed = n_before_dedup - len(unique_idx)
    if n_removed > 0:
        print(f"\n  Deduplicated: removed {n_removed} duplicate source(s)")
    
    # Sort by gaussian source index (the indexes of sources in the Gaussian splat)
    sort_order = np.argsort(merged_source_gaussian_indices)
    merged_source_indices = merged_source_indices[sort_order]
    merged_source_positions = merged_source_positions[sort_order]
    merged_geodesic_distances = merged_geodesic_distances[sort_order]
    merged_source_gaussian_indices = merged_source_gaussian_indices[sort_order]
    
    print(f"\n  Merged data:")
    print(f"    Total sources: {len(merged_source_indices)}")
    print(f"    Gaussians: {len(gaussian_positions)}")
    print(f"    Geodesic distances shape: {merged_geodesic_distances.shape}")
    print(f"    Source Gaussian indices shape: {merged_source_gaussian_indices.shape}")
    
    # Save merged results
    output_path = output_folder / "geodesic_distance" / "gt_geodesic.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        gaussian_positions=gaussian_positions,
        source_indices=merged_source_indices,
        source_positions=merged_source_positions,
        geodesic_distances=merged_geodesic_distances,
        closest_mesh_indices=closest_mesh_indices,
        closest_mesh_distances=closest_mesh_distances,
        source_gaussian_indices=merged_source_gaussian_indices
    )
    
    print(f"\n  Saved complete ground truth to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"\n  Distance statistics:")
    print(f"    Min: {merged_geodesic_distances.min():.6f}")
    print(f"    Max: {merged_geodesic_distances.max():.6f}")
    print(f"    Mean: {merged_geodesic_distances.mean():.6f}")
    print(f"    Std: {merged_geodesic_distances.std():.6f}")


def save_computation_metadata(
    output_folder: Path,
    args: argparse.Namespace,
    num_gaussians: int,
    num_sources: int,
    surface_name: Optional[str] = None
) -> None:
    """
    Save metadata about the geodesic distance computation.
    
    Args:
        output_folder: Output folder path
        args: Command line arguments
        num_gaussians: Number of Gaussians
        num_sources: Total number of sources
        surface_name: Surface name
    """
    metadata = {
        'computation_info': {
            'timestamp': datetime.now().isoformat(),
            'script': 'compute_gaussian_geodesic_distances.py',
            'description': 'Ground truth geodesic distances on Gaussian splats'
        },
        'surface': {
            'name': surface_name if surface_name else args.surface,
            'type': args.surface if args.surface else 'Unknown',
            'data_root': str(args.data_root),
            'mesh_level': args.mesh_level
        },
        'gaussian_data': {
            'source_folder': str(args.gaussian_output),
            'iteration': args.iteration if args.iteration else 'auto (highest)',
            'num_gaussians': num_gaussians
        },
        'source_generation': {
            'method': 'parametric_mesh',
            'source_mesh_resolution': args.source_mesh_resolution,
            'total_sources': num_sources,
            'source_selection': args.source_selection
        },
        'distance_computation': {
            'method': 'Mahalanobis' if args.use_mahalanobis else 'Euclidean',
            'description': 'Gaussian covariance-based distance' if args.use_mahalanobis else 'Standard Euclidean distance',
            'geodesic_algorithm': getattr(args, 'geodesic_method', 'mmp')
        },
        'parameters': {
            'use_mahalanobis': args.use_mahalanobis,
            'seed': args.seed,
            'source_start': args.source_start,
            'source_end': args.source_end
        }
    }
    
    # Save metadata
    metadata_path = output_folder / "geodesic_distance" / "computation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Metadata saved to: {metadata_path}")
