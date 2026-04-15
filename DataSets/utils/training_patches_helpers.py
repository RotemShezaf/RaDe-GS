#!/usr/bin/env python3
"""
Helper functions for creating Gaussian training patches.

This module contains utility functions for:
- Computing Gaussian normals from principal axes
- Creating individual training examples
- Generating batches of training examples
- Metadata and configuration handling

Dataclasses:
- ``GaussianData``: bundles per-point arrays (positions, scales, …)
- ``PatchConfig``: bundles generation parameters (ring, attributes, …)

These functions are used by create_gaussian_training_patches.py
"""

from ast import If
from dataclasses import dataclass, field

import numpy as np
import argparse
import multiprocessing
from numpy import linalg as LA
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# Import from canonical locations
from DataSets.utils.config_utils import get_ring_size_mapping


# ---------------------------------------------------------------------------
# Dataclasses for bundled argument passing
# ---------------------------------------------------------------------------

@dataclass
class GaussianData:
    """Bundle of per-point Gaussian arrays used throughout patch generation.

    Avoids passing ``positions, scales, rotations, opacities, normals,
    sh_features, per_point_nn_distances`` as separate arguments.
    """
    positions: np.ndarray                        # (N, 3)
    scales: Optional[np.ndarray] = None          # (N, 3)
    rotations: Optional[np.ndarray] = None       # (N, 4)
    opacities: Optional[np.ndarray] = None       # (N, 1)
    normals: Optional[np.ndarray] = None         # (N, 3)
    sh_features: Optional[np.ndarray] = None     # (N, K)
    per_point_nn_distances: Optional[np.ndarray] = None  # (N,)


@dataclass
class PatchConfig:
    """Generation parameters for training patch creation.

    Avoids passing ``ring, normalization_factor, nn_mean, attributes, …``
    as a long positional-argument list.
    """
    ring: int = 2
    normalization_factor: float = 1.0
    nn_mean: float = 1.0
    attributes: List[str] = field(default_factory=lambda: ["xyz"])
    use_mahalanobis: bool = False
    use_r1_min_val: bool = False
    mask_attributes: List[str] = field(default_factory=list)
    mask_constant: float = -10.0
    ring_size_mapping: Optional[Dict] = None
    normalize_per_patch: bool = False
    disable_outlier_filtering: bool = False
    surface_type: Optional[str] = None
    # Outlier filtering parameters (training path)
    outlier_median_multiplier: float = 3.0
    outlier_threshold_floor: float = 2.0
    outlier_hard_cap: float = 500.0
    outlier_fallback_multiplier: float = 5.0
    outlier_fallback_floor: float = 3.0
    # Outlier filtering parameters (inference path — _filter_outliers_without_pu)
    outlier_max_removal_fraction: Optional[float] = None
    # When True, outlier neighbors have their geodesic set to mask_constant
    # instead of being removed entirely.  This preserves spatial (xyz) info.
    mask_outliers_only: bool = False


# ---------------------------------------------------------------------------
# Module-level shared data for multiprocessing workers.
# On Linux (fork), child processes inherit the parent's address space
# copy-on-write, so we avoid pickling large arrays.
# ---------------------------------------------------------------------------
_mp_shared: Optional[Dict] = None


def _init_mp_worker(shared: Dict) -> None:
    """Initializer for Pool workers – stores a reference to shared data."""
    global _mp_shared
    _mp_shared = shared


def _generate_chunk(params: Tuple) -> List[np.ndarray]:
    """
    Worker function: generate training examples for a chunk of iterations.

    Reads shared data from the module-level ``_mp_shared`` dict.

    Args:
        params: Tuple of (iter_start, iter_end, base_seed, ring, num_sources,
                num_train_points, normalization_factor, nn_mean, attributes,
                use_mahalanobis, use_r1_min_val, mask_attributes, mask_constant,
                ring_size_mapping, normalize_per_patch, near_source_oversample,
                surface_type)

    Returns:
        List of example arrays generated in this chunk.
    """
    global _mp_shared
    d = _mp_shared

    (iter_start, iter_end, base_seed, ring, num_sources,
     num_train_points, normalization_factor, nn_mean, attributes,
     use_mahalanobis, use_r1_min_val, mask_attributes, mask_constant,
     ring_size_mapping, normalize_per_patch, near_source_oversample,
     surface_type, disable_outlier_filtering,
     outlier_median_multiplier, outlier_threshold_floor,
     outlier_hard_cap, outlier_fallback_multiplier,
     outlier_fallback_floor) = params

    gdata = GaussianData(
        positions=d['positions'],
        scales=d['scales'],
        rotations=d['rotations'],
        opacities=d['opacities'],
        normals=d['normals'],
        sh_features=d['sh_features'],
        per_point_nn_distances=d['per_point_nn_distances'],
    )
    pcfg = PatchConfig(
        ring=ring,
        normalization_factor=normalization_factor,
        nn_mean=nn_mean,
        attributes=list(attributes),
        use_mahalanobis=use_mahalanobis,
        use_r1_min_val=use_r1_min_val,
        mask_attributes=list(mask_attributes),
        mask_constant=mask_constant,
        ring_size_mapping=ring_size_mapping,
        normalize_per_patch=normalize_per_patch,
        surface_type=surface_type,
        disable_outlier_filtering=disable_outlier_filtering,
        outlier_median_multiplier=outlier_median_multiplier,
        outlier_threshold_floor=outlier_threshold_floor,
        outlier_hard_cap=outlier_hard_cap,
        outlier_fallback_multiplier=outlier_fallback_multiplier,
        outlier_fallback_floor=outlier_fallback_floor,
    )

    geodesic_data = d['geodesic_data']
    ring_nbrs_dict = d['ring_nbrs_dict']
    ring1_nbrs = d['ring1_nbrs']
    inv_ring1 = d['inverse_ring1_nbrs']

    num_gaussians = len(gdata.positions)
    all_source_indices = geodesic_data['source_gaussian_indices']
    all_geodesic_distances = geodesic_data['geodesic_distances']

    rng = np.random.RandomState(base_seed + iter_start)
    examples: List[np.ndarray] = []
    patches_with_outliers = 0
    total_neighbors_removed = 0
    total_patches = 0

    for i in range(iter_start, iter_end):
        # Randomly select sources
        if num_sources <= len(all_source_indices):
            selected_source_idxs = rng.choice(
                len(all_source_indices), num_sources, replace=False
            )
        else:
            selected_source_idxs = np.arange(len(all_source_indices))

        selected_distances = all_geodesic_distances[selected_source_idxs]
        min_distances = selected_distances.min(axis=0)

        source_gaussian_idxs = all_source_indices[selected_source_idxs]
        available_points = np.setdiff1d(np.arange(num_gaussians), source_gaussian_idxs)

        if len(available_points) < num_train_points:
            train_points = available_points
        else:
            train_points = _sample_train_points(
                rng, available_points, num_train_points, min_distances,
                near_source_oversample,
            )

        for point_idx in train_points:
            _ex_stats = {}
            example = create_train_example(
                point_idx, min_distances,
                ring_nbrs_dict[pcfg.ring], ring1_nbrs,
                gaussian_data=gdata, patch_config=pcfg,
                _stats_out=_ex_stats,
                inverse_ring1_nbrs=inv_ring1,
            )
            if example is not None:
                examples.append(example)
                total_patches += 1
                nr = _ex_stats.get('num_removed', 0)
                if nr > 0:
                    patches_with_outliers += 1
                    total_neighbors_removed += nr

    return examples, {
        'patches_with_outliers': patches_with_outliers,
        'total_neighbors_removed': total_neighbors_removed,
        'total_patches': total_patches,
    }


def _sample_train_points(
    rng: np.random.RandomState,
    available_points: np.ndarray,
    num_train_points: int,
    min_distances: np.ndarray,
    near_source_oversample: float,
) -> np.ndarray:
    """Sample training points with optional near-source oversampling.

    When ``near_source_oversample > 0``, **all** ``num_train_points`` are
    drawn from a single weighted distribution that smoothly transitions
    from heavy near-source bias to uniform far from the source.

    The weighting function is::

        w(d) = exp(-alpha * d / median_d) + 1

    where ``median_d`` is the median non-zero geodesic distance among
    ``available_points`` and ``alpha`` controls the sharpness.  The ``+1``
    floor ensures that far-away points always retain at least uniform
    probability — i.e.  the distribution *collapses* to uniform at large
    distances.  ``near_source_oversample`` controls ``alpha``:

    * 0.0 → uniform (alpha = 0, all weights equal).
    * 1.0 → very aggressive near-source bias (alpha = 10).

    Args:
        rng:  Seeded random state.
        available_points: Array of candidate point indices.
        num_train_points: Total number of training points to return.
        min_distances: (N_gaussians,) minimum geodesic distance from source(s).
        near_source_oversample: Strength of near-source bias (0–1).

    Returns:
        Array of selected point indices (length ≤ num_train_points).
    """
    if near_source_oversample <= 0 or len(available_points) <= num_train_points:
        return rng.choice(available_points, num_train_points, replace=False)

    dists = min_distances[available_points]
    median_d = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    alpha = near_source_oversample * 10.0  # maps [0,1] → [0,10]

    weights = np.exp(-alpha * dists / median_d) + 1.0
    weights /= weights.sum()

    return rng.choice(available_points, num_train_points, replace=False, p=weights)


def normalize_neighborhood(
    valid_neighborhood: np.ndarray,
    beyond_neighborhood: np.ndarray,
    p_features: np.ndarray,
    r1_neighborhood: np.ndarray,
    p_u: float,
    r1_min_val: float,
    attributes: List[str],
    current_normalization: float,
    nn_mean: float,
    mask_constant: float,
    sh_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Normalize neighborhood features for a training example.

    This is the offline (data-generation) normalization stage:
    1. Shift geodesic distances so that the minimum visited-neighbor geodesic
       becomes zero.
    2. Scale spatial (xyz, scale, euclidean_distances) and geodesic columns by
       ``nn_mean / current_normalization`` — matching the global/per-patch factor
       used during training data generation.

    This function is the inverse of :func:`denormalize_neighborhood`.

    Args:
        valid_neighborhood: (N_valid, entry_size) array for neighbors with
            geodesic ≤ p_u (known distances).
        beyond_neighborhood: (N_beyond, entry_size) array for neighbors with
            geodesic > p_u.  Their geodesic column is overwritten with
            ``mask_constant``; spatial features are kept and scaled.
        p_features: (point_feature_size,) point feature vector.
        r1_neighborhood: (N_r1, entry_size) ring-1 neighbor features.
        p_u: Target geodesic distance for the center point.
        r1_min_val: Minimum ring-1 geodesic distance.
        attributes: Ordered list of attribute names (same as used to build
            the feature arrays).
        current_normalization: Per-patch normalization factor
            (``per_point_nn_distances[nbrs].mean()`` or global mean NN dist).
        nn_mean: Target mean nearest-neighbor distance after normalization.
        mask_constant: Sentinel value used to mark invalid/unknown geodesic.
        sh_features: Full SH feature array (needed to determine per-point SH
            dimension when ``"sh"`` is in attributes).

    Returns:
        Tuple ``(valid_neighborhood, beyond_neighborhood, p_features,
        r1_neighborhood, p_u, r1_min_val, min_input)`` where all arrays are
        normalized in-place and ``min_input`` is the geodesic shift that was
        subtracted (required for :func:`denormalize_neighborhood`).
    """
    # Step 1: Shift geodesic distances to have zero minimum (valid only)
    min_input = valid_neighborhood[:, -1].min() if len(valid_neighborhood) > 0 else p_u
    valid_neighborhood[:, -1] = valid_neighborhood[:, -1] - min_input
    p_u = p_u - min_input
    r1_min_val = r1_min_val - min_input

    # Set geodesic distance to mask_constant for beyond-p_u neighbors
    # (their spatial features remain intact)
    if len(beyond_neighborhood) > 0:
        beyond_neighborhood[:, -1] = mask_constant

    # Step 2: Scale spatial features for ALL neighbors (valid + beyond)
    for part in [valid_neighborhood, beyond_neighborhood]:
        if len(part) == 0:
            continue
        attr_index = 0
        for attr in attributes:
            if attr == "xyz":
                part[:, attr_index:attr_index + 3] = (
                    part[:, attr_index:attr_index + 3] / current_normalization
                ) * nn_mean
                attr_index += 3
            elif attr == "scale":
                part[:, attr_index:attr_index + 3] = (
                    part[:, attr_index:attr_index + 3] / current_normalization
                ) * nn_mean
                attr_index += 3
            elif attr == "normals":
                attr_index += 3
            elif attr == "_aug_normals":
                attr_index += 3  # unit vectors, no scaling
            elif attr == "opacity":
                attr_index += 1
            elif attr == "rotation":
                attr_index += 4
            elif attr == "sh":
                attr_index += sh_features.shape[1] if sh_features is not None else 0
            elif attr == "euclidean_distances":
                part[:, attr_index:attr_index + 1] = (
                    part[:, attr_index:attr_index + 1] / current_normalization
                ) * nn_mean
                attr_index += 1

    # Scale geodesic distances for valid neighborhood only
    valid_neighborhood[:, -1] = (
        valid_neighborhood[:, -1] / current_normalization
    ) * nn_mean

    # Scale ring-1 neighborhood and point features
    attr_index = 0
    for attr in attributes:
        if attr == "xyz":
            r1_neighborhood[:, attr_index:attr_index + 3] = (
                r1_neighborhood[:, attr_index:attr_index + 3] / current_normalization
            ) * nn_mean
            attr_index += 3
        elif attr == "scale":
            r1_neighborhood[:, attr_index:attr_index + 3] = (
                r1_neighborhood[:, attr_index:attr_index + 3] / current_normalization
            ) * nn_mean
            p_features[attr_index:attr_index + 3] = (
                p_features[attr_index:attr_index + 3] / current_normalization
            ) * nn_mean
            attr_index += 3
        elif attr == "normals":
            attr_index += 3
        elif attr == "_aug_normals":
            attr_index += 3  # unit vectors, no scaling
        elif attr == "opacity":
            attr_index += 1
        elif attr == "rotation":
            attr_index += 4
        elif attr == "sh":
            attr_index += sh_features.shape[1] if sh_features is not None else 0
        elif attr == "euclidean_distances":
            r1_neighborhood[:, attr_index:attr_index + 1] = (
                r1_neighborhood[:, attr_index:attr_index + 1] / current_normalization
            ) * nn_mean
            attr_index += 1

    p_u = (p_u / current_normalization) * nn_mean
    r1_min_val = (r1_min_val / current_normalization) * nn_mean

    return (
        valid_neighborhood, beyond_neighborhood, p_features,
        r1_neighborhood, p_u, r1_min_val, min_input,
    )


def denormalize_neighborhood(
    neighborhood: np.ndarray,
    point_features: np.ndarray,
    target: float,
    r1_min_val: Optional[float],
    min_input: float,
    current_normalization: float,
    nn_mean: float,
    attributes: List[str],
    mask_constant: float,
    sh_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float]]:
    """
    Invert :func:`normalize_neighborhood`.

    Converts features that were normalized by ``normalize_neighborhood`` back
    to their original (unnormalized) scale.  This is the first of two
    denormalization steps at inference time (the second being the inverse of
    ``GaussianPatchDataset._normalize_patches``).

    The inverse mapping is:
    * Geodesic (valid entries):
      ``geo_orig = geo_norm * current_normalization / nn_mean + min_input``
    * Spatial (xyz, scale, euclidean_distances):
      ``x_orig = x_norm * current_normalization / nn_mean``
    * Target and r1_min_val follow the same formula as geodesic.

    Args:
        neighborhood: (N, entry_size) combined neighborhood array (valid +
            beyond + padded).  Beyond/padded entries have geodesic ==
            ``mask_constant`` and are left unchanged.
        point_features: (point_feature_size,) normalized point feature vector.
        target: Normalized target geodesic distance (p_u).
        r1_min_val: Normalized ring-1 minimum, or None.
        min_input: The shift that was subtracted during normalization (returned
            by :func:`normalize_neighborhood`).
        current_normalization: Per-patch normalization factor used during
            normalization.
        nn_mean: Target mean nearest-neighbor distance used during normalization.
        attributes: Ordered list of attribute names.
        mask_constant: Sentinel value for invalid/unknown geodesic.
        sh_features: Full SH array (needed for SH dimension).

    Returns:
        Tuple ``(neighborhood, point_features, target, r1_min_val)`` with
        denormalized values.  Arrays are *not* modified in-place — copies are
        returned.
    """
    if nn_mean == 0:
        return neighborhood, point_features, target, r1_min_val

    scale = current_normalization / nn_mean
    neighborhood = neighborhood.copy()
    point_features = point_features.copy()

    # Denormalize spatial features
    attr_index = 0
    for attr in attributes:
        if attr == "xyz":
            neighborhood[:, attr_index:attr_index + 3] *= scale
            attr_index += 3
        elif attr == "scale":
            neighborhood[:, attr_index:attr_index + 3] *= scale
            point_features[attr_index:attr_index + 3] *= scale
            attr_index += 3
        elif attr == "normals":
            attr_index += 3
        elif attr == "_aug_normals":
            attr_index += 3  # unit vectors, no denormalization needed
        elif attr == "opacity":
            attr_index += 1
        elif attr == "rotation":
            attr_index += 4
        elif attr == "sh":
            attr_index += sh_features.shape[1] if sh_features is not None else 0
        elif attr == "euclidean_distances":
            neighborhood[:, attr_index:attr_index + 1] *= scale
            attr_index += 1

    # Denormalize geodesic distances (only for non-masked entries)
    geo_col = neighborhood[:, -1].copy()
    valid_geo = geo_col != mask_constant
    geo_col[valid_geo] = geo_col[valid_geo] * scale + min_input
    neighborhood[:, -1] = geo_col

    # Denormalize target and r1_min_val
    target = float(target) * scale + min_input
    if r1_min_val is not None:
        r1_min_val = float(r1_min_val) * scale + min_input

    return neighborhood, point_features, target, r1_min_val


def compute_gaussian_normals(
    scales: np.ndarray,
    rotations: np.ndarray
) -> np.ndarray:
    """
    Compute approximate normals for Gaussians based on their principal axis.
    Uses the direction of smallest scale as the normal.
    
    The intuition is that a Gaussian splat representing a surface will be
    "flat" in the normal direction (smallest scale).
    
    Args:
        scales: (N, 3) array of scales for each Gaussian
        rotations: (N, 4) array of quaternions (w, x, y, z) for each Gaussian
    
    Returns:
        (N, 3) array of unit normal vectors
    """
    # Import here to avoid circular imports
    from GenerateData.utils.data_generation_utils import build_rotation
    
    # Build rotation matrices from quaternions
    R = build_rotation(rotations)  # (N, 3, 3)
    
    # Get the axis corresponding to the smallest scale (most compressed direction)
    min_scale_idx = np.argmin(scales, axis=1)  # (N,)
    
    # Extract the column of R corresponding to the smallest scale axis
    # R[:, :, min_scale_idx] gives us the normal direction
    normals = np.zeros((len(scales), 3))
    for i in range(len(scales)):
        normals[i] = R[i, :, min_scale_idx[i]]
    
    return normals


def _filter_outliers_without_pu(
    nbrs_xyz: np.ndarray,
    nbrs_u: np.ndarray,
    r1_nbrs_xyz: np.ndarray,
    r1_nbrs_u: np.ndarray,
    sentinel: float = 1e10,
    max_removal_fraction: Optional[float] = None,
) -> np.ndarray:
    """
    Outlier filter that does not depend on the center point's geodesic (p_u).

    Used during inference when p_u is an artificial threshold rather than a
    real geodesic distance.

    Approach — two-stage inter-neighbor geodesic consistency:

    **Stage 0 — clean ring-1 references.**
      Ring-1 neighbors can themselves be fold neighbors.  We identify them
      by computing each ring-1 neighbor's *median* geodesic gradient against
      all other ring-1 neighbors.  Fold ring-1 neighbors will have a high
      median gradient because they are geodesically far from the majority.
      We remove ring-1 neighbors whose median gradient exceeds the
      overall-median × 5 (floored at 3.0).

    **Stage 1 — baseline gradient** from the cleaned ring-1 pairwise
      geodesic gradients.

    **Stage 2 — per-neighbor gradient check** against cleaned ring-1
      references: flag visited ring-k neighbors whose minimum gradient
      against the cleaned references exceeds an adaptive threshold.

    **Stage 3 — direct geo/euc hard cap.**
      Any visited neighbor whose *shifted* geodesic / euclidean-distance-
      from-center ratio exceeds a hard cap (50) is flagged as an outlier.
      The shift subtracts ``min(vis_u)`` so the ratio measures local
      geodesic gradient rather than absolute distance from the source.
      This catches the remaining edge cases where fold neighbors
      "validate" each other.

    **Max removal cap** (optional):
      When *max_removal_fraction* is set (e.g. 0.5 for 50 %), at most that
      fraction of *visited* neighbors may be removed.  If the stages above
      would remove more, the least-outlying flagged neighbors (lowest
      ``min_grads``) are kept to respect the cap.

    Unvisited neighbors (geodesic >= *sentinel*) are always kept — they will
    be masked as "beyond" later in the pipeline.

    Args:
        max_removal_fraction: Maximum fraction of visited neighbors that
            may be removed (0.0–1.0).  ``None`` disables the cap.

    Returns:
        Boolean inlier mask of shape ``(len(nbrs_u),)``.
    """
    eps = 1e-8
    n_nbrs = len(nbrs_u)
    inlier_mask = np.ones(n_nbrs, dtype=bool)

    # Only use visited ring-1 neighbors as the trusted reference set.
    r1_visited = r1_nbrs_u < sentinel
    if r1_visited.sum() < 2:
        # Not enough ring-1 references — fall back to geo/euc hard cap only.
        visited_nbr = nbrs_u < sentinel
        if visited_nbr.any():
            euc_from_center = LA.norm(nbrs_xyz[visited_nbr], axis=1)
            vis_geo = nbrs_u[visited_nbr]
            shifted_geo = vis_geo - vis_geo.min()
            ratio = shifted_geo / np.maximum(euc_from_center, eps)
            hard_outlier = ratio > 50.0
            if hard_outlier.any():
                idx = np.where(visited_nbr)[0]
                inlier_mask[idx[hard_outlier]] = False
        return inlier_mask

    r1_xyz_v = r1_nbrs_xyz[r1_visited]
    r1_u_v = r1_nbrs_u[r1_visited]
    n_r1_v = len(r1_u_v)

    # --- Stage 0: clean ring-1 fold neighbors ---
    # Compute each ring-1 neighbor's median geodesic gradient to all others.
    r1_eucl = LA.norm(
        r1_xyz_v[:, np.newaxis, :] - r1_xyz_v[np.newaxis, :, :], axis=2
    )                                                     # (R, R)
    r1_geo_diff = np.abs(
        r1_u_v[:, np.newaxis] - r1_u_v[np.newaxis, :]
    )                                                     # (R, R)
    r1_grad_matrix = r1_geo_diff / np.maximum(r1_eucl, eps)

    # Per-neighbor median gradient (exclude self diagonal).
    np.fill_diagonal(r1_grad_matrix, 0.0)
    if n_r1_v > 2:
        # Median of off-diagonal elements per row.
        r1_per_nbr_median = np.array([
            np.median(np.concatenate([r1_grad_matrix[i, :i],
                                       r1_grad_matrix[i, i+1:]]))
            for i in range(n_r1_v)
        ])
        overall_median = np.median(r1_per_nbr_median)
        r1_clean_thresh = max(overall_median * 5.0, 3.0)
        r1_clean_mask = r1_per_nbr_median <= r1_clean_thresh
    else:
        # Only 2 ring-1 neighbors — keep both (can't determine which is fold).
        r1_clean_mask = np.ones(n_r1_v, dtype=bool)

    # Also apply geo/euc hard cap to ring-1 themselves.
    r1_euc_from_center = LA.norm(r1_xyz_v, axis=1)
    r1_shifted_geo = r1_u_v - r1_u_v.min()
    r1_ratio = r1_shifted_geo / np.maximum(r1_euc_from_center, eps)
    r1_clean_mask &= r1_ratio <= 50.0

    if r1_clean_mask.sum() < 2:
        # Cleaning removed too many — fall back to uncleaned ring-1.
        r1_clean_mask = np.ones(n_r1_v, dtype=bool)

    r1_xyz_clean = r1_xyz_v[r1_clean_mask]
    r1_u_clean = r1_u_v[r1_clean_mask]
    n_r1_clean = len(r1_u_clean)

    # --- Stage 1: baseline gradient from cleaned ring-1 pairwise ---
    if n_r1_clean >= 2:
        r1c_eucl = LA.norm(
            r1_xyz_clean[:, np.newaxis, :] - r1_xyz_clean[np.newaxis, :, :],
            axis=2,
        )
        r1c_geo_diff = np.abs(
            r1_u_clean[:, np.newaxis] - r1_u_clean[np.newaxis, :]
        )
        r1c_grad = r1c_geo_diff / np.maximum(r1c_eucl, eps)
        triu = np.triu_indices(n_r1_clean, k=1)
        baseline = np.median(r1c_grad[triu])
    else:
        baseline = 1.0  # safe default

    # 5× baseline, floored at 3.0, capped at 50.0
    threshold = min(max(baseline * 5.0, 3.0), 50.0)

    # --- Stage 2: check each visited ring-k neighbor ---
    visited_nbr = nbrs_u < sentinel
    if not visited_nbr.any():
        return inlier_mask

    vis_xyz = nbrs_xyz[visited_nbr]             # (M_v, 3)
    vis_u = nbrs_u[visited_nbr]                 # (M_v,)

    # (M_v, R_clean) pairwise Euclidean distances to cleaned ring-1 refs
    dists = LA.norm(
        vis_xyz[:, np.newaxis, :] - r1_xyz_clean[np.newaxis, :, :], axis=2
    )
    geo_diffs = np.abs(vis_u[:, np.newaxis] - r1_u_clean[np.newaxis, :])
    grads = geo_diffs / np.maximum(dists, eps)   # (M_v, R_clean)
    min_grads = grads.min(axis=1)                # (M_v,)

    outlier = min_grads > threshold

    # --- Stage 3: geo/euc hard cap ---
    euc_from_center = LA.norm(vis_xyz, axis=1)
    shifted_geo = vis_u - vis_u.min()
    ratio = shifted_geo / np.maximum(euc_from_center, eps)
    outlier |= ratio > 50.0

    if outlier.any():
        idx = np.where(visited_nbr)[0]

        # --- Max removal cap ---
        if max_removal_fraction is not None:
            n_visited = int(visited_nbr.sum())
            max_remove = max(int(np.floor(n_visited * max_removal_fraction)), 0)
            n_flagged = int(outlier.sum())
            if n_flagged > max_remove:
                # Keep the least-outlying flagged neighbors.
                # Sort flagged indices by min_grads (ascending) and only
                # remove the *max_remove* worst (highest gradient).
                flagged_idx = np.where(outlier)[0]
                order = np.argsort(min_grads[flagged_idx])  # ascending
                # The first (n_flagged - max_remove) are "least bad" → un-flag
                keep_count = n_flagged - max_remove
                outlier[flagged_idx[order[:keep_count]]] = False

        inlier_mask[idx[outlier]] = False

    return inlier_mask


def build_inverse_ring1(ring1_nbrs: Dict[int, np.ndarray], num_points: int) -> Dict[int, np.ndarray]:
    """Build the inverse ring-1 neighbourhood map.

    For each point *v*, the inverse ring-1 is the set of all points *P* that
    have *v* in **their** ring-1 neighbourhood — i.e.
    ``inverse_ring1[v] = {P : v ∈ ring1_nbrs[P]}``.

    KNN graphs are NOT symmetric, so ``ring1_nbrs[A]`` containing B does NOT
    imply ``ring1_nbrs[B]`` contains A.  The inverse map captures the
    "who points at me" direction, identical to ``reverse_ring_neighbors`` in
    ``fast_marching.py``.

    Args:
        ring1_nbrs: Mapping ``point_idx → array of ring-1 neighbour indices``.
        num_points: Total number of points in the scene.

    Returns:
        Dict mapping each point to an ``np.ndarray`` of its inverse-ring-1
        neighbour indices.
    """
    inv: Dict[int, List[int]] = {i: [] for i in range(num_points)}
    for point_idx, nbrs in ring1_nbrs.items():
        for nbr in nbrs:
            inv[int(nbr)].append(int(point_idx))
    return {k: np.array(v, dtype=np.intp) for k, v in inv.items()}


def create_train_example(
    point_idx: int,
    geodesic_distances: np.ndarray,
    ring_nbrs: Dict[int, np.ndarray],
    ring1_nbrs: Dict[int, np.ndarray],
    gaussian_data: 'GaussianData' = None,
    patch_config: 'PatchConfig' = None,
    _stats_out: Optional[Dict] = None,
    skip_fold_check: bool = True,
    inverse_ring1_nbrs: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """
    Create a single training example for a point.

    Calling convention::

        example = create_train_example(
            point_idx, geodesic_distances, ring_nbrs, ring1_nbrs,
            gaussian_data=gdata, patch_config=pcfg,
        )

    Args:
        point_idx: Index of the point to create example for
        geodesic_distances: (N,) geodesic distances from source
        ring_nbrs: Dictionary mapping point index to ring-k neighbor indices
        ring1_nbrs: Dictionary mapping point index to ring-1 neighbor indices
        gaussian_data: ``GaussianData`` bundle (positions, scales, …).
        patch_config: ``PatchConfig`` bundle (ring, attributes, …).
        _stats_out: Optional dict for collecting outlier stats.
        skip_fold_check: Outlier filter mode (True=inference, False=training,
            None=disabled).
        inverse_ring1_nbrs: Optional inverse ring-1 map for r1_min_val.

    Returns:
        Training example array or None if invalid (e.g., too many neighbors)
    """
    if gaussian_data is None:
        raise ValueError("gaussian_data is required")
    if patch_config is None:
        raise ValueError("patch_config is required")

    # Resolve from dataclasses
    positions = gaussian_data.positions
    normals = gaussian_data.normals
    scales = gaussian_data.scales
    rotations = gaussian_data.rotations
    opacities = gaussian_data.opacities
    sh_features = gaussian_data.sh_features
    per_point_nn_distances = gaussian_data.per_point_nn_distances

    ring = patch_config.ring
    normalization_factor = patch_config.normalization_factor
    nn_mean = patch_config.nn_mean
    attributes = patch_config.attributes
    use_mahalanobis = patch_config.use_mahalanobis
    use_r1_min_val = patch_config.use_r1_min_val
    mask_attributes = patch_config.mask_attributes
    mask_constant = patch_config.mask_constant
    ring_size_mapping = patch_config.ring_size_mapping
    normalize_per_patch = patch_config.normalize_per_patch
    surface_type = patch_config.surface_type
    outlier_median_multiplier = patch_config.outlier_median_multiplier
    outlier_threshold_floor = patch_config.outlier_threshold_floor
    outlier_hard_cap = patch_config.outlier_hard_cap
    outlier_fallback_multiplier = patch_config.outlier_fallback_multiplier
    outlier_fallback_floor = patch_config.outlier_fallback_floor
    outlier_max_removal_fraction = patch_config.outlier_max_removal_fraction
    mask_outliers_only = patch_config.mask_outliers_only
    if patch_config.disable_outlier_filtering:
        skip_fold_check = None  # sentinel: skip ALL outlier filtering
    max_num_nbrs = get_ring_size_mapping(ring, use_mahalanobis, ring_size_mapping)
    
    # Get neighbors
    nbrs = ring_nbrs[point_idx]
    r1_nbrs = ring1_nbrs[point_idx]
    
    # Get point data
    p_xyz = positions[point_idx]
    p_u = geodesic_distances[point_idx]
    
    # Get neighbor relative positions
    nbrs_xyz = positions[nbrs] - p_xyz
    nbrs_euclidean_distances = LA.norm(nbrs_xyz, axis=1)
    nbrs_u = geodesic_distances[nbrs]

    # Get ring-1 neighbor data
    r1_nbrs_xyz = positions[r1_nbrs] - p_xyz
    r1_nbrs_euclidean_distances = LA.norm(r1_nbrs_xyz, axis=1)
    r1_nbrs_u = geodesic_distances[r1_nbrs]
    
    # ---- Outlier filtering ----
    # Remove neighbors from other surface folds (Euclidean-close but
    # geodesically inconsistent with the center point).
    eps = 1e-8

    if skip_fold_check is None:
        # Outlier filtering entirely disabled (e.g. polynomial surfaces
        # with no folds).
        inlier_mask = np.ones(len(nbrs_u), dtype=bool)
        num_removed = 0
    elif skip_fold_check:
        # p_u is artificial (e.g. build_input sets it to max(visited)+1).
        # Use the p_u-independent filter based on inter-neighbor consistency.
        inlier_mask = _filter_outliers_without_pu(
            nbrs_xyz, nbrs_u, r1_nbrs_xyz, r1_nbrs_u,
            max_removal_fraction=outlier_max_removal_fraction,
        )
        num_removed = int((~inlier_mask).sum())
    else:
        # Training path: p_u is the real geodesic distance.
        # |geo(S,N) - geo(S,P)| / euc(N,P) should be bounded on the same
        # surface sheet.  Fold neighbors violate this.
        geo_discrepancy = np.abs(nbrs_u - p_u) / np.maximum(nbrs_euclidean_distances, eps)
        r1_geo_discrepancy = np.abs(geodesic_distances[r1_nbrs] - p_u) / np.maximum(r1_nbrs_euclidean_distances, eps)
        median_disc = np.mean(r1_geo_discrepancy)
        adaptive_threshold = max(median_disc * outlier_median_multiplier, outlier_threshold_floor)
        outlier_threshold = min(adaptive_threshold, outlier_hard_cap)
        inlier_mask = geo_discrepancy <= outlier_threshold
        num_removed = int((~inlier_mask).sum())
        # Fallback: if filter removes everything, relax threshold
        if inlier_mask.sum() == 0:
            inlier_mask = geo_discrepancy <= max(median_disc * outlier_fallback_multiplier, outlier_fallback_floor)
            num_removed = int((~inlier_mask).sum())

    if _stats_out is not None:
        _stats_out['num_removed'] = num_removed
    if mask_outliers_only:
        # Keep all neighbors but mask outlier geodesics to mask_constant.
        # Spatial (xyz) features are preserved.
        nbrs_u[~inlier_mask] = mask_constant
    else:
        nbrs = nbrs[inlier_mask]
        nbrs_xyz = nbrs_xyz[inlier_mask]
        nbrs_euclidean_distances = nbrs_euclidean_distances[inlier_mask]
        nbrs_u = nbrs_u[inlier_mask]
    

    


    # Sanity check (training only): the ring-1 neighbor with minimum geodesic
    # must not be a fold outlier.  Ring-1 neighbors bypass the outlier filter
    # above, so we skip this example if the r1_min neighbor looks like a fold
    # neighbor — otherwise r1_min_val would be unreliable.
    if skip_fold_check is not None and not skip_fold_check and len(r1_nbrs_u) > 0:
        r1_disc = np.abs(r1_nbrs_u - p_u) / np.maximum(r1_nbrs_euclidean_distances, eps)
        r1_min_idx = np.argmin(r1_nbrs_u)
        if r1_disc[r1_min_idx] > outlier_threshold:
            Warning(f"Skipping point {point_idx} because its ring-1 minimum neighbor looks like a fold outlier (disc={r1_disc[r1_min_idx]:.2f} > threshold={outlier_threshold:.2f})")
            print(f"Ring-1 neighbors' geodesic discrepancies: {r1_disc}")
            return None  # r1_min_val is contaminated by a fold neighbor
    
    # Build feature arrays based on attributes
    nbrs_features = []
    r1_nbrs_features = []
    p_features = []
    
    for attr in attributes:
        if attr == "xyz":
            nbrs_features.append(nbrs_xyz)
            r1_nbrs_features.append(r1_nbrs_xyz)
            p_features.append(np.zeros((3,)))  # Point relative position is zero
        elif attr == "opacity":
            if opacities is None:
                raise ValueError("opacities data required for 'opacity' attribute")
            nbrs_features.append(opacities[nbrs])
            r1_nbrs_features.append(opacities[r1_nbrs])
            p_features.append(opacities[point_idx])
        elif attr == "scale":
            if scales is None:
                raise ValueError("scales data required for 'scale' attribute")
            nbrs_features.append(scales[nbrs])
            r1_nbrs_features.append(scales[r1_nbrs])
            p_features.append(scales[point_idx])
        elif attr == "rotation":
            if rotations is None:
                raise ValueError("rotations data required for 'rotation' attribute")
            nbrs_features.append(rotations[nbrs])
            r1_nbrs_features.append(rotations[r1_nbrs])
            p_features.append(rotations[point_idx])
        elif attr == "sh":
            if sh_features is None:
                raise ValueError("sh_features data required for 'sh' attribute")
            nbrs_features.append(sh_features[nbrs])
            r1_nbrs_features.append(sh_features[r1_nbrs])
            p_features.append(sh_features[point_idx])
        elif attr == "normals":
            if normals is None:
                raise ValueError("normals data required for 'normals' attribute")
            nbrs_features.append(normals[nbrs])
            r1_nbrs_features.append(normals[r1_nbrs])
            p_features.append(normals[point_idx])
        elif attr == "euclidean_distances":
            nbrs_features.append(np.expand_dims(nbrs_euclidean_distances, axis=1))
            r1_nbrs_features.append(np.expand_dims(r1_nbrs_euclidean_distances, axis=1))
            p_features.append(np.array([0.0]))  # Point to itself distance is zero
        elif attr == "_aug_normals":
            if surface_type is None:
                raise ValueError("surface_type required for '_aug_normals' attribute")
            from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial_normal
            # Compute analytical normals from RAW positions (before centering)
            raw_nbr_pos = positions[nbrs]  # (N, 3) absolute positions
            nbr_normals = evaluate_polynomial_normal(
                raw_nbr_pos[:, 0], raw_nbr_pos[:, 1], surface_type)
            nbrs_features.append(nbr_normals)
            # Ring-1 neighbors
            raw_r1_pos = positions[r1_nbrs]
            r1_normals = evaluate_polynomial_normal(
                raw_r1_pos[:, 0], raw_r1_pos[:, 1], surface_type)
            r1_nbrs_features.append(r1_normals)
            # Point normal
            p_normal = evaluate_polynomial_normal(
                np.array([p_xyz[0]]), np.array([p_xyz[1]]), surface_type)
            p_features.append(p_normal.flatten())
    
    # Add geodesic distances at the end (always included for neighbors)
    nbrs_features.append(np.expand_dims(nbrs_u, axis=1))
    r1_nbrs_features.append(np.expand_dims(r1_nbrs_u, axis=1))
    
    # Concatenate all features
    neighborhood = np.concatenate(nbrs_features, axis=1)
    r1_neighborhood = np.concatenate(r1_nbrs_features, axis=1)
    p_features = np.concatenate(p_features, axis=0) if p_features else np.array([])
    
    # Separate valid (geo <= p_u) from beyond (geo > p_u) neighbors.
    # Keep ALL neighbors' spatial features — only mask geodesic for beyond ones.
    # This preserves the full spatial extent of the patch so that normalization
    # (max_dist) is consistent whether we see all neighbors or only a few
    # visited ones during Fast Marching inference.
    valid_mask_arr = neighborhood[:, -1] <= p_u
    valid_neighborhood = neighborhood[valid_mask_arr]
    beyond_neighborhood = neighborhood[~valid_mask_arr]

    # If still more neighbors than max_num_nbrs, keep only the closest
    # by Euclidean distance.
    if len(valid_neighborhood) > max_num_nbrs:
        keep_idx = np.argsort(nbrs_euclidean_distances[valid_mask_arr])[:max_num_nbrs]
        valid_neighborhood = valid_neighborhood[keep_idx]
        beyond_neighborhood = []
    
    # Get ring-1 minimum for dropout augmentation.
    # Use inverse ring: for point v, inverse_ring1 is the set of all points P
    # that have v in *their* ring-1 neighbourhood.  This matches the direction
    # used by Fast Marching (reverse_ring_neighbors).
    if inverse_ring1_nbrs is not None and point_idx in inverse_ring1_nbrs:
        inv_nbrs = inverse_ring1_nbrs[point_idx]
        if len(inv_nbrs) > 0:
            inv_nbrs_u = geodesic_distances[inv_nbrs]
            r1_min_val = inv_nbrs_u.min()
        else:
            r1_min_val = r1_nbrs_u.min() if len(r1_nbrs_u) > 0 else p_u
    else:
        r1_min_val = r1_nbrs_u.min() if len(r1_nbrs_u) > 0 else p_u

    # Determine normalization factor: per-patch or global
    if normalize_per_patch and per_point_nn_distances is not None:
        current_normalization = per_point_nn_distances[nbrs].mean() if len(nbrs) > 0 else normalization_factor
    else:
        current_normalization = normalization_factor

    # Normalize: shift + scale geodesic and spatial features
    (valid_neighborhood, beyond_neighborhood, p_features, r1_neighborhood,
     p_u, r1_min_val, _min_input) = normalize_neighborhood(
        valid_neighborhood=valid_neighborhood,
        beyond_neighborhood=beyond_neighborhood,
        p_features=p_features,
        r1_neighborhood=r1_neighborhood,
        p_u=p_u,
        r1_min_val=r1_min_val,
        attributes=attributes,
        current_normalization=current_normalization,
        nn_mean=nn_mean,
        mask_constant=mask_constant,
        sh_features=sh_features,
    )

    # Combine: valid first, then beyond-p_u (with real xyz, masked geodesic)
    # Fill remaining slots with fully masked padding entries
    num_beyond_to_keep = min(len(beyond_neighborhood), max_num_nbrs - len(valid_neighborhood))
    parts = [valid_neighborhood]
    if num_beyond_to_keep > 0:
        parts.append(beyond_neighborhood[:num_beyond_to_keep])
    neighborhood = np.vstack(parts)
    
    # Pad remaining slots by duplicating random existing neighbors and
    # masking only their geodesic distance.  This keeps all other features
    # (xyz, normals, opacity, …) looking like real data so that downstream
    # normalization (opacity min-max, scale pc_norm, SH clamping) is not
    # distorted by arbitrary constant padding values.
    pad_num = max_num_nbrs - neighborhood.shape[0]
    if pad_num > 0:
        if neighborhood.shape[0] == 0:
            # Safety net: should never happen after the outlier-filter fallback,
            # but if a ring genuinely has no neighbors, fill with fully-masked
            # rows rather than crashing.
            entry_size = neighborhood.shape[1]
            neighborhood = np.zeros((max_num_nbrs, entry_size), dtype=neighborhood.dtype)
            neighborhood[:, -1] = mask_constant
            pad_num = 0
        else:
            source_indices = np.random.choice(neighborhood.shape[0], pad_num, replace=True)
            padding = neighborhood[source_indices].copy()
            padding[:, -1] = mask_constant   # only geodesic distance is masked
            neighborhood = np.vstack([neighborhood, padding])
    
    assert neighborhood.shape[0] == max_num_nbrs, f"Expected {max_num_nbrs} neighbors, got {neighborhood.shape[0]}"
    
    # Construct final example: [neighborhood_features..., point_features, r1_min_val?, target]
    if use_r1_min_val:
        example = np.append(neighborhood.flatten(), p_features.flatten())
        example = np.append(example, r1_min_val)
        example = np.append(example, p_u)
    else:
        example = np.append(neighborhood.flatten(), p_features.flatten())
        example = np.append(example, p_u)
    
    return example


def generate_training_examples(
    geodesic_data: Dict,
    ring_nbrs_dict: Dict,
    ring1_nbrs: Dict,
    num_iterations: int,
    num_sources: int,
    num_train_points: int,
    gaussian_data: 'GaussianData' = None,
    patch_config: 'PatchConfig' = None,
    seed: int = 42,
    verbose: bool = True,
    num_workers: Optional[int] = None,
    _stats_out: Optional[Dict] = None,
    near_source_oversample: float = 0.0,
) -> np.ndarray:
    """
    Generate training examples by randomly sampling sources and training points.

    Calling convention::

        examples = generate_training_examples(
            geodesic_data, ring_nbrs_dict, ring1_nbrs,
            num_iterations, num_sources, num_train_points,
            gaussian_data=gdata, patch_config=pcfg, seed=42,
        )

    Args:
        geodesic_data: Dictionary with precomputed geodesic distances
        ring_nbrs_dict: Dictionary of ring -> {point_idx -> neighbor_indices}
        ring1_nbrs: Dictionary of point_idx -> ring-1 neighbor indices
        num_iterations: Number of training iterations
        num_sources: Number of source points per iteration
        num_train_points: Number of training points per iteration
        gaussian_data: ``GaussianData`` bundle.
        patch_config: ``PatchConfig`` bundle.
        seed: Random seed
        verbose: Print progress information
        num_workers: Number of parallel workers.
        _stats_out: Optional dict for collecting outlier stats.
        near_source_oversample: Fraction (0–1) of ``num_train_points`` that
            are sampled with inverse-geodesic-distance weighting so that
            points closer to the source are more likely to be selected.
            0.0 (default) means uniform sampling (legacy behaviour).
            0.5 means half the points are distance-weighted, half uniform.

    Returns:
        (M, D) array of training examples
    """
    if gaussian_data is None:
        raise ValueError("gaussian_data is required")
    if patch_config is None:
        raise ValueError("patch_config is required")

    positions = gaussian_data.positions
    ring = patch_config.ring
    attributes = patch_config.attributes

    if verbose:
        print(f"\nGenerating training examples (ring {ring}):")
        print(f"  Iterations: {num_iterations}")
        print(f"  Sources per iteration: {num_sources}")
        print(f"  Train points per iteration: {num_train_points}")
        print(f"  Attributes: {', '.join(attributes)}")
        if near_source_oversample > 0:
            print(f"  Near-source oversample ratio: {near_source_oversample:.0%}")

    # ------------------------------------------------------------------
    # Decide how many workers to use
    # ------------------------------------------------------------------
    if num_workers is None:
        num_workers = min(int(0.1 * multiprocessing.cpu_count()), num_iterations, 2)
    num_workers = max(1, min(num_workers, num_iterations))

    # ------------------------------------------------------------------
    # Sequential path (1 worker or very few iterations)
    # ------------------------------------------------------------------
    if num_workers <= 1:
        return _generate_sequential(
            gaussian_data, patch_config,
            geodesic_data, ring_nbrs_dict, ring1_nbrs,
            num_iterations, num_sources, num_train_points,
            seed, verbose,
            _stats_out=_stats_out,
            near_source_oversample=near_source_oversample,
        )

    # ------------------------------------------------------------------
    # Parallel path – distribute iterations across workers
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Workers: {num_workers}")

    # Precompute inverse ring-1 for r1_min_val computation
    _inv_ring1 = build_inverse_ring1(ring1_nbrs, len(gaussian_data.positions))

    # Pack shared read-only data into a dict (inherited via fork on Linux)
    global _mp_shared
    _mp_shared = {
        'positions': gaussian_data.positions,
        'normals': gaussian_data.normals,
        'geodesic_data': geodesic_data,
        'ring_nbrs_dict': ring_nbrs_dict,
        'ring1_nbrs': ring1_nbrs,
        'inverse_ring1_nbrs': _inv_ring1,
        'scales': gaussian_data.scales,
        'rotations': gaussian_data.rotations,
        'opacities': gaussian_data.opacities,
        'sh_features': gaussian_data.sh_features,
        'per_point_nn_distances': gaussian_data.per_point_nn_distances,
    }

    # Split iterations into roughly equal chunks
    chunk_boundaries = np.linspace(0, num_iterations, num_workers + 1, dtype=int)
    params_list = []
    for w in range(num_workers):
        params_list.append((
            int(chunk_boundaries[w]),
            int(chunk_boundaries[w + 1]),
            seed,
            patch_config.ring,
            num_sources,
            num_train_points,
            patch_config.normalization_factor,
            patch_config.nn_mean,
            list(patch_config.attributes),
            patch_config.use_mahalanobis,
            patch_config.use_r1_min_val,
            list(patch_config.mask_attributes),
            patch_config.mask_constant,
            patch_config.ring_size_mapping,
            patch_config.normalize_per_patch,
            near_source_oversample,
            patch_config.surface_type,
            patch_config.disable_outlier_filtering,
            patch_config.outlier_median_multiplier,
            patch_config.outlier_threshold_floor,
            patch_config.outlier_hard_cap,
            patch_config.outlier_fallback_multiplier,
            patch_config.outlier_fallback_floor,
        ))

    all_examples: List[np.ndarray] = []
    agg_stats: Dict = {'patches_with_outliers': 0, 'total_neighbors_removed': 0, 'total_patches': 0}
    with multiprocessing.Pool(
        num_workers,
        initializer=_init_mp_worker,
        initargs=(_mp_shared,),
    ) as pool:
        results_iter = pool.imap_unordered(_generate_chunk, params_list)
        if verbose:
            results_iter = tqdm(
                results_iter, total=num_workers, desc="Generating examples (parallel)"
            )
        for chunk_result in results_iter:
            chunk_examples, chunk_stats = chunk_result
            all_examples.extend(chunk_examples)
            for k in agg_stats:
                agg_stats[k] += chunk_stats.get(k, 0)

    if _stats_out is not None:
        _stats_out.update(agg_stats)
    _mp_shared = None  # release reference

    if len(all_examples) == 0:
        raise ValueError("No valid examples generated!")

    return np.vstack(all_examples)


def _generate_sequential(
    gaussian_data: 'GaussianData', patch_config: 'PatchConfig',
    geodesic_data, ring_nbrs_dict, ring1_nbrs,
    num_iterations, num_sources, num_train_points,
    seed, verbose,
    _stats_out: Optional[Dict] = None,
    near_source_oversample: float = 0.0,
) -> np.ndarray:
    """Sequential fallback for ``generate_training_examples``."""
    np.random.seed(seed)

    positions = gaussian_data.positions
    ring = patch_config.ring

    num_gaussians = len(positions)
    all_source_indices = geodesic_data['source_gaussian_indices']
    all_geodesic_distances = geodesic_data['geodesic_distances']

    # Precompute inverse ring-1 for r1_min_val computation
    inv_ring1 = build_inverse_ring1(ring1_nbrs, num_gaussians)

    examples: List[np.ndarray] = []
    patches_with_outliers = 0
    total_neighbors_removed = 0
    total_patches = 0

    iterator = tqdm(range(num_iterations), desc="Generating examples") if verbose else range(num_iterations)

    for i in iterator:
        if num_sources <= len(all_source_indices):
            selected_source_idxs = np.random.choice(
                len(all_source_indices), num_sources, replace=False
            )
        else:
            if i == 0 and verbose:
                print(f"Warning: Requested {num_sources} sources but only {len(all_source_indices)} available")
            selected_source_idxs = np.arange(len(all_source_indices))

        selected_distances = all_geodesic_distances[selected_source_idxs]
        min_distances = selected_distances.min(axis=0)

        source_gaussian_idxs = all_source_indices[selected_source_idxs]
        available_points = np.setdiff1d(np.arange(num_gaussians), source_gaussian_idxs)

        if len(available_points) < num_train_points:
            train_points = available_points
        else:
            train_points = _sample_train_points(
                np.random.RandomState(seed + i), available_points,
                num_train_points, min_distances, near_source_oversample,
            )

        for point_idx in train_points:
            _ex_stats = {}
            example = create_train_example(
                point_idx, min_distances,
                ring_nbrs_dict[ring], ring1_nbrs,
                gaussian_data=gaussian_data, patch_config=patch_config,
                _stats_out=_ex_stats,
                inverse_ring1_nbrs=inv_ring1,
            )
            if example is not None:
                examples.append(example)
                total_patches += 1
                nr = _ex_stats.get('num_removed', 0)
                if nr > 0:
                    patches_with_outliers += 1
                    total_neighbors_removed += nr

    if _stats_out is not None:
        _stats_out.update({
            'patches_with_outliers': patches_with_outliers,
            'total_neighbors_removed': total_neighbors_removed,
            'total_patches': total_patches,
        })
    if len(examples) == 0:
        raise ValueError("No valid examples generated!")

    return np.vstack(examples)


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file with command-line arguments.
    Command-line arguments take precedence over config file.
    
    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments
    
    Returns:
        Updated arguments namespace
    """
    # Map config keys to argument names
    config_mapping = {
        'gaussian_output': 'gaussian_output',
        'geodesic_data': 'geodesic_data',
        'output_dir': 'output_dir',
        'iteration': 'iteration',
        'num_iterations': 'num_iterations',
        'num_sources': 'num_sources',
        'num_train_points': 'num_train_points',
        'seed': 'seed',
        'use_mahalanobis': 'use_mahalanobis',
        'use_r1_min_val': 'use_r1_min_val',
        'n_neighbors': 'n_neighbors',
        'rings': 'rings',
        'attributes': 'attributes',
        'constant_val': 'constant_val',
        'mask_attributes': 'mask_attributes',
        'mask_constant': 'mask_constant',
        'normalize_per_patch': 'normalize_per_patch',
        'num_output_workers': 'num_output_workers',
        'adaptive_target_ring': 'adaptive_target_ring',
        'adaptive_target_neighbors': 'adaptive_target_neighbors',
        'adaptive_k_boost': 'adaptive_k_boost',
        'adaptive_max_mean_cut': 'adaptive_max_mean_cut',
        'adaptive_max_steps': 'adaptive_max_steps',
        'disable_outlier_filtering': 'disable_outlier_filtering',
        'outlier_median_multiplier': 'outlier_median_multiplier',
        'outlier_threshold_floor': 'outlier_threshold_floor',
        'outlier_hard_cap': 'outlier_hard_cap',
        'outlier_fallback_multiplier': 'outlier_fallback_multiplier',
        'outlier_fallback_floor': 'outlier_fallback_floor',
        'outlier_max_removal_fraction': 'outlier_max_removal_fraction',
        'mask_outliers_only': 'mask_outliers_only',
    }
    
    # Apply config values for path arguments if not explicitly set
    for config_key, arg_name in config_mapping.items():
        if config_key in config and config[config_key] is not None:
            current_value = getattr(args, arg_name, None)
            
            # For paths and iteration, only use config if command-line arg is None
            if arg_name in ['gaussian_output', 'geodesic_data', 'output_dir', 'iteration']:
                if current_value is None:
                    setattr(args, arg_name, config[config_key])
            # Check if argument has default value (wasn't set on command line)
            elif arg_name == 'rings' and current_value == [2, 3]:
                setattr(args, arg_name, config[config_key])
            elif arg_name == 'attributes' and current_value == ['xyz']:
                setattr(args, arg_name, config[config_key])
            elif arg_name == 'mask_attributes' and current_value == []:
                setattr(args, arg_name, config[config_key])
            elif arg_name in ['num_iterations', 'num_sources', 'num_train_points', 'seed', 
                              'n_neighbors', 'constant_val', 'mask_constant',
                              'num_output_workers',
                              'adaptive_target_ring', 'adaptive_target_neighbors',
                              'adaptive_k_boost', 'adaptive_max_mean_cut',
                              'adaptive_max_steps',
                              'outlier_median_multiplier', 'outlier_threshold_floor',
                              'outlier_hard_cap', 'outlier_fallback_multiplier',
                              'outlier_fallback_floor',
                              'outlier_max_removal_fraction']:
                setattr(args, arg_name, config[config_key])
            elif arg_name in ['mask_outliers_only'] and not current_value:
                setattr(args, arg_name, config[config_key])
            elif arg_name in ['use_mahalanobis', 'use_r1_min_val', 'disable_outlier_filtering'] and not current_value:
                setattr(args, arg_name, config[config_key])
    
    # Store ring_size_mapping if present
    if 'ring_size_mapping' in config:
        args.ring_size_mapping = config['ring_size_mapping']
    
    # Store dataset info if present
    if 'dataset' in config:
        args.dataset_info = config['dataset']
    
    return args


def compute_scale_stats(scales: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics about Gaussian scales.
    
    Args:
        scales: (N, 3) array of Gaussian scales
    
    Returns:
        Dictionary with mean, min, max, std statistics
    """
    return {
        'mean': float(scales.mean()),
        'min': float(scales.min()),
        'max': float(scales.max()),
        'std': float(scales.std())
    }
