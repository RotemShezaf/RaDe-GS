"""
Input builder for creating model inputs from Gaussian splat data.

This module handles:
- Building neighborhood features for model input
- Creating valid masks for attention
- Normalizing features appropriately via GaussianPatchDataset
- Extracting only visited neighbors for Fast Marching

The full two-stage normalization pipeline is:

Stage 1 (offline, inside ``create_train_example``):
    Applied by :func:`DataSets.utils.training_patches_helpers.normalize_neighborhood`.
    1. Shift geodesic distances so that the minimum visited-neighbor geodesic
       becomes zero (shift = ``min_input``).
    2. Scale spatial and geodesic values by ``nn_mean / current_normalization``.

Stage 2 (online, inside ``GaussianPatchDataset.get_item_from_raw``):
    Applied by :meth:`DataSets.gaussian_dataset.GaussianPatchDataset._normalize_patches`.
    3. Attribute-specific normalization (opacity min-max, SH scaling, scale PCA).
    4. Centre xyz by the point position, then divide xyz and geodesic by
       ``max_dist`` (max Euclidean distance to any real neighbour).

Denormalization must be applied in reverse: undo stage 2 first (via
:meth:`GaussianPatchDataset._denormalize_results`), then undo stage 1
(via :func:`denormalize_neighborhood`).
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class GaussianInputBuilder:
    """
    Builder for creating model inputs from Gaussian splat data.

    Converts Gaussian splat data and neighborhood information into the format
    expected by the GaussianPatchTransformer model.  The builder owns a
    :class:`DataSets.gaussian_dataset.GaussianPatchDataset` instance that is
    initialized (in ``inference_only`` mode, without loading ``.npy`` files)
    from ``dataset_config`` at construction time.  All attribute names,
    neighbor counts, and normalization parameters are read directly from the
    dataset config so callers do not need to repeat them.

    For Fast Marching propagation the builder filters neighbors to only those
    whose geodesic distance is already known (visited), while keeping all
    ring-k neighbours' spatial features visible to the model (beyond-p_u
    entries retain their xyz columns but receive ``mask_constant`` for their
    geodesic column).

    Usage::

        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config="DataSets/configs/polynomial/combined_polynomial_ring2.yaml",
            ring=2,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )

        # During Fast Marching step for point u:
        result = builder.build_input(
            point_idx=u,
            all_neighbor_indices=ring_nbrs[u],
            neighbor_distances=distances[ring_nbrs[u]],
            visited_mask=visited,
        )
        if result is not None:
            neighborhood, point_features, valid_mask, build_info = result
            raw_pred = model(neighborhood, point_features, valid_mask).item()
            dist_u = builder.denormalize_result(raw_pred, build_info)
    """

    def __init__(
        self,
        positions: np.ndarray,
        dataset_config: Union[str, Path, Dict],
        ring: int,
        normalization_factor: float = 1.0,
        scales: Optional[np.ndarray] = None,
        rotations: Optional[np.ndarray] = None,
        opacities: Optional[np.ndarray] = None,
        sh_features: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        per_point_nn_distances: Optional[np.ndarray] = None,
        inference_transforms=None,
        device: Optional[str] = None,
        # --- Inference transforms from config (alternative to callable) ---
        transforms_config: Optional[List] = None,
    ):
        """
        Initialize the input builder.

        Creates an inference-only :class:`DataSets.gaussian_dataset.GaussianPatchDataset`
        from the given config.  All attribute names, ``mask_constant``,
        ``max_neighbors``, and ``nn_mean`` are read from the dataset, so they
        do not need to be passed separately.

        When the config contains ``n_neighbors``, ring-1 and ring-k neighbours
        are computed internally (with optional adaptive kNN).  The computed
        ``mean_nn_dist`` and ``per_point_nn_distances`` override the
        ``normalization_factor`` and ``per_point_nn_distances`` arguments.
        The resulting neighbourhoods are stored as :attr:`ring1_neighbors`
        and :attr:`ring_neighbors`.

        When ``transforms_config`` is provided (and ``inference_transforms``
        is ``None``), deterministic inference transforms are built
        automatically from the training transforms config list.

        Args:
            positions: Gaussian centre positions, shape ``(N, 3)``.
            dataset_config: Path to a dataset config YAML file **or** a config
                dict (same format as used by ``GaussianPatchDataset``).
            ring: Ring level (determines max neighbours).  Must match the ring
                used to train the checkpoint.
            normalization_factor: Global mean nearest-neighbour distance used
                as the per-patch normalization fallback when
                ``per_point_nn_distances`` is not provided.
            scales: Gaussian scale vectors, shape ``(N, 3)``, optional.
            rotations: Gaussian rotations as quaternions ``(w,x,y,z)``,
                shape ``(N, 4)``, optional.
            opacities: Gaussian opacities, shape ``(N,)`` or ``(N, 1)``,
                optional.
            sh_features: Spherical-harmonics feature vectors, shape
                ``(N, K)``, optional.
            normals: Surface normals, shape ``(N, 3)``, optional.
            per_point_nn_distances: Per-point nearest-neighbour distances,
                shape ``(N,)``.  When provided, per-patch normalization is
                used inside ``create_train_example``, exactly matching the
                training setting ``normalize_per_patch=True``.
            inference_transforms: Deterministic transform callable applied to
                ``(neighborhood, point_features)`` after dataset normalization
                (e.g. ``GaussianPatchCanonicalRotate``).
            device: Torch device string (e.g. ``'cuda'``, ``'cpu'``).
                Defaults to CUDA when available, otherwise CPU.
            transforms_config: Training transform config list (from YAML).
                Used to build inference transforms when ``inference_transforms``
                is ``None``.
        """
        from DataSets.gaussian_dataset import GaussianPatchDataset

        # ── Build inference transforms from config if needed ──────────
        if inference_transforms is None and transforms_config is not None:
            from models.const import build_inference_transforms
            # Load dataset config to get mask_constant / attributes for
            # build_inference_transforms — peek into the config before
            # creating the full dataset.
            if isinstance(dataset_config, dict):
                _ds_cfg = dataset_config
            else:
                import yaml
                with open(dataset_config) as _f:
                    _ds_cfg = yaml.safe_load(_f)
            _mask_constant = float(_ds_cfg.get('mask_constant', -10.0))
            _attributes = _ds_cfg.get('attributes', ['xyz'])
            inference_transforms = build_inference_transforms(
                transforms_config, _attributes, _mask_constant,
            )

        # Create dataset from config (inference_only skips .npy data loading).
        # Inference transforms are passed to the dataset so they are applied
        # once inside get_item_from_raw(), matching the __getitem__ path.
        self.dataset = GaussianPatchDataset(
            config=dataset_config,
            ring=ring,
            transform=inference_transforms,
            inference_only=True,
        )

        # Read all configuration from the dataset so callers don't duplicate it
        self.attributes: List[str] = self.dataset.attributes
        self.mask_constant: float = self.dataset.mask_constant
        self.nn_mean: float = float(self.dataset.config.get('nn_mean', 1.0))
        self.ring: int = ring
        self.max_neighbors: int = self.dataset.max_neighbors

        # Gaussian splat data
        self.num_points: int = len(positions)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.scales = scales
        self.rotations = rotations
        self.opacities = (
            np.asarray(opacities).flatten() if opacities is not None else None
        )
        self.sh_features = sh_features
        self.normals = normals
        self.normalization_factor = float(normalization_factor)
        self.per_point_nn_distances = (
            np.asarray(per_point_nn_distances, dtype=np.float32)
            if per_point_nn_distances is not None else None
        )

        # ── Resolve kNN / adaptive params from config ──
        _cfg = self.dataset.config
        n_neighbors = _cfg.get('n_neighbors', None)
        use_mahalanobis = _cfg.get('use_mahalanobis', False)
        adaptive_target_ring = _cfg.get('adaptive_target_ring', None)
        adaptive_target_neighbors = _cfg.get('adaptive_target_neighbors', None)
        adaptive_k_boost = _cfg.get('adaptive_k_boost', 20)
        adaptive_max_mean_cut = _cfg.get('adaptive_max_mean_cut', 2.0)
        adaptive_max_steps = _cfg.get('adaptive_max_steps', 5)

        # ── Compute ring neighbours if n_neighbors was provided ───────
        self.ring1_neighbors: Optional[Dict[int, np.ndarray]] = None
        self.ring_neighbors: Optional[Dict[int, np.ndarray]] = None
        if n_neighbors is not None:
            self._compute_neighborhoods(
                n_neighbors=n_neighbors,
                use_mahalanobis=use_mahalanobis,
                adaptive_target_ring=adaptive_target_ring,
                adaptive_target_neighbors=adaptive_target_neighbors,
                adaptive_k_boost=adaptive_k_boost,
                adaptive_max_mean_cut=adaptive_max_mean_cut,
                adaptive_max_steps=adaptive_max_steps,
            )

        # Pre-build dataclass bundles for create_train_example calls
        from DataSets.utils.training_patches_helpers import GaussianData, PatchConfig

        # create_train_example expects opacities with shape (N, 1)
        opacities_2d = (
            self.opacities.reshape(-1, 1)
            if self.opacities is not None
            else None
        )
        self._gaussian_data = GaussianData(
            positions=self.positions,
            scales=self.scales,
            rotations=self.rotations,
            opacities=opacities_2d,
            normals=self.normals,
            sh_features=self.sh_features,
            per_point_nn_distances=self.per_point_nn_distances,
        )
        self._patch_config = PatchConfig(
            ring=self.ring,
            normalization_factor=self.normalization_factor,
            nn_mean=self.nn_mean,
            attributes=self.attributes,
            use_mahalanobis=self.dataset.config.get('use_mahalanobis', False),
            use_r1_min_val=self.dataset.use_r1_min,
            mask_attributes=self.dataset.config.get('mask_attributes', []),
            mask_constant=self.mask_constant,
            ring_size_mapping=self.dataset.config.get('ring_size_mapping', None),
            normalize_per_patch=self.per_point_nn_distances is not None,
            disable_outlier_filtering=self.dataset.config.get('disable_outlier_filtering', False),
        )

        self.inference_transforms = inference_transforms
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_neighborhoods(
        self,
        n_neighbors: int,
        use_mahalanobis: Optional[bool],
        adaptive_target_ring: Optional[int],
        adaptive_target_neighbors: Optional[int],
        adaptive_k_boost: int,
        adaptive_max_mean_cut: float,
        adaptive_max_steps: int,
    ) -> None:
        """Compute ring-1 and ring-k neighbours and update internal state.

        Sets :attr:`ring1_neighbors`, :attr:`ring_neighbors`,
        :attr:`normalization_factor`, and :attr:`per_point_nn_distances`.
        """
        # Several directories on sys.path (DataSets/, geodesic_propagation/)
        # contain a ``utils/`` regular package (with __init__.py) that shadows
        # the project-root ``utils/`` namespace package (no __init__.py).
        # This breaks ``from utils.general_utils import …`` inside
        # data_generation_utils.  Temporarily remove offending paths and evict
        # any stale ``utils`` entries from the module cache.
        import sys as _sys
        import os as _os

        _project_root = _os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))
        )
        _paths_removed = []
        for _p in list(_sys.path):
            if _os.path.normpath(_p) == _os.path.normpath(_project_root):
                continue  # keep project root
            _candidate = _os.path.join(_p, 'utils', '__init__.py')
            if _os.path.isfile(_candidate):
                _sys.path.remove(_p)
                _paths_removed.append(_p)

        # Evict any cached 'utils' that came from a non-root location.
        for _k in list(_sys.modules):
            if _k == 'utils' or _k.startswith('utils.'):
                _m = _sys.modules[_k]
                _mpath = list(getattr(_m, '__path__', []))
                _mfile = getattr(_m, '__file__', '') or ''
                _all_locs = _mpath + ([_mfile] if _mfile else [])
                if any(
                    not _os.path.normpath(loc).startswith(
                        _os.path.normpath(_project_root) + _os.sep
                    )
                    or _os.path.normpath(loc).startswith(
                        _os.path.normpath(_project_root) + _os.sep + 'DataSets'
                    )
                    or _os.path.normpath(loc).startswith(
                        _os.path.normpath(_project_root) + _os.sep + 'geodesic_propagation'
                    )
                    for loc in _all_locs
                    if loc
                ):
                    del _sys.modules[_k]
        try:
            from GenerateData.utils.data_generation_utils import (
                ring1_neighbors_gaussians,
                adaptive_ring1_neighbors,
                get_neighborhood_by_ring,
            )
        finally:
            for _p in _paths_removed:
                if _p not in _sys.path:
                    _sys.path.append(_p)

        if use_mahalanobis is None:
            use_mahalanobis = self.dataset.config.get('use_mahalanobis', False)

        # --- Ring-1 (adaptive or fixed) ---
        if adaptive_target_ring is not None and adaptive_target_neighbors is not None:
            ring1_nbrs, mean_nn_dist, per_point_nn_dist = adaptive_ring1_neighbors(
                vertices=self.positions,
                target_ring=adaptive_target_ring,
                target_ring_neighbors=adaptive_target_neighbors,
                n_neighbors_base=n_neighbors,
                k_boost=adaptive_k_boost,
                adaptive_max_mean_cut=adaptive_max_mean_cut,
                adaptive_max_steps=adaptive_max_steps,
                use_mahalanobis=use_mahalanobis,
                gaussian_scales=self.scales,
                gaussian_rotations=self.rotations,
            )
        else:
            ring1_nbrs, mean_nn_dist, per_point_nn_dist = ring1_neighbors_gaussians(
                vertices=self.positions,
                n_neighbors=n_neighbors,
                use_mahalanobis=use_mahalanobis,
                gaussian_scales=self.scales if use_mahalanobis else None,
                gaussian_rotations=self.rotations if use_mahalanobis else None,
            )

        self.ring1_neighbors = ring1_nbrs
        self.normalization_factor = float(mean_nn_dist)
        self.per_point_nn_distances = np.asarray(per_point_nn_dist, dtype=np.float32)

        # --- Ring-k ---
        ring_nbrs: Dict[int, np.ndarray] = {}
        for i in range(self.num_points):
            ring_nbrs[i] = get_neighborhood_by_ring(i, self.ring, ring1_nbrs)
        self.ring_neighbors = ring_nbrs

    def _get_current_normalization(self, neighbor_indices: np.ndarray) -> float:
        """Return the per-patch normalization factor matching training.

        When ``per_point_nn_distances`` was provided, returns
        ``per_point_nn_distances[neighbor_indices].mean()``, matching the
        ``normalize_per_patch=True`` path in training data generation.
        Falls back to ``self.normalization_factor`` otherwise.
        """
        if (
            self.per_point_nn_distances is not None
            and len(neighbor_indices) > 0
        ):
            return float(self.per_point_nn_distances[neighbor_indices].mean())
        return self.normalization_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_input(
        self,
        point_idx: int,
        all_neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        ring1_neighbor_indices: Optional[np.ndarray] = None,
        visited_mask: Optional[np.ndarray] = None,
        visited_set: Optional[Set[int]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]]:
        """
        Build model input for a single Fast Marching step.

        Steps:

        1. Resolve which of ``all_neighbor_indices`` are already visited
           (via ``visited_mask`` or ``visited_set``).
        2. Construct a scratch ``geodesic_distances`` array of size N where
           visited neighbours carry their real distances and all others receive
           a large sentinel (``1e12``).  ``point_idx`` itself is assigned
           ``max(visited_dists) + 1.0`` so that ``create_train_example``
           classifies visited neighbours as "valid" and unvisited ones as
           "beyond".
        3. Call :func:`DataSets.utils.training_patches_helpers.create_train_example`
           to produce a raw flat example array with stage-1 normalization
           already applied.
        4. Call :meth:`DataSets.gaussian_dataset.GaussianPatchDataset.get_item_from_raw`
           for stage-2 normalization (attribute-specific + xyz/geodesic
           max-distance scaling).
        5. Return the normalized tensors plus a ``build_info`` dict containing
           the values needed for :meth:`denormalize_result`.

        Args:
            point_idx: Index of the point whose geodesic distance is to be
                predicted.
            all_neighbor_indices: Full ring-k stencil (ALL neighbours, both
                visited and unvisited).
            neighbor_distances: Geodesic distances for entries in
                ``all_neighbor_indices``.  Entries for unvisited neighbours
                are ignored; supply zeros or any placeholder.
            ring1_neighbor_indices: Ring-1 neighbour indices used for the
                r1_min_val dropout augmentation.  Falls back to
                ``all_neighbor_indices`` when ``None``.
            visited_mask: Boolean array of shape ``(N,)`` — ``True`` for
                finalized points.
            visited_set: Set of visited point indices (alternative to
                ``visited_mask``; slower for large point clouds).

        Returns:
            ``(neighborhood, point_features, valid_mask, build_info)`` or
            ``None`` if ``create_train_example`` rejects the patch (too many
            valid neighbours for this ring).

            ``build_info`` is a :class:`dict` with keys:

            * ``'min_input'`` – minimum visited geodesic before normalization
            * ``'current_normalization'`` – per-patch normalization factor
            * ``'max_dist'`` – ``max_dist`` from ``_normalize_patches``
            * ``'neighborhood'`` – same tensor as the first return value
            * ``'point_features'`` – same tensor as the second return value
        """
        from DataSets.utils.training_patches_helpers import create_train_example

        UNVISITED_SENTINEL = 1e12

        # ------ Resolve visited neighbours --------------------------------
        if visited_mask is not None:
            vis_local = visited_mask[all_neighbor_indices]
            visited_nbr_indices = all_neighbor_indices[vis_local]
            visited_dists = neighbor_distances[visited_nbr_indices]
        elif visited_set is not None:
            vis_local = np.array(
                [idx in visited_set for idx in all_neighbor_indices], dtype=bool
            )
            visited_nbr_indices = all_neighbor_indices[vis_local]
            visited_dists = neighbor_distances[visited_nbr_indices]
        else:
            # All neighbours treated as visited
            visited_nbr_indices = all_neighbor_indices
            visited_dists = neighbor_distances[all_neighbor_indices]

        # ------ Compute denorm parameters BEFORE any normalization ----------
        if len(visited_dists) > 0:
            min_input = float(np.min(visited_dists))
            p_u_threshold = float(np.max(visited_dists)) + 1.0
        else:
            min_input = 0.0
            p_u_threshold = 1.0

        current_normalization = self._get_current_normalization(all_neighbor_indices)

        # ------ Build scratch geodesic array --------------------------------
        geo_scratch = np.full(self.num_points, UNVISITED_SENTINEL, dtype=np.float64)
        if len(visited_nbr_indices) > 0:
            geo_scratch[visited_nbr_indices] = visited_dists.astype(np.float64)
        geo_scratch[point_idx] = p_u_threshold

        # ------ Neighbour dicts for create_train_example --------------------
        ring_nbrs_dict = {point_idx: all_neighbor_indices}
        r1_nbrs = (
            ring1_neighbor_indices
            if ring1_neighbor_indices is not None
            else all_neighbor_indices
        )
        ring1_nbrs_dict = {point_idx: r1_nbrs}

        # ------ Read config options from dataset ----------------------------
        # (now stored in self._patch_config during __init__)

        # ------ Stage 1: create raw example with offline normalization ------
        raw_example = create_train_example(
            point_idx=point_idx,
            geodesic_distances=geo_scratch,
            ring_nbrs=ring_nbrs_dict,
            ring1_nbrs=ring1_nbrs_dict,
            gaussian_data=self._gaussian_data,
            patch_config=self._patch_config,
            skip_fold_check=True,
        )

        if raw_example is None:
            return None

        # ------ Stage 2: dataset normalisation pipeline ---------------------
        neighborhood, point_features, _target, valid_mask, norm_params = \
            self.dataset.get_item_from_raw(raw_example, return_norm_params=True)

        # Move to the configured device
        neighborhood = neighborhood.to(self.device)
        point_features = point_features.to(self.device)
        valid_mask = valid_mask.to(self.device)

        # NOTE: inference transforms are applied inside
        # dataset.get_item_from_raw() — no need to apply them again here.

        build_info = {
            'min_input': min_input,
            'current_normalization': current_normalization,
            'max_dist': norm_params['max_dist'],
            'neighborhood': neighborhood,
            'point_features': point_features,
        }

        return neighborhood, point_features, valid_mask, build_info

    def denormalize_result(
        self,
        raw_pred: float,
        build_info: Dict,
    ) -> float:
        """
        Convert a raw model prediction back to an absolute geodesic distance.

        Applies the inverse of the full two-stage normalization pipeline in
        reverse order:

        1. **Undo stage 2** (:meth:`GaussianPatchDataset._denormalize_results`):
           multiply target and geodesic distances by ``build_info['max_dist']``.

        2. **Undo stage 1** (:func:`denormalize_neighborhood`):
           multiply by ``current_normalization / nn_mean`` then add
           ``min_input``, giving the absolute geodesic distance.

        Args:
            raw_pred: Scalar model output.
            build_info: Dict returned as the fourth element of
                :meth:`build_input`.

        Returns:
            Predicted absolute geodesic distance.  Clamped so it is never
            smaller than ``build_info['min_input']``.
        """
        from DataSets.utils.training_patches_helpers import denormalize_neighborhood

        neighborhood = build_info['neighborhood']
        point_features = build_info['point_features']
        min_input = build_info['min_input']
        current_normalization = build_info['current_normalization']
        max_dist = build_info['max_dist']

        target_tensor = torch.tensor(float(raw_pred))
        nbhd_cpu = neighborhood.cpu()
        pf_cpu = point_features.cpu()

        # Step 1: undo _normalize_patches (undo max_dist scaling)
        nbhd_dn, pf_dn, target_dn = self.dataset._denormalize_results(
            nbhd_cpu, pf_cpu, target_tensor, {'max_dist': max_dist}
        )

        # Step 2: undo normalize_neighborhood (undo shift + per-patch scale)
        _, _, result, _ = denormalize_neighborhood(
            neighborhood=nbhd_dn.numpy(),
            point_features=pf_dn.numpy(),
            target=float(target_dn),
            r1_min_val=None,
            min_input=min_input,
            current_normalization=current_normalization,
            nn_mean=self.nn_mean,
            attributes=self.attributes,
            mask_constant=self.mask_constant,
            sh_features=self.sh_features,
        )

        return max(float(result), min_input)

    def get_neighbor_feature_dim(self) -> int:
        """Return the size of a single neighbour feature vector (entry_size)."""
        return self.dataset.features_entry_size

    def get_point_feature_dim(self) -> int:
        """Return the length of the point feature vector."""
        return len(self.dataset.point_indices)
