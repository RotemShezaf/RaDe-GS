"""
FPS (Furthest Point Sampling) utilities for Gaussian splatting data.

Provides ``fps_gs`` which selects a subset of Gaussians via FPS,
using any combination of attributes (xyz, scale, rotation, opacity, sh,
normals, …) to compute inter-point distances — matching the attribute
system in ``DataSets/gaussian_dataset.py``.

Typical usage
-------------
>>> from utils.misc import fps_gs
>>> idx = fps_gs(
...     positions, n=5000, attributes=["xyz"],
...     scales=scales, rotations=rotations,
...     opacities=opacities, device="cuda",
... )
>>> positions_ds = positions[idx]

Or to downsample *all* arrays at once:

>>> from utils.misc import fps_downsample_gaussians
>>> ds = fps_downsample_gaussians(
...     positions, n=5000,
...     scales=scales, rotations=rotations, opacities=opacities,
... )
>>> ds["positions"]   # (5000, 3)
>>> ds["indices"]     # (5000,)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

try:
    from pointnet2_ops import pointnet2_utils
    _HAS_POINTNET2_CUDA = torch.cuda.is_available()
except ImportError:
    pointnet2_utils = None  # type: ignore[assignment]
    _HAS_POINTNET2_CUDA = False

try:
    import fpsample as _fpsample
except ImportError:
    _fpsample = None  # type: ignore[assignment]


# ── FPS dispatch ─────────────────────────────────────────────────────────────

def _fps_dispatch(
    packed: np.ndarray,
    n: int,
    device: Union[str, torch.device] = "cuda",
    start_idx: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """Run FPS, choosing the best available backend.

    Priority:
      1. **pointnet2_utils** on CUDA (fastest, but no ``start_idx`` support —
         protected-indices logic is handled in the caller).
      2. **fpsample** on CPU (Rust-backed, fast, supports ``start_idx`` for
         protected / seed points natively).
      3. Raises ``RuntimeError`` if neither backend is available.

    Parameters
    ----------
    packed : ndarray, shape ``(N, K)``
        Packed attribute array (float32).
    n : int
        Number of points to select.
    device : str or torch.device
        CUDA device (only used when pointnet2 is available).
    start_idx : int or list[int], optional
        Starting / protected indices.  Supported natively by *fpsample*;
        ignored when the CUDA backend is used (caller handles protection).

    Returns
    -------
    indices : ndarray of int64, shape ``(n,)``
    """
    # ── CUDA path ─────────────────────────────────────────────────────
    if _HAS_POINTNET2_CUDA and start_idx is None:
        data_t = torch.from_numpy(packed).float().unsqueeze(0).to(device)
        fps_idx = pointnet2_utils.furthest_point_sample(data_t, n)  # (1, n)
        return fps_idx[0].cpu().numpy().astype(np.int64)

    # ── CPU path via fpsample (Rust-backed) ───────────────────────────
    if _fpsample is not None:
        idx = _fpsample.fps_sampling(
            packed.astype(np.float32), n, start_idx=start_idx,
        )
        return np.asarray(idx, dtype=np.int64)

    raise RuntimeError(
        "No FPS backend available. Install either:\n"
        "  • pointnet2_ops  (pip install pointnet2_ops, requires CUDA)\n"
        "  • fpsample       (pip install fpsample, CPU-only, Rust-backed)"
    )


# ── Attribute layout (mirrors DataSets/utils/data_transformation_utils.py) ──

_ATTR_SIZE = {
    "xyz": 3,
    "normals": 3,
    "scale": 3,
    "sh": 3,
    "rotation": 4,
    "opacity": 1,
    "euclidean_distances": 1,
    "geodesic_distance": 1,
}


def get_attribute_size(attr: str) -> int:
    """Return the number of channels for *attr*."""
    return _ATTR_SIZE.get(attr, 1)


# ── Core helpers ─────────────────────────────────────────────────────────────

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure *arr* is ``(N, K)`` — squeeze trailing length-1 dims and
    expand 1-D arrays to ``(N, 1)``."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _pack_attributes(
    attributes: Sequence[str],
    positions: np.ndarray,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    sh_features: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack selected Gaussian attributes into a single ``(N, K)`` array.

    Only attributes listed in *attributes* are included, in order.
    """
    parts: list[np.ndarray] = []
    for attr in attributes:
        if attr == "xyz":
            parts.append(_to_2d(positions))
        elif attr == "scale":
            if scales is None:
                raise ValueError("scales required when 'scale' in attributes")
            parts.append(_to_2d(scales))
        elif attr == "rotation":
            if rotations is None:
                raise ValueError("rotations required when 'rotation' in attributes")
            parts.append(_to_2d(rotations))
        elif attr == "opacity":
            if opacities is None:
                raise ValueError("opacities required when 'opacity' in attributes")
            parts.append(_to_2d(opacities))
        elif attr == "sh":
            if sh_features is None:
                raise ValueError("sh_features required when 'sh' in attributes")
            parts.append(_to_2d(sh_features))
        elif attr == "normals":
            if normals is None:
                raise ValueError("normals required when 'normals' in attributes")
            parts.append(_to_2d(normals))
        else:
            raise ValueError(f"Unknown attribute '{attr}' for FPS packing")
    return np.concatenate(parts, axis=1)


# ── Public API ───────────────────────────────────────────────────────────────

def fps_gs(
    positions: np.ndarray,
    n: int,
    attributes: List[str] = ("xyz",),
    *,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    sh_features: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    protected_indices: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
) -> np.ndarray:
    """Furthest-point-sample *n* Gaussians using the given *attributes*.

    Unlike the ``Gaussian_MAE`` version which takes a pre-packed
    ``(B, N, K)`` tensor with a fixed column layout, this function takes
    separate per-attribute arrays as stored by GaussianDataCPU /
    ``load_gaussian_data_cpu``, and packs only the requested attributes
    for the FPS distance computation.

    Parameters
    ----------
    positions : ndarray, shape ``(N, 3)``
        Gaussian centre positions (always required, even if 'xyz' is not
        in *attributes*, for the return mapping).
    n : int
        Number of points to sample.
    attributes : list of str
        Which attributes to use for the FPS distance metric.
        Valid names match the dataset config: ``'xyz'``, ``'scale'``,
        ``'rotation'``, ``'opacity'``, ``'sh'``, ``'normals'``.
        Default: ``['xyz']`` (position-only FPS).
    scales, rotations, opacities, sh_features, normals : ndarray, optional
        Per-Gaussian attribute arrays.  Only those listed in *attributes*
        need be supplied.
    protected_indices : ndarray of int, optional
        Indices that must appear in the output regardless of FPS selection.
        FPS is run to fill the remaining ``n - len(protected_indices)``
        slots from non-protected points, and the two sets are merged.
    device : str or torch.device
        CUDA device for ``pointnet2_utils``.

    Returns
    -------
    indices : ndarray of int64, shape ``(n,)``
        Indices into the original ``(N, …)`` arrays.
    """
    N = positions.shape[0]
    if N <= n:
        return np.arange(N, dtype=np.int64)

    # Normalise protected_indices
    if protected_indices is not None and len(protected_indices) > 0:
        protected: Optional[np.ndarray] = np.unique(
            np.asarray(protected_indices, dtype=np.int64)
        )
        n_protected = len(protected)
        if n_protected >= n:
            return protected[:n]
    else:
        protected = None
        n_protected = 0

    packed = _pack_attributes(
        attributes, positions,
        scales=scales, rotations=rotations,
        opacities=opacities, sh_features=sh_features,
        normals=normals,
    )  # (N, K)

    # ── fpsample path (supports start_idx natively) ───────────────────
    if _fpsample is not None and not _HAS_POINTNET2_CUDA:
        start = protected.tolist() if protected is not None else None
        return np.sort(_fps_dispatch(packed, n, device, start_idx=start))

    # ── CUDA path (no native start_idx → manual protect-then-FPS) ─────
    if protected is not None:
        all_idx = np.arange(N, dtype=np.int64)
        mask = np.ones(N, dtype=bool)
        mask[protected] = False
        remaining = all_idx[mask]

        n_fps = n - n_protected
        if len(remaining) <= n_fps:
            return np.sort(np.concatenate([protected, remaining]))

        packed_rem = packed[remaining]
        fps_local = _fps_dispatch(packed_rem, n_fps, device)
        fps_original = remaining[fps_local]
        return np.sort(np.concatenate([protected, fps_original]))

    return _fps_dispatch(packed, n, device)


def fps_downsample_gaussians(
    positions: np.ndarray,
    n: int,
    attributes: List[str] = ("xyz",),
    *,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    sh_features: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    protected_indices: Optional[np.ndarray] = None,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, np.ndarray]:
    """FPS-downsample a full set of Gaussian arrays and return a dict.

    Convenience wrapper around :func:`fps_gs` that also slices every
    supplied array to the selected subset.

    Returns
    -------
    dict with keys:
        ``'indices'``, ``'positions'``, and every non-None optional array
        (``'scales'``, ``'rotations'``, ``'opacities'``,
        ``'sh_features'``, ``'normals'``), each sliced to ``(n, …)``.
    """
    idx = fps_gs(
        positions, n, attributes,
        scales=scales, rotations=rotations,
        opacities=opacities, sh_features=sh_features,
        normals=normals, protected_indices=protected_indices,
        device=device,
    )

    result: Dict[str, np.ndarray] = {
        "indices": idx,
        "positions": positions[idx],
    }
    if scales is not None:
        result["scales"] = scales[idx]
    if rotations is not None:
        result["rotations"] = rotations[idx]
    if opacities is not None:
        result["opacities"] = opacities[idx]
    if sh_features is not None:
        result["sh_features"] = sh_features[idx]
    if normals is not None:
        result["normals"] = normals[idx]
    return result
