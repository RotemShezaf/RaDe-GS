"""
Constants and registries for GaussianPatchTransformer training.

Provides:
  TRANSFORM_REGISTRY  – maps transform name  -> class
  Compose             – chains multiple transforms with the dataset's calling convention
  build_transforms    – builds a transform (or Compose) from a list of config dicts/strings
"""

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_root = Path(__file__).parent.parent
_root_str = str(_root)
_datasets_dir = str(_root / "DataSets")

# Ensure project root is importable
if _root_str not in sys.path:
    sys.path.append(_root_str)

# data_transformation.py uses bare `from utils.xxx import ...` which requires
# DataSets/ on the path so `utils` resolves to DataSets/utils.
# We prepend it temporarily so the import succeeds, then move it to the back
# so it doesn't shadow models/utils.py for later imports.
_already_present = _datasets_dir in sys.path
if not _already_present:
    sys.path.insert(0, _datasets_dir)

from DataSets.data_transformation import (
    PointcloudRandomInputDropout,
    GaussianPatchDropout,
    GaussianPatchRotate,
    GaussianPatchRandomFlip,
    GaussianPatchCanonicalRotate,
    SparseContextDropout,
    GaussianPatchSurfacePerturb,
    GeodesicNoiseAugmentation,
)

if not _already_present:
    sys.path.remove(_datasets_dir)
    sys.path.append(_datasets_dir)

# DataSets/utils got cached as the 'utils' package in sys.modules during the
# import above.  Remove it so that subsequent imports of `utils` (e.g. in
# models/transformer.py) find models/utils.py instead.
for _key in list(sys.modules.keys()):
    if _key == "utils" or _key.startswith("utils."):
        del sys.modules[_key]


# ── Registry ──────────────────────────────────────────────────────────────────

TRANSFORM_REGISTRY: Dict[str, type] = {
    "PointcloudRandomInputDropout": PointcloudRandomInputDropout,
    "GaussianPatchDropout":         GaussianPatchDropout,
    "GaussianPatchRotate":          GaussianPatchRotate,
    "GaussianPatchRandomFlip":      GaussianPatchRandomFlip,
    "GaussianPatchCanonicalRotate": GaussianPatchCanonicalRotate,
    "SparseContextDropout":         SparseContextDropout,
    "GaussianPatchSurfacePerturb":  GaussianPatchSurfacePerturb,
    "GeodesicNoiseAugmentation":    GeodesicNoiseAugmentation,
}

# Transforms that are deterministic and must be applied at inference time.
# Stochastic augmentations (dropout, random rotation, random flip) are
# training-only and must be excluded.
INFERENCE_SAFE_TRANSFORMS = {
    "GaussianPatchCanonicalRotate",
}


# ── Compose ───────────────────────────────────────────────────────────────────

class Compose:
    """
    Chain multiple transforms.

    Each transform is called with the dataset's convention::

        neighborhood, point_features = t(neighborhood, point_features, r1_min_val)
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        for t in self.transforms:
            neighborhood, point_features = t(neighborhood, point_features, r1_min_val, **kwargs)
        return neighborhood, point_features

    def __repr__(self) -> str:
        lines = [f"Compose(["]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__},")
        lines.append("])")
        return "\n".join(lines)


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_single_transform(
    cfg: Union[str, Dict[str, Any]],
    attributes: List[str],
    mask_constant: float,
) -> Any:
    """
    Instantiate one transform from a name string or a config dict.

    ``attributes`` and ``mask_constant`` are injected automatically for
    constructors that accept them; any extra keys in *cfg* are forwarded
    verbatim as keyword arguments.
    """
    if isinstance(cfg, str):
        name = cfg
        extra_kwargs: Dict[str, Any] = {}
    else:
        cfg = dict(cfg)          # shallow copy so we can pop safely
        name = cfg.pop("name")
        extra_kwargs = cfg

    if name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available: {sorted(TRANSFORM_REGISTRY)}"
        )

    cls = TRANSFORM_REGISTRY[name]
    init_params = inspect.signature(cls.__init__).parameters

    kwargs: Dict[str, Any] = {}
    if "attributes" in init_params:
        kwargs["attributes"] = attributes
    if "mask_constant" in init_params:
        kwargs["mask_constant"] = mask_constant

    # User-supplied values override injected ones
    kwargs.update(extra_kwargs)

    return cls(**kwargs)


def build_transforms(
    transforms_cfg: Optional[List[Union[str, Dict[str, Any]]]],
    attributes: List[str],
    mask_constant: float = -10.0,
) -> Optional[Any]:
    """
    Build a transform (or ``Compose``) from a list of transform config entries.

    Each entry is either:
    * a string  – transform name with default parameters
    * a dict    – ``{"name": "...", **kwargs}``

    Args:
        transforms_cfg: List of transform configs.  ``None`` or empty list
                        returns ``None`` (no augmentation).
        attributes:     Attribute list forwarded to transforms that need it.
        mask_constant:  Mask value forwarded to transforms that need it.

    Returns:
        A single transform, a ``Compose``, or ``None``.

    Example YAML::

        transforms:
          - name: GaussianPatchRotate
          - name: GaussianPatchDropout
            max_dropout_ratio: 0.3
          - name: GaussianPatchRandomFlip
            flip_prob: 0.5
    """
    if not transforms_cfg:
        return None

    built = [_build_single_transform(c, attributes, mask_constant) for c in transforms_cfg]
    return built[0] if len(built) == 1 else Compose(built)


def build_inference_transforms(
    transforms_cfg: Optional[List[Union[str, Dict[str, Any]]]],
    attributes: List[str],
    mask_constant: float = -10.0,
) -> Optional[Any]:
    """
    Build transforms for inference, keeping only deterministic (inference-safe)
    transforms from the training config.

    Stochastic augmentations (dropout, random rotation, random flip) are
    filtered out.  Only transforms listed in ``INFERENCE_SAFE_TRANSFORMS``
    are retained.

    Args:
        transforms_cfg: Training transform config list (from YAML).
        attributes:     Attribute list forwarded to transforms that need it.
        mask_constant:  Mask value forwarded to transforms that need it.

    Returns:
        A single transform, a ``Compose``, or ``None``.
    """
    if not transforms_cfg:
        return None

    filtered = []
    for cfg in transforms_cfg:
        name = cfg if isinstance(cfg, str) else cfg.get("name", cfg)
        if name in INFERENCE_SAFE_TRANSFORMS:
            filtered.append(cfg)

    if not filtered:
        return None

    return build_transforms(filtered, attributes, mask_constant)
