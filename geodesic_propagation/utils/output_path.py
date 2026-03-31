"""
Build structured output directories from model and gaussian paths.

When multiple shapes or models are evaluated, results must be stored in
separate directories to avoid overwriting.  This module derives a compact
directory hierarchy from ``model_path`` and ``gaussian_dir``.

Example output structure::

    geodesic_propagation/eval_output/
        combined_tosca_ring3/
            blue_texture/cat2/high_res/
                model_vs_gt.csv
                model_vs_gt_plots.png
            blue_texture/gorilla8/high_res/
                ...
        one_source/combined_polynomial_ring3/
            blue_texture/Saddle/level_04/light_4/
                ...
"""

from pathlib import Path, PurePosixPath
from typing import Optional


def _extract_model_name(model_path: Optional[str]) -> str:
    """Derive a compact model identifier from the checkpoint path.

    Typical patterns:
        checkpoints/combined_tosca_ring3/best_model.pth  → combined_tosca_ring3
        checkpoints/one_source/combined_polynomial_ring3/best_model.pth
            → one_source/combined_polynomial_ring3
    """
    if not model_path:
        return "unknown_model"
    p = PurePosixPath(str(model_path))
    # Walk up from the .pth file to find the checkpoints/ prefix
    parts = p.parts
    # Find the 'checkpoints' component
    try:
        ckpt_idx = parts.index("checkpoints")
        # Everything between 'checkpoints' and the .pth filename
        subparts = parts[ckpt_idx + 1 : -1]  # exclude final file
        if subparts:
            return str(PurePosixPath(*subparts))
    except ValueError:
        pass
    # Fallback: parent directory name
    return p.parent.name or "unknown_model"


def _extract_gaussian_subpath(gaussian_dir: Optional[str]) -> str:
    """Derive a compact identifier from the gaussian output directory.

    Strips known prefixes and suffixes so only the distinguishing portion
    remains.

    Typical patterns:
        TrainData/TOSCA/SyntheticColmapData/blue_texture/cat2/high_res/
            decoupled_appearance/output
            → blue_texture/cat2/high_res
        TrainData/Polynomial/SyntheticColmapData/blue_texture/Saddle/
            level_04/light_4/output
            → blue_texture/Saddle/level_04/light_4
    """
    if not gaussian_dir:
        return "unknown_gaussian"
    p = PurePosixPath(str(gaussian_dir))
    parts = list(p.parts)

    # Strip trailing 'output' or 'decoupled_appearance/output'
    while parts and parts[-1] in ("output", "decoupled_appearance"):
        parts.pop()

    # Strip known prefix components
    trim_prefixes = {"TrainData", "TOSCA", "Polynomial", "SyntheticColmapData"}
    while parts and parts[0] in trim_prefixes:
        parts.pop(0)
    # Also strip leading '/' if present
    if parts and parts[0] == "/":
        parts.pop(0)

    if parts:
        return str(PurePosixPath(*parts))
    return "unknown_gaussian"


def build_eval_output_dir(
    base_dir: str,
    model_path: Optional[str],
    gaussian_dir: Optional[str],
) -> Path:
    """Construct a structured output directory.

    Parameters
    ----------
    base_dir : str
        Base output directory (e.g. ``geodesic_propagation/eval_output``).
    model_path : str or None
        Path to the model checkpoint.
    gaussian_dir : str or None
        Path to the Gaussian splat output directory.

    Returns
    -------
    Path
        ``base_dir / <model_name> / <gaussian_subpath>``
    """
    model_name = _extract_model_name(model_path)
    gaussian_sub = _extract_gaussian_subpath(gaussian_dir)
    return Path(base_dir) / model_name / gaussian_sub
