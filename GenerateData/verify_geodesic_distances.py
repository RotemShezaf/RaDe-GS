#!/usr/bin/env python3
"""
Verify geodesic distances computed on a Gaussian mesh.

This script performs sanity checks on geodesic distance data:

1. Non-negativity: all distances ≥ 0
2. Self-distance: d(source_i, nearest_gaussian_i) bounded by source-
   to-Gaussian on-surface Euclidean distance (sources are parametric
   grid points, NOT Gaussians, so self-distance > 0 is expected)
3. Triangle inequality: d(A,C) ≤ d(A,B_gauss) + d(B,C) + slack
   (slack accounts for source ≠ Gaussian offset)
4. Euclidean lower-bound: geodesic ≥ on-surface Euclidean distance
   (requires mesh for on-surface positions)
5. Symmetry: d(source_i, gauss_j) ≈ d(source_j, gauss_i) + offsets
6. Distance statistics & NaN/Inf check
7. Closest-mesh-mapping check
8. Optional comparison with reference (GT-mesh) geodesics

Outputs a JSON report alongside the geodesic data.

Usage::

    python verify_geodesic_distances.py \\
        --gaussian_output <path_to_output> \\
        [--reference <path_to_reference_gt_geodesic.npz>] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

INF_THRESHOLD = 1e6  # distances above this are treated as effectively infinite


# ── helpers ──────────────────────────────────────────────────────────────────

def load_geodesic_data(npz_path: Path) -> dict:
    """Load geodesic data from an npz file."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def _try_load_mesh(output_dir: Path):
    """Try to load the Gaussian mesh; returns (vertices, faces, gauss_vi) or Nones."""
    mesh_dir = output_dir / "geodesic_mesh"
    npz = mesh_dir / "geodesic_mesh_data.npz"
    if not npz.exists():
        return None, None, None
    d = np.load(npz, allow_pickle=True)
    return d["vertices"], d["faces"], d["gaussian_vertex_indices"]


# ── individual checks ───────────────────────────────────────────────────────

def check_non_negativity(distances: np.ndarray) -> dict:
    n_negative = int(np.sum(distances < 0))
    return {
        "passed": n_negative == 0,
        "n_negative": n_negative,
        "min_value": float(np.min(distances)),
        "pct_negative": float(100 * n_negative / max(distances.size, 1)),
    }


def check_self_distance(
    distances: np.ndarray,
    source_gaussian_indices: np.ndarray,
    source_indices: np.ndarray | None,
    mesh_vertices: np.ndarray | None,
    gaussian_vertex_indices: np.ndarray | None,
) -> dict:
    """Check source → nearest-Gaussian geodesic distance.

    Sources are parametric-grid points snapped to the nearest mesh vertex,
    so self-distance = geodesic(source_mesh_vert, nearest_gaussian_vert).
    This is expected to be > 0.  We verify:

    * self-geodesic ≥ 0
    * self-geodesic ≥ on-surface Euclidean to nearest Gaussian (if mesh available)
    * distribution looks reasonable
    """
    n_sources = len(source_gaussian_indices)
    self_dists = np.array([
        distances[i, source_gaussian_indices[i]]
        for i in range(n_sources)
    ])

    result: dict = {
        "n_sources": n_sources,
        "min": float(self_dists.min()),
        "max": float(self_dists.max()),
        "mean": float(self_dists.mean()),
        "median": float(np.median(self_dists)),
        "n_negative": int((self_dists < 0).sum()),
    }

    # If mesh is available, compare with Euclidean source→Gaussian on-surface
    if mesh_vertices is not None and source_indices is not None and gaussian_vertex_indices is not None:
        src_pos = mesh_vertices[source_indices]
        gauss_pos_on_surf = mesh_vertices[gaussian_vertex_indices]
        gauss_tree = KDTree(gauss_pos_on_surf)
        eucl_dists, _ = gauss_tree.query(src_pos)
        ratio = self_dists / np.maximum(eucl_dists, 1e-15)
        # Geodesic ≥ Euclidean always, so ratio should be ≥ 1
        # Skip sources where both are ~0 (source IS its nearest Gaussian)
        nontrivial = eucl_dists > 1e-10
        if nontrivial.any():
            n_bad = int((ratio[nontrivial] < 1 - 1e-4).sum())
        else:
            n_bad = 0
        result["euclidean_comparison"] = {
            "max_eucl_to_nearest_gauss": float(eucl_dists.max()),
            "mean_eucl_to_nearest_gauss": float(eucl_dists.mean()),
            "ratio_geodesic_over_euclidean_min": float(ratio[nontrivial].min()) if nontrivial.any() else float("nan"),
            "ratio_geodesic_over_euclidean_median": float(np.median(ratio[nontrivial])) if nontrivial.any() else float("nan"),
            "n_below_euclidean": n_bad,
            "n_sources_at_gaussian": int((~nontrivial).sum()),
        }
        result["passed"] = bool((self_dists >= 0).all() and n_bad == 0)
    else:
        result["passed"] = bool((self_dists >= 0).all())
        result["note"] = "No mesh available; only checked non-negativity"

    return result


def check_triangle_inequality(
    distances: np.ndarray,
    source_gaussian_indices: np.ndarray,
    n_samples: int = 50000,
    seed: int = 42,
) -> dict:
    """Sample random triples and check triangle inequality.

    d(A, C) ≤ d(A, B_gauss) + d(B, C) + self_dist_B

    where self_dist_B = d(B, B_gauss) accounts for source B not being
    exactly at Gaussian B.
    """
    rng = np.random.RandomState(seed)
    n_sources, n_gauss = distances.shape

    if n_sources < 2:
        return {"passed": True, "n_tested": 0, "note": "fewer than 2 sources"}

    # Pre-compute self-distances (source → its nearest Gaussian)
    self_dists = np.array([
        distances[i, source_gaussian_indices[i]] for i in range(n_sources)
    ])

    src_a = rng.randint(0, n_sources, size=n_samples)
    src_b = rng.randint(0, n_sources, size=n_samples)
    gauss_c = rng.randint(0, n_gauss, size=n_samples)

    d_ac = distances[src_a, gauss_c]
    b_gauss = source_gaussian_indices[src_b]
    d_ab = distances[src_a, b_gauss]
    d_bc = distances[src_b, gauss_c]
    slack = self_dists[src_b]  # offset because source_B ≠ Gaussian_B

    violations = d_ac > d_ab + d_bc + slack + 1e-5
    n_violations = int(violations.sum())
    if n_violations > 0:
        mags = d_ac[violations] - d_ab[violations] - d_bc[violations] - slack[violations]
    else:
        mags = np.array([])

    return {
        "passed": n_violations == 0,
        "n_tested": n_samples,
        "n_violations": n_violations,
        "pct_violations": float(100 * n_violations / n_samples),
        "max_violation": float(mags.max()) if n_violations > 0 else 0.0,
        "mean_violation": float(mags.mean()) if n_violations > 0 else 0.0,
    }


def check_euclidean_lower_bound(
    distances: np.ndarray,
    source_indices: np.ndarray,
    mesh_vertices: np.ndarray | None,
    gaussian_vertex_indices: np.ndarray | None,
    tolerance: float = 1e-5,
) -> dict:
    """Check geodesic ≥ Euclidean using ON-SURFACE positions.

    Uses mesh vertex positions (not original off-surface Gaussian positions)
    so the comparison is geometrically valid.  If no mesh is available,
    skip with an informational note.
    """
    n_sources, n_gauss = distances.shape

    if mesh_vertices is None or gaussian_vertex_indices is None:
        return {"passed": None, "note": "No mesh available; skipped"}

    # On-surface positions
    src_pos = mesh_vertices[source_indices]        # (n_sources, 3)
    gauss_pos = mesh_vertices[gaussian_vertex_indices]  # (n_gauss, 3)

    # Sample pairs (full would be huge)
    rng = np.random.RandomState(42)
    n_test = min(500_000, n_sources * n_gauss)
    si = rng.randint(0, n_sources, size=n_test)
    gi = rng.randint(0, n_gauss, size=n_test)

    eucl = np.linalg.norm(src_pos[si] - gauss_pos[gi], axis=1)
    geod = distances[si, gi]

    violations = geod < eucl - tolerance
    n_violations = int(violations.sum())
    valid = ~violations
    ratios = geod[valid] / np.maximum(eucl[valid], 1e-15)

    return {
        "passed": n_violations == 0,
        "n_tested": n_test,
        "n_violations": n_violations,
        "pct_violations": float(100 * n_violations / n_test),
        "max_violation": float((eucl[violations] - geod[violations]).max()) if n_violations > 0 else 0.0,
        "geodesic_to_euclidean_ratio": {
            "min": float(ratios.min()) if len(ratios) > 0 else None,
            "median": float(np.median(ratios)) if len(ratios) > 0 else None,
            "mean": float(ratios.mean()) if len(ratios) > 0 else None,
            "max": float(ratios.max()) if len(ratios) > 0 else None,
        },
    }


def check_symmetry(
    distances: np.ndarray,
    source_gaussian_indices: np.ndarray,
    source_indices: np.ndarray | None,
    mesh_vertices: np.ndarray | None,
    gaussian_vertex_indices: np.ndarray | None,
) -> dict:
    """Check symmetry: d(src_i → gauss_j) ≈ d(src_j → gauss_i).

    Because source_i is NOT at gauss_i, perfect symmetry is not expected.
    The expected asymmetry is bounded by the source-to-nearest-Gaussian
    offsets:  |d(i→gj) - d(j→gi)| ≤ self_i + self_j
    """
    n_sources = len(source_gaussian_indices)
    if n_sources < 2:
        return {"passed": True, "n_tested": 0, "note": "fewer than 2 sources"}

    self_dists = np.array([
        distances[i, source_gaussian_indices[i]] for i in range(n_sources)
    ])

    diffs = []
    rel_diffs = []
    adjusted_violations = 0
    n_tested = 0

    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            gi = source_gaussian_indices[i]
            gj = source_gaussian_indices[j]
            d_ij = distances[i, gj]
            d_ji = distances[j, gi]
            diff = abs(d_ij - d_ji)
            diffs.append(diff)
            denom = max(d_ij, d_ji, 1e-15)
            rel_diffs.append(diff / denom)
            # Expected asymmetry bound
            bound = self_dists[i] + self_dists[j] + 1e-5
            if diff > bound:
                adjusted_violations += 1
            n_tested += 1

    diffs = np.array(diffs)
    rel_diffs = np.array(rel_diffs)

    return {
        "passed": adjusted_violations == 0,
        "n_tested": n_tested,
        "n_raw_asymmetric_gt5pct": int((rel_diffs > 0.05).sum()),
        "n_adjusted_violations": adjusted_violations,
        "absolute_diff": {
            "min": float(diffs.min()),
            "median": float(np.median(diffs)),
            "mean": float(diffs.mean()),
            "max": float(diffs.max()),
        },
        "relative_diff": {
            "min": float(rel_diffs.min()),
            "median": float(np.median(rel_diffs)),
            "mean": float(rel_diffs.mean()),
            "max": float(rel_diffs.max()),
        },
    }


def check_distance_statistics(distances: np.ndarray) -> dict:
    n_inf = int(np.sum(np.isinf(distances)))
    n_nan = int(np.sum(np.isnan(distances)))
    n_above_threshold = int(np.sum(distances >= INF_THRESHOLD))
    passed = (n_inf == 0) and (n_nan == 0) and (n_above_threshold == 0)
    return {
        "passed": passed,
        "shape": list(distances.shape),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "median": float(np.median(distances)),
        "p5": float(np.percentile(distances, 5)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p95": float(np.percentile(distances, 95)),
        "n_zeros": int(np.sum(distances == 0)),
        "n_inf": n_inf,
        "n_nan": n_nan,
        "n_above_threshold": n_above_threshold,
        "inf_threshold": INF_THRESHOLD,
    }


def check_closest_mesh_mapping(
    closest_mesh_indices: np.ndarray,
    closest_mesh_distances: np.ndarray,
) -> dict:
    return {
        "n_gaussians": int(len(closest_mesh_indices)),
        "unique_mesh_vertices_used": int(len(np.unique(closest_mesh_indices))),
        "closest_distance_max": float(np.max(closest_mesh_distances)),
        "closest_distance_mean": float(np.mean(closest_mesh_distances)),
        "closest_distance_median": float(np.median(closest_mesh_distances)),
        "n_zero_distance": int(np.sum(closest_mesh_distances == 0)),
        "note": (
            "For Gaussian-mesh mode, all distances should be 0 "
            "(Gaussians ARE mesh vertices)"
        ),
    }


def check_gaussian_consistency(
    output_dir: Path,
    gaussian_positions: np.ndarray,
    closest_mesh_indices: np.ndarray,
    closest_mesh_distances: np.ndarray,
    mesh_vertices: np.ndarray | None,
    gaussian_vertex_indices: np.ndarray | None,
    surface_type: str | None,
    tolerance: float = 1e-4,
) -> dict:
    """Cross-check PLY, mesh, and geodesic data for consistency.

    Checks:
    1. PLY positions == geodesic gaussian_positions (exact)
    2. Gaussian count consistent across PLY, mesh, and geodesic data
    3. closest_mesh_indices == gaussian_vertex_indices (Gaussian-mesh mode)
    4. closest_mesh_distances all == 0 (Gaussian-mesh mode)
    5. gaussian_vertex_indices valid (in range, unique)
    6. mesh_vertices[gi] == project(ply_positions, surface) (if surface provided)
    """
    from GenerateData.utils.geodesic_mesh_utils import project_gaussians_to_surface

    result: dict = {"sub_checks": {}}
    all_passed = True

    # ── 1. Load PLY and compare with geodesic gaussian_positions ──────
    try:
        from utils.load_utils import find_available_iterations
        from plyfile import PlyData

        point_cloud_dir = output_dir / "point_cloud"
        iterations = find_available_iterations(output_dir)
        if not iterations:
            result["sub_checks"]["ply_match"] = {
                "passed": None,
                "note": "No PLY iterations found",
            }
        else:
            highest_iter = max(iterations)
            ply_path = point_cloud_dir / f"iteration_{highest_iter}" / "point_cloud.ply"
            plydata = PlyData.read(str(ply_path))
            vertex = plydata.elements[0]
            ply_xyz = np.stack([
                np.asarray(vertex["x"]),
                np.asarray(vertex["y"]),
                np.asarray(vertex["z"]),
            ], axis=1).astype(np.float32)

            # Count match
            count_match = len(ply_xyz) == len(gaussian_positions)
            # Value match
            if count_match:
                exact = np.array_equal(ply_xyz, gaussian_positions)
                max_diff = float(np.max(np.abs(ply_xyz - gaussian_positions)))
            else:
                exact = False
                max_diff = float("inf")

            ply_passed = count_match and exact
            if not ply_passed:
                all_passed = False
            result["sub_checks"]["ply_match"] = {
                "passed": ply_passed,
                "ply_iteration": highest_iter,
                "ply_count": int(len(ply_xyz)),
                "geodesic_count": int(len(gaussian_positions)),
                "count_match": count_match,
                "exact_match": exact,
                "max_diff": max_diff,
            }
    except Exception as e:
        result["sub_checks"]["ply_match"] = {
            "passed": None,
            "note": f"Could not load PLY: {e}",
        }

    # ── 2. Gaussian count consistency across mesh and geodesic ────────
    n_geodesic = len(gaussian_positions)
    counts = {"geodesic_gaussian_positions": n_geodesic}
    if gaussian_vertex_indices is not None:
        counts["mesh_gaussian_vertex_indices"] = int(len(gaussian_vertex_indices))
    counts["closest_mesh_indices"] = int(len(closest_mesh_indices))
    all_same = len(set(counts.values())) == 1
    if not all_same:
        all_passed = False
    result["sub_checks"]["count_consistency"] = {
        "passed": all_same,
        "counts": counts,
    }

    # ── 3. closest_mesh_indices == gaussian_vertex_indices ────────────
    if gaussian_vertex_indices is not None:
        idx_match = np.array_equal(closest_mesh_indices, gaussian_vertex_indices)
        if not idx_match:
            all_passed = False
            n_differ = int(np.sum(closest_mesh_indices != gaussian_vertex_indices))
        else:
            n_differ = 0
        result["sub_checks"]["indices_match"] = {
            "passed": idx_match,
            "n_differ": n_differ,
        }
    else:
        result["sub_checks"]["indices_match"] = {
            "passed": None,
            "note": "No mesh gaussian_vertex_indices available",
        }

    # ── 4. closest_mesh_distances all zero (Gaussian-mesh mode) ──────
    all_zero = bool(np.all(closest_mesh_distances == 0))
    if not all_zero:
        all_passed = False
    result["sub_checks"]["distances_zero"] = {
        "passed": all_zero,
        "max": float(np.max(closest_mesh_distances)),
        "n_nonzero": int(np.sum(closest_mesh_distances != 0)),
    }

    # ── 5. gaussian_vertex_indices valid and unique ──────────────────
    if gaussian_vertex_indices is not None and mesh_vertices is not None:
        n_verts = len(mesh_vertices)
        in_range = bool((gaussian_vertex_indices >= 0).all()
                        and (gaussian_vertex_indices < n_verts).all())
        n_unique = int(len(np.unique(gaussian_vertex_indices)))
        is_unique = n_unique == len(gaussian_vertex_indices)
        gi_ok = in_range and is_unique
        if not gi_ok:
            all_passed = False
        result["sub_checks"]["vertex_indices_valid"] = {
            "passed": gi_ok,
            "in_range": in_range,
            "n_unique": n_unique,
            "n_total": int(len(gaussian_vertex_indices)),
            "is_unique": is_unique,
            "n_mesh_vertices": n_verts,
        }
    else:
        result["sub_checks"]["vertex_indices_valid"] = {
            "passed": None,
            "note": "No mesh data available",
        }

    # ── 6. mesh_vertices[gi] == project(positions, surface) ─────────
    if (surface_type is not None and gaussian_vertex_indices is not None
            and mesh_vertices is not None):
        projected = project_gaussians_to_surface(
            gaussian_positions, surface_type,
        )
        mesh_at_gi = mesh_vertices[gaussian_vertex_indices]
        diff = np.linalg.norm(mesh_at_gi - projected, axis=1)
        n_exceed = int(np.sum(diff > tolerance))
        proj_ok = n_exceed == 0
        if not proj_ok:
            all_passed = False
        result["sub_checks"]["projection_match"] = {
            "passed": proj_ok,
            "n_exceed_tolerance": n_exceed,
            "tolerance": tolerance,
            "max_diff": float(diff.max()),
            "mean_diff": float(diff.mean()),
        }
    else:
        result["sub_checks"]["projection_match"] = {
            "passed": None,
            "note": "Skipped (no surface type or mesh)",
        }

    result["passed"] = all_passed
    return result


def check_batch_cache(output_dir: Path) -> dict:
    """Scan mesh_batch_cache for inf/NaN values and report per-file details."""
    cache_dir = output_dir / "geodesic_distance" / "mesh_batch_cache"
    if not cache_dir.is_dir():
        return {"passed": None, "note": "No mesh_batch_cache directory found"}

    batch_files = sorted(cache_dir.glob("mesh_batch_*.npz"))
    if not batch_files:
        return {"passed": None, "note": "No batch files found in cache"}

    total_files = len(batch_files)
    bad_files = []
    total_inf = 0
    total_nan = 0
    total_above_threshold = 0
    total_values = 0

    for bf in batch_files:
        d = np.load(bf)
        if "distances" not in d:
            continue
        dists = d["distances"]
        n_inf = int(np.isinf(dists).sum())
        n_nan = int(np.isnan(dists).sum())
        n_above = int((dists >= INF_THRESHOLD).sum())
        total_inf += n_inf
        total_nan += n_nan
        total_above_threshold += n_above
        total_values += dists.size

        if n_inf > 0 or n_nan > 0 or n_above > 0:
            source_indices = d["source_indices"].tolist() if "source_indices" in d else []
            bad_files.append({
                "file": bf.name,
                "source_indices": source_indices,
                "n_inf": n_inf,
                "n_nan": n_nan,
                "n_above_threshold": n_above,
                "n_values": int(dists.size),
            })

    passed = total_inf == 0 and total_nan == 0 and total_above_threshold == 0
    result = {
        "passed": passed,
        "total_batch_files": total_files,
        "total_values_checked": total_values,
        "total_inf": total_inf,
        "total_nan": total_nan,
        "total_above_threshold": total_above_threshold,
        "inf_threshold": INF_THRESHOLD,
        "n_bad_files": len(bad_files),
    }
    if bad_files:
        result["bad_files"] = bad_files
    return result


def compare_with_reference(
    distances: np.ndarray,
    source_positions: np.ndarray,
    source_gaussian_indices: np.ndarray,
    ref_distances: np.ndarray,
    ref_source_positions: np.ndarray,
    ref_source_gaussian_indices: np.ndarray,
    gaussian_positions: np.ndarray,
) -> dict:
    """Compare geodesic distances with a reference (e.g., GT-mesh geodesics).

    Since sources may differ, we match by nearest Gaussian index instead
    of spatial position (which may be off-surface vs on-surface).
    """
    # Match by source_gaussian_indices
    ref_gauss_map = {int(gi): i for i, gi in enumerate(ref_source_gaussian_indices)}
    matched_new = []
    matched_ref = []
    for i, gi in enumerate(source_gaussian_indices):
        gi_int = int(gi)
        if gi_int in ref_gauss_map:
            matched_new.append(i)
            matched_ref.append(ref_gauss_map[gi_int])

    n_matched = len(matched_new)
    if n_matched == 0:
        return {
            "passed": None,
            "n_matched_sources": 0,
            "note": "No matching source-Gaussian pairs between new and reference",
        }

    new_dists = distances[matched_new]
    ref_dists = ref_distances[matched_ref]

    abs_diff = np.abs(new_dists - ref_dists)
    # Avoid div-by-zero for zero ref distances
    safe_ref = np.maximum(np.abs(ref_dists), 1e-10)
    rel_diff = abs_diff / safe_ref

    return {
        "n_matched_sources": n_matched,
        "n_total_pairs": int(abs_diff.size),
        "absolute_diff": {
            "min": float(np.min(abs_diff)),
            "median": float(np.median(abs_diff)),
            "mean": float(np.mean(abs_diff)),
            "max": float(np.max(abs_diff)),
            "p95": float(np.percentile(abs_diff, 95)),
            "p99": float(np.percentile(abs_diff, 99)),
        },
        "relative_diff": {
            "min": float(np.min(rel_diff)),
            "median": float(np.median(rel_diff)),
            "mean": float(np.mean(rel_diff)),
            "max": float(np.max(rel_diff)),
            "p95": float(np.percentile(rel_diff, 95)),
            "p99": float(np.percentile(rel_diff, 99)),
        },
        "correlation": float(np.corrcoef(new_dists.ravel(), ref_dists.ravel())[0, 1]),
    }


def check_source_projection(
    source_positions: np.ndarray,
    source_indices: np.ndarray,
    gaussian_positions: np.ndarray,
    source_gaussian_indices: np.ndarray,
    surface_type: str,
    mesh_vertices: np.ndarray | None,
    tolerance: float = 1e-4,
) -> dict:
    """Verify source/Gaussian relationship on a polynomial surface.

    ``source_positions`` are the original parametric grid vertices used to
    select sources.  The actual geodesic source vertex on the mesh is
    ``mesh_vertices[source_indices[i]]``.  We check:

    1. **Source-to-mesh distance** — how far the original grid point is from
       the mesh vertex that represents it.
    2. **Mesh vertex ≈ projection** — the mesh source vertex should equal
       ``project_gaussians_to_surface(gaussian_positions[sgi], surface_type)``
       (i.e. the closest point on the surface to the associated Gaussian).
    3. **Gaussian off-surface distance** — how far each Gaussian source
       center is from its projection onto the surface.
    """
    from GenerateData.utils.geodesic_mesh_utils import project_gaussians_to_surface

    result: dict = {"n_sources": int(len(source_positions))}

    # ── 1. source_positions vs mesh vertices at source_indices ─────────
    if mesh_vertices is not None:
        mesh_src = mesh_vertices[source_indices]
        src_mesh_dist = np.linalg.norm(source_positions - mesh_src, axis=1)
        result["source_to_mesh_vertex"] = {
            "min": float(src_mesh_dist.min()),
            "mean": float(src_mesh_dist.mean()),
            "max": float(src_mesh_dist.max()),
        }
    else:
        mesh_src = None
        result["source_to_mesh_vertex"] = {"note": "no mesh available"}

    # ── 2. mesh vertex ≈ projection of Gaussian onto surface ──────────
    gauss_at_src = gaussian_positions[source_gaussian_indices]
    projected = project_gaussians_to_surface(gauss_at_src, surface_type)

    if mesh_src is not None:
        proj_mesh_dist = np.linalg.norm(mesh_src - projected, axis=1)
        n_proj_mismatch = int(np.sum(proj_mesh_dist > tolerance))
        result["mesh_vs_projection"] = {
            "n_exceed_tolerance": n_proj_mismatch,
            "tolerance": tolerance,
            "dist_max": float(proj_mesh_dist.max()),
            "dist_mean": float(proj_mesh_dist.mean()),
        }
    else:
        n_proj_mismatch = 0
        result["mesh_vs_projection"] = {"note": "no mesh available; skipped"}

    # ── 3. Gaussian source → projection distance (off-surface offset) ─
    gauss_proj_dist = np.linalg.norm(gauss_at_src - projected, axis=1)
    result["gaussian_to_projection"] = {
        "min": float(gauss_proj_dist.min()),
        "mean": float(gauss_proj_dist.mean()),
        "max": float(gauss_proj_dist.max()),
        "median": float(np.median(gauss_proj_dist)),
    }

    result["passed"] = n_proj_mismatch == 0
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify geodesic distance computation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gaussian_output", type=str, required=True,
        help="Path to Gaussian splatting output folder.",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to a reference gt_geodesic.npz for comparison.",
    )
    parser.add_argument(
        "--surface", type=str, default=None,
        choices=["Paraboloid", "Saddle", "HyperbolicParaboloid"],
        help="Polynomial surface type.  When provided, verifies that "
             "source positions are correct projections onto the surface.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed results.",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t0 = time.time()

    output_dir = Path(args.gaussian_output)
    geodesic_path = output_dir / "geodesic_distance" / "gt_geodesic.npz"

    if not geodesic_path.exists():
        print(f"Error: {geodesic_path} not found")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  Geodesic Distance Verification")
    print(f"{'='*72}")
    print(f"  Data: {geodesic_path}")

    data = load_geodesic_data(geodesic_path)
    distances = data["geodesic_distances"]
    source_positions = data["source_positions"]
    source_indices = data["source_indices"]
    gaussian_positions = data["gaussian_positions"]
    source_gaussian_indices = data["source_gaussian_indices"]
    closest_mesh_indices = data["closest_mesh_indices"]
    closest_mesh_distances = data["closest_mesh_distances"]

    n_sources, n_gauss = distances.shape
    print(f"  Sources: {n_sources}, Gaussians: {n_gauss}")
    print(f"  Total distance pairs: {distances.size:,}")

    # Try to load the Gaussian mesh for on-surface position checks
    mesh_verts, mesh_faces, gauss_vi = _try_load_mesh(output_dir)
    has_mesh = mesh_verts is not None
    if has_mesh:
        print(f"  Mesh loaded: {len(mesh_verts)} vertices, {len(mesh_faces)} faces")
    else:
        print(f"  No Gaussian mesh found (Euclidean checks will use off-surface positions)")

    report: dict = {
        "data_path": str(geodesic_path),
        "n_sources": n_sources,
        "n_gaussians": n_gauss,
        "has_mesh": has_mesh,
        "checks": {},
    }

    n_checks = 9 + bool(args.surface) + bool(args.reference)

    # ── 1. Non-negativity ──────────────────────────────────────────────
    print(f"\n  [1/{n_checks}] Non-negativity ... ", end="", flush=True)
    result = check_non_negativity(distances)
    print("PASS" if result["passed"] else f"FAIL ({result['n_negative']} negative)")
    report["checks"]["non_negativity"] = result

    # ── 2. Self-distance ───────────────────────────────────────────────
    print(f"  [2/{n_checks}] Self-distance ... ", end="", flush=True)
    result = check_self_distance(
        distances, source_gaussian_indices,
        source_indices, mesh_verts, gauss_vi,
    )
    if result["passed"]:
        print(f"PASS (max={result['max']:.6f}, geodesic ≥ Euclidean)")
    elif result["passed"] is False:
        print(f"FAIL")
    else:
        print(f"OK (no mesh for full check, max={result['max']:.6f})")
    if args.verbose:
        print(f"        self-dist: min={result['min']:.6f}, median={result['median']:.6f}, "
              f"max={result['max']:.6f}")
        if "euclidean_comparison" in result:
            ec = result["euclidean_comparison"]
            print(f"        eucl to nearest gauss: mean={ec['mean_eucl_to_nearest_gauss']:.6f}, "
                  f"max={ec['max_eucl_to_nearest_gauss']:.6f}")
            print(f"        geodesic/euclidean ratio: min={ec['ratio_geodesic_over_euclidean_min']:.4f}")
    report["checks"]["self_distance"] = result

    # ── 3. Triangle inequality ─────────────────────────────────────────
    print(f"  [3/{n_checks}] Triangle inequality ... ", end="", flush=True)
    result = check_triangle_inequality(distances, source_gaussian_indices)
    status = "PASS" if result["passed"] else f"FAIL ({result['n_violations']}/{result['n_tested']})"
    print(status)
    if args.verbose and result.get("n_violations", 0) > 0:
        print(f"        max violation: {result['max_violation']:.6f}")
    report["checks"]["triangle_inequality"] = result

    # ── 4. Euclidean lower bound ───────────────────────────────────────
    print(f"  [4/{n_checks}] Geodesic ≥ Euclidean (on-surface) ... ", end="", flush=True)
    result = check_euclidean_lower_bound(
        distances, source_indices, mesh_verts, gauss_vi,
    )
    if result.get("passed") is None:
        print("SKIP (no mesh)")
    elif result["passed"]:
        print("PASS")
    else:
        print(f"FAIL ({result['n_violations']}/{result['n_tested']} violations)")
    if args.verbose and result.get("geodesic_to_euclidean_ratio"):
        r = result["geodesic_to_euclidean_ratio"]
        print(f"        geodesic/euclidean ratio: median={r['median']:.4f}, "
              f"mean={r['mean']:.4f}")
    report["checks"]["euclidean_lower_bound"] = result

    # ── 5. Symmetry ────────────────────────────────────────────────────
    print(f"  [5/{n_checks}] Symmetry (adjusted for source offsets) ... ", end="", flush=True)
    result = check_symmetry(
        distances, source_gaussian_indices,
        source_indices, mesh_verts, gauss_vi,
    )
    if result["passed"]:
        print("PASS")
    else:
        print(f"FAIL ({result['n_adjusted_violations']} adjusted violations)")
    if args.verbose and result["n_tested"] > 0:
        print(f"        raw: {result['n_raw_asymmetric_gt5pct']} pairs with >5% relative diff")
        print(f"        abs diff: median={result['absolute_diff']['median']:.6f}, "
              f"max={result['absolute_diff']['max']:.6f}")
        print(f"        rel diff: median={result['relative_diff']['median']:.6f}, "
              f"max={result['relative_diff']['max']:.6f}")
    report["checks"]["symmetry"] = result

    # ── 6. Distance statistics ─────────────────────────────────────────
    print(f"  [6/{n_checks}] Distance statistics ... ", end="", flush=True)
    result = check_distance_statistics(distances)
    if result["passed"]:
        print("PASS")
    else:
        parts = []
        if result["n_nan"] > 0:
            parts.append(f"{result['n_nan']} NaN")
        if result["n_inf"] > 0:
            parts.append(f"{result['n_inf']} Inf")
        if result["n_above_threshold"] > 0:
            parts.append(f"{result['n_above_threshold']} above {INF_THRESHOLD:.0e}")
        print(f"FAIL ({', '.join(parts)})")
    if args.verbose:
        print(f"        range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"        mean={result['mean']:.4f}, std={result['std']:.4f}")
        print(f"        NaN: {result['n_nan']}, Inf: {result['n_inf']}, Zeros: {result['n_zeros']}, >threshold: {result['n_above_threshold']}")
    report["checks"]["distance_statistics"] = result

    # ── 7. Closest mesh mapping ────────────────────────────────────────
    print(f"  [7/{n_checks}] Closest mesh mapping ... ", end="", flush=True)
    result = check_closest_mesh_mapping(closest_mesh_indices, closest_mesh_distances)
    all_zero = result["n_zero_distance"] == result["n_gaussians"]
    print("OK (Gaussian-mesh: all zero)" if all_zero
          else f"OK (max_dist={result['closest_distance_max']:.6f})")
    report["checks"]["closest_mesh_mapping"] = result

    # ── 8. Gaussian consistency (PLY ↔ mesh ↔ geodesic) ──────────────
    print(f"  [8/{n_checks}] Gaussian consistency (PLY ↔ mesh ↔ geodesic) ... ", end="", flush=True)
    result = check_gaussian_consistency(
        output_dir, gaussian_positions,
        closest_mesh_indices, closest_mesh_distances,
        mesh_verts, gauss_vi, args.surface,
    )
    if result["passed"]:
        print("PASS")
    elif result["passed"] is False:
        failed_subs = [k for k, v in result["sub_checks"].items()
                       if v.get("passed") is False]
        print(f"FAIL ({', '.join(failed_subs)})")
    else:
        print("SKIP")
    if args.verbose:
        for name, sub in result["sub_checks"].items():
            status = "PASS" if sub.get("passed") is True else (
                "FAIL" if sub.get("passed") is False else "SKIP")
            extra = ""
            if name == "ply_match" and "ply_count" in sub:
                extra = f" (PLY iter {sub['ply_iteration']}: {sub['ply_count']} gaussians, max_diff={sub['max_diff']:.2e})"
            elif name == "count_consistency":
                extra = f" ({sub['counts']})"
            elif name == "indices_match" and "n_differ" in sub:
                extra = f" ({sub['n_differ']} differ)"
            elif name == "distances_zero":
                extra = f" (max={sub['max']:.2e}, {sub['n_nonzero']} nonzero)"
            elif name == "vertex_indices_valid" and "n_unique" in sub:
                extra = f" ({sub['n_unique']}/{sub['n_total']} unique, in_range={sub['in_range']})"
            elif name == "projection_match" and "max_diff" in sub:
                extra = f" (max_diff={sub['max_diff']:.2e}, {sub['n_exceed_tolerance']} exceed tol)"
            print(f"        {name}: {status}{extra}")
    report["checks"]["gaussian_consistency"] = result

    # ── 9. Batch cache inf/NaN check ──────────────────────────────────
    print(f"  [9/{n_checks}] Batch cache inf/NaN scan ... ", end="", flush=True)
    result = check_batch_cache(output_dir)
    if result.get("passed") is None:
        print("SKIP (no cache)")
    elif result["passed"]:
        print(f"PASS ({result['total_batch_files']} files, {result['total_values_checked']:,} values)")
    else:
        parts = []
        if result['total_inf'] > 0:
            parts.append(f"{result['total_inf']} inf")
        if result['total_nan'] > 0:
            parts.append(f"{result['total_nan']} nan")
        if result['total_above_threshold'] > 0:
            parts.append(f"{result['total_above_threshold']} above {INF_THRESHOLD:.0e}")
        print(f"FAIL ({result['n_bad_files']} bad files: {', '.join(parts)})")
        if args.verbose and result.get("bad_files"):
            for bf in result["bad_files"]:
                print(f"        {bf['file']}: sources={bf['source_indices']}, "
                      f"inf={bf['n_inf']}, nan={bf['n_nan']}, >thr={bf['n_above_threshold']}")
    report["checks"]["batch_cache"] = result

    # ── 10 (optional). Source projection onto polynomial surface ────────
    check_idx = 10
    if args.surface:
        print(f"  [{check_idx}/{n_checks}] Source projection check ({args.surface}) ... ", end="", flush=True)
        result = check_source_projection(
            source_positions, source_indices, gaussian_positions,
            source_gaussian_indices, args.surface, mesh_verts,
        )
        if result["passed"]:
            smv = result.get("source_to_mesh_vertex", {})
            gtp = result["gaussian_to_projection"]
            print(f"PASS (src→mesh_vert max={smv.get('max', 0):.4f}, gauss→proj max={gtp['max']:.4f})")
        else:
            mp = result["mesh_vs_projection"]
            print(f"FAIL ({mp['n_exceed_tolerance']} mesh-vs-projection exceed tol={mp['tolerance']:.0e}, max={mp['dist_max']:.6f})")
        if args.verbose:
            smv = result.get("source_to_mesh_vertex", {})
            if "max" in smv:
                print(f"        source_pos → mesh vertex: mean={smv['mean']:.6f}, max={smv['max']:.6f}")
            gtp = result["gaussian_to_projection"]
            print(f"        gauss source → projection: mean={gtp['mean']:.6f}, max={gtp['max']:.6f}, median={gtp['median']:.6f}")
        report["checks"]["source_projection"] = result
        check_idx += 1

    # ── Reference comparison (optional) ────────────────────────────────
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            print(f"\n  [{check_idx}/{n_checks}] Comparing with reference: {ref_path.name}")
            ref_data = load_geodesic_data(ref_path)
            result = compare_with_reference(
                distances=distances,
                source_positions=source_positions,
                source_gaussian_indices=source_gaussian_indices,
                ref_distances=ref_data["geodesic_distances"],
                ref_source_positions=ref_data["source_positions"],
                ref_source_gaussian_indices=ref_data["source_gaussian_indices"],
                gaussian_positions=gaussian_positions,
            )
            report["checks"]["reference_comparison"] = result

            if result["n_matched_sources"] > 0:
                print(f"        Matched {result['n_matched_sources']} sources "
                      f"({result['n_total_pairs']:,} pairs)")
                print(f"        Abs diff: median={result['absolute_diff']['median']:.6f}, "
                      f"max={result['absolute_diff']['max']:.6f}, "
                      f"p95={result['absolute_diff']['p95']:.6f}")
                print(f"        Rel diff: median={result['relative_diff']['median']:.6f}, "
                      f"max={result['relative_diff']['max']:.6f}")
                print(f"        Correlation: {result['correlation']:.6f}")
            else:
                print(f"        No matching sources found")
        else:
            print(f"\n  [{check_idx}/{n_checks}] Reference file not found: {ref_path}")

    # ── Summary ────────────────────────────────────────────────────────
    all_checks = report["checks"]
    n_pass = sum(1 for c in all_checks.values() if c.get("passed") is True)
    n_fail = sum(1 for c in all_checks.values() if c.get("passed") is False)
    n_info = sum(1 for c in all_checks.values() if "passed" not in c or c.get("passed") is None)

    report["summary"] = {
        "passed": n_pass,
        "failed": n_fail,
        "info_only": n_info,
        "all_passed": n_fail == 0,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    print(f"\n{'='*72}")
    print(f"  SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_info} info/skip")
    print(f"{'='*72}")

    report_path = output_dir / "geodesic_distance" / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}\n")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
