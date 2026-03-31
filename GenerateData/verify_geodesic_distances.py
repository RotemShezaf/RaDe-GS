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
    return {
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
        "n_inf": int(np.sum(np.isinf(distances))),
        "n_nan": int(np.sum(np.isnan(distances))),
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

    n_checks = 8 if args.reference else 7

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
    has_issues = result["n_nan"] > 0 or result["n_inf"] > 0
    print("WARN" if has_issues else "OK")
    if args.verbose:
        print(f"        range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"        mean={result['mean']:.4f}, std={result['std']:.4f}")
        print(f"        NaN: {result['n_nan']}, Inf: {result['n_inf']}, Zeros: {result['n_zeros']}")
    report["checks"]["distance_statistics"] = result

    # ── 7. Closest mesh mapping ────────────────────────────────────────
    print(f"  [7/{n_checks}] Closest mesh mapping ... ", end="", flush=True)
    result = check_closest_mesh_mapping(closest_mesh_indices, closest_mesh_distances)
    all_zero = result["n_zero_distance"] == result["n_gaussians"]
    print("OK (Gaussian-mesh: all zero)" if all_zero
          else f"OK (max_dist={result['closest_distance_max']:.6f})")
    report["checks"]["closest_mesh_mapping"] = result

    # ── 8. Reference comparison ────────────────────────────────────────
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            print(f"\n  [8/{n_checks}] Comparing with reference: {ref_path.name}")
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
            print(f"\n  [8/{n_checks}] Reference file not found: {ref_path}")

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


if __name__ == "__main__":
    main()
