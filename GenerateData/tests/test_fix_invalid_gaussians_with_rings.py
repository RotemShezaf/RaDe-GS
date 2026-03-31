#!/usr/bin/env python3
"""Tests for _fix_invalid_gaussians_with_rings.

Verifies:
  - Ring points are added around Gaussians in bad triangles
  - Adjacent Gaussians share ring points (deduplication)
  - Gaussian positions are never moved
  - All Gaussians remain in the mesh
  - Output vertices lie on the surface
  - No-op when no bad Gaussian-touching triangles exist
  - Integration with refine_bad_triangles (first iteration ring fix)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.geodesic_mesh_utils import (
    _fix_invalid_gaussians_with_rings,
    _triangle_quality,
    build_surface_delaunay,
    refine_bad_triangles,
    _filter_boundary_long_edges,
)
from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_surface_points(surface_type, n=200, seed=42, x_range=(-1, 1)):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(x_range[0], x_range[1], (n, 2))
    z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
    return np.column_stack([xy, z])


def _build_mesh_with_slivers(surface_type, n_gauss=100):
    """Build a mesh that has bad Gaussian-touching triangles.

    Injects near-collinear Gaussian points that guarantee slivers.
    """
    rng = np.random.default_rng(42)
    # Regular Gaussians
    xy = rng.uniform(-0.8, 0.8, (n_gauss, 2))
    # Add collinear Gaussians to create slivers
    sliver_xy = np.array([
        [0.0, 0.0],
        [0.001, 0.0],     # nearly identical x → sliver with first
        [0.002, 0.0001],  # another near-collinear point
    ])
    xy = np.vstack([xy, sliver_xy])
    total_gauss = len(xy)
    z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
    pts = np.column_stack([xy, z])

    verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
    gauss_idx = np.arange(total_gauss, dtype=np.int32)
    is_gaussian = np.zeros(len(verts), dtype=bool)
    is_gaussian[gauss_idx] = True
    return verts, faces, is_gaussian, gauss_idx


def _count_bad_gauss_triangles(vertices, faces, is_gaussian, ar_thresh=3.0,
                                angle_thresh=15.0, edge_thresh=None):
    """Count Gaussian-touching bad triangles."""
    ar, min_angle, longest = _triangle_quality(vertices, faces)
    touches = (
        is_gaussian[faces[:, 0]]
        | is_gaussian[faces[:, 1]]
        | is_gaussian[faces[:, 2]]
    )
    bad = touches & ((ar > ar_thresh) | (min_angle < angle_thresh))
    if edge_thresh is not None:
        bad = bad | (touches & (longest > edge_thresh))
    return int(bad.sum())


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(params=["Paraboloid", "Saddle", "HyperbolicParaboloid"])
def surface_type(request):
    return request.param


# ── Tests for _fix_invalid_gaussians_with_rings ──────────────────────────────

class TestFixInvalidGaussiansWithRings:

    def test_ring_points_added(self, surface_type):
        """Ring points are inserted around Gaussians in bad triangles."""
        verts, faces, is_gauss, _ = _build_mesh_with_slivers(surface_type)
        n_before = len(verts)
        # Ensure there are bad triangles
        n_bad = _count_bad_gauss_triangles(verts, faces, is_gauss)
        if n_bad == 0:
            pytest.skip("No bad Gaussian-touching triangles in this mesh")

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
            verbose=True,
        )
        assert len(v2) > n_before, "Ring points should have been added"

    def test_gaussian_positions_unchanged(self, surface_type):
        """Gaussian vertex positions remain exactly the same."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)
        gauss_before = verts[gauss_idx].copy()

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
        )
        # Gaussian indices are the first len(gauss_idx) vertices (unchanged)
        np.testing.assert_allclose(
            v2[gauss_idx], gauss_before, atol=1e-12,
            err_msg="Gaussian vertices must not be moved",
        )

    def test_gaussians_still_marked(self, surface_type):
        """is_gaussian mask is correctly extended for new ring points."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
        )
        assert len(ig2) == len(v2), "is_gaussian must match vertex count"
        # Original Gaussians still marked
        for gi in gauss_idx:
            assert ig2[gi], f"Gaussian {gi} lost its mark"
        # New vertices are NOT Gaussians
        for i in range(len(verts), len(v2)):
            assert not ig2[i], f"Ring vertex {i} should not be Gaussian"

    def test_all_vertices_on_surface(self, surface_type):
        """All vertices (old + new ring points) lie on the surface."""
        verts, faces, is_gauss, _ = _build_mesh_with_slivers(surface_type)

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
        )
        expected_z = evaluate_polynomial(v2[:, 0], v2[:, 1], surface_type)
        np.testing.assert_allclose(v2[:, 2], expected_z, atol=1e-10)

    def test_face_indices_valid(self, surface_type):
        """All face indices are within bounds."""
        verts, faces, is_gauss, _ = _build_mesh_with_slivers(surface_type)

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    def test_noop_when_no_bad_triangles(self, surface_type):
        """When all Gaussian-touching triangles are fine, nothing changes."""
        # Build a nice regular mesh with well-spaced Gaussians
        pts = _make_surface_points(surface_type, n=50, seed=7)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        is_gauss = np.zeros(len(verts), dtype=bool)
        is_gauss[:50] = True

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=10.0,
            surface_type=surface_type,
            gauss_ar_threshold=100.0,   # very permissive → no bad triangles
            gauss_angle_threshold=0.1,
        )
        # No vertices added
        assert len(v2) == len(verts)
        np.testing.assert_array_equal(f2, faces)

    def test_deduplication_reduces_ring_points(self):
        """Two adjacent Gaussians share ring points via deduplication.

        The dedup spacing is target_r_3d * 0.4 = max_edge * 0.65 * 0.4.
        Place Gaussians close enough that their ring points fall within
        the same dedup cell.
        """
        surface_type = "Paraboloid"
        max_edge = 0.5
        # dedup_spacing = 0.5 * 0.65 * 0.4 = 0.13
        # Place Gaussians closer than dedup_spacing so ring points aimed
        # at each other get merged.
        xy = np.array([
            [0.0, 0.0],
            [0.05, 0.0],  # distance 0.05 < dedup_spacing 0.13
        ])
        z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
        # Add surrounding points so we get a mesh
        rng = np.random.default_rng(42)
        extra_xy = rng.uniform(-0.5, 0.5, (30, 2))
        extra_z = evaluate_polynomial(extra_xy[:, 0], extra_xy[:, 1], surface_type)
        all_xy = np.vstack([xy, extra_xy])
        all_z = np.concatenate([z, extra_z])
        pts = np.column_stack([all_xy, all_z])

        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        is_gauss = np.zeros(len(verts), dtype=bool)
        is_gauss[:2] = True  # Only the first two are Gaussians

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=max_edge,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
            verbose=True,
        )
        # With two Gaussians, 12 ring points would be generated naively.
        # Deduplication should reduce this count.
        n_added = len(v2) - len(verts)
        assert n_added < 12, (
            f"Expected deduplication to merge some ring points, "
            f"but {n_added} were added (full count would be 12)"
        )

    def test_ring_points_improve_structure(self, surface_type):
        """Ring insertion adds vertices, giving subsequent refinement
        more degrees of freedom.  The total face count should increase
        (more, smaller triangles around bad regions)."""
        verts, faces, is_gauss, _ = _build_mesh_with_slivers(surface_type)
        n_bad_before = _count_bad_gauss_triangles(verts, faces, is_gauss)
        if n_bad_before == 0:
            pytest.skip("No bad Gaussian-touching triangles")

        v2, f2, ig2 = _fix_invalid_gaussians_with_rings(
            verts, faces, is_gauss, max_edge_length=0.5,
            surface_type=surface_type,
            gauss_ar_threshold=3.0, gauss_angle_threshold=15.0,
        )
        # More vertices and faces → finer mesh near bad Gaussians
        assert len(v2) > len(verts)
        assert len(f2) > len(faces)


# ── Integration with refine_bad_triangles ────────────────────────────────────

class TestRefineWithRingFix:
    """Verify that refine_bad_triangles calls the ring fix in the first round."""

    def test_refine_adds_ring_points_first(self, surface_type):
        """refine_bad_triangles should add ring points (more vertices than
        just smoothing + splitting would produce in iteration 0)."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)
        n_before = len(verts)

        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=1,
            ring_fix_invalid_gaussians=True,
            verbose=True,
        )
        # Vertices should increase due to ring + splitting
        assert len(v2) > n_before

    def test_gaussian_positions_preserved_with_ring_fix(self, surface_type):
        """Gaussian positions must still be exactly preserved after ring fix
        + refine."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)
        gauss_pos_before = verts[gauss_idx].copy()

        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=3,
            ring_fix_invalid_gaussians=True,
        )
        np.testing.assert_allclose(
            v2[g2], gauss_pos_before, atol=1e-12,
            err_msg="Gaussian positions changed during ring fix + refine",
        )

    def test_all_gaussians_retained_with_ring_fix(self, surface_type):
        """No Gaussians are lost during ring fix + refine."""
        n_gauss = 103  # 100 + 3 sliver-inducing
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)
        expected_n = len(gauss_idx)

        _, _, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=3,
            ring_fix_invalid_gaussians=True,
        )
        assert len(g2) == expected_n, (
            f"Expected {expected_n} Gaussians, got {len(g2)}"
        )

    def test_quality_improves_with_ring_fix(self, surface_type):
        """Bad-triangle fraction should not increase substantially after
        full refinement (ring fix + smoothing + splitting)."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)

        def _bad_frac(v, f):
            ar, ma, _ = _triangle_quality(v, f)
            n_bad = int(((ar > 5.0) | (ma < 10.0)).sum())
            return n_bad / max(len(f), 1)

        frac_before = _bad_frac(verts, faces)

        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=5,
            ring_fix_invalid_gaussians=True,
        )
        frac_after = _bad_frac(v2, f2)
        # Allow a small tolerance — ring insertion may create new faces
        assert frac_after <= frac_before + 0.05, (
            f"Bad-triangle fraction increased too much: "
            f"{frac_before:.3f} → {frac_after:.3f}"
        )

    def test_no_ring_fix_when_zero_iterations(self, surface_type):
        """max_iterations=0 should skip ring fix entirely."""
        verts, faces, is_gauss, gauss_idx = _build_mesh_with_slivers(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=0,
        )
        np.testing.assert_array_equal(v2, verts)
        np.testing.assert_array_equal(f2, faces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
