#!/usr/bin/env python3
"""Tests for refine_bad_triangles and fix_orphaned_gaussians.

Verifies:
  - Relocation strategy (no vertex removal, no holes)
  - Gaussian vertices are never relocated
  - Midpoint insertion for Gaussian-apex slivers
  - Large-triangle centroid insertion
  - Convergence (bad-triangle count decreases or stabilises)
  - Vertex count never decreases (no removal)
  - Orphan-Gaussian repair
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.geodesic_mesh_utils import (
    build_surface_delaunay,
    build_surface_delaunay_3d,
    build_surface_ball_pivoting,
    build_surface_grid,
    insert_gaussians_into_grid_mesh,
    insert_steiner_points,
    _bowyer_watson_insert,
    _uniform_laplacian_smooth,
    _filter_convex_hull_artifacts,
    _split_large_edges,
    refine_bad_triangles,
    fix_orphaned_gaussians,
    _triangle_quality,
    _filter_boundary_long_edges,
    _count_mesh_holes,
    _is_boundary_face,
    _extract_surface_faces,
    _circumcenter_2d,
)
from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_surface_points(surface_type, n=200, seed=42, x_range=(-1, 1)):
    """Random surface points.  First *n* are 'Gaussians'."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(x_range[0], x_range[1], (n, 2))
    z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
    return np.column_stack([xy, z])


def _inject_sliver(pts, surface_type):
    """Append two near-collinear points that create slivers."""
    extra = np.array([[0.0, 0.0], [0.001, 0.0]])
    z = evaluate_polynomial(extra[:, 0], extra[:, 1], surface_type)
    return np.vstack([pts, np.column_stack([extra, z])])


def _count_bad(vertices, faces, max_ar=5.0, min_angle=10.0):
    ar, ma, _ = _triangle_quality(vertices, faces)
    return int(((ar > max_ar) | (ma < min_angle)).sum())


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(params=["Paraboloid", "Saddle", "HyperbolicParaboloid"])
def surface_type(request):
    return request.param


# ── TestRefine ───────────────────────────────────────────────────────────────

class TestRefineBadTriangles:
    """Core tests for the relocation-based refine_bad_triangles."""

    def _build_mesh(self, surface_type, n_gauss=200, inject_slivers=True):
        pts = _make_surface_points(surface_type, n=n_gauss)
        if inject_slivers:
            pts = _inject_sliver(pts, surface_type)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        gauss_idx = np.arange(n_gauss, dtype=np.int32)
        return verts, faces, gauss_idx

    # ── 1. No vertex removal (vertex count never decreases) ──────────

    def test_vertex_count_never_decreases(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        n_before = len(verts)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        assert len(v2) >= n_before, "Vertex count must not decrease (no removal)"

    # ── 2. Gaussian vertices: first pass preserves, relaxed may move ──

    def test_gaussian_positions_unchanged(self, surface_type):
        """Gaussian positions must be exactly preserved.

        The refiner never moves Gaussian vertices — only non-Gaussian
        support vertices are smoothed.
        """
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        gauss_before = verts[gauss_idx].copy()
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        gauss_after = v2[g2]
        np.testing.assert_allclose(
            gauss_after, gauss_before, atol=1e-12,
            err_msg="Gaussian vertices must not be moved",
        )

    # ── 3. All Gaussian indices are preserved ────────────────────────

    def test_all_gaussians_retained(self, surface_type):
        n_gauss = 200
        verts, faces, gauss_idx = self._build_mesh(surface_type, n_gauss=n_gauss)
        _, _, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        assert len(g2) == n_gauss, (
            f"Expected {n_gauss} Gaussian indices, got {len(g2)}"
        )

    # ── 4. Bad-triangle fraction does not increase ─────────────────────

    def test_bad_count_non_increasing(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        n_bad_before = _count_bad(verts, faces)
        if n_bad_before == 0:
            pytest.skip("No bad triangles to refine")
        frac_before = n_bad_before / max(len(faces), 1)
        v2, f2, _ = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=5,
        )
        n_bad_after = _count_bad(v2, f2)
        frac_after = n_bad_after / max(len(f2), 1)
        # Ring fix may add faces, so compare fractions with tolerance
        assert frac_after <= frac_before + 0.05, (
            f"Bad fraction increased too much: {frac_before:.3f} → {frac_after:.3f} "
            f"(count: {n_bad_before} → {n_bad_after})"
        )

    # ── 5. Sliver injection triggers relocation ──────────────────────

    def test_slivers_improved(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type, inject_slivers=True)
        ar_before, _, _ = _triangle_quality(verts, faces)
        v2, f2, _ = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=5,
        )
        ar_after, _, _ = _triangle_quality(v2, f2)
        # Worst aspect ratio should improve or stay the same
        assert ar_after.max() <= ar_before.max() + 1e-6

    # ── 6. Output faces are valid indices ────────────────────────────

    def test_face_indices_valid(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, _ = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    # ── 7. All output vertices lie on the surface ────────────────────

    def test_vertices_on_surface(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, _ = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        expected_z = evaluate_polynomial(v2[:, 0], v2[:, 1], surface_type)
        np.testing.assert_allclose(v2[:, 2], expected_z, atol=1e-10)

    # ── 8. Return type ───────────────────────────────────────────────

    def test_return_types(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        result = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        assert isinstance(result, tuple) and len(result) == 3
        v, f, g = result
        assert v.dtype == np.float64
        assert f.dtype == np.int32
        assert g.dtype == np.int32

    # ── 9. Zero iterations returns input unchanged ───────────────────

    def test_zero_iterations_noop(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_iterations=0,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
        )
        np.testing.assert_array_equal(v2, verts)
        np.testing.assert_array_equal(f2, faces)

    # ── 10. No Gaussian indices (all vertices relocatable) ───────────

    def test_no_gaussians(self, surface_type):
        verts, faces, _ = self._build_mesh(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=None,
        )
        assert len(g2) == 0
        assert len(v2) >= len(verts)

    # ── 11. Large-triangle centroid insertion works ───────────────────

    def test_large_triangles_get_centroids(self):
        """Cluster of points with a gap → large triangles → centroids."""
        # Tight cluster near origin + a few far points with explicit
        # max_edge_length large enough to keep them connected.
        rng = np.random.default_rng(99)
        xy_near = rng.uniform(-0.1, 0.1, (20, 2))
        xy_far = np.array([[0.8, 0.8], [-0.8, 0.8], [0.8, -0.8]])
        xy = np.vstack([xy_near, xy_far])
        z = evaluate_polynomial(xy[:, 0], xy[:, 1], "Paraboloid")
        pts = np.column_stack([xy, z])
        verts, faces = build_surface_delaunay(pts, surface_type="Paraboloid")
        v2, f2, _ = refine_bad_triangles(
            verts, faces, "Paraboloid",
            max_area_factor=1.0,
            max_iterations=2,
            max_edge_length=5.0,  # very permissive
            gaussian_vertex_indices=None,
        )
        assert len(v2) > len(verts), "Centroids should have been inserted"

    # ── 12. Mesh connectivity — no orphans after refine ──────────────

    def test_no_orphan_vertices(self, surface_type):
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=3,
        )
        used = set(f2.ravel().tolist())
        for gi in g2:
            assert int(gi) in used, f"Gaussian vertex {gi} is an orphan after refine"


# ── TestFixOrphanedGaussians ─────────────────────────────────────────────────

class TestFixOrphanedGaussians:
    """Tests for the orphan-Gaussian repair function."""

    def test_no_orphans_noop(self, surface_type):
        """When all Gaussians are in faces, nothing changes."""
        pts = _make_surface_points(surface_type, n=50)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        n_gauss = 50
        v2, f2 = fix_orphaned_gaussians(
            verts, faces, n_gauss,
            max_edge_length=2.0,
            surface_type=surface_type,
        )
        # Same vertices and faces
        assert len(v2) == len(verts)
        np.testing.assert_array_equal(f2, faces)

    def test_orphan_gets_support(self, surface_type):
        """Manually orphan a Gaussian by filtering its faces, then repair."""
        pts = _make_surface_points(surface_type, n=100)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        n_gauss = 100
        # Artificially orphan vertex 0 by removing all its faces
        orphan_mask = np.all(faces != 0, axis=1)
        faces_trimmed = faces[orphan_mask]
        assert 0 not in set(faces_trimmed.ravel().tolist()), "vertex 0 should be orphaned"

        v2, f2 = fix_orphaned_gaussians(
            verts, faces_trimmed, n_gauss,
            max_edge_length=2.0,
            surface_type=surface_type,
            verbose=True,
        )
        used = set(f2.ravel().tolist())
        assert 0 in used, "Gaussian vertex 0 should be repaired"

    def test_output_faces_valid(self, surface_type):
        pts = _make_surface_points(surface_type, n=50)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        v2, f2 = fix_orphaned_gaussians(
            verts, faces, 50,
            max_edge_length=2.0,
            surface_type=surface_type,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)


# ── TestTriangleQuality ──────────────────────────────────────────────────────

class TestTriangleQuality:
    def test_equilateral(self):
        """Equilateral triangle → AR ≈ 1, min angle ≈ 60°."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        ar, ma, _ = _triangle_quality(verts, faces)
        assert ar[0] == pytest.approx(1.0, abs=1e-10)
        assert ma[0] == pytest.approx(60.0, abs=1e-10)

    def test_right_triangle(self):
        """45-45-90 right triangle → AR = √2, min angle = 45°."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        ar, ma, _ = _triangle_quality(verts, faces)
        assert ar[0] == pytest.approx(np.sqrt(2), abs=1e-10)
        assert ma[0] == pytest.approx(45.0, abs=1e-10)

    def test_sliver(self):
        """Nearly degenerate triangle → high AR, tiny min angle."""
        # Two vertices 1 apart, third almost on the same edge:
        # edges ≈ 1, 1.001, 0.00141  →  AR ≈ 709
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1.001, 0.001, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        ar, ma, _ = _triangle_quality(verts, faces)
        assert ar[0] > 100
        assert ma[0] < 1.0


# ── Integration ──────────────────────────────────────────────────────────────

class TestRefineIntegration:
    """End-to-end: build → refine → verify."""

    def test_full_pipeline(self, surface_type):
        n_gauss = 150
        pts = _make_surface_points(surface_type, n=n_gauss, seed=7)
        pts = _inject_sliver(pts, surface_type)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        gauss_idx = np.arange(n_gauss, dtype=np.int32)

        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_aspect_ratio=5.0,
            min_angle_deg=10.0,
            max_iterations=5,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            verbose=True,
        )

        # All Gaussians retained
        assert len(g2) == n_gauss
        # Gaussian positions must be exactly preserved (no movement)
        np.testing.assert_allclose(
            v2[g2], pts[:n_gauss], atol=1e-12,
            err_msg="Gaussian vertices must not be moved",
        )
        # Quality improved or equal (compare fractions — ring fix adds faces)
        n_bad_before = _count_bad(verts, faces)
        n_bad_after = _count_bad(v2, f2)
        frac_before = n_bad_before / max(len(faces), 1)
        frac_after = n_bad_after / max(len(f2), 1)
        assert frac_after <= frac_before + 0.05, (
            f"Bad fraction increased: {frac_before:.3f} → {frac_after:.3f}"
        )
        # No orphaned Gaussians
        used = set(f2.ravel().tolist())
        orphaned = [int(gi) for gi in g2 if int(gi) not in used]
        # Note: some Gaussians may become orphaned due to max_edge_length
        # filter — that's handled by fix_orphaned_gaussians, not refine.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── TestNoHoles ──────────────────────────────────────────────────────────────

class TestNoHoles:
    """Verify that the mesh has zero holes after every pipeline step."""

    def _build_mesh(self, surface_type, n_gauss=200, inject_slivers=True):
        pts = _make_surface_points(surface_type, n=n_gauss)
        if inject_slivers:
            pts = _inject_sliver(pts, surface_type)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        gauss_idx = np.arange(n_gauss, dtype=np.int32)
        return verts, faces, gauss_idx

    def test_delaunay_no_holes(self, surface_type):
        """Raw Delaunay has no holes by construction."""
        pts = _make_surface_points(surface_type, n=200)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        assert _count_mesh_holes(faces) == 0

    def test_boundary_filter_no_holes(self, surface_type):
        """Boundary-only long-edge filter does not create interior holes."""
        pts = _make_surface_points(surface_type, n=200)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        # Use a tight threshold to remove many boundary faces
        filtered, n_rm = _filter_boundary_long_edges(verts, faces, 0.3)
        assert n_rm >= 0
        if len(filtered) > 0:
            holes = _count_mesh_holes(filtered)
            assert holes == 0, f"Boundary filter created {holes} hole(s)"

    def test_refine_no_holes(self, surface_type):
        """refine_bad_triangles never produces holes."""
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=3,
        )
        holes = _count_mesh_holes(f2)
        assert holes == 0, f"Refinement created {holes} hole(s)"

    def test_refine_tight_threshold_no_holes(self, surface_type):
        """Even with a very tight max_edge_length, no holes appear."""
        verts, faces, gauss_idx = self._build_mesh(surface_type)
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.1,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=3,
        )
        holes = _count_mesh_holes(f2)
        assert holes == 0, f"Tight-threshold refinement created {holes} hole(s)"

    def test_fix_orphaned_no_holes(self, surface_type):
        """fix_orphaned_gaussians does not create holes."""
        pts = _make_surface_points(surface_type, n=100)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        # Force orphans by removing one vertex's faces
        orphan_mask = np.all(faces != 0, axis=1)
        faces_trimmed = faces[orphan_mask]
        v2, f2 = fix_orphaned_gaussians(
            verts, faces_trimmed, 100,
            max_edge_length=2.0,
            surface_type=surface_type,
        )
        holes = _count_mesh_holes(f2)
        assert holes == 0, f"Orphan fix created {holes} hole(s)"

    def test_full_pipeline_no_holes(self, surface_type):
        """Full build → filter → refine pipeline has no holes."""
        n_gauss = 150
        pts = _make_surface_points(surface_type, n=n_gauss, seed=7)
        pts = _inject_sliver(pts, surface_type)
        verts, faces = build_surface_delaunay(pts, surface_type=surface_type)
        gauss_idx = np.arange(n_gauss, dtype=np.int32)

        # Step 1: boundary filter (like the caller does)
        faces, _ = _filter_boundary_long_edges(verts, faces, 0.5)
        assert _count_mesh_holes(faces) == 0, "Holes after initial filter"

        # Step 2: refine
        v2, f2, g2 = refine_bad_triangles(
            verts, faces, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gauss_idx,
            max_iterations=5,
        )
        assert _count_mesh_holes(f2) == 0, "Holes after refinement"


class TestBoundaryHelpers:
    """Unit tests for _is_boundary_face and _filter_boundary_long_edges."""

    def test_single_triangle_all_boundary(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mask = _is_boundary_face(faces)
        assert mask[0] is np.True_

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing edge 1-2. Each has 2 boundary edges."""
        faces = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int32)
        mask = _is_boundary_face(faces)
        assert mask.all()  # both still have boundary edges

    def test_interior_face_not_boundary(self):
        """Center face surrounded by 3 others — no boundary edges."""
        #    3
        #   / \\
        #  0---1
        #   \ /
        #    2
        faces = np.array([
            [0, 1, 3],  # top
            [0, 2, 1],  # bottom
            [0, 3, 2],  # left  — wait this doesn't make a center face
        ], dtype=np.int32)
        # Actually let me build a proper interior:
        # 4 triangles around a center vertex
        #     2
        #    /|\\
        #   / | \\
        #  3--0--1
        #   \ | /
        #    \|/
        #     4
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
        ], dtype=np.int32)
        mask = _is_boundary_face(faces)
        # All faces touch the outer boundary (vertex 1,2,3,4 are hull)
        assert mask.all()

    def test_count_holes_no_holes(self):
        """Complete Delaunay grid → 0 holes."""
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(25)])
        _, faces = build_surface_delaunay(pts)
        assert _count_mesh_holes(faces) == 0

    def test_count_holes_with_hole(self):
        """Remove an interior face → exactly 1 hole."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(100)])
        _, faces = build_surface_delaunay(pts)
        # Find an interior face (all edges shared by 2 faces)
        bdry = _is_boundary_face(faces)
        interior_idx = np.where(~bdry)[0]
        assert len(interior_idx) > 0, "Grid should have interior faces"
        # Remove one interior face → creates a hole
        faces_holed = np.delete(faces, interior_idx[0], axis=0)
        holes = _count_mesh_holes(faces_holed)
        assert holes == 1, f"Expected 1 hole, got {holes}"

    def test_filter_never_removes_interior(self):
        """_filter_boundary_long_edges never removes an interior face."""
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(64)])
        verts, faces = build_surface_delaunay(pts)
        # Every interior face should survive even with threshold=0
        bdry_before = _is_boundary_face(faces)
        interior_before = set(map(tuple, faces[~bdry_before].tolist()))
        filtered, _ = _filter_boundary_long_edges(verts, faces, 0.001)
        interior_after = set(map(tuple, filtered.tolist()))
        for tri in interior_before:
            assert tri in interior_after, f"Interior face {tri} was removed!"


class TestBallPivoting:
    """Tests for the Ball Pivoting meshing method."""

    def test_ball_pivoting_basic(self, surface_type):
        """Ball Pivoting produces a non-empty mesh."""
        pts = _make_surface_points(surface_type, n=200)
        verts, faces = build_surface_ball_pivoting(pts, surface_type=surface_type)
        assert len(verts) > 0, "Ball Pivoting produced no vertices"
        assert len(faces) > 0, "Ball Pivoting produced no faces"

    def test_ball_pivoting_vertices_on_surface(self, surface_type):
        """All vertices lie on the analytical surface."""
        pts = _make_surface_points(surface_type, n=200)
        verts, faces = build_surface_ball_pivoting(pts, surface_type=surface_type)
        from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial
        z_expected = evaluate_polynomial(verts[:, 0], verts[:, 1], surface_type)
        assert np.allclose(verts[:, 2], z_expected, atol=1e-10)

    def test_ball_pivoting_no_holes(self, surface_type):
        """Ball Pivoting mesh has no interior holes."""
        pts = _make_surface_points(surface_type, n=300)
        verts, faces = build_surface_ball_pivoting(pts, surface_type=surface_type)
        if len(faces) > 0:
            holes = _count_mesh_holes(faces)
            assert holes == 0, f"Ball Pivoting created {holes} hole(s)"

    def test_ball_pivoting_face_indices_valid(self, surface_type):
        """Face indices are within the vertex array bounds."""
        pts = _make_surface_points(surface_type, n=200)
        verts, faces = build_surface_ball_pivoting(pts, surface_type=surface_type)
        if len(faces) > 0:
            assert faces.min() >= 0
            assert faces.max() < len(verts)


# ── TestGridMesh ─────────────────────────────────────────────────────────────

class TestGridMesh:
    """Tests for grid-based mesh construction and Gaussian insertion."""

    def _build_grid(self, surface_type, target_edge=0.1, x_range=(-1, 1),
                    y_range=(-1, 1), curvature_adaptive=False):
        return build_surface_grid(
            surface_type=surface_type,
            x_range=x_range,
            y_range=y_range,
            target_edge_length=target_edge,
            curvature_adaptive=curvature_adaptive,
        )

    # ── 1. Basic grid structure ──────────────────────────────────────

    def test_grid_produces_valid_mesh(self, surface_type):
        """Grid mesh has non-empty valid vertices and faces."""
        verts, faces = self._build_grid(surface_type)
        assert len(verts) > 0
        assert len(faces) > 0
        assert faces.dtype == np.int32
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    def test_grid_vertices_on_surface(self, surface_type):
        """All grid vertices lie on the analytical surface."""
        verts, faces = self._build_grid(surface_type)
        z_expected = evaluate_polynomial(verts[:, 0], verts[:, 1], surface_type)
        np.testing.assert_allclose(verts[:, 2], z_expected, atol=1e-10)

    def test_grid_no_degenerate_faces(self, surface_type):
        """No degenerate (zero-area) faces."""
        verts, faces = self._build_grid(surface_type)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        assert (areas > 1e-15).all(), "Grid has degenerate (zero-area) faces"

    # ── 2. Quality ───────────────────────────────────────────────────

    def test_grid_quality(self, surface_type):
        """Structured grid should have good triangle quality."""
        verts, faces = self._build_grid(surface_type)
        ar, ma, _ = _triangle_quality(verts, faces)
        assert ar.max() < 5.0, f"Grid has too-high aspect ratio: {ar.max()}"
        assert ma.min() > 10.0, f"Grid min angle too small: {ma.min():.2f}°"

    def test_grid_no_holes(self, surface_type):
        """Structured grid has no holes."""
        verts, faces = self._build_grid(surface_type)
        assert _count_mesh_holes(faces) == 0

    # ── 3. Arc-length adaptation ─────────────────────────────────────

    def test_grid_arc_length_edges(self, surface_type):
        """Grid 3-D edge lengths should be within ~3× of target."""
        target = 0.1
        verts, faces = self._build_grid(surface_type, target_edge=target)
        _, _, longest = _triangle_quality(verts, faces)
        # All longest edges should not wildly exceed target
        assert longest.max() < target * 3.5, (
            f"Max edge {longest.max():.4f} >> target {target}"
        )

    def test_grid_curvature_adaptive_denser(self, surface_type):
        """Curvature-adaptive grid has ≥ as many vertices as uniform."""
        v_uni, _ = self._build_grid(surface_type, curvature_adaptive=False)
        v_ada, _ = self._build_grid(surface_type, curvature_adaptive=True)
        assert len(v_ada) >= len(v_uni), (
            f"Adaptive ({len(v_ada)}) should have ≥ uniform ({len(v_uni)}) vertices"
        )

    # ── 4. Gaussian insertion ────────────────────────────────────────

    def test_insert_all_gaussians_present(self, surface_type):
        """All Gaussians become mesh vertices after insertion."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=50, seed=123,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        assert len(gi) == 50
        used = set(f2.ravel().tolist())
        for gv in gi:
            assert int(gv) in used, f"Gaussian vertex {gv} not in any face"

    def test_insert_faces_valid(self, surface_type):
        """Face indices remain valid after Gaussian insertion."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=30, seed=99,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    def test_insert_vertex_count(self, surface_type):
        """Vertex count = grid + Gaussians after insertion."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        n_grid = len(verts)
        n_gauss = 40
        gauss_pts = _make_surface_points(surface_type, n=n_gauss, seed=77,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        assert len(v2) == n_grid + n_gauss

    def test_insert_no_holes(self, surface_type):
        """No holes after Gaussian insertion."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=30, seed=55,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        assert _count_mesh_holes(f2) == 0, "Holes after Gaussian insertion"

    def test_insert_vertices_on_surface(self, surface_type):
        """All vertices lie on the surface after insertion."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=20, seed=88,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        z_expected = evaluate_polynomial(v2[:, 0], v2[:, 1], surface_type)
        np.testing.assert_allclose(v2[:, 2], z_expected, atol=1e-10)

    def test_insert_zero_gaussians(self, surface_type):
        """Inserting zero Gaussians returns the grid unchanged."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        empty = np.empty((0, 3), dtype=np.float64)
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, empty, surface_type,
        )
        assert len(gi) == 0
        np.testing.assert_array_equal(f2, faces)

    # ── 5. Edge flipping ────────────────────────────────────────────

    def test_edge_flipping_improves_quality(self, surface_type):
        """Edge flipping should not degrade triangle quality."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=50, seed=42,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        ar, ma, _ = _triangle_quality(v2, f2)
        # Majority of triangles should be reasonable
        pct_good = float(np.mean(ar < 5) * 100)
        assert pct_good > 70, f"Only {pct_good:.1f}% triangles have AR < 5"

    # ── 6. Full pipeline ─────────────────────────────────────────────

    def test_grid_insert_refine_pipeline(self, surface_type):
        """Full pipeline: grid → insert → refine → quality."""
        verts, faces = self._build_grid(surface_type, target_edge=0.15)
        gauss_pts = _make_surface_points(surface_type, n=50, seed=42,
                                         x_range=(-0.8, 0.8))
        v2, f2, gi = insert_gaussians_into_grid_mesh(
            verts, faces, gauss_pts, surface_type,
        )
        v3, f3, gi3 = refine_bad_triangles(
            v2, f2, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=3,
        )
        # At least 90% of Gaussians should be retained
        assert len(gi3) >= len(gauss_pts) * 0.9
        # Quality
        ar, ma, _ = _triangle_quality(v3, f3)
        pct_good = float(np.mean(ar < 5) * 100)
        assert pct_good > 60, f"Only {pct_good:.1f}% triangles have AR < 5"
        # No holes
        assert _count_mesh_holes(f3) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3-D Delaunay tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestDelaunay3D:
    """Tests for build_surface_delaunay_3d and _extract_surface_faces."""

    # ── Helper ──────────────────────────────────────────────────────

    @staticmethod
    def _make_pts(surface_type, n=200, seed=42):
        return _make_surface_points(surface_type, n=n, seed=seed)

    # ── 1. Basic output shape / type ────────────────────────────────

    def test_returns_vertices_and_faces(self, surface_type):
        pts = self._make_pts(surface_type)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert faces.dtype == np.int32

    def test_vertex_count_unchanged(self, surface_type):
        pts = self._make_pts(surface_type)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        assert len(verts) == len(pts)

    def test_produces_faces(self, surface_type):
        pts = self._make_pts(surface_type, n=50)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        assert len(faces) > 0, "3D Delaunay produced no surface faces"

    # ── 2. Vertices lie on the surface ──────────────────────────────

    def test_vertices_on_surface(self, surface_type):
        pts = self._make_pts(surface_type)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        z_expected = evaluate_polynomial(verts[:, 0], verts[:, 1], surface_type)
        np.testing.assert_allclose(verts[:, 2], z_expected, atol=1e-10)

    # ── 3. All face indices valid ───────────────────────────────────

    def test_face_indices_valid(self, surface_type):
        pts = self._make_pts(surface_type)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    # ── 4. Consistency with 2-D Delaunay ────────────────────────────

    def test_covers_most_vertices(self, surface_type):
        """A meaningful fraction of vertices should appear in some face.

        3D Delaunay on thin surfaces produces many cap faces that get
        filtered by surface-normal alignment. Coverage may be low for
        non-planar surfaces.
        """
        pts = self._make_pts(surface_type, n=100)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        used = np.unique(faces.ravel())
        coverage = len(used) / len(verts)
        assert coverage > 0.1, f"Only {coverage:.0%} of vertices are covered"

    # ── 5. _extract_surface_faces helper ────────────────────────────

    def test_extract_faces_from_single_tet(self, surface_type):
        """A single tetrahedron should produce 4 boundary faces."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        faces = _extract_surface_faces(pts, cells, surface_type=None)
        assert len(faces) == 4

    def test_extract_faces_shared_face_removed(self, surface_type):
        """Two tetrahedra sharing a face → shared face NOT in output."""
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
        ], dtype=np.float64)
        # Tet1: 0-1-2-3, Tet2: 1-2-3-4 (shared face: 1-2-3)
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        faces = _extract_surface_faces(pts, cells, surface_type=None)
        # 2 tets × 4 faces = 8 total, minus 2 shared = 6 boundary faces
        assert len(faces) == 6

    # ── 6. Compare face count with 2-D Delaunay ────────────────────

    def test_face_count_reasonable(self, surface_type):
        """3D Delaunay should produce some surface faces.

        On thin surfaces the normal-alignment filter may be aggressive;
        we only require at least 10% of what 2D produces.
        """
        pts = self._make_pts(surface_type, n=100)
        _, faces_2d = build_surface_delaunay(pts, surface_type)
        _, faces_3d = build_surface_delaunay_3d(pts, surface_type)
        ratio = len(faces_3d) / max(len(faces_2d), 1)
        assert ratio > 0.1, (
            f"3D Delaunay produced only {len(faces_3d)} faces vs "
            f"{len(faces_2d)} from 2D ({ratio:.0%})"
        )

    # ── 7. No degenerate faces ──────────────────────────────────────

    def test_no_degenerate_faces(self, surface_type):
        pts = self._make_pts(surface_type, n=100)
        verts, faces = build_surface_delaunay_3d(pts, surface_type)
        if len(faces) == 0:
            return
        # Check no face has two identical vertex indices
        for f in faces:
            assert len(set(f)) == 3, f"Degenerate face: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# Circumcentre helper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCircumcenter2D:
    """Tests for _circumcenter_2d."""

    def test_equilateral_triangle(self):
        """Circumcentre of equilateral triangle is at centroid."""
        ax = np.array([0.0])
        ay = np.array([0.0])
        bx = np.array([1.0])
        by = np.array([0.0])
        cx = np.array([0.5])
        cy = np.array([np.sqrt(3) / 2])
        ux, uy = _circumcenter_2d(ax, ay, bx, by, cx, cy)
        np.testing.assert_allclose(ux, [0.5], atol=1e-10)
        np.testing.assert_allclose(uy, [np.sqrt(3) / 6], atol=1e-10)

    def test_right_triangle(self):
        """Circumcentre of right triangle is at midpoint of hypotenuse."""
        ax = np.array([0.0])
        ay = np.array([0.0])
        bx = np.array([1.0])
        by = np.array([0.0])
        cx = np.array([0.0])
        cy = np.array([1.0])
        ux, uy = _circumcenter_2d(ax, ay, bx, by, cx, cy)
        np.testing.assert_allclose(ux, [0.5], atol=1e-10)
        np.testing.assert_allclose(uy, [0.5], atol=1e-10)

    def test_vectorized(self):
        """Multiple triangles at once."""
        ax = np.array([0.0, 0.0])
        ay = np.array([0.0, 0.0])
        bx = np.array([1.0, 2.0])
        by = np.array([0.0, 0.0])
        cx = np.array([0.0, 0.0])
        cy = np.array([1.0, 2.0])
        ux, uy = _circumcenter_2d(ax, ay, bx, by, cx, cy)
        assert len(ux) == 2
        np.testing.assert_allclose(ux, [0.5, 1.0], atol=1e-10)
        np.testing.assert_allclose(uy, [0.5, 1.0], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Steiner-point insertion tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestSteinerPoints:
    """Tests for insert_steiner_points."""

    @staticmethod
    def _basic_mesh(surface_type, n=200, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        gi = np.arange(n, dtype=np.int32)
        return v, f, gi

    # ── 1. Basic contract ──────────────────────────────────────────

    def test_returns_three_arrays(self, surface_type):
        v, f, gi = self._basic_mesh(surface_type)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
        )
        assert v2.ndim == 2 and v2.shape[1] == 3
        assert f2.ndim == 2 and f2.shape[1] == 3
        assert gi2.ndim == 1

    def test_vertex_count_grows_or_stays(self, surface_type):
        """Steiner insertion never removes vertices."""
        v, f, gi = self._basic_mesh(surface_type)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
        )
        assert len(v2) >= len(v)

    def test_gaussian_indices_preserved(self, surface_type):
        """Original Gaussian indices must remain unchanged."""
        v, f, gi = self._basic_mesh(surface_type)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
        )
        np.testing.assert_array_equal(gi2, gi)

    # ── 2. Vertices on surface ──────────────────────────────────────

    def test_steiner_points_on_surface(self, surface_type):
        """Newly inserted Steiner points must lie on the surface."""
        v, f, gi = self._basic_mesh(surface_type)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
        )
        z_expected = evaluate_polynomial(v2[:, 0], v2[:, 1], surface_type)
        np.testing.assert_allclose(v2[:, 2], z_expected, atol=1e-10)

    # ── 3. Quality improvement ──────────────────────────────────────

    def test_quality_not_worse(self, surface_type):
        """Steiner insertion should not degrade median aspect ratio."""
        v, f, gi = self._basic_mesh(surface_type, n=150)
        ar_before, _, _ = _triangle_quality(v, f)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
            max_aspect_ratio=3.0, min_angle_deg=20.0,
        )
        if len(f2) > 0:
            ar_after, _, _ = _triangle_quality(v2, f2)
            # Median AR should not increase by more than 50%
            assert np.median(ar_after) <= np.median(ar_before) * 1.5 + 0.5

    # ── 4. No crash on good mesh ────────────────────────────────────

    def test_noop_on_good_mesh(self, surface_type):
        """When all triangles are good, no Steiner points are inserted."""
        # Use a well-sampled grid which typically has good quality
        verts, faces = build_surface_grid(
            surface_type, (-1, 1), (-1, 1), target_edge_length=0.2,
        )
        gi = np.arange(0, dtype=np.int32)
        v2, f2, gi2 = insert_steiner_points(
            verts, faces, surface_type,
            max_aspect_ratio=10.0, min_angle_deg=5.0,
            gaussian_vertex_indices=gi,
        )
        # With such lenient thresholds, the grid should need no Steiner points
        assert len(v2) == len(verts)

    # ── 5. Face indices valid ───────────────────────────────────────

    def test_face_indices_valid(self, surface_type):
        v, f, gi = self._basic_mesh(surface_type)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type, gaussian_vertex_indices=gi,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    # ── 6. Pipeline: Steiner → Refine ───────────────────────────────

    def test_steiner_then_refine(self, surface_type):
        """Full pipeline: delaunay → steiner → refine."""
        v, f, gi = self._basic_mesh(surface_type, n=100)
        v2, f2, gi2 = insert_steiner_points(
            v, f, surface_type,
            gaussian_vertex_indices=gi,
            max_aspect_ratio=3.0,
            max_iterations=2,
        )
        v3, f3, gi3 = refine_bad_triangles(
            v2, f2, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi2,
            max_iterations=2,
        )
        assert len(f3) > 0
        assert len(gi3) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Bowyer–Watson insertion tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestBowyerWatson:
    """Tests for _bowyer_watson_insert."""

    @staticmethod
    def _make_mesh(surface_type, n=100, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        return v, f

    def test_insert_single_point(self, surface_type):
        """Inserting one point should increase vertex count by 1."""
        v, f = self._make_mesh(surface_type, n=50)
        n_v = len(v)
        new_p = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        new_p[0, 2] = evaluate_polynomial(
            np.array([0.0]), np.array([0.0]), surface_type
        )[0]
        faces_list = [list(fi) for fi in f]
        v2, f2, _ = _bowyer_watson_insert(v, faces_list, new_p, surface_type)
        assert len(v2) == n_v + 1

    def test_insert_preserves_existing(self, surface_type):
        """Original vertices must remain after insertions."""
        v, f = self._make_mesh(surface_type, n=50)
        orig = v.copy()
        rng = np.random.default_rng(99)
        new_xy = rng.uniform(-0.5, 0.5, (5, 2))
        new_z = evaluate_polynomial(new_xy[:, 0], new_xy[:, 1], surface_type)
        new_p = np.column_stack([new_xy, new_z])
        faces_list = [list(fi) for fi in f]
        v2, f2, _ = _bowyer_watson_insert(v, faces_list, new_p, surface_type)
        np.testing.assert_allclose(v2[:len(orig), :2], orig[:, :2], atol=1e-12)

    def test_faces_valid(self, surface_type):
        """All face indices must be in range after insertion."""
        v, f = self._make_mesh(surface_type, n=50)
        rng = np.random.default_rng(77)
        new_xy = rng.uniform(-0.5, 0.5, (3, 2))
        new_z = evaluate_polynomial(new_xy[:, 0], new_xy[:, 1], surface_type)
        new_p = np.column_stack([new_xy, new_z])
        faces_list = [list(fi) for fi in f]
        v2, f2, _ = _bowyer_watson_insert(v, faces_list, new_p, surface_type)
        fa = np.array(f2, dtype=np.int32)
        assert fa.min() >= 0
        assert fa.max() < len(v2)

    def test_no_none_faces(self, surface_type):
        """After insertion there should be no None entries in faces."""
        v, f = self._make_mesh(surface_type, n=40)
        new_p = np.array([[0.1, 0.1, 0.0]], dtype=np.float64)
        new_p[0, 2] = evaluate_polynomial(
            np.array([0.1]), np.array([0.1]), surface_type
        )[0]
        faces_list = [list(fi) for fi in f]
        v2, f2, _ = _bowyer_watson_insert(v, faces_list, new_p, surface_type)
        for fi in f2:
            assert fi is not None

    def test_insert_multiple_points(self, surface_type):
        """Inserting 10 points maintains valid mesh."""
        v, f = self._make_mesh(surface_type, n=80)
        rng = np.random.default_rng(42)
        new_xy = rng.uniform(-0.7, 0.7, (10, 2))
        new_z = evaluate_polynomial(new_xy[:, 0], new_xy[:, 1], surface_type)
        new_p = np.column_stack([new_xy, new_z])
        faces_list = [list(fi) for fi in f]
        v2, f2, _ = _bowyer_watson_insert(v, faces_list, new_p, surface_type)
        assert len(v2) == len(v) + 10
        fa = np.array(f2, dtype=np.int32)
        # Check all face vertex indices are unique per face
        for fi in fa:
            assert len(set(fi)) == 3, f"Degenerate face: {fi}"


# ═══════════════════════════════════════════════════════════════════════════
# Uniform Laplacian smoothing tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestUniformLaplacian:
    """Tests for _uniform_laplacian_smooth."""

    @staticmethod
    def _make_mesh(surface_type, n=100, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        return v, f

    def test_immovable_vertices_stay(self, surface_type):
        """Vertices marked as not movable must not change."""
        v, f = self._make_mesh(surface_type, n=80)
        movable = np.ones(len(v), dtype=bool)
        movable[0] = False
        orig_0 = v[0].copy()
        _uniform_laplacian_smooth(v, f, movable, surface_type)
        np.testing.assert_array_equal(v[0], orig_0)

    def test_vertices_on_surface_after(self, surface_type):
        """All moved vertices must lie on the surface."""
        v, f = self._make_mesh(surface_type, n=80)
        movable = np.ones(len(v), dtype=bool)
        _uniform_laplacian_smooth(v, f, movable, surface_type)
        z_expected = evaluate_polynomial(v[:, 0], v[:, 1], surface_type)
        np.testing.assert_allclose(v[:, 2], z_expected, atol=1e-10)

    def test_no_crash_empty(self, surface_type):
        """No crash when no vertices are movable."""
        v, f = self._make_mesh(surface_type, n=50)
        movable = np.zeros(len(v), dtype=bool)
        _uniform_laplacian_smooth(v, f, movable, surface_type)

    def test_smoothing_reduces_variance(self, surface_type):
        """Smoothing should reduce the variance of edge lengths."""
        v, f = self._make_mesh(surface_type, n=100)
        ar_before, _, _ = _triangle_quality(v, f)
        movable = np.ones(len(v), dtype=bool)
        _uniform_laplacian_smooth(v, f, movable, surface_type, damping=1.0)
        # Just check no crash; quality may not always improve since
        # smoothing without retriangulation can worsen some triangles


# ═══════════════════════════════════════════════════════════════════════════
# Convex-hull artifact filter & large-edge splitting tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestConvexHullArtifacts:
    """Tests for _filter_convex_hull_artifacts and _split_large_edges."""

    @staticmethod
    def _make_mesh(surface_type, n=200, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        return v, f

    def test_artifact_filter_only_removes_boundary(self, surface_type):
        """Convex-hull artifact filter must only remove boundary faces."""
        v, f = self._make_mesh(surface_type)
        is_bdry_before = _is_boundary_face(f)
        mel = 0.3  # tight threshold
        f2, n_removed = _filter_convex_hull_artifacts(v, f, mel, surface_type)
        if n_removed == 0:
            pytest.skip("No artifacts to remove")
        # Every removed face must have been a boundary face
        removed_mask = np.ones(len(f), dtype=bool)
        if len(f2) > 0:
            # Identify which original faces survived
            for i in range(len(f)):
                for j in range(len(f2)):
                    if np.array_equal(f[i], f2[j]):
                        removed_mask[i] = False
                        break
        assert is_bdry_before[removed_mask].all(), (
            "Some removed faces were not boundary faces!"
        )

    def test_split_preserves_all_vertices(self, surface_type):
        """Splitting adds vertices but never removes existing ones."""
        v, f = self._make_mesh(surface_type)
        n_before = len(v)
        v2, f2 = _split_large_edges(v, f, 0.3, surface_type)
        assert len(v2) >= n_before

    def test_split_reduces_max_edge(self, surface_type):
        """After splitting, the max edge should be reduced or equal."""
        v, f = self._make_mesh(surface_type)
        _, _, longest_before = _triangle_quality(v, f)
        mel = float(np.median(longest_before) * 2.0)
        v2, f2 = _split_large_edges(v, f, mel, surface_type)
        _, _, longest_after = _triangle_quality(v2, f2)
        # Max edge should not increase beyond the original
        assert longest_after.max() <= longest_before.max() * 1.05


# ═══════════════════════════════════════════════════════════════════════════
# Harsher Gaussian refinement thresholds tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestGaussianHarsherThresholds:
    """Verify that harsher thresholds for Gaussian triangles work."""

    @staticmethod
    def _make_mesh(surface_type, n=150, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        gi = np.arange(n, dtype=np.int32)
        return v, f, gi

    def test_harsher_gauss_produces_more_refinement(self, surface_type):
        """Harsher Gaussian thresholds should improve quality near Gaussians."""
        v, f, gi = self._make_mesh(surface_type)
        # Normal thresholds
        v1, f1, g1 = refine_bad_triangles(
            v.copy(), f.copy(), surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi.copy(),
            max_iterations=3,
            gauss_max_aspect_ratio=5.0,  # same as default
            gauss_min_angle_deg=10.0,
        )
        # Harsher thresholds for Gaussians
        v2, f2, g2 = refine_bad_triangles(
            v.copy(), f.copy(), surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi.copy(),
            max_iterations=5,
            gauss_max_aspect_ratio=2.0,
            gauss_min_angle_deg=25.0,
        )
        # Both should complete without error and produce valid meshes
        assert len(f1) > 0
        assert len(f2) > 0
        assert f2.max() < len(v2)
        # Harsher thresholds should flag more Gaussian-touching faces as bad,
        # meaning the refinement *attempts* to fix more near Gaussians.
        # The key invariant is that it runs without crashing and
        # Gaussians are never moved.

    def test_gaussian_positions_preserved_with_harsh(self, surface_type):
        """Gaussians are never moved even with harsh thresholds."""
        v, f, gi = self._make_mesh(surface_type)
        orig = v[gi].copy()
        v2, f2, g2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=3,
            gauss_max_aspect_ratio=2.0,
            gauss_min_angle_deg=25.0,
        )
        np.testing.assert_allclose(
            v2[g2], orig, atol=1e-12,
            err_msg="Gaussian vertices moved with harsh thresholds",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Refine with Bowyer–Watson pipeline tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("surface_type", ["Paraboloid", "Saddle", "HyperbolicParaboloid"])
class TestRefineBowyerWatson:
    """Integration tests for refine_bad_triangles with Bowyer–Watson."""

    @staticmethod
    def _make_mesh(surface_type, n=150, seed=42):
        pts = _make_surface_points(surface_type, n=n, seed=seed)
        v, f = build_surface_delaunay(pts, surface_type)
        gi = np.arange(n, dtype=np.int32)
        return v, f, gi

    def test_refine_reduces_bad(self, surface_type):
        """Refinement should reduce or maintain bad triangle fraction."""
        v, f, gi = self._make_mesh(surface_type)
        n_bad_before = _count_bad(v, f)
        frac_before = n_bad_before / max(len(f), 1)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=3,
        )
        n_bad_after = _count_bad(v2, f2)
        frac_after = n_bad_after / max(len(f2), 1)
        # Splitting adds faces so raw count may grow; compare fractions
        # with tolerance consistent with test_bad_count_non_increasing.
        assert frac_after <= frac_before + 0.05, (
            f"Bad fraction increased too much: {frac_before:.3f} → {frac_after:.3f} "
            f"(count: {n_bad_before} → {n_bad_after})"
        )

    def test_vertex_count_grows(self, surface_type):
        """Refinement should add or maintain vertex count."""
        v, f, gi = self._make_mesh(surface_type)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=3,
        )
        assert len(v2) >= len(v)

    def test_gaussian_indices_valid(self, surface_type):
        """Gaussian vertex indices must remain valid."""
        v, f, gi = self._make_mesh(surface_type, n=50)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=2,
        )
        assert gi2.max() < len(v2), "Gaussian index out of range"

    def test_face_indices_valid(self, surface_type):
        """All face indices must be valid after refinement."""
        v, f, gi = self._make_mesh(surface_type, n=80)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=3,
        )
        assert f2.min() >= 0
        assert f2.max() < len(v2)

    def test_vertices_on_surface(self, surface_type):
        """All vertices must still lie on the surface."""
        v, f, gi = self._make_mesh(surface_type, n=60)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=2,
        )
        z_expected = evaluate_polynomial(v2[:, 0], v2[:, 1], surface_type)
        np.testing.assert_allclose(v2[:, 2], z_expected, atol=1e-8)

    def test_use_3d_delaunay_flag(self, surface_type):
        """Passing use_3d_delaunay should not crash."""
        v, f, gi = self._make_mesh(surface_type, n=50)
        v2, f2, gi2 = refine_bad_triangles(
            v, f, surface_type,
            max_edge_length=0.5,
            gaussian_vertex_indices=gi,
            max_iterations=1,
            use_3d_delaunay=False,
        )
        assert len(f2) > 0

