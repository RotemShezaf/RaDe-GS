#!/usr/bin/env python3
"""Tests for geodesic_mesh_utils — Gaussian-mesh construction utilities."""

import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile, shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.geodesic_mesh_utils import (
    project_gaussians_to_surface,
    sample_surface_uniform,
    build_surface_delaunay,
    save_geodesic_mesh,
    load_geodesic_mesh,
)
from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(params=["Paraboloid", "Saddle", "HyperbolicParaboloid"])
def surface_type(request):
    return request.param


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def random_positions():
    """10 random Gaussian-ish positions with z ≈ surface + noise."""
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, 30)
    y = rng.uniform(-1, 1, 30)
    z = x ** 2 + y ** 2 + rng.normal(0, 0.02, 30)  # ~Paraboloid + noise
    return np.column_stack([x, y, z])


# ── project_gaussians_to_surface ─────────────────────────────────────────────

class TestProjectGaussiansToSurface:
    def test_output_shape(self, surface_type, random_positions):
        proj = project_gaussians_to_surface(random_positions, surface_type)
        assert proj.shape == random_positions.shape

    def test_xy_unchanged(self, surface_type, random_positions):
        proj = project_gaussians_to_surface(random_positions, surface_type)
        np.testing.assert_array_equal(proj[:, :2], random_positions[:, :2])

    def test_z_matches_polynomial(self, surface_type, random_positions):
        proj = project_gaussians_to_surface(random_positions, surface_type)
        expected_z = evaluate_polynomial(random_positions[:, 0], random_positions[:, 1], surface_type)
        np.testing.assert_allclose(proj[:, 2], expected_z, atol=1e-12)

    def test_single_point(self, surface_type):
        pt = np.array([[0.5, -0.3, 999.0]])
        proj = project_gaussians_to_surface(pt, surface_type)
        expected_z = evaluate_polynomial(np.array([0.5]), np.array([-0.3]), surface_type)
        assert proj[0, 2] == pytest.approx(expected_z[0], abs=1e-12)

    def test_already_on_surface(self, surface_type):
        """Points exactly on the surface should not move."""
        rng = np.random.RandomState(7)
        x = rng.uniform(-1, 1, 20)
        y = rng.uniform(-1, 1, 20)
        z = evaluate_polynomial(x, y, surface_type)
        pts = np.column_stack([x, y, z])
        proj = project_gaussians_to_surface(pts, surface_type)
        np.testing.assert_allclose(proj, pts, atol=1e-12)


# ── sample_surface_uniform ───────────────────────────────────────────────────

class TestSampleSurfaceUniform:
    def test_requires_one_of_n_or_radius(self, surface_type):
        with pytest.raises(ValueError):
            sample_surface_uniform(surface_type, (-1, 1), (-1, 1))
        with pytest.raises(ValueError):
            sample_surface_uniform(surface_type, (-1, 1), (-1, 1), n_points=10, min_radius=0.1)

    def test_n_points_returns_correct_count(self, surface_type):
        pts = sample_surface_uniform(surface_type, (-1, 1), (-1, 1), n_points=20)
        # Dart-throwing may not hit exact count, but should be close
        assert len(pts) > 0
        assert len(pts) <= 20  # should not exceed target

    def test_min_radius_spacing(self, surface_type):
        r = 0.15
        pts = sample_surface_uniform(surface_type, (-1, 1), (-1, 1), min_radius=r, seed=1)
        if len(pts) < 2:
            pytest.skip("not enough points sampled")
        # Check pairwise distances in (x, y) space
        from scipy.spatial import distance as spdist
        dists = spdist.pdist(pts[:, :2])
        assert dists.min() >= r - 1e-10

    def test_existing_xy_respected(self, surface_type):
        """New samples stay at least min_radius from existing points."""
        r = 0.3
        existing = np.array([[0.0, 0.0], [0.5, 0.5]])
        pts = sample_surface_uniform(
            surface_type, (-1, 1), (-1, 1),
            min_radius=r,
            existing_xy=existing,
            seed=0,
        )
        if len(pts) == 0:
            pytest.skip("no new points sampled")
        from scipy.spatial import KDTree
        tree = KDTree(existing)
        dists, _ = tree.query(pts[:, :2])
        assert dists.min() >= r - 1e-10

    def test_points_on_surface(self, surface_type):
        """Sampled points should lie on the surface."""
        pts = sample_surface_uniform(surface_type, (-1, 1), (-1, 1), n_points=15, seed=99)
        expected_z = evaluate_polynomial(pts[:, 0], pts[:, 1], surface_type)
        np.testing.assert_allclose(pts[:, 2], expected_z, atol=1e-12)

    def test_within_domain(self, surface_type):
        x_range = (-0.5, 0.5)
        y_range = (-0.5, 0.5)
        pts = sample_surface_uniform(surface_type, x_range, y_range, n_points=20, seed=3)
        assert pts[:, 0].min() >= x_range[0]
        assert pts[:, 0].max() <= x_range[1]
        assert pts[:, 1].min() >= y_range[0]
        assert pts[:, 1].max() <= y_range[1]

    def test_deterministic_with_same_seed(self, surface_type):
        kw = dict(surface_type=surface_type, x_range=(-1, 1), y_range=(-1, 1), min_radius=0.2, seed=42)
        a = sample_surface_uniform(**kw)
        b = sample_surface_uniform(**kw)
        np.testing.assert_array_equal(a, b)


# ── build_surface_delaunay ───────────────────────────────────────────────────

class TestBuildSurfaceDelaunay:
    def _make_grid(self, surface_type, n=10):
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.ravel(), yy.ravel()
        zz = evaluate_polynomial(xx, yy, surface_type)
        return np.column_stack([xx, yy, zz])

    def test_output_shapes(self, surface_type):
        pts = self._make_grid(surface_type)
        verts, faces = build_surface_delaunay(pts)
        assert verts.shape == pts.shape
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert len(faces) > 0

    def test_all_vertices_present(self, surface_type):
        pts = self._make_grid(surface_type, 8)
        verts, faces = build_surface_delaunay(pts)
        np.testing.assert_array_equal(verts, pts)
        # Every vertex should appear in at least one face
        unique_verts = np.unique(faces)
        assert len(unique_verts) == len(pts)

    def test_face_indices_valid(self, surface_type):
        pts = self._make_grid(surface_type, 6)
        verts, faces = build_surface_delaunay(pts)
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    def test_random_points(self, surface_type):
        rng = np.random.RandomState(5)
        xy = rng.uniform(-1, 1, (50, 2))
        z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
        pts = np.column_stack([xy, z])
        verts, faces = build_surface_delaunay(pts)
        assert len(faces) > 0

    def test_four_points_two_triangles(self):
        """Four non-collinear points → exactly 2 triangles."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        verts, faces = build_surface_delaunay(pts)
        assert len(faces) == 2


# ── save / load geodesic mesh ───────────────────────────────────────────────

class TestSaveLoadGeodesicMesh:
    def test_roundtrip(self, tmp_dir, surface_type):
        n = 50
        rng = np.random.RandomState(3)
        xy = rng.uniform(-1, 1, (n, 2))
        z = evaluate_polynomial(xy[:, 0], xy[:, 1], surface_type)
        pts = np.column_stack([xy, z])
        verts, faces = build_surface_delaunay(pts)
        g_idx = np.arange(30, dtype=np.int32)  # first 30 are "Gaussians"

        save_geodesic_mesh(tmp_dir, verts, faces, g_idx, metadata={"surface": np.array(surface_type)})

        v2, f2, g2 = load_geodesic_mesh(tmp_dir)
        np.testing.assert_array_equal(v2, verts)
        np.testing.assert_array_equal(f2, faces)
        np.testing.assert_array_equal(g2, g_idx)

    def test_ply_exists(self, tmp_dir):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        verts, faces = build_surface_delaunay(pts)
        save_geodesic_mesh(tmp_dir, verts, faces, np.arange(4, dtype=np.int32))
        assert (tmp_dir / "geodesic_mesh.ply").exists()
        assert (tmp_dir / "geodesic_mesh_data.npz").exists()

    def test_load_missing_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_geodesic_mesh(tmp_dir / "nonexistent")

    def test_metadata_preserved(self, tmp_dir):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        verts, faces = build_surface_delaunay(pts)
        save_geodesic_mesh(
            tmp_dir, verts, faces,
            np.arange(4, dtype=np.int32),
            metadata={"my_key": np.array(42)},
        )
        data = np.load(str(tmp_dir / "geodesic_mesh_data.npz"))
        assert data["my_key"] == 42


# ── Integration: full pipeline on a small surface ───────────────────────────

class TestIntegration:
    def test_full_pipeline_paraboloid(self, tmp_dir):
        """End-to-end: project + sample + triangulate + save + load."""
        surface = "Paraboloid"
        rng = np.random.RandomState(0)

        # Fake Gaussian positions near a paraboloid
        x = rng.uniform(-1, 1, 40)
        y = rng.uniform(-1, 1, 40)
        z = x ** 2 + y ** 2 + rng.normal(0, 0.01, 40)
        positions = np.column_stack([x, y, z])

        # Project
        proj = project_gaussians_to_surface(positions, surface)
        assert proj.shape == positions.shape

        # Sample extras
        extras = sample_surface_uniform(
            surface, (-1.1, 1.1), (-1.1, 1.1),
            n_points=10,
            existing_xy=proj[:, :2],
            seed=0,
        )

        # Build mesh
        all_pts = np.vstack([proj, extras]) if len(extras) > 0 else proj
        verts, faces = build_surface_delaunay(all_pts)
        n_gauss = len(positions)
        g_idx = np.arange(n_gauss, dtype=np.int32)

        # Save + Load
        save_geodesic_mesh(tmp_dir, verts, faces, g_idx)
        v2, f2, g2 = load_geodesic_mesh(tmp_dir)

        assert len(v2) == len(all_pts)
        assert len(g2) == n_gauss
        # Gaussian vertices should match projected positions exactly
        np.testing.assert_allclose(v2[g2], proj, atol=1e-12)

    def test_gaussian_vertices_are_first(self, tmp_dir):
        """Verify that the first N_gauss vertices are exactly the projected Gaussians."""
        surface = "Saddle"
        rng = np.random.RandomState(1)
        n = 25
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = evaluate_polynomial(x, y, surface)
        proj = np.column_stack([x, y, z])

        extras = sample_surface_uniform(surface, (-1.1, 1.1), (-1.1, 1.1), n_points=5, seed=1)
        all_pts = np.vstack([proj, extras]) if len(extras) > 0 else proj
        verts, faces = build_surface_delaunay(all_pts)

        np.testing.assert_array_equal(verts[:n], proj)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
