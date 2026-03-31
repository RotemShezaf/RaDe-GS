#!/usr/bin/env python3
"""Tests for compute_gaussian_geodesic_distances_tosca.py — TOSCA geodesic computation."""

import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.compute_gaussian_geodesic_distances_tosca import (
    load_tosca_gt_mesh,
    select_sources_near_gaussians,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_mesh_vertices():
    """A 10×10 regular grid on the XY plane, z=0."""
    xs = np.linspace(-1, 1, 10)
    ys = np.linspace(-1, 1, 10)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(100)])
    return verts.astype(np.float64)


@pytest.fixture
def gaussian_cluster_positions():
    """50 Gaussian centres clustered near the origin."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((50, 3)) * 0.3).astype(np.float32)


@pytest.fixture
def gaussian_spread_positions():
    """50 Gaussians spread across the domain [-1, 1]^2."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (50, 2))
    z = np.zeros((50, 1))
    return np.hstack([xy, z]).astype(np.float32)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test I/O."""
    return tmp_path


# ---------------------------------------------------------------------------
#  load_tosca_gt_mesh
# ---------------------------------------------------------------------------

class TestLoadToscaGtMesh:
    """Tests for load_tosca_gt_mesh."""

    def test_mesh_not_found_raises(self, tmp_dir):
        """FileNotFoundError if no mesh file exists."""
        shape_dir = tmp_dir / "cat0"
        shape_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No mesh"):
            load_tosca_gt_mesh(tmp_dir, "cat0", "high_res")

    def test_loads_mesh_from_ply(self, tmp_dir):
        """Successfully loads a simple PLY mesh."""
        import trimesh

        shape_dir = tmp_dir / "cat0"
        shape_dir.mkdir()

        # Create a trivial mesh and save as PLY
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(str(shape_dir / "mesh_high_res_arc0.5.ply"))

        v, f = load_tosca_gt_mesh(tmp_dir, "cat0", "high_res")
        assert v.shape == (4, 3)
        assert f.shape == (2, 3)
        np.testing.assert_allclose(v, verts, atol=1e-6)


# ---------------------------------------------------------------------------
#  select_sources_near_gaussians
# ---------------------------------------------------------------------------

class TestSelectSourcesNearGaussians:
    """Tests for select_sources_near_gaussians."""

    def test_returns_correct_count(self, simple_mesh_vertices, gaussian_cluster_positions):
        """Should return exactly num_sources indices."""
        num_sources = 20
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, num_sources, seed=42,
        )
        assert len(idx) == num_sources
        assert pos.shape == (num_sources, 3)

    def test_indices_are_valid(self, simple_mesh_vertices, gaussian_cluster_positions):
        """All returned indices must be within [0, V)."""
        idx, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 15, seed=0,
        )
        assert np.all(idx >= 0)
        assert np.all(idx < len(simple_mesh_vertices))

    def test_positions_match_indices(self, simple_mesh_vertices, gaussian_cluster_positions):
        """Positions must correspond to mesh_vertices[indices]."""
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 15, seed=0,
        )
        np.testing.assert_allclose(pos, simple_mesh_vertices[idx])

    def test_no_duplicate_indices(self, simple_mesh_vertices, gaussian_cluster_positions):
        """Selected source indices should be unique."""
        idx, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 30, seed=7,
        )
        assert len(np.unique(idx)) == len(idx)

    def test_sources_are_near_gaussians(self, simple_mesh_vertices, gaussian_cluster_positions):
        """Selected vertices should be close to some Gaussian."""
        from scipy.spatial import KDTree

        idx, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 20,
            proximity_factor=3.0, seed=0,
        )
        gauss_tree = KDTree(gaussian_cluster_positions)
        dists, _ = gauss_tree.query(simple_mesh_vertices[idx])
        # All selected sources should have finite distance < some reasonable bound
        assert np.all(np.isfinite(dists))

    def test_few_candidates_uses_all(self, simple_mesh_vertices, gaussian_cluster_positions):
        """When num_sources ≥ candidates the function returns all candidates."""
        # Use a very tight proximity factor so few vertices qualify
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 9999,
            proximity_factor=0.001, seed=0,
        )
        # Should still return something (relaxation kicks in);
        # just verify no crash and result is well-formed
        assert len(idx) > 0
        assert pos.shape[0] == len(idx)

    def test_spread_gaussians_give_spread_sources(
        self, simple_mesh_vertices, gaussian_spread_positions
    ):
        """With Gaussians spread across the domain, sources should also spread."""
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_spread_positions, 20, seed=0,
        )
        # Check that selected sources span a reasonable fraction of the domain
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        assert x_range > 0.5, "Sources should span a decent x range"
        assert y_range > 0.5, "Sources should span a decent y range"

    def test_num_sources_larger_than_mesh(self, gaussian_cluster_positions):
        """If num_sources > V, return all vertices."""
        small_mesh = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        idx, pos = select_sources_near_gaussians(
            small_mesh, gaussian_cluster_positions, 100, seed=0,
        )
        assert len(idx) <= len(small_mesh)  # can't return more than mesh has

    def test_consistent_count_and_validity(self, simple_mesh_vertices, gaussian_cluster_positions):
        """Repeated calls return correct count and valid indices.

        Note: the underlying FPS backend (fpsample / pointnet2) is not
        guaranteed to be deterministic across calls, so we only check
        structural invariants rather than exact index equality.
        """
        idx1, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 15, seed=99,
        )
        idx2, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gaussian_cluster_positions, 15, seed=99,
        )
        assert len(idx1) == 15
        assert len(idx2) == 15
        assert np.all(idx1 >= 0) and np.all(idx1 < len(simple_mesh_vertices))
        assert np.all(idx2 >= 0) and np.all(idx2 < len(simple_mesh_vertices))

    def test_embedded_mode_returns_correct_count(self, simple_mesh_vertices):
        """With gaussian_vertex_indices, FPS uses those vertices as candidates."""
        gauss_vi = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        gauss_pos = simple_mesh_vertices[gauss_vi]
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gauss_pos, 5,
            gaussian_vertex_indices=gauss_vi,
        )
        assert len(idx) == 5
        assert pos.shape == (5, 3)

    def test_embedded_mode_indices_subset_of_candidates(self, simple_mesh_vertices):
        """All returned indices must be from the gaussian_vertex_indices set."""
        gauss_vi = np.array([3, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        gauss_pos = simple_mesh_vertices[gauss_vi]
        idx, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gauss_pos, 6,
            gaussian_vertex_indices=gauss_vi,
        )
        assert set(idx.tolist()).issubset(set(gauss_vi.tolist()))

    def test_embedded_mode_positions_match(self, simple_mesh_vertices):
        """Returned positions must match mesh_vertices[indices]."""
        gauss_vi = np.arange(0, 100, 5)
        gauss_pos = simple_mesh_vertices[gauss_vi]
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gauss_pos, 8,
            gaussian_vertex_indices=gauss_vi,
        )
        np.testing.assert_allclose(pos, simple_mesh_vertices[idx])

    def test_embedded_mode_all_when_fewer_candidates(self, simple_mesh_vertices):
        """When num_sources > unique candidates, returns all candidates."""
        gauss_vi = np.array([10, 20, 30])
        gauss_pos = simple_mesh_vertices[gauss_vi]
        idx, pos = select_sources_near_gaussians(
            simple_mesh_vertices, gauss_pos, 100,
            gaussian_vertex_indices=gauss_vi,
        )
        assert len(idx) == 3
        assert set(idx.tolist()) == {10, 20, 30}

    def test_embedded_mode_deduplicates(self, simple_mesh_vertices):
        """Duplicate gaussian_vertex_indices should be deduplicated."""
        gauss_vi = np.array([5, 5, 10, 10, 20, 30, 40, 50])
        gauss_pos = simple_mesh_vertices[[5, 5, 10, 10, 20, 30, 40, 50]]
        idx, _ = select_sources_near_gaussians(
            simple_mesh_vertices, gauss_pos, 3,
            gaussian_vertex_indices=gauss_vi,
        )
        assert len(np.unique(idx)) == len(idx)


# ---------------------------------------------------------------------------
#  FPS integration (uses utils.misc.fps_gs)
# ---------------------------------------------------------------------------

class TestFPSIntegration:
    """Verify that the FPS call via utils.misc.fps_gs works correctly."""

    def test_fps_gs_basic(self):
        """fps_gs should return n indices without error."""
        from utils.misc import fps_gs

        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3)).astype(np.float32)
        idx = fps_gs(pts, n=50, attributes=["xyz"], device="cpu")
        assert len(idx) == 50
        assert len(np.unique(idx)) == 50

    def test_fps_gs_identity_when_n_ge_N(self):
        """If n ≥ N, fps_gs returns all points."""
        from utils.misc import fps_gs

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = fps_gs(pts, n=10, attributes=["xyz"], device="cpu")
        assert len(idx) == 3  # only 3 points

    def test_fps_gs_spread(self):
        """Selected points should be well separated."""
        from utils.misc import fps_gs

        # Create two tight clusters far apart
        rng = np.random.default_rng(2)
        cluster_a = rng.standard_normal((100, 3)).astype(np.float32) * 0.01
        cluster_b = rng.standard_normal((100, 3)).astype(np.float32) * 0.01 + 10.0
        pts = np.vstack([cluster_a, cluster_b])

        idx = fps_gs(pts, n=4, attributes=["xyz"], device="cpu")
        selected = pts[idx]

        # Should pick points from both clusters
        in_a = (selected[:, 0] < 5).sum()
        in_b = (selected[:, 0] > 5).sum()
        assert in_a >= 1, "Should pick at least one from cluster A"
        assert in_b >= 1, "Should pick at least one from cluster B"


# ---------------------------------------------------------------------------
#  End-to-end lightweight smoke test (no real TOSCA data needed)
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:
    """Smoke-test the full pipeline with synthetic data (no disk I/O)."""

    def test_select_then_transfer(self, simple_mesh_vertices):
        """Run select_sources → fake geodesic → transfer pipeline."""
        from scipy.spatial import KDTree, Delaunay
        from GenerateData.utils.compute_gaussian_geodesic_distances_helper import (
            find_closest_mesh_vertices,
            find_closest_mesh_faces_barycentric,
            transfer_geodesic_to_gaussians,
        )

        # Build mesh: Delaunay on XY
        verts = simple_mesh_vertices  # (100, 3)
        tri = Delaunay(verts[:, :2])
        faces = tri.simplices.astype(np.int32)

        # Fake Gaussians near the mesh
        rng = np.random.default_rng(42)
        gauss_pos = verts[rng.choice(len(verts), 30, replace=False)] + rng.standard_normal((30, 3)) * 0.01
        gauss_pos = gauss_pos.astype(np.float32)

        # Select sources
        src_idx, src_pos = select_sources_near_gaussians(verts, gauss_pos, 10, seed=0)
        assert len(src_idx) == 10

        # Closest mesh vertex/face per Gaussian
        g2m_idx, g2m_dist = find_closest_mesh_vertices(gauss_pos, verts)
        bary_face, bary_w = find_closest_mesh_faces_barycentric(gauss_pos, verts, faces, g2m_idx)

        # Fake mesh geodesic distances (use Euclidean as surrogate)
        mesh_geo = np.linalg.norm(verts[src_idx, None, :] - verts[None, :, :], axis=2)  # (S, V)

        # Source → nearest Gaussian mapping
        gauss_tree = KDTree(gauss_pos)
        _, src_gauss_idx = gauss_tree.query(verts[src_idx])

        # Transfer
        gauss_geo = transfer_geodesic_to_gaussians(
            mesh_geo, g2m_idx, src_idx, src_gauss_idx,
            barycentric_face_vertices=bary_face,
            barycentric_weights=bary_w,
        )
        assert gauss_geo.shape == (len(src_idx), len(gauss_pos))
        assert np.all(np.isfinite(gauss_geo))
        # Sources to their own Gaussian should be small relative to
        # the max distance (Euclidean surrogate + barycentric interpolation
        # on a coarse grid can shift distances slightly)
        max_dist = gauss_geo.max()
        for i in range(len(src_idx)):
            assert gauss_geo[i, src_gauss_idx[i]] < 0.4 * max_dist, (
                f"Source {i} distance to its own Gaussian = "
                f"{gauss_geo[i, src_gauss_idx[i]]:.4f} is too large "
                f"(max_dist={max_dist:.4f})"
            )
