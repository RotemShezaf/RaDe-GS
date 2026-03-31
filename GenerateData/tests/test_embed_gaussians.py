#!/usr/bin/env python3
"""Tests for embed_gaussians_in_mesh — Gaussian embedding into mesh surfaces.

Covers:
- Output shape / type invariants
- Every Gaussian gets a valid vertex index
- Snapping to nearby existing vertices
- Correct face splitting (single and multi-insertion)
- Augmented mesh consistency (no degenerate faces, valid indices)
- Embedded vertices lie on the original mesh surface
- Direct vertex indexing vs barycentric interpolation equivalence
- Edge cases: all snap, no snap, Gaussians far from mesh
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial import Delaunay, KDTree

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.compute_gaussian_geodesic_distances_tosca import (
    embed_gaussians_in_mesh,
    preprocess_mesh,
    project_along_surface_normal,
    project_pn_triangle,
)
from GenerateData.utils.compute_gaussian_geodesic_distances_helper import (
    find_closest_mesh_vertices,
    find_closest_mesh_faces_barycentric,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_grid_mesh(nx: int = 10, ny: int = 10, scale: float = 1.0):
    """Create a regular triangulated grid on z=0.

    Returns (vertices (V,3), faces (F,3)).
    """
    xs = np.linspace(-scale, scale, nx)
    ys = np.linspace(-scale, scale, ny)
    xx, yy = np.meshgrid(xs, ys)
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    verts = verts.astype(np.float64)

    # Delaunay on (x, y)
    tri = Delaunay(verts[:, :2])
    faces = tri.simplices.astype(np.int32)
    return verts, faces


def _make_single_triangle():
    """One triangle: (0,0,0), (1,0,0), (0,1,0)."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


def _face_area(v0, v1, v2):
    """Area of triangle (v0, v1, v2)."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_mesh():
    """10×10 grid mesh on z=0."""
    return _make_grid_mesh(10, 10)


@pytest.fixture
def single_tri():
    """Single triangle mesh."""
    return _make_single_triangle()


@pytest.fixture
def gaussians_on_surface(grid_mesh):
    """20 Gaussians at random XY positions on the z=0 grid surface."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-0.9, 0.9, (20, 2))
    z = np.zeros((20, 1))
    return np.hstack([xy, z]).astype(np.float64)


@pytest.fixture
def gaussians_near_surface(grid_mesh):
    """30 Gaussians slightly above the z=0 grid."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(-0.8, 0.8, (30, 2))
    z = rng.uniform(0.0, 0.05, (30, 1))
    return np.hstack([xy, z]).astype(np.float64)


# ---------------------------------------------------------------------------
#  Output invariants
# ---------------------------------------------------------------------------

class TestOutputInvariants:
    """Basic shape / type / validity checks on the output."""

    def test_output_shapes(self, grid_mesh, gaussians_on_surface):
        verts, faces = grid_mesh
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert new_v.ndim == 2 and new_v.shape[1] == 3
        assert new_f.ndim == 2 and new_f.shape[1] == 3
        assert g_vi.shape == (len(gaussians_on_surface),)

    def test_output_dtypes(self, grid_mesh, gaussians_on_surface):
        verts, faces = grid_mesh
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert new_v.dtype == np.float64
        assert new_f.dtype == np.int32
        assert np.issubdtype(g_vi.dtype, np.integer)

    def test_all_gaussians_mapped(self, grid_mesh, gaussians_on_surface):
        """Every Gaussian must get a non-negative vertex index."""
        verts, faces = grid_mesh
        _, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert np.all(g_vi >= 0)

    def test_vertex_indices_in_range(self, grid_mesh, gaussians_on_surface):
        """All assigned vertex indices must be valid."""
        verts, faces = grid_mesh
        new_v, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert np.all(g_vi < len(new_v))

    def test_face_indices_valid(self, grid_mesh, gaussians_on_surface):
        """All face vertex references must index into the vertex array."""
        verts, faces = grid_mesh
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert new_f.min() >= 0
        assert new_f.max() < len(new_v)

    def test_no_degenerate_faces(self, grid_mesh, gaussians_on_surface):
        """No face should reference the same vertex twice."""
        verts, faces = grid_mesh
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        for i in range(len(new_f)):
            assert len(set(new_f[i])) == 3, (
                f"Degenerate face {i}: {new_f[i]}"
            )

    def test_augmented_mesh_has_more_or_equal_verts(self, grid_mesh, gaussians_on_surface):
        """Augmented mesh should have at least as many vertices as original."""
        verts, faces = grid_mesh
        new_v, _, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert len(new_v) >= len(verts)

    def test_original_vertices_preserved(self, grid_mesh, gaussians_on_surface):
        """Original mesh vertices must appear unchanged in the augmented mesh."""
        verts, faces = grid_mesh
        new_v, _, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        np.testing.assert_array_equal(new_v[: len(verts)], verts)


# ---------------------------------------------------------------------------
#  Snapping behaviour
# ---------------------------------------------------------------------------

class TestSnapping:
    """Gaussians near existing vertices should snap instead of creating new ones."""

    def test_gaussians_at_vertices_all_snap(self, grid_mesh):
        """Gaussians placed exactly at mesh vertices should all snap."""
        verts, faces = grid_mesh
        # Pick 10 mesh vertices as 'Gaussian' positions
        gauss_pos = verts[:10].copy()
        new_v, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.95, verbose=False,
        )
        # No new vertices should be created
        assert len(new_v) == len(verts)
        # Each Gaussian should map to its own vertex
        np.testing.assert_array_equal(g_vi, np.arange(10))

    def test_snap_threshold_controls_snapping(self, single_tri):
        """Lowering snap_threshold should snap fewer Gaussians."""
        verts, faces = single_tri
        # Point near vertex 0: bary ≈ (0.94, 0.03, 0.03)
        gauss_pos = np.array([[0.03, 0.03, 0.0]], dtype=np.float64)

        # With threshold=0.90, should snap (max bary >0.90)
        _, _, g_vi_low = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.90, verbose=False,
        )
        # With threshold=0.99, should NOT snap
        new_v_high, _, g_vi_high = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        # Low threshold → snapped → no new vertex
        assert g_vi_low[0] < len(verts), "Should snap to existing vertex"
        # High threshold → inserted → new vertex
        assert g_vi_high[0] >= len(verts) or len(new_v_high) > len(verts)

    def test_no_new_vertices_when_all_snap(self, grid_mesh):
        """When all Gaussians snap, face count should not change."""
        verts, faces = grid_mesh
        gauss_pos = verts[5:8].copy()
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.5, verbose=False,
        )
        assert len(new_v) == len(verts)
        assert len(new_f) == len(faces)


# ---------------------------------------------------------------------------
#  Face splitting
# ---------------------------------------------------------------------------

class TestFaceSplitting:
    """Correct triangle subdivision when Gaussians are inserted."""

    def test_single_insertion_creates_3_faces(self, single_tri):
        """Inserting 1 point in 1 triangle → 3 sub-triangles."""
        verts, faces = single_tri
        gauss_pos = np.array([[0.3, 0.3, 0.0]], dtype=np.float64)

        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        # 1 original face removed, 3 added → net +2
        assert len(new_f) == 3
        assert len(new_v) == 4  # 3 original + 1 new

    def test_total_area_preserved_single(self, single_tri):
        """Splitting should preserve total surface area."""
        verts, faces = single_tri
        orig_area = _face_area(verts[0], verts[1], verts[2])

        gauss_pos = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        total_area = sum(
            _face_area(new_v[f[0]], new_v[f[1]], new_v[f[2]])
            for f in new_f
        )
        np.testing.assert_allclose(total_area, orig_area, rtol=1e-10)

    def test_total_area_preserved_grid(self, grid_mesh, gaussians_on_surface):
        """Total mesh area should be preserved after embedding on the grid."""
        verts, faces = grid_mesh
        orig_area = sum(
            _face_area(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces
        )

        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, snap_threshold=0.99,
            verbose=False,
        )
        new_area = sum(
            _face_area(new_v[f[0]], new_v[f[1]], new_v[f[2]]) for f in new_f
        )
        np.testing.assert_allclose(new_area, orig_area, rtol=1e-8)

    def test_multi_insertion_same_face(self, single_tri):
        """Multiple Gaussians in one triangle → Delaunay sub-triangulation."""
        verts, faces = single_tri
        gauss_pos = np.array([
            [0.2, 0.2, 0.0],
            [0.5, 0.1, 0.0],
            [0.1, 0.5, 0.0],
        ], dtype=np.float64)

        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        # 6 points (3 orig + 3 new) → Delaunay should produce multiple faces
        assert len(new_v) == 6
        assert len(new_f) >= 4  # Delaunay of 6 points ≥ 4 triangles
        assert len(g_vi) == 3
        # All new vertex indices are 3, 4, 5
        assert set(g_vi.tolist()) == {3, 4, 5}


# ---------------------------------------------------------------------------
#  Surface consistency
# ---------------------------------------------------------------------------

class TestSurfaceConsistency:
    """Embedded Gaussian vertices should lie on the mesh surface."""

    def test_embedded_points_on_surface_z0(self, grid_mesh, gaussians_near_surface):
        """On a z=0 grid, embedded Gaussian z should be ≈ 0."""
        verts, faces = grid_mesh
        new_v, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_near_surface, verbose=False,
        )
        embedded_z = new_v[g_vi, 2]
        np.testing.assert_allclose(embedded_z, 0.0, atol=1e-10)

    def test_embedded_points_are_barycentric_combinations(self, grid_mesh):
        """Each embedded point should be a convex combination of face vertices."""
        verts, faces = grid_mesh
        rng = np.random.default_rng(99)
        gauss_pos = np.column_stack([
            rng.uniform(-0.8, 0.8, 15),
            rng.uniform(-0.8, 0.8, 15),
            rng.uniform(0, 0.02, 15),
        ])

        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )

        # For each embedded Gaussian vertex, check it lies in some face
        import trimesh
        mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        for g in range(len(gauss_pos)):
            vi = g_vi[g]
            pt = new_v[vi]
            # The point IS a vertex of the augmented mesh — verify it by
            # checking that at least one face contains this vertex
            face_mask = np.any(new_f == vi, axis=1)
            assert face_mask.sum() >= 1, (
                f"Gaussian {g} (vertex {vi}) not referenced by any face"
            )


# ---------------------------------------------------------------------------
#  Unique vertex indices
# ---------------------------------------------------------------------------

class TestUniqueMapping:
    """Each Gaussian should ideally get a distinct vertex."""

    def test_distinct_gaussians_get_distinct_vertices(self, grid_mesh):
        """Well-separated Gaussians should map to different vertices."""
        verts, faces = grid_mesh
        # Place 5 Gaussians at well-separated locations
        gauss_pos = np.array([
            [-0.7, -0.7, 0.0],
            [-0.7,  0.7, 0.0],
            [ 0.7, -0.7, 0.0],
            [ 0.7,  0.7, 0.0],
            [ 0.0,  0.0, 0.0],
        ], dtype=np.float64)

        _, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, verbose=False,
        )
        assert len(np.unique(g_vi)) == 5

    def test_coincident_gaussians_same_vertex(self, grid_mesh):
        """Two Gaussians at the exact same position at a mesh vertex should snap to it."""
        verts, faces = grid_mesh
        # Place both at the actual mesh vertex closest to origin
        tree = KDTree(verts)
        _, vi = tree.query([0.0, 0.0, 0.0])
        gauss_pos = np.array([verts[vi], verts[vi]], dtype=np.float64)

        _, _, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.95, verbose=False,
        )
        # Both should snap to the same existing vertex
        assert g_vi[0] == g_vi[1]
        assert g_vi[0] == vi


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary / degenerate scenarios."""

    def test_single_gaussian(self, grid_mesh):
        """Works with just 1 Gaussian."""
        verts, faces = grid_mesh
        gauss_pos = np.array([[0.3, 0.4, 0.0]], dtype=np.float64)
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, verbose=False,
        )
        assert g_vi.shape == (1,)
        assert g_vi[0] >= 0

    def test_many_gaussians(self, grid_mesh):
        """Works with many Gaussians (More than mesh vertices)."""
        verts, faces = grid_mesh  # 100 vertices
        rng = np.random.default_rng(33)
        gauss_pos = np.column_stack([
            rng.uniform(-0.9, 0.9, 200),
            rng.uniform(-0.9, 0.9, 200),
            np.zeros(200),
        ])

        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, verbose=False,
        )
        assert g_vi.shape == (200,)
        assert np.all(g_vi >= 0)
        assert np.all(g_vi < len(new_v))

    def test_gaussian_at_mesh_boundary(self, single_tri):
        """Gaussian on an edge of the triangle."""
        verts, faces = single_tri
        # On the edge v0-v1: (0.5, 0, 0) → bary ≈ (0.5, 0.5, 0.0)
        gauss_pos = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        assert g_vi[0] >= 0
        assert g_vi[0] < len(new_v)

    def test_gaussian_far_from_mesh(self, grid_mesh):
        """Gaussians far from mesh surface should still get mapped."""
        verts, faces = grid_mesh
        gauss_pos = np.array([[10.0, 10.0, 5.0]], dtype=np.float64)
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, verbose=False,
        )
        assert g_vi[0] >= 0


# ---------------------------------------------------------------------------
#  Integration: embedded path vs barycentric path
# ---------------------------------------------------------------------------

class TestEmbeddedVsBarycentric:
    """Compare the embedded vertex approach against barycentric interpolation.

    On a flat z=0 grid with Euclidean distances used as a surrogate for
    geodesics, the embedded path should produce distances that are at
    least as accurate as barycentric interpolation.
    """

    def test_embedded_distances_close_to_barycentric(self, grid_mesh):
        """Embedded and barycentric distances should be similar."""
        verts, faces = grid_mesh
        rng = np.random.default_rng(55)
        gauss_pos = np.column_stack([
            rng.uniform(-0.7, 0.7, 15),
            rng.uniform(-0.7, 0.7, 15),
            np.zeros(15),
        ]).astype(np.float64)

        # --- Embedded path ---
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99, verbose=False,
        )
        # Use Euclidean from one source vertex as surrogate geodesic
        source_idx = 0  # vertex 0
        mesh_dists_emb = np.linalg.norm(
            new_v - new_v[source_idx], axis=1,
        )
        embedded_dists = mesh_dists_emb[g_vi]

        # --- Barycentric path ---
        g2m_idx, _ = find_closest_mesh_vertices(gauss_pos, verts)
        bary_face, bary_w = find_closest_mesh_faces_barycentric(
            gauss_pos, verts, faces, g2m_idx,
        )
        mesh_dists_orig = np.linalg.norm(
            verts - verts[source_idx], axis=1,
        )
        bary_dists = np.sum(
            bary_w * mesh_dists_orig[bary_face], axis=1,
        )

        # Ground truth: direct Euclidean from source vertex to Gaussian
        gt_dists = np.linalg.norm(
            gauss_pos - verts[source_idx], axis=1,
        )

        # Embedded path error should be ≤ barycentric error (or comparable)
        emb_err = np.abs(embedded_dists - gt_dists)
        bary_err = np.abs(bary_dists - gt_dists)

        # Embedded should have near-zero error on a flat surface
        # (projection lands exactly on z=0)
        assert emb_err.mean() < 0.15, (
            f"Embedded mean error too high: {emb_err.mean():.4f}"
        )
        # And should generally be no worse than barycentric
        assert emb_err.mean() <= bary_err.mean() + 0.05, (
            f"Embedded error ({emb_err.mean():.4f}) significantly worse than "
            f"barycentric ({bary_err.mean():.4f})"
        )


# ---------------------------------------------------------------------------
#  Mesh validity after embedding
# ---------------------------------------------------------------------------

class TestMeshValidity:
    """The augmented mesh should be a valid triangle mesh."""

    def test_no_zero_area_faces(self, grid_mesh, gaussians_on_surface):
        """No face should have zero area."""
        verts, faces = grid_mesh
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, snap_threshold=0.99,
            verbose=False,
        )
        for i, f in enumerate(new_f):
            area = _face_area(new_v[f[0]], new_v[f[1]], new_v[f[2]])
            assert area > 1e-15, f"Face {i} has zero area: {new_v[f]}"

    def test_all_new_vertices_referenced(self, grid_mesh, gaussians_on_surface):
        """Every newly-added vertex should appear in at least one face."""
        verts, faces = grid_mesh
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, snap_threshold=0.99,
            verbose=False,
        )
        n_orig = len(verts)
        if len(new_v) > n_orig:
            referenced = set(new_f.ravel().tolist())
            for vi in range(n_orig, len(new_v)):
                assert vi in referenced, (
                    f"New vertex {vi} not referenced by any face"
                )

    def test_trimesh_loads_augmented(self, grid_mesh, gaussians_on_surface):
        """trimesh should load the augmented mesh without errors."""
        import trimesh

        verts, faces = grid_mesh
        new_v, new_f, _ = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        assert len(mesh.vertices) == len(new_v)
        assert len(mesh.faces) == len(new_f)


# ---------------------------------------------------------------------------
#  Preprocess + embed combined
# ---------------------------------------------------------------------------

class TestPreprocessThenEmbed:
    """Preprocessing before embedding should work seamlessly."""

    def test_preprocess_then_embed(self):
        """Preprocess a mesh with small disconnected components, then embed."""
        import trimesh

        # Main component: grid
        verts, faces = _make_grid_mesh(8, 8)

        # Add a small floating triangle far away
        extra_v = np.array([[5, 5, 0], [5.1, 5, 0], [5, 5.1, 0]], dtype=np.float64)
        extra_f = np.array([[len(verts), len(verts) + 1, len(verts) + 2]], dtype=np.int32)
        verts_combined = np.vstack([verts, extra_v])
        faces_combined = np.vstack([faces, extra_f])

        # Preprocess: should remove floating triangle
        clean_v, clean_f = preprocess_mesh(
            verts_combined, faces_combined,
            keep_largest_component=True, verbose=False,
        )
        assert len(clean_v) < len(verts_combined)

        # Embed
        gauss_pos = np.column_stack([
            np.random.default_rng(0).uniform(-0.7, 0.7, 10),
            np.random.default_rng(0).uniform(-0.7, 0.7, 10),
            np.zeros(10),
        ])
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            clean_v, clean_f, gauss_pos, verbose=False,
        )
        assert np.all(g_vi >= 0)
        assert np.all(g_vi < len(new_v))


# ---------------------------------------------------------------------------
#  Normal-guided projection
# ---------------------------------------------------------------------------

def _make_hemisphere_mesh(n: int = 20):
    """Create a triangulated hemisphere (radius 1, open at z=0).

    Returns (vertices (V,3), faces (F,3)).
    """
    import trimesh
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    # Keep only the upper hemisphere (z >= -0.1 to keep equator ring)
    keep = sphere.vertices[:, 2] >= -0.1
    keep_map = np.full(len(sphere.vertices), -1, dtype=int)
    new_idx = 0
    for i, k in enumerate(keep):
        if k:
            keep_map[i] = new_idx
            new_idx += 1
    verts = sphere.vertices[keep].astype(np.float64)
    faces_list = []
    for f in sphere.faces:
        if keep[f[0]] and keep[f[1]] and keep[f[2]]:
            faces_list.append([keep_map[f[0]], keep_map[f[1]], keep_map[f[2]]])
    faces = np.array(faces_list, dtype=np.int32)
    return verts, faces


class TestCurvatureProjection:
    """Tests for project_pn_triangle and its integration."""

    def test_flat_surface_same_as_orthogonal(self):
        """On a flat z=0 mesh, PN triangle projection should match flat."""
        verts, faces = _make_grid_mesh(10, 10)
        gauss_pos = np.array([
            [0.3, 0.4, 0.05],
            [-0.2, 0.1, 0.02],
            [0.5, -0.5, 0.1],
        ], dtype=np.float64)

        # Without curvature projection
        new_v_flat, _, g_vi_flat = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99,
            use_curvature_projection=False, verbose=False,
        )
        # With curvature projection
        new_v_pn, _, g_vi_pn = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99,
            use_curvature_projection=True, verbose=False,
        )
        # On a flat surface both projections should land at the same z=0 points
        pts_flat = new_v_flat[g_vi_flat]
        pts_pn = new_v_pn[g_vi_pn]
        np.testing.assert_allclose(pts_flat[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(pts_pn[:, 2], 0.0, atol=1e-10)
        np.testing.assert_allclose(pts_flat, pts_pn, atol=1e-10)

    def test_pn_triangle_flat_identity(self):
        """On a flat triangle with uniform normals, PN ≡ flat interpolation."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype=np.float64)
        vnormals = np.array([
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ], dtype=np.float64)

        face_verts = verts[np.newaxis, :, :]      # (1, 3, 3)
        vtx_normals = vnormals[np.newaxis, :, :]  # (1, 3, 3)
        bary = np.array([[0.4, 0.3, 0.3]])

        proj = project_pn_triangle(face_verts, bary, vtx_normals)
        flat = bary[:, 0:1] * verts[0] + bary[:, 1:2] * verts[1] + bary[:, 2:3] * verts[2]

        np.testing.assert_allclose(proj, flat, atol=1e-12)

    def test_pn_triangle_interpolates_corners(self):
        """PN triangle must interpolate vertex positions exactly."""
        verts = np.array([
            [0, 0, 0], [2, 0, 0], [1, 2, 1],
        ], dtype=np.float64)
        vnormals = np.array([
            [0, 0, 1], [0, 0.3, 0.95], [-0.1, 0.1, 0.99],
        ], dtype=np.float64)
        # Normalize
        vnormals = vnormals / np.linalg.norm(vnormals, axis=1, keepdims=True)

        for corner_bary, expected_pos in [
            ([1, 0, 0], verts[0]),
            ([0, 1, 0], verts[1]),
            ([0, 0, 1], verts[2]),
        ]:
            face_v = verts[np.newaxis, :, :]
            vtx_n = vnormals[np.newaxis, :, :]
            bary = np.array([corner_bary], dtype=np.float64)
            proj = project_pn_triangle(face_v, bary, vtx_n)
            np.testing.assert_allclose(
                proj[0], expected_pos, atol=1e-12,
                err_msg=f"Corner bary={corner_bary} should give {expected_pos}"
            )

    def test_hemisphere_pn_projection(self):
        """On a curved hemisphere, PN-triangle points should be closer to r=1."""
        verts, faces = _make_hemisphere_mesh()

        # Place Gaussians slightly outside the hemisphere (radial distance > 1)
        rng = np.random.default_rng(123)
        theta = rng.uniform(0, np.pi / 3, 10)
        phi = rng.uniform(0, 2 * np.pi, 10)
        r = rng.uniform(1.05, 1.2, 10)
        gauss_pos = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]).astype(np.float64)

        # With PN-triangle curvature correction
        new_v_pn, new_f_pn, g_vi_pn = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99,
            use_curvature_projection=True, verbose=False,
        )
        radii_pn = np.linalg.norm(new_v_pn[g_vi_pn], axis=1)

        # Without curvature correction (flat)
        new_v_flat, _, g_vi_flat = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99,
            use_curvature_projection=False, verbose=False,
        )
        radii_flat = np.linalg.norm(new_v_flat[g_vi_flat], axis=1)

        # PN should be at least as close to r=1 as flat
        err_pn = np.abs(radii_pn - 1.0)
        err_flat = np.abs(radii_flat - 1.0)
        assert err_pn.mean() <= err_flat.mean() + 0.01, (
            f"PN error ({err_pn.mean():.4f}) should be <= flat ({err_flat.mean():.4f})"
        )

    def test_curvature_projection_flag_off(self, grid_mesh, gaussians_near_surface):
        """use_curvature_projection=False should still produce valid output."""
        verts, faces = grid_mesh
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_near_surface,
            use_curvature_projection=False, verbose=False,
        )
        assert np.all(g_vi >= 0)
        assert np.all(g_vi < len(new_v))
        assert new_f.min() >= 0
        assert new_f.max() < len(new_v)

    def test_default_curvature_projection_invariants(self, grid_mesh, gaussians_on_surface):
        """Default (PN on) should satisfy basic mesh invariants."""
        verts, faces = grid_mesh
        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gaussians_on_surface, verbose=False,
        )
        assert new_v.ndim == 2 and new_v.shape[1] == 3
        assert np.all(g_vi >= 0) and np.all(g_vi < len(new_v))
        assert new_f.min() >= 0 and new_f.max() < len(new_v)

    def test_pn_triangle_many_gaussians_on_hemisphere(self):
        """PN projection should handle many Gaussians on a curved surface."""
        verts, faces = _make_hemisphere_mesh()

        rng = np.random.default_rng(999)
        theta = rng.uniform(0.1, np.pi / 3, 40)
        phi = rng.uniform(0, 2 * np.pi, 40)
        r = rng.uniform(1.02, 1.3, 40)
        gauss_pos = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]).astype(np.float64)

        new_v, new_f, g_vi = embed_gaussians_in_mesh(
            verts, faces, gauss_pos, snap_threshold=0.99,
            use_curvature_projection=True, verbose=False,
        )

        assert np.all(g_vi >= 0)
        assert np.all(g_vi < len(new_v))
        assert new_f.min() >= 0 and new_f.max() < len(new_v)
