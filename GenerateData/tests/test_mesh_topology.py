#!/usr/bin/env python3
"""Tests for mesh topology: non-manifold edges, duplicate faces, and the fix."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.geodesic_mesh_utils import (
    _detect_non_manifold_edges,
    _detect_duplicate_faces,
    _count_mesh_holes,
    _fill_small_holes,
    _fill_holes_with_center_vertex,
    fix_non_manifold_mesh,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_clean_fan():
    """A simple 4-triangle fan around a center vertex — fully manifold."""
    # Vertices: center + 4 corners of a square
    verts = np.array([
        [0, 0, 0],   # 0: center
        [1, 0, 0],   # 1
        [0, 1, 0],   # 2
        [-1, 0, 0],  # 3
        [0, -1, 0],  # 4
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
    ], dtype=np.int32)
    return verts, faces


# ── Tests: _detect_duplicate_faces ───────────────────────────────────────────

class TestDetectDuplicateFaces:
    def test_no_duplicates(self):
        _, faces = _make_clean_fan()
        n_dup, dup_idx = _detect_duplicate_faces(faces)
        assert n_dup == 0
        assert len(dup_idx) == 0

    def test_exact_duplicate(self):
        _, faces = _make_clean_fan()
        # Append exact copy of first face
        faces = np.vstack([faces, faces[0:1]])
        n_dup, dup_idx = _detect_duplicate_faces(faces)
        assert n_dup == 1
        assert 4 in dup_idx  # the appended row

    def test_reversed_winding_duplicate(self):
        _, faces = _make_clean_fan()
        # Append reversed first face: [0,1,2] → [2,1,0]
        rev = faces[0:1, ::-1].copy()
        faces = np.vstack([faces, rev])
        n_dup, dup_idx = _detect_duplicate_faces(faces)
        assert n_dup == 1

    def test_rotated_duplicate(self):
        _, faces = _make_clean_fan()
        # Append rotated first face: [0,1,2] → [1,2,0]
        rot = np.array([[1, 2, 0]], dtype=np.int32)
        faces = np.vstack([faces, rot])
        n_dup, dup_idx = _detect_duplicate_faces(faces)
        assert n_dup == 1

    def test_multiple_duplicates(self):
        _, faces = _make_clean_fan()
        # Duplicate face 0 twice and face 2 once
        extra = np.vstack([faces[0:1], faces[0:1], faces[2:3]])
        faces = np.vstack([faces, extra])
        n_dup, dup_idx = _detect_duplicate_faces(faces)
        assert n_dup == 3


# ── Tests: _detect_non_manifold_edges ────────────────────────────────────────

class TestDetectNonManifoldEdges:
    def test_clean_mesh_has_none(self):
        _, faces = _make_clean_fan()
        n_nm, details = _detect_non_manifold_edges(faces)
        assert n_nm == 0

    def test_duplicate_face_creates_non_manifold(self):
        _, faces = _make_clean_fan()
        # Duplicate face[0] = [0,1,2] — edges (0,1), (0,2), (1,2) each get +1
        faces = np.vstack([faces, faces[0:1]])
        n_nm, details = _detect_non_manifold_edges(faces)
        # Edge (0,1) is shared by faces [0,1,2] (×2) + [0,4,1] = 3 faces
        # Edge (0,2) is shared by faces [0,1,2] (×2) + [0,2,3] = 3 faces
        # Edge (1,2) is shared by faces [0,1,2] (×2) = 2 (still ok, boundary)
        assert n_nm >= 1

    def test_extra_face_on_edge(self):
        """Manually create a non-manifold edge with 3 distinct faces."""
        faces = np.array([
            [0, 1, 2],  # face on left of edge (0,1)
            [0, 1, 3],  # face on right of edge (0,1)
            [0, 1, 4],  # third face on edge (0,1) — non-manifold!
        ], dtype=np.int32)
        n_nm, details = _detect_non_manifold_edges(faces)
        assert n_nm == 1
        edge, count = details[0]
        assert set(edge) == {0, 1}
        assert count == 3


# ── Tests: fix_non_manifold_mesh ─────────────────────────────────────────────

class TestFixNonManifoldMesh:
    def test_clean_mesh_unchanged(self):
        _, faces = _make_clean_fan()
        faces_fixed = fix_non_manifold_mesh(faces)
        assert len(faces_fixed) == len(faces)

    def test_removes_duplicate_faces(self):
        _, faces = _make_clean_fan()
        faces = np.vstack([faces, faces[0:1]])
        assert len(faces) == 5
        faces_fixed = fix_non_manifold_mesh(faces, verbose=True)
        assert len(faces_fixed) == 4
        # No duplicates remain
        n_dup, _ = _detect_duplicate_faces(faces_fixed)
        assert n_dup == 0

    def test_resolves_non_manifold_from_dup(self):
        _, faces = _make_clean_fan()
        faces = np.vstack([faces, faces[0:1]])
        faces_fixed = fix_non_manifold_mesh(faces)
        n_nm, _ = _detect_non_manifold_edges(faces_fixed)
        assert n_nm == 0

    def test_resolves_non_manifold_extra_face(self):
        """3 distinct faces sharing edge (0,1) → fix keeps only 2."""
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ], dtype=np.int32)
        faces_fixed = fix_non_manifold_mesh(faces)
        assert len(faces_fixed) == 2
        n_nm, _ = _detect_non_manifold_edges(faces_fixed)
        assert n_nm == 0

    def test_prefers_gaussian_faces(self):
        """When resolving non-manifold edges, prefer keeping Gaussian faces."""
        faces = np.array([
            [0, 1, 2],  # no Gaussian
            [0, 1, 3],  # contains Gaussian vertex 3
            [0, 1, 4],  # no Gaussian
        ], dtype=np.int32)
        gauss_idx = np.array([3], dtype=np.int32)
        faces_fixed = fix_non_manifold_mesh(faces, gauss_idx)
        assert len(faces_fixed) == 2
        # Face with vertex 3 must survive
        has_3 = (faces_fixed == 3).any(axis=1)
        assert has_3.any()


# ── Tests: real mesh data (integration) ──────────────────────────────────────

class TestRealMeshTopology:
    """Test topology of actual generated meshes (if available)."""

    MESH_PATHS = [
        "TrainData/Polynomial/SyntheticColmapData/blue_texture/HyperbolicParaboloid/level_02/light_1/output/geodesic_mesh/geodesic_mesh_data.npz",
        "TrainData/Polynomial/SyntheticColmapData/blue_texture/HyperbolicParaboloid/level_03/light_1/output/geodesic_mesh/geodesic_mesh_data.npz",
        "TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_02/light_4/output/geodesic_mesh/geodesic_mesh_data.npz",
    ]

    @pytest.fixture(params=MESH_PATHS)
    def mesh_data(self, request):
        path = project_root / request.param
        if not path.exists():
            pytest.skip(f"Mesh not found: {path}")
        data = np.load(str(path))
        return data["faces"], data.get("gaussian_vertex_indices", None)

    def test_fix_removes_non_manifold(self, mesh_data):
        """After fix_non_manifold_mesh, no non-manifold edges remain."""
        faces, gauss_idx = mesh_data
        faces_fixed = fix_non_manifold_mesh(faces, gauss_idx)
        n_nm, _ = _detect_non_manifold_edges(faces_fixed)
        assert n_nm == 0, f"Still {n_nm} non-manifold edges after fix"

    def test_fix_removes_duplicates(self, mesh_data):
        """After fix, no duplicate faces remain."""
        faces, gauss_idx = mesh_data
        faces_fixed = fix_non_manifold_mesh(faces, gauss_idx)
        n_dup, _ = _detect_duplicate_faces(faces_fixed)
        assert n_dup == 0, f"Still {n_dup} duplicate faces after fix"

    def test_fix_preserves_gaussian_vertices(self, mesh_data):
        """Fix should not break Gaussian vertex coverage."""
        faces, gauss_idx = mesh_data
        if gauss_idx is None:
            pytest.skip("No gaussian_vertex_indices in mesh")
        faces_fixed = fix_non_manifold_mesh(faces, gauss_idx)
        gauss_set = set(gauss_idx.tolist())
        verts_in_mesh = set(faces_fixed.ravel().tolist())
        missing = gauss_set - verts_in_mesh
        # At most a handful may be lost from non-manifold resolution
        assert len(missing) <= 5, (
            f"{len(missing)} Gaussian vertices missing from fixed mesh"
        )

    def test_fix_creates_no_holes(self, mesh_data):
        """After fix, no interior holes should remain."""
        faces, gauss_idx = mesh_data
        faces_fixed = fix_non_manifold_mesh(faces, gauss_idx)
        n_holes = _count_mesh_holes(faces_fixed)
        assert n_holes == 0, f"Fix left {n_holes} hole(s) in mesh"


# ── Tests: _count_mesh_holes ─────────────────────────────────────────────────

class TestCountMeshHoles:
    def test_closed_fan_no_holes(self):
        _, faces = _make_clean_fan()
        assert _count_mesh_holes(faces) == 0

    def test_empty_faces(self):
        faces = np.empty((0, 3), dtype=np.int32)
        assert _count_mesh_holes(faces) == 0

    def test_single_triangle(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        assert _count_mesh_holes(faces) == 0  # boundary but no hole

    def test_annular_mesh_has_one_hole(self):
        """A mesh with outer boundary and an inner hole (annular region)."""
        faces = np.array([
            [0, 1, 5], [0, 5, 4],  # bottom
            [1, 2, 6], [1, 6, 5],  # right
            [2, 3, 7], [2, 7, 6],  # top
            [3, 0, 4], [3, 4, 7],  # left
        ], dtype=np.int32)
        assert _count_mesh_holes(faces) == 1


# ── Tests: _fill_small_holes ─────────────────────────────────────────────────

class TestFillSmallHoles:
    def test_no_holes_no_fill(self):
        _, faces = _make_clean_fan()
        fill = _fill_small_holes(faces)
        assert len(fill) == 0

    def test_fills_quad_hole(self):
        """Annular mesh with 4-vertex inner hole gets filled."""
        faces = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int32)
        fill = _fill_small_holes(faces, max_hole_size=20)
        assert len(fill) == 2  # 4-vertex hole → 2 triangles
        all_faces = np.vstack([faces, fill])
        assert _count_mesh_holes(all_faces) == 0

    def test_fills_triangle_hole(self):
        """Mesh with a 3-vertex inner hole."""
        # Outer hex + inner triangle hole
        faces = np.array([
            [0, 1, 4], [0, 4, 3],  # bottom
            [1, 2, 5], [1, 5, 4],  # right
            [2, 0, 3], [2, 3, 5],  # left
        ], dtype=np.int32)
        n_holes = _count_mesh_holes(faces)
        if n_holes == 0:
            pytest.skip("This mesh has no hole")
        fill = _fill_small_holes(faces, max_hole_size=20)
        all_faces = np.vstack([faces, fill])
        assert _count_mesh_holes(all_faces) == 0

    def test_ignores_large_holes(self):
        """Holes larger than max_hole_size are not filled."""
        faces = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int32)
        fill = _fill_small_holes(faces, max_hole_size=2)
        assert len(fill) == 0  # hole has 4 verts > max_hole_size=2

    def test_fill_does_not_create_duplicates(self):
        """Fill faces should not duplicate existing faces."""
        faces = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int32)
        fill = _fill_small_holes(faces)
        all_faces = np.vstack([faces, fill])
        n_dup, _ = _detect_duplicate_faces(all_faces)
        assert n_dup == 0

    def test_fill_does_not_create_non_manifold(self):
        """Fill faces should not create non-manifold edges."""
        faces = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int32)
        fill = _fill_small_holes(faces)
        all_faces = np.vstack([faces, fill])
        n_nm, _ = _detect_non_manifold_edges(all_faces)
        assert n_nm == 0


# ── Tests: fix_non_manifold_mesh with holes ──────────────────────────────────

class TestFixNonManifoldMeshHoles:
    def test_fix_does_not_create_holes(self):
        """fix_non_manifold_mesh should never leave interior holes."""
        _, faces = _make_clean_fan()
        # Add a duplicate to trigger the fix
        faces = np.vstack([faces, faces[0:1]])
        fixed = fix_non_manifold_mesh(faces)
        assert _count_mesh_holes(fixed) == 0

    def test_fix_cleans_all_topology_issues(self):
        """fix_non_manifold_mesh should leave mesh with 0 dups, 0 NM, 0 holes."""
        _, faces = _make_clean_fan()
        # Add duplicates and extra faces
        extra = np.array([[0, 1, 5]], dtype=np.int32)  # new non-manifold
        faces = np.vstack([faces, faces[0:1], faces[1:2], extra])
        gauss_idx = np.array([0], dtype=np.int32)
        fixed = fix_non_manifold_mesh(faces, gauss_idx)
        n_dup, _ = _detect_duplicate_faces(fixed)
        n_nm, _ = _detect_non_manifold_edges(fixed)
        n_holes = _count_mesh_holes(fixed)
        assert n_dup == 0, f"Duplicates remain: {n_dup}"
        assert n_nm == 0, f"Non-manifold edges remain: {n_nm}"
        assert n_holes == 0, f"Holes remain: {n_holes}"


class TestCenterVertexFill:
    """Tests for the center-vertex hole fill fallback."""

    @staticmethod
    def _make_fill_scenario():
        """Build inputs for _fill_holes_with_center_vertex.

        Creates a hexagonal annular mesh and manually constructs a
        conflicting-hole group where fan fill would create NM edges.
        """
        import math

        # Build vertices on paraboloid z = x² + y²
        verts = []
        # Outer ring r=2 (verts 0-5)
        for i in range(6):
            a = math.radians(60 * i)
            x, y = 2 * math.cos(a), 2 * math.sin(a)
            verts.append([x, y, x * x + y * y])
        # Inner ring r=1 (verts 6-11)
        for i in range(6):
            a = math.radians(60 * i)
            x, y = math.cos(a), math.sin(a)
            verts.append([x, y, x * x + y * y])
        verts = np.array(verts, dtype=np.float64)

        # Annular faces (no inner fill → inner hole)
        faces_list = []
        for i in range(6):
            o0, o1 = i, (i + 1) % 6
            i0, i1 = 6 + i, 6 + (i + 1) % 6
            faces_list.append([o0, o1, i1])
            faces_list.append([o0, i1, i0])
        faces = np.array(faces_list, dtype=np.int32)

        # Edge count of the mesh
        edge_count: dict = {}
        for f in faces:
            s = sorted(f)
            for e in ((s[0], s[1]), (s[0], s[2]), (s[1], s[2])):
                edge_count[e] = edge_count.get(e, 0) + 1

        existing_set = set()
        for f in faces:
            existing_set.add(tuple(sorted(f)))

        # Simulate conflicting fill: pretend these two fan triangles
        # were generated by _fill_small_holes but would conflict.
        # Fan from vertex 6 over the inner hole: first 2 faces
        conflicting_fills = [
            [6, 7, 8],
            [6, 8, 9],
        ]

        return verts, faces, edge_count, existing_set, conflicting_fills

    def test_center_vertex_fill_produces_faces(self):
        """_fill_holes_with_center_vertex should produce fan triangles."""
        verts, faces, edge_count, existing_set, fills = self._make_fill_scenario()
        new_faces, new_verts = _fill_holes_with_center_vertex(
            [fills], faces, verts, "Paraboloid",
            existing_set, edge_count, verbose=False,
        )
        assert len(new_verts) == 1, "Expected 1 new center vertex"
        assert len(new_faces) >= 3, "Expected at least 3 fan triangles"

    def test_center_vertex_on_paraboloid(self):
        """New center vertex z should equal x² + y² (paraboloid)."""
        verts, faces, edge_count, existing_set, fills = self._make_fill_scenario()
        _, new_verts = _fill_holes_with_center_vertex(
            [fills], faces, verts, "Paraboloid",
            existing_set, edge_count, verbose=False,
        )
        for v in new_verts:
            expected_z = v[0] ** 2 + v[1] ** 2
            assert abs(v[2] - expected_z) < 1e-6, (
                f"Vertex {v} not on paraboloid (expected z={expected_z})"
            )

    def test_center_vertex_no_surface_type(self):
        """Without surface_type, center vertex z is the average."""
        verts, faces, edge_count, existing_set, fills = self._make_fill_scenario()
        _, new_verts = _fill_holes_with_center_vertex(
            [fills], faces, verts, None,
            existing_set, edge_count, verbose=False,
        )
        assert len(new_verts) == 1
        hole_vids = sorted({int(v) for f in fills for v in f})
        expected_z = verts[hole_vids, 2].mean()
        assert abs(new_verts[0][2] - expected_z) < 1e-6

    def test_fix_returns_tuple_with_vertices(self):
        """fix_non_manifold_mesh returns (faces, verts) when vertices given."""
        verts, faces = _make_clean_fan()
        result = fix_non_manifold_mesh(faces, vertices=verts)
        assert isinstance(result, tuple) and len(result) == 2

    def test_fix_returns_faces_only_without_vertices(self):
        """fix_non_manifold_mesh returns faces array when no vertices given."""
        _, faces = _make_clean_fan()
        result = fix_non_manifold_mesh(faces)
        assert isinstance(result, np.ndarray)
