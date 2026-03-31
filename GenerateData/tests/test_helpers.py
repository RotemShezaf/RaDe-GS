#!/usr/bin/env python3
"""Tests for compute_gaussian_geodesic_distances_helper.py - geodesic computation helpers."""

import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.compute_gaussian_geodesic_distances_helper import (
    generate_source_mesh_and_map,
    compute_geodesic_distances_for_sources,
    map_indexes_between_gaussian_and_surfaces,
    find_closest_mesh_vertices,
    find_closest_mesh_faces_barycentric,
    project_points_to_triangles_vectorized,
    transfer_geodesic_to_gaussians,
    save_partial_results,
    merge_partial_results,
    find_missing_sources,
    merge_geodesic_data,
    save_computation_metadata,
)


class TestGenerateSourceMeshAndMap:
    """Tests for generate_source_mesh_and_map function."""
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.generate_surface_mesh')
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.KDTree')
    def test_generate_source_mesh_and_map(self, mock_kdtree, mock_generate_mesh):
        """Test source mesh generation and mapping to nearest Gaussians."""
        # Setup mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.randn(100, 3)
        mock_generate_mesh.return_value = (mock_mesh, None, 0.1)
        
        # Setup mock KDTree
        mock_tree = Mock()
        mock_tree.query.return_value = (
            np.random.rand(100),  # distances
            np.random.randint(0, 500, 100)  # Gaussian indices
        )
        mock_kdtree.return_value = mock_tree
        
        # Setup Gaussian positions (replaces gt_vertices)
        gaussian_positions = np.random.randn(500, 3)
        
        # Call function
        source_gaussian_indices, source_positions = generate_source_mesh_and_map(
            surface_type="Paraboloid",
            source_mesh_resolution=10,
            gaussian_positions=gaussian_positions,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0)
        )
        
        # Verify calls
        mock_generate_mesh.assert_called_once()
        mock_kdtree.assert_called_once_with(gaussian_positions)
        mock_tree.query.assert_called_once()
        
        # Verify outputs
        assert source_gaussian_indices.shape[0] == 100  # one per source vertex
        assert source_positions.shape == (100, 3)


class TestComputeGeodesicDistancesForSources:
    """Tests for compute_geodesic_distances_for_sources function."""
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.exact_geodesic_via_vtp_vertex_distance')
    def test_compute_geodesic_basic(self, mock_vtp):
        """Test basic geodesic distance computation."""
        # Create simple mesh (triangle)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)
        
        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32)
        
        source_indices = np.array([0, 1])
        
        # Mock VTP to return geodesic distances
        mock_distances = np.array([
            [0.0, 1.0, 1.5, 2.0],  # From source 0
            [1.0, 0.0, 1.5, 1.0],  # From source 1
        ], dtype=np.float32)
        mock_vtp.return_value = mock_distances
        
        # Compute geodesic distances (sequential mode)
        distances = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=False,
            n_jobs=None
        )
        
        # Verify
        assert distances.shape == (2, 4)
        np.testing.assert_array_equal(distances, mock_distances)
        mock_vtp.assert_called_once()
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.compute_exact_geodesic')
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.exact_geodesic_via_vtp_vertex_distance')
    def test_compute_geodesic_with_fallback(self, mock_vtp, mock_mmp):
        """Test geodesic computation with fallback to MMP for problematic sources."""
        vertices = np.random.randn(100, 3).astype(np.float64)
        faces = np.random.randint(0, 100, (50, 3)).astype(np.int32)
        source_indices = np.array([0, 1, 2])
        
        # Mock VTP to return distances with one problematic source (>10000)
        mock_vtp_distances = np.array([
            [0.0] + [1.0] * 99,     # Good source
            [0.0] + [15000.0] * 99, # Problematic source (max > 10000)
            [0.0] + [2.0] * 99,     # Good source
        ], dtype=np.float32)
        mock_vtp.return_value = mock_vtp_distances
        
        # Mock MMP to fix problematic source
        mock_mmp_distances = np.array([
            [0.0] + [5.0] * 99,  # Fixed distance
        ], dtype=np.float32)
        mock_mmp.return_value = mock_mmp_distances
        
        # Compute
        distances = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=False
        )
        
        # Verify shape
        assert distances.shape == (3, 100)
        
        # Verify VTP was called
        mock_vtp.assert_called_once()
        
        # Verify MMP was called for problematic source (index 1)
        mock_mmp.assert_called_once()
        called_args = mock_mmp.call_args[1]
        np.testing.assert_array_equal(called_args['sources_id'], np.array([1]))
        
        # Verify the problematic source was fixed
        np.testing.assert_array_equal(distances[1, :], mock_mmp_distances[0, :])
    
    def test_compute_geodesic_empty_sources(self):
        """Test with empty source indices."""
        vertices = np.random.randn(10, 3).astype(np.float64)
        faces = np.random.randint(0, 10, (5, 3)).astype(np.int32)
        source_indices = np.array([], dtype=np.int32)
        
        # Should return empty array with correct shape
        distances = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=False,
            n_jobs=None
        )
        
        assert distances.shape == (0, 10)
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.exact_geodesic_via_vtp_vertex_distance')
    def test_compute_geodesic_single_source(self, mock_vtp):
        """Test with single source."""
        vertices = np.random.randn(50, 3).astype(np.float64)
        faces = np.random.randint(0, 50, (25, 3)).astype(np.int32)
        source_indices = np.array([5])
        
        # Mock VTP
        mock_distances = np.random.rand(1, 50).astype(np.float32)
        mock_vtp.return_value = mock_distances
        
        distances = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=True,
            n_jobs=1
        )
        
        assert distances.shape == (1, 50)
        np.testing.assert_array_equal(distances, mock_distances)
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.exact_geodesic_via_vtp_vertex_distance')
    def test_compute_geodesic_distance_properties(self, mock_vtp):
        """Test that computed distances have expected properties."""
        vertices = np.random.randn(20, 3).astype(np.float64)
        faces = np.random.randint(0, 20, (10, 3)).astype(np.int32)
        source_indices = np.array([0, 5, 10])
        
        # Create realistic distance matrix
        # Distance from source to itself should be 0
        mock_distances = np.random.rand(3, 20).astype(np.float32) * 10
        mock_distances[0, 0] = 0.0  # Source 0 to vertex 0
        mock_distances[1, 5] = 0.0  # Source 5 to vertex 5
        mock_distances[2, 10] = 0.0 # Source 10 to vertex 10
        mock_vtp.return_value = mock_distances
        
        distances = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=False,
            n_jobs=None
        )
        
        # Verify distances from sources to themselves are 0
        assert distances[0, 0] == 0.0
        assert distances[1, 5] == 0.0
        assert distances[2, 10] == 0.0
        
        # Verify all distances are non-negative
        assert np.all(distances >= 0)
    
    @patch('GenerateData.utils.compute_gaussian_geodesic_distances_helper.exact_geodesic_via_vtp_vertex_distance')
    def test_compute_geodesic_parallel(self, mock_vtp):
        """Test parallel geodesic distance computation mode."""
        vertices = np.random.randn(20, 3).astype(np.float64)
        faces = np.random.randint(0, 20, (10, 3)).astype(np.int32)
        source_indices = np.array([0, 5, 10])
        
        # Mock VTP for sequential mode (parallel mode bypasses this)
        mock_distances = np.random.rand(3, 20).astype(np.float32) * 10
        mock_distances[0, 0] = 0.0
        mock_distances[1, 5] = 0.0
        mock_distances[2, 10] = 0.0
        mock_vtp.return_value = mock_distances
        
        # Test sequential mode (n_jobs=None or 1)
        distances_seq = compute_geodesic_distances_for_sources(
            vertices,
            faces,
            source_indices,
            verbose=False,
            n_jobs=1
        )
        
        # Verify shape
        assert distances_seq.shape == (3, 20)
        
        # Verify VTP was called in sequential mode
        assert mock_vtp.called
        
        # Note: Testing actual parallel execution with multiprocessing is complex
        # due to pickling requirements. The parallel code path is tested by
        # integration tests and manual testing.


class TestMapIndexesBetweenGaussianAndSurfaces:
    """Tests for map_indexes_between_gaussian_and_surfaces function."""
    
    @patch('GenerateData.utils.data_generation_utils.map_points_to_surface')
    def test_euclidean_mapping(self, mock_map_points):
        """Test Euclidean distance-based mapping."""
        # Setup mock to return expected indices
        mock_map_points.return_value = np.array([0, 1, 2])
        
        # Create sample data
        source_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        dest_surface = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ], dtype=np.float64)
        
        source_idx = np.array([0, 1, 2])
        
        # Map with Euclidean distance
        result_map = map_indexes_between_gaussian_and_surfaces(
            source_idx,
            source_points,
            dest_surface,
            use_mahalanobis=False
        )
        
        # Check mappings (should map to closest vertices)
        assert result_map.shape == (3,)
        # Verify mock was called
        mock_map_points.assert_called_once()
    
    @patch('GenerateData.utils.data_generation_utils.map_points_to_surface')
    def test_mahalanobis_mapping(self, mock_map_points):
        """Test Mahalanobis distance-based mapping."""
        # Setup mock to return indices
        mock_map_points.return_value = np.array([0, 1, 2])
        
        # Create sample data
        source_points = np.random.randn(10, 3).astype(np.float32)
        dest_surface = np.random.randn(100, 3).astype(np.float64)
        dest_scales = np.random.rand(100, 3).astype(np.float32)
        dest_rotations = np.array([
            [1.0, 0.0, 0.0, 0.0] for _ in range(100)
        ], dtype=np.float32)
        
        source_idx = np.array([0, 1, 2])
        
        # Map with Mahalanobis distance
        result_map = map_indexes_between_gaussian_and_surfaces(
            source_idx,
            source_points,
            dest_surface,
            use_mahalanobis=True,
            dest_scales=dest_scales,
            dest_rotations=dest_rotations
        )
        
        # Verify mock was called
        mock_map_points.assert_called_once()
        assert result_map.shape == (3,)


class TestFindClosestMeshVertices:
    """Tests for find_closest_mesh_vertices function."""
    
    @patch('GenerateData.utils.data_generation_utils.map_points_to_surface')
    def test_basic_mahalanobis_distance(self, mock_map_points):
        """Test Mahalanobis distance computation."""
        # Setup mock
        mock_map_points.return_value = (np.array([0]), np.array([0.0]))
        
        # Create simple case: 1 Gaussian, 3 mesh vertices
        gaussian_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        gaussian_scales = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # Isotropic
        gaussian_rotations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # Identity
        
        mesh_vertices = np.array([
            [0.0, 0.0, 0.0],  # Same position
            [1.0, 0.0, 0.0],  # Distance 1 along x
            [2.0, 0.0, 0.0],  # Distance 2 along x
        ], dtype=np.float64)
        
        closest_indices, closest_distances = find_closest_mesh_vertices(
            gaussian_positions,
            mesh_vertices,
            use_mahalanobis=False,
            gaussian_scales=gaussian_scales,
            gaussian_rotations=gaussian_rotations
        )
        
        # Should map to closest vertex (index 0)
        assert closest_indices[0] == 0
        assert closest_distances[0] == 0.0


class TestTransferGeodesicToGaussians:
    """Tests for transfer_geodesic_to_gaussians function."""
    
    def test_transfer_geodesic_distances(self):
        """Test transferring geodesic distances from mesh to Gaussians."""
        # Create sample mapping
        gaussian_to_mesh_map = np.array([0, 1, 2, 0])  # 4 Gaussians
        
        # Create geodesic distances (3 sources, 3 mesh vertices)
        mesh_geodesic_distances = np.array([
            [0.0, 1.0, 2.0],  # Source 0
            [1.0, 0.0, 1.0],  # Source 1
            [2.0, 1.0, 0.0],  # Source 2
        ], dtype=np.float32)
        
        # Source mesh and Gaussian indices
        source_mesh_indices = np.array([0, 1, 2])
        source_gaussian_indices = np.array([0, 1, 2])
        
        # Transfer to Gaussians
        gaussian_geodesic_distances = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances,
            gaussian_to_mesh_map,
            source_mesh_indices,
            source_gaussian_indices
        )
        
        # Check shape
        assert gaussian_geodesic_distances.shape == (3, 4)
        
        # Check transferred values
        assert gaussian_geodesic_distances[0, 0] == 0.0  # Source 0 -> Gaussian 0 (mesh 0)
        assert gaussian_geodesic_distances[0, 1] == 1.0  # Source 0 -> Gaussian 1 (mesh 1)
        assert gaussian_geodesic_distances[0, 2] == 2.0  # Source 0 -> Gaussian 2 (mesh 2)
        assert gaussian_geodesic_distances[0, 3] == 0.0  # Source 0 -> Gaussian 3 (mesh 0)

    def test_transfer_preserves_row_order_with_unsorted_gaussian_indices(self):
        """Row ordering must match input source order, NOT sorted by source_gaussian_indices.

        Regression test: a previous bug sorted rows by argsort(source_gaussian_indices)
        inside transfer_geodesic_to_gaussians, creating a mismatch between the returned
        distance matrix and the metadata arrays (source_indices, source_gaussian_indices)
        that were saved alongside it.
        """
        # 4 mesh vertices, 5 Gaussians
        gaussian_to_mesh_map = np.array([0, 1, 2, 3, 0])

        # 3 sources: mesh vertices 2, 0, 3 (note: NOT sorted)
        mesh_geodesic_distances = np.array([
            [2.0, 1.0, 0.0, 1.5],  # Source at mesh-vertex 2
            [0.0, 1.0, 2.0, 3.0],  # Source at mesh-vertex 0
            [1.5, 0.5, 1.0, 0.0],  # Source at mesh-vertex 3
        ], dtype=np.float32)

        source_mesh_indices = np.array([2, 0, 3])
        # Gaussian indices are intentionally unsorted
        source_gaussian_indices = np.array([4, 1, 3])

        result = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances,
            gaussian_to_mesh_map,
            source_mesh_indices,
            source_gaussian_indices,
        )

        assert result.shape == (3, 5)

        # Row 0 must still correspond to mesh-vertex 2 (source_mesh_indices[0])
        # Gaussian 0 -> mesh 0: dist from mesh 2 = 2.0
        assert result[0, 0] == 2.0
        # Gaussian 2 -> mesh 2: dist from mesh 2 = 0.0  (self)
        assert result[0, 2] == 0.0

        # Row 1 must still correspond to mesh-vertex 0
        assert result[1, 0] == 0.0  # Gaussian 0 -> mesh 0 (self)
        assert result[1, 2] == 2.0  # Gaussian 2 -> mesh 2

        # Row 2 must still correspond to mesh-vertex 3
        assert result[2, 3] == 0.0  # Gaussian 3 -> mesh 3 (self)
        assert result[2, 0] == 1.5  # Gaussian 0 -> mesh 0

    def test_barycentric_interpolation(self):
        """transfer_geodesic_to_gaussians with barycentric mode interpolates correctly.

        Setup: a single equilateral-ish triangle with vertices 0,1,2.
        One source at vertex 0.  Gaussian is the centroid of the triangle,
        so barycentric weights are all 1/3 and the interpolated distance should be
        (D[0]+D[1]+D[2]) / 3.
        """
        # Mesh: 3 vertices, 1 face
        # Source at vertex 0: distances to vertices [0.0, 1.0, 2.0]
        mesh_geodesic_distances = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)  # (1, 3)
        gaussian_to_mesh_indices = np.array([0])  # not used in bary mode but required
        source_mesh_indices      = np.array([0])
        source_gaussian_indices  = np.array([0])

        # Gaussian at centroid of face 0 -> equal barycentric weights
        barycentric_face_vertices = np.array([[0, 1, 2]], dtype=np.int64)  # (1, 3)
        barycentric_weights       = np.array([[1/3, 1/3, 1/3]], dtype=np.float64)

        result = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances,
            gaussian_to_mesh_indices,
            source_mesh_indices,
            source_gaussian_indices,
            barycentric_face_vertices=barycentric_face_vertices,
            barycentric_weights=barycentric_weights,
        )

        assert result.shape == (1, 1)
        expected = (0.0 + 1.0 + 2.0) / 3.0
        assert abs(result[0, 0] - expected) < 1e-9

    def test_barycentric_vertex_position(self):
        """When Gaussian sits on vertex 0, barycentric gives same result as snapping."""
        mesh_geodesic_distances = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)  # (1, 3)
        gaussian_to_mesh_indices = np.array([0])
        source_mesh_indices      = np.array([0])
        source_gaussian_indices  = np.array([0])

        # Gaussian at vertex 0: bary = [1, 0, 0]
        barycentric_face_vertices = np.array([[0, 1, 2]], dtype=np.int64)
        barycentric_weights       = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        result = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances,
            gaussian_to_mesh_indices,
            source_mesh_indices,
            source_gaussian_indices,
            barycentric_face_vertices=barycentric_face_vertices,
            barycentric_weights=barycentric_weights,
        )

        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 0.0) < 1e-9  # coincides with source vertex

    def test_fallback_when_no_barycentric(self):
        """Without barycentric args the function falls back to vertex snapping."""
        gaussian_to_mesh_map = np.array([0, 1, 2, 0])
        mesh_geodesic_distances = np.array([
            [0.0, 1.0, 2.0],
        ], dtype=np.float32)
        source_mesh_indices     = np.array([0])
        source_gaussian_indices = np.array([0])

        result = transfer_geodesic_to_gaussians(
            mesh_geodesic_distances,
            gaussian_to_mesh_map,
            source_mesh_indices,
            source_gaussian_indices,
            # no barycentric args
        )

        assert result.shape == (1, 4)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 1.0
        assert result[0, 3] == 0.0  # Gaussian 3 maps to mesh vertex 0


class TestProjectPointsToTrianglesVectorized:
    """Tests for project_points_to_triangles_vectorized."""

    def test_point_at_vertex_a(self):
        """Point coincident with vertex A should return bary=[1,0,0], dist=0."""
        A = np.array([[0.0, 0.0, 0.0]])
        B = np.array([[1.0, 0.0, 0.0]])
        C = np.array([[0.0, 1.0, 0.0]])
        P = A.copy()
        bary, dists = project_points_to_triangles_vectorized(P, A, B, C)
        np.testing.assert_allclose(bary[0], [1.0, 0.0, 0.0], atol=1e-9)
        assert dists[0] < 1e-9

    def test_point_at_centroid(self):
        """Point at centroid should have equal barycentric weights and dist=0."""
        A = np.array([[0.0, 0.0, 0.0]])
        B = np.array([[1.0, 0.0, 0.0]])
        C = np.array([[0.0, 1.0, 0.0]])
        P = (A + B + C) / 3.0
        bary, dists = project_points_to_triangles_vectorized(P, A, B, C)
        np.testing.assert_allclose(bary[0], [1/3, 1/3, 1/3], atol=1e-9)
        assert dists[0] < 1e-9

    def test_point_above_triangle(self):
        """Point directly above centroid projects to centroid, bary = [1/3,1/3,1/3]."""
        A = np.array([[0.0, 0.0, 0.0]])
        B = np.array([[1.0, 0.0, 0.0]])
        C = np.array([[0.0, 1.0, 0.0]])
        centroid = (A + B + C) / 3.0
        P = centroid + np.array([[0.0, 0.0, 2.0]])  # 2 units above
        bary, dists = project_points_to_triangles_vectorized(P, A, B, C)
        np.testing.assert_allclose(bary[0], [1/3, 1/3, 1/3], atol=1e-6)
        np.testing.assert_allclose(dists[0], 2.0, atol=1e-6)

    def test_barycentric_weights_sum_to_one(self):
        """Barycentric weights must always sum to 1 for arbitrary points."""
        rng = np.random.default_rng(0)
        N = 200
        A = rng.standard_normal((N, 3))
        B = rng.standard_normal((N, 3))
        C = rng.standard_normal((N, 3))
        P = rng.standard_normal((N, 3))
        bary, _ = project_points_to_triangles_vectorized(P, A, B, C)
        np.testing.assert_allclose(bary.sum(axis=1), np.ones(N), atol=1e-9)
        assert (bary >= -1e-9).all(), "Some barycentric weights are negative"


class TestFindClosestMeshFacesBarycentric:
    """Tests for find_closest_mesh_faces_barycentric."""

    def _simple_mesh(self):
        """A 2-triangle mesh on the xy-plane.

        Vertices:  0=(0,0,0)  1=(1,0,0)  2=(0,1,0)  3=(1,1,0)
        Faces:     0=(0,1,2)  1=(1,3,2)
        """
        verts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[1,3,2]], dtype=np.int64)
        return verts, faces

    def test_point_at_vertex(self):
        """Gaussian exactly at vertex 0 should map to face 0 with bary [1,0,0]."""
        verts, faces = self._simple_mesh()
        point = np.array([[0.0, 0.0, 0.0]])
        closest_v = np.array([0])
        fv, bw = find_closest_mesh_faces_barycentric(point, verts, faces, closest_v)
        # vertex 0 belongs to face 0 only
        assert 0 in fv[0]
        v0_pos = np.where(fv[0] == 0)[0][0]
        assert abs(bw[0, v0_pos] - 1.0) < 1e-6

    def test_point_above_face_centroid(self):
        """Gaussian above face-0 centroid should project onto face 0 with [1/3,1/3,1/3]."""
        verts, faces = self._simple_mesh()
        centroid = verts[faces[0]].mean(axis=0)
        point = centroid[None, :] + np.array([[0, 0, 0.5]])  # above centroid
        # closest vertex in this mesh is one of {0,1,2} — find it manually
        from scipy.spatial import KDTree
        tree = KDTree(verts)
        _, cv = tree.query(point)
        fv, bw = find_closest_mesh_faces_barycentric(point, verts, faces, cv.ravel())
        np.testing.assert_allclose(bw[0], [1/3, 1/3, 1/3], atol=1e-6)

    def test_output_shapes(self):
        """Output arrays have the right shapes."""
        verts, faces = self._simple_mesh()
        G = 5
        rng = np.random.default_rng(1)
        points = rng.uniform(0, 1, (G, 3))
        points[:, 2] = 0  # keep on plane
        from scipy.spatial import KDTree
        _, cv = KDTree(verts).query(points)
        fv, bw = find_closest_mesh_faces_barycentric(points, verts, faces, cv)
        assert fv.shape == (G, 3)
        assert bw.shape == (G, 3)
        np.testing.assert_allclose(bw.sum(axis=1), np.ones(G), atol=1e-9)


class TestSaveAndMergeResults:
    """Tests for save/merge partial results functions."""
    
    def test_save_partial_results(self, tmp_path):
        """Test saving partial results to file."""
        # Create sample data
        gaussian_positions = np.random.randn(3, 3).astype(np.float32)
        source_indices = np.array([10, 20], dtype=np.int32)
        source_positions = np.random.randn(2, 3).astype(np.float32)
        geodesic_distances = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
        ], dtype=np.float32)
        closest_mesh_indices = np.array([0, 1, 2], dtype=np.int32)
        closest_mesh_distances = np.array([0.1, 0.2, 0.15], dtype=np.float32)
        source_gaussian_indices = np.array([0, 1], dtype=np.int32)
        
        # Save to temp file
        output_path = tmp_path / "test_partial.npz"
        
        save_partial_results(
            output_path,
            gaussian_positions,
            source_indices,
            source_positions,
            geodesic_distances,
            closest_mesh_indices,
            closest_mesh_distances,
            source_gaussian_indices
        )
        
        # Check file exists
        assert output_path.exists()
        
        # Load and verify
        data = np.load(output_path)
        np.testing.assert_array_equal(data['geodesic_distances'], geodesic_distances)
        np.testing.assert_array_equal(data['source_indices'], source_indices)
        np.testing.assert_array_equal(data['gaussian_positions'], gaussian_positions)
    
    def test_merge_partial_results(self, tmp_path):
        """Test merging multiple partial result files."""
        # Create proper directory structure
        output_dir = tmp_path / "output"
        partial_dir = output_dir / "geodesic_distance" / "gt_partial"
        partial_dir.mkdir(parents=True)
        
        # Use consistent gaussian_positions across all partial files
        gaussian_positions = np.random.randn(100, 3).astype(np.float32)
        
        # Save 3 partial results with proper format
        for i in range(3):
            source_indices = np.array([i*2, i*2+1], dtype=np.int32)
            source_positions = np.random.randn(2, 3).astype(np.float32)
            geodesic_distances = np.random.rand(2, 100).astype(np.float32)
            closest_mesh_indices = np.random.randint(0, 1000, 100).astype(np.int32)
            closest_mesh_distances = np.random.rand(100).astype(np.float32)
            source_gaussian_indices = np.array([i*2, i*2+1], dtype=np.int32)
            
            filename = partial_dir / f"sources_range_{i}.npz"
            np.savez(filename,
                    gaussian_positions=gaussian_positions,  # Same for all files
                    source_indices=source_indices,
                    source_positions=source_positions,
                    geodesic_distances=geodesic_distances,
                    closest_mesh_indices=closest_mesh_indices,
                    closest_mesh_distances=closest_mesh_distances,
                    source_gaussian_indices=source_gaussian_indices)
        
        # Merge results (function returns None, saves to file)
        merge_partial_results(output_dir, verbose=False)
        
        # Check that merged file was created
        merged_file = output_dir / "geodesic_distance" / "gt_geodesic.npz"
        assert merged_file.exists(), f"Expected file not found: {merged_file}"


class TestFindMissingSourcesAndMerge:
    """Tests for find_missing_sources and merge_geodesic_data functions."""
    
    def test_find_missing_sources(self, tmp_path):
        """Test finding missing sources in partial results."""
        # All sources are [0, 1, 2, 3, 4]
        all_sources = np.array([0, 1, 2, 3, 4])
        
        # Existing sources are [0, 2, 4]
        existing_sources = np.array([0, 2, 4], dtype=np.int32)
        
        # Find missing
        missing, missing_mask = find_missing_sources(all_sources, existing_sources)
        
        # Should find [1, 3]
        expected_missing = np.array([1, 3])
        np.testing.assert_array_equal(sorted(missing), sorted(expected_missing))
    
    def test_merge_geodesic_data(self):
        """Test merging old and new geodesic data."""
        # Create old data (3 sources)
        old_source_indices = np.array([0, 2, 4])
        old_source_positions = np.random.randn(3, 3).astype(np.float32)
        old_geodesic_distances = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ], dtype=np.float32)
        
        # Create new data (2 sources)
        new_source_indices = np.array([1, 3])
        new_source_positions = np.random.randn(2, 3).astype(np.float32)
        new_geodesic_distances = np.array([
            [1.2, 0.0, 1.3],
            [2.1, 1.4, 0.0],
        ], dtype=np.float32)
        
        # Merge
        merged_indices, merged_positions, merged_distances = merge_geodesic_data(
            old_source_indices, old_source_positions, old_geodesic_distances,
            new_source_indices, new_source_positions, new_geodesic_distances
        )
        
        # Check result (5 sources total)
        assert merged_distances.shape == (5, 3)
        assert len(merged_indices) == 5
        assert merged_positions.shape == (5, 3)
        
        # Check ordering (should be sorted by source index)
        expected_indices = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(merged_indices, expected_indices)


class TestSaveComputationMetadata:
    """Tests for save_computation_metadata function."""
    
    def test_save_metadata(self, tmp_path):
        """Test saving computation metadata."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()
        
        # Create mock args with all required attributes
        from argparse import Namespace
        args = Namespace(
            surface="Paraboloid",
            data_root="/tmp/data",
            mesh_level=2,
            source_mesh_resolution=20,
            iteration=30000,
            mapping_method="euclidean",
            gaussian_output="/tmp/gaussian_output",
            use_mahalanobis=False,
            partial_start=0,
            partial_end=None,
            source_selection="uniform",  # Added
            seed=42,  # Added
            source_start=None,  # Added
            source_end=None  # Added
        )
        
        save_computation_metadata(
            output_folder,
            args,
            num_gaussians=10000,
            num_sources=400,
            surface_name="Paraboloid"
        )
        
        # Check file exists
        metadata_file = output_folder / "geodesic_distance" / "computation_metadata.json"
        assert metadata_file.exists()
        
        # Load and verify
        with open(metadata_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["surface"]["name"] == "Paraboloid"
        assert loaded["surface"]["mesh_level"] == 2
        assert loaded["source_generation"]["total_sources"] == 400
        assert "timestamp" in loaded["computation_info"]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('GenerateData.utils.data_generation_utils.map_points_to_surface')
    def test_empty_gaussian_positions(self, mock_map_points):
        """Test handling of empty Gaussian data."""
        mock_map_points.return_value = np.array([])
        
        source_idx = np.array([], dtype=np.int32)
        source_points = np.array([], dtype=np.float32).reshape(0, 3)
        dest_surface = np.random.randn(100, 3)
        
        result = map_indexes_between_gaussian_and_surfaces(
            source_idx,
            source_points,
            dest_surface,
            use_mahalanobis=False
        )
        
        assert result.shape == (0,)
    
    @patch('GenerateData.utils.data_generation_utils.map_points_to_surface')
    def test_single_gaussian(self, mock_map_points):
        """Test with single Gaussian."""
        mock_map_points.return_value = np.array([0])
        
        source_idx = np.array([0])
        source_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        dest_surface = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float64)
        
        result = map_indexes_between_gaussian_and_surfaces(
            source_idx,
            source_points,
            dest_surface,
            use_mahalanobis=False
        )
        
        assert result.shape == (1,)
        assert result[0] == 0  # Should map to closest (first) vertex


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
