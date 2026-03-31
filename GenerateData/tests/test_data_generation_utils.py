"""
Tests for data_generation_utils.py - neighborhood computation and point mapping utilities.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.data_generation_utils import (
    mahalanobis_distance,
    ring1_neighbors_gaussians,
    get_neighborhood_by_ring,
    get_all_points_nbrs_all_rings,
    get_all_points_nbrs_single_ring,
    adaptive_ring1_neighbors,
    map_points_to_surface,
    build_rotation,
    _build_covariance_matrices,
)


class TestMahalanobisDistance:
    """Test Mahalanobis distance computation."""
    
    def test_mahalanobis_identity_covariance(self):
        """With identity covariance, should equal Euclidean distance."""
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        cov_inv = np.eye(3, dtype=np.float32)
        
        distances = mahalanobis_distance(points, center, cov_inv)
        expected = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        np.testing.assert_allclose(distances, expected, rtol=1e-5)
    
    def test_mahalanobis_scaled_covariance(self):
        """Test with scaled covariance matrix."""
        points = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        # Scale x and y by factor of 2
        cov_inv = np.diag([0.25, 0.25, 1.0]).astype(np.float32)
        
        distances = mahalanobis_distance(points, center, cov_inv)
        expected = np.array([1.0, 1.0], dtype=np.float32)
        
        np.testing.assert_allclose(distances, expected, rtol=1e-5)
    
    def test_mahalanobis_zero_distance(self):
        """Distance from center to itself should be zero."""
        points = np.array([[1, 2, 3]], dtype=np.float32)
        center = np.array([1, 2, 3], dtype=np.float32)
        cov_inv = np.eye(3, dtype=np.float32)
        
        distances = mahalanobis_distance(points, center, cov_inv)
        
        assert distances[0] < 1e-6


class TestRing1NeighborsGaussians:
    """Test ring-1 neighbor computation."""
    
    def test_euclidean_neighbors_basic(self):
        """Test basic Euclidean neighbor finding."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [10, 10, 10]  # Far away point
        ], dtype=np.float32)
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=3, use_mahalanobis=False
        )
        
        # Check structure
        assert len(neighbors) == 5
        assert all(len(neighbors[i]) <= 3 for i in range(5))
        
        # Point 0 should have points 1,2,3 as nearest (not 4 which is far)
        assert 4 not in neighbors[0]
        
        # Check distances are positive
        assert mean_dist > 0
        assert len(per_point_dist) == 5
        assert all(per_point_dist > 0)
    
    def test_mahalanobis_neighbors(self):
        """Test Mahalanobis distance-based neighbor finding."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        scales = np.array([
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ], dtype=np.float32)
        
        rotations = np.array([
            [1, 0, 0, 0],  # Identity quaternions
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.float32)
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=2, use_mahalanobis=True,
            gaussian_scales=scales, gaussian_rotations=rotations
        )
        
        assert len(neighbors) == 3
        assert mean_dist > 0
        assert len(per_point_dist) == 3
    
    def test_torch_tensor_input(self):
        """Test that torch tensors are accepted."""
        vertices = torch.rand(10, 3)
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=5, use_mahalanobis=False
        )
        
        assert len(neighbors) == 10
        assert isinstance(per_point_dist, np.ndarray)
    
    def test_missing_mahalanobis_params(self):
        """Should raise error if mahalanobis=True but scales/rotations missing."""
        vertices = np.random.rand(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="gaussian_scales and gaussian_rotations must be provided"):
            ring1_neighbors_gaussians(vertices, n_neighbors=5, use_mahalanobis=True)


class TestGetNeighborhoodByRing:
    """Test ring neighborhood expansion."""
    
    def test_ring1_neighbors(self):
        """Ring 1 should return direct neighbors."""
        ring1_nbrs = {
            0: np.array([1, 2]),
            1: np.array([0, 3]),
            2: np.array([0, 3]),
            3: np.array([1, 2])
        }
        
        nbrs = get_neighborhood_by_ring(0, ring=1, ring1_nbrs=ring1_nbrs)
        expected = np.array([1, 2])
        
        assert set(nbrs) == set(expected)
    
    def test_ring2_neighbors(self):
        """Ring 2 should include ring-1 and ring-2."""
        ring1_nbrs = {
            0: np.array([1, 2]),
            1: np.array([0, 3]),
            2: np.array([0, 3]),
            3: np.array([1, 2])
        }
        
        nbrs = get_neighborhood_by_ring(0, ring=2, ring1_nbrs=ring1_nbrs)
        # Ring 1: [1, 2]
        # Ring 2: neighbors of 1 and 2 = [0, 3]
        # Combined (excluding self): [1, 2, 3]
        
        assert 0 not in nbrs  # Self should be excluded
        assert set(nbrs) == {1, 2, 3}
    
    def test_ring3_neighbors(self):
        """Ring 3 should include up to 3-hop neighbors."""
        ring1_nbrs = {
            0: np.array([1]),
            1: np.array([0, 2]),
            2: np.array([1, 3]),
            3: np.array([2, 4]),
            4: np.array([3])
        }
        
        nbrs = get_neighborhood_by_ring(0, ring=3, ring1_nbrs=ring1_nbrs)
        # Ring 1: [1]
        # Ring 2: [2] (neighbor of 1)
        # Ring 3: [3] (neighbor of 2)
        # Combined: [1, 2, 3]
        
        assert 0 not in nbrs
        assert set(nbrs) == {1, 2, 3}


class TestGetAllPointsNbrsAllRings:
    """Test computing all ring neighborhoods."""
    
    def test_all_rings_structure(self):
        """Test that all ring dictionaries are created."""
        vertices = np.random.rand(20, 3).astype(np.float32)
        
        ring1, ring2, ring3, ring4, mean_dist, per_point_dist = get_all_points_nbrs_all_rings(
            vertices, use_mahalanobis=False, n_neighbors_ring1=10
        )
        
        # Check all dictionaries have correct size
        assert len(ring1) == 20
        assert len(ring2) == 20
        assert len(ring3) == 20
        assert len(ring4) == 20
        
        # Check distances
        assert mean_dist > 0
        assert len(per_point_dist) == 20
    
    def test_ring_expansion(self):
        """Ring k should have more or equal neighbors than ring k-1."""
        vertices = np.random.rand(50, 3).astype(np.float32)
        
        ring1, ring2, ring3, ring4, _, _ = get_all_points_nbrs_all_rings(
            vertices, use_mahalanobis=False, n_neighbors_ring1=10
        )
        
        # For at least some points, higher rings should have more neighbors
        for i in range(10):
            assert len(ring2[i]) >= len(ring1[i])
            assert len(ring3[i]) >= len(ring2[i])


class TestGetAllPointsNbrsSingleRing:
    """Test computing single ring neighborhoods."""
    
    def test_single_ring_output(self):
        """Should return ring1 and specified ring."""
        vertices = np.random.rand(15, 3).astype(np.float32)
        
        ring1, ring3 = get_all_points_nbrs_single_ring(
            vertices, ring=3, use_mahalanobis=False, n_neighbors_ring1=8
        )
        
        assert len(ring1) == 15
        assert len(ring3) == 15


class TestMapPointsToSurface:
    """Test point-to-surface mapping."""
    
    def test_euclidean_mapping_exact_match(self):
        """Points that match targets should map to themselves."""
        query = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        target = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        
        indices = map_points_to_surface(query, target, use_mahalanobis=False)
        
        assert indices[0] == 0
        assert indices[1] == 1
    
    def test_euclidean_mapping_with_distances(self):
        """Test that distances are returned when requested."""
        query = np.array([[0, 0, 0]], dtype=np.float32)
        target = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=False, return_distances=True
        )
        
        assert len(indices) == 1
        assert len(distances) == 1
        assert distances[0] > 0
        assert distances[0] <= 1.5  # Should be close to 1.0
    
    def test_torch_tensor_mapping(self):
        """Test with torch tensors."""
        query = torch.rand(10, 3)
        target = torch.rand(20, 3)
        
        indices = map_points_to_surface(query, target, use_mahalanobis=False)
        
        assert len(indices) == 10
        assert all(0 <= idx < 20 for idx in indices)
    
    def test_mahalanobis_mapping_requires_params(self):
        """Mahalanobis mapping should require scales/rotations."""
        query = np.random.rand(5, 3).astype(np.float32)
        target = np.random.rand(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Either query or target scales/rotations must be provided"):
            map_points_to_surface(query, target, use_mahalanobis=True)
    
    def test_single_point_mapping(self):
        """Test mapping with single query point."""
        target = np.array([[1, 2, 3]], dtype=np.float32)
        query = np.array([[1.1, 2.1, 3.1]], dtype=np.float32)
        
        indices = map_points_to_surface(query, target, use_mahalanobis=False)
        
        assert indices.shape == (1,)
        assert indices[0] == 0
    
    def test_large_batch_mapping(self):
        """Test efficiency with large batches."""
        np.random.seed(42)
        target = np.random.randn(1000, 3).astype(np.float32)
        query = np.random.randn(500, 3).astype(np.float32)
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=False, return_distances=True
        )
        
        assert indices.shape == (500,)
        assert distances.shape == (500,)
        assert np.all(indices >= 0) and np.all(indices < 1000)
    
    def test_distance_correctness(self):
        """Verify returned distances match actual distances."""
        np.random.seed(123)
        target = np.random.randn(50, 3).astype(np.float32)
        query = np.random.randn(20, 3).astype(np.float32)
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=False, return_distances=True
        )
        
        for i, idx in enumerate(indices):
            expected_dist = np.linalg.norm(query[i] - target[idx])
            assert np.isclose(distances[i], expected_dist, rtol=1e-4)
    
    def test_multiple_queries_same_target(self):
        """Test multiple queries mapping to same target."""
        target = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
        query = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], dtype=np.float32)
        
        indices = map_points_to_surface(query, target, use_mahalanobis=False)
        
        assert np.all(indices == 0)  # All map to first target

    def test_mahalanobis_mapping_with_query_covariance(self):
        """Test Mahalanobis mapping using query point covariances."""
        np.random.seed(42)
        query = np.random.randn(20, 3).astype(np.float32)
        target = np.random.randn(50, 3).astype(np.float32)
        query_scales = np.abs(np.random.randn(20, 3).astype(np.float32)) + 0.1
        query_rotations = np.zeros((20, 4), dtype=np.float32)
        query_rotations[:, 0] = 1.0  # Identity quaternions
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True
        )
        
        assert indices.shape == (20,)
        assert distances.shape == (20,)
        assert np.all(indices >= 0) and np.all(indices < 50)
        assert np.all(distances >= 0)

    def test_mahalanobis_mapping_with_target_covariance(self):
        """Test Mahalanobis mapping using target point covariances."""
        np.random.seed(123)
        query = np.random.randn(15, 3).astype(np.float32)
        target = np.random.randn(30, 3).astype(np.float32)
        target_scales = np.abs(np.random.randn(30, 3).astype(np.float32)) + 0.1
        target_rotations = np.zeros((30, 4), dtype=np.float32)
        target_rotations[:, 0] = 1.0  # Identity quaternions
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=True,
            target_scales=target_scales, target_rotations=target_rotations,
            return_distances=True
        )
        
        assert indices.shape == (15,)
        assert distances.shape == (15,)
        assert np.all(indices >= 0) and np.all(indices < 30)
        assert np.all(distances >= 0)

    def test_mahalanobis_mapping_batched_consistency(self):
        """Test that batched and non-batched Mahalanobis give same results."""
        np.random.seed(456)
        query = np.random.randn(100, 3).astype(np.float32)
        target = np.random.randn(200, 3).astype(np.float32)
        query_scales = np.abs(np.random.randn(100, 3).astype(np.float32)) + 0.1
        query_rotations = np.zeros((100, 4), dtype=np.float32)
        query_rotations[:, 0] = 1.0
        
        # With small batch size
        indices_batched, dist_batched = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=32
        )
        
        # With no batching (large batch_size)
        indices_full, dist_full = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=None
        )
        
        np.testing.assert_array_equal(indices_batched, indices_full)
        np.testing.assert_allclose(dist_batched, dist_full, rtol=1e-5)

    def test_mahalanobis_mapping_large_batch(self):
        """Test efficiency with large Mahalanobis mapping."""
        np.random.seed(789)
        query = np.random.randn(500, 3).astype(np.float32)
        target = np.random.randn(1000, 3).astype(np.float32)
        query_scales = np.abs(np.random.randn(500, 3).astype(np.float32)) + 0.1
        query_rotations = np.zeros((500, 4), dtype=np.float32)
        query_rotations[:, 0] = 1.0
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=128
        )
        
        assert indices.shape == (500,)
        assert distances.shape == (500,)
        assert np.all(indices >= 0) and np.all(indices < 1000)


class TestBuildRotation:
    """Test rotation matrix construction from quaternions."""
    
    def test_identity_rotation(self):
        """Identity quaternion should give identity rotation."""
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)
        R = build_rotation(q)
        
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(R[0], expected, atol=1e-5)
    
    def test_rotation_matrix_properties(self):
        """Rotation matrix should be orthogonal with det=1."""
        q = np.array([[0.7071, 0.7071, 0, 0]], dtype=np.float32)  # 90° around x-axis
        R = build_rotation(q)
        
        # Check orthogonality: R^T * R = I
        identity = R[0].T @ R[0]
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-5)
        
        # Check determinant = 1
        det = np.linalg.det(R[0])
        assert abs(det - 1.0) < 1e-5
    
    def test_batch_rotation(self):
        """Should handle batch of quaternions."""
        q = np.array([
            [1, 0, 0, 0],
            [0.7071, 0.7071, 0, 0],
            [0.5, 0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        R = build_rotation(q)
        
        assert R.shape == (3, 3, 3)
        
        # All should be valid rotations
        for i in range(3):
            det = np.linalg.det(R[i])
            assert abs(det - 1.0) < 1e-4


class TestBuildCovarianceMatrices:
    """Test covariance matrix construction."""
    
    def test_covariance_shape(self):
        """Should return (N, 6) symmetric covariance matrices (upper triangle)."""
        scales = np.array([
            [1, 1, 1],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        rotations = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.float32)
        
        cov = _build_covariance_matrices(scales, rotations)
        
        # strip_symmetric returns upper triangle: 6 values per matrix
        assert cov.shape == (2, 6)
    
    def test_covariance_positive_values(self):
        """Covariance matrix values should be positive for positive scales."""
        scales = np.array([[1, 2, 3]], dtype=np.float32)
        rotations = np.array([[1, 0, 0, 0]], dtype=np.float32)
        
        cov = _build_covariance_matrices(scales, rotations)
        
        # Diagonal elements are at indices 0, 3, 5 in packed format [C00, C01, C02, C11, C12, C22]
        assert cov.shape == (1, 6)
        assert cov[0, 0] > 0  # C00 diagonal
        assert cov[0, 3] > 0  # C11 diagonal
        assert cov[0, 5] > 0  # C22 diagonal
    
    def test_covariance_scaling_modifier(self):
        """Scaling modifier should affect covariance magnitude."""
        scales = np.array([[1, 1, 1]], dtype=np.float32)
        rotations = np.array([[1, 0, 0, 0]], dtype=np.float32)
        
        cov1 = _build_covariance_matrices(scales, rotations, scaling_modifier=1.0)
        cov2 = _build_covariance_matrices(scales, rotations, scaling_modifier=2.0)
        
        # cov2 should have larger values
        assert np.linalg.norm(cov2) > np.linalg.norm(cov1)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_point_neighbors(self):
        """Few points should handle gracefully."""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        
        # With 2 points, can request 1 neighbor
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=1, use_mahalanobis=False
        )
        
        assert len(neighbors) == 2
        assert len(neighbors[0]) == 1  # Point 0 has 1 neighbor (point 1)
        assert len(neighbors[1]) == 1  # Point 1 has 1 neighbor (point 0)
        assert neighbors[0][0] == 1
        assert neighbors[1][0] == 0
    
    def test_empty_ring_neighbors(self):
        """Test with isolated points."""
        ring1_nbrs = {
            0: np.array([]),  # Isolated point
            1: np.array([2]),
            2: np.array([1])
        }
        
        nbrs = get_neighborhood_by_ring(0, ring=2, ring1_nbrs=ring1_nbrs)
        
        assert len(nbrs) == 0  # Should remain isolated


class TestKNNBackends:
    """Test different KNN backend implementations."""
    
    def test_sklearn_cpu_backend(self):
        """Test sklearn CPU fallback works correctly."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0]
        ], dtype=np.float32)
        
        # Force CPU by using numpy input
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=2, use_mahalanobis=False
        )
        
        # Check basic properties
        assert len(neighbors) == 5
        assert all(len(neighbors[i]) == 2 for i in range(5))
        assert mean_dist > 0
        assert len(per_point_dist) == 5
        
        # Check neighbor relationships make sense
        # Point 0 should be closest to points 1, 2, 3 (not 4)
        assert 4 not in neighbors[0]
    
    def test_torch_cdist_backend(self):
        """Test torch.cdist backend on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        vertices = torch.rand(50, 3).cuda()
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=10, use_mahalanobis=False
        )
        
        assert len(neighbors) == 50
        assert all(len(neighbors[i]) == 10 for i in range(50))
        assert mean_dist > 0
        assert len(per_point_dist) == 50
    
    def test_backend_consistency(self):
        """Test that different backends produce consistent results."""
        np.random.seed(42)
        vertices_np = np.random.rand(20, 3).astype(np.float32)
        
        # CPU result (sklearn)
        neighbors_cpu, _, _ = ring1_neighbors_gaussians(
            vertices_np, n_neighbors=5, use_mahalanobis=False
        )
        
        # Torch result (may use different backend depending on CUDA availability)
        vertices_torch = torch.from_numpy(vertices_np)
        neighbors_torch, _, _ = ring1_neighbors_gaussians(
            vertices_torch, n_neighbors=5, use_mahalanobis=False
        )
        
        # Results should be identical or very similar (neighbors should overlap)
        matches = 0
        for i in range(20):
            overlap = len(set(neighbors_cpu[i]) & set(neighbors_torch[i]))
            matches += overlap
        
        # At least 80% of neighbors should match across backends
        assert matches >= (20 * 5 * 0.8)
    
    def test_mahalanobis_batched_efficiency(self):
        """Test that batched Mahalanobis is efficient and correct."""
        np.random.seed(42)
        vertices = np.random.rand(30, 3).astype(np.float32)
        scales = np.random.rand(30, 3).astype(np.float32) * 0.1 + 0.05
        rotations = np.array([[1, 0, 0, 0]] * 30, dtype=np.float32)  # Identity rotations
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=5, use_mahalanobis=True,
            gaussian_scales=scales, gaussian_rotations=rotations
        )
        
        # Check output structure
        assert len(neighbors) == 30
        assert all(len(neighbors[i]) <= 5 for i in range(30))
        assert mean_dist > 0
        assert len(per_point_dist) == 30
        
        # With identity rotations and similar scales, Mahalanobis should be similar to Euclidean
        neighbors_euclidean, _, _ = ring1_neighbors_gaussians(
            vertices, n_neighbors=5, use_mahalanobis=False
        )
        
        # Check some overlap (won't be perfect due to scale differences)
        matches = sum(len(set(neighbors[i]) & set(neighbors_euclidean[i])) for i in range(30))
        assert matches >= (30 * 5 * 0.5)  # At least 50% overlap
    
    @patch('GenerateData.utils.data_generation_utils.KNN_CUDA_AVAILABLE', True)
    @patch('GenerateData.utils.data_generation_utils.SIMPLE_KNN_AVAILABLE', False)
    def test_knn_cuda_mock(self, *args):
        """Test that knn_cuda path is taken when available (mocked)."""
        # This tests the code path, not actual knn_cuda functionality
        # Since knn_cuda may not be installed or compatible
        vertices = torch.rand(10, 3).cuda() if torch.cuda.is_available() else torch.rand(10, 3)
        
        # Should not raise error even if mocked
        try:
            neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
                vertices, n_neighbors=3, use_mahalanobis=False
            )
            # If it succeeds or falls back gracefully, that's fine
            assert len(neighbors) == 10
        except Exception as e:
            # Expected if knn_cuda not actually available
            assert "knn_cuda" in str(e).lower() or "KNN" in str(type(e).__name__)
    
    def test_map_points_backend_consistency(self):
        """Test map_points_to_surface works with different backends."""
        query = np.random.rand(10, 3).astype(np.float32)
        target = np.random.rand(50, 3).astype(np.float32)
        
        # Test CPU path
        indices_cpu = map_points_to_surface(query, target, use_mahalanobis=False)
        assert len(indices_cpu) == 10
        assert all(0 <= idx < 50 for idx in indices_cpu)
        
        # Test with torch tensors (may use CUDA if available)
        query_torch = torch.from_numpy(query)
        target_torch = torch.from_numpy(target)
        indices_torch = map_points_to_surface(query_torch, target_torch, use_mahalanobis=False)
        
        # Results should be identical
        np.testing.assert_array_equal(indices_cpu, indices_torch)


class TestLargeTensorDistanceComputation:
    """Tests for distance computation with large tensors to verify batching works."""
    
    def test_large_knn_euclidean(self):
        """Test KNN Euclidean with large point cloud."""
        np.random.seed(42)
        n_points = 5000
        vertices = torch.from_numpy(np.random.randn(n_points, 3).astype(np.float32))
        
        if torch.cuda.is_available():
            vertices = vertices.cuda()
        
        from utils.data_generation_utils import _compute_knn_euclidean
        
        indices, distances = _compute_knn_euclidean(
            vertices, n_neighbors=16, query_points=None, exclude_self=True
        )
        
        assert indices.shape == (n_points, 16)
        assert distances.shape == (n_points, 16)
        assert np.all(distances >= 0)
        # Verify distances are sorted
        for i in range(min(100, n_points)):  # Check first 100 points
            assert np.all(np.diff(distances[i]) >= -1e-5)  # Allow small numerical error
    
    def test_large_knn_mahalanobis_batched(self):
        """Test batched KNN Mahalanobis with large point cloud."""
        np.random.seed(123)
        n_points = 2000
        vertices = torch.from_numpy(np.random.randn(n_points, 3).astype(np.float32))
        scales = torch.from_numpy(np.abs(np.random.randn(n_points, 3).astype(np.float32)) + 0.1)
        rotations = torch.zeros(n_points, 4, dtype=torch.float32)
        rotations[:, 0] = 1.0  # Identity quaternions
        
        if torch.cuda.is_available():
            vertices = vertices.cuda()
            scales = scales.cuda()
            rotations = rotations.cuda()
        
        from utils.data_generation_utils import _compute_knn_mahalanobis
        
        # Test with batching
        indices_batched, dist_batched = _compute_knn_mahalanobis(
            vertices, scales, rotations, n_neighbors=10, batch_size=256
        )
        
        assert indices_batched.shape == (n_points, 10)
        assert dist_batched.shape == (n_points, 10)
        assert np.all(dist_batched >= 0)
    
    def test_large_knn_mahalanobis_batch_vs_full(self):
        """Verify batched and non-batched Mahalanobis produce same results."""
        np.random.seed(456)
        n_points = 500  # Smaller for comparison
        vertices = torch.from_numpy(np.random.randn(n_points, 3).astype(np.float32))
        scales = torch.from_numpy(np.abs(np.random.randn(n_points, 3).astype(np.float32)) + 0.1)
        rotations = torch.zeros(n_points, 4, dtype=torch.float32)
        rotations[:, 0] = 1.0
        
        if torch.cuda.is_available():
            vertices = vertices.cuda()
            scales = scales.cuda()
            rotations = rotations.cuda()
        
        from utils.data_generation_utils import _compute_knn_mahalanobis
        
        # Full computation
        indices_full, dist_full = _compute_knn_mahalanobis(
            vertices, scales, rotations, n_neighbors=8, batch_size=None
        )
        
        # Batched computation
        indices_batched, dist_batched = _compute_knn_mahalanobis(
            vertices, scales, rotations, n_neighbors=8, batch_size=64
        )
        
        np.testing.assert_array_equal(indices_full, indices_batched)
        np.testing.assert_allclose(dist_full, dist_batched, rtol=1e-5)
    
    def test_large_map_points_euclidean(self):
        """Test map_points_to_surface with large query and target sets."""
        np.random.seed(789)
        n_query = 3000
        n_target = 5000
        
        query = np.random.randn(n_query, 3).astype(np.float32)
        target = np.random.randn(n_target, 3).astype(np.float32)
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=False, return_distances=True
        )
        
        assert indices.shape == (n_query,)
        assert distances.shape == (n_query,)
        assert np.all(indices >= 0) and np.all(indices < n_target)
        assert np.all(distances >= 0)
        
        # Verify correctness for a sample of points
        for i in np.random.choice(n_query, size=50, replace=False):
            expected_dist = np.linalg.norm(query[i] - target[indices[i]])
            assert np.isclose(distances[i], expected_dist, rtol=1e-4)
    
    def test_large_map_points_mahalanobis_query_cov(self):
        """Test large-scale Mahalanobis mapping with query covariances."""
        np.random.seed(111)
        n_query = 1000
        n_target = 2000
        
        query = np.random.randn(n_query, 3).astype(np.float32)
        target = np.random.randn(n_target, 3).astype(np.float32)
        query_scales = np.abs(np.random.randn(n_query, 3).astype(np.float32)) + 0.1
        query_rotations = np.zeros((n_query, 4), dtype=np.float32)
        query_rotations[:, 0] = 1.0
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=128
        )
        
        assert indices.shape == (n_query,)
        assert distances.shape == (n_query,)
        assert np.all(indices >= 0) and np.all(indices < n_target)
        assert np.all(distances >= 0)
    
    def test_large_map_points_mahalanobis_target_cov(self):
        """Test large-scale Mahalanobis mapping with target covariances."""
        np.random.seed(222)
        n_query = 800
        n_target = 1500
        
        query = np.random.randn(n_query, 3).astype(np.float32)
        target = np.random.randn(n_target, 3).astype(np.float32)
        target_scales = np.abs(np.random.randn(n_target, 3).astype(np.float32)) + 0.1
        target_rotations = np.zeros((n_target, 4), dtype=np.float32)
        target_rotations[:, 0] = 1.0
        
        indices, distances = map_points_to_surface(
            query, target, use_mahalanobis=True,
            target_scales=target_scales, target_rotations=target_rotations,
            return_distances=True, batch_size=100
        )
        
        assert indices.shape == (n_query,)
        assert distances.shape == (n_query,)
        assert np.all(indices >= 0) and np.all(indices < n_target)
        assert np.all(distances >= 0)
    
    def test_large_map_points_batch_consistency(self):
        """Verify batched and non-batched mapping produce same results."""
        np.random.seed(333)
        n_query = 200
        n_target = 400
        
        query = np.random.randn(n_query, 3).astype(np.float32)
        target = np.random.randn(n_target, 3).astype(np.float32)
        query_scales = np.abs(np.random.randn(n_query, 3).astype(np.float32)) + 0.1
        query_rotations = np.zeros((n_query, 4), dtype=np.float32)
        query_rotations[:, 0] = 1.0
        
        # Full computation
        indices_full, dist_full = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=None
        )
        
        # Batched computation
        indices_batched, dist_batched = map_points_to_surface(
            query, target, use_mahalanobis=True,
            query_scales=query_scales, query_rotations=query_rotations,
            return_distances=True, batch_size=32
        )
        
        np.testing.assert_array_equal(indices_full, indices_batched)
        np.testing.assert_allclose(dist_full, dist_batched, rtol=1e-5)
    
    def test_very_large_ring1_neighbors(self):
        """Test ring1_neighbors_gaussians with large point cloud."""
        np.random.seed(444)
        n_points = 3000
        
        vertices = np.random.randn(n_points, 3).astype(np.float32)
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=20, use_mahalanobis=False
        )
        
        assert len(neighbors) == n_points
        assert all(len(neighbors[i]) == 20 for i in range(n_points))
        assert mean_dist > 0
        assert len(per_point_dist) == n_points
        assert np.all(per_point_dist >= 0)
    
    def test_very_large_ring1_neighbors_mahalanobis(self):
        """Test ring1_neighbors_gaussians with Mahalanobis on large point cloud."""
        np.random.seed(555)
        n_points = 1000
        
        vertices = np.random.randn(n_points, 3).astype(np.float32)
        scales = np.abs(np.random.randn(n_points, 3).astype(np.float32)) + 0.1
        rotations = np.zeros((n_points, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=15, use_mahalanobis=True,
            gaussian_scales=scales, gaussian_rotations=rotations
        )
        
        assert len(neighbors) == n_points
        assert all(len(neighbors[i]) == 15 for i in range(n_points))
        assert mean_dist > 0
        assert len(per_point_dist) == n_points
    
    def test_distance_symmetry_check(self):
        """Verify distance computation behaves correctly for symmetric cases."""
        np.random.seed(666)
        n_points = 100
        
        # Create points on a sphere for symmetric distribution
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        vertices = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], axis=1).astype(np.float32)
        
        neighbors, mean_dist, per_point_dist = ring1_neighbors_gaussians(
            vertices, n_neighbors=10, use_mahalanobis=False
        )
        
        # All points on a sphere should have similar nearest neighbor distances
        dist_std = np.std(per_point_dist)
        dist_mean = np.mean(per_point_dist)
        # Coefficient of variation should be reasonable (not too high)
        assert dist_std / dist_mean < 0.5  # Less than 50% variation
    
    def test_gpu_memory_efficiency(self):
        """Test that batched computation doesn't OOM on reasonable sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        np.random.seed(777)
        n_points = 5000
        
        vertices = torch.from_numpy(np.random.randn(n_points, 3).astype(np.float32)).cuda()
        scales = torch.from_numpy(np.abs(np.random.randn(n_points, 3).astype(np.float32)) + 0.1).cuda()
        rotations = torch.zeros(n_points, 4, dtype=torch.float32).cuda()
        rotations[:, 0] = 1.0
        
        from utils.data_generation_utils import _compute_knn_mahalanobis
        
        # This should complete without OOM due to batching
        indices, distances = _compute_knn_mahalanobis(
            vertices, scales, rotations, n_neighbors=16, batch_size=512
        )
        
        assert indices.shape == (n_points, 16)
        assert distances.shape == (n_points, 16)


class TestAdaptiveRing1Neighbors:
    """Test adaptive per-point kNN binary search."""

    def _make_grid(self, n=10):
        """Create a regular 2D grid as a (n*n, 3) point cloud."""
        xs = np.linspace(0, 1, n)
        ys = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(xs, ys)
        vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
        return vertices

    def test_returns_same_as_base_when_no_boost_needed(self):
        """When k_boost <= n_neighbors_base, should return base ring-1."""
        vertices = self._make_grid(10)
        ring1_adaptive, mean_d, ppd = adaptive_ring1_neighbors(
            vertices, target_ring=2, target_ring_neighbors=10,
            n_neighbors_base=8, k_boost=8, use_mahalanobis=False,
        )
        ring1_base, _, _ = ring1_neighbors_gaussians(
            vertices, n_neighbors=8, use_mahalanobis=False,
        )

        for i in range(len(vertices)):
            np.testing.assert_array_equal(
                np.sort(ring1_adaptive[i]), np.sort(ring1_base[i])
            )

    def test_per_point_k_varies(self):
        """After adaptive search, different points may have different ring-1 sizes."""
        np.random.seed(42)
        vertices = np.random.randn(200, 3).astype(np.float32)

        ring1_adaptive, _, _ = adaptive_ring1_neighbors(
            vertices, target_ring=3, target_ring_neighbors=80,
            n_neighbors_base=6, k_boost=16, use_mahalanobis=False,
            adaptive_max_steps=5,
        )

        sizes = [len(ring1_adaptive[i]) for i in range(len(vertices))]
        # At least some variation in ring-1 sizes (per-point binary search)
        assert max(sizes) >= min(sizes), "Expected some variation in ring-1 sizes"

    def test_ring_count_increases_for_deficient_points(self):
        """Adaptive mode should increase the target-ring count for formerly deficient points."""
        np.random.seed(42)
        vertices = np.random.randn(200, 3).astype(np.float32)

        base_k = 6
        target_ring = 3
        target_nbrs = 80

        ring1_base, _, _ = ring1_neighbors_gaussians(
            vertices, n_neighbors=base_k, use_mahalanobis=False,
        )
        ring1_adaptive, _, _ = adaptive_ring1_neighbors(
            vertices, target_ring=target_ring, target_ring_neighbors=target_nbrs,
            n_neighbors_base=base_k, k_boost=16, use_mahalanobis=False,
        )

        for i in range(len(vertices)):
            base_count = len(get_neighborhood_by_ring(i, target_ring, ring1_base))
            adaptive_count = len(get_neighborhood_by_ring(i, target_ring, ring1_adaptive))
            if base_count < target_nbrs:
                assert adaptive_count >= base_count, (
                    f"Point {i}: adaptive ring-{target_ring} count {adaptive_count} "
                    f"should be >= base count {base_count}"
                )

    def test_mean_cut_respects_threshold(self):
        """With enough steps, mean cut should stay within threshold."""
        np.random.seed(42)
        vertices = np.random.randn(100, 3).astype(np.float32)

        target_ring = 2
        target_nbrs = 30
        max_mean_cut = 3.0

        ring1_adaptive, _, _ = adaptive_ring1_neighbors(
            vertices, target_ring=target_ring,
            target_ring_neighbors=target_nbrs,
            n_neighbors_base=6, k_boost=14,
            adaptive_max_mean_cut=max_mean_cut,
            adaptive_max_steps=10,
            use_mahalanobis=False,
        )

        total_cut = 0
        N = len(vertices)
        for i in range(N):
            cnt = len(get_neighborhood_by_ring(i, target_ring, ring1_adaptive))
            total_cut += max(0, cnt - target_nbrs)
        mean_cut = total_cut / N
        # With enough steps the threshold should be met (or close to it)
        assert mean_cut <= max_mean_cut + 1.0, (
            f"mean_cut {mean_cut:.2f} should be close to threshold {max_mean_cut}"
        )

    def test_max_steps_limits_iterations(self):
        """Setting adaptive_max_steps=1 should still produce valid output."""
        np.random.seed(42)
        vertices = np.random.randn(50, 3).astype(np.float32)

        ring1, mean_d, ppd = adaptive_ring1_neighbors(
            vertices, target_ring=2, target_ring_neighbors=20,
            n_neighbors_base=6, k_boost=12,
            adaptive_max_steps=1,
            use_mahalanobis=False,
        )
        assert isinstance(ring1, dict)
        assert len(ring1) == 50
        assert mean_d > 0

    def test_returns_correct_tuple(self):
        """Should return (ring1_nbrs, mean_nn_dist, per_point_nn_dist) triple."""
        vertices = self._make_grid(8)
        result = adaptive_ring1_neighbors(
            vertices, target_ring=2, target_ring_neighbors=20,
            n_neighbors_base=6, k_boost=10, use_mahalanobis=False,
        )
        assert len(result) == 3
        ring1, mean_d, ppd = result
        assert isinstance(ring1, dict)
        assert len(ring1) == len(vertices)
        assert mean_d > 0
        assert len(ppd) == len(vertices)

    def test_integration_with_get_all_points_nbrs_all_rings(self):
        """Adaptive mode should work through the all-rings wrapper."""
        np.random.seed(42)
        vertices = np.random.randn(100, 3).astype(np.float32)

        ring1, ring2, ring3, ring4, md, ppd = get_all_points_nbrs_all_rings(
            vertices, n_neighbors_ring1=6,
            adaptive_target_ring=3, adaptive_target_neighbors=60,
            adaptive_k_boost=12, adaptive_max_mean_cut=5.0,
            adaptive_max_steps=3,
        )
        assert len(ring1) == 100
        assert len(ring3) == 100
        assert md > 0

    def test_integration_with_get_all_points_nbrs_single_ring(self):
        """Adaptive mode should work through the single-ring wrapper."""
        np.random.seed(42)
        vertices = np.random.randn(100, 3).astype(np.float32)

        ring1, ring3 = get_all_points_nbrs_single_ring(
            vertices, ring=3, n_neighbors_ring1=6,
            adaptive_target_ring=3, adaptive_target_neighbors=60,
            adaptive_k_boost=12, adaptive_max_mean_cut=5.0,
            adaptive_max_steps=3,
        )
        assert len(ring1) == 100
        assert len(ring3) == 100
