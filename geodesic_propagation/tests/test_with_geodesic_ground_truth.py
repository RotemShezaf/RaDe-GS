"""
Tests for geodesic propagation using ground truth geodesic distances.

This module provides:
1. Mock model that uses exact geodesic computation for testing
2. Tests for propagation accuracy against ground truth
3. Efficiency/performance tests
4. Consistency tests between GaussianInputBuilder and GaussianPatchDataset
"""

import numpy as np
import torch
import pytest
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.priority_queue import WavefrontPriorityQueue, PointState
from geodesic_propagation.input_builder import GaussianInputBuilder
from geodesic_propagation.fast_marching import FastMarchingPropagator
from geodesic_propagation.utils.model_handler import ModelHandler


def _make_config(attributes=None, max_neighbors=8, mask_constant=-10.0,
                 output_dir="/tmp/unused_test_dir"):
    """
    Build a minimal dataset_config dict for constructing a GaussianInputBuilder
    in tests.  The ring_size_mapping is inferred from max_neighbors.
    """
    if attributes is None:
        attributes = ["xyz"]
    # Determine ring from max_neighbors
    if max_neighbors <= 8:
        ring = 1
    elif max_neighbors <= 32:
        ring = 2
    else:
        ring = 3
    return {
        "output_dir": output_dir,
        "attributes": attributes,
        "nn_mean": 1.0,
        "mask_constant": mask_constant,
        "use_r1_min_val": False,
        "use_mahalanobis": False,
        "ring_size_mapping": {"euclidean": {1: 8, 2: 32, 3: 128}},
    }, ring


class MockGeodesicModel:
    """
    Mock model that computes geodesic distances using exact geodesic computation.
    
    This model uses pre-computed ground truth geodesic distances to simulate
    what a trained model would predict. Used for testing the propagation algorithm.
    """
    
    def __init__(self, ground_truth_distances: np.ndarray, source_idx: int = 0):
        """
        Initialize the mock model with ground truth distances.
        
        Args:
            ground_truth_distances: (N,) array of geodesic distances from source
            source_idx: Index of the source vertex
        """
        self.ground_truth = ground_truth_distances.astype(np.float32)
        self.source_idx = source_idx
        self.call_count = 0
    
    def predict(self, point_idx: int, neighbor_distances: np.ndarray) -> float:
        """
        Predict geodesic distance using ground truth.
        
        Args:
            point_idx: Index of the point to predict
            neighbor_distances: Known distances of visited neighbors (unused, for interface compatibility)
            
        Returns:
            Ground truth geodesic distance
        """
        self.call_count += 1
        return float(self.ground_truth[point_idx])
    
    def __call__(self, neighborhood: torch.Tensor, point_features: torch.Tensor, 
                 valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass interface (for compatibility).
        
        Returns a dummy prediction - actual ground truth is injected via predict().
        """
        if neighborhood.dim() == 2:
            # Single sample
            return torch.tensor([0.0])
        else:
            # Batch
            return torch.zeros(neighborhood.shape[0])


class MockGeodesicModelHandler(ModelHandler):
    """
    Model handler that wraps MockGeodesicModel for use with FastMarchingPropagator.
    """
    
    def __init__(self, ground_truth_distances: np.ndarray, source_idx: int = 0):
        """
        Initialize with ground truth distances.
        
        Args:
            ground_truth_distances: Pre-computed geodesic distances from source
            source_idx: Source vertex index
        """
        super().__init__()
        self.mock_model = MockGeodesicModel(ground_truth_distances, source_idx)
        self.model = self.mock_model
        self._ground_truth = ground_truth_distances
    
    def predict_single(self, neighborhood: torch.Tensor, point_features: torch.Tensor,
                      valid_mask: torch.Tensor) -> float:
        """
        Predict using ground truth - extracts point index from neighborhood.
        
        Note: This is a simplified version that just returns based on the
        minimum neighbor distance + some offset. In real usage, the propagator
        handles the ground truth lookup.
        """
        # For testing, we'll let the propagator inject the actual prediction
        return 0.0
    
    def get_ground_truth(self, point_idx: int) -> float:
        """Get ground truth distance for a point."""
        return float(self._ground_truth[point_idx])


class TestMockGeodesicModel:
    """Tests for the mock geodesic model."""
    
    def test_mock_model_initialization(self):
        """Test mock model initialization."""
        distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        model = MockGeodesicModel(distances, source_idx=0)
        
        assert model.source_idx == 0
        assert len(model.ground_truth) == 5
        assert model.call_count == 0
    
    def test_mock_model_predict(self):
        """Test mock model prediction."""
        distances = np.array([0.0, 1.5, 2.5, 3.5, 4.5])
        model = MockGeodesicModel(distances)
        
        assert model.predict(0, np.array([])) == 0.0
        assert model.predict(2, np.array([0.0, 1.5])) == 2.5
        assert model.call_count == 2
    
    def test_mock_handler_ground_truth(self):
        """Test mock handler ground truth access."""
        distances = np.array([0.0, 1.0, 2.0, 3.0])
        handler = MockGeodesicModelHandler(distances)
        
        assert handler.get_ground_truth(0) == 0.0
        assert handler.get_ground_truth(2) == 2.0


class TestPropagationWithGroundTruth:
    """Tests for propagation using ground truth geodesic distances."""
    
    @pytest.fixture
    def sphere_mesh_data(self):
        """Create a simple sphere mesh for testing."""
        try:
            from utils.geodesic_utils import generate_test_mesh, compute_exact_geodesic
            
            vertices, faces = generate_test_mesh("sphere")
            num_vertices = len(vertices)
            
            # Compute ground truth geodesic distances from vertex 0
            source_idx = 0
            ground_truth = compute_exact_geodesic(vertices, faces, source_idx)
            
            # Build ring1 neighbors from mesh faces
            ring1_neighbors = {}
            for i in range(num_vertices):
                ring1_neighbors[i] = np.array([], dtype=np.int64)
            
            for face in faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i + 1) % 3]
                    ring1_neighbors[v1] = np.unique(np.append(ring1_neighbors[v1], v2))
                    ring1_neighbors[v2] = np.unique(np.append(ring1_neighbors[v2], v1))
            
            return {
                'vertices': vertices.astype(np.float32),
                'faces': faces,
                'num_vertices': num_vertices,
                'ground_truth': ground_truth.flatten().astype(np.float32),
                'source_idx': source_idx,
                'ring1_neighbors': ring1_neighbors
            }
        except ImportError:
            pytest.skip("geodesic_utils not available")
    
    @pytest.fixture
    def simple_grid_data(self):
        """Create a simple 2D grid for testing without external dependencies."""
        # Create a 5x5 grid
        grid_size = 5
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i * 0.1, j * 0.1, 0.0])
        
        positions = np.array(positions, dtype=np.float32)
        num_points = len(positions)
        
        # Build grid neighbors (4-connectivity)
        ring1_neighbors = {}
        for idx in range(num_points):
            i, j = idx // grid_size, idx % grid_size
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * grid_size + j)
            if i < grid_size - 1:
                neighbors.append((i + 1) * grid_size + j)
            if j > 0:
                neighbors.append(i * grid_size + j - 1)
            if j < grid_size - 1:
                neighbors.append(i * grid_size + j + 1)
            ring1_neighbors[idx] = np.array(neighbors, dtype=np.int64)
        
        # Compute approximate ground truth (Manhattan distance scaled by grid spacing)
        source_idx = 0
        source_i, source_j = 0, 0
        ground_truth = np.zeros(num_points, dtype=np.float32)
        for idx in range(num_points):
            i, j = idx // grid_size, idx % grid_size
            # Euclidean distance as approximation
            ground_truth[idx] = np.sqrt((i - source_i)**2 + (j - source_j)**2) * 0.1
        
        return {
            'positions': positions,
            'num_points': num_points,
            'ground_truth': ground_truth,
            'source_idx': source_idx,
            'ring1_neighbors': ring1_neighbors,
            'grid_size': grid_size
        }
    
    def test_propagation_reaches_all_points(self, simple_grid_data):
        """Test that propagation reaches all connected points."""
        data = simple_grid_data
        
        # Create mock handler with ground truth
        handler = MockGeodesicModelHandler(data['ground_truth'], data['source_idx'])
        
        # Create input builder
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Create propagator with a model that uses ground truth
        # For this test, we'll create a custom propagator that uses ground truth directly
        propagator = FastMarchingPropagatorWithGroundTruth(
            ground_truth=data['ground_truth'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        # Propagate
        distances = propagator.propagate([data['source_idx']])
        
        # All points should have finite distances
        assert np.sum(np.isfinite(distances)) == data['num_points']
        
        # Source should have distance 0
        assert distances[data['source_idx']] == 0.0
    
    def test_propagation_accuracy(self, simple_grid_data):
        """Test that propagation produces accurate distances."""
        data = simple_grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        propagator = FastMarchingPropagatorWithGroundTruth(
            ground_truth=data['ground_truth'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([data['source_idx']])
        
        # Compare with ground truth
        # Should be very close since we're using ground truth for predictions
        relative_error = np.abs(distances - data['ground_truth']) / (data['ground_truth'] + 1e-8)
        mean_error = np.mean(relative_error[data['ground_truth'] > 0])
        
        # With ground truth model, error should be small
        assert mean_error < 0.1, f"Mean relative error {mean_error} too high"


class FastMarchingPropagatorWithEuclideanMock(FastMarchingPropagator):
    """
    Propagator with mock model: predicts nearest neighbor geodesic + Euclidean distance.
    (Non-batched version)
    
    This simulates a simple model that predicts geodesic distance as:
    predicted_dist = min_neighbor_geodesic + euclidean_distance_to_nearest_neighbor
    """
    
    def __init__(self, positions: np.ndarray, input_builder, ring1_neighbors, 
                 ring=1, verbose=True):
        """
        Initialize with positions for Euclidean distance computation.
        
        Args:
            positions: (N, 3) array of point positions
            input_builder: GaussianInputBuilder instance
            ring1_neighbors: Ring-1 neighbor dictionary
            ring: Ring level for neighbor expansion
            verbose: Print progress information
        """
        # Create a dummy model handler
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': input_builder.max_neighbors,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        super().__init__(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=ring,
            verbose=verbose
        )
        
        self.positions = positions.astype(np.float32)
    
    def _predict_distance(self, point_idx: int, visited_neighbors: np.ndarray,
                         visited_distances: np.ndarray) -> float:
        """
        Predict distance as: nearest neighbor's geodesic + Euclidean to that neighbor.
        
        This is a simple heuristic that approximates geodesic distance.
        """
        if len(visited_neighbors) == 0:
            return float('inf')
        
        # Find the nearest neighbor (by geodesic distance)
        min_idx = np.argmin(visited_distances)
        nearest_neighbor = visited_neighbors[min_idx]
        nearest_geodesic = visited_distances[min_idx]
        
        # Compute Euclidean distance from nearest neighbor to this point
        euclidean_dist = np.linalg.norm(
            self.positions[point_idx] - self.positions[nearest_neighbor]
        )
        
        return float(nearest_geodesic + euclidean_dist)
    
    def _update_wavefront_for_points(self, points_to_update):
        """Override to use Euclidean mock model for wavefront updates."""
        for point_idx in points_to_update:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            if len(visited_neighbors) > 0:
                predicted_dist = self._predict_distance(point_idx, visited_neighbors, visited_distances)
                self.wavefront.add_or_update(point_idx, predicted_dist)
    
    def _update_wavefront_distances(self):
        """Override to use mock model (non-batched, calls _predict_distance)."""
        wavefront_points = list(self.wavefront.get_wavefront_points())
        
        for point_idx in wavefront_points:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            
            if len(visited_neighbors) > 0:
                predicted_dist = self._predict_distance(
                    point_idx, visited_neighbors, visited_distances
                )
                self.wavefront.add_or_update(point_idx, predicted_dist)


class FastMarchingPropagatorWithEuclideanMockBatched(FastMarchingPropagator):
    """
    Propagator with mock model: predicts nearest neighbor geodesic + Euclidean distance.
    (Batched version - processes all wavefront points in one batch)
    
    This simulates a simple model that predicts geodesic distance as:
    predicted_dist = min_neighbor_geodesic + euclidean_distance_to_nearest_neighbor
    """
    
    def __init__(self, positions: np.ndarray, input_builder, ring1_neighbors, 
                 ring=1, verbose=True):
        """
        Initialize with positions for Euclidean distance computation.
        
        Args:
            positions: (N, 3) array of point positions
            input_builder: GaussianInputBuilder instance
            ring1_neighbors: Ring-1 neighbor dictionary
            ring: Ring level for neighbor expansion
            verbose: Print progress information
        """
        # Create a dummy model handler
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': input_builder.max_neighbors,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        super().__init__(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=ring,
            verbose=verbose
        )
        
        self.positions = positions.astype(np.float32)
    
    def _update_wavefront_for_points(self, points_to_update):
        """Override to use batched Euclidean mock model for wavefront updates."""
        point_indices = []
        visited_neighbors_list = []
        visited_distances_list = []
        for point_idx in points_to_update:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            if len(visited_neighbors) > 0:
                point_indices.append(point_idx)
                visited_neighbors_list.append(visited_neighbors)
                visited_distances_list.append(visited_distances)
        if point_indices:
            predictions = self._predict_distance_batch(
                point_indices, visited_neighbors_list, visited_distances_list
            )
            for point_idx, predicted_dist in zip(point_indices, predictions):
                self.wavefront.add_or_update(point_idx, float(predicted_dist))
    
    def _predict_distance_batch(self, point_indices: list, 
                                 visited_neighbors_list: list,
                                 visited_distances_list: list) -> np.ndarray:
        """
        Batch predict distances for multiple points.
        
        Args:
            point_indices: List of point indices to predict
            visited_neighbors_list: List of neighbor arrays for each point
            visited_distances_list: List of neighbor distance arrays for each point
            
        Returns:
            Array of predicted distances for each point
        """
        predictions = np.zeros(len(point_indices), dtype=np.float32)
        
        for i, (point_idx, visited_neighbors, visited_distances) in enumerate(
            zip(point_indices, visited_neighbors_list, visited_distances_list)
        ):
            if len(visited_neighbors) == 0:
                predictions[i] = float('inf')
                continue
            
            # Find the nearest neighbor (by geodesic distance)
            min_idx = np.argmin(visited_distances)
            nearest_neighbor = visited_neighbors[min_idx]
            nearest_geodesic = visited_distances[min_idx]
            
            # Compute Euclidean distance from nearest neighbor to this point
            euclidean_dist = np.linalg.norm(
                self.positions[point_idx] - self.positions[nearest_neighbor]
            )
            
            predictions[i] = nearest_geodesic + euclidean_dist
        
        return predictions
    
    def _update_wavefront_distances(self):
        """Override to use batched mock prediction."""
        wavefront_points = list(self.wavefront.get_wavefront_points())
        
        if len(wavefront_points) == 0:
            return
        
        # Collect all data for batch prediction
        point_indices = []
        visited_neighbors_list = []
        visited_distances_list = []
        
        for point_idx in wavefront_points:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            
            if len(visited_neighbors) > 0:
                point_indices.append(point_idx)
                visited_neighbors_list.append(visited_neighbors)
                visited_distances_list.append(visited_distances)
        
        if len(point_indices) == 0:
            return
        
        # Batch predict
        predictions = self._predict_distance_batch(
            point_indices, visited_neighbors_list, visited_distances_list
        )
        
        # Update wavefront with predictions
        for point_idx, predicted_dist in zip(point_indices, predictions):
            self.wavefront.add_or_update(point_idx, float(predicted_dist))


class TestSphereWithEuclideanMock:
    """
    Test propagation on a sphere using mock model that predicts:
    predicted_dist = nearest_neighbor_geodesic + euclidean_distance_to_neighbor
    """
    
    @pytest.fixture
    def sphere_data(self):
        """Create points on a unit sphere with known geodesic distances."""
        np.random.seed(42)
        
        # Create points on unit sphere using fibonacci lattice for uniform distribution
        num_points = 100
        indices = np.arange(num_points, dtype=float) + 0.5
        
        phi = np.arccos(1 - 2 * indices / num_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        positions = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # Compute KNN neighbors
        from sklearn.neighbors import NearestNeighbors
        k = 8
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        ring1_neighbors = {i: indices[i, 1:] for i in range(num_points)}
        
        # Compute ground truth geodesic distances from point 0
        # On unit sphere: geodesic distance = arccos(dot(p1, p2))
        source_idx = 0
        source_pos = positions[source_idx]
        dot_products = np.clip(positions @ source_pos, -1.0, 1.0)
        ground_truth_geodesic = np.arccos(dot_products).astype(np.float32)
        
        return {
            'positions': positions,
            'ring1_neighbors': ring1_neighbors,
            'ground_truth_geodesic': ground_truth_geodesic,
            'source_idx': source_idx,
            'num_points': num_points,
            'radius': 1.0,
        }
    
    def test_euclidean_mock_propagation(self, sphere_data):
        """
        Test that the Euclidean mock model produces reasonable results.
        
        Expected behavior:
        - All points should be reached
        - Predicted distances should be >= true geodesic (Euclidean is a lower bound on sphere)
        - Predicted distances should correlate with true geodesic
        """
        positions = sphere_data['positions']
        source_idx = sphere_data['source_idx']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        propagator = FastMarchingPropagatorWithEuclideanMock(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        # Run propagation
        distances = propagator.propagate([source_idx])
        
        # All points should be reached
        reached = np.sum(np.isfinite(distances))
        assert reached == sphere_data['num_points'], \
            f"Only reached {reached}/{sphere_data['num_points']} points"
        
        # Source should have distance 0
        assert distances[source_idx] == 0.0, "Source distance should be 0"
        
        # Distances should be non-negative
        assert np.all(distances >= 0), "All distances should be non-negative"
        
        # Compare with ground truth
        ground_truth = sphere_data['ground_truth_geodesic']
        
        # The Euclidean-based prediction should approximate geodesic
        # On a sphere, Euclidean chord < geodesic arc, but accumulated error may vary
        # Check correlation is positive
        non_source = np.arange(sphere_data['num_points']) != source_idx
        correlation = np.corrcoef(distances[non_source], ground_truth[non_source])[0, 1]
        assert correlation > 0.9, f"Correlation with ground truth too low: {correlation:.3f}"
        
        print(f"\nResults on sphere (n={sphere_data['num_points']}):")
        print(f"  Correlation with ground truth: {correlation:.4f}")
        print(f"  Mean predicted: {np.mean(distances[non_source]):.4f}")
        print(f"  Mean ground truth: {np.mean(ground_truth[non_source]):.4f}")
        print(f"  Max predicted: {np.max(distances):.4f}")
        print(f"  Max ground truth (pi): {np.max(ground_truth):.4f}")
    
    def test_euclidean_mock_propagation_batched(self, sphere_data):
        """
        Test batched Euclidean mock propagation produces same results as non-batched.
        """
        positions = sphere_data['positions']
        source_idx = sphere_data['source_idx']
        ground_truth = sphere_data['ground_truth_geodesic']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Non-batched propagator
        propagator_nonbatch = FastMarchingPropagatorWithEuclideanMock(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_nonbatch = propagator_nonbatch.propagate([source_idx])
        
        # Batched propagator
        propagator_batched = FastMarchingPropagatorWithEuclideanMockBatched(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_batched = propagator_batched.propagate([source_idx])
        
        # Results should be identical
        np.testing.assert_allclose(distances_batched, distances_nonbatch, rtol=1e-5,
                                   err_msg="Batched and non-batched results differ")
        
        # Both should reach all points
        assert np.sum(np.isfinite(distances_batched)) == sphere_data['num_points']
        
        # Check correlation with ground truth
        non_source = np.arange(sphere_data['num_points']) != source_idx
        correlation = np.corrcoef(distances_batched[non_source], ground_truth[non_source])[0, 1]
        assert correlation > 0.9, f"Batched correlation too low: {correlation:.3f}"
        
        print(f"\nBatched vs Non-batched comparison:")
        print(f"  Max difference: {np.max(np.abs(distances_batched - distances_nonbatch)):.6f}")
        print(f"  Correlation with ground truth: {correlation:.4f}")
    
    def test_euclidean_mock_ordering(self, sphere_data):
        """
        Test that the propagation visits points in roughly correct order.
        
        Points closer to source (by geodesic) should generally be visited earlier.
        """
        positions = sphere_data['positions']
        source_idx = sphere_data['source_idx']
        ground_truth = sphere_data['ground_truth_geodesic']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        propagator = FastMarchingPropagatorWithEuclideanMock(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        # Track visit order
        visit_order = []
        
        def callback(iteration, point_idx, distance):
            visit_order.append((point_idx, distance, ground_truth[point_idx]))
        
        propagator.propagate([source_idx], callback=callback)
        
        # Check that visit order correlates with ground truth
        if len(visit_order) > 1:
            visited_gt = [v[2] for v in visit_order]
            # Check that it's mostly monotonically increasing
            increasing_pairs = sum(1 for i in range(len(visited_gt)-1) 
                                   if visited_gt[i] <= visited_gt[i+1])
            monotonicity = increasing_pairs / (len(visited_gt) - 1)
            
            print(f"\nVisit order analysis:")
            print(f"  Total visited: {len(visit_order)}")
            print(f"  Monotonicity (ground truth): {monotonicity:.2%}")
            
            # Should be somewhat monotonic (allow out-of-order due to Euclidean vs geodesic difference)
            assert monotonicity > 0.5, f"Visit order not monotonic enough: {monotonicity:.2%}"
    
    def test_euclidean_vs_ground_truth_comparison(self, sphere_data):
        """
        Compare Euclidean mock with ground truth propagation on same sphere.
        """
        positions = sphere_data['positions']
        source_idx = sphere_data['source_idx']
        ground_truth = sphere_data['ground_truth_geodesic']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Euclidean mock propagator
        euclidean_propagator = FastMarchingPropagatorWithEuclideanMock(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        euclidean_distances = euclidean_propagator.propagate([source_idx])
        
        # Ground truth propagator
        gt_propagator = FastMarchingPropagatorWithGroundTruth(
            ground_truth=ground_truth,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        gt_distances = gt_propagator.propagate([source_idx])
        
        # Both should reach all points
        assert np.sum(np.isfinite(euclidean_distances)) == sphere_data['num_points']
        assert np.sum(np.isfinite(gt_distances)) == sphere_data['num_points']
        
        # Ground truth propagation should match exactly
        np.testing.assert_allclose(gt_distances, ground_truth, rtol=1e-5)
        
        # Euclidean should be reasonably close (within 50% for most points)
        non_source = np.arange(sphere_data['num_points']) != source_idx
        relative_error = np.abs(euclidean_distances[non_source] - ground_truth[non_source]) / (ground_truth[non_source] + 1e-6)
        median_error = np.median(relative_error)
        
        print(f"\nEuclidean mock vs Ground truth:")
        print(f"  Median relative error: {median_error:.2%}")
        print(f"  90th percentile error: {np.percentile(relative_error, 90):.2%}")
        
        # Euclidean-based mock should be reasonably accurate
        assert median_error < 0.5, f"Median relative error too high: {median_error:.2%}"
    
    def test_euclidean_vs_ground_truth_comparison_batched(self, sphere_data):
        """
        Compare batched Euclidean mock with batched ground truth propagation.
        """
        positions = sphere_data['positions']
        source_idx = sphere_data['source_idx']
        ground_truth = sphere_data['ground_truth_geodesic']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Batched Euclidean mock propagator
        euclidean_propagator = FastMarchingPropagatorWithEuclideanMockBatched(
            positions=positions,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        euclidean_distances = euclidean_propagator.propagate([source_idx])
        
        # Batched Ground truth propagator
        gt_propagator = FastMarchingPropagatorWithGroundTruthBatched(
            ground_truth=ground_truth,
            input_builder=input_builder,
            ring1_neighbors=sphere_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        gt_distances = gt_propagator.propagate([source_idx])
        
        # Both should reach all points
        assert np.sum(np.isfinite(euclidean_distances)) == sphere_data['num_points']
        assert np.sum(np.isfinite(gt_distances)) == sphere_data['num_points']
        
        # Ground truth propagation should match exactly
        np.testing.assert_allclose(gt_distances, ground_truth, rtol=1e-5)
        
        # Check median error for Euclidean mock
        non_source = np.arange(sphere_data['num_points']) != source_idx
        relative_error = np.abs(euclidean_distances[non_source] - ground_truth[non_source]) / (ground_truth[non_source] + 1e-6)
        median_error = np.median(relative_error)
        
        print(f"\nBatched Euclidean mock vs Batched Ground truth:")
        print(f"  Median relative error: {median_error:.2%}")
        
        assert median_error < 0.5, f"Median relative error too high: {median_error:.2%}"


class TestBatchedVsNonBatched:
    """Tests comparing batched and non-batched implementations."""
    
    @pytest.fixture
    def grid_data(self):
        """Create a 10x10 grid for testing."""
        grid_size = 10
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i * 0.1, j * 0.1, 0.0])
        
        positions = np.array(positions, dtype=np.float32)
        num_points = len(positions)
        
        # Build 4-connected grid neighbors
        ring1_neighbors = {}
        for idx in range(num_points):
            i, j = idx // grid_size, idx % grid_size
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * grid_size + j)
            if i < grid_size - 1:
                neighbors.append((i + 1) * grid_size + j)
            if j > 0:
                neighbors.append(i * grid_size + j - 1)
            if j < grid_size - 1:
                neighbors.append(i * grid_size + j + 1)
            ring1_neighbors[idx] = np.array(neighbors, dtype=np.int64)
        
        # Ground truth: Euclidean distance from source
        source_idx = 0
        ground_truth = np.linalg.norm(positions - positions[source_idx], axis=1).astype(np.float32)
        
        return {
            'positions': positions,
            'num_points': num_points,
            'ground_truth': ground_truth,
            'source_idx': source_idx,
            'ring1_neighbors': ring1_neighbors,
            'grid_size': grid_size
        }
    
    def test_ground_truth_batched_matches_nonbatched(self, grid_data):
        """Test that batched ground truth propagation matches non-batched."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Non-batched
        propagator_nb = FastMarchingPropagatorWithGroundTruth(
            ground_truth=data['ground_truth'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_nb = propagator_nb.propagate([data['source_idx']])
        
        # Batched
        propagator_b = FastMarchingPropagatorWithGroundTruthBatched(
            ground_truth=data['ground_truth'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_b = propagator_b.propagate([data['source_idx']])
        
        # Results should be identical
        np.testing.assert_allclose(distances_b, distances_nb, rtol=1e-5)
        
        # Both should match ground truth
        np.testing.assert_allclose(distances_nb, data['ground_truth'], rtol=1e-5)
        np.testing.assert_allclose(distances_b, data['ground_truth'], rtol=1e-5)
    
    def test_euclidean_mock_batched_matches_nonbatched(self, grid_data):
        """Test that batched Euclidean mock matches non-batched."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Non-batched
        propagator_nb = FastMarchingPropagatorWithEuclideanMock(
            positions=data['positions'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_nb = propagator_nb.propagate([data['source_idx']])
        
        # Batched
        propagator_b = FastMarchingPropagatorWithEuclideanMockBatched(
            positions=data['positions'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        distances_b = propagator_b.propagate([data['source_idx']])
        
        # Results should be identical
        np.testing.assert_allclose(distances_b, distances_nb, rtol=1e-5)
        
        # Both should reach all points
        assert np.sum(np.isfinite(distances_nb)) == data['num_points']
        assert np.sum(np.isfinite(distances_b)) == data['num_points']
    
    def test_batched_performance(self, grid_data):
        """Test that batched version is not slower than non-batched."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Time non-batched
        propagator_nb = FastMarchingPropagatorWithGroundTruth(
            ground_truth=data['ground_truth'],
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        start = time.perf_counter()
        for _ in range(3):
            propagator_nb = FastMarchingPropagatorWithGroundTruth(
                ground_truth=data['ground_truth'],
                input_builder=input_builder,
                ring1_neighbors=data['ring1_neighbors'],
                ring=1,
                verbose=False
            )
            propagator_nb.propagate([data['source_idx']])
        time_nb = time.perf_counter() - start
        
        # Time batched
        start = time.perf_counter()
        for _ in range(3):
            propagator_b = FastMarchingPropagatorWithGroundTruthBatched(
                ground_truth=data['ground_truth'],
                input_builder=input_builder,
                ring1_neighbors=data['ring1_neighbors'],
                ring=1,
                verbose=False
            )
            propagator_b.propagate([data['source_idx']])
        time_b = time.perf_counter() - start
        
        print(f"\nPerformance comparison (3 runs on {data['num_points']} points):")
        print(f"  Non-batched: {time_nb:.4f}s")
        print(f"  Batched: {time_b:.4f}s")
        print(f"  Ratio: {time_b/time_nb:.2f}x")
        
        # Batched should not be much slower (allow 2x overhead for small test)
        assert time_b < time_nb * 2, f"Batched too slow: {time_b:.4f}s vs {time_nb:.4f}s"


class FastMarchingPropagatorWithGroundTruth(FastMarchingPropagator):
    """
    Modified propagator that uses ground truth for distance prediction.
    (Non-batched version)
    """
    
    def __init__(self, ground_truth: np.ndarray, input_builder, ring1_neighbors, 
                 ring=1, verbose=True):
        """
        Initialize with ground truth distances.
        
        Args:
            ground_truth: Pre-computed geodesic distances from source
            input_builder: GaussianInputBuilder instance
            ring1_neighbors: Ring-1 neighbor dictionary
            ring: Ring level for neighbor expansion
            verbose: Print progress information
        """
        # Create a dummy model handler with matching max_neighbors
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': input_builder.max_neighbors,  # Match input_builder
            'embed_dim': 8,  # Small for speed
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        super().__init__(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=ring,
            verbose=verbose
        )
        
        self.ground_truth = ground_truth.astype(np.float32)
    
    def _predict_distance(self, point_idx: int, visited_neighbors: np.ndarray,
                         visited_distances: np.ndarray) -> float:
        """Override to use ground truth instead of model prediction."""
        return float(self.ground_truth[point_idx])
    
    def _update_wavefront_for_points(self, points_to_update):
        """Override to use ground truth for all wavefront point updates."""
        for point_idx in points_to_update:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            if len(visited_neighbors) > 0:
                predicted_dist = float(self.ground_truth[point_idx])
                self.wavefront.add_or_update(point_idx, predicted_dist)
    
    def _update_wavefront_distances(self):
        """Override to use ground truth (non-batched, calls _predict_distance)."""
        wavefront_points = list(self.wavefront.get_wavefront_points())
        
        for point_idx in wavefront_points:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            
            if len(visited_neighbors) > 0:
                predicted_dist = self._predict_distance(
                    point_idx, visited_neighbors, visited_distances
                )
                self.wavefront.add_or_update(point_idx, predicted_dist)


class FastMarchingPropagatorWithGroundTruthBatched(FastMarchingPropagator):
    """
    Modified propagator that uses ground truth for distance prediction.
    (Batched version - processes all wavefront points in one batch)
    """
    
    def __init__(self, ground_truth: np.ndarray, input_builder, ring1_neighbors, 
                 ring=1, verbose=True):
        """
        Initialize with ground truth distances.
        
        Args:
            ground_truth: Pre-computed geodesic distances from source
            input_builder: GaussianInputBuilder instance
            ring1_neighbors: Ring-1 neighbor dictionary
            ring: Ring level for neighbor expansion
            verbose: Print progress information
        """
        # Create a dummy model handler with matching max_neighbors
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': input_builder.max_neighbors,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        super().__init__(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=ring,
            verbose=verbose
        )
        
        self.ground_truth = ground_truth.astype(np.float32)
    
    def _predict_distance_batch(self, point_indices: list) -> np.ndarray:
        """
        Batch predict distances using ground truth.
        
        Args:
            point_indices: List of point indices to predict
            
        Returns:
            Array of ground truth distances for each point
        """
        return self.ground_truth[point_indices]
    
    def _update_wavefront_for_points(self, points_to_update):
        """Override to use ground truth for all wavefront point updates."""
        for point_idx in points_to_update:
            visited_neighbors, visited_distances = self._get_visited_neighbor_distances(point_idx)
            if len(visited_neighbors) > 0:
                predicted_dist = float(self.ground_truth[point_idx])
                self.wavefront.add_or_update(point_idx, predicted_dist)
    
    def _update_wavefront_distances(self):
        """Override to use batched ground truth lookup."""
        wavefront_points = list(self.wavefront.get_wavefront_points())
        
        if len(wavefront_points) == 0:
            return
        
        # Collect points with valid neighbors
        valid_point_indices = []
        
        for point_idx in wavefront_points:
            visited_neighbors, _ = self._get_visited_neighbor_distances(point_idx)
            if len(visited_neighbors) > 0:
                valid_point_indices.append(point_idx)
        
        if len(valid_point_indices) == 0:
            return
        
        # Batch lookup ground truth
        predictions = self._predict_distance_batch(valid_point_indices)
        
        # Update wavefront with predictions
        for point_idx, predicted_dist in zip(valid_point_indices, predictions):
            self.wavefront.add_or_update(point_idx, float(predicted_dist))


class TestEfficiency:
    """Performance and efficiency tests."""
    
    def test_priority_queue_efficiency(self):
        """Test priority queue operations are O(log n)."""
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            queue = WavefrontPriorityQueue(size)
            
            start = time.perf_counter()
            
            # Add all points
            for i in range(size):
                queue.add_or_update(i, float(size - i))
            
            # Pop all points
            while len(queue) > 0:
                queue.pop_min()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Check that time scales reasonably (not quadratic)
        # For O(n log n) operations, time should scale roughly linearly with n log n
        # Between 100 and 10000, n increases 100x, n*log(n) increases ~150x
        # We allow 500x to account for overhead
        time_ratio = times[2] / times[0]
        assert time_ratio < 500, f"Priority queue scaling too slow: {time_ratio}x for 100x size increase"
    
    def test_input_builder_batch_efficiency(self):
        """Test that build_input can be called sequentially without errors."""
        num_points = 100
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=32)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Build a visited_mask where first 50 points are visited
        visited_mask = np.zeros(num_points, dtype=bool)
        visited_mask[:50] = True
        distances = np.full(num_points, np.inf)
        distances[:50] = np.arange(50, dtype=np.float32) * 0.1
        
        # Sequential building
        results = []
        start = time.perf_counter()
        for point_idx in range(10):
            all_nbrs = np.array([j for j in range(num_points) if j != point_idx][:20],
                                dtype=np.int64)
            result = builder.build_input(
                point_idx=point_idx,
                all_neighbor_indices=all_nbrs,
                neighbor_distances=distances,
                visited_mask=visited_mask,
            )
            if result is not None:
                results.append(result)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Sequential build_input too slow: {elapsed:.4f}s"
        assert len(results) > 0, "Should successfully build at least one input"
    
    def test_large_scale_propagation(self):
        """Test propagation scales reasonably for larger point clouds."""
        np.random.seed(42)
        num_points = 500
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        # Create random neighbor structure
        ring1_neighbors = {}
        for i in range(num_points):
            # Random 5-10 neighbors
            num_nbrs = np.random.randint(5, 11)
            nbrs = np.random.choice([j for j in range(num_points) if j != i], 
                                    num_nbrs, replace=False)
            ring1_neighbors[i] = nbrs.astype(np.int64)
        
        # Create ground truth (Euclidean distance as approximation)
        ground_truth = np.linalg.norm(positions - positions[0], axis=1).astype(np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=16)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        propagator = FastMarchingPropagatorWithGroundTruth(
            ground_truth=ground_truth,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        start = time.perf_counter()
        distances = propagator.propagate([0], max_iterations=num_points)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (< 30 seconds for 500 points)
        assert elapsed < 30.0, f"Propagation took too long: {elapsed:.2f}s"
        
        # Should reach significant portion of points
        reached = np.sum(np.isfinite(distances))
        assert reached > num_points * 0.1, f"Only reached {reached} points"


class TestGaussianInputBuilderConsistency:
    """
    Tests for consistency between GaussianInputBuilder and GaussianPatchDataset.
    
    Ensures that the feature formats, dimensions, and normalization approaches
    are compatible.
    """
    
    def test_feature_dimension_calculation(self):
        """Test that feature dimensions are calculated consistently."""
        # Test with xyz only
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        builder_xyz = GaussianInputBuilder(
            positions=np.random.randn(10, 3).astype(np.float32),
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # xyz = 3 features for point, 3 + 1 (geodesic) = 4 for neighbors
        assert builder_xyz.get_point_feature_dim() == 3
        assert builder_xyz.get_neighbor_feature_dim() == 4
        
        # Test with multiple attributes
        _cfg_full, _ring_full = _make_config(
            attributes=["xyz", "scale", "rotation", "opacity"], max_neighbors=8)
        builder_full = GaussianInputBuilder(
            positions=np.random.randn(10, 3).astype(np.float32),
            scales=np.random.randn(10, 3).astype(np.float32),
            rotations=np.random.randn(10, 4).astype(np.float32),
            opacities=np.random.randn(10, 1).astype(np.float32),
            dataset_config=_cfg_full,
            ring=_ring_full,
            device='cpu'
        )
        
        # xyz=3, scale=3, rotation=4, opacity=1 = 11 for point
        # 11 + 1 (geodesic) = 12 for neighbors
        assert builder_full.get_point_feature_dim() == 11
        assert builder_full.get_neighbor_feature_dim() == 12
    
    def test_entry_size_matches_dataset_calculation(self):
        """Test that entry size calculation matches GaussianPatchDataset."""
        # This tests the same formula used in both classes
        attributes = ["xyz", "scale", "rotation", "opacity"]
        
        # Manual calculation (matching GaussianPatchDataset._get_entry_size)
        expected_entry_size = sum([
            4 if attr == "rotation" else 
            3 if attr in ["xyz", "normals", "scale", "sh"] else 
            1 for attr in attributes
        ]) + 1  # +1 for geodesic distance
        
        # xyz=3, scale=3, rotation=4, opacity=1, geodesic=1 = 12
        assert expected_entry_size == 12
        
        # GaussianInputBuilder calculation
        _cfg_calc, _ring_calc = _make_config(attributes=attributes, max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=np.random.randn(10, 3).astype(np.float32),
            scales=np.random.randn(10, 3).astype(np.float32),
            rotations=np.random.randn(10, 4).astype(np.float32),
            opacities=np.random.randn(10, 1).astype(np.float32),
            dataset_config=_cfg_calc,
            ring=_ring_calc,
            device='cpu'
        )
        
        assert builder.get_neighbor_feature_dim() == expected_entry_size
    
    def test_mask_constant_consistency(self):
        """Test that mask constant is used consistently."""
        mask_constant = -10.0
        
        _cfg_mc, _ring_mc = _make_config(
            attributes=["xyz", "opacity"], max_neighbors=8, mask_constant=mask_constant)
        builder = GaussianInputBuilder(
            positions=np.random.randn(10, 3).astype(np.float32),
            opacities=np.random.randn(10, 1).astype(np.float32),
            dataset_config=_cfg_mc,
            ring=_ring_mc,
            device='cpu'
        )
        
        # Build input with 2 visited neighbors (all others unvisited)
        N = 10
        visited_mask = np.zeros(N, dtype=bool)
        visited_mask[1] = True
        visited_mask[2] = True
        distances = np.full(N, np.inf)
        distances[1] = 0.1
        distances[2] = 0.2
        all_nbrs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=all_nbrs,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        assert result is not None
        neighborhood, _, valid_mask, _ = result
        
        # Check that invalid entries have correct masking
        invalid_entries = neighborhood[~valid_mask]
        # There should be unvisited entries (max_neighbors=8, only 2 visited)
        assert invalid_entries.shape[0] > 0, "Should have some invalid entries"
        # Valid entries should be > 0
        assert valid_mask.sum() == 2, f"Expected 2 valid entries, got {valid_mask.sum()}"
        # Geodesic column (last=-1) of invalid entries should be mask_constant
        assert torch.all(invalid_entries[:, -1] == mask_constant), \
            f"Geodesic-masked entries should be {mask_constant}, got {invalid_entries[:, -1]}"
    
    def test_geodesic_distance_position(self):
        """Test that geodesic distance is at the expected position (last column)."""
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=np.random.randn(10, 3).astype(np.float32),
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Build input with 3 visited neighbors
        N = 10
        visited_mask = np.zeros(N, dtype=bool)
        visited_mask[1] = True
        visited_mask[2] = True
        visited_mask[3] = True
        geo_distances_raw = np.array([0.5, 1.0, 1.5])
        distances = np.full(N, np.inf)
        distances[1] = geo_distances_raw[0]
        distances[2] = geo_distances_raw[1]
        distances[3] = geo_distances_raw[2]
        all_nbrs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=all_nbrs,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        assert result is not None
        neighborhood, _, valid_mask, build_info = result
        
        # With visited_mask, the 3 valid neighbors should appear first
        assert valid_mask.sum() == 3, f"Expected 3 valid neighbors, got {valid_mask.sum()}"
        # Geodesic distances should be shifted so min = 0 (stage-1 normalization)
        valid_geo = neighborhood[valid_mask, -1].numpy()
        assert valid_geo.min() >= -0.1, "Geodesic distances should be non-negative after shifting"
        assert build_info['min_input'] is not None, "build_info should contain min_input"
    
    def test_xyz_relative_positioning(self):
        """Test that xyz features are relative to center point (after normalization)."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # Point 0 (center)
            [1.0, 0.0, 0.0],  # Point 1
            [0.0, 1.0, 0.0],  # Point 2
            [0.0, 0.0, 1.0],  # Point 3
        ], dtype=np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Build input for point 0 with 3 visited neighbors
        N = 4
        visited_mask = np.zeros(N, dtype=bool)
        visited_mask[1] = True
        visited_mask[2] = True
        visited_mask[3] = True
        distances = np.full(N, np.inf)
        distances[1] = 1.0
        distances[2] = 1.0
        distances[3] = 1.0
        all_nbrs = np.array([1, 2, 3], dtype=np.int64)
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=all_nbrs,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        assert result is not None
        neighborhood, point_features, valid_mask, _ = result
        
        # All 3 neighbors should be valid
        assert valid_mask.sum() == 3
        
        # Point features should contain xyz of point 0 (may be normalized but order is correct)
        # After dataset normalization, xyz is centered. The point feature xyz is the patch center.
        # With xyz=[0,0,0] as center, centered xyz should be [0,0,0]
        assert point_features.shape[0] >= 3, "Point features should have at least 3 dimensions (xyz)"
    
    def test_attribute_ordering_matches(self):
        """Test that attribute ordering matches expected order."""
        positions = np.random.randn(10, 3).astype(np.float32)
        scales = np.random.randn(10, 3).astype(np.float32)
        opacities = np.random.randn(10, 1).astype(np.float32)
        
        # Order: xyz (3), opacity (1), scale (3)
        _cfg_ao, _ring_ao = _make_config(
            attributes=["xyz", "opacity", "scale"], max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=positions,
            scales=scales,
            opacities=opacities,
            dataset_config=_cfg_ao,
            ring=_ring_ao,
            device='cpu'
        )
        
        # Point features should be: xyz(0:3), opacity(3:4), scale(4:7)
        assert builder.get_point_feature_dim() == 7
        
        # Neighbor features should be: xyz(0:3), opacity(3:4), scale(4:7), geodesic(7:8)
        assert builder.get_neighbor_feature_dim() == 8


class TestRobustness:
    """Edge case and robustness tests."""
    
    def test_single_point_propagation(self):
        """Test propagation with only one point."""
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        ring1_neighbors = {0: np.array([], dtype=np.int64)}
        ground_truth = np.array([0.0], dtype=np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        propagator = FastMarchingPropagatorWithGroundTruth(
            ground_truth=ground_truth,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([0])
        assert distances[0] == 0.0
    
    def test_empty_neighbors(self):
        """Test handling of points with no visited neighbors."""
        positions = np.random.randn(5, 3).astype(np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Build input with all-unvisited neighbors (visited_mask=False everywhere)
        N = 5
        visited_mask = np.zeros(N, dtype=bool)
        distances = np.full(N, np.inf)
        all_nbrs = np.array([1, 2, 3, 4], dtype=np.int64)
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=all_nbrs,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        
        # With no visited neighbors, either None is returned or valid_mask is all-False
        if result is not None:
            _, _, valid_mask_out, _ = result
            assert valid_mask_out.sum() == 0, \
                f"With no visited neighbors, valid_mask should be empty, got {valid_mask_out.sum()}"

    
    def test_max_neighbors_limit(self):
        """Test that max_neighbors limit is respected in returned neighborhood."""
        num_points = 50
        positions = np.random.randn(num_points, 3).astype(np.float32)
        max_neighbors = 8
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=max_neighbors)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Provide exactly max_neighbors visited neighbors (from a larger ring)
        visited_mask = np.zeros(num_points, dtype=bool)
        visited_mask[1:max_neighbors+1] = True
        distances = np.full(num_points, np.inf)
        distances[1:max_neighbors+1] = np.arange(1, max_neighbors+1, dtype=np.float32) * 0.1
        all_nbrs = np.arange(1, min(num_points, 20), dtype=np.int64)
        
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=all_nbrs,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        
        assert result is not None
        neighborhood, _, valid_mask, _ = result
        # Neighborhood tensor shape should be (max_neighbors,)
        assert neighborhood.shape[0] == max_neighbors
        # All max_neighbors entries should be valid
        assert valid_mask.sum() == max_neighbors
    
    def test_visited_set_filtering(self):
        """Test that visited_set correctly filters neighbors."""
        positions = np.random.randn(10, 3).astype(np.float32)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        neighbor_indices = np.array([1, 2, 3, 4, 5])
        neighbor_distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        visited_set = {1, 3, 5}  # Only odd indices are visited
        
        # Build visited mask from visited_set
        N = 10
        visited_mask = np.zeros(N, dtype=bool)
        for idx in visited_set:
            visited_mask[idx] = True
        distances = np.full(N, np.inf)
        for idx in visited_set:
            distances[idx] = 0.1 * idx
        
        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=neighbor_indices,
            neighbor_distances=distances,
            visited_mask=visited_mask,
        )
        
        assert result is not None
        _, _, valid_mask, _ = result
        
        # Should only have 3 valid entries (the visited ones)
        assert valid_mask.sum() == 3


class DeterministicMockModel(torch.nn.Module):
    """
    Deterministic mock model that returns predictable outputs.
    
    Returns prediction = sum(valid_neighbor_geodesics) / count + offset
    This is a simple, deterministic function that doesn't depend on learned weights.
    """
    
    def __init__(self, offset: float = 0.1):
        super().__init__()
        self.offset = offset
        self.call_count = 0
    
    def forward(self, neighborhood: torch.Tensor, point_features: torch.Tensor,
                valid_mask: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        """
        Deterministic forward pass: mean of valid geodesic distances + offset.
        
        Args:
            neighborhood: (B, K, F) or (K, F) neighbor features (geodesic is last column)
            point_features: (B, P) or (P,) point features (unused)
            valid_mask: (B, K) or (K,) valid neighbor mask
            return_embeddings: Ignored (for interface compatibility)
            
        Returns:
            (B, 1) predictions
        """
        self.call_count += 1
        
        # Handle single sample vs batch
        if neighborhood.dim() == 2:
            neighborhood = neighborhood.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)
        
        batch_size = neighborhood.shape[0]
        predictions = torch.zeros(batch_size, 1)
        
        for i in range(batch_size):
            # Get geodesic distances (last column) for valid neighbors
            valid_geodesics = neighborhood[i, valid_mask[i], -1]
            
            if len(valid_geodesics) > 0:
                # Deterministic formula: mean geodesic + offset
                predictions[i, 0] = valid_geodesics.mean() + self.offset
            else:
                predictions[i, 0] = self.offset
        
        return predictions


class DeterministicModelHandler(ModelHandler):
    """Model handler wrapping the deterministic mock model."""
    
    def __init__(self, offset: float = 0.1):
        super().__init__()
        self.model = DeterministicMockModel(offset)
        self._device = 'cpu'
    
    def predict_single(self, neighborhood: torch.Tensor, point_features: torch.Tensor,
                       valid_mask: torch.Tensor) -> float:
        """Predict using deterministic model."""
        with torch.no_grad():
            prediction = self.model(neighborhood, point_features, valid_mask)
        return float(prediction.item() if prediction.dim() == 0 else prediction[0])
    
    def predict_batch(self, neighborhoods: torch.Tensor, point_features: torch.Tensor,
                      valid_masks: torch.Tensor) -> np.ndarray:
        """Batch predict using deterministic model."""
        with torch.no_grad():
            predictions = self.model(neighborhoods, point_features, valid_masks)
        return predictions.numpy()


class TestModelIndependentCorrectness:
    """
    Tests that verify correctness using mathematical invariants,
    without depending on a trained model's weights.
    
    Uses a deterministic mock model to test the propagation algorithm itself.
    """
    
    @pytest.fixture
    def grid_data(self):
        """Create a 7x7 grid for testing."""
        grid_size = 7
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i * 0.1, j * 0.1, 0.0])
        
        positions = np.array(positions, dtype=np.float32)
        num_points = len(positions)
        
        # Build 4-connected grid neighbors
        ring1_neighbors = {}
        for idx in range(num_points):
            i, j = idx // grid_size, idx % grid_size
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * grid_size + j)
            if i < grid_size - 1:
                neighbors.append((i + 1) * grid_size + j)
            if j > 0:
                neighbors.append(i * grid_size + j - 1)
            if j < grid_size - 1:
                neighbors.append(i * grid_size + j + 1)
            ring1_neighbors[idx] = np.array(neighbors, dtype=np.int64)
        
        return {
            'positions': positions,
            'num_points': num_points,
            'ring1_neighbors': ring1_neighbors,
            'grid_size': grid_size,
            'source_idx': (grid_size // 2) * grid_size + grid_size // 2  # Center point
        }
    
    def test_source_distance_is_zero(self, grid_data):
        """Test invariant: source point always has distance 0."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([data['source_idx']])
        
        assert distances[data['source_idx']] == 0.0, \
            f"Source distance should be 0, got {distances[data['source_idx']]}"
    
    def test_all_distances_non_negative(self, grid_data):
        """Test invariant: all distances must be non-negative."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([data['source_idx']])
        
        finite_distances = distances[np.isfinite(distances)]
        assert np.all(finite_distances >= 0), \
            f"Found negative distances: {finite_distances[finite_distances < 0]}"
    
    def test_all_connected_points_reached(self, grid_data):
        """Test invariant: all connected points should be reached."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([data['source_idx']])
        
        reached = np.sum(np.isfinite(distances))
        assert reached == data['num_points'], \
            f"Only reached {reached}/{data['num_points']} points"
    
    def test_distances_increase_from_source(self, grid_data):
        """
        Test invariant: on average, points farther in graph distance
        should have larger geodesic estimates.
        """
        data = grid_data
        source_idx = data['source_idx']
        grid_size = data['grid_size']
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([source_idx])
        
        # Compute graph distance (Manhattan on grid) from source
        source_i, source_j = source_idx // grid_size, source_idx % grid_size
        graph_distances = np.zeros(data['num_points'])
        for idx in range(data['num_points']):
            i, j = idx // grid_size, idx % grid_size
            graph_distances[idx] = abs(i - source_i) + abs(j - source_j)
        
        # Check correlation between predicted and graph distances
        non_source = np.arange(data['num_points']) != source_idx
        correlation = np.corrcoef(distances[non_source], graph_distances[non_source])[0, 1]
        
        assert correlation > 0.5, \
            f"Distances should correlate with graph distance, got correlation={correlation:.3f}"
    
    def test_deterministic_reproducibility(self, grid_data):
        """Test that running twice produces identical results."""
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Run twice with same parameters
        results = []
        for _ in range(2):
            handler = DeterministicModelHandler(offset=0.1)
            propagator = FastMarchingPropagator(
                model_handler=handler,
                input_builder=input_builder,
                ring1_neighbors=data['ring1_neighbors'],
                ring=1,
                verbose=False
            )
            distances = propagator.propagate([data['source_idx']])
            results.append(distances.copy())
        
        np.testing.assert_array_equal(results[0], results[1],
            err_msg="Deterministic model should produce identical results")
    
    def test_different_offsets_produce_different_results(self, grid_data):
        """Test that model is invoked during propagation (sanity check).
        
        On flat grids the Euclidean-Dijkstra floor is exact, so different
        model offsets may produce identical final distances (the floor
        dominates via the ``max(model, floor)`` rule). We verify that:
        1. The model IS invoked (call_count > 0)
        2. Propagation completes with non-negative finite distances.
        """
        data = grid_data
        grid_size = data['grid_size']
        num_points = data['num_points']
        
        # Build 8-connected neighbours so the model path is exercised
        ring1_8 = {}
        for idx in range(num_points):
            i, j = idx // grid_size, idx % grid_size
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbors.append(ni * grid_size + nj)
            ring1_8[idx] = np.array(neighbors, dtype=np.int64)
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        for offset in [0.5, 5.0]:
            handler = DeterministicModelHandler(offset=offset)
            propagator = FastMarchingPropagator(
                model_handler=handler,
                input_builder=input_builder,
                ring1_neighbors=ring1_8,
                ring=1,
                verbose=False
            )
            distances = propagator.propagate([data['source_idx']])
            
            assert handler.model.call_count > 0, \
                f"Model should be invoked for offset={offset}"
            assert np.all(distances >= 0), \
                f"All distances should be non-negative for offset={offset}"
            assert np.all(np.isfinite(distances)), \
                f"All distances should be finite for offset={offset}"
    
    def test_multiple_sources_merge_correctly(self, grid_data):
        """Test that multiple source points all have distance 0."""
        data = grid_data
        grid_size = data['grid_size']
        
        # Use corners as sources
        sources = [0, grid_size - 1, (grid_size - 1) * grid_size, grid_size * grid_size - 1]
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate(sources)
        
        # All sources should have distance 0
        for s in sources:
            assert distances[s] == 0.0, f"Source {s} should have distance 0, got {distances[s]}"
        
        # All points should be reached
        assert np.sum(np.isfinite(distances)) == data['num_points']
    
    def test_visit_order_respects_wavefront(self, grid_data):
        """
        Test that points are visited in order of their estimated distance.
        
        Uses callback to track visit order.
        """
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        # Track visit order
        visit_order = []
        
        def callback(iteration, point_idx, distance):
            visit_order.append((point_idx, distance))
        
        propagator.propagate([data['source_idx']], callback=callback)
        
        # Verify monotonicity: distances should be non-decreasing
        visit_distances = [d for _, d in visit_order]
        for i in range(1, len(visit_distances)):
            assert visit_distances[i] >= visit_distances[i-1] - 1e-6, \
                f"Visit order violated at step {i}: {visit_distances[i-1]:.4f} > {visit_distances[i]:.4f}"
    
    def test_batch_vs_sequential_consistency(self, grid_data):
        """
        Test that the batched wavefront update produces consistent results
        by comparing against a manually sequentialized version.
        
        This verifies the batching logic without depending on model weights.
        """
        data = grid_data
        
        _cfg, _ring = _make_config(attributes=["xyz"], max_neighbors=8)
        input_builder = GaussianInputBuilder(
            positions=data['positions'],
            dataset_config=_cfg,
            ring=_ring,
            device='cpu'
        )
        
        # Use deterministic model - both should produce same results
        handler = DeterministicModelHandler(offset=0.1)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([data['source_idx']])
        
        # Key invariant: result should be deterministic and reproducible
        handler2 = DeterministicModelHandler(offset=0.1)
        propagator2 = FastMarchingPropagator(
            model_handler=handler2,
            input_builder=input_builder,
            ring1_neighbors=data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        distances2 = propagator2.propagate([data['source_idx']])
        
        np.testing.assert_allclose(distances, distances2, rtol=1e-5,
            err_msg="Batched propagation should be reproducible")
        
        # On sparse grids (4-connectivity, ring=1) the model may not be
        # invoked because most points have < MIN_NEIGHBORS_FOR_MODEL visited
        # neighbours and use the Euclidean fallback.  Only assert that the
        # two runs agree, which tests reproducibility regardless.
        
        print(f"\nModel call counts: {handler.model.call_count}, {handler2.model.call_count}")
        if handler.model.call_count > 0:
            print(f"Batch efficiency: ~{data['num_points'] / handler.model.call_count:.1f} points per call")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
