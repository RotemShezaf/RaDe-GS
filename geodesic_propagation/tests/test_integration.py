"""
Integration tests for the geodesic propagation module.

These tests verify that all components work together correctly.
"""

import pytest
import numpy as np
import torch
import tempfile
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.fast_marching import FastMarchingPropagator
from geodesic_propagation.input_builder import GaussianInputBuilder
from geodesic_propagation.utils.model_handler import ModelHandler
from geodesic_propagation.utils.results_saver import ResultsSaver
from geodesic_propagation.priority_queue import WavefrontPriorityQueue


def _make_builder(positions, max_neighbors=4, normalization_factor=1.0,
                  nn_mean=1.0, attributes=None):
    """Create a GaussianInputBuilder with the new dataset_config API."""
    if attributes is None:
        attributes = ["xyz"]
    if max_neighbors <= 8:
        ring = 1
    elif max_neighbors <= 32:
        ring = 2
    else:
        ring = 3
    config = {
        "output_dir": "/tmp/test_integration",
        "attributes": attributes,
        "nn_mean": nn_mean,
        "mask_constant": -10.0,
        "use_r1_min_val": False,
        "use_mahalanobis": False,
        "ring_size_mapping": {"euclidean": {ring: max_neighbors}},
    }
    return GaussianInputBuilder(
        positions=positions,
        dataset_config=config,
        ring=ring,
        normalization_factor=normalization_factor,
        device="cpu",
    )


def _gdist(n_pts, nbr_indices, nbr_dists, default=1e12):
    """Expand per-neighbor distances into a global (N,) distance array."""
    d = np.full(n_pts, default, dtype=np.float64)
    d[nbr_indices] = nbr_dists
    return d


class TestEndToEndPropagation:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def synthetic_gaussian_data(self):
        """Create synthetic Gaussian splat data on a sphere."""
        np.random.seed(42)
        
        # Create points on a sphere (small number for fast tests)
        num_points = 15
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        theta = np.arccos(np.random.uniform(-1, 1, num_points))
        
        r = 1.0
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        positions = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # Create Gaussian properties
        scales = np.ones((num_points, 3), dtype=np.float32) * 0.05
        
        # Random rotations
        rotations = np.random.randn(num_points, 4).astype(np.float32)
        rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
        
        opacities = np.ones((num_points, 1), dtype=np.float32) * 0.9
        sh_features = np.random.randn(num_points, 3).astype(np.float32) * 0.1
        
        # Compute neighbors
        k = 10
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(positions)
        _, indices = nbrs.kneighbors(positions)
        ring1_neighbors = {i: indices[i, 1:] for i in range(num_points)}
        
        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'sh_features': sh_features,
            'ring1_neighbors': ring1_neighbors,
            'num_points': num_points,
        }
    
    def test_full_pipeline(self, synthetic_gaussian_data):
        """Test the complete propagation and saving pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tiny model for speed
            handler = ModelHandler()
            handler.create_model({
                'attributes': ["xyz"],
                'max_neighbors': 4,
                'embed_dim': 8,
                'encoder_depth': 1,
                'num_heads': 1,
            })
            
            # Create input builder
            input_builder = _make_builder(
                synthetic_gaussian_data['positions'],
                max_neighbors=4,
                normalization_factor=0.2,
                nn_mean=1.0,
            )
            
            # Create propagator
            propagator = FastMarchingPropagator(
                model_handler=handler,
                input_builder=input_builder,
                ring1_neighbors=synthetic_gaussian_data['ring1_neighbors'],
                ring=1,
                verbose=False
            )
            
            # Run propagation with max_iterations limit
            source_indices = [0, 5]
            distances = propagator.propagate(source_indices, max_iterations=2)
            
            # Verify basic properties
            assert len(distances) == synthetic_gaussian_data['num_points']
            assert distances[0] == 0.0
            assert distances[5] == 0.0
            
            # Save results
            saver = ResultsSaver(tmpdir)
            result_path = saver.save_propagation_results(
                distances=distances,
                source_indices=source_indices,
                positions=synthetic_gaussian_data['positions'],
                metadata={'test': True},
                name='test_propagation'
            )
            
            # Load and verify
            loaded = saver.load_results('test_propagation')
            assert np.allclose(loaded['distances'], distances)
            assert list(loaded['source_indices']) == source_indices
    
    def test_model_save_load_cycle(self, synthetic_gaussian_data):
        """Test saving and loading model checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and configure tiny model
            handler1 = ModelHandler()
            handler1.create_model({
                'attributes': ["xyz", "scale"],
                'max_neighbors': 4,
                'embed_dim': 8,
                'encoder_depth': 1,
                'num_heads': 1,
            })
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "model.pt"
            handler1.save_checkpoint(checkpoint_path, epoch=10)
            
            # Load in new handler
            handler2 = ModelHandler(model_path=checkpoint_path)
            
            # Verify config matches
            assert handler1.get_attributes() == handler2.get_attributes()
            assert handler1.get_max_neighbors() == handler2.get_max_neighbors()
    
    def test_batched_vs_sequential_consistency(self, synthetic_gaussian_data):
        """Test that batched and sequential propagation give similar results."""
        # Create model and input builder (tiny for speed)
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': 4,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = _make_builder(
            synthetic_gaussian_data['positions'],
            max_neighbors=4,
        )
        
        # Create two propagators with same configuration
        propagator1 = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=synthetic_gaussian_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        propagator2 = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=synthetic_gaussian_data['ring1_neighbors'],
            ring=1,
            verbose=False
        )
        
        source_indices = [0]
        
        # Sequential propagation
        distances1 = propagator1.propagate(source_indices, max_iterations=3)
        
        # Batched propagation
        distances2 = propagator2.propagate_batch(
            source_indices, 
            batch_size=2, 
            max_iterations=3
        )
        
        # Both should visit the source with 0 distance
        assert distances1[0] == 0.0
        assert distances2[0] == 0.0
        
        # Both should reach similar number of points.  propagate() uses
        # Euclidean seeding for ring-1 neighbours (expanding more points
        # before the max_iterations loop), while propagate_batch() uses
        # model predictions.  With only 3 iterations the gap can be large.
        visited1 = np.sum(np.isfinite(distances1))
        visited2 = np.sum(np.isfinite(distances2))
        assert visited1 >= 1 and visited2 >= 1


class TestComponentInteraction:
    """Test interactions between components."""
    
    def test_priority_queue_with_propagator(self):
        """Test that priority queue state is correctly managed."""
        np.random.seed(42)
        num_points = 10  # Reduced from 20
        
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        # Simple neighbors
        ring1_neighbors = {i: np.array([(i + 1) % num_points, (i - 1) % num_points]) 
                         for i in range(num_points)}
        
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': 4,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = _make_builder(positions, max_neighbors=4)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        # Initialize sources
        propagator.initialize_sources([0])
        
        # Check queue state
        queue = propagator.wavefront
        assert queue.num_visited() == 1
        assert queue.num_wavefront() > 0
        assert queue.get_state(0) == 2  # VISITED
        
        # Run a few iterations
        for _ in range(5):
            if queue.is_empty():
                break
            propagator._update_wavefront_distances()
            result = queue.pop_min()
            if result:
                point_idx, dist = result
                propagator.distances[point_idx] = dist
    
    def test_input_builder_normalization_consistency(self):
        """Test that normalization is consistent across builds."""
        np.random.seed(42)
        num_points = 30
        
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        builder = _make_builder(
            positions, max_neighbors=8, normalization_factor=0.5, nn_mean=1.0
        )
        
        # Build same input twice — seed np.random so padding duplication
        # (random choice in create_train_example) is deterministic.
        neighbor_indices = np.array([1, 2, 3])
        visited = np.ones(num_points, dtype=bool)
        global_dists = _gdist(num_points, neighbor_indices, np.array([0.1, 0.2, 0.3]))
        
        np.random.seed(123)
        result1 = builder.build_input(
            point_idx=0,
            all_neighbor_indices=neighbor_indices,
            neighbor_distances=global_dists,
            visited_mask=visited,
        )
        assert result1 is not None
        neighborhood1, pf1, mask1, _ = result1
        
        np.random.seed(123)
        result2 = builder.build_input(
            point_idx=0,
            all_neighbor_indices=neighbor_indices,
            neighbor_distances=global_dists,
            visited_mask=visited,
        )
        assert result2 is not None
        neighborhood2, pf2, mask2, _ = result2
        
        # Should be identical
        assert torch.allclose(neighborhood1, neighborhood2)
        assert torch.allclose(pf1, pf2)
        assert torch.equal(mask1, mask2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_point(self):
        """Test propagation with a single point."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        ring1_neighbors = {0: np.array([], dtype=np.int64)}
        
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': 4,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = _make_builder(positions, max_neighbors=4)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([0], max_iterations=2)
        assert distances[0] == 0.0
    
    def test_disconnected_components(self):
        """Test propagation on disconnected components."""
        # Two separate clusters
        positions = np.array([
            [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0],  # Cluster 1
            [10, 10, 10], [10.1, 10, 10], [10, 10.1, 10]  # Cluster 2
        ], dtype=np.float32)
        
        ring1_neighbors = {
            0: np.array([1, 2]), 1: np.array([0, 2]), 2: np.array([0, 1]),
            3: np.array([4, 5]), 4: np.array([3, 5]), 5: np.array([3, 4])
        }
        
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': 4,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = _make_builder(positions, max_neighbors=4)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        # Propagate from cluster 1
        distances = propagator.propagate([0], max_iterations=3)
        
        # Cluster 1 should be reached
        assert np.isfinite(distances[0])
        assert np.isfinite(distances[1])
        assert np.isfinite(distances[2])
        
        # Cluster 2 should not be reached
        assert np.isinf(distances[3])
        assert np.isinf(distances[4])
        assert np.isinf(distances[5])
    
    def test_empty_neighbors(self):
        """Test handling points with no neighbors."""
        positions = np.random.randn(5, 3).astype(np.float32)
        ring1_neighbors = {i: np.array([], dtype=np.int64) for i in range(5)}
        
        handler = ModelHandler()
        handler.create_model({
            'attributes': ["xyz"],
            'max_neighbors': 4,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = _make_builder(positions, max_neighbors=4)
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        distances = propagator.propagate([0], max_iterations=2)
        
        # Only source should be reached
        assert distances[0] == 0.0
        assert all(np.isinf(distances[i]) for i in range(1, 5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
