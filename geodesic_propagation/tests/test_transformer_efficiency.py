"""
Efficiency tests for GaussianPatchTransformer model.

Tests to understand and benchmark model inference performance.
"""

import pytest
import numpy as np
import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.GaussianPatchTransformer import (
    GaussianPatchTransformer,
    create_gaussian_patch_transformer
)


class TestTransformerInferenceSpeed:
    """Benchmark transformer inference speed."""
    
    @pytest.fixture(scope='class')
    def tiny_model(self):
        """Create a tiny model for fast testing."""
        return create_gaussian_patch_transformer(
            attributes=['xyz'],
            max_neighbors=4,
            embed_dim=8,
            encoder_depth=1,
            num_heads=1,
        )
    
    @pytest.fixture(scope='class')
    def small_model(self):
        """Create a small model."""
        return create_gaussian_patch_transformer(
            attributes=['xyz'],
            max_neighbors=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
        )
    
    @pytest.fixture(scope='class')
    def medium_model(self):
        """Create a medium model (typical for training)."""
        return create_gaussian_patch_transformer(
            attributes=['xyz', 'scale', 'rotation', 'opacity'],
            max_neighbors=32,
            embed_dim=256,
            encoder_depth=4,
            num_heads=8,
        )
    
    def test_single_vs_batch_inference_tiny(self, tiny_model):
        """Test that batching is faster than sequential for tiny model."""
        tiny_model.eval()
        n_samples = 16
        
        # Create data
        single_batch = torch.randn(1, 4, 4)
        single_pf = torch.randn(1, 3)
        single_mask = torch.ones(1, 4, dtype=torch.bool)
        
        batch = torch.randn(n_samples, 4, 4)
        batch_pf = torch.randn(n_samples, 3)
        batch_mask = torch.ones(n_samples, 4, dtype=torch.bool)
        
        # Warmup
        with torch.no_grad():
            tiny_model(single_batch, single_pf, single_mask)
        
        # Sequential inference
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(n_samples):
                tiny_model(single_batch, single_pf, single_mask)
        sequential_time = time.perf_counter() - t0
        
        # Batched inference
        t0 = time.perf_counter()
        with torch.no_grad():
            tiny_model(batch, batch_pf, batch_mask)
        batched_time = time.perf_counter() - t0
        
        print(f"\nTiny model ({n_samples} samples):")
        print(f"  Sequential: {sequential_time*1000:.1f}ms ({sequential_time/n_samples*1000:.2f}ms/sample)")
        print(f"  Batched: {batched_time*1000:.1f}ms ({batched_time/n_samples*1000:.2f}ms/sample)")
        print(f"  Speedup: {sequential_time/batched_time:.1f}x")
        
        # Batching should be faster
        assert batched_time < sequential_time, "Batched should be faster than sequential"
    
    def test_batch_size_scaling(self, tiny_model):
        """Test how inference time scales with batch size."""
        tiny_model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        times_per_sample = []
        
        for bs in batch_sizes:
            batch = torch.randn(bs, 4, 4)
            batch_pf = torch.randn(bs, 3)
            batch_mask = torch.ones(bs, 4, dtype=torch.bool)
            
            # Warmup
            with torch.no_grad():
                tiny_model(batch, batch_pf, batch_mask)
            
            # Measure
            n_iters = max(10, 100 // bs)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    tiny_model(batch, batch_pf, batch_mask)
            elapsed = time.perf_counter() - t0
            
            time_per_sample = elapsed / (n_iters * bs) * 1000
            times_per_sample.append(time_per_sample)
        
        print("\nBatch size scaling (tiny model):")
        for bs, t in zip(batch_sizes, times_per_sample):
            print(f"  Batch {bs:2d}: {t:.3f}ms/sample")
        
        # Larger batches should be more efficient per sample
        assert times_per_sample[-1] < times_per_sample[0], \
            "Larger batches should be more efficient"
    
    def test_model_size_comparison(self, tiny_model, small_model):
        """Compare inference time across model sizes."""
        batch_size = 8
        
        results = {}
        
        for name, model, max_neighbors, entry_size, point_dim in [
            ('tiny', tiny_model, 4, 4, 3),
            ('small', small_model, 16, 4, 3),
        ]:
            model.eval()
            
            batch = torch.randn(batch_size, max_neighbors, entry_size)
            batch_pf = torch.randn(batch_size, point_dim)
            batch_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
            
            # Warmup
            with torch.no_grad():
                model(batch, batch_pf, batch_mask)
            
            # Measure
            n_iters = 20
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    model(batch, batch_pf, batch_mask)
            elapsed = time.perf_counter() - t0
            
            time_per_batch = elapsed / n_iters * 1000
            results[name] = time_per_batch
        
        print(f"\nModel size comparison (batch={batch_size}):")
        for name, t in results.items():
            print(f"  {name}: {t:.2f}ms/batch")
        
        # Tiny should be faster than small
        assert results['tiny'] < results['small'], \
            "Tiny model should be faster than small"
    
    def test_inference_components_breakdown(self, tiny_model):
        """Profile different parts of the forward pass."""
        tiny_model.eval()
        batch_size = 8
        
        batch = torch.randn(batch_size, 4, 4)
        batch_pf = torch.randn(batch_size, 3)
        batch_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        
        # Full forward pass
        n_iters = 50
        
        with torch.no_grad():
            tiny_model(batch, batch_pf, batch_mask)  # warmup
        
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                tiny_model(batch, batch_pf, batch_mask)
        full_time = (time.perf_counter() - t0) / n_iters * 1000
        
        print(f"\nFull forward pass: {full_time:.2f}ms")
        
        # This is informational - just verify it runs
        assert full_time > 0


class TestTransformerMemory:
    """Test memory usage of transformer."""
    
    def test_memory_scaling_with_batch_size(self):
        """Test how memory scales with batch size."""
        model = create_gaussian_patch_transformer(
            attributes=['xyz'],
            max_neighbors=16,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
        )
        model.eval()
        
        # This test just verifies no memory errors occur
        batch_sizes = [1, 8, 32, 64]
        
        for bs in batch_sizes:
            batch = torch.randn(bs, 16, 4)
            batch_pf = torch.randn(bs, 3)
            batch_mask = torch.ones(bs, 16, dtype=torch.bool)
            
            with torch.no_grad():
                output = model(batch, batch_pf, batch_mask)
            
            assert output.shape == (bs, 1)
        
        print(f"\nMemory test passed for batch sizes: {batch_sizes}")


class TestBatchedPropagatorEfficiency:
    """Test the batched propagator implementation efficiency."""
    
    def test_batched_update_wavefront(self):
        """Test that batched wavefront update is efficient."""
        from geodesic_propagation.fast_marching import FastMarchingPropagator
        from geodesic_propagation.input_builder import GaussianInputBuilder
        from geodesic_propagation.utils.model_handler import ModelHandler
        from sklearn.neighbors import NearestNeighbors
        
        np.random.seed(42)
        num_points = 50
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        # Create neighbors
        nbrs = NearestNeighbors(n_neighbors=8).fit(positions)
        _, indices = nbrs.kneighbors(positions)
        ring1_neighbors = {i: indices[i, 1:] for i in range(num_points)}
        
        # Create model and propagator
        handler = ModelHandler()
        handler.create_model({
            'attributes': ['xyz'],
            'max_neighbors': 8,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config={
                'output_dir': '/tmp/test_transformer_eff',
                'attributes': ['xyz'],
                'nn_mean': 1.0,
                'mask_constant': -10.0,
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {'euclidean': {1: 8}},
            },
            ring=1,
            device='cpu'
        )
        
        propagator = FastMarchingPropagator(
            model_handler=handler,
            input_builder=input_builder,
            ring1_neighbors=ring1_neighbors,
            ring=1,
            verbose=False
        )
        
        # Initialize with multiple sources to get a larger wavefront
        propagator.initialize_sources([0, 10, 20])
        wavefront_size = propagator.wavefront.num_wavefront()
        
        print(f"\nWavefront size: {wavefront_size}")
        
        # Time a single wavefront update (batched)
        t0 = time.perf_counter()
        propagator._update_wavefront_distances()
        update_time = time.perf_counter() - t0
        
        print(f"Batched wavefront update: {update_time*1000:.1f}ms")
        print(f"Time per wavefront point: {update_time/wavefront_size*1000:.2f}ms")
        
        # Should complete in reasonable time
        assert update_time < 1.0, f"Wavefront update too slow: {update_time:.2f}s"
    
    def test_propagation_with_different_wavefront_sizes(self):
        """Test propagation efficiency with different wavefront sizes."""
        from geodesic_propagation.fast_marching import FastMarchingPropagator
        from geodesic_propagation.input_builder import GaussianInputBuilder
        from geodesic_propagation.utils.model_handler import ModelHandler
        from sklearn.neighbors import NearestNeighbors
        
        np.random.seed(42)
        num_points = 30
        positions = np.random.randn(num_points, 3).astype(np.float32)
        
        nbrs = NearestNeighbors(n_neighbors=6).fit(positions)
        _, indices = nbrs.kneighbors(positions)
        ring1_neighbors = {i: indices[i, 1:] for i in range(num_points)}
        
        handler = ModelHandler()
        handler.create_model({
            'attributes': ['xyz'],
            'max_neighbors': 8,
            'embed_dim': 8,
            'encoder_depth': 1,
            'num_heads': 1,
        })
        
        input_builder = GaussianInputBuilder(
            positions=positions,
            dataset_config={
                'output_dir': '/tmp/test_transformer_eff',
                'attributes': ['xyz'],
                'nn_mean': 1.0,
                'mask_constant': -10.0,
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {'euclidean': {1: 8}},
            },
            ring=1,
            device='cpu'
        )
        
        # Test with 1 source vs multiple sources
        times = {}
        
        for n_sources, sources in [(1, [0]), (3, [0, 10, 20])]:
            propagator = FastMarchingPropagator(
                model_handler=handler,
                input_builder=input_builder,
                ring1_neighbors=ring1_neighbors,
                ring=1,
                verbose=False
            )
            
            t0 = time.perf_counter()
            propagator.propagate(sources, max_iterations=5)
            times[n_sources] = time.perf_counter() - t0
        
        print(f"\nPropagation time (5 iterations):")
        for n, t in times.items():
            print(f"  {n} source(s): {t*1000:.1f}ms")
        
        # Both should complete in reasonable time
        for t in times.values():
            assert t < 2.0, f"Propagation too slow: {t:.2f}s"


class TestTransformerBottleneckAnalysis:
    """Detailed analysis of transformer bottlenecks."""
    
    def test_forward_pass_profiling(self):
        """Profile the forward pass to identify bottlenecks."""
        model = create_gaussian_patch_transformer(
            attributes=['xyz'],
            max_neighbors=8,
            embed_dim=32,
            encoder_depth=2,
            num_heads=4,
        )
        model.eval()
        
        batch_size = 4
        batch = torch.randn(batch_size, 8, 4)
        batch_pf = torch.randn(batch_size, 3)
        batch_mask = torch.ones(batch_size, 8, dtype=torch.bool)
        
        # Warmup
        with torch.no_grad():
            model(batch, batch_pf, batch_mask)
        
        # Profile with different numbers of iterations
        for n_iters in [10, 50, 100]:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    model(batch, batch_pf, batch_mask)
            elapsed = time.perf_counter() - t0
            
            per_iter = elapsed / n_iters * 1000
            print(f"  {n_iters} iters: {per_iter:.2f}ms/iter")
        
        # Verify timing is consistent
        assert True  # Informational test
    
    def test_overhead_analysis(self):
        """Analyze overhead from torch operations."""
        # Test pure tensor operations overhead
        batch_size = 8
        dim = 32
        
        x = torch.randn(batch_size, 8, dim)
        w = torch.randn(dim, dim)
        
        # Warmup
        _ = x @ w
        
        # Time matmul
        n_iters = 1000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = x @ w
        matmul_time = (time.perf_counter() - t0) / n_iters * 1000
        
        print(f"\nPure matmul ({x.shape} @ {w.shape}): {matmul_time:.4f}ms")
        
        # The overhead should be minimal for such small operations
        assert matmul_time < 1.0, "Matmul overhead too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
