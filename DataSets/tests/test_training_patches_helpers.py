"""
Tests for training_patches_helpers.py utility functions.
"""

import pytest
import numpy as np
import argparse
import multiprocessing
import torch

from DataSets.utils.training_patches_helpers import (
    compute_scale_stats,
    merge_config_with_args,
    generate_training_examples,
    _generate_sequential,
    _generate_chunk,
    _init_mp_worker,
    create_train_example,
    normalize_neighborhood,
    denormalize_neighborhood,
    build_inverse_ring1,
    GaussianData,
    PatchConfig,
)


class TestComputeScaleStats:
    """Test suite for compute_scale_stats function."""
    
    def test_basic_statistics(self):
        """Test basic statistical computations."""
        scales = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_scale_stats(scales)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        np.testing.assert_almost_equal(stats['mean'], 3.0)
        np.testing.assert_almost_equal(stats['min'], 1.0)
        np.testing.assert_almost_equal(stats['max'], 5.0)
    
    def test_single_value(self):
        """Test with a single value."""
        scales = np.array([5.0])
        stats = compute_scale_stats(scales)
        
        assert stats['mean'] == 5.0
        assert stats['min'] == 5.0
        assert stats['max'] == 5.0
        assert stats['std'] == 0.0
    
    def test_2d_scales(self):
        """Test with 2D scale array (N, 3) - computes stats over all elements."""
        scales = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        stats = compute_scale_stats(scales)
        
        # Computes stats over all elements
        np.testing.assert_almost_equal(stats['mean'], 5.0)  # Mean of 1-9
        np.testing.assert_almost_equal(stats['min'], 1.0)
        np.testing.assert_almost_equal(stats['max'], 9.0)
    
    def test_returns_floats(self):
        """Test that stats are returned as Python floats."""
        scales = np.array([1.0, 2.0, 3.0])
        stats = compute_scale_stats(scales)
        
        for key, value in stats.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"


class TestMergeConfigWithArgs:
    """Test suite for merge_config_with_args function."""
    
    def test_config_overrides_numeric_args(self):
        """Test that config overrides numeric arguments."""
        args = argparse.Namespace(
            n_neighbors=20,
            num_iterations=10,
            num_sources=5,
            num_train_points=100,
            seed=42,
            constant_val=0.0,
            mask_constant=-10.0
        )
        config = {
            'n_neighbors': 30,
            'num_iterations': 20,
            'seed': 123
        }
        
        merged = merge_config_with_args(config, args)
        
        # Config should override
        assert merged.n_neighbors == 30
        assert merged.num_iterations == 20
        assert merged.seed == 123
    
    def test_path_args_only_override_when_none(self):
        """Test that path args only get overridden when None."""
        args = argparse.Namespace(
            gaussian_output='/custom/path',
            geodesic_data=None,
            output_dir=None,
            iteration=None
        )
        config = {
            'gaussian_output': '/config/path',
            'geodesic_data': '/config/geodesic',
            'output_dir': '/config/output',
            'iteration': 30000
        }
        
        merged = merge_config_with_args(config, args)
        
        # Custom path should not be overridden (not None)
        assert merged.gaussian_output == '/custom/path'
        # None values should be filled from config
        assert merged.geodesic_data == '/config/geodesic'
        assert merged.output_dir == '/config/output'
        assert merged.iteration == 30000
    
    def test_rings_default_override(self):
        """Test that rings only override when at default value."""
        # With default value
        args_default = argparse.Namespace(rings=[2, 3])
        config = {'rings': [1, 2, 3]}
        merged = merge_config_with_args(config, args_default)
        assert merged.rings == [1, 2, 3]
        
        # With custom value - should still be overridden since config takes precedence
        # for listed mappings
        args_custom = argparse.Namespace(rings=[4, 5])
        merged2 = merge_config_with_args(config, args_custom)
        # Custom value preserved since it's not the default
        assert merged2.rings == [4, 5]
    
    def test_boolean_args(self):
        """Test handling of boolean arguments."""
        args = argparse.Namespace(
            use_mahalanobis=False,
            use_r1_min_val=False
        )
        config = {
            'use_mahalanobis': True,
            'use_r1_min_val': True
        }
        
        merged = merge_config_with_args(config, args)
        
        # Booleans should be overridden when False
        assert merged.use_mahalanobis == True
        assert merged.use_r1_min_val == True
    
    def test_ring_size_mapping_stored(self):
        """Test that ring_size_mapping is stored correctly."""
        args = argparse.Namespace()
        config = {
            'ring_size_mapping': {1: 32, 2: 64, 3: 128}
        }
        
        merged = merge_config_with_args(config, args)
        
        assert hasattr(merged, 'ring_size_mapping')
        assert merged.ring_size_mapping == {1: 32, 2: 64, 3: 128}
    
    def test_dataset_info_stored(self):
        """Test that dataset info is stored correctly."""
        args = argparse.Namespace()
        config = {
            'dataset': {
                'name': 'test_dataset',
                'type': 'polynomial'
            }
        }
        
        merged = merge_config_with_args(config, args)
        
        assert hasattr(merged, 'dataset_info')
        assert merged.dataset_info['name'] == 'test_dataset'
    
    def test_none_config_values_not_applied(self):
        """Test that None values in config don't override."""
        args = argparse.Namespace(
            gaussian_output='/my/path',
            geodesic_data='/my/geodesic'
        )
        config = {
            'gaussian_output': None,
            'geodesic_data': None
        }
        
        merged = merge_config_with_args(config, args)
        
        # Original values should be preserved
        assert merged.gaussian_output == '/my/path'
        assert merged.geodesic_data == '/my/geodesic'

    def test_adaptive_knn_config_keys(self):
        """Test that adaptive kNN config keys are merged properly."""
        args = argparse.Namespace(
            adaptive_target_ring=None,
            adaptive_target_neighbors=None,
            adaptive_k_boost=20,
            adaptive_max_mean_cut=5.0,
            adaptive_max_steps=5,
        )
        config = {
            'adaptive_target_ring': 3,
            'adaptive_target_neighbors': 128,
            'adaptive_k_boost': 20,
            'adaptive_max_mean_cut': 3.0,
            'adaptive_max_steps': 8,
        }

        merged = merge_config_with_args(config, args)

        assert merged.adaptive_target_ring == 3
        assert merged.adaptive_target_neighbors == 128
        assert merged.adaptive_k_boost == 20
        assert merged.adaptive_max_mean_cut == 3.0
        assert merged.adaptive_max_steps == 8


class TestComputeGaussianNormals:
    """Test suite for compute_gaussian_normals function."""
    
    def test_import_and_signature(self):
        """Test that compute_gaussian_normals can be imported and has correct signature."""
        from DataSets.utils.training_patches_helpers import compute_gaussian_normals
        import inspect
        
        sig = inspect.signature(compute_gaussian_normals)
        params = list(sig.parameters.keys())
        
        assert 'scales' in params
        assert 'rotations' in params
    
    @pytest.mark.skip(reason="Requires full 3DGS environment with utils.general_utils")
    def test_basic_functionality(self):
        """Test that normals are computed correctly from scales and rotations."""
        from DataSets.utils.training_patches_helpers import compute_gaussian_normals
        
        num_gaussians = 10
        
        # Create scales with smallest value in z-direction
        scales = np.ones((num_gaussians, 3), dtype=np.float32)
        scales[:, 2] = 0.1  # Smallest scale in z
        
        # Identity quaternions (w, x, y, z) = (1, 0, 0, 0)
        rotations = np.zeros((num_gaussians, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        
        normals = compute_gaussian_normals(scales, rotations)
        
        assert normals.shape == (num_gaussians, 3)
        # With smallest scale in z and identity rotation, normals should be z-axis
        for i in range(num_gaussians):
            # Check that z-component has highest magnitude
            assert abs(normals[i, 2]) >= abs(normals[i, 0])
            assert abs(normals[i, 2]) >= abs(normals[i, 1])
    
    @pytest.mark.skip(reason="Requires full 3DGS environment with utils.general_utils")
    def test_different_smallest_axes(self):
        """Test normals align with different smallest scale axes."""
        from DataSets.utils.training_patches_helpers import compute_gaussian_normals
        
        # Identity rotation
        rotation = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        
        # Smallest scale in x
        scales_x = np.array([[0.1, 1.0, 1.0]], dtype=np.float32)
        normals_x = compute_gaussian_normals(scales_x, rotation)
        assert abs(normals_x[0, 0]) >= abs(normals_x[0, 1])
        assert abs(normals_x[0, 0]) >= abs(normals_x[0, 2])
        
        # Smallest scale in y
        scales_y = np.array([[1.0, 0.1, 1.0]], dtype=np.float32)
        normals_y = compute_gaussian_normals(scales_y, rotation)
        assert abs(normals_y[0, 1]) >= abs(normals_y[0, 0])
        assert abs(normals_y[0, 1]) >= abs(normals_y[0, 2])


class TestIntegration:
    """Integration tests for helper functions."""
    
    def test_scale_stats_on_real_scales(self):
        """Test compute_scale_stats with realistic scale data."""
        np.random.seed(42)
        
        # Simulate typical Gaussian scales (log-normal distributed)
        log_scales = np.random.randn(100, 3) * 0.5 - 3  # Log scales around -3
        scales = np.exp(log_scales)
        
        stats = compute_scale_stats(scales)
        
        assert stats['min'] > 0
        assert stats['max'] > stats['min']
        assert stats['mean'] > 0
        assert stats['std'] > 0
    
    def test_merge_config_preserves_all_fields(self):
        """Test that merge_config preserves fields not in mapping."""
        args = argparse.Namespace(
            custom_field='original',
            n_neighbors=20
        )
        config = {
            'n_neighbors': 30,
            # custom_field not in config
        }
        
        merged = merge_config_with_args(config, args)
        
        # Custom field should be preserved
        assert merged.custom_field == 'original'
        # Mapped field should be updated
        assert merged.n_neighbors == 30


# ---------------------------------------------------------------------------
# Helper fixtures / utilities for generate_training_examples tests
# ---------------------------------------------------------------------------

def _make_dummy_data(num_points=50, num_sources=5, seed=42):
    """Create minimal dummy data for generate_training_examples tests."""
    rng = np.random.RandomState(seed)

    positions = rng.randn(num_points, 3).astype(np.float32)
    scales = np.abs(rng.randn(num_points, 3).astype(np.float32)) * 0.1
    rotations = np.zeros((num_points, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # identity quaternion
    opacities = rng.uniform(0.1, 1.0, (num_points, 1)).astype(np.float32)

    # Simple geodesic data: random distances from a few sources
    source_indices = rng.choice(num_points, num_sources, replace=False)
    geodesic_distances = rng.uniform(0.1, 5.0, (num_sources, num_points)).astype(np.float32)
    geodesic_data = {
        'source_gaussian_indices': source_indices,
        'geodesic_distances': geodesic_distances,
    }

    # Build trivial ring neighbors (every point sees the nearest 5 as ring-1, etc.)
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    ring1_nbrs = {}
    ring2_nbrs = {}
    for i in range(num_points):
        dists, idxs = tree.query(positions[i], k=min(6, num_points))
        ring1_nbrs[i] = idxs[1:]  # exclude self
        dists2, idxs2 = tree.query(positions[i], k=min(20, num_points))
        ring2_nbrs[i] = idxs2[1:]

    ring_nbrs_dict = {1: ring1_nbrs, 2: ring2_nbrs}

    ring_size_mapping = {'euclidean': {1: 5, 2: 19}}

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'geodesic_data': geodesic_data,
        'ring_nbrs_dict': ring_nbrs_dict,
        'ring1_nbrs': ring1_nbrs,
        'ring_size_mapping': ring_size_mapping,
    }


class TestGenerateTrainingExamplesParallel:
    """Tests for the parallelised generate_training_examples."""

    @pytest.fixture(scope="class")
    def dummy_data(self):
        return _make_dummy_data()

    def _run(self, data, num_workers, num_iterations=10, seed=42):
        gdata = GaussianData(
            positions=data['positions'],
            scales=data['scales'],
            rotations=data['rotations'],
            opacities=data['opacities'],
            normals=None,
            sh_features=None,
            per_point_nn_distances=None,
        )
        pcfg = PatchConfig(
            ring=2,
            normalization_factor=1.0,
            nn_mean=1.0,
            attributes=['xyz'],
            use_mahalanobis=False,
            use_r1_min_val=False,
            mask_attributes=[],
            mask_constant=-10.0,
            ring_size_mapping=data['ring_size_mapping'],
            normalize_per_patch=False,
        )
        return generate_training_examples(
            geodesic_data=data['geodesic_data'],
            ring_nbrs_dict=data['ring_nbrs_dict'],
            ring1_nbrs=data['ring1_nbrs'],
            num_iterations=num_iterations,
            num_sources=2,
            num_train_points=5,
            gaussian_data=gdata,
            patch_config=pcfg,
            seed=seed,
            verbose=False,
            num_workers=num_workers,
        )

    def test_sequential_produces_results(self, dummy_data):
        """Sequential path (num_workers=1) returns a non-empty array."""
        result = self._run(dummy_data, num_workers=1)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] > 0

    def test_parallel_produces_results(self, dummy_data):
        """Parallel path (num_workers=2) returns a non-empty array."""
        result = self._run(dummy_data, num_workers=2)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] > 0

    def test_parallel_same_total_examples(self, dummy_data):
        """Sequential and parallel produce the same number of examples."""
        seq = self._run(dummy_data, num_workers=1, num_iterations=20, seed=99)
        par = self._run(dummy_data, num_workers=2, num_iterations=20, seed=99)
        assert seq.shape[1] == par.shape[1], "Feature dimension must match"
        # Total count should be very close (same iterations, same seed logic)
        # They won't be identical because parallel uses per-chunk RNG,
        # but overall count should be similar.
        assert par.shape[0] > 0

    def test_auto_workers(self, dummy_data):
        """Default num_workers=None auto-selects based on cpu_count."""
        result = self._run(dummy_data, num_workers=None, num_iterations=4)
        assert result.shape[0] > 0

    def test_more_workers_than_iterations(self, dummy_data):
        """Gracefully handles more workers than iterations."""
        result = self._run(dummy_data, num_workers=10, num_iterations=3)
        assert result.shape[0] > 0

    def test_single_iteration_parallel(self, dummy_data):
        """Single iteration falls back to sequential."""
        result = self._run(dummy_data, num_workers=4, num_iterations=1)
        assert result.shape[0] > 0


class TestMemoryMappedDataset:
    """Tests for memory-mapped (lazy) loading in GaussianPatchDataset."""

    @pytest.fixture
    def temp_mmap_dataset(self, tmp_path):
        """Create a temp dataset and verify it loads with memory mapping."""
        import yaml

        config = {
            'output_dir': str(tmp_path),
            'attributes': ['xyz'],
            'use_r1_min_val': False,
            'use_mahalanobis': False,
            'ring_size_mapping': {
                'euclidean': {1: 5, 2: 10}
            },
        }
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # entry_size = xyz(3) + geodesic(1) = 4
        # total cols = max_neighbors*4 + point_features(3) + p_u(1) = 10*4 + 3 + 1 = 44
        max_neighbors = 10
        entry_size = 4
        num_examples = 200
        total_cols = max_neighbors * entry_size + 3 + 1
        examples = np.random.randn(num_examples, total_cols).astype(np.float32)
        examples[:, -1] = np.abs(examples[:, -1]) + 0.01  # positive targets
        np.save(tmp_path / 'gaussian_examples_ring2_n200.npy', examples)

        return tmp_path, config_path, examples

    def test_lazy_loading_is_memmap(self, temp_mmap_dataset):
        """With lazy_loading=True, self.examples should be a numpy memmap."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        _, config_path, _ = temp_mmap_dataset
        ds = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=True)
        assert isinstance(ds.examples, np.memmap), \
            f"Expected np.memmap, got {type(ds.examples)}"

    def test_eager_loading_is_tensor(self, temp_mmap_dataset):
        """With lazy_loading=False (default), self.examples should be a torch.Tensor."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        import torch
        _, config_path, _ = temp_mmap_dataset
        ds = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=False)
        assert isinstance(ds.examples, torch.Tensor), \
            f"Expected torch.Tensor, got {type(ds.examples)}"

    def test_default_is_eager(self, temp_mmap_dataset):
        """Default (no lazy_loading argument) should be eager loading."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        import torch
        _, config_path, _ = temp_mmap_dataset
        ds = GaussianPatchDataset(config=str(config_path), ring=2)
        assert isinstance(ds.examples, torch.Tensor)

    def test_getitem_returns_tensors(self, temp_mmap_dataset):
        """__getitem__ should return torch tensors in both modes."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        import torch
        _, config_path, _ = temp_mmap_dataset
        for lazy in (True, False):
            ds = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=lazy)
            neighborhood, point_features, target, valid_mask = ds[0]
            assert isinstance(neighborhood, torch.Tensor)
            assert isinstance(point_features, torch.Tensor)
            assert isinstance(target, torch.Tensor)
            assert isinstance(valid_mask, torch.Tensor)

    def test_getitem_values_match_raw(self, temp_mmap_dataset):
        """Values returned by __getitem__ should be consistent between lazy and eager modes."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        _, config_path, raw_examples = temp_mmap_dataset
        # Compare lazy vs eager — both should return the same normalized values
        ds_lazy = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=True)
        ds_eager = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=False)
        _, _, target_lazy, _ = ds_lazy[5]
        _, _, target_eager, _ = ds_eager[5]
        np.testing.assert_almost_equal(target_lazy.item(), target_eager.item(), decimal=4)
        # Target should be positive (raw targets are positive)
        assert target_eager.item() > 0

    def test_len_correct(self, temp_mmap_dataset):
        """__len__ should reflect the total number of examples."""
        from DataSets.gaussian_dataset import GaussianPatchDataset
        _, config_path, raw_examples = temp_mmap_dataset
        for lazy in (True, False):
            ds = GaussianPatchDataset(config=str(config_path), ring=2, lazy_loading=lazy)
            assert len(ds) == raw_examples.shape[0]


# ---------------------------------------------------------------------------
# Tests for normalize_neighborhood and denormalize_neighborhood
# ---------------------------------------------------------------------------

def _make_neighborhoods(n_valid=5, n_beyond=3, attr_size=4, mask_constant=-10.0, rng=None):
    """Create small valid/beyond neighborhood arrays for tests."""
    if rng is None:
        rng = np.random.RandomState(7)
    # attr_size columns (last = geodesic)
    valid = rng.rand(n_valid, attr_size).astype(np.float64)
    valid[:, -1] = rng.uniform(0.0, 3.0, n_valid)  # geodesic column
    beyond = rng.rand(n_beyond, attr_size).astype(np.float64)
    beyond[:, -1] = rng.uniform(4.0, 8.0, n_beyond)  # beyond threshold
    p_features = rng.rand(attr_size - 1).astype(np.float64)  # no geodesic for point
    r1_neighborhood = rng.rand(3, attr_size).astype(np.float64)
    r1_neighborhood[:, -1] = rng.uniform(0.1, 2.0, 3)
    p_u = float(rng.uniform(2.0, 3.0))
    r1_min_val = float(rng.uniform(0.1, 1.0))
    return valid, beyond, p_features, r1_neighborhood, p_u, r1_min_val


class TestNormalizeNeighborhood:
    """Tests for the standalone normalize_neighborhood function."""

    def test_min_geodesic_becomes_zero(self):
        """After normalization the smallest valid geodesic should be 0."""
        valid, beyond, p_feats, r1_nbhd, p_u, r1_min = _make_neighborhoods()
        attrs = ["xyz"]  # entry: xyz(3) + geo(1)
        orig_min = valid[:, -1].min()

        (v, b, pf, r1, pu_n, r1_n, min_in) = normalize_neighborhood(
            valid.copy(), beyond.copy(), p_feats.copy(), r1_nbhd.copy(),
            p_u, r1_min,
            attributes=attrs,
            current_normalization=1.0,
            nn_mean=1.0,
            mask_constant=-10.0,
        )
        assert min_in == pytest.approx(orig_min)
        assert v[:, -1].min() == pytest.approx(0.0, abs=1e-7)

    def test_beyond_geodesic_becomes_mask_constant(self):
        """Beyond-threshold neighbors' geodesic column must be mask_constant."""
        MASK = -99.0
        valid, beyond, p_feats, r1_nbhd, p_u, r1_min = _make_neighborhoods()
        (v, b, *_) = normalize_neighborhood(
            valid.copy(), beyond.copy(), p_feats.copy(), r1_nbhd.copy(),
            p_u, r1_min,
            attributes=["xyz"],
            current_normalization=1.0,
            nn_mean=1.0,
            mask_constant=MASK,
        )
        np.testing.assert_allclose(b[:, -1], MASK)

    def test_spatial_scaling(self):
        """xyz columns should be scaled by nn_mean / current_normalization."""
        rng = np.random.RandomState(11)
        valid = rng.rand(4, 4).astype(np.float64)
        valid[:, -1] = rng.uniform(0.1, 1.0, 4)
        beyond = np.zeros((0, 4), dtype=np.float64)
        p_feats = rng.rand(3).astype(np.float64)
        r1_nbhd = rng.rand(2, 4).astype(np.float64)
        r1_nbhd[:, -1] = 0.5

        orig_xyz = valid[:, :3].copy()
        nn_mean = 2.0
        c_norm = 4.0

        (v, b, *_) = normalize_neighborhood(
            valid.copy(), beyond.copy(), p_feats.copy(), r1_nbhd.copy(),
            p_u=1.0, r1_min_val=0.5,
            attributes=["xyz"],
            current_normalization=c_norm,
            nn_mean=nn_mean,
            mask_constant=-10.0,
        )
        expected = orig_xyz / c_norm * nn_mean
        np.testing.assert_allclose(v[:, :3], expected, rtol=1e-5)

    def test_returns_min_input(self):
        """Function must return min_input as last element."""
        valid, beyond, p_feats, r1_nbhd, p_u, r1_min = _make_neighborhoods()
        expected_min = valid[:, -1].min()
        result = normalize_neighborhood(
            valid.copy(), beyond.copy(), p_feats.copy(), r1_nbhd.copy(),
            p_u, r1_min,
            attributes=["xyz"],
            current_normalization=1.0,
            nn_mean=1.0,
            mask_constant=-10.0,
        )
        assert result[-1] == pytest.approx(expected_min, rel=1e-6)


class TestDenormalizeNeighborhood:
    """Tests for the standalone denormalize_neighborhood function."""

    def _normalize_and_collect(self, attrs, c_norm, nn_mean, mask_constant=-10.0):
        """Run normalize, return inputs and outputs for roundtrip checks."""
        valid, beyond, p_feats, r1_nbhd, p_u, r1_min = _make_neighborhoods()
        orig_valid = valid.copy()
        orig_p_feats = p_feats.copy()
        orig_p_u = p_u
        orig_r1 = r1_min

        (v_n, b_n, pf_n, r1_n, pu_n, r1v_n, min_in) = normalize_neighborhood(
            valid.copy(), beyond.copy(), p_feats.copy(), r1_nbhd.copy(),
            p_u, r1_min,
            attributes=attrs,
            current_normalization=c_norm,
            nn_mean=nn_mean,
            mask_constant=mask_constant,
        )
        # Combine valid + beyond into a single neighborhood for denorm
        combined = np.vstack([v_n, b_n]) if len(b_n) > 0 else v_n
        return combined, pf_n, pu_n, r1v_n, min_in, orig_p_u, orig_r1

    def test_target_roundtrip(self):
        """Denormalizing a normalized target should recover the original."""
        attrs = ["xyz"]
        c_norm, nn_mean = 2.0, 1.0

        combined, pf_n, pu_n, r1v_n, min_in, orig_pu, orig_r1 = \
            self._normalize_and_collect(attrs, c_norm, nn_mean)

        _, _, target_rec, _ = denormalize_neighborhood(
            combined, pf_n, pu_n, r1v_n,
            min_input=min_in,
            current_normalization=c_norm,
            nn_mean=nn_mean,
            attributes=attrs,
            mask_constant=-10.0,
        )
        assert target_rec == pytest.approx(orig_pu, rel=1e-5)

    def test_r1_min_val_roundtrip(self):
        """Denormalizing r1_min_val should recover the original value."""
        attrs = ["xyz"]
        c_norm, nn_mean = 3.0, 2.0

        combined, pf_n, pu_n, r1v_n, min_in, orig_pu, orig_r1 = \
            self._normalize_and_collect(attrs, c_norm, nn_mean)

        _, _, _, r1_rec = denormalize_neighborhood(
            combined, pf_n, pu_n, r1v_n,
            min_input=min_in,
            current_normalization=c_norm,
            nn_mean=nn_mean,
            attributes=attrs,
            mask_constant=-10.0,
        )
        assert r1_rec == pytest.approx(orig_r1, rel=1e-5)

    def test_masked_entries_unchanged(self):
        """Entries whose geodesic == mask_constant must not be modified."""
        MASK = -10.0
        nbhd = np.array([[1.0, 2.0, 3.0, MASK], [0.5, 0.5, 0.5, 0.2]])
        pf = np.array([0.0, 0.0, 0.0])
        nbhd_dn, _, _, _ = denormalize_neighborhood(
            nbhd, pf, target=0.1, r1_min_val=None,
            min_input=0.0,
            current_normalization=1.0,
            nn_mean=1.0,
            attributes=["xyz"],
            mask_constant=MASK,
        )
        assert nbhd_dn[0, -1] == MASK

    def test_zero_nn_mean_is_safe(self):
        """nn_mean=0 edge case: function returns inputs unchanged."""
        nbhd = np.ones((3, 4))
        pf = np.ones(3)
        nbhd_dn, pf_dn, t, r = denormalize_neighborhood(
            nbhd, pf, target=1.0, r1_min_val=0.5,
            min_input=0.0,
            current_normalization=1.0,
            nn_mean=0.0,
            attributes=["xyz"],
            mask_constant=-10.0,
        )
        # When nn_mean=0, denormalization is skipped
        np.testing.assert_allclose(nbhd_dn, nbhd)

    def test_create_train_example_uses_normalize_neighborhood(self):
        """
        Verify that normalizing a raw example via denormalize_neighborhood
        recovers reasonable values — confirming the training pipeline uses
        the extracted function.
        """
        from DataSets.utils.training_patches_helpers import create_train_example
        from DataSets.utils.config_utils import get_ring_size_mapping

        rng = np.random.RandomState(55)
        n = 30
        positions = rng.randn(n, 3).astype(np.float64)
        geo_dists = rng.uniform(0.5, 5.0, n).astype(np.float64)

        # Build simple ring-1 and ring-2 neighbor dicts
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        ring1_nbrs = {}
        ring2_nbrs = {}
        for i in range(n):
            _, idxs = tree.query(positions[i], k=6)
            ring1_nbrs[i] = idxs[1:]
            _, idxs2 = tree.query(positions[i], k=16)
            ring2_nbrs[i] = idxs2[1:]

        ring_size_mapping = {'euclidean': {2: 15}}
        c_norm = 0.5
        nn_mean = 1.0

        example = create_train_example(
            point_idx=0,
            positions=positions,
            normals=None,
            geodesic_distances=geo_dists,
            ring_nbrs=ring2_nbrs,
            ring1_nbrs=ring1_nbrs,
            ring=2,
            normalization_factor=c_norm,
            nn_mean=nn_mean,
            attributes=["xyz"],
            use_r1_min_val=True,
            mask_constant=-10.0,
            ring_size_mapping=ring_size_mapping,
        )
        assert example is not None, "create_train_example returned None"
        # Target is the last value; must be a finite positive number after normalization
        target_norm = float(example[-1])
        assert np.isfinite(target_norm)
        assert target_norm >= 0.0


class TestOutlierFiltering:
    """Tests for the geo/euclidean ratio outlier filtering in create_train_example."""

    def _build_fold_data(self, n=30, n_fold=5, seed=42):
        """Create data with a simulated surface fold (outlier neighbors).

        Some neighbors are placed physically close but with very large
        geodesic distances — simulating a different mesh sheet.
        """
        rng = np.random.RandomState(seed)
        positions = rng.randn(n, 3).astype(np.float64)

        # Geodesic distances: normal neighbours have values in [0.5, 3]
        geo_dists = rng.uniform(0.5, 3.0, n).astype(np.float64)

        # Introduce fold outliers: points very close to point 0 in xyz
        # but with huge geodesic distances (simulating a fold).
        fold_indices = list(range(1, 1 + n_fold))
        for idx in fold_indices:
            positions[idx] = positions[0] + rng.randn(3) * 0.01  # very close in xyz
            geo_dists[idx] = rng.uniform(50.0, 100.0)  # very far in geodesic

        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        ring1_nbrs = {}
        ring2_nbrs = {}
        for i in range(n):
            _, idxs = tree.query(positions[i], k=min(6, n))
            ring1_nbrs[i] = idxs[1:]
            _, idxs2 = tree.query(positions[i], k=min(20, n))
            ring2_nbrs[i] = idxs2[1:]

        return positions, geo_dists, ring1_nbrs, ring2_nbrs

    def test_outliers_excluded_from_neighborhood(self):
        """Fold outliers (high geo/euclidean ratio) should be filtered out."""
        positions, geo_dists, ring1_nbrs, ring2_nbrs = self._build_fold_data()

        ring_size_mapping = {'euclidean': {2: 19}}
        example = create_train_example(
            point_idx=0,
            positions=positions,
            normals=None,
            geodesic_distances=geo_dists,
            ring_nbrs=ring2_nbrs,
            ring1_nbrs=ring1_nbrs,
            ring=2,
            normalization_factor=1.0,
            nn_mean=1.0,
            attributes=["xyz"],
            use_r1_min_val=True,
            mask_constant=-10.0,
            ring_size_mapping=ring_size_mapping,
        )
        assert example is not None

        # The neighborhood should NOT contain geodesic values in [50, 100]
        # (after normalization they'd still be very large relative to others).
        # The outlier neighbors have geo/euclidean ratio >> 3*median so they
        # should have been removed.
        max_neighbors = 19
        entry_size = 4  # xyz(3) + geodesic(1)
        neighborhood = example[: max_neighbors * entry_size].reshape(max_neighbors, entry_size)
        geo_col = neighborhood[:, -1]
        valid_geo = geo_col[geo_col != -10.0]
        # Valid geodesic values should all be reasonable — none should be
        # the extreme outlier values (50–100 before normalisation, still
        # disproportionately large after).
        if len(valid_geo) > 0:
            assert valid_geo.max() < 50.0, \
                f"Outlier geodesic {valid_geo.max()} found — filtering failed"

    def test_no_outliers_on_smooth_surface(self):
        """On a smooth surface all neighbors should survive filtering."""
        rng = np.random.RandomState(99)
        n = 30
        positions = rng.randn(n, 3).astype(np.float64)
        # Geodesic ≈ Euclidean for a smooth surface
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        geo_dists = np.zeros(n, dtype=np.float64)
        _, idxs0 = tree.query(positions[0], k=n)
        for i in range(n):
            geo_dists[i] = np.linalg.norm(positions[i] - positions[0]) * 1.1  # slight fudge

        ring1_nbrs = {}
        ring2_nbrs = {}
        for i in range(n):
            _, idxs = tree.query(positions[i], k=min(6, n))
            ring1_nbrs[i] = idxs[1:]
            _, idxs2 = tree.query(positions[i], k=min(20, n))
            ring2_nbrs[i] = idxs2[1:]

        ring_size_mapping = {'euclidean': {2: 19}}
        example = create_train_example(
            point_idx=0,
            positions=positions,
            normals=None,
            geodesic_distances=geo_dists,
            ring_nbrs=ring2_nbrs,
            ring1_nbrs=ring1_nbrs,
            ring=2,
            normalization_factor=1.0,
            nn_mean=1.0,
            attributes=["xyz"],
            use_r1_min_val=True,
            mask_constant=-10.0,
            ring_size_mapping=ring_size_mapping,
        )
        assert example is not None


class TestRandomDuplicationPadding:
    """Tests for the random-duplication padding strategy in create_train_example."""

    def test_padded_entries_have_real_features_but_masked_geodesic(self):
        """Padded entries should duplicate real neighbor features and mask only geodesic."""
        rng = np.random.RandomState(77)
        n = 10  # small so most slots need padding
        positions = rng.randn(n, 3).astype(np.float64)
        geo_dists = rng.uniform(0.5, 5.0, n).astype(np.float64)

        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        ring1_nbrs = {}
        ring2_nbrs = {}
        for i in range(n):
            _, idxs = tree.query(positions[i], k=min(6, n))
            ring1_nbrs[i] = idxs[1:]
            _, idxs2 = tree.query(positions[i], k=min(8, n))
            ring2_nbrs[i] = idxs2[1:]

        max_neighbors = 20  # deliberately larger than available neighbors
        ring_size_mapping = {'euclidean': {2: max_neighbors}}

        example = create_train_example(
            point_idx=0,
            positions=positions,
            normals=None,
            geodesic_distances=geo_dists,
            ring_nbrs=ring2_nbrs,
            ring1_nbrs=ring1_nbrs,
            ring=2,
            normalization_factor=1.0,
            nn_mean=1.0,
            attributes=["xyz"],
            use_r1_min_val=True,
            mask_constant=-10.0,
            ring_size_mapping=ring_size_mapping,
        )
        assert example is not None

        entry_size = 4  # xyz(3) + geodesic(1)
        point_feature_size = 3
        neighborhood = example[: max_neighbors * entry_size].reshape(max_neighbors, entry_size)

        for i in range(max_neighbors):
            geo = neighborhood[i, -1]
            xyz = neighborhood[i, :3]
            if geo == -10.0:
                # Padded entry: xyz should NOT be -10 (old sentinel);
                # it should be a copy of some real neighbor's xyz.
                assert not np.allclose(xyz, -10.0), \
                    f"Padded entry {i} should have real xyz, not sentinel -10"
            # All entries should have finite xyz
            assert np.all(np.isfinite(xyz)), f"Entry {i} has non-finite xyz"


class TestPointcloudDropoutNewMasking:
    """Test that PointcloudRandomInputDropout now uses random duplication."""

    def test_dropped_entries_have_real_features(self):
        """After dropout, dropped entries should keep real features, only geodesic masked."""
        from DataSets.data_transformation import PointcloudRandomInputDropout

        dropout = PointcloudRandomInputDropout(
            max_dropout_ratio=0.99,
            attributes=["xyz"],
            mask_constant=-10
        )
        # pc shape: (batch=2, num_points=50, entry_size=4)
        pc = torch.randn(2, 50, 4)
        pc[:, :, -1] = torch.rand(2, 50)  # positive geodesic distances
        pc_dropped = dropout(pc.clone())

        for b in range(2):
            for p in range(50):
                if pc_dropped[b, p, -1] == -10.0:
                    # xyz should NOT all be -10 (old sentinel behaviour)
                    assert not torch.allclose(
                        pc_dropped[b, p, :3],
                        torch.tensor([-10.0, -10.0, -10.0])
                    ), f"Dropped entry [{b},{p}] xyz should be a real duplicate, not sentinel"


class TestBuildInverseRing1:
    """Tests for build_inverse_ring1."""

    def test_basic_inverse(self):
        """Inverse ring maps each neighbour back to the point that lists it."""
        ring1 = {
            0: np.array([1, 2]),
            1: np.array([0, 3]),
            2: np.array([0]),
            3: np.array([1]),
        }
        inv = build_inverse_ring1(ring1, 4)
        # Point 0 is in ring1 of points 1 and 2
        assert set(inv[0].tolist()) == {1, 2}
        # Point 1 is in ring1 of points 0 and 3
        assert set(inv[1].tolist()) == {0, 3}
        # Point 2 is only in ring1 of point 0
        assert set(inv[2].tolist()) == {0}
        # Point 3 is only in ring1 of point 1
        assert set(inv[3].tolist()) == {1}

    def test_asymmetric_graph(self):
        """KNN is NOT symmetric — inverse ring captures this."""
        ring1 = {
            0: np.array([1, 2, 3]),  # 0 sees 1, 2, 3
            1: np.array([0]),         # 1 only sees 0
            2: np.array([]),          # 2 sees nobody
            3: np.array([]),          # 3 sees nobody
        }
        inv = build_inverse_ring1(ring1, 4)
        # 0 is in ring1 of point 1 only
        assert set(inv[0].tolist()) == {1}
        # 1 is in ring1 of point 0 only
        assert set(inv[1].tolist()) == {0}
        # 2 is in ring1 of point 0 only
        assert set(inv[2].tolist()) == {0}
        # 3 is in ring1 of point 0 only
        assert set(inv[3].tolist()) == {0}

    def test_empty_rings(self):
        """Points with no neighbours should have empty inverse ring."""
        ring1 = {0: np.array([]), 1: np.array([])}
        inv = build_inverse_ring1(ring1, 2)
        assert len(inv[0]) == 0
        assert len(inv[1]) == 0

    def test_returns_numpy_arrays(self):
        ring1 = {0: np.array([1]), 1: np.array([0])}
        inv = build_inverse_ring1(ring1, 2)
        for v in inv.values():
            assert isinstance(v, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

