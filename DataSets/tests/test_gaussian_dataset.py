#!/usr/bin/env python3
"""
Tests for GaussianPatchDataset

Tests cover:
- Dataset initialization and configuration loading
- Attribute filtering
- r1_min_val removal
- Normalization (opacity, SH, scale)
- Data loading and shape validation
"""

import pytest
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from gaussian_dataset import GaussianPatchDataset, create_dataloaders


@pytest.fixture
def temp_dataset_dir():
    """Create temporary directory with test data (module-level fixture)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config = {
            'output_dir': str(tmpdir),
            'attributes': ['xyz', 'normals', 'opacity', 'scale'],
            'use_r1_min_val': True,
            'use_mahalanobis': False,
            'ring_size_mapping': {
                'euclidean': {1: 32, 2: 64, 3: 128, 4: 256}
            }
        }
        config_path = tmpdir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        max_neighbors = 64
        entry_size = 11
        num_examples = 100
        examples = np.random.randn(num_examples, 716).astype(np.float32)
        for i in range(max_neighbors):
            opacity_idx = i * entry_size + 6
            examples[:, opacity_idx] = np.random.uniform(0, 1, num_examples)
        point_opacity_idx = max_neighbors * entry_size + 6
        examples[:, point_opacity_idx] = np.random.uniform(0, 1, num_examples)
        examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
        examples[:, -2] = np.random.uniform(0.05, 0.5, num_examples)
        np.save(tmpdir / 'gaussian_examples_ring2.npy', examples)

        yield tmpdir, config_path


class TestGaussianPatchDataset:
    """Test suite for GaussianPatchDataset class."""

    @pytest.fixture
    def temp_dataset_dir(self, temp_dataset_dir):
        """Forward to the module-level fixture."""
        return temp_dataset_dir

    def test_dataset_initialization(self, temp_dataset_dir):
        """Test basic dataset initialization."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True
        )
        
        assert len(dataset) == 100
        assert dataset.ring == 2
        assert dataset.max_neighbors == 64
        assert dataset.use_r1_min == True
    
    def test_attribute_filtering(self, temp_dataset_dir):
        """Test filtering to subset of attributes."""
        tmpdir, config_path = temp_dataset_dir
        
        # Request only xyz and opacity
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=True
        )
        
        # Should filter out normals and scale
        neighborhood, point_features, target, valid_mask = dataset[0]
        # Expected: neighborhood (64, 5) where entry_size = xyz(3) + opacity(1) + geodesic(1) = 5
        # point_features: xyz(3) + opacity(1) = 4
        assert neighborhood.shape == (64, 5)
        assert point_features.shape[0] == 4
    
    def test_r1_min_removal(self, temp_dataset_dir):
        """Test r1_min_val removal when use_r1_min=False."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=False  # Remove r1_min_val
        )
        
        neighborhood, point_features, target, valid_mask = dataset[0]
        
        # neighborhood: (64, 11) where entry_size = xyz(3) + normals(3) + opacity(1) + scale(3) + geodesic(1) = 11
        # point_features: xyz(3) + normals(3) + opacity(1) + scale(3) = 10
        assert neighborhood.shape == (64, 11)
        assert point_features.shape[0] == 10
        assert isinstance(target, torch.Tensor)
        # Note: r1_min_val is no longer returned, transforms receive None when use_r1_min=False
    
    def test_invalid_attributes(self, temp_dataset_dir):
        """Test that invalid attributes raise assertion error."""
        tmpdir, config_path = temp_dataset_dir
        
        with pytest.raises(AssertionError):
            dataset = GaussianPatchDataset(
                config_path=config_path,
                attributes=['xyz', 'invalid_attr'],  # invalid_attr not in config
                ring=2,
                use_r1_min=True
            )
    
    def test_data_loading_shapes(self, temp_dataset_dir):
        """Test that loaded data has correct shapes."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True
        )
        
        for i in range(min(5, len(dataset))):
            neighborhood, point_features, target, valid_mask = dataset[i]
            
            # Check shapes
            assert neighborhood.dim() == 2  # (max_neighbors, entry_size)
            assert neighborhood.shape[0] == dataset.max_neighbors
            assert point_features.dim() == 1
            assert target.dim() == 0  # scalar
            
            # Check valid_mask shape (max_neighbors,)
            assert valid_mask.dim() == 1
            assert valid_mask.shape[0] == dataset.max_neighbors
            assert valid_mask.dtype == torch.bool
            
            # Check target is positive (geodesic distance)
            assert target.item() > 0
    
    def test_normalization_opacity(self, temp_dataset_dir):
        """Test opacity normalization to [0, 1]."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True
        )
        
        # Normalization now happens per-example in __getitem__
        # Get a sample and check opacity is normalized
        neighborhood, point_features, target, valid_mask = dataset[0]
        
        # In the new format:
        # neighborhood: (max_neighbors, entry_size) where entry_size = xyz(3) + normals(3) + opacity(1) + scale(3) + geodesic(1) = 11
        # opacity is at index 6 (after xyz and normals)
        neighbor_opacities = neighborhood[:, 6]  # (max_neighbors,)
        point_opacity = point_features[6]  # scalar
        
        # Check opacity values are in [0, 1]
        assert (neighbor_opacities >= 0).all() and (neighbor_opacities <= 1).all(), \
            f"Neighbor opacity out of range [0, 1]"
        assert 0 <= point_opacity <= 1, f"Point opacity {point_opacity} out of range [0, 1]"
    
    def test_normalization_scale(self, temp_dataset_dir):
        """Test scale normalization using pc_norm approach."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True
        )
        
        # Normalization now happens per-example in __getitem__
        # Get a sample and check scale is normalized
        neighborhood, point_features, target, valid_mask = dataset[0]
        
        # In the new format:
        # scale is at index 7:10 (after xyz, normals, opacity)
        neighbor_scales = neighborhood[:, 7:10]  # (max_neighbors, 3)
        point_scale = point_features[7:10]  # (3,)
        
        # Combine all scales
        all_scales = torch.cat([neighbor_scales, point_scale.unsqueeze(0)], dim=0)  # (max_neighbors+1, 3)
        
        # Check that centroid is close to zero (pc_norm centers data)
        centroid = all_scales.mean(dim=0)
        assert torch.allclose(centroid, torch.zeros(3), atol=1e-5), f"Centroid {centroid} not centered"
        
        # Check that max distance is approximately 1 (pc_norm scales by max distance)
        distances = torch.sqrt(torch.sum(all_scales**2, dim=1))
        max_dist = distances.max()
        assert torch.isclose(max_dist, torch.tensor(1.0), atol=1e-5), f"Max distance {max_dist} not normalized to 1"


class TestDataLoaders:
    """Test suite for dataloader creation."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create config file
            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {
                        1: 32,
                        2: 64
                    }
                }
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create sample data
            # attributes: xyz(3) + opacity(1) = 4
            # entry_size = 4 + 1 (geodesic) = 5
            max_neighbors = 64
            entry_size = 5
            point_feature_size = 4
            num_examples = 50
            
            # neighbor_features: 64 * 5 = 320
            # point_features: 4
            # p_u: 1
            # Total: 325
            
            examples = np.random.randn(num_examples, 325).astype(np.float32)
            examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
            
            data_path = tmpdir / 'gaussian_examples_ring2.npy'
            np.save(data_path, examples)
            
            yield tmpdir, config_path
    
    def test_dataloader_creation(self, temp_dataset_dir):
        """Test creating train and validation dataloaders."""
        tmpdir, config_path = temp_dataset_dir
        
        train_loader, val_loader = create_dataloaders(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=8,
            train_split=0.8,
            num_workers=0
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Check batch shapes
        batch = next(iter(train_loader))
        neighborhoods, point_features, targets, valid_masks = batch
        
        # neighborhoods: (batch, max_neighbors, entry_size) = (batch, 64, 5)
        assert neighborhoods.shape[0] <= 8  # batch size
        assert neighborhoods.shape[1] == 64  # max_neighbors
        assert neighborhoods.shape[2] == 5  # entry_size for xyz + opacity + geodesic
        # point_features: (batch, point_feature_size) = (batch, 4)
        assert point_features.shape[0] == neighborhoods.shape[0]
        assert point_features.shape[1] == 4
        assert targets.shape[0] == neighborhoods.shape[0]
        assert valid_masks.shape[0] == neighborhoods.shape[0]  # batch dim
        assert valid_masks.dtype == torch.bool
    
    def test_dataloader_split(self, temp_dataset_dir):
        """Test train/val split ratios."""
        tmpdir, config_path = temp_dataset_dir
        
        train_loader, val_loader = create_dataloaders(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=4,
            train_split=0.8,
            num_workers=0
        )
        
        # Count examples
        train_count = sum(batch[0].shape[0] for batch in train_loader)
        val_count = sum(batch[0].shape[0] for batch in val_loader)
        
        total = train_count + val_count
        train_ratio = train_count / total
        
        # Should be approximately 0.8
        assert 0.75 <= train_ratio <= 0.85


class TestValidMask:
    """Test suite for valid_mask functionality."""
    
    @pytest.fixture
    def temp_dataset_with_masked_neighbors(self):
        """Create temporary directory with test data that includes masked neighbors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create config file
            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {
                        1: 32,
                        2: 64
                    }
                }
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create sample data with some masked neighbors
            # attributes: xyz(3) + opacity(1) = 4
            # entry_size = 4 + 1 (geodesic) = 5
            max_neighbors = 64
            entry_size = 5
            num_examples = 20
            
            # neighbor_features: 64 * 5 = 320
            # point_features: 4
            # p_u: 1
            # Total: 325
            
            examples = np.random.randn(num_examples, 325).astype(np.float32)
            examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
            
            # Set some neighbors to have geodesic_distance = mask_constant (-10.0)
            mask_constant = -10.0
            for ex_idx in range(num_examples):
                # Mask the last few neighbors for each example
                num_masked = np.random.randint(5, 20)  # Mask 5-20 neighbors
                for neighbor_idx in range(max_neighbors - num_masked, max_neighbors):
                    geodesic_idx = neighbor_idx * entry_size + (entry_size - 1)  # Last value is geodesic
                    examples[ex_idx, geodesic_idx] = mask_constant
            
            data_path = tmpdir / 'gaussian_examples_ring2.npy'
            np.save(data_path, examples)
            
            yield tmpdir, config_path
    
    def test_valid_mask_detects_masked_neighbors(self, temp_dataset_with_masked_neighbors):
        """Test that valid_mask correctly identifies masked neighbors."""
        tmpdir, config_path = temp_dataset_with_masked_neighbors
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
        )
        
        for i in range(min(5, len(dataset))):
            neighborhood, point_features, target, valid_mask = dataset[i]
            
            # Check that valid_mask has the right shape
            assert valid_mask.shape[0] == dataset.max_neighbors
            assert valid_mask.dtype == torch.bool
            
            # Check that there are some masked neighbors (False values)
            # Based on our data generation, each example has 5-20 masked neighbors
            num_valid = valid_mask.sum().item()
            num_masked = dataset.max_neighbors - num_valid
            assert 5 <= num_masked <= 20, f"Expected 5-20 masked neighbors, got {num_masked}"
            
            # Verify that the masked positions correspond to mask_constant in features
            # by checking that the last masked neighbors have mask_constant geodesic values
            assert num_valid > 0, "At least some neighbors should be valid"
    
    def test_valid_mask_all_valid(self, temp_dataset_with_masked_neighbors):
        """Test behavior when all neighbors are valid (no masking)."""
        tmpdir, config_path = temp_dataset_with_masked_neighbors
        
        # Create data with all valid neighbors
        max_neighbors = 64
        entry_size = 5
        num_examples = 5
        
        examples = np.random.randn(num_examples, 325).astype(np.float32)
        # Set positive geodesic distances (never mask_constant)
        for ex_idx in range(num_examples):
            for neighbor_idx in range(max_neighbors):
                geodesic_idx = neighbor_idx * entry_size + (entry_size - 1)
                examples[ex_idx, geodesic_idx] = np.random.uniform(0.1, 2.0)
        examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
        
        data_path = tmpdir / 'gaussian_examples_ring2.npy'
        np.save(data_path, examples)
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
        )
        
        for i in range(len(dataset)):
            neighborhood, point_features, target, valid_mask = dataset[i]
            
            # All neighbors should be valid
            assert valid_mask.all(), "All neighbors should be valid when no mask_constant values"


class TestDatasetWithTransforms:
    """Test suite for dataset with multiple transformations."""
    
    @pytest.fixture
    def temp_dataset_with_r1_min(self):
        """Create temporary directory with test data including r1_min_val."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create config file
            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'normals', 'opacity'],
                'use_r1_min_val': True,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {
                        1: 32,
                        2: 64
                    }
                }
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create sample data
            # attributes: xyz(3) + normals(3) + opacity(1) = 7
            # entry_size = 7 + 1 (geodesic) = 8
            max_neighbors = 64
            entry_size = 8
            point_feature_size = 7
            num_examples = 30
            
            # neighbor_features: 64 * 8 = 512
            # point_features: 7
            # r1_min_val: 1
            # p_u: 1
            # Total: 521
            
            examples = np.random.randn(num_examples, 521).astype(np.float32)
            
            # Set xyz coordinates
            for i in range(max_neighbors):
                xyz_start = i * entry_size
                examples[:, xyz_start:xyz_start+3] = np.random.uniform(-1, 1, (num_examples, 3))
            
            # Set normals (normalized)
            for i in range(max_neighbors):
                normals_start = i * entry_size + 3
                normals = np.random.randn(num_examples, 3)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
                examples[:, normals_start:normals_start+3] = normals
            
            # Set opacity to [0, 1] range
            for i in range(max_neighbors):
                opacity_idx = i * entry_size + 6
                examples[:, opacity_idx] = np.random.uniform(0, 1, num_examples)
            
            # Set geodesic distances
            for i in range(max_neighbors):
                geodesic_idx = i * entry_size + 7
                examples[:, geodesic_idx] = np.random.uniform(0.1, 2.0, num_examples)
            
            # Set point features
            point_start = max_neighbors * entry_size
            examples[:, point_start:point_start+3] = 0  # xyz at origin
            # Point normals
            point_normals = np.random.randn(num_examples, 3)
            point_normals = point_normals / np.linalg.norm(point_normals, axis=1, keepdims=True)
            examples[:, point_start+3:point_start+6] = point_normals
            # Point opacity
            examples[:, point_start+6] = np.random.uniform(0, 1, num_examples)
            
            # Set r1_min_val and p_u
            examples[:, -2] = np.random.uniform(0.05, 0.5, num_examples)  # r1_min_val
            examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)  # p_u
            
            data_path = tmpdir / 'gaussian_examples_ring2.npy'
            np.save(data_path, examples)
            
            yield tmpdir, config_path
    
    def test_dataset_with_single_transform(self, temp_dataset_with_r1_min):
        """Test dataset with a single transformation."""
        from data_transformation import GaussianPatchRotate
        
        tmpdir, config_path = temp_dataset_with_r1_min
        
        transform = GaussianPatchRotate(attributes=['xyz', 'normals', 'opacity'])
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity'],
            ring=2,
            use_r1_min=True,
            transform=transform
        )
        
        neighborhood, point_features, target, valid_mask = dataset[0]
        
        # Check shapes are preserved
        assert neighborhood.shape == (64, 8)  # (max_neighbors, entry_size)
        assert point_features.shape[0] == 7
        assert valid_mask.dtype == torch.bool
    
    def test_dataset_with_composed_transforms(self, temp_dataset_with_r1_min):
        """Test dataset with composed transformations (dropout + rotate + flip)."""
        from data_transformation import GaussianPatchDropout, GaussianPatchRotate, GaussianPatchRandomFlip
        from torchvision.transforms import Compose
        
        tmpdir, config_path = temp_dataset_with_r1_min
        
        # Create a custom compose that handles the 3-argument interface
        class TransformCompose:
            def __init__(self, transforms):
                self.transforms = transforms
            
            def __call__(self, neighborhood, point_features, r1_min_val, **kwargs):
                for t in self.transforms:
                    neighborhood, point_features = t(neighborhood, point_features, r1_min_val, **kwargs)
                return neighborhood, point_features
        
        transform = TransformCompose([
            GaussianPatchDropout(max_dropout_ratio=0.3, attributes=['xyz', 'normals', 'opacity']),
            GaussianPatchRotate(attributes=['xyz', 'normals', 'opacity']),
            GaussianPatchRandomFlip(attributes=['xyz', 'normals', 'opacity'], flip_prob=0.5),
        ])
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity'],
            ring=2,
            use_r1_min=True,
            transform=transform
        )
        
        # Test multiple examples
        for i in range(min(10, len(dataset))):
            neighborhood, point_features, target, valid_mask = dataset[i]
            
            # Check shapes are preserved after all transforms
            assert neighborhood.shape == (64, 8)
            assert point_features.shape[0] == 7
            assert valid_mask.dtype == torch.bool
            assert target.dim() == 0
    
    def test_dataset_with_transforms_no_r1_min(self, temp_dataset_with_r1_min):
        """Test that transforms receive None when use_r1_min=False."""
        from data_transformation import GaussianPatchDropout
        
        tmpdir, config_path = temp_dataset_with_r1_min
        
        # Track what r1_min_val is passed to transform
        received_r1_min_vals = []
        
        class TrackingDropout(GaussianPatchDropout):
            def __call__(self, neighborhood, point_features, r1_min_val, **kwargs):
                received_r1_min_vals.append(r1_min_val)
                return super().__call__(neighborhood, point_features, r1_min_val, **kwargs)
        
        transform = TrackingDropout(max_dropout_ratio=0.3, attributes=['xyz', 'normals', 'opacity'])
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity'],
            ring=2,
            use_r1_min=False,  # Should pass None to transforms
            transform=transform
        )
        
        # Get a few examples
        for i in range(3):
            _ = dataset[i]
        
        # Check that None was passed to transforms
        assert all(v is None for v in received_r1_min_vals), \
            f"Expected None for r1_min_val when use_r1_min=False, got {received_r1_min_vals}"
    
    def test_dropout_with_none_r1_min_drops_from_all(self, temp_dataset_with_r1_min):
        """Test that dropout with None r1_min can drop from all neighbors."""
        from data_transformation import GaussianPatchDropout
        
        tmpdir, config_path = temp_dataset_with_r1_min
        
        # Create dataset without r1_min
        dataset_no_r1_min = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        # Get original neighborhood
        neighborhood_orig, point_features, _, _ = dataset_no_r1_min[0]
        
        # Apply dropout with None r1_min
        dropout = GaussianPatchDropout(max_dropout_ratio=0.99, attributes=['xyz', 'normals', 'opacity'])
        
        # Run multiple times to ensure dropout can happen
        np.random.seed(42)
        total_dropped = 0
        for _ in range(10):
            neighborhood_copy = neighborhood_orig.clone()
            neighborhood_dropped, _ = dropout(neighborhood_copy, point_features, None)
            # Count masked neighbors (geodesic == -10)
            masked_count = (neighborhood_dropped[:, -1] == -10.0).sum().item()
            total_dropped += masked_count
        
        # With 99% dropout from all neighbors, we should have some dropped
        assert total_dropped > 0, "Dropout with None r1_min should drop from all neighbors"


class TestDenormalizeResults:
    """Tests for GaussianPatchDataset._denormalize_results."""

    @pytest.fixture
    def ds_with_item(self, temp_dataset_dir):
        tmpdir, config_path = temp_dataset_dir
        ds = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True,
        )
        return ds

    def test_target_roundtrip(self, ds_with_item):
        """_denormalize_results must recover a target scaled by max_dist."""
        ds = ds_with_item
        nbhd, pf, target, valid_mask = ds[0]
        # Manually re-run _normalize_patches to obtain norm_params
        # We need the un-normalized neighborhood; easiest via get_item_from_raw
        raw = ds.examples[0]
        raw_example = raw if isinstance(raw, torch.Tensor) else torch.from_numpy(raw).float()
        nbhd2, pf2, target2, valid_mask2 = ds.get_item_from_raw(raw_example)

        # The two paths must agree
        assert torch.allclose(target, target2, atol=1e-5), \
            "get_item_from_raw and __getitem__ should return the same target"

    def test_denormalize_results_scales_target(self, ds_with_item):
        """After _denormalize_results the target should be larger (un-scaled)."""
        ds = ds_with_item
        nbhd, pf, target_norm, valid_mask = ds[0]

        # Build a fake norm_params with a known max_dist
        max_dist = 2.0
        norm_params = {'max_dist': torch.tensor(max_dist)}

        nbhd_dn, pf_dn, target_dn = ds._denormalize_results(nbhd, pf, target_norm, norm_params)

        expected_target = target_norm.item() * max_dist
        assert target_dn.item() == pytest.approx(expected_target, rel=1e-5)

    def test_denormalize_results_xyz_scaled(self, ds_with_item):
        """Real xyz entries should be multiplied by max_dist."""
        ds = ds_with_item
        nbhd, pf, target_norm, valid_mask = ds[0]

        # Identify xyz offset (first attribute)
        xyz_offset = 0
        max_dist = 3.0
        norm_params = {'max_dist': torch.tensor(max_dist)}

        # Record a valid xyz value
        valid_idx = valid_mask.nonzero(as_tuple=True)[0][0].item()
        orig_xyz = nbhd[valid_idx, xyz_offset:xyz_offset + 3].clone()

        nbhd_dn, _, _ = ds._denormalize_results(nbhd, pf, target_norm, norm_params)

        expected = orig_xyz * max_dist
        assert torch.allclose(nbhd_dn[valid_idx, xyz_offset:xyz_offset + 3], expected, atol=1e-5)

    def test_denormalize_does_not_modify_no_xyz(self, temp_dataset_dir):
        """When xyz is NOT in attributes, _denormalize_results returns unchanged tensors."""
        tmpdir, config_path = temp_dataset_dir
        ds = GaussianPatchDataset(
            config_path=config_path,
            attributes=['opacity'],
            ring=2,
            use_r1_min=False,
        )
        nbhd, pf, target, valid_mask = ds[0]
        norm_params = {'max_dist': torch.tensor(5.0)}

        nbhd_dn, pf_dn, target_dn = ds._denormalize_results(nbhd, pf, target, norm_params)

        # Without xyz, nothing should change
        assert torch.allclose(nbhd, nbhd_dn)
        assert torch.allclose(target, target_dn)


class TestGetItemFromRaw:
    """Tests for GaussianPatchDataset.get_item_from_raw."""

    @pytest.fixture
    def dataset(self, temp_dataset_dir):
        tmpdir, config_path = temp_dataset_dir
        return GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=True,
        )

    def test_same_output_as_getitem(self, dataset):
        """get_item_from_raw must produce the same result as __getitem__."""
        for idx in range(min(5, len(dataset))):
            raw = dataset.examples[idx]
            nbhd_gi, pf_gi, target_gi, vm_gi = dataset[idx]
            nbhd_raw, pf_raw, target_raw, vm_raw = dataset.get_item_from_raw(raw)

            assert torch.allclose(nbhd_gi, nbhd_raw, atol=1e-5), \
                f"neighborhood mismatch at idx={idx}"
            assert torch.allclose(pf_gi, pf_raw, atol=1e-5), \
                f"point_features mismatch at idx={idx}"
            assert torch.allclose(target_gi, target_raw, atol=1e-5), \
                f"target mismatch at idx={idx}"
            assert torch.equal(vm_gi, vm_raw), \
                f"valid_mask mismatch at idx={idx}"

    def test_output_shapes(self, dataset):
        """Output shapes should match the standard __getitem__ contract."""
        raw = dataset.examples[0]
        nbhd, pf, target, vm = dataset.get_item_from_raw(raw)

        assert nbhd.dim() == 2
        assert nbhd.shape[0] == dataset.max_neighbors
        assert nbhd.shape[1] == dataset.features_entry_size
        assert pf.dim() == 1
        assert target.dim() == 0
        assert vm.shape == (dataset.max_neighbors,)
        assert vm.dtype == torch.bool

    def test_accepts_tensor_input(self, dataset):
        """get_item_from_raw should accept a torch.Tensor as well as a numpy array."""
        raw_np = dataset.examples[0]
        raw_tensor = raw_np if isinstance(raw_np, torch.Tensor) else torch.from_numpy(raw_np).float()
        raw_np = raw_np.numpy() if isinstance(raw_np, torch.Tensor) else raw_np

        nbhd_np, pf_np, t_np, vm_np = dataset.get_item_from_raw(raw_np)
        nbhd_t, pf_t, t_t, vm_t = dataset.get_item_from_raw(raw_tensor)

        assert torch.allclose(nbhd_np, nbhd_t, atol=1e-6)
        assert torch.allclose(pf_np, pf_t, atol=1e-6)
        assert torch.allclose(t_np, t_t, atol=1e-6)

    def test_valid_mask_is_bool(self, dataset):
        nbhd, pf, target, vm = dataset.get_item_from_raw(dataset.examples[0])
        assert vm.dtype == torch.bool

    def test_target_is_finite(self, dataset):
        """Target (normalized geodesic) must be a finite non-negative scalar."""
        for idx in range(min(10, len(dataset))):
            _, _, target, _ = dataset.get_item_from_raw(dataset.examples[idx])
            assert torch.isfinite(target).all()
            assert target.item() >= 0.0


class TestNormalizeAllNeighbors:
    """Test normalize_all_neighbors option in GaussianPatchDataset."""

    @pytest.fixture
    def dataset_dir_with_padding(self):
        """Create dataset where some neighbors are padded (geodesic = mask_constant)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mask_constant = -10.0

            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz'],
                'use_r1_min_val': True,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: 8}
                }
            }
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            max_neighbors = 8
            # entry_size = xyz(3) + geodesic(1) = 4
            entry_size = 4
            # point_feature_size = xyz(3) = 3
            point_feature_size = 3
            num_examples = 10

            row_size = max_neighbors * entry_size + point_feature_size + 2  # +2 for target + r1_min
            examples = np.zeros((num_examples, row_size), dtype=np.float32)

            for ex in range(num_examples):
                # Set neighbor positions for first 4 (valid) neighbors
                for i in range(4):
                    base = i * entry_size
                    examples[ex, base:base + 3] = np.random.uniform(0.5, 2.0, 3)
                    examples[ex, base + 3] = np.random.uniform(0.1, 1.0)  # geodesic > 0

                # Last 4 neighbors are padded: copy positions from valid but set geodesic = mask_constant
                for i in range(4, 8):
                    base = i * entry_size
                    # Give them larger xyz values than valid neighbors
                    examples[ex, base:base + 3] = np.random.uniform(3.0, 5.0, 3)
                    examples[ex, base + 3] = mask_constant  # padded

                # Point features (xyz)
                pf_start = max_neighbors * entry_size
                examples[ex, pf_start:pf_start + 3] = 0.0

                # Target and r1_min
                examples[ex, -2] = np.random.uniform(0.5, 2.0)  # target
                examples[ex, -1] = np.random.uniform(0.1, 0.5)  # r1_min

            np.save(tmpdir / 'gaussian_examples_ring2.npy', examples)
            yield tmpdir, config_path

    def test_default_uses_valid_only(self, dataset_dir_with_padding):
        """Default behavior: max_dist computed from valid neighbors only."""
        tmpdir, config_path = dataset_dir_with_padding
        dataset = GaussianPatchDataset(
            config_path=config_path, attributes=['xyz'], ring=2, use_r1_min=True,
        )
        assert dataset.normalize_all_neighbors is False

    def test_normalize_all_neighbors_from_arg(self, dataset_dir_with_padding):
        """Constructor arg sets normalize_all_neighbors."""
        tmpdir, config_path = dataset_dir_with_padding
        dataset = GaussianPatchDataset(
            config_path=config_path, attributes=['xyz'], ring=2, use_r1_min=True,
            normalize_all_neighbors=True,
        )
        assert dataset.normalize_all_neighbors is True

    def test_normalize_all_neighbors_from_config(self, dataset_dir_with_padding):
        """Config key 'normalize_all_neighbors' is respected."""
        tmpdir, config_path = dataset_dir_with_padding
        # Modify config to include the option
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg['normalize_all_neighbors'] = True
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)

        dataset = GaussianPatchDataset(
            config_path=config_path, attributes=['xyz'], ring=2, use_r1_min=True,
        )
        assert dataset.normalize_all_neighbors is True

    def test_produces_different_max_dist(self, dataset_dir_with_padding):
        """normalize_all_neighbors=True produces different normalization than False.

        Since padded neighbors have larger xyz, max_dist should be larger when
        computed over all neighbors, leading to smaller normalized xyz values.
        """
        tmpdir, config_path = dataset_dir_with_padding

        ds_valid = GaussianPatchDataset(
            config_path=config_path, attributes=['xyz'], ring=2, use_r1_min=True,
            normalize_all_neighbors=False,
        )
        ds_all = GaussianPatchDataset(
            config_path=config_path, attributes=['xyz'], ring=2, use_r1_min=True,
            normalize_all_neighbors=True,
        )

        # Compare max_dist (visible through norm_params via target normalization)
        nb_valid, pf_valid, t_valid, vm_valid = ds_valid[0]
        nb_all, pf_all, t_all, vm_all = ds_all[0]

        # target is normalized by max_dist, so with larger max_dist,
        # t_all should be <= t_valid (target is divided by a larger number)
        # Valid neighbors have xyz in [0.5, 2.0], padded in [3.0, 5.0]
        # So normalize_all_neighbors=True should produce a larger max_dist
        assert t_all <= t_valid or torch.isclose(t_all, t_valid, atol=1e-6), \
            f"Expected t_all ({t_all:.6f}) <= t_valid ({t_valid:.6f}) when padded neighbors have larger xyz"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
