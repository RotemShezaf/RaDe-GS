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


class TestGaussianPatchDataset:
    """Test suite for GaussianPatchDataset class."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create config file
            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'normals', 'opacity', 'scale'],
                'use_r1_min_val': True,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {
                        1: 32,
                        2: 64,
                        3: 128,
                        4: 256
                    }
                }
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Create sample data for ring 2
            # attributes: xyz(3) + normals(3) + opacity(1) + scale(3) = 10
            # entry_size = 10 + 1 (geodesic) = 11
            max_neighbors = 64
            entry_size = 11
            point_feature_size = 10  # no geodesic in point features
            num_examples = 100
            
            # neighbor_features: 64 * 11 = 704
            # point_features: 10
            # r1_min_val: 1
            # p_u: 1
            # Total: 704 + 10 + 1 + 1 = 716
            
            examples = np.random.randn(num_examples, 716).astype(np.float32)
            
            # Set opacity to [0, 1] range
            for i in range(max_neighbors):
                opacity_idx = i * entry_size + 6  # xyz(3) + normals(3) + opacity
                examples[:, opacity_idx] = np.random.uniform(0, 1, num_examples)
            
            # Set point opacity
            point_opacity_idx = max_neighbors * entry_size + 6
            examples[:, point_opacity_idx] = np.random.uniform(0, 1, num_examples)
            
            # Set positive geodesic distances
            examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
            examples[:, -2] = np.random.uniform(0.05, 0.5, num_examples)  # r1_min_val
            
            data_path = tmpdir / 'gaussian_examples_ring2.npy'
            np.save(data_path, examples)
            
            yield tmpdir, config_path
    
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
        features, target = dataset[0]
        # Expected: 64 neighbors * (3 xyz + 1 opacity + 1 geodesic) + (3 xyz + 1 opacity) point 
        # = 64 * 5 + 4 + 1 = 325 features (excluding target)
        assert features.shape[0] == 324
    
    def test_r1_min_removal(self, temp_dataset_dir):
        """Test r1_min_val removal when use_r1_min=False."""
        tmpdir, config_path = temp_dataset_dir
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'normals', 'opacity', 'scale'],
            ring=2,
            use_r1_min=False  # Remove r1_min_val
        )
        
        features, target = dataset[0]
        
        # Should not include r1_min_val in features
        # 64 neighbors * 11 + 10 point = 714 (no r1_min_val)
        assert features.shape[0] == 714
        assert isinstance(target, torch.Tensor)
    
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
            features, target = dataset[i]
            
            # Check shapes
            assert features.dim() == 1
            assert target.dim() == 0  # scalar
            
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
        
        # Call normalization
        dataset._normalize_examples()
        
        # Check opacity values are in [0, 1]
        entry_size = 11
        max_neighbors = 64
        
        for i in range(min(10, len(dataset))):
            example = dataset.examples[i]
            
            # Check neighbor opacities
            for neighbor_idx in range(max_neighbors):
                opacity_idx = neighbor_idx * entry_size + 6
                opacity = example[opacity_idx].item()
                assert 0 <= opacity <= 1, f"Neighbor opacity {opacity} out of range [0, 1]"
            
            # Check point opacity
            point_opacity_idx = max_neighbors * entry_size + 6
            point_opacity = example[point_opacity_idx].item()
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
        
        # Call normalization
        dataset._normalize_examples()
        
        # Check that scales are normalized (should be centered and within reasonable range)
        entry_size = 11
        max_neighbors = 64
        
        for i in range(min(10, len(dataset))):
            example = dataset.examples[i]
            
            # Collect all scale values
            scales = []
            for neighbor_idx in range(max_neighbors):
                scale_idx = neighbor_idx * entry_size + 7  # xyz(3) + normals(3) + opacity(1)
                scales.append(example[scale_idx:scale_idx+3].numpy())
            
            # Point scale
            point_scale_idx = max_neighbors * entry_size + 7
            scales.append(example[point_scale_idx:point_scale_idx+3].numpy())
            
            scales = np.array(scales)
            
            # Check that centroid is close to zero (pc_norm centers data)
            centroid = scales.mean(axis=0)
            assert np.allclose(centroid, 0, atol=1e-5), f"Centroid {centroid} not centered"
            
            # Check that max distance is approximately 1 (pc_norm scales by max distance)
            distances = np.sqrt(np.sum(scales**2, axis=1))
            max_dist = distances.max()
            assert np.isclose(max_dist, 1.0, atol=1e-5), f"Max distance {max_dist} not normalized to 1"


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
        features, targets = batch
        
        assert features.shape[0] <= 8  # batch size
        assert features.shape[1] == 324  # feature dim
        assert targets.shape[0] == features.shape[0]
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
