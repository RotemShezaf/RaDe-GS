#!/usr/bin/env python3
"""
Tests for Multi-source Dataset functionality

Tests cover:
- CombinedGaussianPatchDataset with multi-source config
- CombinedGaussianPatchDataset with list of config paths
- Weighted sampling support
- create_combined_dataloaders function
- Data source utilities in config_utils.py
- Gaussian outputs file (.txt) support for multi-output single datasets
"""

import pytest
import torch
import numpy as np
import tempfile
import yaml
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gaussian_dataset import (
    GaussianPatchDataset,
    CombinedGaussianPatchDataset,
    create_combined_dataloaders,
    create_dataloaders
)
from DataSets.utils.config_utils import (
    load_config,
    get_data_sources,
    is_multi_source_config,
    get_all_source_output_dirs,
    merge_source_config,
    validate_shared_settings,
    SHARED_CONFIG_KEYS,
    is_gaussian_outputs_file,
    load_gaussian_outputs_file,
    resolve_gaussian_outputs,
)
from DataSets.utils.data_transformation_utils import get_entry_size


class TestConfigUtils:
    """Test suite for config utility functions."""
    
    def test_single_source_config(self):
        """Test get_data_sources with single-source config."""
        config = {
            'gaussian_output': '/path/to/output',
            'geodesic_data': '/path/to/geodesic.npz',
            'output_dir': '/path/to/patches'
        }
        
        sources = get_data_sources(config)
        assert len(sources) == 1
        assert sources[0]['gaussian_output'] == '/path/to/output'
        assert sources[0]['output_dir'] == '/path/to/patches'
        assert sources[0]['weight'] == 1.0
        assert sources[0]['name'] == 'default'
    
    def test_multi_source_config(self):
        """Test get_data_sources with multi-source config."""
        config = {
            'output_dir': '/base/path',
            'data_sources': [
                {
                    'name': 'source1',
                    'gaussian_output': '/path/to/output1',
                    'weight': 2.0,
                    'output_dir': '/custom/path1'
                },
                {
                    'name': 'source2',
                    'gaussian_output': '/path/to/output2',
                    'weight': 0.5
                    # No output_dir - should derive from base
                }
            ]
        }
        
        sources = get_data_sources(config)
        assert len(sources) == 2
        
        # First source with explicit output_dir
        assert sources[0]['name'] == 'source1'
        assert sources[0]['gaussian_output'] == '/path/to/output1'
        assert sources[0]['output_dir'] == '/custom/path1'
        assert sources[0]['weight'] == 2.0
        
        # Second source with derived output_dir
        assert sources[1]['name'] == 'source2'
        assert sources[1]['output_dir'] == '/base/path/source2'
        assert sources[1]['weight'] == 0.5
    
    def test_is_multi_source_config(self):
        """Test is_multi_source_config detection."""
        single_config = {'gaussian_output': '/path', 'output_dir': '/out'}
        multi_config = {'data_sources': [{'name': 'test'}]}
        
        assert not is_multi_source_config(single_config)
        assert is_multi_source_config(multi_config)
    
    def test_get_all_source_output_dirs(self):
        """Test getting all output directories from config."""
        config = {
            'output_dir': '/base',
            'data_sources': [
                {'name': 'src1', 'output_dir': '/custom/src1'},
                {'name': 'src2'}  # Will derive as /base/src2
            ]
        }
        
        dirs = get_all_source_output_dirs(config)
        assert len(dirs) == 2
        assert dirs[0] == Path('/custom/src1')
        assert dirs[1] == Path('/base/src2')
    
    def test_merge_source_config(self):
        """Test merging master config with per-source settings."""
        master_config = {
            'attributes': ['xyz', 'opacity'],
            'rings': [2, 3],
            'ring_size_mapping': {'euclidean': {2: 32}},
            'use_mahalanobis': False,
            'use_r1_min_val': True,
            'mask_constant': -10.0,
            'output_dir': '/base'
        }
        
        source = {
            'name': 'paraboloid',
            'gaussian_output': '/path/to/gaussian',
            'output_dir': '/path/to/output',
            'weight': 1.5
        }
        
        merged = merge_source_config(master_config, source)
        
        # Shared settings from master
        assert merged['attributes'] == ['xyz', 'opacity']
        assert merged['rings'] == [2, 3]
        assert merged['use_mahalanobis'] == False
        assert merged['use_r1_min_val'] == True
        
        # Per-source settings
        assert merged['output_dir'] == '/path/to/output'
        assert merged['gaussian_output'] == '/path/to/gaussian'
        assert merged['source_name'] == 'paraboloid'
        assert merged['weight'] == 1.5
    
    def test_validate_shared_settings_pass(self):
        """Test validation passes when settings are identical."""
        configs = [
            {'attributes': ['xyz'], 'rings': [2], 'use_mahalanobis': False},
            {'attributes': ['xyz'], 'rings': [2], 'use_mahalanobis': False}
        ]
        
        # Should not raise
        validate_shared_settings(configs)
    
    def test_validate_shared_settings_fail(self):
        """Test validation fails when settings differ."""
        configs = [
            {'attributes': ['xyz'], 'rings': [2], 'use_mahalanobis': False},
            {'attributes': ['xyz', 'opacity'], 'rings': [2], 'use_mahalanobis': False}
        ]
        
        with pytest.raises(ValueError, match="Shared setting 'attributes' differs"):
            validate_shared_settings(configs)


class TestCombinedDatasetWithConfigList:
    """Test suite for CombinedGaussianPatchDataset with list of config paths."""
    
    @pytest.fixture
    def two_source_datasets(self):
        """Create two separate datasets for combined testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Common settings
            max_neighbors = 32
            entry_size = 5  # xyz(3) + opacity(1) + geodesic(1)
            point_feature_size = 4  # xyz(3) + opacity(1)
            
            configs = []
            config_paths = []
            
            for source_idx, (name, num_examples) in enumerate([('source_a', 50), ('source_b', 30)]):
                source_dir = tmpdir / name
                source_dir.mkdir()
                
                config = {
                    'output_dir': str(source_dir),
                    'attributes': ['xyz', 'opacity'],
                    'use_r1_min_val': False,
                    'use_mahalanobis': False,
                    'ring_size_mapping': {
                        'euclidean': {2: max_neighbors}
                    }
                }
                
                config_path = source_dir / 'config.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Create data
                # Total: 32*5 + 4 + 1 = 165
                examples = np.random.randn(num_examples, 165).astype(np.float32)
                examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
                
                data_path = source_dir / 'gaussian_examples_ring2.npy'
                np.save(data_path, examples)
                
                configs.append(config)
                config_paths.append(config_path)
            
            yield tmpdir, config_paths, configs
    
    def test_combined_dataset_creation(self, two_source_datasets):
        """Test creating combined dataset from list of configs."""
        tmpdir, config_paths, configs = two_source_datasets
        
        dataset = CombinedGaussianPatchDataset(
            config=config_paths,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == 80  # 50 + 30
        assert len(dataset.datasets) == 2
        assert dataset.dataset_names == ['config', 'config']  # stem of config.yaml
    
    def test_combined_dataset_indexing(self, two_source_datasets):
        """Test that indexing correctly maps to source datasets."""
        tmpdir, config_paths, configs = two_source_datasets
        
        dataset = CombinedGaussianPatchDataset(
            config=config_paths,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        # Index 0-49 should come from first dataset
        for i in range(50):
            assert dataset.get_source_index(i) == 0
        
        # Index 50-79 should come from second dataset
        for i in range(50, 80):
            assert dataset.get_source_index(i) == 1
        
        # Test data access
        neighborhood, point_features, target, valid_mask = dataset[0]
        assert neighborhood.shape == (32, 5)
        
        neighborhood, point_features, target, valid_mask = dataset[75]
        assert neighborhood.shape == (32, 5)
    
    def test_combined_dataset_with_weights(self, two_source_datasets):
        """Test sample weights for weighted sampling."""
        tmpdir, config_paths, configs = two_source_datasets
        
        weights = [2.0, 0.5]
        dataset = CombinedGaussianPatchDataset(
            config=config_paths,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            weights=weights
        )
        
        sample_weights = dataset.get_sample_weights()
        assert len(sample_weights) == 80
        
        # First 50 samples should have weight 2.0
        assert all(sample_weights[:50] == 2.0)
        
        # Last 30 samples should have weight 0.5
        assert all(sample_weights[50:] == 0.5)


class TestCombinedDatasetWithMultiSourceConfig:
    """Test suite for CombinedGaussianPatchDataset with multi-source config file.
    
    These tests verify in-memory config merging - NO separate config.yaml per source is needed.
    """
    
    @pytest.fixture
    def multi_source_config_setup(self):
        """Create multi-source config and data directories.
        
        Note: No per-source config.yaml files are created - the combined dataset
        uses in-memory config merging from the master config.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Common settings (will be merged into each source config in-memory)
            max_neighbors = 32
            entry_size = 5
            
            source_dirs = []
            for name, num_examples, weight in [
                ('paraboloid', 40, 1.0),
                ('saddle', 60, 1.5),
                ('sphere', 20, 0.5)
            ]:
                source_dir = tmpdir / name
                source_dir.mkdir()
                source_dirs.append((name, source_dir, num_examples, weight))
                
                # Create data files (NO config.yaml per source - that's the point!)
                examples = np.random.randn(num_examples, 165).astype(np.float32)
                examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
                np.save(source_dir / 'gaussian_examples_ring2.npy', examples)
                
                # Create metadata (optional, for print_info)
                metadata = {
                    'surface': {'name': name, 'texture': 'blue'},
                    'gaussian_data': {'num_gaussians': 1000}
                }
                with open(source_dir / 'generation_metadata.json', 'w') as f:
                    json.dump(metadata, f)
            
            # Create multi-source config file with ALL shared settings
            # This is the SINGLE SOURCE OF TRUTH
            multi_config = {
                'output_dir': str(tmpdir),
                
                # Shared settings (apply to all sources)
                'attributes': ['xyz', 'opacity'],
                'rings': [2],
                'use_mahalanobis': False,
                'use_r1_min_val': False,
                'mask_constant': -10.0,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                },
                
                # Per-source settings
                'data_sources': [
                    {
                        'name': name,
                        'gaussian_output': str(source_dir / 'gaussian_output'),
                        'output_dir': str(source_dir),
                        'weight': weight
                    }
                    for name, source_dir, _, weight in source_dirs
                ]
            }
            
            multi_config_path = tmpdir / 'combined_config.yaml'
            with open(multi_config_path, 'w') as f:
                yaml.dump(multi_config, f)
            
            yield tmpdir, multi_config_path, source_dirs
    
    def test_multi_source_config_loading(self, multi_source_config_setup):
        """Test loading combined dataset from multi-source config."""
        tmpdir, multi_config_path, source_dirs = multi_source_config_setup
        
        dataset = CombinedGaussianPatchDataset(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == 120  # 40 + 60 + 20
        assert len(dataset.datasets) == 3
        assert dataset.dataset_names == ['paraboloid', 'saddle', 'sphere']
    
    def test_multi_source_weights_from_config(self, multi_source_config_setup):
        """Test that weights are correctly loaded from multi-source config."""
        tmpdir, multi_config_path, source_dirs = multi_source_config_setup
        
        dataset = CombinedGaussianPatchDataset(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        expected_weights = [1.0, 1.5, 0.5]
        assert dataset.dataset_weights == expected_weights
        
        sample_weights = dataset.get_sample_weights()
        assert len(sample_weights) == 120
        
        # Check weight distribution
        assert all(sample_weights[:40] == 1.0)   # paraboloid
        assert all(sample_weights[40:100] == 1.5)  # saddle
        assert all(sample_weights[100:] == 0.5)   # sphere
    
    def test_multi_source_print_info(self, multi_source_config_setup, capsys):
        """Test print_info method for combined dataset."""
        tmpdir, multi_config_path, source_dirs = multi_source_config_setup
        
        dataset = CombinedGaussianPatchDataset(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        dataset.print_info()
        captured = capsys.readouterr()
        
        assert 'Combined Gaussian Patch Dataset' in captured.out
        assert 'Total sources: 3' in captured.out
        assert 'paraboloid' in captured.out
        assert 'saddle' in captured.out
        assert 'sphere' in captured.out
        assert 'weight=1.0' in captured.out
        assert 'weight=1.5' in captured.out
        assert 'weight=0.5' in captured.out


class TestCreateCombinedDataloaders:
    """Test suite for create_combined_dataloaders function."""
    
    @pytest.fixture
    def combined_dataset_config(self):
        """Create config for combined dataloaders testing (in-memory merging)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            max_neighbors = 32
            
            # Create two sources - data files only, no per-source config.yaml
            for name, num_examples in [('src1', 100), ('src2', 50)]:
                source_dir = tmpdir / name
                source_dir.mkdir()
                
                # Only create data files - config is merged from master in-memory
                examples = np.random.randn(num_examples, 165).astype(np.float32)
                examples[:, -1] = np.random.uniform(0.1, 2.0, num_examples)
                np.save(source_dir / 'gaussian_examples_ring2.npy', examples)
            
            # Create multi-source config with ALL settings (single source of truth)
            multi_config = {
                'output_dir': str(tmpdir),
                
                # Shared settings
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                },
                
                # Per-source settings
                'data_sources': [
                    {'name': 'src1', 'output_dir': str(tmpdir / 'src1'), 'weight': 1.0},
                    {'name': 'src2', 'output_dir': str(tmpdir / 'src2'), 'weight': 2.0}
                ]
            }
            
            multi_config_path = tmpdir / 'combined.yaml'
            with open(multi_config_path, 'w') as f:
                yaml.dump(multi_config, f)
            
            yield tmpdir, multi_config_path
    
    def test_create_combined_dataloaders_basic(self, combined_dataset_config):
        """Test basic dataloader creation from combined config."""
        tmpdir, multi_config_path = combined_dataset_config
        
        train_loader, val_loader = create_combined_dataloaders(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=16,
            train_split=0.8,
            num_workers=0
        )
        
        # Check total examples
        train_count = sum(batch[0].shape[0] for batch in train_loader)
        val_count = sum(batch[0].shape[0] for batch in val_loader)
        
        assert train_count + val_count == 150
        assert 0.75 <= train_count / 150 <= 0.85
    
    def test_create_combined_dataloaders_with_weighted_sampling(self, combined_dataset_config):
        """Test dataloader creation with weighted sampling."""
        tmpdir, multi_config_path = combined_dataset_config
        
        train_loader, val_loader = create_combined_dataloaders(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=16,
            train_split=0.8,
            num_workers=0,
            use_weighted_sampling=True
        )
        
        # Just check that dataloaders work
        batch = next(iter(train_loader))
        assert batch[0].shape[0] <= 16
        assert batch[0].shape[1] == 32  # max_neighbors
        assert batch[0].shape[2] == 5   # entry_size
    
    def test_create_combined_dataloaders_batch_shapes(self, combined_dataset_config):
        """Test batch shapes from combined dataloaders."""
        tmpdir, multi_config_path = combined_dataset_config
        
        train_loader, val_loader = create_combined_dataloaders(
            config=multi_config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=8,
            num_workers=0
        )
        
        neighborhoods, point_features, targets, valid_masks = next(iter(train_loader))
        
        assert neighborhoods.shape == (8, 32, 5)  # (batch, max_neighbors, entry_size)
        assert point_features.shape == (8, 4)     # (batch, point_feature_size)
        assert targets.shape == (8,)              # (batch,)
        assert valid_masks.shape == (8, 32)       # (batch, max_neighbors)


class TestDataTransformationUtilsIntegration:
    """Test suite for data_transformation_utils integration."""
    
    def test_get_entry_size_basic(self):
        """Test get_entry_size with various attribute combinations."""
        from DataSets.utils.data_transformation_utils import get_entry_size
        
        # xyz only
        assert get_entry_size(['xyz']) == 4  # 3 + 1 (geodesic)
        assert get_entry_size(['xyz'], include_geodesic=False) == 3
        
        # Multiple attributes
        assert get_entry_size(['xyz', 'opacity']) == 5  # 3 + 1 + 1
        assert get_entry_size(['xyz', 'normals', 'opacity']) == 8  # 3 + 3 + 1 + 1
        assert get_entry_size(['xyz', 'rotation']) == 8  # 3 + 4 + 1
    
    def test_get_masked_entry_values(self):
        """Test get_masked_entry produces correct masked values."""
        from DataSets.utils.data_transformation_utils import get_masked_entry
        
        mask_constant = -10.0
        masked = get_masked_entry(['xyz', 'opacity', 'rotation'], mask_constant)
        
        assert masked.shape == (1, 9)  # 3 + 1 + 4 + 1
        
        # xyz should be mask_constant
        assert torch.all(masked[0, :3] == mask_constant)
        
        # opacity should be 0.0
        assert masked[0, 3] == 0.0
        
        # rotation should be identity quaternion [1, 0, 0, 0]
        assert torch.allclose(masked[0, 4:8], torch.tensor([1., 0., 0., 0.]))
        
        # geodesic should be mask_constant
        assert masked[0, -1] == mask_constant
    
    def test_get_point_feature_size(self):
        """Test get_point_feature_size excludes geodesic."""
        from DataSets.utils.data_transformation_utils import get_point_feature_size
        
        assert get_point_feature_size(['xyz']) == 3
        assert get_point_feature_size(['xyz', 'opacity']) == 4
        assert get_point_feature_size(['xyz', 'normals', 'rotation']) == 10


class TestBackwardCompatibility:
    """Test suite for backward compatibility with existing code."""
    
    @pytest.fixture
    def single_source_dataset(self):
        """Create single-source dataset for backward compatibility testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: 32}
                }
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            examples = np.random.randn(50, 165).astype(np.float32)
            examples[:, -1] = np.random.uniform(0.1, 2.0, 50)
            np.save(tmpdir / 'gaussian_examples_ring2.npy', examples)
            
            yield tmpdir, config_path
    
    def test_single_config_to_combined_dataset(self, single_source_dataset):
        """Test that single config can be loaded via CombinedGaussianPatchDataset."""
        tmpdir, config_path = single_source_dataset
        
        # Should work with single config path
        dataset = CombinedGaussianPatchDataset(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == 50
        assert len(dataset.datasets) == 1
    
    def test_regular_dataset_still_works(self, single_source_dataset):
        """Test that GaussianPatchDataset still works as before."""
        tmpdir, config_path = single_source_dataset
        
        dataset = GaussianPatchDataset(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == 50
        neighborhood, point_features, target, valid_mask = dataset[0]
        assert neighborhood.shape == (32, 5)
    
    def test_create_dataloaders_unchanged(self, single_source_dataset):
        """Test that create_dataloaders function still works."""
        tmpdir, config_path = single_source_dataset
        
        train_loader, val_loader = create_dataloaders(
            config_path=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=8,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        assert len(batch) == 4  # neighborhood, point_features, target, valid_mask


class TestConfigValidation:
    """Test suite for config validation in combined datasets."""
    
    @pytest.fixture
    def mismatched_config_setup(self):
        """Create two datasets with mismatched shared settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            max_neighbors = 32
            
            # Create first source with ['xyz', 'opacity']
            source1_dir = tmpdir / 'source1'
            source1_dir.mkdir()
            
            config1 = {
                'output_dir': str(source1_dir),
                'attributes': ['xyz', 'opacity'],  # Different attributes!
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                }
            }
            config1_path = source1_dir / 'config.yaml'
            with open(config1_path, 'w') as f:
                yaml.dump(config1, f)
            
            examples1 = np.random.randn(30, 165).astype(np.float32)
            examples1[:, -1] = np.random.uniform(0.1, 2.0, 30)
            np.save(source1_dir / 'gaussian_examples_ring2.npy', examples1)
            
            # Create second source with ['xyz'] only - MISMATCHED!
            source2_dir = tmpdir / 'source2'
            source2_dir.mkdir()
            
            config2 = {
                'output_dir': str(source2_dir),
                'attributes': ['xyz'],  # Different attributes!
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                }
            }
            config2_path = source2_dir / 'config.yaml'
            with open(config2_path, 'w') as f:
                yaml.dump(config2, f)
            
            examples2 = np.random.randn(20, 129).astype(np.float32)  # Different size
            examples2[:, -1] = np.random.uniform(0.1, 2.0, 20)
            np.save(source2_dir / 'gaussian_examples_ring2.npy', examples2)
            
            yield tmpdir, [config1_path, config2_path]
    
    def test_combined_dataset_rejects_mismatched_attributes(self, mismatched_config_setup):
        """Test that combining datasets with different attributes raises an error."""
        tmpdir, config_paths = mismatched_config_setup
        
        with pytest.raises(ValueError, match="Shared setting 'attributes' differs"):
            CombinedGaussianPatchDataset(
                config=config_paths,
                attributes=['xyz'],  # Request subset
                ring=2,
                use_r1_min=False
            )
    
    def test_dict_config_for_dataset(self):
        """Test that GaussianPatchDataset accepts dict config directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            max_neighbors = 32
            
            # Create data file
            examples = np.random.randn(50, 165).astype(np.float32)
            examples[:, -1] = np.random.uniform(0.1, 2.0, 50)
            np.save(tmpdir / 'gaussian_examples_ring2.npy', examples)
            
            # Pass config as dict (simulating in-memory merge)
            config_dict = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                }
            }
            
            # Use new 'config' parameter with dict
            dataset = GaussianPatchDataset(
                config=config_dict,
                attributes=['xyz', 'opacity'],
                ring=2,
                use_r1_min=False
            )
            
            assert len(dataset) == 50
            assert dataset._config_source == 'dict'


class TestGaussianOutputsFile:
    """Test suite for Gaussian outputs file (.txt) utility functions."""
    
    def test_is_gaussian_outputs_file_txt(self):
        """Test that .txt files are correctly identified."""
        assert is_gaussian_outputs_file("path/to/outputs.txt") == True
        assert is_gaussian_outputs_file("/absolute/path/gaussian_outputs.txt") == True
    
    def test_is_gaussian_outputs_file_non_txt(self):
        """Test that non-.txt paths are not identified as outputs files."""
        assert is_gaussian_outputs_file("path/to/output") == False
        assert is_gaussian_outputs_file("path/to/output.yaml") == False
        assert is_gaussian_outputs_file("path/to/output.npy") == False
    
    def test_is_gaussian_outputs_file_none(self):
        """Test that None returns False."""
        assert is_gaussian_outputs_file(None) == False
    
    def test_load_gaussian_outputs_file(self):
        """Test loading paths from a .txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            txt_path = tmpdir / "gaussian_outputs.txt"
            
            txt_path.write_text(
                "/path/to/output1\n"
                "/path/to/output2\n"
                "/path/to/output3\n"
            )
            
            paths = load_gaussian_outputs_file(txt_path)
            assert len(paths) == 3
            assert paths[0] == "/path/to/output1"
            assert paths[1] == "/path/to/output2"
            assert paths[2] == "/path/to/output3"
    
    def test_load_gaussian_outputs_file_with_comments_and_blanks(self):
        """Test that comments and blank lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            txt_path = tmpdir / "gaussian_outputs.txt"
            
            txt_path.write_text(
                "# This is a comment\n"
                "/path/to/output1\n"
                "\n"
                "# Another comment\n"
                "/path/to/output2\n"
                "   \n"
            )
            
            paths = load_gaussian_outputs_file(txt_path)
            assert len(paths) == 2
            assert paths[0] == "/path/to/output1"
            assert paths[1] == "/path/to/output2"
    
    def test_load_gaussian_outputs_file_not_found(self):
        """Test FileNotFoundError for missing .txt file."""
        with pytest.raises(FileNotFoundError):
            load_gaussian_outputs_file("/nonexistent/path/outputs.txt")
    
    def test_load_gaussian_outputs_file_empty(self):
        """Test ValueError for empty .txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            txt_path = tmpdir / "empty.txt"
            txt_path.write_text("# only comments\n\n")
            
            with pytest.raises(ValueError, match="No valid paths"):
                load_gaussian_outputs_file(txt_path)
    
    def test_resolve_gaussian_outputs_single_path(self):
        """Test resolve returns single-element list for a regular path."""
        result = resolve_gaussian_outputs("/path/to/output")
        assert result == ["/path/to/output"]
    
    def test_resolve_gaussian_outputs_txt_file(self):
        """Test resolve loads paths from .txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            txt_path = tmpdir / "outputs.txt"
            txt_path.write_text("/path/a\n/path/b\n")
            
            result = resolve_gaussian_outputs(str(txt_path))
            assert result == ["/path/a", "/path/b"]
    
    def test_resolve_gaussian_outputs_none(self):
        """Test resolve returns empty list for None."""
        result = resolve_gaussian_outputs(None)
        assert result == []


class TestGaussianOutputsFileInConfig:
    """Test that gaussian_output .txt files work in config contexts."""
    
    def test_single_source_config_with_txt_file(self):
        """Test get_data_sources with a .txt file as gaussian_output."""
        config = {
            'gaussian_output': '/path/to/outputs.txt',
            'output_dir': '/path/to/patches'
        }
        
        sources = get_data_sources(config)
        assert len(sources) == 1
        assert sources[0]['gaussian_output'] == '/path/to/outputs.txt'
    
    def test_multi_source_config_with_txt_file_in_source(self):
        """Test get_data_sources with a .txt file in one of the data_sources."""
        config = {
            'output_dir': '/base/path',
            'data_sources': [
                {
                    'name': 'multi_output_source',
                    'gaussian_output': '/path/to/outputs.txt',
                    'output_dir': '/custom/path1',
                    'weight': 1.0,
                },
                {
                    'name': 'single_source',
                    'gaussian_output': '/path/to/single_output',
                    'output_dir': '/custom/path2',
                    'weight': 2.0,
                }
            ]
        }
        
        sources = get_data_sources(config)
        assert len(sources) == 2
        assert sources[0]['gaussian_output'] == '/path/to/outputs.txt'
        assert is_gaussian_outputs_file(sources[0]['gaussian_output'])
        assert sources[1]['gaussian_output'] == '/path/to/single_output'
        assert not is_gaussian_outputs_file(sources[1]['gaussian_output'])
    
    @pytest.fixture
    def multi_output_dataset(self):
        """Create a dataset generated from multiple Gaussian outputs (simulated).
        
        Simulates the result of create_gaussian_training_patches.py processing
        a .txt file with multiple Gaussian output paths. The combined .npy file
        is already created (as the script would produce).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            max_neighbors = 32
            
            # Simulate 3 Gaussian outputs each contributing examples
            # After combination, all examples are in one .npy file
            num_examples_per_output = [20, 30, 15]
            total_examples = sum(num_examples_per_output)
            
            # Create combined data (as the creation script would produce)
            examples = np.random.randn(total_examples, 165).astype(np.float32)
            examples[:, -1] = np.random.uniform(0.1, 2.0, total_examples)
            np.save(tmpdir / f'gaussian_examples_ring2_n{total_examples}.npy', examples)
            
            # Create the txt file listing the original Gaussian outputs
            txt_path = tmpdir / "gaussian_outputs.txt"
            txt_path.write_text(
                "/path/to/gaussian/output1\n"
                "/path/to/gaussian/output2\n"
                "/path/to/gaussian/output3\n"
            )
            
            # Create config (as the creation script would produce)
            config = {
                'output_dir': str(tmpdir),
                'gaussian_output': str(txt_path),
                'gaussian_output_paths': [
                    '/path/to/gaussian/output1',
                    '/path/to/gaussian/output2',
                    '/path/to/gaussian/output3',
                ],
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                },
                'num_gaussian_outputs': 3,
            }
            
            config_path = tmpdir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            yield tmpdir, config_path, total_examples
    
    def test_dataset_from_multi_output(self, multi_output_dataset):
        """Test loading a dataset generated from multiple Gaussian outputs."""
        tmpdir, config_path, total_examples = multi_output_dataset
        
        dataset = GaussianPatchDataset(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == total_examples
        
        # Verify data access works
        neighborhood, point_features, target, valid_mask = dataset[0]
        assert neighborhood.shape == (32, 5)
        assert point_features.shape[0] == 4
    
    def test_dataset_from_multi_output_config_has_txt_reference(self, multi_output_dataset):
        """Test that the saved config has the original .txt file reference."""
        tmpdir, config_path, _ = multi_output_dataset
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert is_gaussian_outputs_file(config['gaussian_output'])
        assert config['num_gaussian_outputs'] == 3
        assert len(config['gaussian_output_paths']) == 3
    
    def test_dataloader_from_multi_output(self, multi_output_dataset):
        """Test creating dataloaders from a multi-output dataset."""
        tmpdir, config_path, total_examples = multi_output_dataset
        
        train_loader, val_loader = create_dataloaders(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False,
            batch_size=8,
            train_split=0.8,
            num_workers=0
        )
        
        # Verify total examples match
        train_count = sum(batch[0].shape[0] for batch in train_loader)
        val_count = sum(batch[0].shape[0] for batch in val_loader)
        assert train_count + val_count == total_examples


class TestMultiOutputInCombinedDataset:
    """Test that multi-output sources work inside CombinedGaussianPatchDataset."""
    
    @pytest.fixture
    def combined_with_multi_output(self):
        """Create combined dataset where one source uses multi-output.
        
        Source 1: Regular single-output dataset (40 examples)
        Source 2: Multi-output dataset (60 examples combined from 3 outputs)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            max_neighbors = 32
            
            # Source 1: regular single-output
            src1_dir = tmpdir / 'source1'
            src1_dir.mkdir()
            examples1 = np.random.randn(40, 165).astype(np.float32)
            examples1[:, -1] = np.random.uniform(0.1, 2.0, 40)
            np.save(src1_dir / 'gaussian_examples_ring2_n40.npy', examples1)
            
            # Source 2: simulated multi-output (already combined)
            src2_dir = tmpdir / 'source2'
            src2_dir.mkdir()
            examples2 = np.random.randn(60, 165).astype(np.float32)
            examples2[:, -1] = np.random.uniform(0.1, 2.0, 60)
            np.save(src2_dir / 'gaussian_examples_ring2_n60.npy', examples2)
            
            # Create txt file for source 2
            txt_path = tmpdir / "source2_outputs.txt"
            txt_path.write_text("/path/to/output_a\n/path/to/output_b\n/path/to/output_c\n")
            
            # Create multi-source config
            multi_config = {
                'output_dir': str(tmpdir),
                'attributes': ['xyz', 'opacity'],
                'use_r1_min_val': False,
                'use_mahalanobis': False,
                'ring_size_mapping': {
                    'euclidean': {2: max_neighbors}
                },
                'data_sources': [
                    {
                        'name': 'single_output',
                        'gaussian_output': '/path/to/single/output',
                        'output_dir': str(src1_dir),
                        'weight': 1.0,
                    },
                    {
                        'name': 'multi_output',
                        'gaussian_output': str(txt_path),
                        'output_dir': str(src2_dir),
                        'weight': 1.5,
                    }
                ]
            }
            
            config_path = tmpdir / 'combined.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(multi_config, f)
            
            yield tmpdir, config_path
    
    def test_combined_dataset_with_multi_output_source(self, combined_with_multi_output):
        """Test that CombinedGaussianPatchDataset works with multi-output sources."""
        tmpdir, config_path = combined_with_multi_output
        
        dataset = CombinedGaussianPatchDataset(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        assert len(dataset) == 100  # 40 + 60
        assert len(dataset.datasets) == 2
        assert dataset.dataset_names == ['single_output', 'multi_output']
    
    def test_combined_weights_with_multi_output(self, combined_with_multi_output):
        """Test that weights are correct in the combined dataset."""
        tmpdir, config_path = combined_with_multi_output
        
        dataset = CombinedGaussianPatchDataset(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        sample_weights = dataset.get_sample_weights()
        assert len(sample_weights) == 100
        
        # First 40 have weight 1.0 (single_output)
        assert all(sample_weights[:40] == 1.0)
        # Last 60 have weight 1.5 (multi_output)
        assert all(sample_weights[40:] == 1.5)
    
    def test_data_access_across_sources(self, combined_with_multi_output):
        """Test that data can be accessed from both sources."""
        tmpdir, config_path = combined_with_multi_output
        
        dataset = CombinedGaussianPatchDataset(
            config=config_path,
            attributes=['xyz', 'opacity'],
            ring=2,
            use_r1_min=False
        )
        
        # Access from source 1
        neighborhood, point_features, target, valid_mask = dataset[0]
        assert neighborhood.shape == (32, 5)
        
        # Access from source 2
        neighborhood, point_features, target, valid_mask = dataset[50]
        assert neighborhood.shape == (32, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
