# Tests for Gaussian Dataset and Data Transformations

This directory contains comprehensive tests for the Gaussian patch dataset, data transformation utilities, and multi-source dataset support.

## Test Files

### test_gaussian_dataset.py
Tests for `GaussianPatchDataset` class:
- Dataset initialization and configuration loading
- Attribute filtering functionality
- r1_min_val removal
- Normalization methods (opacity, SH, scale)
- Data loading and shape validation
- Dataloader creation and splitting
- Valid mask functionality
- Transform integration
- `normalize_all_neighbors` option (max_dist over all vs valid-only neighbors)

### test_multi_source_dataset.py
Tests for multi-source dataset functionality:
- `CombinedGaussianPatchDataset` with multi-source config file
- `CombinedGaussianPatchDataset` with list of config paths
- Weighted sampling support
- `create_combined_dataloaders` function
- Data source utilities (`get_data_sources`, `is_multi_source_config`, etc.)
- Backward compatibility with single-source configs
- Config utility functions from `data_transformation_utils.py`

### test_data_transformation.py
Tests for data transformation classes:
- `PointcloudRandomInputDropout`: Random point dropout augmentation
- `GaussianPatchRotate`: Random rotation augmentation
- `GaussianPatchCanonicalRotate`: Canonical rotation based on center of mass
- Transformation composition and integration

### test_rotation.py
Tests for rotation utility functions:
- Quaternion operations
- Rotation matrix conversions
- Canonical rotation alignment

### test_training_patches_helpers.py
Tests for training patch generation helper functions:
- Scale statistics computation
- `merge_config_with_args` — config-to-args merging (including adaptive kNN keys)
- Gaussian normals computation
- Integration: full config merge round-trip
- Parallel example generation
- Memory-mapped dataset loading
- `normalize_neighborhood` / `denormalize_neighborhood`

## Running Tests

### Run all tests:
```bash
cd /home/rotem.shezaf/RaDe-GS/DataSets
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_gaussian_dataset.py -v
python -m pytest tests/test_multi_source_dataset.py -v
python -m pytest tests/test_data_transformation.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_gaussian_dataset.py::TestGaussianPatchDataset -v
python -m pytest tests/test_multi_source_dataset.py::TestCombinedDatasetWithMultiSourceConfig -v
```

### Run specific test method:
```bash
python -m pytest tests/test_gaussian_dataset.py::TestGaussianPatchDataset::test_normalization_opacity -v
python -m pytest tests/test_multi_source_dataset.py::TestConfigUtils::test_multi_source_config -v
```

### Run with coverage:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Coverage

### GaussianPatchDataset Tests:
- ✓ Basic initialization
- ✓ Configuration loading and validation
- ✓ Attribute filtering with subset selection
- ✓ r1_min_val removal
- ✓ Invalid attribute detection
- ✓ Data shape validation
- ✓ Opacity normalization (min-max per example)
- ✓ Scale normalization (pc_norm approach)
- ✓ SH feature transformation
- ✓ Dataloader creation
- ✓ Train/validation split ratios
- ✓ Valid mask functionality

### Normalize All Neighbors Tests:
- ✓ Default uses valid-only neighbors for max_dist
- ✓ Constructor arg sets normalize_all_neighbors
- ✓ Config key 'normalize_all_neighbors' is respected
- ✓ Produces different normalization than valid-only

### Multi-source Dataset Tests:
- ✓ Single-source config parsing
- ✓ Multi-source config parsing
- ✓ Per-source output directories
- ✓ Weight loading from config
- ✓ Combined dataset creation (list of configs)
- ✓ Combined dataset creation (multi-source config)
- ✓ Dataset indexing and source mapping
- ✓ Sample weight generation for WeightedRandomSampler
- ✓ Combined dataloaders creation
- ✓ Weighted sampling in dataloaders
- ✓ Backward compatibility with single-source code

### Data Transformation Tests:
- ✓ Random input dropout
- ✓ Dropout masking behavior
- ✓ Random rotation shape preservation
- ✓ XYZ coordinate rotation
- ✓ Normal vector preservation and normalization
- ✓ Quaternion composition and normalization
- ✓ Opacity preservation during rotation
- ✓ Geodesic distance preservation
- ✓ Canonical rotation alignment
- ✓ Canonical rotation determinism
- ✓ Transformation composition

## Requirements

```bash
pip install pytest pytest-cov
```

## Multi-source Dataset Usage

The multi-source dataset support allows combining training data from multiple surfaces:

### Configuration Format

```yaml
# combined_config.yaml
output_dir: "TrainData/patches/combined"

data_sources:
  - name: "paraboloid"
    gaussian_output: "path/to/paraboloid/output"
    output_dir: "TrainData/patches/paraboloid"  # Per-source output
    weight: 1.0  # Sampling weight
    
  - name: "saddle"
    gaussian_output: "path/to/saddle/output"
    output_dir: "TrainData/patches/saddle"
    weight: 1.5  # Higher weight = more samples

# Shared parameters
attributes: ['xyz', 'normals']
rings: [2, 3]
```

### Generating Data

```bash
# Generate patches for all sources (each gets its own output directory)
python DataSets/create_gaussian_training_patches.py --config combined_config.yaml
```

### Loading Combined Dataset

```python
from DataSets.gaussian_dataset import CombinedGaussianPatchDataset, create_combined_dataloaders

# Load from multi-source config
dataset = CombinedGaussianPatchDataset(
    config='combined_config.yaml',
    attributes=['xyz'],
    ring=2
)

# Or create dataloaders directly
train_loader, val_loader = create_combined_dataloaders(
    config='combined_config.yaml',
    attributes=['xyz'],
    ring=2,
    use_r1_min=False,
    batch_size=32,
    use_weighted_sampling=True  # Use source weights for balanced sampling
)
```

## Notes

- Tests use temporary directories for data generation
- Sample data is created programmatically to ensure consistency
- Tests validate both correctness and numerical stability
- Normalization tests check mathematical properties (centering, scaling)
- Rotation tests verify geometric properties (orthogonality, magnitude preservation)
- Multi-source tests verify correct data aggregation and weight handling
