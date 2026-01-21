# Tests for Gaussian Dataset and Data Transformations

This directory contains comprehensive tests for the Gaussian patch dataset and data transformation utilities.

## Test Files

### test_gaussian_dataset.py
Tests for `GaussianPatchDataset` class:
- Dataset initialization and configuration loading
- Attribute filtering functionality
- r1_min_val removal
- Normalization methods (opacity, SH, scale)
- Data loading and shape validation
- Dataloader creation and splitting

### test_data_transformation.py
Tests for data transformation classes:
- `PointcloudRandomInputDropout`: Random point dropout augmentation
- `GaussianPatchRotate`: Random rotation augmentation
- `GaussianPatchCanonicalRotate`: Canonical rotation based on center of mass
- Transformation composition and integration

## Running Tests

### Run all tests:
```bash
cd /home/rotem.shezaf/RaDe-GS/GenerateData
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_gaussian_dataset.py -v
python -m pytest tests/test_data_transformation.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_gaussian_dataset.py::TestGaussianPatchDataset -v
```

### Run specific test method:
```bash
python -m pytest tests/test_gaussian_dataset.py::TestGaussianPatchDataset::test_normalization_opacity -v
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

## Notes

- Tests use temporary directories for data generation
- Sample data is created programmatically to ensure consistency
- Tests validate both correctness and numerical stability
- Normalization tests check mathematical properties (centering, scaling)
- Rotation tests verify geometric properties (orthogonality, magnitude preservation)
