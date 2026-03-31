# Testing Documentation

This directory contains comprehensive tests for the Gaussian Patch Transformer models.

## Test Structure

```
tests/
├── __init__.py              # Test module initialization
├── README.md                # This file
├── test_utils.py            # Tests for utility functions (utils.py)
├── test_transformer.py      # Tests for transformer components (transformer.py)
├── test_model.py            # Tests for GaussianPatchTransformer model
└── test_integration.py      # Integration tests with real dataset
```

## Running Tests

### Prerequisites

All tests require:
- PyTorch
- timm
- numpy

Integration tests additionally require:
- Access to a GaussianPatchDataset configuration file
- Generated dataset files

### Individual Test Files

Run each test file independently:

```bash
cd /home/rotem.shezaf/RaDe-GS/models/tests

# Test utility functions
python test_utils.py

# Test transformer components
python test_transformer.py

# Test GaussianPatchTransformer model
python test_model.py

# Integration tests (requires dataset)
python test_integration.py --config /path/to/config.yaml --ring 2
```

### All Tests at Once

Run all non-integration tests:

```bash
cd /home/rotem.shezaf/RaDe-GS/models/tests
python test_utils.py && python test_transformer.py && python test_model.py
```

## Test Files

### test_utils.py

Tests for utility functions in `utils.py`:

- **test_get_attribute_dim**: Validates attribute dimension calculation for Gaussian features
- **test_create_mlp**: Tests MLP layer creation with various configurations
- **test_drop_path**: Validates DropPath stochastic depth implementation
- **test_attention**: Tests self-attention mechanism
- **test_cross_attention**: Tests cross-attention between query and context
- **test_feedforward**: Validates feedforward network (MLP) in transformer blocks
- **test_geometric_positional_encoding**: Tests geometric positional encoding from xyz coordinates

**Run:**
```bash
python test_utils.py
```

**Expected output:** All tests pass with ✓ marks

---

### test_transformer.py

Tests for transformer components in `transformer.py`:

- **test_gaussian_patch_encoder**: Tests encoding of neighbor features (with geodesic distances)
- **test_point_feature_encoder**: Tests encoding of central point features (without geodesic)
- **test_transformer_encoder_block**: Tests single transformer encoder block
- **test_transformer_decoder_block**: Tests single transformer decoder block with cross-attention
- **test_transformer_encoder**: Tests full transformer encoder stack
- **test_transformer_decoder**: Tests full transformer decoder stack
- **test_transformer_pipeline**: Tests complete encoder-decoder pipeline end-to-end

**Run:**
```bash
python test_transformer.py
```

**Expected output:** All tests pass, showing shapes at each stage

---

### test_model.py

Tests for the complete GaussianPatchTransformer model:

- **test_model_initialization**: Tests model creation with various configurations
- **test_feature_extraction**: Tests extraction of neighbor and point features from flattened input
- **test_forward_pass**: Tests forward pass with dummy data
- **test_forward_with_embeddings**: Tests forward pass with intermediate embedding extraction
- **test_loss_computation**: Tests different loss functions (MSE, L1, Smooth L1)
- **test_metrics_computation**: Tests evaluation metrics (MAE, RMSE, relative error)
- **test_different_attribute_combinations**: Tests model with various attribute subsets
- **test_model_parameters**: Tests parameter counting for different model sizes
- **test_gradient_flow**: Tests backward pass and gradient computation

**Run:**
```bash
python test_model.py
```

**Expected output:** All tests pass, including gradient flow verification

---

### test_integration.py

Integration tests with real GaussianPatchDataset:

- **test_with_dataset**: Tests model with real dataset examples
  - Loads dataset from config
  - Creates matching model configuration
  - Tests single example and batch predictions
  - Computes loss and metrics
  - Verifies gradient computation

- **test_training_loop**: Tests actual training iterations
  - Creates dataloader
  - Trains for a few epochs
  - Evaluates on full dataset
  - Reports final metrics

**Run:**
```bash
# Basic integration test
python test_integration.py --config /path/to/config.yaml --ring 2

# With training loop test (slower)
python test_integration.py \
    --config /path/to/config.yaml \
    --ring 2 \
    --test_training \
    --num_epochs 3 \
    --batch_size 32
```

**Arguments:**
- `--config`: Path to dataset configuration YAML file (required)
- `--ring`: Ring number to load from dataset (default: 2)
- `--test_training`: Run training loop test
- `--num_epochs`: Number of epochs for training test (default: 3)
- `--batch_size`: Batch size for training test (default: 32)

**Expected output:**
- Dataset information
- Model architecture details
- Predictions on single example and batches
- Loss and metrics
- Gradient flow verification
- (If --test_training) Training progress and final evaluation

---

## Test Coverage

### Utility Functions (utils.py)
- ✓ Attribute dimension calculation
- ✓ MLP creation with various configurations
- ✓ DropPath stochastic depth
- ✓ Self-attention mechanism
- ✓ Cross-attention mechanism
- ✓ Feedforward networks
- ✓ Fetures positional encoding

### Transformer Components (transformer.py)
- ✓ Gaussian patch encoder
- ✓ Point feature encoder
- ✓ Transformer encoder blocks
- ✓ Transformer decoder blocks
- ✓ Full encoder stack with positional encoding
- ✓ Full decoder stack with cross-attention
- ✓ End-to-end pipeline

### GaussianPatchTransformer Model
- ✓ Model initialization and configuration
- ✓ Feature extraction from flattened input
- ✓ Forward pass
- ✓ Embedding extraction
- ✓ Loss computation (MSE, L1, Smooth L1)
- ✓ Metrics computation (MAE, RMSE, relative error, max error)
- ✓ Different attribute combinations
- ✓ Parameter counting
- ✓ Gradient flow

### Integration with Real Data
- ✓ Loading from GaussianPatchDataset
- ✓ Matching model to dataset configuration
- ✓ Single example prediction
- ✓ Batch prediction
- ✓ Loss and metrics on real data
- ✓ Gradient computation with real data
- ✓ Training loop
- ✓ Evaluation

## Common Issues

### Import Errors

If you encounter import errors:

```bash
# Ensure you're in the tests directory
cd /home/rotem.shezaf/RaDe-GS/models/tests

# Or use full python path
PYTHONPATH=/home/rotem.shezaf/RaDe-GS python test_utils.py
```

### CUDA Out of Memory

For integration tests with large batches:

```bash
# Reduce batch size
python test_integration.py --config config.yaml --ring 2 --batch_size 8

# Or run on CPU
CUDA_VISIBLE_DEVICES="" python test_integration.py --config config.yaml --ring 2
```

### Dataset Not Found

For integration tests, ensure:
1. Dataset has been generated using `create_gaussian_training_patches.py`
2. Config YAML file path is correct
3. Ring number matches available data files

## Writing New Tests

To add new tests, follow this pattern:

```python
def test_new_feature():
    """Test description."""
    print("Testing new feature...")
    
    # Setup
    model = create_model()
    input_data = create_dummy_data()
    
    # Execute
    output = model(input_data)
    
    # Verify
    assert output.shape == expected_shape
    assert output satisfies some condition
    
    print("✓ New feature test passed!")
```

Add your test function to the appropriate file and include it in `run_all_tests()`.

## Performance Benchmarks

Expected test execution times (approximate):

- `test_utils.py`: ~2-5 seconds
- `test_transformer.py`: ~5-10 seconds
- `test_model.py`: ~10-20 seconds
- `test_integration.py` (basic): ~30-60 seconds
- `test_integration.py` (with training): ~5-15 minutes (depends on dataset size)

## Continuous Integration

To use these tests in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    cd models/tests
    python test_utils.py
    python test_transformer.py
    python test_model.py
```

For integration tests, you'll need to:
1. Generate or download a test dataset
2. Set up the dataset config path
3. Run with appropriate arguments

## Contributing

When adding new features to the model:

1. Write tests for the new functionality
2. Add tests to the appropriate test file
3. Update this README with test descriptions
4. Ensure all existing tests still pass
5. Run integration tests to verify end-to-end functionality

## Support

If you encounter issues with tests:

1. Check that all dependencies are installed
2. Verify you're using compatible PyTorch version
3. Ensure dataset files are properly generated (for integration tests)
4. Check CUDA availability for GPU tests

For questions or bug reports, please open an issue in the RaDe-GS repository.
