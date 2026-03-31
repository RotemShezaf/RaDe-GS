# Using Transforms with GaussianPatchDataset

## Overview

The `GaussianPatchDataset` supports data augmentation through transforms. Transforms modify the input data during training to improve model robustness and generalization.

## Quick Start

```python
from gaussian_dataset import GaussianPatchDataset
from data_transformation import GaussianPatchDropout, GaussianPatchRotate

# Create a transform
transform = GaussianPatchDropout(
    max_dropout_ratio=0.3,
    attributes=["xyz", "normals", "opacity"],
    max_neighbors=64,
    use_r1_min_val=True,
    mask_constant=-10
)

# Create dataset with transform
dataset = GaussianPatchDataset(
    config_path="config.yaml",
    attributes=["xyz", "normals", "opacity"],
    ring=2,
    use_r1_min=True,
    transform=transform  # Pass transform here
)

# Use in training
for features, target in dataset:
    # Features are automatically augmented
    pass
```

## Available Transforms

### 1. GaussianPatchDropout
Randomly drops neighbors whose geodesic distance is greater than `r1_min_val`.

**Parameters:**
- `max_dropout_ratio`: Maximum ratio of neighbors to drop (0 to 0.99)
- `attributes`: List of attributes (must match dataset)
- `max_neighbors`: Number of neighbors in patch
- `use_r1_min_val`: Whether r1_min_val is available
- `mask_constant`: Value to use for masked entries (default: -10)

**Example:**
```python
dropout = GaussianPatchDropout(
    max_dropout_ratio=0.3,  # Drop up to 30% of far neighbors
    attributes=["xyz", "normals", "opacity"],
    max_neighbors=64,
    use_r1_min_val=True,
    mask_constant=-10
)
```

### 2. GaussianPatchRotate
Applies random rotation around the y-axis to XYZ coordinates, normals, and quaternions.

**Parameters:**
- `attributes`: List of attributes to rotate
- `max_neighbors`: Number of neighbors in patch
- `use_r1_min_val`: Whether r1_min_val is in the data

**Example:**
```python
rotation = GaussianPatchRotate(
    attributes=["xyz", "normals", "opacity", "rotation"],
    max_neighbors=64,
    use_r1_min_val=True
)
```

### 3. GaussianPatchCanonicalRotate
Applies deterministic canonical rotation to align center of mass with a target direction.

**Parameters:**
- `attributes`: List of attributes to rotate
- `max_neighbors`: Number of neighbors in patch
- `use_r1_min_val`: Whether r1_min_val is in the data
- `target_direction`: Target direction vector (default: [0, 1, 0])

**Example:**
```python
canonical = GaussianPatchCanonicalRotate(
    attributes=["xyz", "normals", "opacity"],
    max_neighbors=64,
    use_r1_min_val=True,
    target_direction=[0., 1., 0.]
)
```

### 4. SparseContextDropout
Simulates the Fast Marching inference regime where most neighbors are unvisited.
Randomly keeps only K valid neighbors (K drawn from a small-value distribution),
masking the rest. Kept neighbors are the closest by geodesic distance (wavefront order).
This bridges the train–inference distribution gap for near-source predictions.

**Parameters:**
- `min_valid`: Minimum number of valid neighbors to keep (default: 1)
- `max_valid_ratio`: Maximum fraction of valid neighbors to keep when applied (default: 0.3)
- `p`: Probability of applying this transform per example (default: 0.3)
- `attributes`: List of attributes
- `mask_constant`: Sentinel value for masked geodesic distance (default: -10.0)

**Example:**
```python
sparse_dropout = SparseContextDropout(
    min_valid=1,
    max_valid_ratio=0.3,
    p=0.3,
    attributes=["xyz"],
    mask_constant=-10.0,
)
```

**YAML config:**
```yaml
transforms:
  - name: SparseContextDropout
    min_valid: 1
    max_valid_ratio: 0.3
    p: 0.3
```

### 5. GaussianPatchSurfacePerturb
Adds small random noise along the surface normal at each neighbor, simulating
off-surface Gaussian displacements in reconstructed scenes.

When the dataset includes `_aug_normals`, perturbation is **directed along the normal**.
When `_aug_normals` are *not* available, falls back to isotropic noise.

**Parameters:**
- `surface_type`: Kept for config back-compat; ignored at runtime.
- `max_offset`: Maximum perturbation magnitude in normalized patch coordinates (default: 0.02)
- `p`: Probability of applying per example (default: 0.5)
- `max_ratio`: Maximum fraction of valid neighbors to perturb per example (default: 1.0 = all). The actual fraction is drawn uniformly from [0, max_ratio].
- `attributes`: List of attributes
- `mask_constant`: Sentinel value (default: -10.0)

**Example:**
```python
perturb = GaussianPatchSurfacePerturb(
    max_offset=0.02,
    p=0.5,
    max_ratio=0.8,
    attributes=["xyz"],
)
```

**YAML config:**
```yaml
transforms:
  - name: GaussianPatchSurfacePerturb
    max_offset: 0.02
    p: 0.5
    max_ratio: 0.8
```

> **`_aug_normals`:** Include `_aug_normals` in dataset attributes to enable
> directed (along-normal) perturbation. Normals are computed from raw positions
> at data generation time and passed to transforms via `kwargs`.

### 6. GeodesicNoiseAugmentation
Simulates `min_input` prediction error during Fast Marching inference.
A single additive offset is drawn per example and applied uniformly to
valid neighbours' normalised geodesic distances.

**Parameters:**
- `max_noise`: Scale factor for the offset relative to the geodesic range in the patch (default: 0.1)
- `p`: Probability of applying per example (default: 0.5)
- `max_ratio`: Maximum fraction of valid neighbors whose geodesic values are perturbed per example (default: 1.0 = all). The actual fraction is drawn uniformly from [0, max_ratio]. Unperturbed neighbors keep their original geodesic value.
- `attributes`: Unused, kept for registry injection consistency
- `mask_constant`: Sentinel value (default: -10.0)

**Example:**
```python
geo_noise = GeodesicNoiseAugmentation(
    max_noise=0.01,
    p=0.5,
    max_ratio=0.7,
    attributes=["xyz"],
)
```

**YAML config:**
```yaml
transforms:
  - name: GeodesicNoiseAugmentation
    max_noise: 0.01
    p: 0.5
    max_ratio: 0.7
```

## Composing Multiple Transforms

Use `ComposeTransforms` to chain multiple transforms:

```python
class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

# Compose dropout and rotation
composed = ComposeTransforms([
    GaussianPatchDropout(max_dropout_ratio=0.2, ...),
    GaussianPatchRotate(...)
])

dataset = GaussianPatchDataset(..., transform=composed)
```

## Training vs Validation

**Important:** Apply transforms only to training data, not validation data!

```python
# Training: with augmentation
train_transform = ComposeTransforms([
    GaussianPatchDropout(...),
    GaussianPatchRotate(...)
])

train_dataset = GaussianPatchDataset(
    config_path="config.yaml",
    ...,
    transform=train_transform  # Apply augmentation
)

# Validation: no augmentation
val_dataset = GaussianPatchDataset(
    config_path="config.yaml",
    ...,
    transform=None  # No augmentation
)
```

## Custom Transforms

Create custom transforms by implementing a callable class:

```python
class MyCustomTransform:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, batch):
        # batch shape: (batch_size, feature_dim)
        # Modify batch here
        # Important: Don't modify last 1-2 values (targets)
        return batch

# Use it
dataset = GaussianPatchDataset(..., transform=MyCustomTransform(...))
```

## Best Practices

1. **Match attributes**: Ensure transform attributes match dataset attributes
2. **Match max_neighbors**: Use same max_neighbors in transform and dataset
3. **Match use_r1_min_val**: Keep consistent between transform and dataset
4. **Training only**: Apply augmentation to training data, not validation/test
5. **Compose carefully**: Order matters - apply dropout before rotation typically
6. **Test first**: Verify transforms work on a small batch before full training

## Example Training Script

```python
import torch
from torch.utils.data import DataLoader
from gaussian_dataset import GaussianPatchDataset
from data_transformation import GaussianPatchDropout, GaussianPatchRotate

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch

# Training augmentation
train_transform = ComposeTransforms([
    GaussianPatchDropout(max_dropout_ratio=0.3, attributes=["xyz", "normals", "opacity"], 
                         max_neighbors=64, use_r1_min_val=True),
    GaussianPatchRotate(attributes=["xyz", "normals", "opacity"], 
                        max_neighbors=64, use_r1_min_val=True)
])

# Create datasets
train_dataset = GaussianPatchDataset(
    config_path="config.yaml",
    attributes=["xyz", "normals", "opacity"],
    ring=2,
    use_r1_min=True,
    transform=train_transform
)

val_dataset = GaussianPatchDataset(
    config_path="config.yaml",
    attributes=["xyz", "normals", "opacity"],
    ring=2,
    use_r1_min=True,
    transform=None  # No augmentation
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    for features, targets in train_loader:
        # features are automatically augmented
        outputs = model(features)
        loss = criterion(outputs, targets)
        # ... backprop
    
    # Validation
    model.eval()
    with torch.no_grad():
        for features, targets in val_loader:
            # No augmentation here
            outputs = model(features)
            # ... compute metrics
```

## See Also

- [example_transform_usage.py](example_transform_usage.py) - Complete examples
- [data_transformation.py](data_transformation.py) - Transform implementations
- [tests/test_data_transformation.py](tests/test_data_transformation.py) - Transform tests

## Loss Functions

The `GaussianPatchTransformer.get_loss()` method supports these loss types:

| `loss_type` | Description |
|---|---|
| `mse` | Mean Squared Error |
| `l1` | Mean Absolute Error |
| `smooth_l1` | Smooth L1 / Huber-like |
| `huber` | Huber loss (set `huber_delta`) |
| `log_mse` | MSE in log-distance space |
| `log_l1` | L1 in log-distance space |
| `blended_mse` | 50/50 blend of linear MSE and log-space MSE — penalises relative errors equally at all distance scales while retaining strong gradients for large errors |
