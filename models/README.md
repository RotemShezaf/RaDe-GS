# Gaussian Patch Transformer

A transformer-based architecture for predicting geodesic distances on Gaussian splatting patches.

## Overview

The Gaussian Patch Transformer is designed to predict geodesic distances for points in a 3D Gaussian splatting representation. The model:

- **Encoder-only architecture** – uses CLS token pooling similar to BERT/ViT
- **Encoder-decoder architecture** – XYZ geodesic encoder + geodesic-distance decoder (`attention_type="geodesic_enc_dec"`)
- **Split-embedding encoder-only** – interleaved attribute/geodesic tokens with type embeddings and mask only on geodesic tokens (`attention_type="split"`)
- **Split-embedding encoder-decoder** – encoder on attributes (standard attention, no mask), decoder on geodesic with cross-attention (`attention_type="split_enc_dec"`)
- **Handles variable neighbor counts** – uses attention masking for padded/invalid neighbors
- **Configurable point attributes** – can use all, subset, or no point features
- **Sorts neighbors by Euclidean distance** – for positional encoding consistency
- **Flexible pooling** – masked max, mean, or both concatenated (`pool` parameter)
- **Three pluggable encoder types** – Conv1d PointNet-style (default), MLP linear, or residual MLP (`encoder_type`)
- **Three positional encoding strategies** – sorted-order index, 3-D spatial sinusoidal, or learned relative-position attention bias (`pos_encoding_type`)
- **Geodesic self-attention** – optional geodesic self-attention replacing Q·K^T with GDS-based weights computed from token XYZ positions (`attention_type`)

## Project Structure

```
models/
├── __init__.py                          # Module exports
├── args.py                              # CLI / YAML argument parsing
├── const.py                             # Transform registry & Compose helper
├── utils.py                             # Utility modules (Attention, PE, RelativePositionBias, …)
├── transformer.py                       # Encoder/decoder blocks & all encoder variants
├── GaussianPatchTransformer.py          # Main model implementation
├── trainer.py                           # Trainer class with early stopping
├── train_gaussian_patch_transformer.py  # Training script (CLI)
├── configs/                             # Training configuration files
│   ├── default_train.yaml                   # Default training hyperparameters
│   ├── combined_polynomial_ring2.yaml       # Baseline: linear encoder, index PE
│   ├── combined_polynomial_ring2_conv.yaml  # Conv encoder, spatial PE
│   ├── combined_polynomial_ring2_spatial_maxmean.yaml  # Conv, spatial, max_mean pool, log_l1
│   └── testing.yaml                         # Quick 100K-sample sanity-check config
└── tests/                               # Test suite
    ├── __init__.py
    ├── README.md                        # Testing documentation
    ├── test_utils.py                    # Tests for utility functions (31 tests)
    ├── test_transformer.py              # Tests for transformer components (10 tests)
    ├── test_model.py                    # Tests for GaussianPatchTransformer
    ├── test_trainer.py                  # Tests for Trainer class
    └── test_integration.py              # Integration tests
```

## Architecture

The model follows an encoder-only transformer architecture with CLS token:

```
Input:
├── neighborhood: (batch, max_neighbors, entry_size)  # includes geodesic distance
├── point_features: (batch, point_feature_dim)        # no geodesic distance
└── valid_mask: (batch, max_neighbors)                # True for valid neighbors

    │
    ├── Sort neighbors by Euclidean distance from point
    │
    ├── Encode neighbor features → neighbor tokens  (conv / linear / residual)
    │
    ├── (Optional) Encode point features → point token
    │
    ├── Prepend learnable CLS token
    │       │
    │       v
    │   [CLS, (point), neighbor_1, ..., neighbor_N]
    │       │
    │       v
    │   Positional encoding  (index / spatial / relative_bias)
    │       │
    │       v
    │   Transformer Encoder (with attention masking + optional per-layer attn bias)
    │       │
    │       v
    │   Extract CLS output + masked-pooled neighbors (max / mean / max_mean)
    │       │
    │       v
    │   Prediction Head (MLP → LeakyReLU)
    │       │
    │       v
    └── Geodesic Distance (scalar, non-negative)
```

### Key Components

#### Encoders (`encoder_type`)

| Type | Neighbor Encoder | Point Encoder | Details |
|------|-----------------|---------------|---------|
| `conv` *(default)* | `GaussianPatchEncoder` | `PointFeatureEncoder` | Conv1d PointNet-style: per-token encoding → global max-pool context injection → per-token refinement. Uses GroupNorm + LeakyReLU. |
| `linear` | `GaussianPatchLinearEncoder` | `PointFeatureLinearEncoder` | MLP (Linear → LayerNorm → GELU). Original implementation. |
| `residual` | `GaussianPatchLinearEncoderResidual` | `PointFeatureLinearResidualEncoder` | MLP with a 128→128→128 skip-connection block. No LayerNorm; LeakyReLU activations. |

Switch via `encoder_type` in the YAML config or `--encoder_type` on the CLI.

#### Positional Encoding (`pos_encoding_type`)

| Type | When Applied | Description |
|------|-------------|-------------|
| `index` *(default)* | Once, before layer 1 | Sinusoidal encoding of sorted-neighbor order (1, 2, …, N). |
| `spatial` | Once, before layer 1 | Sinusoidal encoding of relative 3-D XYZ displacement; requires `xyz` in attributes. |
| `relative_bias` | At every layer | Learned per-head bias added to attention logits from pairwise 3-D positions (`RelativePositionBias`). Keeps geometry alive through all layers. Requires `xyz` in attributes. |

#### Other Components

- **CLS Token**: Learnable token prepended to the sequence for aggregation.
- **TransformerEncoderBlock**: Pre-norm transformer block (LayerNorm → Attention → DropPath → LayerNorm → FFN → DropPath). Accepts optional `attn_bias` for relative-position bias.
- **Pooling** (`pool`): Aggregates neighbor tokens — `max` (default), `mean`, or `max_mean` (concatenation of both; head input becomes 3×embed_dim).
- **Geodesic Self-Attention** (`attention_type`): `standard` (default) uses vanilla multi-head self-attention (Q·K^T). `geodesic` replaces Q·K^T with attention weights from a *Graph-based Geodesic Distance Score* (GDS) matrix computed on-the-fly from token XYZ positions: K-NN graph construction + vectorised tropical-semiring shortest paths (Algorithm 1 from *Geodesic Self-Attention for 3D Point Clouds*, NeurIPS 2022). Attention formula: `softmax(-GDS / τ) · V(X)` with a learnable per-head temperature τ. Requires `"xyz"` in `attributes`. Can be combined with any positional encoding type. `geodesic_enc_dec` uses a full encoder-decoder transformer: the encoder processes raw XYZ with geodesic self-attention (no masking), while the decoder processes raw geodesic distances with standard self-attention + cross-attention to the encoder (valid mask applied only on the decoder side). Requires `"xyz"` in `attributes`.
- **Split Embedding** (`attention_type="split"`): Creates two tokens per neighbor — an attribute token (all Gaussian attributes except geodesic) and a geodesic token (scalar geodesic distance via `GeodesicEmbedding`). Tokens are interleaved: `[CLS, (point?), attr_1, geo_1, attr_2, geo_2, …]`. A learnable type embedding (3 types: CLS/point, attribute, geodesic) is added. Attention mask applied only to geodesic tokens. Uses standard self-attention.
- **Split Embedding Encoder-Decoder** (`attention_type="split_enc_dec"`): Encoder processes all Gaussian attributes (no geodesic) with standard self-attention and no mask. Decoder processes geodesic distances via `GeodesicEmbedding` with self-attention + cross-attention to the encoder. Valid mask applied only on the decoder side. Supports all encoder types.
- **Prediction Head**: MLP (Linear → LayerNorm → GELU → Dropout) × 2 → Linear → LeakyReLU. Final LeakyReLU ensures non-negative output while avoiding dead gradients near zero.

## Installation

The module uses the following dependencies:
- PyTorch >= 2.4.0 (for NumPy 2.0 compatibility)
- timm >= 0.9.0 (for weight initialization)
- numpy >= 2.0.0
- tqdm (for training)
- wandb >= 0.17.0 (optional, for experiment tracking)

## Usage

### Basic Usage

```python
from models import GaussianPatchTransformer, create_gaussian_patch_transformer
from DataSets.gaussian_dataset import GaussianPatchDataset
from torch.utils.data import DataLoader

# Create model
model = create_gaussian_patch_transformer(
    attributes=["xyz", "opacity", "scale", "rotation", "sh"],
    max_neighbors=32,
    embed_dim=384,
    encoder_depth=6,
    num_heads=8
)

# Load dataset (returns 4 values: neighborhood, point_features, target, valid_mask)
dataset = GaussianPatchDataset(
    config_path="path/to/config.yaml",
    attributes=["xyz", "opacity", "scale", "rotation", "sh"],
    ring=2,
    use_r1_min=False
)

# Forward pass
neighborhood, point_features, target, valid_mask = dataset[0]
predictions = model(
    neighborhood.unsqueeze(0),      # (1, max_neighbors, entry_size)
    point_features.unsqueeze(0),    # (1, point_feature_dim)
    valid_mask.unsqueeze(0)         # (1, max_neighbors)
)
```

### Training

Train the model using the Trainer class or the CLI script:

#### Using the Trainer Class (Recommended)

```python
from models import GaussianPatchTransformer
from models.trainer import Trainer, EarlyStopping
from torch.utils.data import DataLoader
import torch

# Create model
model = GaussianPatchTransformer(
    attributes=["xyz", "opacity", "scale"],
    max_neighbors=32
)

# Set up training components
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    early_stopping=early_stopping,
    save_dir="./checkpoints",
    loss_type='mse'
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)

# Make predictions
predictions = trainer.predict(test_loader)
```

#### Using the CLI Script

The training script supports three layers of configuration (later overrides earlier):
1. Built-in defaults (in `args.py`)
2. YAML config file (`--train_config`)
3. Explicit CLI arguments

```bash
# Minimal – dataset config + ring on CLI
python models/train_gaussian_patch_transformer.py \
    --dataset_config DataSets/configs/saddle_all.yaml --ring 2

# With training config file
python models/train_gaussian_patch_transformer.py \
    --train_config models/configs/default_train.yaml \
    --dataset_config DataSets/configs/combined_polynomial_all.yaml --ring 2

# Override any value from the command line
python models/train_gaussian_patch_transformer.py \
    --train_config models/configs/combined_polynomial_ring2.yaml \
    --batch_size 128 --learning_rate 5e-4 --use_wandb
```

#### Training Configuration Files

Training configs live in `models/configs/` and are structured with nested sections
(`dataset`, `model`, `training`, `early_stopping_cfg`, `infrastructure`, `logging`).
See [models/configs/default_train.yaml](configs/default_train.yaml) for all available keys.

Key logging options:
- **`log_file`** – path to a log file (console + file); `null` for console only
- **`use_wandb`** – enable Weights & Biases experiment tracking
- **`wandb_run_name`** – human-readable name for the W&B run

#### Data Augmentation

Augmentation transforms are configured under the `dataset.transforms` key in the training
config YAML, or passed as transform names via the `--transforms` CLI flag.
Transforms are applied to every training example before batching (not applied to validation).

```yaml
# In your training config (e.g. models/configs/combined_polynomial_ring2.yaml)
dataset:
  use_r1_min: true   # required for GaussianPatchDropout
  transforms:
    - name: GaussianPatchRotate                # random SO(3) rotation
    - name: GaussianPatchRandomFlip            # random axis flip
      flip_prob: 0.5
    - name: GaussianPatchDropout               # geodesic-distance-based dropout
      max_dropout_ratio: 0.3
```

Or via CLI (default parameters only):

```bash
python models/train_gaussian_patch_transformer.py \
    --train_config models/configs/default_train.yaml \
    --dataset_config DataSets/configs/saddle_all.yaml \
    --transforms GaussianPatchRotate GaussianPatchRandomFlip
```

**Available transforms** (see `models/const.py` → `TRANSFORM_REGISTRY`):

| Name | Description |
|------|-------------|
| `GaussianPatchRotate` | Random SO(3) rotation of xyz, normals, and rotation quaternions |
| `GaussianPatchRandomFlip` | Random reflection along x / y / z axes (`flip_prob`, `flip_axes`) |
| `GaussianPatchDropout` | Drop neighbors beyond r1\_min threshold (`max_dropout_ratio`; needs `use_r1_min`) |
| `GaussianPatchCanonicalRotate` | Deterministic centre-of-mass alignment to a target direction |
| `PointcloudRandomInputDropout` | Uniform random neighbor dropout (`max_dropout_ratio`) |

Transforms are built and chained via `build_transforms()` in `models/const.py`.
Multiple transforms are composed automatically (applied left-to-right).

#### Automated Training Script

A convenience bash script creates a tmux session, allocates a GPU via SLURM,
and launches training:

```bash
bash scripts/train_combined_polynomial.sh
```

See the script header for customisation options (node, GPUs, ring, config overrides).

### Early Stopping

The `EarlyStopping` class monitors validation metrics and stops training when no improvement is seen:

```python
from models.trainer import EarlyStopping

# Stop if validation MAE doesn't improve for 10 epochs
early_stopping = EarlyStopping(
    patience=10,          # Epochs to wait before stopping
    min_delta=0.001,      # Minimum change to qualify as improvement
    mode='min',           # 'min' for loss/MAE, 'max' for accuracy
)
```

### Point Attributes Configuration

The `point_attributes` parameter controls which attributes are used for the point token:

```python
# Use all attributes for point token (default)
model = GaussianPatchTransformer(
    attributes=["xyz", "opacity", "scale"],
    point_attributes=None  # Uses all attributes
)

# Use subset of attributes for point token
model = GaussianPatchTransformer(
    attributes=["xyz", "opacity", "scale"],
    point_attributes=["xyz", "opacity"]  # Only xyz and opacity
)

# No point token (encoder processes only neighbors)
model = GaussianPatchTransformer(
    attributes=["xyz", "opacity", "scale"],
    point_attributes=[]  # No point token
)
```

## Model Configuration

### Parameters

- **attributes** (List[str]): Gaussian attributes to use for neighbors
  - Options: `"xyz"`, `"opacity"`, `"scale"`, `"rotation"`, `"sh"`, `"normals"`
  - Default: `["xyz"]`

- **point_attributes** (Optional[List[str]]): Attributes for point token
  - `None`: Use all attributes (default)
  - `[]`: No point token
  - `["xyz", "opacity"]`: Use only specified attributes

- **max_neighbors** (int): Maximum number of neighbors in a patch
  - Default: 32

- **embed_dim** (int): Dimension of token embeddings
  - Default: 384

- **encoder_depth** (int): Number of transformer encoder blocks
  - Default: 6

- **num_heads** (int): Number of attention heads
  - Default: 8

- **mlp_ratio** (float): Ratio of MLP hidden dim to embedding dim
  - Default: 4.0

- **dropout** (float): Dropout rate
  - Default: 0.0

- **attn_dropout** (float): Attention dropout rate
  - Default: 0.0

- **drop_path_rate** (float): Stochastic depth rate
  - Default: 0.1

- **pool** (str): Pooling strategy for aggregating neighbor tokens before the prediction head
  - `"max"`: masked max-pool (default)
  - `"mean"`: masked mean-pool
  - `"max_mean"`: concatenate max and mean (head input becomes 3×embed_dim)

- **pos_encoding_type** (str): Positional encoding applied to neighbor tokens
  - `"index"`: sinusoidal encoding of sorted-neighbor order (default)
  - `"spatial"`: sinusoidal encoding of relative 3-D XYZ position (requires `"xyz"` in `attributes`)
  - `"relative_bias"`: learned per-head attention bias from pairwise 3-D positions, injected at every layer (requires `"xyz"` in `attributes`)

- **attention_type** (str): Self-attention mechanism
  - `"standard"`: vanilla multi-head self-attention (default)
  - `"geodesic"`: geodesic self-attention — replaces Q·K^T with `softmax(-GDS/τ)` where GDS is computed from XYZ via K-NN + tropical-semiring shortest paths.  Learnable per-head temperature τ.  Requires `"xyz"` in `attributes`.
  - `"geodesic_enc_dec"`: encoder-decoder architecture — encoder uses geodesic self-attention on raw XYZ (no mask), decoder processes raw geodesic distances with cross-attention to encoder (mask applied on decoder only). Requires `"xyz"` in `attributes`.
  - `"split"`: split-embedding encoder-only — creates interleaved attribute/geodesic tokens per neighbor with type embedding; attention mask on geodesic tokens only.
  - `"split_enc_dec"`: split-embedding encoder-decoder — encoder on attributes (standard attention, no mask), decoder on geodesic with cross-attention.

- **decoder_depth** (int): Number of decoder blocks (only for `geodesic_enc_dec`)
  - Default: 3

- **encoder_type** (str): Encoder implementation for both neighbor and point encoders
  - `"conv"`: Conv1d PointNet-style — per-token encoding → global max-pool injection → refinement; GroupNorm + LeakyReLU (default)
  - `"linear"`: original MLP (Linear + LayerNorm + GELU) encoder
  - `"residual"`: MLP with 128→128→128 skip-connection block, no LayerNorm, LeakyReLU

## Input Format

The model expects three separate tensors:

1. **neighborhood**: `(batch, max_neighbors, entry_size)`
   - Contains neighbor features with geodesic distance as last entry
   - Entry size = attribute_dim + 1 (for geodesic)

2. **point_features**: `(batch, point_feature_dim)`
   - Contains point features without geodesic distance
   - Feature dim = attribute_dim

3. **valid_mask**: `(batch, max_neighbors)`
   - Boolean mask, True for valid neighbors, False for padded/invalid

This format matches the output of `GaussianPatchDataset`.

## Loss Functions

```python
# Mean Squared Error (default)
loss = model.get_loss(predictions, targets, loss_type='mse')

# L1 Loss
loss = model.get_loss(predictions, targets, loss_type='l1')

# Smooth L1 Loss
loss = model.get_loss(predictions, targets, loss_type='smooth_l1')

# Log-space MSE – penalises relative error equally across all distance scales
loss = model.get_loss(predictions, targets, loss_type='log_mse')

# Log-space L1
loss = model.get_loss(predictions, targets, loss_type='log_l1')
```

All five loss types are also selectable via the YAML config key `loss_type` or the `--loss_type` CLI flag.

## Evaluation Metrics

```python
metrics = model.get_metrics(predictions, targets)
# Returns: {
#     'mae': mean absolute error,
#     'rmse': root mean squared error,
#     'relative_error_pct': mean relative error as percentage,
#     'max_error': maximum absolute error
# }
```

## Advanced Features

### Return Intermediate Embeddings

```python
output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
# Returns dict with:
# - 'prediction': final geodesic distance prediction
# - 'neighbor_tokens': initial neighbor embeddings
# - 'point_token': initial point embedding (None if no point token)
# - 'cls_output': CLS token output after encoder
# - 'max_pooled': max-pooled neighbor features (or mean/both, depending on pool)
# - 'encoder_output': full encoder output
# - 'sorted_indices': indices used for neighbor sorting
# - 'attention_mask': attention mask used in encoder
```

### Model Information

```python
model.print_model_info()
# Prints architecture details, feature dimensions, parameter counts
```

## Testing

Run tests with pytest:

```bash
# Run all model tests
cd models
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_model.py -v       # Model tests
python -m pytest tests/test_trainer.py -v     # Trainer tests
python -m pytest tests/test_transformer.py -v # Transformer tests (10 tests)
python -m pytest tests/test_utils.py -v       # Utility tests (31 tests)
python -m pytest tests/test_integration.py -v # Integration tests

# Run with coverage
python -m pytest tests/ -v --cov=. --cov-report=html
```

### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_model.py` | GaussianPatchTransformer, GaussianPatchSplitTransformer, SplitEmbeddingEncoderDecoderTransformer forward pass, loss, metrics, point_attributes, pool, encoder_type |
| `test_trainer.py` | Trainer class, early stopping, checkpoints, batch training |
| `test_transformer.py` | Conv1d + Linear encoder/decoder blocks (10 tests) |
| `test_utils.py` | Attribute dimensions, positional encoding, `sinusoidal`, `SpatialPositionalEncoding` (31 tests) |

## Performance Tips

1. **Batch Size**: Start with 32, adjust based on GPU memory
2. **Learning Rate**: 1e-4 works well with AdamW optimizer
3. **Attention Masking**: Automatically handles variable neighbor counts
4. **Point Attributes**: Use `point_attributes=[]` if point features don't help
5. **Attention Heads**: Use 8 for good performance/efficiency trade-off

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Reduce embed_dim
- Reduce encoder_depth

### Poor Convergence
- Check learning rate (try 1e-5 to 1e-3)
- Verify data normalization
- Try different loss function (smooth_l1 can be more stable)

### Overfitting
- Increase dropout and drop_path_rate
- Use data augmentation – add transforms to the `dataset.transforms` config key
  (e.g. `GaussianPatchRotate`, `GaussianPatchRandomFlip`; see [Data Augmentation](#data-augmentation) above)

## License

See the LICENSE.md file in the root directory of RaDe-GS.
