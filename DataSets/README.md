# DataSets Module

This module provides PyTorch datasets and utilities for training geodesic distance prediction models on Gaussian splats.

## Overview

The DataSets module handles the complete pipeline for:
1. **Generating** training patches from Gaussian splat 
2. **Loading** training data into PyTorch datasets
3. **Transforming** data with augmentations during training
4. **Combining** multiple data sources for diverse training

## Directory Structure

```
DataSets/
├── README.md                               # This file
├── TRANSFORM_USAGE.md                      # Detailed guide for data transforms
├── create_gaussian_training_patches.py     # Generate patches for polynomial surfaces
├── create_tosca_training_patches.py        # Generate patches for TOSCA shapes (area-weighted)
├── gaussian_dataset.py                     # PyTorch Dataset classes
├── data_transformation.py             # Data augmentation transforms
├── example_transform_usage.py         # Examples for using transforms
├── configs/                                # Configuration files
│   ├── README.md                           # Config documentation
│   ├── combined_polynomial.yaml            # Multi-source polynomial example
│   ├── saddle.yaml                         # Single-source polynomial example
│   └── tosca/                              # Auto-generated TOSCA configs (by generate_tosca_training_patches.sh)
├── utils/                             # Utility modules
│   ├── config_utils.py                # Config loading and validation
│   ├── data_transformation_utils.py   # Core transformation utilities
│   ├── rotation_conversions.py        # Quaternion/rotation utilities
│   └── training_patches_helpers.py    # Training patch generation helpers
└── tests/                             # Test suite
    ├── test_gaussian_dataset.py       # Dataset tests
    ├── test_data_transformation.py    # Transform tests
    ├── test_multi_source_dataset.py   # Multi-source tests
    ├── test_rotation.py               # Rotation utility tests
    └── test_training_patches_helpers.py # Helper function tests
```

## Quick Start

### 1. Generate Training Data

#### Polynomial Surfaces

Use `create_gaussian_training_patches.py` with a config file:

```bash
python DataSets/create_gaussian_training_patches.py --config DataSets/configs/saddle.yaml
```

Or specify paths directly:

```bash
python DataSets/create_gaussian_training_patches.py \
    --gaussian_output path/to/gaussian/output \
    --geodesic_data path/to/geodesic.npz \
    --output_dir path/to/output
```

#### TOSCA Shapes

Use `create_tosca_training_patches.py` — the TOSCA-specific counterpart that samples sources with **area-weighted uniform probability** over the mesh surface (see [TOSCA Training Patches](#tosca-training-patches) below):

```bash
python DataSets/create_tosca_training_patches.py \
    --gaussian_output TrainData/TOSCA/SyntheticColmapData/colors_texture/cat0/high_res/light_0/output \
    --tosca_data_root TrainData/TOSCA/processed \
    --output_dir TrainData/datasets/gaussian_patches/tosca_cat0_colors
```

Or use the shell driver to generate patches for all shapes at once:

```bash
bash scripts/generate_tosca_training_patches.sh
```

#### FPS Downsampling (Optional)

Both patch generation scripts support **Furthest Point Sampling (FPS)** to
reduce the point-cloud size before neighbourhood / KNN queries. This is useful
when the Gaussian output is very dense and you want lighter, more uniform
training patches.

```bash
# Position-only FPS — downsample to 20 000 Gaussians
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/saddle.yaml \
    --fps_target 20000

# Attribute-aware FPS (uses xyz + scale for distance metric)
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/saddle.yaml \
    --fps_target 20000 --fps_attributes xyz scale

# Same options work for TOSCA patches
python DataSets/create_tosca_training_patches.py \
    --gaussian_output path/to/tosca/output \
    --tosca_data_root TrainData/TOSCA/processed \
    --output_dir path/to/output \
    --fps_target 20000 --fps_attributes xyz
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fps_target` | `int` | `0` (disabled) | Target number of Gaussians after FPS. `0` = no downsampling. |
| `--fps_attributes` | `str…` | `xyz` | Space-separated attribute names used for the FPS distance metric. Valid: `xyz`, `scale`, `rotation`, `opacity`, `sh`, `normals`. |

**Source protection**: GT geodesic source Gaussians are always protected — they
are never dropped by FPS, regardless of the target count. FPS selects the
remaining `fps_target − num_sources` points from the non-source pool.

The underlying utility is `utils.misc.fps_gs`, which wraps
`pointnet2_utils.furthest_point_sample` and accepts a `protected_indices`
parameter to guarantee specific points survive.

#### Near-Source Oversampling (Optional)

By default, training source points in each patch are sampled uniformly.
With `--near_source_oversample`, **all** training points are drawn from a
single weighted distribution that heavily oversamples points near the source
yet **collapses to uniform** at large distances:

```
w(d) = exp(-alpha * d / median_d) + 1
```

where `alpha = near_source_oversample * 10` and `median_d` is the median
non-zero geodesic distance.  The `+1` floor ensures far-away points always
retain at least uniform probability.

```bash
# Controls near-source bias strength (0 = uniform, 1 = aggressive)
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/saddle.yaml \
    --near_source_oversample 0.3
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--near_source_oversample` | `float` | `0.0` (disabled) | Near-source bias strength (0–1).  `0.0` = uniform sampling.  Higher values give stronger near-source weighting that smoothly collapses to uniform far from the source. |

---

#### Using a Gaussian Outputs File (.txt)

To generate a single dataset from multiple Gaussian outputs, create a `.txt` file
listing all the Gaussian output paths (one per line):

```text
# gaussian_outputs.txt
path/to/gaussian/output1
path/to/gaussian/output2
path/to/gaussian/output3
```

Then reference the `.txt` file in the config or command line:

```bash
python DataSets/create_gaussian_training_patches.py \
    --gaussian_output gaussian_outputs.txt \
    --output_dir path/to/output
```

Or in a YAML config:

```yaml
gaussian_output: "gaussian_outputs.txt"
output_dir: "path/to/output"
```

Samples from all listed Gaussian outputs will be combined into a single dataset.

### 2. Load Data in PyTorch

```python
from DataSets.gaussian_dataset import GaussianPatchDataset, create_dataloaders

# Single-source dataset
dataset = GaussianPatchDataset(
    config="DataSets/configs/saddle.yaml",
    ring=2,
    use_r1_min=True
)

# Create train/val dataloaders
train_loader, val_loader = create_dataloaders(
    config="DataSets/configs/saddle.yaml",
    ring=2,
    batch_size=32,
    use_r1_min=True
)
```

### 3. Use Data Augmentation

```python
from DataSets.data_transformation import (
    GaussianPatchRandomRotation,
    SparseContextDropout,
    GaussianPatchSurfacePerturb,
    Compose,
)

# SparseContextDropout now unifies both GaussianPatchDropout and sparse-context
# behaviour via the p_dropout parameter:
#   - With prob p_dropout: GaussianPatchDropout-style filtering (drop valid
#     neighbours with geodesic > r1_min_val)
#   - With prob 1-p_dropout: sparse-context mode (keep only K closest valid
#     neighbours, simulating near-source sparsity)
transform = Compose([
    GaussianPatchRandomRotation(attributes=["xyz", "normal"]),
    SparseContextDropout(
        min_valid=1, max_valid_ratio=0.3, p=0.3,
        p_dropout=0.3, max_dropout_ratio=0.5,
        attributes=["xyz"],
    ),
    GaussianPatchSurfacePerturb(max_offset=0.02, p=0.5, max_ratio=1.0, attributes=["xyz"]),
])

# Apply to dataset
dataset = GaussianPatchDataset(
    config="DataSets/configs/saddle.yaml",
    ring=2,
    use_r1_min=True,
    transform=transform,
)
```

**Directed perturbation with `_aug_normals`:**
Include `_aug_normals` in the dataset `attributes` list to enable directed
(along-normal) perturbation.  Analytical surface normals are computed from raw
positions during data generation.  The dataset passes them to transforms via
`kwargs`; rotation and flip transforms also rotate `_aug_normals` consistently.
Without `_aug_normals`, the transform falls back to isotropic noise.

See [TRANSFORM_USAGE.md](TRANSFORM_USAGE.md) for full documentation on each
transform, including YAML config examples and combined-dataset guidance.

## TOSCA Training Patches

`create_tosca_training_patches.py` is a full-featured counterpart of
`create_gaussian_training_patches.py` designed for irregular organic TOSCA
shapes (cats, humans, horses, …).

### Why area-weighted sampling?

Polynomial surfaces have a regular NxN grid of precomputed sources, which is
approximately uniform in 3D.  TOSCA meshes are **irregular** — heavily
tessellated regions would be over-represented by naive uniform sampling from
the source set.  This script instead assigns each source a probability
proportional to the **mesh surface area** surrounding it:

$$w_s = \frac{A(v_s)}{\sum_j A(v_j)}$$

where $A(v)$ is the per-vertex barycentric area (one-third of each adjacent
face area).  Sources are then drawn without replacement using
`np.random.choice(..., p=source_weights)`.

### Prerequisites

1. **Preprocessed PLY meshes** — run `GenerateData/preprocess_tosca.py` to
   produce `TrainData/TOSCA/processed/{shape}/mesh_high_res_*.ply`.
2. **Gaussian reconstructions** — produced by
   `GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py` + `train.py`.
3. **Precomputed geodesic distances** — produced by
   `GenerateData/compute_gaussian_geodesic_distances.py`, saved as
   `{gaussian_output}/geodesic_distance/gt_geodesic.npz`.

### TOSCA-specific config fields

| Field | Default | Description |
|---|---|---|
| `tosca_data_root` | `TrainData/TOSCA/processed` | Root of preprocessed PLY meshes |
| `shape` | auto-inferred from path | Shape name, e.g. `cat0`, `centaur1` |

All other config fields are identical to the polynomial version.

### TOSCA config example

```yaml
# Auto-generated by scripts/generate_tosca_training_patches.sh
dataset_class: GaussianPatchDataset

gaussian_output: "DataSets/configs/tosca/gaussian_sources/tosca_cat0_colors_all.txt"
geodesic_data: null
output_dir: "TrainData/datasets/gaussian_patches/tosca_cat0_colors"
tosca_data_root: "TrainData/TOSCA/processed"
shape: "cat0"

num_iterations: 1000
num_sources: 4
num_train_points: 50
seed: 42

use_mahalanobis: false
n_neighbors: 10
# Adaptive kNN (optional): boost per-point k so ring-k count reaches target
# adaptive_target_ring: 3
# adaptive_target_neighbors: 128
# adaptive_k_boost: 20
# adaptive_max_mean_cut: 5.0
# adaptive_max_steps: 5
normalize_per_patch: true
rings: [2, 3]

attributes:
  - xyz

nn_mean: 1
use_r1_min_val: true
mask_constant: -10.0
```

### Automated pipeline

```bash
# Generate patches for all auto-detected TOSCA shapes (parallel)
bash scripts/generate_tosca_training_patches.sh

# Specific shapes only
bash scripts/generate_tosca_training_patches.sh --shapes "cat0,cat1"

# Also produce a combined multi-source config
bash scripts/generate_tosca_training_patches.sh --combined_config
```

The shell script auto-generates the per-shape YAML configs and `.txt` source
lists under `DataSets/configs/tosca/`, then invokes
`create_tosca_training_patches.py` for each shape + texture combination.

---

## Multi-Source vs Multi-Output

There are two distinct ways to use multiple Gaussian outputs:

### Multi-Output (Gaussian Outputs File)

Use a `.txt` file listing multiple Gaussian output paths when you want a **single unified dataset** with samples from all listed outputs. This is useful when you have multiple reconstructions of similar surfaces or multiple resolutions and want them merged.

```yaml
# Config with gaussian outputs file
gaussian_output: "path/to/gaussian_outputs.txt"
output_dir: "path/to/output"
attributes: ["xyz"]
rings: [2, 3]
```

The `.txt` file contains one Gaussian output path per line (lines starting with `#` are ignored):

```text
# gaussian_outputs.txt
path/to/gaussian/output1
path/to/gaussian/output2
path/to/gaussian/output3
```

Load as a single dataset:

```python
dataset = GaussianPatchDataset(
    config="path/to/output/config.yaml",
    ring=2,
    use_r1_min=True
)
```

### Combined Data Configuration (data_sources)

Use `data_sources` when you want to **combine conceptually different datasets** (e.g., different surface types) for training diversity. Each source is an
independent dataset that can be weighted and sampled differently.

```python
from DataSets.gaussian_dataset import CombinedGaussianPatchDataset, create_combined_dataloaders

# Using combined data config
dataset = CombinedGaussianPatchDataset(
    config="DataSets/configs/combined_polynomial.yaml",
    ring=2,
    use_r1_min=True
)

# Or combine multiple single-source configs
dataset = CombinedGaussianPatchDataset(
    config=[
        "DataSets/configs/saddle.yaml",
        "DataSets/configs/paraboloid.yaml"
    ],
    ring=2,
    weights=[1.0, 2.0]  # Sample more from paraboloid
)

# Create dataloaders with weighted sampling
train_loader, val_loader = create_combined_dataloaders(
    config="DataSets/configs/combined_polynomial.yaml",
    ring=2,
    batch_size=32,
    use_weighted_sampling=True
)
```

Each source in a combined data config can also use a `.txt` file for its `gaussian_output`:

```yaml
data_sources:
  - name: "paraboloid_variants"
    gaussian_output: "path/to/paraboloid_outputs.txt"  # txt file with multiple paths
    output_dir: "path/to/combined/paraboloid"
    weight: 1.0
  - name: "saddle_single"
    gaussian_output: "path/to/saddle/output"  # single path
    output_dir: "path/to/combined/saddle"
    weight: 1.0
```

### Partial Loading with GaussianPatchSubsetDataset

When a full dataset has millions of examples (e.g. the combined polynomial dataset has ~4.5M), training one epoch can take a very long time. `GaussianPatchSubsetDataset` inherits `GaussianPatchDataset` and keeps only a reproducible random subset, without regenerating any data files.

```python
from DataSets.gaussian_dataset import GaussianPatchSubsetDataset

# Keep at most 500 000 examples
dataset = GaussianPatchSubsetDataset(
    config_path="DataSets/configs/saddle_all.yaml",
    ring=2,
    max_examples=500_000,   # hard cap
    subset_seed=42,         # reproducible shuffle
)

# Alternatively, keep 25 % of the data
dataset = GaussianPatchSubsetDataset(
    config_path="DataSets/configs/saddle_all.yaml",
    ring=2,
    subset_fraction=0.25,
    subset_seed=42,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_examples` | `int \| None` | `None` | Keep at most this many examples. Takes precedence over `subset_fraction`. |
| `subset_fraction` | `float \| None` | `None` | Fraction of examples to keep `(0, 1]`. |
| `subset_seed` | `int` | `42` | RNG seed — guarantees the same subset across runs. |

**Behaviour:**
- All constructor arguments of `GaussianPatchDataset` are accepted and forwarded.
- If neither `max_examples` nor `subset_fraction` is given the full dataset is returned (identical to `GaussianPatchDataset`).
- Indices are sorted after sampling to preserve storage order, improving cache / memmap locality.
- When lazy loading is active the selected rows are materialised into an in-memory tensor so subsequent indexing is fast.

**From the training script / YAML config** the subset can be controlled without changing any code:

```yaml
# models/configs/combined_polynomial_ring2.yaml  (dataset section)
dataset:
  max_examples: 500000   # cap at 500 k   (null = full dataset)
  subset_fraction: null  # alternative: e.g. 0.25
  subset_seed: 42
```

Or via CLI override:

```bash
python models/train_gaussian_patch_transformer.py \
    --train_config models/configs/combined_polynomial_ring2.yaml \
    --max_examples 200000
```

The training script automatically upgrades a `GaussianPatchDataset` to `GaussianPatchSubsetDataset` when either parameter is non-null, and applies a `torch.utils.data.Subset` wrapper for `CombinedGaussianPatchDataset`.

## Configuration Files

The `dataset_class` field in a config tells the training script which dataset class to instantiate:

| `dataset_class` | Use when… |
|-----------------|-----------|
| `GaussianPatchDataset` | Single data source, full dataset |
| `GaussianPatchSubsetDataset` | Single data source, capped subset (see above) |
| `CombinedGaussianPatchDataset` | Multiple data sources combined |

### Single-Source Config

```yaml
# Dataset class used by the training script to load this config
dataset_class: GaussianPatchDataset

# Path to Gaussian splat output (single folder or .txt file listing multiple folders)
gaussian_output: "path/to/gaussian/output"

# Path to precomputed geodesic distances (optional, auto-detected)
geodesic_data: null

# Output directory for training patches
output_dir: "path/to/output"

# Generation parameters
num_iterations: 1000
num_sources: 3
num_train_points: 15
seed: 42

# Neighborhood settings
use_mahalanobis: true
n_neighbors: 10
# adaptive_target_ring: 3        # optional: enable adaptive per-point k
# adaptive_target_neighbors: 128  # desired ring-k count
# adaptive_k_boost: 20           # max boosted k
rings: [2, 3]

# Feature attributes
attributes: ["xyz", "normal", "opacity", "scale", "rotation"]
mask_constant: -10.0
```

### Single-Source Config with Gaussian Outputs File

```yaml
# Dataset class used by the training script to load this config
dataset_class: GaussianPatchDataset

# .txt file listing multiple Gaussian output folders (one per line)
gaussian_output: "path/to/gaussian_outputs.txt"

# Output directory for the combined training patches
output_dir: "path/to/output"

# All other parameters same as single-source config
num_iterations: 1000
attributes: ["xyz"]
rings: [2, 3]
```

### Combined Data Config (data_sources)

```yaml
# Dataset class used by the training script to load this config
dataset_class: CombinedGaussianPatchDataset

# Base output directory
output_dir: "path/to/combined/output"

# Multiple data sources (each can use a single path or a .txt file)
data_sources:
  - name: "surface_1"
    gaussian_output: "path/to/surface1/output"
    output_dir: "path/to/combined/output/surface_1"
    surface_type: "Paraboloid"  # Used by GaussianPatchSurfacePerturb
    weight: 1.0

  - name: "surface_2"
    gaussian_output: "path/to/surface2/output"
    output_dir: "path/to/combined/output/surface_2"
    surface_type: "Saddle"
    weight: 2.0  # Higher weight = more samples

  - name: "surface_3_multi"
    gaussian_output: "path/to/surface3_outputs.txt"  # .txt file with multiple paths
    output_dir: "path/to/combined/output/surface_3"
    surface_type: "HyperbolicParaboloid"
    weight: 1.0

# Shared parameters (must match across sources)
attributes: ["xyz", "normal"]
rings: [2, 3]
use_mahalanobis: true
```

## Data Bundles: `GaussianData` and `PatchConfig`

`training_patches_helpers.py` exports two dataclasses that bundle the many
arguments previously passed individually to `create_train_example` and
`generate_training_examples`:

```python
from DataSets.utils.training_patches_helpers import GaussianData, PatchConfig

gdata = GaussianData(
    positions=positions,      # (N, 3)
    scales=scales,            # (N, 3) or None
    rotations=rotations,      # (N, 4) or None
    opacities=opacities,      # (N, 1) or None
    normals=normals,          # (N, 3) or None
    sh_features=sh_features,  # (N, K) or None
    per_point_nn_distances=per_point_nn_distances,  # (N,) or None
)

pcfg = PatchConfig(
    ring=2,
    normalization_factor=mean_nn_dist,
    nn_mean=1.0,
    attributes=["xyz", "opacity", "scale", "rotation", "sh"],
    use_mahalanobis=False,
    use_r1_min_val=True,
    mask_attributes=[],
    mask_constant=-10.0,
    ring_size_mapping={"euclidean": {2: 48}},
    normalize_per_patch=False,
)

examples = generate_training_examples(
    geodesic_data, ring_nbrs_dict, ring1_nbrs,
    num_iterations, num_sources, num_train_points,
    gaussian_data=gdata, patch_config=pcfg,
    seed=42,
)
```

The old positional-argument signatures are still accepted for backward
compatibility.

---

## Normalization and Denormalization API

The training pipeline applies normalization in **two successive stages**.
Understanding this split is important when running inference via the
`geodesic_propagation` module.

### Stage 1 — Offline per-patch normalization (`training_patches_helpers.py`)

Applied at *data-generation time* inside `create_train_example`:

1. **Geodesic shift**: subtract the minimum visited-neighbor geodesic distance
   (`min_input`) so the smallest known distance becomes zero.
2. **Spatial + geodesic scaling**: divide xyz, scale, euclidean-distance, and
   geodesic columns by `current_normalization` and multiply by `nn_mean`.
   `current_normalization` is either the global mean NN distance or a per-patch
   mean computed from the neighbors' own NN distances.

The normalization logic is now factored into two module-level functions:

```python
from DataSets.utils.training_patches_helpers import (
    normalize_neighborhood,   # apply stage-1 normalization
    denormalize_neighborhood,  # invert stage-1 normalization
)

# Normalize
(valid_nbhd, beyond_nbhd, p_feats, r1_nbhd, p_u_norm,
 r1_norm, min_input) = normalize_neighborhood(
    valid_neighborhood=valid_nbhd,
    beyond_neighborhood=beyond_nbhd,
    p_features=p_feats,
    r1_neighborhood=r1_nbhd,
    p_u=p_u,
    r1_min_val=r1_min,
    attributes=["xyz"],
    current_normalization=mean_nn_dist,
    nn_mean=1.0,
    mask_constant=-10.0,
)

# Denormalize (e.g. after model prediction)
nbhd_orig, pf_orig, target_orig, r1_orig = denormalize_neighborhood(
    neighborhood=nbhd_norm,
    point_features=pf_norm,
    target=model_output,
    r1_min_val=None,
    min_input=min_input,
    current_normalization=mean_nn_dist,
    nn_mean=1.0,
    attributes=["xyz"],
    mask_constant=-10.0,
)
```

### Stage 2 — Online per-example normalization (`GaussianPatchDataset`)

Applied at *dataset-loading time* inside `_normalize_patches` (which now
subsumes the formerly-separate `_normalize_xyz_geodesic`):

1. **Attribute-specific normalization**: opacity min-max, SH scaling, scale
   point-cloud normalization.
2. **XYZ centering**: subtract the point's own position from all neighbors'
   xyz features (so the patch is centred at the origin).
3. **Max-distance scaling**: divide xyz and geodesic columns by the maximum
   Euclidean distance to any *real* (non-padded) neighbor (`max_dist`).
   By default, only *valid* neighbors (geodesic ≠ mask_constant) contribute
   to `max_dist`. Set `normalize_all_neighbors: true` in the dataset config
   (or pass `normalize_all_neighbors=True` to `GaussianPatchDataset`) to
   compute `max_dist` over **all** non-padded neighbors including duplicated
   padding with valid xyz.

`_normalize_patches` now returns a `norm_params` dict containing `max_dist`
so that predictions can be denormalized:

```python
neighborhood, point_features, target, r1_min_val, norm_params = \
    dataset._normalize_patches(neighborhood, point_features, target, r1_min_val)

# --- run model ---

# Denormalize: step 2 first, then step 1
neighborhood, point_features, target_denorm = dataset._denormalize_results(
    neighborhood, point_features, model_prediction, norm_params
)
# target_denorm is now in "stage-1 normalized" space;
# apply denormalize_neighborhood(...) to fully recover absolute distances.
```

### `get_item_from_raw` — process a raw training example

`GaussianPatchDataset.get_item_from_raw(raw_example)` applies the full
stage-2 normalization pipeline to a raw flat example array produced by
`create_train_example`, returning the same 4-tuple as `__getitem__`:

```python
neighborhood, point_features, target, valid_mask = \
    dataset.get_item_from_raw(raw_example)
```

This is used by `GaussianInputBuilder.build_input_via_dataset` to guarantee
that inference uses exactly the same normalization as training.

---

## Outlier Filtering

During training-patch generation, ring-$k$ neighborhoods on folded or
self-intersecting surfaces can contain **outlier neighbors** — points that are
close in Euclidean space but far on the surface (different mesh sheets).
These outliers distort both the spatial extent of the patch and the geodesic
distance distribution, hurting model generalisation.

`create_train_example` applies an adaptive **geodesic / Euclidean ratio
filter** immediately after retrieving the ring neighbors:

1. For each neighbor $i$, compute
   $r_i = \frac{d_{\text{geo}}(i)}{d_{\text{euc}}(i)}$.
2. Compute the median ratio $\tilde{r}$.
3. Any neighbor with
   $r_i > \max(5\tilde{r},\; 2)$
   is removed from the neighborhood.

The minimum bound of 2 prevents over-filtering on smooth surfaces where all
ratios are nearly identical. The 5× multiplier removes only ~1 % of neighbors
(the most extreme outliers from mesh folds) while preserving all legitimate
ring neighbors. After outlier removal, if the neighborhood still exceeds
`max_num_nbrs`, the closest neighbors by Euclidean distance are kept.

### `ring_size_mapping` considerations

After outlier filtering, effective ring sizes shrink slightly.  Current
defaults were determined from a comprehensive ring statistics analysis
(`DataSets/analyze_ring_statistics.py`) across all training shapes:

| Dataset     | ring-2 | ring-3 | Basis                              |
|-------------|--------|--------|------------------------------------|
| TOSCA       |     64 |    192 | max p99 = 60 / 185 across 14 shapes |
| Polynomial  |     48 |    128 | max p99 = 48 / 119 across 3 surfaces |

These values cover ≥ p99 of observed post-filter neighbor counts.  The
surplus slots are padded with duplicated neighbor features that do not
distort downstream normalization (see below).  Reducing `ring_size_mapping`
would discard real neighbors in the 1 % of dense patches.

### Adaptive kNN

Some points at the edge of the kNN graph or in locally sparse regions end up
with very few ring-$k$ neighbors even when the `ring_size_mapping` cap is not
reached. This can cause zero-prediction at evaluation time because the local
neighborhood graph creates geodesic "local minima" (no ring-$k$ neighbor has a
smaller geodesic distance).

To address this, the **adaptive kNN** mode (`adaptive_ring1_neighbors`)
performs a **per-point binary search** on the ring-1 neighbor count:

1. Compute kNN with `k_boost` once to obtain the full candidate neighbour
   arrays for every point as a 2-D array.
2. **Precompute** a lookup table of ring-$k$ counts for each uniform $k$
   from `n_neighbors` to `k_boost` via sparse-matrix reachability.
3. Each point $i$ maintains its own binary-search interval
   $[\text{lo}_i, \text{hi}_i]$ initialised to
   $[\text{n\_neighbors}, \text{k\_boost}]$.
4. At each step $\text{mid}_i = (\text{lo}_i + \text{hi}_i) // 2$; the
   ring-$k$ count is **looked up** from the precomputed table via numpy
   fancy-indexing — **O(N) per step, no matrix multiplication in the loop**.
5. If the looked-up count exceeds `adaptive_target_neighbors` (positive *cut*),
   $\text{hi}_i$ is lowered; otherwise $\text{lo}_i$ is raised.
6. The search stops when the **mean cut** across all points is
   ≤ `adaptive_max_mean_cut`, or after `adaptive_max_steps` iterations.

Enable it in any config by adding:

```yaml
adaptive_target_ring: 3            # ring level to optimize
adaptive_target_neighbors: 128     # desired ring-k count
adaptive_k_boost: 20               # max boosted k (upper bound of binary search)
adaptive_max_mean_cut: 5.0         # stop when mean cut ≤ this (default: 5.0)
adaptive_max_steps: 5              # max binary-search iterations (default: 5)
```

Or via command-line:

```bash
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/polynomial/combined_polynomial_all_one_source.yaml \
    --adaptive_target_ring 3 --adaptive_target_neighbors 128 --adaptive_k_boost 20 \
    --adaptive_max_mean_cut 5.0 --adaptive_max_steps 5
```

---

## Masking & Padding Strategy

### Padding (data-generation time)

When a ring neighborhood has fewer neighbors than `max_num_nbrs`, the
remaining slots must be padded.  The **new strategy** duplicates random
existing neighbors and sets **only their geodesic distance** to
`mask_constant` (`-10.0`):

```python
source_indices = np.random.choice(neighborhood.shape[0], pad_num, replace=True)
padding = neighborhood[source_indices].copy()
padding[:, -1] = mask_constant
```

This replaces the previous approach of filling padded entries with constant
sentinel values (`get_masked_entry`), which injected extreme values into the
feature distribution.  Because `_normalize_patches` normalises opacity
(min-max), scale (centroid + max-distance), and SH features across **all**
neighbor entries, constant sentinels would distort these statistics.  With the
new approach, padded entries carry realistic feature values and are invisible
to normalisation.

### Dropout (training time)

`GaussianPatchDropout` masks dropped neighbors by setting **only geodesic
distance** to `mask_constant`, preserving all other features.
`PointcloudRandomInputDropout` now follows the same principle: it duplicates a
random non-dropped entry and masks only the geodesic column.

### Valid-mask detection

```python
valid_mask = geodesic_distances != mask_constant   # True for valid, False for masked
```

This single criterion detects **both** padded entries and beyond-regime
neighbors (geodesic > p_u).  The model should only attend to valid entries.

---

## Available Attributes

Training patches can include various Gaussian attributes:

| Attribute | Size | Description |
|-----------|------|-------------|
| `xyz` | 3 | Position coordinates (always relative to center) |
| `normal` | 3 | Normal vector from smallest scale axis |
| `opacity` | 1 | Gaussian opacity |
| `scale` | 3 | Gaussian scales (log-space) |
| `rotation` | 4 | Rotation quaternion (w, x, y, z) |
| `sh` | varies | Spherical harmonics (SH degree 0-3) |
| `euclidean` | 1 | Euclidean distance to center |
| `geodesic` | 1 | Target geodesic distance |

## Data Format

Each training example has shape `(k, entry_size)` where:
- `k` = number of neighbors (from ring size mapping)
- `entry_size` = sum of attribute sizes + 2 (for geodesic and r1_min)

The last two columns are:
- `[:, -2]`: Target geodesic distance (`p_u`)
- `[:, -1]`: Ring-1 minimum distance (`r1_min_val`) for dropout augmentation.
  Computed from the **inverse ring-1** neighbourhood: for point *v*,
  `r1_min_val = min geodesic over {P : v ∈ ring1_nbrs[P]}`.  This matches
  the `reverse_ring_neighbors` direction used by Fast Marching.

## Available Transforms

| Transform | Description |
|-----------|-------------|
| `GaussianPatchRandomRotation` | Random SO(3) rotation of positions and normals |
| `SparseContextDropout` | Unified dropout replacing `GaussianPatchDropout`.  With prob `p_dropout`: drop valid neighbours beyond `r1_min_val`; otherwise keep only K closest (sparse-context mode). |
| `GaussianPatchDropout` | *(legacy)* Random neighbor dropout based on geodesic distance — superseded by `SparseContextDropout` |
| `GaussianPatchSurfacePerturb` | Perturb neighbour XYZ along surface normals |
| `GeodesicNoiseAugmentation` | Simulate min_input prediction error on geodesic distances |
| `PointcloudRandomInputDropout` | Random dropout of neighbor points |
| `Compose` | Compose multiple transforms |

See [TRANSFORM_USAGE.md](TRANSFORM_USAGE.md) for detailed examples.

## Testing

Run the test suite:

```bash
# All DataSets tests
python -m pytest DataSets/tests/ -v

# Specific test file
python -m pytest DataSets/tests/test_gaussian_dataset.py -v

# With coverage
python -m pytest DataSets/tests/ --cov=DataSets --cov-report=html
```

Current test coverage: **90 tests** across 5 test files.

## Module Dependencies

```
DataSets/
├── gaussian_dataset.py                      # Main entry point
│   ├── utils/config_utils.py                # Config loading
│   └── utils/data_transformation_utils.py   # Feature calculations
├── data_transformation.py                   # Transforms
│   ├── utils/data_transformation_utils.py
│   └── utils/rotation_conversions.py
├── create_gaussian_training_patches.py      # Polynomial patch generation
│   ├── utils/config_utils.py
│   ├── utils/training_patches_helpers.py
│   ├── utils/misc.py → fps_gs              # Optional FPS downsampling
│   └── GenerateData/...                     # External dependencies
└── create_tosca_training_patches.py         # TOSCA patch generation (area-weighted)
    ├── utils/config_utils.py
    ├── utils/training_patches_helpers.py    # Re-uses create_train_example
    ├── utils/misc.py → fps_gs              # Optional FPS downsampling
    ├── trimesh                              # Mesh loading + area computation
    ├── scipy.spatial.KDTree                # Source-to-vertex mapping
    └── GenerateData/...                    # External dependencies
```

## Related Modules

- `GenerateData/`: Scripts for creating synthetic datasets and computing geodesics
- `deep_eikonal_revisit/`: Neural network training for geodesic prediction
- `Gaussian_MAE/`: Transformer-based geodesic prediction model
