# Configuration Files for Gaussian Training Example Generation

This directory contains YAML configuration files for generating training examples
from different Gaussian splat datasets, organized by dataset family.

## Directory Structure

```
DataSets/configs/
├── README.md                         # This file
├── polynomial/                       # Polynomial surface configs
│   ├── gaussian_sources/             # .txt files listing Gaussian output paths
│   │   ├── paraboloid_blue_all.txt
│   │   ├── saddle_blue_all.txt
│   │   └── hyperbolic_paraboloid_blue_all.txt
│   ├── paraboloid_all.yaml           # Paraboloid – xyz only
│   ├── paraboloid_all_scale_opacity.yaml
│   ├── paraboloid_all_mahalanobis.yaml
│   ├── saddle.yaml                   # Single-output quick experiment
│   ├── saddle_all.yaml
│   ├── saddle_all_scale_opacity.yaml
│   ├── saddle_all_mahalanobis.yaml
│   ├── hyperbolic_paraboloid_all.yaml
│   ├── hyperbolic_paraboloid_all_scale_opacity.yaml
│   ├── hyperbolic_paraboloid_all_mahalanobis.yaml
│   ├── combined_polynomial.yaml      # Legacy 3-surface combo
│   ├── combined_polynomial_all.yaml  # All 3 surfaces × 15 outputs = 45
│   ├── combined_polynomial_all_scale_opacity.yaml
│   ├── combined_polynomial_all_mahalanobis.yaml
│   ├── new_dataset/                  # One-source variants with augmentations
│   │   ├── combined_polynomial_all_one_source.yaml
│   │   ├── paraboloid_all_one_source.yaml
│   │   ├── saddle_all_one_source.yaml
│   │   └── hyperbolic_paraboloid_all_one_source.yaml
│   └── outlier_filtering/            # Outlier filtering with tunable params
│       ├── combined_polynomial_all_one_source.yaml   # 1 source
│       ├── combined_polynomial_all.yaml              # 4 sources
│       ├── paraboloid_all_one_source.yaml
│       ├── paraboloid_all.yaml
│       ├── saddle_all_one_source.yaml
│       ├── saddle_all.yaml
│       ├── hyperbolic_paraboloid_all_one_source.yaml
│       └── hyperbolic_paraboloid_all.yaml
│
└── tosca/                            # TOSCA mesh configs
    ├── gaussian_sources/             # .txt files listing Gaussian output paths
    │   ├── tosca_all_blue.txt        # All 148 poses combined
    │   ├── tosca_cat_blue.txt        # Per-animal source lists
    │   ├── tosca_centaur_blue.txt
    │   ├── ...
    │   └── tosca_wolf_blue.txt
    ├── tosca_cat.yaml                # Per-animal – xyz only
    ├── tosca_cat_scale_opacity.yaml  # Per-animal – xyz + scale + opacity
    ├── tosca_centaur.yaml
    ├── ...
    ├── tosca_wolf_scale_opacity.yaml
    ├── combined_tosca_all.yaml       # All 12 animals – xyz only
    ├── combined_tosca_all_scale_opacity.yaml  # All 12 animals – xyz + scale + opacity
    └── outlier_filtering/            # Outlier filtering with tunable params
        ├── combined_tosca_all.yaml   # All 9 animals – outlier filtering enabled
        ├── tosca_cat.yaml
        ├── tosca_centaur.yaml
        ├── ...
        └── tosca_wolf.yaml
```

---

## Polynomial Configs (`polynomial/`)

### Single-Surface Configs

- **`saddle.yaml`** — Saddle (z = x² − y²), single output, useful for quick experiments.

### Per-Surface, All Levels & Lights

Each config combines **all** COLMAP levels (02, 03, 04) and lighting conditions (0–4),
giving 15 Gaussian outputs per surface. They reference `.txt` files in
`polynomial/gaussian_sources/` and use `dataset_class: GaussianPatchDataset`.

| Config | Surface | Outputs | Attributes |
|--------|---------|---------|------------|
| `paraboloid_all.yaml` | Paraboloid (z = x² + y²) | 15 | xyz |
| `saddle_all.yaml` | Saddle (z = x² − y²) | 15 | xyz |
| `hyperbolic_paraboloid_all.yaml` | HyperbolicParaboloid (z = xy) | 15 | xyz |
| `*_scale_opacity.yaml` | (same) | 15 | xyz, scale, opacity |
| `*_mahalanobis.yaml` | (same) | 15 | xyz (Mahalanobis KNN) |

### Multi-Source Combined Configs

Load all 3 polynomial surfaces at once via `CombinedGaussianPatchDataset`.

| Config | Attributes | Total |
|--------|------------|-------|
| `combined_polynomial_all.yaml` | xyz | 45 |
| `combined_polynomial_all_scale_opacity.yaml` | xyz, scale, opacity | 45 |
| `combined_polynomial_all_mahalanobis.yaml` | xyz (Mahalanobis) | 45 |

### Outlier Filtering Configs (`polynomial/outlier_filtering/`)

Configs with outlier filtering **enabled** (`disable_outlier_filtering: false`) and
explicit tunable parameters. Two variants — one source and four sources — without
Mahalanobis configurations.

| Config | Sources | Surfaces |
|--------|---------|----------|
| `combined_polynomial_all_one_source.yaml` | 1 | All 3 |
| `combined_polynomial_all.yaml` | 4 | All 3 |
| `paraboloid_all_one_source.yaml` / `paraboloid_all.yaml` | 1 / 4 | Paraboloid |
| `saddle_all_one_source.yaml` / `saddle_all.yaml` | 1 / 4 | Saddle |
| `hyperbolic_paraboloid_all_one_source.yaml` / `hyperbolic_paraboloid_all.yaml` | 1 / 4 | HyperbolicParaboloid |

---

## TOSCA Configs (`tosca/`)

### Data Overview

12 animal types, 148 total poses, blue_texture, high_res, decoupled_appearance.

| Animal | Poses | Mesh Diagonal |
|--------|-------|---------------|
| cat | 9 | ~223.5 |
| centaur | 6 | ~288.8 |
| david | 15 | ~234.7 |
| dog | 11 | ~198.4 |
| gorilla | 21 | ~188.6 |
| horse | 17 | ~330.0 |
| lioness | 15 | ~290.5 |
| michael | 20 | ~261.9 |
| seahorse | 6 | ~482.8 |
| shark | 1 | ~531.6 |
| victoria | 24 | ~243.4 |
| wolf | 3 | ~200.0 |

### Per-Animal Configs

Each animal has two configs using `GaussianPatchDataset`:

| Config | Attributes |
|--------|------------|
| `tosca_{animal}.yaml` | xyz |
| `tosca_{animal}_scale_opacity.yaml` | xyz, scale, opacity |

### Multi-Source Combined Configs

All 12 animals in a single `CombinedGaussianPatchDataset`:

| Config | Attributes | Total poses |
|--------|------------|-------------|
| `combined_tosca_all.yaml` | xyz | 148 |
| `combined_tosca_all_scale_opacity.yaml` | xyz, scale, opacity | 148 |

### Outlier Filtering Configs (`tosca/outlier_filtering/`)

TOSCA configs with outlier filtering **enabled** (`disable_outlier_filtering: false`)
and explicit tunable parameters. The base TOSCA configs have filtering disabled because
TOSCA meshes can have complex folds; these variants enable it for experimentation.

| Config | Animals | Sources |
|--------|---------|---------|
| `combined_tosca_all.yaml` | All 9 | 4 |
| `tosca_{animal}.yaml` | Single | 2 |

### Gaussian Source Files (`tosca/gaussian_sources/`)

One `.txt` per animal listing all its Gaussian output paths, plus a combined
`tosca_all_blue.txt` listing all 148.

---

## Configuration File Format

```yaml
# Dataset class
dataset_class: GaussianPatchDataset   # or CombinedGaussianPatchDataset

# Paths
gaussian_output: "DataSets/configs/tosca/gaussian_sources/tosca_cat_blue.txt"
geodesic_data: null    # null = auto-detect
output_dir: "TrainData/datasets/gaussian_patches/tosca_cat"
iteration: null        # null = highest available

# Data generation parameters
num_iterations: 1000
num_sources: 4
num_train_points: 50
seed: 42

# Neighbourhood
use_mahalanobis: false
n_neighbors: 10
normalize_per_patch: true

# Ring configuration
rings: [2, 3]
ring_size_mapping:
  euclidean:  { 2: 32, 3: 128, 4: 512 }
  mahalanobis: { 2: 90, 3: 250, 4: 600 }

# Feature attributes (options: xyz, opacity, rotation, sh, normals, scale, euclidean_distances)
attributes:
  - xyz

# Normalization
nn_mean: 1
use_r1_min_val: true
mask_constant: -10.0

# Outlier filtering (training path)
disable_outlier_filtering: false       # true = skip all outlier filtering
outlier_median_multiplier: 3.0         # adaptive_threshold = max(median * this, floor)
outlier_threshold_floor: 2.0           # minimum adaptive threshold
outlier_hard_cap: 500.0                # absolute cap on outlier threshold
outlier_fallback_multiplier: 5.0       # relaxed multiplier when all neighbors filtered
outlier_fallback_floor: 3.0            # floor for fallback threshold

# Optional: compute max_dist over ALL neighbors (including padded) for XYZ normalization.
# Default is false (only valid neighbors contribute to max_dist).
# normalize_all_neighbors: false
```

For `CombinedGaussianPatchDataset` configs, replace `gaussian_output` with a
`data_sources` list — see `combined_tosca_all.yaml` for an example.

---

## Quick Reference

| Goal | Config |
|------|--------|
| Polynomial baseline (xyz) | `polynomial/combined_polynomial_all.yaml` |
| Polynomial with scale+opacity | `polynomial/combined_polynomial_all_scale_opacity.yaml` |
| Polynomial with Mahalanobis KNN | `polynomial/combined_polynomial_all_mahalanobis.yaml` |
| Polynomial + outlier filtering (1 src) | `polynomial/outlier_filtering/combined_polynomial_all_one_source.yaml` |
| Polynomial + outlier filtering (4 src) | `polynomial/outlier_filtering/combined_polynomial_all.yaml` |
| Single polynomial, quick test | `polynomial/saddle.yaml` |
| TOSCA single animal | `tosca/tosca_cat.yaml` |
| TOSCA all animals (xyz) | `tosca/combined_tosca_all.yaml` |
| TOSCA all animals (scale+opacity) | `tosca/combined_tosca_all_scale_opacity.yaml` |
| TOSCA + outlier filtering | `tosca/outlier_filtering/combined_tosca_all.yaml` |

### Python Loading Example

```python
from DataSets.gaussian_dataset import CombinedGaussianPatchDataset

# Polynomial – xyz only
dataset = CombinedGaussianPatchDataset(
    config='DataSets/configs/polynomial/combined_polynomial_all.yaml',
    attributes=['xyz'], ring=3
)

# TOSCA – all animals
dataset = CombinedGaussianPatchDataset(
    config='DataSets/configs/tosca/combined_tosca_all.yaml',
    attributes=['xyz'], ring=2
)
```

### Patch Generation

```bash
# Single surface
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/polynomial/paraboloid_all.yaml

# Single TOSCA animal
python DataSets/create_gaussian_training_patches.py \
    --config DataSets/configs/tosca/tosca_cat.yaml

# All polynomial via script
bash scripts/polynomial/generate_training_patches.sh

# All TOSCA via script
bash scripts/tosca/generate_training_patches.sh
```

---

## Tips

- Start with `combined_polynomial_all.yaml` or `combined_tosca_all.yaml` for maximum diversity.
- Use `_scale_opacity` configs when Gaussian scale/opacity carry geometric meaning.
- Use `_mahalanobis` configs when Gaussian anisotropy should drive neighbourhood selection.
- Increase `num_iterations` for more training examples.
- Include higher `rings` for learning long-range geodesic patterns.
- TOSCA mesh diagonals vary from ~189 to ~532 — keep this in mind for normalisation.
