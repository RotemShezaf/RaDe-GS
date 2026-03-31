# Light ID Quick Reference

## Quick Start

### Single Dataset Generation
```bash
# Generate with default lighting (no light_id)
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Saddle --texture_name colors --colmap_level 2

# Generate with specific light preset
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Saddle --texture_name colors --colmap_level 2 --light_id 2
```

### Batch Processing - Full Pipeline

```bash
# Step 1: Render datasets with all light presets
./scripts/render_all_surfaces.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --colmap_levels "2,3" \
    --light_ids "0,1,2,3,4"

# Step 2: Train on all variants
./scripts/train_polynomial_all.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --levels "02,03" \
    --light_ids "0,1,2,3,4" \
    --outputs "baseline" \
    --iterations 30000

# Step 3: Compute geodesics
./scripts/compute_geodesic_polynomial_all.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --levels "02,03" \
    --light_ids "0,1,2,3,4" \
    --outputs "baseline"
```

## Light ID Presets

| ID | Name | Description | Use Case |
|----|------|-------------|----------|
| 0 | Default | Balanced 7-light setup with indirect | Baseline comparison |
| 1 | Dramatic | Strong frontal lighting, sharp shadows | High-contrast scenes |
| 2 | Warm | Ambient warm lighting, soft shadows | Natural indoor lighting |
| 3 | Cool | Side lighting, enhanced contours | Emphasize geometry |
| 4 | Diffuse | Soft uniform lighting, minimal shadows | Texture detail |

## Common Commands

### Dry Run (Preview)
```bash
# Preview what would be executed
./scripts/render_all_surfaces.sh --light_ids "0,1,2" --dry_run
./scripts/train_polynomial_all.sh --light_ids "0,1,2" --dry_run
./scripts/compute_geodesic_polynomial_all.sh --light_ids "0,1,2" --dry_run
```

### Subset Testing
```bash
# Test with single surface and two light presets
./scripts/render_all_surfaces.sh \
    --surfaces "Saddle" \
    --light_ids "0,2" \
    --colmap_levels "2"
```

### Parallel Processing
```bash
# Render with maximum parallelism
./scripts/render_all_surfaces.sh \
    --light_ids "0,1,2,3,4" \
    --max_parallel 8

# Train with parallel jobs
./scripts/train_polynomial_all.sh \
    --light_ids "0,1,2,3,4" \
    --max_parallel 4
```

## Path Structure

### Dataset Paths
```
# Without light_id
TrainData/Polynomial/SyntheticColmapData/
└── colors_texture/
    └── Saddle/
        └── level_02/

# With light_id=2
TrainData/Polynomial/SyntheticColmapData/
└── colors_texture/
    └── Saddle/
        └── level_02_light_2/
```

### Output Paths
```
# Training output
level_02_light_2/
├── baseline/
│   ├── point_cloud/
│   ├── cameras.json
│   └── ...

# Geodesic output
level_02_light_2/
└── baseline/
    └── geodesic_distance/
        ├── gt_geodesic.npz
        └── ...
```

## Modes Comparison

| Feature | Standard | Decoupled Appearance | Light ID |
|---------|----------|---------------------|----------|
| Trigger | Default (no flags) | `--use_decoupled_appearance` | `--light_id 0-4` |
| Groups | 1 | 10 (random assignment) | 5 (fixed presets) |
| Path suffix | None | None | `_light_{id}` |
| Use case | Baseline | Appearance variation | Systematic comparison |

## Troubleshooting

### Issue: "Source not found"
**Solution**: Ensure datasets are rendered before training:
```bash
# First render
./scripts/render_all_surfaces.sh --light_ids "0,1,2"

# Then train
./scripts/train_polynomial_all.sh --light_ids "0,1,2"
```

### Issue: "Gaussian output not found"
**Solution**: Ensure training is complete before computing geodesics:
```bash
# First train
./scripts/train_polynomial_all.sh --light_ids "0,1,2"

# Then compute geodesics
./scripts/compute_geodesic_polynomial_all.sh --light_ids "0,1,2"
```

### Issue: Inconsistent paths
**Solution**: Use same `--light_ids` across all pipeline stages:
```bash
LIGHT_IDS="0,1,2,3,4"

./scripts/render_all_surfaces.sh --light_ids "$LIGHT_IDS" ...
./scripts/train_polynomial_all.sh --light_ids "$LIGHT_IDS" ...
./scripts/compute_geodesic_polynomial_all.sh --light_ids "$LIGHT_IDS" ...
```

## Testing

### Run Test Suite
```bash
# All tests
cd /home/rotem.shezaf/RaDe-GS
source ~/miniconda3/bin/activate radegs_reproduce
pytest GenerateData/tests/test_rendering_utils.py -v

# Specific test
pytest GenerateData/tests/test_rendering_utils.py::TestLightingConfiguration::test_fixed_light_id_range -v
```

### Quick Validation
```bash
# Test single rendering
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Saddle --light_id 2 --num_views 10

# Verify output
ls -la TrainData/Polynomial/SyntheticColmapData/colors_texture/Saddle/level_02_light_2/
```

## Performance Tips

1. **Use dry_run first**: Preview job count and paths
2. **Adjust parallelism**: Match to CPU cores
3. **Test subset**: Validate with one surface before full batch
4. **Sequential for debugging**: Use `--sequential` to see errors immediately

## Examples by Use Case

### Compare Lighting Impact on Training
```bash
# Train same surface with all lighting presets
./scripts/train_polynomial_all.sh \
    --surfaces "Saddle" \
    --textures "colors" \
    --levels "02" \
    --light_ids "0,1,2,3,4" \
    --outputs "compare_lighting" \
    --iterations 30000
```

### Evaluate Geodesic Accuracy Across Lighting
```bash
# Compute geodesics for all light variants
./scripts/compute_geodesic_polynomial_all.sh \
    --surfaces "Paraboloid,Saddle,HyperbolicParaboloid" \
    --light_ids "0,1,2,3,4" \
    --outputs "baseline" \
    --n_jobs 8
```

### Create Dataset Variants for Ablation Study
```bash
# Render with specific lighting combinations
./scripts/render_all_surfaces.sh \
    --surfaces "Saddle" \
    --textures "colors,wood,marble" \
    --colmap_levels "2,3,4" \
    --light_ids "0,2,4"  # Test default, warm, and diffuse only
```

## Related Documentation

- Detailed guide: [LIGHT_ID_FEATURE.md](LIGHT_ID_FEATURE.md)
- Main README: [README.md](README.md)
- Test suite: [GenerateData/tests/test_rendering_utils.py](GenerateData/tests/test_rendering_utils.py)
