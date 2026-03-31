# Light ID Feature Documentation

## Overview

This document describes the light_id feature that enables systematic comparison of different lighting configurations throughout the entire RaDe-GS pipeline.

## Feature Summary

The system now supports three distinct lighting modes:

1. **Standard Lighting** (default): Balanced 7-light setup with indirect lighting
2. **Decoupled Appearance**: 10 distinct lighting groups with random view shuffling (use_decoupled_appearance)
3. **Fixed Light Presets**: 5 preset lighting configurations (light_id 0-4)

## Components Modified

### 1. Core Rendering System

**File**: `GenerateData/utils/rendering_utils.py`

Key changes:
- Added `light_id` parameter to `_render_images_open3d()`
- Enhanced `setup_lighting_for_group()` to handle three modes:
  - `group_id=-1`: Default balanced lighting
  - `group_id=0-4`: Fixed light_id presets
  - `group_id=5+`: Decoupled appearance variations
- Implemented unique light naming with counter to avoid Open3D warnings
- Added random view shuffling with seed=42 for reproducibility

Light ID Presets:
- **0**: Default balanced lighting (7 lights + indirect)
- **1**: Strong frontal lighting (dramatic shadows)
- **2**: Warm ambient lighting (reduced shadows)
- **3**: Cool side lighting (enhanced contours)
- **4**: Soft diffuse lighting (minimal shadows)

### 2. Dataset Generation Scripts

**Files**: 
- `GenerateData/create_synthetic_colmap_dataset_from_mesh.py`
- `GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py`

Changes:
- Added `--light_id` command-line argument (0-4, default: None)
- Updated path construction to include `_light_{light_id}` suffix when specified
- Logic: Only add light_id to path when `use_decoupled_appearance=False` and `light_id` is provided

Path structure:
```
# Without light_id
{texture}_texture/{surface}/level_{colmap_level}/

# With light_id
{texture}_texture/{surface}/level_{colmap_level}_light_{light_id}/
```

### 3. Batch Processing Scripts

All three batch scripts now support light_id iteration:

#### render_all_surfaces.sh
```bash
# Render with multiple light_ids
./scripts/render_all_surfaces.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --colmap_levels "2,3" \
    --light_ids "0,1,2,3,4"
```

#### compute_geodesic_polynomial_all.sh
```bash
# Compute geodesics for all light_ids
./scripts/compute_geodesic_polynomial_all.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --levels "02,03" \
    --light_ids "0,1,2,3,4" \
    --outputs "output"
```

#### train_polynomial_all.sh
```bash
# Train on all light_id variants
./scripts/train_polynomial_all.sh \
    --surfaces "Paraboloid,Saddle" \
    --textures "colors" \
    --levels "02,03" \
    --light_ids "0,1,2,3,4" \
    --outputs "run1"
```

Changes to batch scripts:
- Added `LIGHT_IDS=""` variable (empty = standard lighting)
- Added `--light_ids` command-line option
- Created `LIGHT_ID_ARRAY` from comma-separated input
- Updated job counting to include light_id loop
- Modified main loops to iterate over light_id array
- Updated path construction with conditional `_light_{light_id}` suffix

### 4. Testing

**File**: `GenerateData/tests/test_rendering_utils.py`

Test suite (17 tests, all passing):
- Point sampling (5 tests): threshold, reproducibility, seeds, color conversion
- UV coordinates (3 tests): range, corners, negative vertices
- Lighting configuration (6 tests): group count, light_id range, mode exclusivity, view shuffling
- Brightness clamping (2 tests): dark pixels, preserve bright
- Integration (1 test): full pipeline mock

New tests added:
- `test_fixed_light_id_range()`: Validates light_id 0-4 range
- `test_light_id_modes_mutually_exclusive()`: Ensures correct mode selection

## Usage Examples

### 1. Generate Dataset with Specific Light ID

```bash
# Generate with light_id=2 (warm ambient lighting)
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Saddle \
    --texture_name colors \
    --colmap_level 2 \
    --light_id 2

# Output: TrainData/Polynomial/SyntheticColmapData/colors_texture/Saddle/level_02_light_2/
```

### 2. Batch Render All Light IDs

```bash
# Render all surfaces with all 5 light presets
./scripts/render_all_surfaces.sh \
    --light_ids "0,1,2,3,4" \
    --surfaces "Paraboloid,Saddle,HyperbolicParaboloid" \
    --textures "colors" \
    --colmap_levels "2,3,4"

# Creates:
# - level_02_light_0/, level_02_light_1/, ..., level_02_light_4/
# - level_03_light_0/, level_03_light_1/, ..., level_03_light_4/
# - level_04_light_0/, level_04_light_1/, ..., level_04_light_4/
```

### 3. Train on All Light Variants

```bash
# Train models for each light_id
./scripts/train_polynomial_all.sh \
    --light_ids "0,1,2,3,4" \
    --surfaces "Paraboloid" \
    --textures "colors" \
    --levels "02" \
    --outputs "baseline" \
    --iterations 30000

# Creates models in:
# - level_02_light_0/baseline/
# - level_02_light_1/baseline/
# - level_02_light_2/baseline/
# - level_02_light_3/baseline/
# - level_02_light_4/baseline/
```

### 4. Compute Geodesics for All Lighting

```bash
# Compute geodesic distances for all light variants
./scripts/compute_geodesic_polynomial_all.sh \
    --light_ids "0,1,2,3,4" \
    --surfaces "Saddle" \
    --textures "colors" \
    --levels "02" \
    --outputs "baseline" \
    --n_jobs 4

# Processes:
# - level_02_light_0/baseline/geodesic_distance/
# - level_02_light_1/baseline/geodesic_distance/
# - ... (for all light_ids)
```

### 5. Backward Compatibility

All scripts maintain backward compatibility. Omitting `--light_ids` uses standard lighting:

```bash
# Without light_ids - uses standard lighting
./scripts/render_all_surfaces.sh \
    --surfaces "Saddle" \
    --textures "colors" \
    --colmap_levels "2"

# Creates: level_02/ (no light_id suffix)
```

## Implementation Details

### Light Naming Strategy

To avoid Open3D duplicate light warnings, unique names are generated using a counter:
- `key_light_{counter}`
- `fill_1_{counter}`
- `fill_2_{counter}`
- etc.

The counter increments globally across all lighting setups.

### Random View Shuffling

For decoupled appearance mode:
- Uses `numpy.random.default_rng(seed=42)` for reproducibility
- Shuffles view indices randomly
- Distributes shuffled views evenly across 10 lighting groups
- Ensures consistent group assignment across runs

### Path Construction Logic

```python
if light_id is not None and not use_decoupled_appearance:
    path = f"{base}/level_{level}_light_{light_id}"
else:
    path = f"{base}/level_{level}"
```

### Batch Script Pattern

All batch scripts follow this pattern:

1. **Variable initialization**: `LIGHT_IDS=""`
2. **Argument parsing**: `--light_ids "$2"`
3. **Array creation**: Handle empty string as single iteration
4. **Job counting**: Include light_id loop in total
5. **Main loop**: Add nested light_id iteration
6. **Path construction**: Conditional suffix based on light_id
7. **Command construction**: Add `--light_id` flag when set

## Testing Results

All 17 tests pass successfully:

```
GenerateData/tests/test_rendering_utils.py::TestPointSampling (5 tests) ✓
GenerateData/tests/test_rendering_utils.py::TestUVCoordinates (3 tests) ✓
GenerateData/tests/test_rendering_utils.py::TestLightingConfiguration (6 tests) ✓
GenerateData/tests/test_rendering_utils.py::TestBrightnessClamping (2 tests) ✓
GenerateData/tests/test_rendering_utils.py::TestIntegration (1 test) ✓

Total: 17 passed in 36.33s
```

## Troubleshooting

### Open3D Light Warnings

If you see warnings like "Cannot add point light because {name} has already been added":
- This is resolved in the current implementation
- Each light setup uses unique counter-based names
- No action needed from users

### Path Not Found Errors

If scripts report missing paths:
1. Verify dataset was generated with same light_id
2. Check that `--light_ids` matches across pipeline stages
3. Use `--dry_run` to preview paths before execution

### Lighting Too Bright/Dark

Brightness clamping is applied to all modes:
- Minimum brightness: 20/255
- Applied uniformly across all light_id presets
- Preserves relative brightness differences

## Future Enhancements

Potential improvements:
1. Add more light_id presets (currently 0-4)
2. Support custom lighting configurations via config files
3. Add lighting comparison visualization tools
4. Integrate with evaluation metrics for lighting impact analysis

## Related Files

- Core rendering: [GenerateData/utils/rendering_utils.py](GenerateData/utils/rendering_utils.py)
- Tests: [GenerateData/tests/test_rendering_utils.py](GenerateData/tests/test_rendering_utils.py)
- Batch scripts:
  - [scripts/render_all_surfaces.sh](scripts/render_all_surfaces.sh)
  - [scripts/compute_geodesic_polynomial_all.sh](scripts/compute_geodesic_polynomial_all.sh)
  - [scripts/train_polynomial_all.sh](scripts/train_polynomial_all.sh)

## Version History

- **v1.0**: Initial implementation
  - Basic light_id support (0-4)
  - Decoupled appearance with 10 groups
  - Random view shuffling
  - Batch script integration
  - Comprehensive test suite
