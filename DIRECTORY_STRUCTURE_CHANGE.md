# Directory Structure Change Summary

## Overview

Changed the directory structure from flat naming (`level_4_light_4`) to hierarchical structure with subdirectories.

## Changes

### Old Structure
```
/blue_texture/HyperbolicParaboloid/level_4_light_4/
/blue_texture/HyperbolicParaboloid/level_4/
```

### New Structure
```
/blue_texture/HyperbolicParaboloid/level_4/light_4/          # When --light_id 4
/blue_texture/HyperbolicParaboloid/level_4/default_light/    # When no light_id
/blue_texture/HyperbolicParaboloid/level_4/decoupled_appearance/  # When --use_decoupled_appearance
```

## Modified Files

1. **GenerateData/create_synthetic_colmap_dataset_from_mesh.py**
   - Changed path construction to create `level_XX/` directory first
   - Then subdirectory: `light_Y/`, `default_light/`, or `decoupled_appearance/`

2. **GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py**
   - Same changes for TOSCA dataset generation

3. **scripts/render_all_surfaces.sh**
   - Updated OUTPUT_DIR construction
   - Changed default LIGHT_IDS from "0,1,2,3,4" to "" (empty = default_light)

4. **scripts/compute_geodesic_polynomial_all.sh**
   - Updated GAUSSIAN_OUTPUT path construction
   - Uses new hierarchical structure

5. **scripts/train_polynomial_all.sh**
   - Updated SOURCE_PATH construction
   - Uses new hierarchical structure

## Path Logic

```python
# Python scripts
base_dir = Path(output_root) / f"{texture}_texture" / surface / f"level_{level:02d}"

if use_decoupled_appearance:
    dataset_dir = base_dir / "decoupled_appearance"
elif light_id is not None:
    dataset_dir = base_dir / f"light_{light_id}"
else:
    dataset_dir = base_dir / "default_light"
```

```bash
# Bash scripts
if [ -n "$light_id" ]; then
    PATH="$BASE/${texture}_texture/$surface/level_${level}/light_${light_id}"
else
    PATH="$BASE/${texture}_texture/$surface/level_${level}/default_light"
fi
```

## Testing Results

✅ All 17 tests pass (4.63s)
✅ Render script dry run: correct paths
✅ Train script dry run: correct paths
✅ Compute geodesic script dry run: correct paths

### Example Outputs

**With light_id:**
```
/colors_texture/Saddle/level_2/light_0/
/colors_texture/Saddle/level_2/light_2/
```

**Without light_id (default):**
```
/colors_texture/Saddle/level_2/default_light/
```

**With decoupled appearance:**
```
/colors_texture/Saddle/level_2/decoupled_appearance/
```

## Benefits

1. **Cleaner organization**: All variants of the same level are grouped together
2. **Easier comparison**: Can easily compare different lighting modes side-by-side
3. **Better scalability**: Adding new lighting modes doesn't clutter the parent directory
4. **Clearer intent**: Directory names explicitly state the mode (`default_light`, `decoupled_appearance`)

## Migration Notes

- Existing datasets with old structure will need to be regenerated or moved
- All scripts updated to use new structure consistently
- No backward compatibility with old path format
