# Depth Normal Loss Bug Fix

## Problem Summary

The `depth_normal_loss` was not learning well during training because **rendered normals were not normalized before computing the dot product error**.

## Root Cause

### The Math
The depth normal loss computes error as:
```python
error = 1 - dot_product(rendered_normal, depth_derived_normal)
```

For two vectors **a** and **b**, the dot product is:
```
a · b = |a| * |b| * cos(θ)
```

For this error metric to be meaningful:
- Both vectors must have magnitude 1.0 (normalized)
- Then: `dot_product = cos(θ)` ranges from [-1, 1]
- And: `error = 1 - cos(θ)` ranges from [0, 2]

### The Bug
In the original code:
1. `depth_middepth_normal` **IS normalized** (via `torch.nn.functional.normalize` in `point_double_to_normal`)
2. `rendered_normal` from rasterization **IS NOT normalized** - it's an alpha-blended weighted average

This means:
- `|rendered_normal|` ≈ 0.4-0.6 (as observed in debug output)
- The dot product is artificially reduced by ~50-60%
- The error metric becomes meaningless
- The network can't learn properly

### Evidence from Debug Output

**Before fix:**
```
Normal magnitudes: rendered=0.48, depth=1.00, middepth=1.00
Dot products: depth=0.46±0.49
loss_normal: 0.046 (artificially low, not improving)
```

**After fix:**
```
Normal magnitudes: rendered=1.00 (normalized), depth=1.00, middepth=1.00
Dot products: depth=0.56±0.49 (realistic)
loss_normal progression: 0.365 → 0.311 → 0.101 → 0.049 → 0.043 (clearly improving!)
```

## The Fix

### Changes Made

**File: `/home/rotem.shezaf/RaDe-GS/train.py`**
**File: `/home/rotem.shezaf/RaDe-GS/train_debug.py`**

```python
# OLD CODE (BUGGY):
normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))

# NEW CODE (FIXED):
# Normalize rendered_normal to ensure proper dot product calculation
# (rendered normals are alpha-blended, so they need normalization)
rendered_normal_normalized = torch.nn.functional.normalize(rendered_normal, dim=0)

depth_ratio = 0.6
normal_error_map = (1 - (rendered_normal_normalized.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
```

### Impact

With this fix:
1. The error metric is now mathematically correct
2. The loss starts at a realistic high value (~0.36 instead of ~0.05)
3. **The loss decreases consistently throughout training** (0.36 → 0.04)
4. The network can properly learn to match normals

## Testing

Tested on Saddle surface (level_03, blue texture):
- 500 iterations with regularization from iter 0
- Loss decreased from 0.365 to 0.043
- Dot products increased from ~0.38 to ~0.56
- Debug visualizations show proper normal alignment improvement

## Recommendation

This fix should be applied to:
1. ✅ `train.py` (already fixed)
2. ✅ `train_debug.py` (already fixed)
3. Any other code that computes normal errors from rasterized normals

## Technical Note

The `depth_derived_normal` vectors ARE properly normalized because:
1. They come from cross products: `normal = cross(dx, dy)`
2. Which are then normalized: `torch.nn.functional.normalize(normal, dim=1)`
3. This happens in `point_double_to_normal()` in `utils/graphics_utils.py`

The `rendered_normal` vectors are NOT normalized because:
1. They come from alpha-blended rasterization: `Σ(α_i * normal_i)`
2. Even if each `normal_i` is normalized, the weighted sum is not
3. Example: `0.5 * [1,0,0] + 0.5 * [0,1,0] = [0.5, 0.5, 0]` has magnitude 0.707, not 1.0

## Date
Fixed: February 3, 2026
