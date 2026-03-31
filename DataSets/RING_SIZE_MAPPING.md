# Ring Size Mapping Reference

This document records the empirically determined `ring_size_mapping` values
used across all dataset configurations.  Values were chosen to cover ≥ p99
of observed post-filter neighbor counts, based on a comprehensive analysis
by `DataSets/analyze_ring_statistics.py`.

## Summary

| Dataset     | Distance | ring-2 | ring-3 | ring-4 | Analysis basis                            |
|-------------|----------|--------|--------|--------|-------------------------------------------|
| TOSCA       | Euclidean     |     64 |    192 |    512 | max p99 = 60 / 185 across 14 shapes       |
| TOSCA       | Mahalanobis   |     90 |    250 |    600 | Mahalanobis distances cover wider area     |
| Polynomial  | Euclidean     |     48 |    128 |    512 | max p99 = 48 / 119 across 3 surfaces      |
| Polynomial  | Mahalanobis   |     90 |    250 |    600 | (symmetric with TOSCA Mahalanobis)         |

## Outlier Filtering

Before counting neighbors, an adaptive outlier filter removes neighbors from
different surface sheets (mesh folds):

- For each neighbor, compute `ratio = geodesic_distance / euclidean_distance`
- Threshold: `max(median(ratio) × 5.0, 2.0)`
- Neighbors exceeding the threshold are excluded

This removes ~1% of neighbors, affecting ~30-50% of points (mostly by
removing just 1-2 outliers per point).

## How values were determined

1. For each shape/surface, load Gaussian splats and compute ring-1/2/3
   neighborhoods (k=10 ring-1 neighbors).
2. Apply the 5× outlier filter to get post-filter neighbor counts.
3. Compute percentiles (p95, p99, max) across 3000 sampled points per shape.
4. Take the maximum p99 across all shapes/surfaces in the dataset.
5. Round up to the nearest multiple of 16.

## Raw Statistics (from analyze_ring_statistics.py)

### TOSCA (14 shapes, Euclidean)

**ring-2:**
- Mean of means: 31.6, mean of p99: 50.4 (range 39-60)
- Post-filter: mean of means: 31.1, mean of p99: 50.8 (range 39-60)
- **Recommended: 64** (max p99 = 60)

**ring-3:**
- Mean of means: 69.7, mean of p99: 142.6 (range 81-185)
- Post-filter: mean of means: 68.3, mean of p99: 133.8 (range 80-170)
- **Recommended: 192** (max p99 = 185; for safety w/ 333 absolute max)

### Polynomial (3 surfaces, Euclidean)

**ring-2:**
- Mean of means: 30.2, mean of p99: 43.0 (range 40-48)
- Post-filter: mean of means: 30.0, mean of p99: 43.3 (range 41-48)
- **Recommended: 48** (max p99 = 48)

**ring-3:**
- Mean of means: 63.1, mean of p99: 98.3 (range 87-120)
- Post-filter: mean of means: 62.5, mean of p99: 97.3 (range 86-119)
- **Recommended: 128** (max p99 = 119)

## Config Files

### TOSCA configs (`DataSets/configs/tosca/`)

All per-shape and combined configs use:
```yaml
ring_size_mapping:
  euclidean:
    2: 64
    3: 192
    4: 512
  mahalanobis:
    2: 90
    3: 250
    4: 600
```

### Polynomial configs (`DataSets/configs/polynomial/`)

All `*_all*.yaml` and `combined_*.yaml` configs use:
```yaml
ring_size_mapping:
  euclidean:
    2: 48
    3: 128
    4: 512
  mahalanobis:
    2: 90
    3: 250
    4: 600
```

## Analysis Tools

- **`DataSets/analyze_ring_statistics.py`**: Compute ring neighbor count
  distributions, geo/euc ratio analysis, and outlier filter impact for
  TOSCA and polynomial datasets.

- **`DataSets/verify_datasets.py`**: Verify generated .npy datasets for
  correctness (no NaN/Inf, correct shapes, padding statistics, etc.)

- **`DataSets/ring_statistics_report.txt`**: Full output from the most
  recent run of `analyze_ring_statistics.py`.
