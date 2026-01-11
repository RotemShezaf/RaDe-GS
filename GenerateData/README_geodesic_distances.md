# Ground Truth Geodesic Distance Calculator for Gaussian Splats

This script computes ground truth geodesic distances by projecting from a high-resolution mesh to Gaussian splat positions (low-resolution point cloud), following the TOSCA methodology.

## Overview

**Analogy to TOSCA:**
- **High-res mesh**: Ground truth polynomial mesh at level 0  
- **Low-res mesh**: Gaussian point cloud (output of training)  
- **Process**: Compute exact geodesics on high-res mesh, then project to Gaussian positions

## Key Concepts

### Partitioning (NUM_SOURCES Pattern)

For large Gaussian datasets, computation can be split into parts for:
- Parallel/distributed processing
- Memory-constrained environments
- Faster iteration during development

**Parameters:**
- `--num_sources`: Number of Gaussians to process per part (e.g., 1000)
- `--part N`: Which part to process (0-indexed)

**How it works:**
```
Total Gaussians = 10,000
NUM_SOURCES = 1000

Part 0: Processes Gaussians [0:1000]
Part 1: Processes Gaussians [1000:2000]
...
Part 9: Processes Gaussians [9000:10000]

Total parts needed: 10
```

The last part may process fewer than NUM_SOURCES if the total doesn't divide evenly.

## Usage

### 1. Process All Gaussians at Once (No Partitioning)

```bash
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --source_vertex_idx 0
```

**Output:** `output/polynomial_experiment/point_cloud/iteration_30000/geodesic/gt_distances_all.npy`

### 2. Process in Parts (Recommended for Large Datasets)

#### Step 1: Determine NUM_SOURCES

Choose based on available memory and dataset size. Typical values: 500-2000.

#### Step 2: Run Each Part

```bash
# Part 0
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --num_sources 1000 \
    --part 0

# Part 1
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --num_sources 1000 \
    --part 1

# Continue for all parts...
```

**Output:** `geodesic/part0.npy`, `geodesic/part1.npy`, etc.

#### Step 3: Unite All Parts

After all parts complete:

```bash
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --surface Paraboloid \
    --unite
```

**Output:** `geodesic/gt_distances_united.npy`

### 3. Process Multiple Iterations

Add `--all_iterations` to process all available iterations:

```bash
# Process all iterations in parts
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --num_sources 1000 \
    --part 0 \
    --all_iterations

# Unite all iterations
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/polynomial_experiment \
    --surface Paraboloid \
    --unite \
    --all_iterations
```

## Parallel Processing

Run parts in parallel on different machines/GPUs:

```bash
# Machine 1
python calculate_gaussian_geodesic_distances.py ... --part 0

# Machine 2
python calculate_gaussian_geodesic_distances.py ... --part 1

# Machine 3
python calculate_gaussian_geodesic_distances.py ... --part 2

# etc.
```

Then collect all `part*.npy` files in the geodesic folder and run `--unite`.

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--gaussian_output` | Yes | Path to Gaussian splatting output folder |
| `--data_root` | Yes* | Path to raw polynomial mesh data |
| `--surface` | Yes | Surface type (Paraboloid, Saddle, HyperbolicParaboloid) |
| `--source_vertex_idx` | No | Source vertex for geodesic computation (default: 0) |
| `--level` | No | GT mesh resolution level (default: 0 = highest) |
| `--iteration` | No | Iteration to process (default: 30000) |
| `--all_iterations` | No | Process all available iterations |
| `--num_sources` | No | Gaussians per part (enables partitioning) |
| `--part` | No | Part number to process (requires --num_sources) |
| `--unite` | No | Unite all parts into single file |
| `--force_recompute` | No | Ignore geodesic distance cache |

*Not required when using `--unite`

## Output Structure

```
output/polynomial_experiment/
└── point_cloud/
    └── iteration_30000/
        ├── point_cloud.ply           # Original Gaussian splats
        └── geodesic/
            ├── part0.npy             # Part 0 distances
            ├── part1.npy             # Part 1 distances
            ├── ...
            └── gt_distances_united.npy  # All parts united (after --unite)
```

## File Formats

- **part{N}.npy**: NumPy array of shape `(num_gaussians_in_part,)` containing geodesic distances
- **gt_distances_united.npy**: NumPy array of shape `(total_gaussians,)` containing all distances
- **gt_distances_all.npy**: Same as united, but created when processing without partitioning

## Cache

Geodesic distance computation on the high-res mesh is cached at:
```
TrainData/Polynomial/raw/{surface}/gt_distances_level{level}_source{vertex}.npy
```

Use `--force_recompute` to recalculate.

## Example Workflow

Complete example for a 5000-Gaussian dataset:

```bash
# 1. Process in 5 parts (1000 Gaussians each)
for part in {0..4}; do
    python calculate_gaussian_geodesic_distances.py \
        --gaussian_output output/saddle_exp \
        --data_root TrainData/Polynomial/raw \
        --surface Saddle \
        --num_sources 1000 \
        --part $part
done

# 2. Verify all parts created
ls output/saddle_exp/point_cloud/iteration_30000/geodesic/
# Should see: part0.npy, part1.npy, part2.npy, part3.npy, part4.npy

# 3. Unite parts
python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/saddle_exp \
    --surface Saddle \
    --unite

# 4. Check result
python -c "import numpy as np; d = np.load('output/saddle_exp/point_cloud/iteration_30000/geodesic/gt_distances_united.npy'); print(f'Shape: {d.shape}, Min: {d.min():.6f}, Max: {d.max():.6f}')"
```

## Troubleshooting

**Error: "Part N is out of range"**
- You specified a part number beyond what's needed
- Calculate: `total_parts = (total_gaussians + num_sources - 1) // num_sources`

**Error: "No part files found"**
- Run processing steps before --unite
- Ensure part files are in the geodesic/ directory

**Warning: "Mismatch in number of Gaussians"**
- Some parts may be missing
- Check all expected part files exist

## Advanced: SLURM Job Array

For HPC clusters with SLURM:

```bash
#!/bin/bash
#SBATCH --array=0-9        # 10 parts
#SBATCH --time=02:00:00
#SBATCH --mem=16G

python calculate_gaussian_geodesic_distances.py \
    --gaussian_output output/experiment \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --num_sources 1000 \
    --part $SLURM_ARRAY_TASK_ID
```
