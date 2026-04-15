# GenerateData

Synthetic data generation pipeline for the RaDe-GS project. This module creates COLMAP-compatible datasets from analytical surfaces (polynomial) and real meshes (TOSCA), renders multi-view images, and computes ground-truth geodesic distances between Gaussian splats.

---

## Directory Structure

```
GenerateData/
├── GenerateRawPolynomialMesh.py          # Generate meshes for polynomial surfaces
├── create_synthetic_colmap_dataset_from_mesh.py      # Render polynomial meshes → COLMAP dataset
├── create_synthetic_colmap_dataset_from_mesh_tosca.py # Render TOSCA meshes → COLMAP dataset
├── preprocess_tosca.py                   # Convert TOSCA .mat files → organized PLY files
├── create_ground_truth_mesh_dataset.py   # Create GT mesh dataset in TNT evaluation format
├── compute_gaussian_geodesic_distances.py       # Geodesic GT for polynomial Gaussians
├── compute_gaussian_geodesic_distances_tosca.py # Geodesic GT for TOSCA Gaussians
├── compute_geodesic_mesh_for_gaussians.py       # Build mesh with Gaussians as vertices
├── compare_geodesic.py                   # Compare geodesic results against ground truth
├── verify_geodesic_distances.py          # Sanity-check geodesic distance files
├── fix_geodesic_row_order.py             # Fix row-ordering bug in gt_geodesic.npz
├── convert_to_logfile.py                 # Convert COLMAP SfM to TanksAndTemples log format
├── scripts/                              # Shell scripts for batch rendering/processing
├── utils/                                # Shared utilities (camera, rendering, I/O, etc.)
├── textures/                             # Texture images and generation scripts
├── tests/                                # Unit tests
├── datasets/                             # Generated dataset outputs (gaussian_patches/)
├── brara/                                # Legacy / experimental scripts
└── TrainData/                            # Symlink or storage for raw training data
```

---

## Pipeline Overview

The end-to-end workflow has three stages:

### 1. Mesh Generation & Preprocessing

| Script | Purpose |
|--------|---------|
| `GenerateRawPolynomialMesh.py` | Generate triangulated meshes at multiple resolutions for three polynomial surfaces: **Paraboloid** ($z = x^2 + y^2$), **Saddle** ($z = x^2 - y^2$), and **Hyperbolic Paraboloid** ($z = x^2 - y^2 + xy$). |
| `preprocess_tosca.py` | Convert TOSCA dataset from MATLAB `.mat` format into organized PLY files with high/low resolution meshes, point clouds, and normals per shape. |

### 2. Synthetic COLMAP Dataset Rendering

These scripts render multi-view images from virtual cameras placed on an orbit around each mesh, producing a full COLMAP-compatible dataset that can be fed directly to `train.py -s <dataset>`.

| Script | Input | Description |
|--------|-------|-------------|
| `create_synthetic_colmap_dataset_from_mesh.py` | Polynomial meshes | Uses a **dual-level mesh system**: one resolution for rendering images, another (higher) for generating the COLMAP sparse point cloud. Supports configurable textures, lighting presets, and camera parameters. Supports `--auto_camera_radius` for automatic camera placement. |
| `create_synthetic_colmap_dataset_from_mesh_tosca.py` | TOSCA meshes | Same concept but for TOSCA shapes. Reads preprocessed PLY files and supports `high_res` / `low_res` modes. Supports `--auto_camera_radius` for automatic camera placement. |

**Output structure** (per surface/shape):
```
<output_root>/
├── images/                    # Rendered JPG images
└── sparse/
    └── 0/
        ├── cameras.{txt,bin}  # Camera intrinsics (PINHOLE model)
        ├── images.{txt,bin}   # Camera extrinsics (poses)
        └── points3D.{txt,bin,ply}  # Sparse 3D point cloud
```

### 3. Ground-Truth Geodesic Distance Computation

After training Gaussian splats on the synthetic datasets, these scripts compute ground-truth geodesic distances between Gaussian centers using the known mesh geometry.

| Script | Domain | Method |
|--------|--------|--------|
| `compute_gaussian_geodesic_distances.py` | Polynomial surfaces | Projects Gaussians onto the nearest face of the GT mesh, then uses **barycentric interpolation** of face-vertex geodesic distances (MMP/VTP algorithm) for O(h²) accuracy. Uses a high-level pipeline (`compute_and_save_geodesic_pipeline`) that splits sources into batches, computes geodesics, transfers to Gaussians, and saves partial results — all in one worker per batch. |
| `compute_gaussian_geodesic_distances_tosca.py` | TOSCA shapes | Same approach but for TOSCA meshes. Supports **Gaussian embedding** (`--embed_gaussians`): each Gaussian is projected onto the mesh surface and inserted as a new vertex, eliminating barycentric interpolation error. Uses **normal-guided projection** by default: instead of orthogonal closest-point projection, each Gaussian is ray-cast along the interpolated vertex normal at the initial projection point, giving more geometrically faithful placement on curved surfaces (falls back to orthogonal projection when the ray misses the triangle). Source vertices are selected via **farthest-point sampling** on embedded Gaussian mesh vertices (when embedding is enabled) or proximity-filtered mesh vertices (legacy mode). Supports both GT and reconstructed meshes. Partial results are merged with automatic deduplication of overlapping source ranges. |
| `compute_geodesic_mesh_for_gaussians.py` | Polynomial surfaces | Alternative approach: inserts Gaussian positions **directly as mesh vertices** (zero transfer error) by projecting (x,y) onto the analytical surface. Supports two insertion modes: global Delaunay (default) and **local per-triangle refinement** (`--local_refinement`), which preserves the grid topology and sub-triangulates only within each grid face containing Gaussians. |

**Output structure:**
```
<gaussian_output>/
└── geodesic_distance/
    ├── gt_partial/
    │   └── sources_batch_{index}_{start}_{end}.npz  # Partial results (per-batch)
    └── gt_geodesic.npz                        # Complete merged ground truth
```

Each `.npz` file contains:
- `gaussian_positions` — (N_gaussians, 3) centers
- `source_indices` — (N_sources,) mesh vertex indices
- `source_positions` — (N_sources, 3) source coordinates
- `geodesic_distances` — (N_sources, N_gaussians) pairwise geodesic distances
- `closest_mesh_indices` — (N_gaussians,) nearest mesh vertex per Gaussian
- `source_gaussian_indices` — (N_sources,) Gaussian index closest to each source

---

## Utility Modules (`utils/`)

| Module | Purpose |
|--------|---------|
| `camera_utils.py` | Camera data classes (`CameraSample`, `CameraIntrinsics`), quaternion ↔ rotation conversions, Fibonacci sphere sampling for uniform view placement, **automatic camera radius** computation from mesh bounding box and FOV. |
| `rendering_utils.py` | Open3D offline rendering: point sampling, orbit camera placement, image rendering with configurable lighting. |
| `lighting.py` | Lighting presets for rendering — standard, 5 light-ID presets, and 10 decoupled appearance groups for appearance diversity training. |
| `io_utils.py` | COLMAP binary/text I/O (cameras, images, points3D), texture loading, dataset validation. |
| `load_utils.py` | Load Gaussian splat PLY files, find available training iterations. |
| `geodesic_mesh_utils.py` | Project Gaussians onto polynomial surfaces, Poisson-disk surface sampling, Delaunay triangulation in (u,v) parameter space. Supports **local per-triangle refinement** (`local_refinement=True`): Gaussians are embedded into the grid mesh by sub-triangulating only the grid faces that contain them, preserving the original grid structure. |
| `data_generation_utils.py` | KNN computation (Euclidean and Mahalanobis), neighborhood ring extraction. Supports knn_cuda, simple-knn, and CPU fallbacks. Includes **adaptive kNN** (`adaptive_ring1_neighbors`) for per-point k adjustment so that a target ring-k count is reached. |
| `compute_gaussian_geodesic_distances_helper.py` | Core helpers for geodesic computation: source mesh generation, barycentric interpolation, partial result merging with automatic deduplication, multiprocessing geodesic solvers. |

---

## Validation & Debugging Tools

| Script | Purpose |
|--------|---------|
| `verify_geodesic_distances.py` | Run sanity checks on geodesic results: non-negativity, self-distance bounds, triangle inequality, Euclidean lower-bound, symmetry, NaN/Inf detection. Outputs a JSON report. |
| `compare_geodesic.py` | Compare geodesic distance files against a ground truth reference. Reports per-source and aggregate error statistics with optional matplotlib plots. |
| `fix_geodesic_row_order.py` | Fix a historical row-ordering bug where `transfer_geodesic_to_gaussians()` sorted rows inconsistently with metadata. Re-merges partial results into a corrected `gt_geodesic.npz`. |
| `create_ground_truth_mesh_dataset.py` | Export GT meshes in TanksAndTemples evaluation format (`.ply`, `.json` bounding volume, `_COLMAP_SfM.log`, `_trans.txt`). |
| `convert_to_logfile.py` | Convert COLMAP SfM reconstruction to TanksAndTemples `.log` camera trajectory format. |

---

## Textures (`textures/`)

Pre-made texture images for rendering, plus a generator script:

- `blue.png`, `pink.jpg`, `stone.jpg`, `colors.png`, `letters.jpg` — Solid and natural textures
- `checkerboard.png`, `checkerboard_blue.png`, `checkerboard_multi.png` — High-frequency patterns for better Gaussian densification
- `generate_highfreq_textures.py` — Script to create custom checkerboard textures with configurable cell size and colors

---

## Shell Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `render_all_surfaces.sh` | Render all three polynomial surfaces with default settings |
| `render_paraboloid.sh` / `render_saddle.sh` / `render_hyperbolic_paraboloid.sh` | Render individual surfaces |
| `run_render_with_gpu.sh` | SLURM wrapper to allocate a GPU node and run rendering |
| `create_patches.sh` | Generate Gaussian training patches from trained reconstructions |

---

## Quick Start

```bash
# 1. Generate polynomial meshes at multiple resolutions
python GenerateData/GenerateRawPolynomialMesh.py --output_dir TrainData/Polynomial/raw

# 2. Render synthetic COLMAP dataset (e.g., Paraboloid with blue texture)
python GenerateData/create_synthetic_colmap_dataset_from_mesh.py \
    --surface Paraboloid \
    --colmap_level 2 \
    --image_mesh_level 1 \
    --num_views 150 \
    --texture_name blue

# 3. Train Gaussian splats (from project root)
python train.py -s TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_02

# 4. Compute ground-truth geodesic distances
python GenerateData/compute_gaussian_geodesic_distances.py \
    --gaussian_output <output_path> \
    --data_root TrainData/Polynomial/raw \
    --surface Paraboloid \
    --source_mesh_resolution 20
```

### TOSCA Workflow

```bash
# 1. Preprocess TOSCA dataset from MATLAB format
python GenerateData/preprocess_tosca.py --input_dir <tosca_matlab_dir> --output_dir TrainData/TOSCA/processed

# 2. Render synthetic COLMAP dataset for a TOSCA shape
python GenerateData/create_synthetic_colmap_dataset_from_mesh_tosca.py \
    --shape cat0 \
    --texture_name blue \
    --num_views 150

# 3. Train Gaussian splats, then compute geodesic distances
python GenerateData/compute_gaussian_geodesic_distances_tosca.py \
    --gaussian_output <output_path> \
    --shape cat0 \
    --mesh_type gt \
    --num_sources 64 \
    --embed_gaussians \
    --geodesic_method mmp
```

---

## Tests

Unit tests are in `tests/`. Run them with:

```bash
cd GenerateData
bash tests/run_tests.sh
```

Key test files:
- `test_rendering_utils.py` — Point sampling, UV generation, lighting configuration, brightness clamping, **auto camera radius** computation
- `test_geodesic_mesh_utils.py` — Surface projection and mesh construction
- `test_data_generation_utils.py` — KNN, ring neighborhoods, adaptive kNN, point-to-surface mapping
- `test_data_generation_utils.py` — KNN and neighborhood computations
- `test_load_utils.py` — PLY loading and iteration discovery
- `test_embed_gaussians.py` — Gaussian-into-mesh embedding, snapping, face splitting, normal-guided projection
- `test_tosca_geodesic.py` — TOSCA geodesic pipeline

---

## Legacy (`brara/`)

Older or experimental scripts kept for reference:
- `create_gaussian_splats.py` — Early Gaussian initialization from polynomial meshes
- `create_gt_by_mmp.py` — GT geodesic computation using PyVista + MMP
- `calculate_gaussian_geodesic_distances.py` — Prior version of the geodesic pipeline with partitioned batch processing
- `debug_colmap_dataset.py` — Print COLMAP dataset summary statistics
- `visualize_colmap_dataset.py` — Open3D visualization of cameras and 3D points
- `generate_data_geodesic_utils.py` — Earlier utility functions for geodesic computation
