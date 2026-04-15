# RaDe-GS Scripts

Utility scripts for rendering synthetic datasets, training Gaussian splatting models,
computing geodesic distances, and generating training patches.

## Directory Structure

```
scripts/
├── README.md                  # This file
├── find_all_nodes.sh          # SLURM node discovery
├── rotem_scripts/             # Personal helper scripts
│
├── polynomial/                # Polynomial surface pipeline
│   ├── run_adaptive_analysis.sh            # Adaptive analysis utility
│   ├── render/                # Rendering & raw mesh generation
│   │   ├── render_all_surfaces.sh              # Render all polynomial surfaces
│   │   ├── render_paraboloid.sh                # Render single surface
│   │   ├── render_saddle.sh
│   │   ├── render_hyperbolic_paraboloid.sh
│   │   └── generate_raw_polynomial_mesh.sh     # Generate raw meshes at multiple levels
│   ├── train_gaussians/       # Gaussian splatting training
│   │   └── train_polynomial_all.sh             # Batch-train Gaussian splatting
│   ├── geodesic/              # Geodesic mesh building & distance computation
│   │   ├── build_geodesic_mesh.sh              # Build mesh for single surface
│   │   ├── build_geodesic_mesh_polynomial_all.sh  # Build mesh for all surfaces
│   │   ├── build_geodesic_mesh_tmux.sh         # tmux launcher for mesh building
│   │   ├── compute_geodesic_batched.sh         # Batched geodesic for single output
│   │   ├── compute_geodesic_polynomial_all.sh  # Geodesic for all polynomial outputs
│   │   └── compute_geodesic_polynomial_all_tmux.sh  # tmux launcher for geodesic
│   ├── patches/               # Training patch generation
│   │   ├── generate_polynomial_training_patches.sh         # Patches (xyz)
│   │   ├── generate_polynomial_training_patches_one_source.sh  # Patches (one source)
│   │   ├── generate_polynomial_training_patches_scale_opacity.sh  # Patches (xyz+scale+opacity)
│   │   ├── generate_polynomial_training_patches_scale_opacity_one_source.sh
│   │   ├── generate_patches_tmux.sh            # tmux launcher (xyz)
│   │   ├── generate_patches_one_source_tmux.sh # tmux launcher (one source)
│   │   ├── generate_patches_scale_opacity_tmux.sh  # tmux launcher (scale+opacity)
│   │   └── generate_patches_scale_opacity_one_source_tmux.sh
│   ├── train_model/           # GaussianPatchTransformer model training
│   │   ├── train_combined_polynomial.sh        # Train combined model (xyz, ring 3)
│   │   ├── train_combined_polynomial_one_source.sh
│   │   ├── train_combined_polynomial_one_source_test.sh
│   │   ├── train_combined_polynomial_test.sh
│   │   ├── train_combined_polynomial_scale_opacity.sh
│   │   └── train_combined_polynomial_scale_opacity_one_source.sh
│   ├── evaluate_scripts/      # Evaluation scripts (one_source config)
│   │   ├── evaluate_geodesic.sh
│   │   ├── evaluate_model_vs_gt.sh
│   │   ├── analyze_nn_stats.sh
│   │   ├── run_all_evaluations.sh
│   │   ├── run_eval_fps.sh
│   │   ├── run_eval_propagation.sh
│   │   └── sweep_knn.sh               # kNN sweep across k values
│   └── new_dataset/           # New dataset pipeline (one-source, updated KNN)
│       ├── patches/           # Training patch generation
│       │   ├── generate_patches_new_dataset_tmux.sh
│       │   └── generate_polynomial_training_patches_new_dataset.sh
│       ├── train/             # GaussianPatchTransformer model training
│       │   ├── train_combined_polynomial_new_dataset.sh       # Ring 3
│       │   ├── train_combined_polynomial_ring2.sh             # Ring 2
│       │   ├── train_combined_polynomial_ring2_conv.sh        # Ring 2, Conv encoder
│       │   ├── train_combined_polynomial_ring2_spatial_maxmean.sh
│       │   └── train_combined_polynomial_ring3_scale_opacity.sh
│       ├── eval_data/         # Level 05 evaluation Gaussian data
│       │   ├── render_eval_surfaces.sh
│       │   ├── train_eval_gaussians.sh
│       │   ├── build_geodesic_mesh_eval.sh
│       │   └── compute_geodesic_eval.sh
│       └── evaluate_scripts/  # Model evaluation (new_dataset config)
│           ├── analyze_nn_stats.sh
│           ├── evaluate_geodesic.sh
│           ├── evaluate_model_vs_gt.sh
│           ├── run_all_evaluations.sh
│           ├── run_eval_fps.sh
│           ├── run_eval_propagation.sh
│           └── sweep_knn.sh
│
└── tosca/                     # TOSCA mesh pipeline
    ├── tosca_animal_map.sh                 # Shared animal→shape index mapping
    ├── run_adaptive_analysis.sh            # Adaptive analysis utility
    ├── analyze_outliers.py                 # Outlier analysis script
    ├── centaur0_bad_points.png             # Visualization
    ├── centaur0_sources.png                # Visualization
    ├── render/                # Rendering
    │   ├── render_all_tosca.sh             # Render all shapes (generic, multi-texture)
    │   ├── render_all_blue.sh              # Render all shapes (blue_texture defaults)
    │   ├── render_tosca_cat0.sh            # Render single shape (quick test)
    │   └── render_tmux.sh                  # tmux launcher for rendering
    ├── train_gaussians/       # Gaussian splatting training & mesh extraction
    │   ├── train_tosca_all.sh              # Batch-train (generic, multi-texture)
    │   ├── train_all_blue.sh               # Train all (blue_texture defaults)
    │   ├── train_and_extract_mesh.sh       # Train + mesh extract single shape
    │   ├── train_tmux.sh                   # tmux launcher for training (GPU)
    │   ├── train_tosca_smart.sh            # Smart batch trainer (animals, auto-detect)
    │   ├── train_tosca_gaussian_all.sh     # Batch-train (shapes/textures/resolutions)
    │   └── mesh_extract_tosca_all.sh       # Extract mesh for all trained outputs
    ├── geodesic/              # Geodesic distance computation
    │   ├── compute_geodesic_tosca_all.sh   # Geodesic for all shapes (generic)
    │   ├── compute_geodesic_all_blue.sh    # Geodesic for all (blue_texture defaults)
    │   ├── compute_geodesic_tosca_batched.sh  # Batched geodesic for single shape
    │   ├── compute_geodesic_tosca_tmux.sh  # tmux launcher for single shape
    │   └── geodesic_all_tmux.sh            # tmux launcher for all shapes
    ├── patches/               # Training patch generation
    │   ├── generate_tosca_training_patches.sh  # Auto-detect shapes + generate patches
    │   ├── generate_training_patches.sh    # Use pre-created configs
    │   └── generate_patches_tmux.sh        # tmux launcher for patch generation
    ├── train_model/           # GaussianPatchTransformer model training
    │   ├── train_combined_tosca.sh         # Train combined model (ring 3)
    │   └── train_combined_tosca_test.sh    # Train combined model (testing config)
    ├── evaluate_scripts/      # Model evaluation (combined_tosca_ring3 config)
    │   ├── evaluate_geodesic.sh
    │   ├── evaluate_model_vs_gt.sh
    │   ├── analyze_nn_stats.sh
    │   ├── run_eval_fps.sh
    │   ├── run_eval_propagation.sh
    │   ├── run_all_evaluations.sh
    │   └── evaluate_all_test_shapes.sh  # Parallel eval on test shapes
    ├── evaluate/              # Geodesic evaluation (legacy)
    │   ├── evaluate_geodesic_single.sh     # Evaluate single test shape (GPU)
    │   ├── evaluate_geodesic_all.sh        # Evaluate all test shapes (GPU)
    │   ├── evaluate_geodesic_single_tmux.sh  # tmux launcher for single-shape eval
    │   └── evaluate_geodesic_all_tmux.sh   # tmux launcher for all-shapes eval
    ├── benchmark/             # Parameter benchmarking
    │   ├── benchmark_params.sh             # Sweep training params, evaluate mesh quality
    │   ├── benchmark_tmux.sh               # tmux+SLURM launcher for benchmark
    │   └── evaluate_benchmark.py           # Per-run evaluation (mesh quality, Gaussian dist, CC)
    └── pipeline/              # Full pipeline orchestration
        └── full_pipeline_tmux.sh           # Render→train→geodesic→patches
```

---

## Quick Start

### Polynomial Pipeline

```bash
# 1. Render synthetic COLMAP datasets
bash scripts/polynomial/render/render_all_surfaces.sh --textures blue --colmap_levels 2

# 2. Train Gaussian splatting
bash scripts/polynomial/train_gaussians/train_polynomial_all.sh --extract_mesh

# 3. Compute geodesic distances
bash scripts/polynomial/geodesic/compute_geodesic_polynomial_all.sh

# 4. Generate training patches
bash scripts/polynomial/patches/generate_polynomial_training_patches.sh

# Or use tmux launchers for cluster nodes:
bash scripts/polynomial/patches/generate_patches_tmux.sh
```

### TOSCA Pipeline (blue_texture, decoupled_appearance)

```bash
# 1. Render all TOSCA shapes
bash scripts/tosca/render/render_all_blue.sh

# 2. Train + extract mesh
bash scripts/tosca/train_gaussians/train_all_blue.sh

# 3. Compute geodesic distances
bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh

# 4. Generate training patches
bash scripts/tosca/patches/generate_training_patches.sh
```

### TOSCA Full Pipeline (tmux, single command)

```bash
# Run all 4 stages in one tmux session on a GPU node
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --node gipdeep7

# Skip render + train (already done), just geodesic + patches
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --skip_render --skip_train

# Specific shapes only
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --shapes "cat0,cat1,dog0"

# Attach to monitor progress
tmux attach -t tosca_pipeline
```

### TOSCA Parameter Benchmark

Sweep training hyperparameters (`min_opacity_prune`, `lambda_multi_view_geo`,
`lambda_distortion`, `densify_grad_threshold`, `big_point_scale_factor`) on a
single shape. Each combination is trained, mesh-extracted, and evaluated against
the ground-truth mesh. Results are saved as per-run JSON reports and a summary CSV.
Optionally logs to Weights & Biases.

```bash
# Benchmark cat0 with default parameter grid
bash scripts/tosca/benchmark/benchmark_params.sh --shape cat0

# Dry run – see all parameter combinations without running
bash scripts/tosca/benchmark/benchmark_params.sh --dry_run

# Run via tmux on a GPU node
bash scripts/tosca/benchmark/benchmark_tmux.sh --node gipdeep7 --shape cat0

# Skip wandb, resume partial benchmark
bash scripts/tosca/benchmark/benchmark_tmux.sh --no_wandb --skip_existing

# Attach to monitor
tmux attach -t tosca_benchmark
```

**Metrics collected per run:**
- Chamfer distance (reconstructed mesh ↔ GT mesh)
- Gaussian → GT mesh surface distance (mean, median, p95, p99)
- Gaussian → closest reconstructed mesh vertex distance
- Number of Gaussians
- Number of connected components in reconstructed mesh
- Accuracy (recon→GT) and completeness (GT→recon)

**Output structure:**
```
output/benchmarks/tosca_params/{shape}_{texture}_{resolution}/
├── benchmark_summary.csv           # Summary table of all runs
├── mop_0.005/                      # min_opacity_prune=0.005
│   ├── benchmark_report.json       # Full evaluation report
│   ├── benchmark_args.txt          # Training args used
│   ├── train.log                   # Training log
│   ├── mesh_extract.log            # Mesh extraction log
│   ├── recon.ply                   # Reconstructed mesh
│   └── point_cloud/                # Gaussian point clouds
├── lmvg_0.1/                       # lambda_multi_view_geo=0.1
│   └── ...
└── ...
```

---

## TOSCA Scripts Detail

### Rendering (`tosca/render/`)

| Script | Purpose |
|--------|---------|
| `render_all_tosca.sh` | Generic renderer (any texture/resolution/light_id combination) |
| `render_all_blue.sh` | Convenience wrapper: blue_texture, high_res, decoupled_appearance |
| `render_tosca_cat0.sh` | Quick single-shape test |
| `render_tmux.sh` | tmux+SLURM launcher for rendering on a compute node |

```bash
# Render specific shapes with blue texture
bash scripts/tosca/render/render_all_blue.sh --shapes "cat0,cat1,dog0"

# Render everything via tmux on gipdeep9
bash scripts/tosca/render/render_tmux.sh --node gipdeep9 --cpus 20
```

### Training (`tosca/train_gaussians/`)

| Script | Purpose |
|--------|---------|
| `train_tosca_all.sh` | Generic batch trainer (any texture/resolution/light_id) |
| `train_all_blue.sh` | Convenience: blue_texture, decoupled_appearance, with mesh extraction |
| `train_and_extract_mesh.sh` | Train + mesh extract for a **single** shape |
| `train_tmux.sh` | tmux+SLURM launcher for training on a GPU node |
| `train_tosca_smart.sh` | Smart batch trainer (animals, decoupled_appearance, GPU node) |
| `train_tosca_gaussian_all.sh` | Batch-train Gaussian splatting (shapes/textures/resolutions + optional mesh) |
| `mesh_extract_tosca_all.sh` | Extract mesh for all trained TOSCA Gaussian outputs |
| `evaluate_gaussian_mesh_quality_single.sh` | Evaluate Gaussian-to-mesh alignment for a single shape (JSON report) |
| `evaluate_gaussian_mesh_quality_all.sh` | Evaluate all TOSCA shapes and print aggregate summary |
| `train_combined_tosca.sh` | Train combined transformer model on TOSCA data (ring 3, gipdeep10) |
| `train_combined_tosca_test.sh` | Train combined transformer model – testing config (ring 3, gipdeep6) |
```bash
# Train single shape
bash scripts/tosca/train_gaussians/train_and_extract_mesh.sh cat0

# Train all shapes, skip already-done
bash scripts/tosca/train_gaussians/train_all_blue.sh --skip_existing

# Train via tmux on gipdeep7 (GPU)
bash scripts/tosca/train_gaussians/train_tmux.sh --node gipdeep7 --shapes "cat0,cat1"

# Batch-train all 9 high-res animals (blue texture, high_res, with mesh extraction)
bash scripts/tosca/train_gaussians/train_tosca_gaussian_all.sh --skip_existing

# Train specific animals only
bash scripts/tosca/train_gaussians/train_tosca_gaussian_all.sh --animals "cat,dog,horse"

# Train specific shapes, both resolutions, in parallel
bash scripts/tosca/train_gaussians/train_tosca_gaussian_all.sh --shapes "cat0,cat2,gorilla5" --resolutions "high_res,low_res" --max_parallel 4

# Extract mesh for all already-trained outputs
bash scripts/tosca/train_gaussians/mesh_extract_tosca_all.sh

# Evaluate Gaussian-mesh quality for a single shape
bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_single.sh cat0

# Evaluate all TOSCA shapes
bash scripts/tosca/train_gaussians/evaluate_gaussian_mesh_quality_all.sh
```

### Combined Model Training (`tosca/train_model/`)

| Script | Purpose |
|--------|---------|
| `train_combined_tosca.sh` | Train transformer on combined TOSCA dataset (ring 3, gipdeep10) |
| `train_combined_tosca_test.sh` | Same with testing config on gipdeep6 |

```bash
# Default: ring 3, combined_tosca_ring3.yaml, gipdeep10
bash scripts/tosca/train_model/train_combined_tosca.sh

# Ring 2 config
bash scripts/tosca/train_model/train_combined_tosca.sh --ring 2 --config models/configs/tosca/combined_tosca_ring2.yaml

# Quick test run on gipdeep6
bash scripts/tosca/train_model/train_combined_tosca_test.sh

# Override hyperparameters on the fly
bash scripts/tosca/train_model/train_combined_tosca_test.sh --extra "--batch_size 64 --no_wandb"

# Attach to monitor
tmux attach -t train_gpt_tosca
tmux attach -t train_gpt_tosca_testing
```

### Geodesic Distances (`tosca/geodesic/`)

| Script | Purpose |
|--------|---------|
| `compute_geodesic_tosca_all.sh` | Generic batch geodesic computation |
| `compute_geodesic_tosca_batched.sh` | Batched geodesic for single shape |
| `compute_geodesic_tosca_tmux.sh` | tmux launcher for single shape |
| `compute_geodesic_all_blue.sh` | Convenience: blue_texture, decoupled_appearance |
| `geodesic_all_tmux.sh` | tmux launcher for all shapes at once |

```bash
# Compute geodesic for all blue_texture shapes
bash scripts/tosca/geodesic/compute_geodesic_all_blue.sh

# Via tmux on gipdeep9
bash scripts/tosca/geodesic/geodesic_all_tmux.sh --node gipdeep9 --cpus 40
```

### Training Patches (`tosca/patches/`)

| Script | Purpose |
|--------|---------|
| `generate_tosca_training_patches.sh` | Auto-detect shapes, generate configs + patches |
| `generate_training_patches.sh` | Use pre-created configs from `DataSets/configs/tosca/` |
| `generate_patches_tmux.sh` | tmux launcher for patch generation |

```bash
# Generate patches for all animals using pre-created configs
bash scripts/tosca/patches/generate_training_patches.sh

# With scale+opacity attributes
bash scripts/tosca/patches/generate_training_patches.sh --scale_opacity

# Specific animals only
bash scripts/tosca/patches/generate_training_patches.sh --animals "cat,dog,horse"

# Via tmux on gipdeep12
bash scripts/tosca/patches/generate_patches_tmux.sh --node gipdeep12 --cpus 40
```

### Full Pipeline (`tosca/pipeline/`)

| Script | Purpose |
|--------|---------|
| `full_pipeline_tmux.sh` | Run all 4 stages sequentially in one tmux session |

```bash
# Everything from scratch
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --node gipdeep7

# Skip stages that are already done
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --skip_render --skip_train

# Dry run to see what would happen
bash scripts/tosca/pipeline/full_pipeline_tmux.sh --dry_run
```
### Model Evaluation (`tosca/evaluate_scripts/`)

Evaluate the **combined_tosca_ring3** model using the same evaluation
scripts as the Polynomial pipeline. All scripts support `--gaussian_dir` to
point at any TOSCA shape.

| Script | Purpose |
|--------|---------|
| `evaluate_geodesic.sh` | End-to-end FM propagation evaluation |
| `evaluate_model_vs_gt.sh` | Model predictions vs GT distances |
| `analyze_nn_stats.sh` | k-NN distance statistics |
| `run_eval_fps.sh` | Evaluation with FPS downsampling |
| `run_eval_propagation.sh` | Debug FM propagation (first N visited) |
| `run_all_evaluations.sh` | Master script launching all 5 above |
| `evaluate_all_test_shapes.sh` | Parallel evaluation on test shapes (gorilla8, horse10, michael16) |

```bash
# Single shape (default: cat2)
bash scripts/tosca/evaluate_scripts/evaluate_model_vs_gt.sh

# Different shape
bash scripts/tosca/evaluate_scripts/evaluate_model_vs_gt.sh \
    --gaussian_dir TrainData/TOSCA/SyntheticColmapData/blue_texture/gorilla8/high_res/decoupled_appearance/output

# All test shapes in parallel
bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh

# Dry run
bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh --dry_run
```

### Geodesic Evaluation (`tosca/evaluate/`)

Evaluate trained geodesic prediction models on TOSCA test shapes.

**Test set** = last pose per animal (12 shapes):
`cat10, centaur5, david14, dog10, gorilla20, horse18, lioness16, michael19, seahorse5, shark0, victoria25, wolf2`

| Script | Purpose |
|--------|--------|
| `evaluate_geodesic_single.sh` | Evaluate on a **single** test shape (requires GPU) |
| `evaluate_geodesic_all.sh` | Evaluate on **all** test shapes sequentially, aggregate metrics |
| `evaluate_geodesic_single_tmux.sh` | tmux+SLURM launcher for single-shape evaluation |
| `evaluate_geodesic_all_tmux.sh` | tmux+SLURM launcher for all-shapes evaluation |

```bash
# Single shape evaluation
bash scripts/tosca/evaluate/evaluate_geodesic_single.sh \
    --model_path checkpoints/my_model/best_model.pth \
    --shape cat10

# All test shapes
bash scripts/tosca/evaluate/evaluate_geodesic_all.sh \
    --model_path checkpoints/my_model/best_model.pth

# Subset of test shapes
bash scripts/tosca/evaluate/evaluate_geodesic_all.sh \
    --model_path checkpoints/my_model/best_model.pth \
    --shapes "cat10,dog10,horse18"

# Via tmux on GPU node
bash scripts/tosca/evaluate/evaluate_geodesic_all_tmux.sh \
    --model_path checkpoints/my_model/best_model.pth \
    --node gipdeep7

# Dry run
bash scripts/tosca/evaluate/evaluate_geodesic_all.sh --model_path <path> --dry_run
```

Results are saved to `geodesic_eval_results/tosca/<shape>/` with:
- Evaluation metrics JSON
- PLY visualization of predicted geodesic distances
- Aggregate metrics across all shapes in `geodesic_eval_results/tosca/aggregate_metrics.json`
---

## Polynomial Scripts Detail

### Rendering (`polynomial/render/`)

| Script | Purpose |
|--------|---------|
| `render_all_surfaces.sh` | Render all polynomial surfaces (parallel, multi-texture) |
| `render_paraboloid.sh` | Render single paraboloid |
| `render_saddle.sh` | Render single saddle |
| `render_hyperbolic_paraboloid.sh` | Render single hyperbolic paraboloid |
| `generate_raw_polynomial_mesh.sh` | Generate raw meshes at multiple resolution levels |

### Training (`polynomial/train_gaussians/`)

| Script | Purpose |
|--------|---------|
| `train_polynomial_all.sh` | Batch-train Gaussian splatting on all polynomial datasets |

### Combined Model Training (`polynomial/train_model/`)

| Script | Purpose |
|--------|---------|
| `train_combined_polynomial.sh` | Train combined transformer model (xyz, ring 3) |
| `train_combined_polynomial_one_source.sh` | Train combined model (one source) |
| `train_combined_polynomial_one_source_test.sh` | Testing config (one source) |
| `train_combined_polynomial_test.sh` | Testing config |
| `train_combined_polynomial_scale_opacity.sh` | Train combined model (xyz+scale+opacity) |
| `train_combined_polynomial_scale_opacity_one_source.sh` | Scale+opacity (one source) |

### Geodesic & Mesh (`polynomial/geodesic/`)

| Script | Purpose |
|--------|---------|
| `compute_geodesic_batched.sh` | Batched geodesic for single output |
| `compute_geodesic_polynomial_all.sh` | Geodesic for all polynomial outputs |
| `build_geodesic_mesh.sh` | Build geodesic mesh for single surface |
| `build_geodesic_mesh_polynomial_all.sh` | Build geodesic mesh for all surfaces |
| `build_geodesic_mesh_tmux.sh` | tmux launcher |
| `validate_geodesic_single.sh` | Validate geodesic data for a single polynomial output (source projection, triangle ineq, etc.) |
| `validate_geodesic_polynomial_all.sh` | Validate all polynomial outputs and print aggregate pass/fail summary |

```bash
# Validate a single output (checks projection, triangle inequality, symmetry, etc.)
bash scripts/polynomial/geodesic/validate_geodesic_single.sh \
    TrainData/Polynomial/SyntheticColmapData/blue_texture/Paraboloid/level_04/light_0/output \
    Paraboloid --verbose

# Validate all polynomial outputs
bash scripts/polynomial/geodesic/validate_geodesic_polynomial_all.sh

# Validate only specific surfaces/levels
bash scripts/polynomial/geodesic/validate_geodesic_polynomial_all.sh \
    --surfaces Paraboloid,Saddle --levels 04 --verbose
```

### Training Patches (`polynomial/patches/`)

| Script | Purpose |
|--------|---------|
| `generate_polynomial_training_patches.sh` | Generate patches (xyz only) |
| `generate_polynomial_training_patches_one_source.sh` | Generate patches (one source) |
| `generate_polynomial_training_patches_scale_opacity.sh` | Generate patches (xyz+scale+opacity) |
| `generate_polynomial_training_patches_scale_opacity_one_source.sh` | Scale+opacity (one source) |
| `generate_patches_tmux.sh` | tmux launcher (xyz) |
| `generate_patches_one_source_tmux.sh` | tmux launcher (one source) |
| `generate_patches_scale_opacity_tmux.sh` | tmux launcher (scale+opacity) |
| `generate_patches_scale_opacity_one_source_tmux.sh` | tmux launcher (scale+opacity, one source) |

### New Dataset Pipeline (`polynomial/new_dataset/`)

Scripts for the new dataset approach (one geodesic source, updated KNN params),
organized into subfolders.

#### Patch Generation (`new_dataset/patches/`)

| Script | Purpose |
|--------|---------|
| `generate_patches_new_dataset_tmux.sh` | tmux+SLURM launcher for patch generation |
| `generate_polynomial_training_patches_new_dataset.sh` | Generate training patches (all 3 surfaces) |

```bash
# Generate patches for all polynomial surfaces
bash scripts/polynomial/new_dataset/patches/generate_polynomial_training_patches_new_dataset.sh

# Via tmux on a cluster node
bash scripts/polynomial/new_dataset/patches/generate_patches_new_dataset_tmux.sh --node gipdeep10 --cpus 70
```

#### Model Training (`new_dataset/train/`)

| Script | Purpose |
|--------|---------|
| `train_combined_polynomial_new_dataset.sh` | Train GaussianPatchTransformer – Ring 3 |
| `train_combined_polynomial_ring2.sh` | Ring 2 |
| `train_combined_polynomial_ring2_conv.sh` | Ring 2, Conv encoder |
| `train_combined_polynomial_ring2_spatial_maxmean.sh` | Ring 2, spatial+max_mean aggregation |
| `train_combined_polynomial_ring3_scale_opacity.sh` | Ring 3 with scale+opacity attributes |

```bash
# Train ring 3 model (default)
bash scripts/polynomial/new_dataset/train/train_combined_polynomial_new_dataset.sh

# Train ring 2 variant
bash scripts/polynomial/new_dataset/train/train_combined_polynomial_ring2.sh

# Override hyperparameters
bash scripts/polynomial/new_dataset/train/train_combined_polynomial_new_dataset.sh --extra "--batch_size 128 --no_wandb"
```

#### Evaluation Data – Level 05 (`new_dataset/eval_data/`)

Creates Gaussian splatting data at **level 05** (a different mesh resolution than levels 02–04
used for training) so the geodesic prediction model can be evaluated on unseen Gaussians.

**Pipeline order:** render → train Gaussians → build geodesic mesh → compute geodesic

| Script | Purpose |
|--------|---------|
| `render_eval_surfaces.sh` | Render level 05 polynomial surfaces (COLMAP-style datasets) |
| `train_eval_gaussians.sh` | Train Gaussian splatting on level 05 data |
| `build_geodesic_mesh_eval.sh` | Build geodesic meshes for level 05 Gaussian outputs |
| `compute_geodesic_eval.sh` | Compute ground-truth geodesic distances for level 05 |

```bash
# 1. Render level 05 evaluation data
bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh

# 2. Train Gaussian splatting on the rendered data
bash scripts/polynomial/new_dataset/eval_data/train_eval_gaussians.sh

# 3. Build geodesic meshes
bash scripts/polynomial/new_dataset/eval_data/build_geodesic_mesh_eval.sh

# 4. Compute geodesic distances
bash scripts/polynomial/new_dataset/eval_data/compute_geodesic_eval.sh

# Dry run any step to preview
bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh --dry_run
bash scripts/polynomial/new_dataset/eval_data/train_eval_gaussians.sh --dry_run

# Process only specific surfaces
bash scripts/polynomial/new_dataset/eval_data/render_eval_surfaces.sh --surfaces Paraboloid
bash scripts/polynomial/new_dataset/eval_data/train_eval_gaussians.sh --surfaces Paraboloid --skip_existing
```

#### Model Evaluation (`new_dataset/evaluate_scripts/`)

Evaluate the **new_dataset** model (`checkpoints/new_dataset/combined_polynomial_ring3/best_model.pth`)
against ground-truth geodesic distances. Mirrors `polynomial/evaluate_scripts/` but uses new_dataset
config and checkpoint paths.

| Script | Purpose |
|--------|---------|
| `run_eval_propagation.sh` | Debug FM propagation – stop after N visited points |
| `sweep_knn.sh` | Sweep kNN k values to find optimal neighborhood size |
| `evaluate_geodesic.sh` | End-to-end geodesic propagation evaluation |
| `evaluate_model_vs_gt.sh` | Compare model predictions to GT distances |
| `analyze_nn_stats.sh` | Analyze k-NN distance statistics |
| `run_eval_fps.sh` | Evaluate with optional FPS downsampling |
| `run_all_evaluations.sh` | Master script – launches all 6 above |

```bash
# Quick propagation test (first 2000 points)
bash scripts/polynomial/new_dataset/evaluate_scripts/run_eval_propagation.sh

# kNN sweep to check neighborhood quality
bash scripts/polynomial/new_dataset/evaluate_scripts/sweep_knn.sh

# Full evaluation suite
bash scripts/polynomial/new_dataset/evaluate_scripts/run_all_evaluations.sh

# Dry run any script to preview
bash scripts/polynomial/new_dataset/evaluate_scripts/run_eval_propagation.sh --dry_run
bash scripts/polynomial/new_dataset/evaluate_scripts/run_all_evaluations.sh --dry_run

# Override defaults
bash scripts/polynomial/new_dataset/evaluate_scripts/run_eval_propagation.sh \
    --node gipdeep12 --max_visited 5000 --device cuda
```

---

## tmux Script Conventions

All tmux scripts share a common pattern:

1. Create (or reuse) a named tmux session
2. `srun` to allocate resources on a SLURM compute node
3. Activate the `geo_splat` conda environment
4. Run the inner script

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--node NAME` | varies | SLURM node name |
| `--cpus N` | varies | CPUs to request |
| `--gpus N` | (training only) | GPUs to request |
| `--time HH:MM:SS` | `24:00:00` | SLURM time limit |
| `--session NAME` | varies | tmux session name |
| `--conda_env NAME` | `geo_splat` | Conda environment |
| `--extra "ARGS"` | | Extra args forwarded to inner script |
| `--dry_run` | | Print without executing |

**Attach/detach:**
```bash
# Attach to a running session
tmux attach -t <session_name>

# Detach (inside tmux): Ctrl+B, then D

# List sessions
tmux ls

# Kill a session
tmux kill-session -t <session_name>
```

---

## Configuration

Scripts reference dataset configurations in `DataSets/configs/`:

```
DataSets/configs/
├── polynomial/      # Polynomial YAML configs + gaussian_sources/*.txt
└── tosca/            # TOSCA YAML configs + gaussian_sources/*.txt
```

See [DataSets/configs/README.md](../DataSets/configs/README.md) for full details.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Permission denied` | `chmod +x scripts/tosca/**/*.sh scripts/polynomial/**/*.sh` |
| `Script not found` | Run from project root: `cd /path/to/RaDe-GS` |
| `ModuleNotFoundError` | Activate conda: `conda activate geo_splat` |
| `CUDA out of memory` | Reduce `--num_views` or run training sequentially |
| `srun: error` | Check node availability: `sinfo -N` |
| `tmux session exists` | Script will reuse it; or `tmux kill-session -t NAME` |
