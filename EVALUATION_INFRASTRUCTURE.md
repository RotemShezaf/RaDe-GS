# Evaluation Infrastructure Changes

## Overview

Major improvements to evaluation scripts and result organization for both
Polynomial and TOSCA datasets.  Changes span shell launcher scripts,
Python evaluation output paths, geodesic distance analysis, and parallel
evaluation of TOSCA test shapes.

---

## 1. Polynomial Evaluation Scripts — Style Alignment

**Directory:** `scripts/polynomial/evaluate_scripts/`

Updated all scripts to match the `new_dataset/evaluate_scripts/` template style:

| Script | Purpose |
|--------|---------|
| `evaluate_geodesic.sh` | Fast Marching geodesic propagation |
| `evaluate_model_vs_gt.sh` | Model accuracy vs ground truth |
| `analyze_nn_stats.sh` | Nearest-neighbor statistics |
| `run_eval_fps.sh` | FPS-downsampled evaluation |
| `run_eval_propagation.sh` | Debug FM propagation |
| `run_all_evaluations.sh` | Master script launching all above |
| `sweep_knn.sh` | **NEW** — kNN sweep across k values |

### Key changes
- Python resolved via `CONDA_BASE/envs/geo_splat/bin/python3` with fallback
- `--refine_passes` support in geodesic and FPS scripts
- Config defaults to `models/configs/one_source/combined_polynomial_ring3.yaml`
- 3-level directory depth (`$SCRIPT_DIR/../../../`)
- Session names without `_nd` suffix

---

## 2. TOSCA Evaluation Scripts

**Directory:** `scripts/tosca/evaluate_scripts/` *(new)*

Created a complete set of evaluation scripts for TOSCA, modeled after the
Polynomial scripts but configured for TOSCA:

| Script | Purpose |
|--------|---------|
| `evaluate_geodesic.sh` | Fast Marching geodesic propagation |
| `evaluate_model_vs_gt.sh` | Model accuracy vs ground truth |
| `analyze_nn_stats.sh` | Nearest-neighbor statistics |
| `run_eval_fps.sh` | FPS-downsampled evaluation |
| `run_eval_propagation.sh` | Debug FM propagation |
| `run_all_evaluations.sh` | Master script launching all above |
| `evaluate_all_test_shapes.sh` | **Parallel evaluation of all test shapes** |

### Configuration defaults
- Config: `models/configs/tosca/combined_tosca_ring3.yaml`
- Checkpoint: `checkpoints/combined_tosca_ring3/best_model.pth`
- Default Gaussian: `TrainData/TOSCA/SyntheticColmapData/blue_texture/cat2/high_res/decoupled_appearance/output`
- Session names use `_tosca` suffix
- All scripts support `EVAL_SHAPE_SUFFIX` env variable for unique tmux session names

---

## 3. Parallel Test Shape Evaluation

**Script:** `scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh`

Identified TOSCA shapes with high resolution that are **not** in the
training set:

| Shape | Status |
|-------|--------|
| `gorilla8` | Test only (not in training) |
| `horse10` | Test only (not in training) |
| `michael16` | Test only (not in training) |

The script:
- Iterates over all test shapes (configurable via `--shapes`)
- Launches the full evaluation pipeline for each shape
- Uses `EVAL_SHAPE_SUFFIX` to create unique tmux sessions per shape
  (e.g. `eval_geodesic_tosca_gorilla8`, `eval_geodesic_tosca_horse10`)
- Verifies gaussian data exists before launching
- Supports `--sequential` and `--dry_run` modes

### Usage
```bash
# Evaluate all test shapes in parallel
bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh

# Evaluate a specific shape
bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh --shapes gorilla8

# Dry run
bash scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh --dry_run
```

---

## 4. Structured Output Directories for Evaluation Results

**Module:** `geodesic_propagation/utils/output_path.py` *(new)*

Evaluation Python scripts now save results in a structured directory hierarchy
that includes the model name and gaussian path, preventing result overrides
when evaluating multiple shapes or models.

### Before
```
geodesic_propagation/eval_output/
    model_vs_gt.csv           ← overwritten by each evaluation
    model_vs_gt_plots.png
```

### After
```
geodesic_propagation/eval_output/
    combined_tosca_ring3/
        blue_texture/cat2/high_res/
            model_vs_gt.csv
            model_vs_gt_plots.png
        blue_texture/gorilla8/high_res/
            model_vs_gt.csv
            model_vs_gt_plots.png
    one_source/combined_polynomial_ring3/
        blue_texture/Saddle/level_04/light_4/
            model_vs_gt.csv
```

### Modified files
| File | Default base dir |
|------|-----------------|
| `geodesic_propagation/evaluate_geodesic.py` | `./geodesic_results` |
| `geodesic_propagation/evaluate_model_vs_gt.py` | `geodesic_propagation/eval_output` |
| `geodesic_propagation/run_eval_fps.py` | `geodesic_propagation/eval_propagation_output` |
| `geodesic_propagation/run_eval_propagation.py` | `geodesic_propagation/eval_propagation_debug` |

The `--output_dir` argument still serves as the **base** directory.  The
model and gaussian subpath are appended automatically via
`build_eval_output_dir()`.

### Path derivation logic
- **Model name:** extracted from checkpoint path by stripping the
  `checkpoints/` prefix and the `.pth` filename
  (`checkpoints/combined_tosca_ring3/best_model.pth` → `combined_tosca_ring3`)
- **Gaussian subpath:** strips `TrainData/*/SyntheticColmapData/` prefix and
  `decoupled_appearance/output` suffix
  (`TrainData/TOSCA/.../blue_texture/cat2/high_res/decoupled_appearance/output`
  → `blue_texture/cat2/high_res`)

---

## 5. Geodesic Valid-Neighbor Analysis

**Script:** `DataSets/analyze_geodesic_valid_neighbors.py` *(new)*
**Launcher:** `scripts/tosca/run_adaptive_analysis.sh` *(rewritten)*

Replaced the old adaptive kNN ring-statistics analysis with a geodesic
distance–based valid-neighbor analysis.  For each shape and GT source, the
script counts how many ring-k neighbors have geodesic distance ≤ the center
point's distance (i.e. would be "valid" / already visited during Fast Marching).

### Features
- **Per-source analysis:** samples a few GT sources, reports percentile
  distribution of valid-neighbor counts
- **Multi-source analysis:** randomly selects `num_sources` sources (like
  training), computes per-point min distances, reports valid-neighbor stats
- Supports both TOSCA and Polynomial datasets
- Configurable ring level, kNN k, number of sources, Mahalanobis option

### Usage
```bash
# Default: TOSCA dataset, ring 3, 10 neighbors, 1 source
bash scripts/tosca/run_adaptive_analysis.sh

# Multi-source analysis with 3 sources
bash scripts/tosca/run_adaptive_analysis.sh --num_sources 3 --num_trials 10

# Specific shapes with Mahalanobis
bash scripts/tosca/run_adaptive_analysis.sh --shapes cat2,gorilla8 --use_mahalanobis

# Polynomial dataset
bash scripts/tosca/run_adaptive_analysis.sh --dataset polynomial
```

### Shell script options
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `tosca` | `tosca`, `polynomial`, or `both` |
| `--shapes` | all | Comma-separated shape filter |
| `--ring` | 3 | Ring level for neighborhoods |
| `--n_neighbors` | 10 | Ring-1 kNN k |
| `--num_sources` | 1 | Sources per multi-source trial |
| `--num_trials` | 5 | Random trials for multi-source |
| `--use_mahalanobis` | off | Use Mahalanobis kNN |
| `--node` | `gipdeep10` | SLURM node |
| `--dry_run` | off | Print without executing |

---

## 6. TMUX Session Name Suffixing

All TOSCA evaluation scripts now append `${EVAL_SHAPE_SUFFIX:-}` to their
tmux session names.  When the parent script `evaluate_all_test_shapes.sh`
sets `EVAL_SHAPE_SUFFIX=_gorilla8`, the tmux sessions become:

```
eval_geodesic_tosca_gorilla8
eval_model_vs_gt_tosca_gorilla8
eval_nn_stats_tosca_gorilla8
eval_fps_tosca_gorilla8
eval_propagation_tosca_gorilla8
```

This prevents session collisions when evaluating multiple shapes in parallel.

---

## Files Changed Summary

### New files
- `scripts/tosca/evaluate_scripts/evaluate_geodesic.sh`
- `scripts/tosca/evaluate_scripts/evaluate_model_vs_gt.sh`
- `scripts/tosca/evaluate_scripts/analyze_nn_stats.sh`
- `scripts/tosca/evaluate_scripts/run_eval_fps.sh`
- `scripts/tosca/evaluate_scripts/run_eval_propagation.sh`
- `scripts/tosca/evaluate_scripts/run_all_evaluations.sh`
- `scripts/tosca/evaluate_scripts/evaluate_all_test_shapes.sh`
- `scripts/polynomial/evaluate_scripts/sweep_knn.sh`
- `geodesic_propagation/utils/output_path.py`
- `DataSets/analyze_geodesic_valid_neighbors.py`

### Modified files
- `scripts/polynomial/evaluate_scripts/*.sh` (6 files — style alignment)
- `scripts/tosca/run_adaptive_analysis.sh` (rewritten for geodesic analysis)
- `geodesic_propagation/evaluate_geodesic.py` (structured output dirs)
- `geodesic_propagation/evaluate_model_vs_gt.py` (structured output dirs)
- `geodesic_propagation/run_eval_fps.py` (structured output dirs)
- `geodesic_propagation/run_eval_propagation.py` (structured output dirs)
- `geodesic_propagation/utils/__init__.py` (export `build_eval_output_dir`)
