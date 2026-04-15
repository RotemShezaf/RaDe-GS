# Geodesic Distance Propagation on Gaussian Splats

This module implements a **Fast Marching Method** for computing geodesic distances on Gaussian splat representations using a learned model for distance prediction.

## Overview

The algorithm propagates geodesic distances from source points to all other points using:

1. **Three disjoint point sets**:
   - **Visited**: Points with finalized geodesic distances
   - **Wavefront**: Points with distances being computed
   - **Unvisited**: Points not yet processed

2. **Learned distance prediction**: Uses a trained `GaussianPatchTransformer` model to predict geodesic distances based on local neighborhood information.

## Algorithm

```
1. Initialize:
   - Mark source points as Visited with distance 0
   - Add neighbors of sources to Wavefront with distance ∞

2. Repeat until all points are Visited:
   a. For each point p in Wavefront:
      - Gather visited neighbors and their distances
      - Build model input from neighborhood features
      - Predict geodesic distance u(p) using the model
      
   b. Mark the minimum-distance Wavefront point p' as Visited
   
   c. Add all Unvisited neighbors of p' to Wavefront

3. Return distances for all points
```

## Module Structure

```
geodesic_propagation/
├── __init__.py                 # Module exports
├── priority_queue.py           # Wavefront priority queue implementation
├── input_builder.py            # Builds model inputs from Gaussian data
├── fast_marching.py            # Core Fast Marching algorithm
│
├── ── Evaluation Scripts ──
├── evaluate_geodesic.py        # Main CLI evaluation script
├── evaluate_model_vs_gt.py     # Per-point model vs GT evaluation (CSV + plots)
├── run_eval_fps.py             # Evaluation with FPS downsampling & GPU
├── run_eval_density_test.py    # Tests algorithm at different cloud densities
│
├── ── Analysis & Comparison ──
├── analyze_nn_stats.py                        # NN distance statistics for a Gaussian scene
├── compare_barycentric_vs_vertex_snapping.py  # Compare barycentric vs vertex-snapping GT
│
├── ── Evaluation Output ──
├── eval_output/                       # Evaluation results
├── eval_output_scale_opacity/         # Evaluation with scale+opacity features
├── eval_output_scale_opacity_4src/    # Evaluation with 4 source points
│
├── debug/                             # All debug & diagnostic scripts
│   ├── ── Model Diagnostics ──
│   ├── debug_model_on_training_data.py            # Sanity-check: run model on training data
│   ├── debug_model_accuracy_vs_gt_distance.py     # Model accuracy vs GT distance (bypasses FM)
│   ├── debug_gt_distances_test.py                 # Replace FM dists with GT to isolate errors
│   ├── debug_one_neighbor_patch.py                # Model prediction with only 1 valid neighbor
│   ├── diagnose_model.py                          # GT-like vs FM inputs: isolate problem source
│   │
│   ├── ── Propagation Diagnostics ──
│   ├── debug_fm_trace.py                  # Step-by-step FM trace, logs every update
│   ├── debug_one_pass.py                  # Single FM iteration, compare to GT
│   ├── debug_source_neighbors.py          # Run FM until ring-1 neighbors visited, compare
│   ├── debug_propagation_detailed.py      # Detailed FM vs training normalization trace
│   ├── debug_accuracy_vs_distance.py      # FM accuracy binned by GT distance
│   │
│   ├── ── Algorithm Investigation ──
│   ├── debug_graph_diameter.py            # Diagnose distance plateau from graph diameter
│   ├── debug_strategy_comparison.py       # Compare FM vs Dijkstra vs hybrid
│   ├── debug_two_pass.py                  # Two-pass: Euclidean Dijkstra → model refinement
│   ├── debug_refined_two_pass.py          # Refined two-pass with training-consistent filtering
│   │
│   ├── ── Data & Normalization ──
│   ├── debug_normalization.py             # Full normalization pipeline diagnostics
│   ├── debug_gt_mapping.py                # GT data consistency check
│   │
│   ├── ── Debug Output Artifacts ──
│   ├── debug_accuracy_vs_distance.csv                # CSV from debug_accuracy_vs_distance.py
│   ├── debug_model_accuracy_vs_gt_distance.csv       # CSV from debug_model_accuracy_vs_gt_distance.py
│   └── debug_output/                                 # Normalization debug visualizations (PNG, CSV)
│
├── utils/
│   ├── __init__.py
│   ├── model_handler.py        # Model loading and inference
│   ├── results_saver.py        # Results saving and export
│   └── output_path.py          # Structured output directories from model + gaussian paths
└── tests/
    ├── __init__.py
    ├── test_priority_queue.py
    ├── test_input_builder.py
    ├── test_fast_marching.py
    ├── test_integration.py
    ├── test_transformer_efficiency.py
    └── test_with_geodesic_ground_truth.py
```

## Usage

### Command Line

```bash
# Basic usage
python -m geodesic_propagation.evaluate_geodesic \
    --model_path /path/to/model.pt \
    --gaussian_output /path/to/gaussian/output \
    --source_indices 0 100 200 \
    --output_dir ./results

# With ground truth comparison
python -m geodesic_propagation.evaluate_geodesic \
    --model_path /path/to/model.pt \
    --gaussian_output /path/to/gaussian/output \
    --source_indices 0 \
    --geodesic_data /path/to/ground_truth.npz \
    --output_dir ./results
```

### Python API

```python
from geodesic_propagation import FastMarchingPropagator, GaussianInputBuilder
from geodesic_propagation.utils import ModelHandler

# Load model
model_handler = ModelHandler(model_path="path/to/model.pt")

# Build input builder — ring-1/ring-k neighbours and inference transforms
# are computed internally.  kNN parameters (n_neighbors, use_mahalanobis,
# adaptive_*) are read automatically from the dataset config; pass them
# explicitly only to override.
input_builder = GaussianInputBuilder(
    positions=positions,
    dataset_config="DataSets/configs/polynomial/combined_polynomial_all.yaml",
    ring=3,
    scales=scales,
    rotations=rotations,
    opacities=opacities,
    device="cuda",
    transforms_config=model_handler.get_transforms_config(),  # builds inference transforms
)

# Create and run propagator
propagator = FastMarchingPropagator(
    model_handler=model_handler,
    input_builder=input_builder,
    ring1_neighbors=input_builder.ring1_neighbors,
    ring_neighbors=input_builder.ring_neighbors,
    ring=3,
)

source_indices = [0, 100, 200]
distances = propagator.propagate(source_indices)
```

Adaptive kNN is also supported — these parameters are read from the
dataset config by default, or can be overridden explicitly:

```python
input_builder = GaussianInputBuilder(
    positions=positions,
    dataset_config="DataSets/configs/polynomial/combined_polynomial_all.yaml",
    ring=3,
    scales=scales, rotations=rotations, opacities=opacities,
    device="cuda",
    # Override config values if needed:
    n_neighbors=10,
    use_mahalanobis=True,
    adaptive_target_ring=3,
    adaptive_target_neighbors=128,
    adaptive_k_boost=20,
)
```

#### `create_propagator` — one-call factory

For the common case, `create_propagator` handles model loading, neighbour
computation, and builder/propagator construction in a single call.  It reads
`dataset_config` and `ring` automatically from the model checkpoint's
companion YAML:

```python
from geodesic_propagation.fast_marching import create_propagator

gaussian_data = {
    "positions": positions,
    "scales": scales,
    "rotations": rotations,
    "opacities": opacities,
    "sh_features": sh_features,
}

propagator = create_propagator(
    model_path="checkpoints/combined_polynomial_ring3/best_model.pt",
    gaussian_data=gaussian_data,
    # dataset_config and ring are read from the checkpoint's YAML;
    # pass them explicitly to override.
    device="cuda",
)
distances = propagator.propagate([0, 100, 200])
```

### Saving Results

```python
from geodesic_propagation.utils import ResultsSaver

saver = ResultsSaver("./output")

# Save propagation results
saver.save_propagation_results(
    distances=distances,
    source_indices=source_indices,
    positions=positions,
    metadata={'experiment': 'test'}
)

# Export PLY for visualization
saver.export_to_ply(
    positions=positions,
    distances=distances,
    output_name="geodesic_viz"
)

# Compare with ground truth
saver.save_evaluation_results(
    predicted_distances=distances,
    ground_truth_distances=gt_distances,
    source_indices=source_indices,
    metrics={'mae': 0.05, 'rmse': 0.07}
)
```

## Key Classes

### WavefrontPriorityQueue

Efficient priority queue for managing the wavefront:
- Min-heap with lazy deletion for fast updates
- Tracks point states (Unvisited/Wavefront/Visited)
- O(log n) insertion and extraction

### GaussianInputBuilder

Builds model inputs from Gaussian data for inference during Fast Marching.
Internally owns an inference-only `GaussianPatchDataset` instance created
from the dataset config, so all attribute lists, neighbour counts, and
normalization constants are set in one place.

- Filters neighbours to only those already visited
- Applies the full two-stage normalization (identical to training)
- Creates valid masks for attention
- Provides round-trip denormalization back to absolute geodesic distances

#### Constructor

```python
builder = GaussianInputBuilder(
    positions=positions,                     # (N, 3) required
    dataset_config="DataSets/configs/polynomial/combined_polynomial_all.yaml",  # path or dict
    ring=3,                                  # must match training
    normalization_factor=mean_nn_dist,       # global mean NN distance (overridden when n_neighbors is set)
    scales=scales,                           # (N, 3) optional
    rotations=rotations,                     # (N, 4) optional
    opacities=opacities,                     # (N,) or (N, 1) optional
    sh_features=sh_features,                 # (N, K) optional
    normals=normals,                         # (N, 3) optional
    per_point_nn_distances=per_point_nn_dist,# (N,) enables per-patch norm (overridden when n_neighbors is set)
    inference_transforms=None,               # deterministic transform callable
    device="cuda",
    # Ring computation (set n_neighbors to compute rings internally)
    n_neighbors=10,                          # ring-1 kNN k
    use_mahalanobis=False,                   # Mahalanobis distance for kNN
    # Adaptive kNN (optional)
    adaptive_target_ring=None,               # ring to optimise
    adaptive_target_neighbors=None,          # desired ring-k cap
    adaptive_k_boost=20,                     # max boosted k
    adaptive_max_mean_cut=2.0,               # convergence threshold
    adaptive_max_steps=5,                    # max binary-search iterations
    # Inference transforms from config
    transforms_config=None,                  # training transform config list (from YAML)
)
```

When ``n_neighbors`` is provided (explicitly or via config), ``ring1_neighbors``
and ``ring_neighbors`` are populated automatically.  The computed ``mean_nn_dist``
and ``per_point_nn_distances`` override the corresponding arguments.

kNN parameters (``n_neighbors``, ``use_mahalanobis``, ``adaptive_target_ring``,
``adaptive_target_neighbors``, ``adaptive_k_boost``, ``adaptive_max_mean_cut``,
``adaptive_max_steps``) are read from ``dataset_config`` by default and only
need to be passed explicitly to override the config values.

When ``transforms_config`` is provided and ``inference_transforms`` is ``None``,
inference-safe transforms are built from the config (stochastic augmentations
are filtered out).

All configuration (`attributes`, `max_neighbors`, `mask_constant`, `nn_mean`,
`ring_size_mapping`, etc.) is read from `dataset_config` — callers never need
to specify them separately.

#### `build_input` — single unified entry point

```python
result = builder.build_input(
    point_idx=u,
    all_neighbor_indices=ring_nbrs[u],    # full ring-k stencil
    neighbor_distances=distances,          # global (N,) array indexed by point
    ring1_neighbor_indices=ring1_nbrs[u],  # for r1_min dropout
    visited_mask=visited,                  # global bool (N,) array
)

if result is not None:
    neighborhood, point_features, valid_mask, build_info = result
```

Internally this calls `create_train_example` (stage-1 normalization) then
`GaussianPatchDataset.get_item_from_raw` (stage-2 normalization), producing
tensors **identical** to those the model saw during training.

`build_info` is a dict with keys:

| Key | Description |
|-----|-------------|
| `min_input` | Minimum visited-neighbour geodesic before shift |
| `current_normalization` | Per-patch normalization factor |
| `max_dist` | Scalar from stage-2 xyz/geodesic scaling |
| `neighborhood` | Same tensor as the first return value |
| `point_features` | Same tensor as the second return value |

#### `denormalize_result` — convert prediction back to absolute distance

```python
absolute_dist = builder.denormalize_result(raw_pred, build_info)
```

Applies the full inverse pipeline in reverse order:
1. Undo stage-2 (`_denormalize_results`): multiply by `max_dist`
2. Undo stage-1 (`denormalize_neighborhood`): scale by
   `current_normalization / nn_mean`, then add `min_input`

The result is clamped to be ≥ `min_input`.

#### Typical Fast Marching loop

```python
# For each wavefront point u:
result = builder.build_input(
    point_idx=u,
    all_neighbor_indices=ring_nbrs[u],
    neighbor_distances=distances,
    ring1_neighbor_indices=ring1_nbrs[u],
    visited_mask=visited,
)
if result is not None:
    neighborhood, point_features, valid_mask, build_info = result
    raw_pred = model(neighborhood, point_features, valid_mask).item()
    distances[u] = builder.denormalize_result(raw_pred, build_info)
```


### FastMarchingPropagator

Core algorithm implementation:
- Manages propagation loop
- Predicts distances using model
- Supports batched inference for efficiency
- Provides progress tracking

### ModelHandler

Model loading and inference:
- Loads checkpoints with configuration
- Handles device placement
- Provides prediction interface

### ResultsSaver

Output management:
- Saves results in NPZ format
- Exports PLY for visualization
- Generates evaluation reports

### `build_eval_output_dir` (from `utils.output_path`)

Constructs structured output directories from model and gaussian paths so
evaluations of different shapes/models don't overwrite each other:

```python
from geodesic_propagation.utils import build_eval_output_dir

path = build_eval_output_dir(
    "geodesic_propagation/eval_output",
    "checkpoints/combined_tosca_ring3/best_model.pth",
    "TrainData/TOSCA/SyntheticColmapData/blue_texture/cat2/high_res/decoupled_appearance/output",
)
# → geodesic_propagation/eval_output/combined_tosca_ring3/blue_texture/cat2/high_res
```

All four evaluation Python scripts (`evaluate_geodesic.py`, `evaluate_model_vs_gt.py`,
`run_eval_fps.py`, `run_eval_propagation.py`) use this automatically.
The `--output_dir` argument serves as the base; model and gaussian subpaths are appended.

## Running Tests

```bash
# Run all tests
cd /path/to/RaDe-GS
pytest geodesic_propagation/tests/ -v

# Run specific test file
pytest geodesic_propagation/tests/test_priority_queue.py -v

# Run with coverage
pytest geodesic_propagation/tests/ -v --cov=geodesic_propagation
```

---

## Evaluation Scripts

### `evaluate_geodesic.py`

**Main CLI evaluation script.** Loads a trained model, runs FM propagation from
specified source points, optionally compares with ground truth, and saves
results + metrics.

```bash
python -m geodesic_propagation.evaluate_geodesic \
    --model_path checkpoints/.../best_model.pth \
    --gaussian_output TrainData/.../output \
    --source_indices 0 100 200 \
    --output_dir ./results
```

### `evaluate_model_vs_gt.py`

**Per-point model evaluation against ground truth** (bypasses FM entirely).
For every point, builds a model input using GT neighbour distances (all
neighbours marked as visited), runs the model, and compares to GT. Outputs
per-point CSV, console tables binned by GT distance, and histogram PNGs.

```bash
python geodesic_propagation/evaluate_model_vs_gt.py \
    --train_config models/configs/combined_polynomial_ring3.yaml
```

### `run_eval_fps.py`

**Evaluation with FPS downsampling and GPU support.** Runs FM propagation
from a GT source point, compares predicted vs GT distances. Supports
optional furthest-point-sampling to reduce cloud size.

```bash
# Full scene on CPU:
python geodesic_propagation/run_eval_fps.py --device cpu --fps_target 0

# Downsampled on GPU via srun:
srun --gres=gpu:1 --pty bash -c \
    'conda activate geo_splat && python geodesic_propagation/run_eval_fps.py --fps_target 10000'
```

### `run_eval_density_test.py`

**Density sweep.** Tests the propagation algorithm at multiple FPS densities
(full dense cloud ~44k points through various downsampled sizes) to find
the density threshold where accuracy degrades.

```bash
srun --gres=gpu:1 --pty bash -c \
    'conda activate geo_splat && python geodesic_propagation/run_eval_density_test.py'
```

---

## Analysis & Comparison Scripts

### `analyze_nn_stats.py`

**Nearest-neighbour distance statistics** for a Gaussian PLY scene. Prints
ASCII histograms of NN distances, useful for choosing normalization factors
and understanding scene density.

```bash
python geodesic_propagation/analyze_nn_stats.py \
    --gaussian_dir TrainData/.../output \
    --n_neighbors 10 --mesh_resolution 0.0058
```

### `compare_barycentric_vs_vertex_snapping.py`

**Compare two GT geodesic NPZ files**: one computed with barycentric
interpolation, one with vertex-snapping. Prints ASCII histograms of the
per-Gaussian distance differences.

```bash
python geodesic_propagation/compare_barycentric_vs_vertex_snapping.py \
    --old_file <vertex_snapping.npz> --old_source_row 0 \
    --new_file <barycentric.npz>      --new_source_row 0
```

---

## Debug Scripts

### Model Diagnostics

These scripts test the model's intrinsic prediction accuracy, independent
of the FM propagation loop.

#### `debug_model_on_training_data.py`

**Quick sanity check.** Loads actual training data through
`GaussianPatchDataset`, feeds it to the model, and prints whether
predictions are reasonable. First thing to run when a new checkpoint
behaves unexpectedly.

#### `debug_model_accuracy_vs_gt_distance.py`

**Model accuracy vs GT distance (bypasses FM).** For every Gaussian, feeds
the model GT neighbour distances (the same setup as training) and checks
prediction accuracy as a function of GT distance from source. Isolates the
model's intrinsic accuracy from FM cascading errors.

Outputs: `debug_model_accuracy_vs_gt_distance.csv`

#### `debug_gt_distances_test.py`

**GT-distance substitution test.** Replaces FM-estimated distances with GT
distances in the model input. If the model predicts well → problem is FM
error accumulation. If the model still fails → problem is model capacity or
input distribution.

#### `debug_one_neighbor_patch.py`

**Single-neighbour edge case.** Builds a patch where only 1 neighbour is
marked as visited (the worst-case during early FM expansion) and runs the
model on it. Compares against the full-neighbourhood prediction.

```bash
python geodesic_propagation/debug/debug_one_neighbor_patch.py \
    --train_config models/configs/combined_polynomial_ring3_dropout99.yaml \
    --source_idx 166
```

#### `diagnose_model.py`

**GT-like vs FM-like inputs.** Feeds the model (a) ground-truth-like inputs
(all neighbours, correct distances) and (b) FM partial-neighbourhood inputs.
Isolates whether the problem is model capacity (A fails) or domain gap
between training and FM inference (A works, B fails).

### Propagation Diagnostics

These scripts trace the FM propagation loop to find where errors arise.

#### `debug_fm_trace.py`

**Step-by-step FM trace.** Logs every `add_or_update` call during
propagation, showing whether distances grow (correct) or collapse
(accumulation bug).

#### `debug_one_pass.py`

**Single FM iteration.** Initializes sources, predicts the initial wavefront,
then runs exactly ONE iteration of `propagate_batch`. Prints detailed
per-point info vs GT. Useful for validating the first expansion step.

#### `debug_source_neighbors.py`

**Ring-1 neighbour convergence.** Runs FM until every ring-1 neighbour of
the source is VISITED (distance finalized). Compares predicted distances of
those neighbours against GT. Tests the most basic correctness: can FM get
the source's immediate neighbours right?

#### `debug_propagation_detailed.py`

**Normalization trace.** Compares model inputs/outputs during FM propagation
vs training data, tracing the two-stage normalization step-by-step to find
mismatches between training and inference pipelines.

#### `debug_accuracy_vs_distance.py`

**Accuracy vs distance profile.** Runs FM for many iterations and measures
how prediction accuracy varies with GT distance from source. For each
finalized point, records predicted distance, GT distance, number of visited
neighbours, and raw model output. Results binned into a table plus per-point
CSV.

Outputs: `debug_accuracy_vs_distance.csv`

### Algorithm Investigation

These scripts test alternative propagation strategies to understand the
design space.

#### `debug_graph_diameter.py`

**Distance plateau diagnosis.** Investigates why predicted distances may
plateau at a fixed value (e.g. ~0.037). Measures the actual graph diameter
at different ring levels and tests whether decoupling the expansion ring
from the model context ring increases the number of accumulation steps.

#### `debug_strategy_comparison.py`

**Strategy comparison.** Compares three approaches:
1. Current FM with learned model
2. Pure Dijkstra (Euclidean edge weights, no model)
3. Hybrid: Dijkstra for initial estimate, model for refinement

Tests the hypothesis that shift-to-zero normalization destroys distance
accumulation.

#### `debug_two_pass.py`

**Two-pass approach.** Pass 1: Euclidean Dijkstra (fast, Corr ≈ 0.94).
Pass 2: model refinement using Dijkstra distances as context. Tests whether
the model can correct Euclidean estimates given reasonably-correct neighbour
distances.

#### `debug_refined_two_pass.py`

**Refined two-pass.** Extends `debug_two_pass.py` with training-consistent
neighbour filtering: filters to neighbours with `dijkstra_dist ≤ dijkstra_dist(target)`,
replicating the training regime where `geo ≤ p_u`.

### Data & Normalization Diagnostics

#### `debug_normalization.py`

**Comprehensive normalization audit.** Loads real Gaussian data + GT, then
traces both the training and inference normalization paths stage by stage.
Outputs per-column statistics, round-trip errors, and histograms.

Outputs: `debug/debug_output/normalization_debug.csv`, `debug/debug_output/normalization_histograms.png`

#### `debug_gt_mapping.py`

**GT data consistency check.** For each source, verifies that
`closest_mesh_indices[source_gaussian_indices[i]] == source_indices[i]`,
i.e. the source Gaussian's closest mesh vertex matches the intended mesh
source. Also prints self-geodesic distances.

---

## Output Directories

| Directory | Contents |
|-----------|----------|
| `debug/` | All debug & diagnostic scripts, CSVs, and output artifacts |
| `debug/debug_output/` | Normalization debug visualizations and CSV from `debug_normalization.py` |
| `eval_output/<model>/<gaussian>/` | Structured evaluation results (auto-partitioned by model + shape) |
| `eval_output_scale_opacity/` | Evaluation results using scale + opacity features |
| `eval_output_scale_opacity_4src/` | Evaluation results with 4 source points |

---

## Dependencies

- numpy
- torch
- scikit-learn
- tqdm
- matplotlib (for PLY export with colormap)
- pointnet2_ops (for FPS downsampling in eval scripts)

## Notes

- The model should be trained using the same attributes specified in the propagator
- Mahalanobis distance for neighbors is recommended for Gaussian splats
- Ring-2 or Ring-3 neighborhoods typically work best
- Batched inference (`propagate_batch`) is faster for large point clouds
- Most debug scripts default to `Paraboloid/level_04/light_0` — edit the
  hardcoded `GAUSSIAN_OUT` / `MODEL_PATH` at the top of each script to
  change the scene or checkpoint
- All debug scripts live under `debug/` and are run from the project root:
  `python geodesic_propagation/debug/<script_name>.py`
