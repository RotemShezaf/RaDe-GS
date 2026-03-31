# Test Suite for GenerateData Utils

This directory contains unit tests for the utilities in `/GenerateData/utils/`.

## Test Files

- `test_load_utils.py` - Tests for Gaussian data loading and CPU-compatible processing
  - GaussianDataCPU class tests
  - Activation functions (exp, sigmoid, quaternion normalization)
  - Spherical harmonics feature handling
  - Edge cases and error handling

- `test_helpers.py` - Tests for geodesic distance computation helper functions

- `test_data_generation_utils.py` - Tests for KNN, neighborhood rings, adaptive KNN, and point mapping
  - Mahalanobis distance computation
  - Ring-1 neighbor finding (Euclidean and Mahalanobis)
  - Ring neighborhood expansion (ring 1-4)
  - `adaptive_ring1_neighbors` — per-point k adjustment for target ring-k count
  - `get_all_points_nbrs_all_rings` / `get_all_points_nbrs_single_ring` with adaptive mode
  - Point-to-surface mapping (Euclidean and Mahalanobis)
  - Source mesh generation and mapping
  - Gaussian-to-mesh mapping (Euclidean and Mahalanobis)
  - Geodesic distance transfer — **vertex snapping and barycentric interpolation modes**
  - `project_points_to_triangles_vectorized` — vectorised Ericson-2005 closest-point-on-triangle
  - `find_closest_mesh_faces_barycentric` — 1-ring face search and barycentric weight computation
  - Partial results saving and merging
  - Metadata handling

## Running Tests

### Interactive with srun (Recommended)

Run all tests interactively with GPU allocation:

```bash
cd /home/rotem.shezaf/RaDe-GS/GenerateData/tests
./run_tests.sh
```

Run specific test file:

```bash
./run_tests.sh test_load_utils.py
```

Run specific test by name:

```bash
./run_tests.sh -k test_get_rotation_normalization
```

### Batch Job with sbatch

Submit as a batch job:

```bash
sbatch run_tests_batch.sh
```

Monitor job status:

```bash
squeue -u $USER
```

View output:

```bash
tail -f test_output_<job_id>.log
```

### Direct pytest (without SLURM)

If running on a node with GPU already allocated:

```bash
conda activate geo_splat
cd /home/rotem.shezaf/RaDe-GS
python -m pytest GenerateData/tests -v
```

## Test Requirements

The tests require the following packages (available in `geo_splat` environment):
- pytest
- numpy
- scipy
- plyfile
- trimesh (for some integration tests)

## Test Coverage

### GaussianDataCPU Class Tests
- ✅ Initialization and data storage
- ✅ Position access (get_xyz)
- ✅ Scale activation (exp)
- ✅ Rotation normalization (unit quaternions, positive first component)
- ✅ Opacity activation (sigmoid)
- ✅ Spherical harmonics (DC and rest coefficients)
- ✅ Feature concatenation
- ✅ Edge cases (extreme values, zero quaternions)

### Helper Function Tests
- ✅ Source mesh generation and mapping
- ✅ Gaussian-to-mesh mapping (Euclidean distance)
- ✅ Gaussian-to-mesh mapping (Mahalanobis distance)
- ✅ Geodesic distance transfer
- ✅ Partial result saving and loading
- ✅ Partial result merging
- ✅ Missing source detection
- ✅ Metadata serialization
- ✅ Edge cases (empty data, single Gaussian)

## Notes

- Tests use mocking for external dependencies (mesh generation, KDTree)
- Some tests require temporary directories (handled by pytest's tmp_path fixture)
- GPU is requested but not strictly required for most tests (CPU fallback available)
- Tests are designed to be fast (< 1 minute total runtime)
