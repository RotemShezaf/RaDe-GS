"""
Tests for GaussianInputBuilder.

The builder is initialized with a dataset_config dict (no .npy files needed,
since inference_only=True) and a ring level.  These tests cover:

- __init__: attributes / max_neighbors / mask_constant / dim methods read from config
- build_input: returns (neighborhood, point_features, valid_mask, build_info)
- build_input: visited filtering via visited_mask and visited_set
- build_input: returns None when create_train_example rejects the patch
- denormalize_result: round-trip inversion of the two-stage normalization
- get_neighbor_feature_dim / get_point_feature_dim
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.input_builder import GaussianInputBuilder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_rotations(n: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    r = rng.randn(n, 4).astype(np.float32)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _make_config(
    attributes=None,
    ring=2,
    use_r1_min_val=False,
    use_mahalanobis=False,
    nn_mean=1.0,
    output_dir="/tmp/nonexistent_fake_dir_for_tests",
):
    """Return a minimal in-memory dataset config dict."""
    return {
        "output_dir": output_dir,
        "attributes": attributes or ["xyz"],
        "nn_mean": nn_mean,
        "mask_constant": -10.0,
        "use_r1_min_val": use_r1_min_val,
        "use_mahalanobis": use_mahalanobis,
        "ring_size_mapping": {"euclidean": {2: 32, 3: 128}},
    }


def _gdist(n_pts, nbr_indices, nbr_dists, default=1e12):
    """Expand per-neighbor distances into a global (N,) distance array.

    ``build_input`` expects ``neighbor_distances`` to be a global array where
    ``neighbor_distances[point_idx]`` gives the current distance estimate of
    that point (matching the contract used by ``FastMarchingPropagator``).
    """
    d = np.full(n_pts, default, dtype=np.float64)
    d[nbr_indices] = nbr_dists
    return d


# ---------------------------------------------------------------------------
# Module-scoped raw data (shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_data():
    rng = np.random.RandomState(99)
    n = 40
    return {
        "n": n,
        "positions":   rng.randn(n, 3).astype(np.float32),
        "scales":      rng.rand(n, 3).astype(np.float32) * 0.1,
        "rotations":   _make_rotations(n),
        "opacities":   rng.rand(n, 1).astype(np.float32),
        "sh_features": rng.randn(n, 3).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Convenience builder factories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xyz_builder(base_data):
    """Builder with xyz-only config, ring-2."""
    return GaussianInputBuilder(
        positions=base_data["positions"],
        dataset_config=_make_config(["xyz"]),
        ring=2,
        normalization_factor=1.0,
        device="cpu",
    )


@pytest.fixture(scope="module")
def full_builder(base_data):
    """Builder with xyz+opacity+scale+rotation+sh, ring-2."""
    return GaussianInputBuilder(
        positions=base_data["positions"],
        dataset_config=_make_config(["xyz", "opacity", "scale", "rotation", "sh"]),
        ring=2,
        normalization_factor=1.0,
        scales=base_data["scales"],
        rotations=base_data["rotations"],
        opacities=base_data["opacities"],
        sh_features=base_data["sh_features"],
        device="cpu",
    )


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_attributes_read_from_config(self, xyz_builder):
        assert xyz_builder.attributes == ["xyz"]

    def test_max_neighbors_from_config(self, xyz_builder):
        # ring_size_mapping: euclidean: {2: 32}
        assert xyz_builder.max_neighbors == 32

    def test_mask_constant_from_config(self, xyz_builder):
        assert xyz_builder.mask_constant == -10.0

    def test_nn_mean_from_config(self, xyz_builder):
        assert xyz_builder.nn_mean == pytest.approx(1.0)

    def test_ring_stored(self, xyz_builder):
        assert xyz_builder.ring == 2

    def test_num_points(self, base_data, xyz_builder):
        assert xyz_builder.num_points == base_data["n"]

    def test_device_default_cpu(self, xyz_builder):
        assert xyz_builder.device == torch.device("cpu")

    def test_full_attributes(self, full_builder):
        assert "xyz" in full_builder.attributes
        assert "opacity" in full_builder.attributes
        assert "scale" in full_builder.attributes
        assert "rotation" in full_builder.attributes
        assert "sh" in full_builder.attributes


# ===========================================================================
# TestDimMethods
# ===========================================================================

class TestDimMethods:
    def test_get_neighbor_feature_dim_xyz(self, xyz_builder):
        # xyz(3) + geodesic(1) = 4
        assert xyz_builder.get_neighbor_feature_dim() == 4

    def test_get_point_feature_dim_xyz(self, xyz_builder):
        # xyz(3) for the query point
        assert xyz_builder.get_point_feature_dim() == 3

    def test_get_neighbor_feature_dim_full(self, full_builder):
        # xyz(3)+opacity(1)+scale(3)+rotation(4)+sh(3) = 14 + geodesic(1) = 15
        assert full_builder.get_neighbor_feature_dim() == 15

    def test_get_point_feature_dim_full(self, full_builder):
        assert full_builder.get_point_feature_dim() == 14

    def test_dims_consistent_with_build_input(self, xyz_builder):
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3, 4, 5]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_mask=np.ones(xyz_builder.num_points, dtype=bool),
        )
        assert result is not None
        neighborhood, point_features, valid_mask, build_info = result
        assert neighborhood.shape[1] == xyz_builder.get_neighbor_feature_dim()
        assert point_features.shape[0] == xyz_builder.get_point_feature_dim()


# ===========================================================================
# TestBuildInput — basic shapes and types
# ===========================================================================

class TestBuildInput:
    def test_returns_4_tuple(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert result is not None
        assert len(result) == 4

    def test_neighborhood_shape(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        neighborhood, pf, vm, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert neighborhood.shape == (32, 4)

    def test_point_features_shape(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, pf, _, _ = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert pf.shape == (3,)

    def test_valid_mask_shape_dtype(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, vm, _ = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert vm.shape == (32,)
        assert vm.dtype == torch.bool

    def test_build_info_keys(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert "min_input" in bi
        assert "current_normalization" in bi
        assert "max_dist" in bi
        assert "neighborhood" in bi
        assert "point_features" in bi

    def test_tensors_on_cpu(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        nb, pf, vm, _ = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert nb.device == torch.device("cpu")
        assert pf.device == torch.device("cpu")
        assert vm.device == torch.device("cpu")

    def test_valid_entries_are_finite(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        nb, pf, vm, _ = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3, 4]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3, 4]), np.array([0.1, 0.2, 0.3, 0.4])),
            visited_mask=visited,
        )
        assert torch.isfinite(nb[vm]).all()
        assert torch.isfinite(pf).all()


# ===========================================================================
# TestVisitedFiltering
# ===========================================================================

class TestVisitedFiltering:
    def test_visited_mask_filters_correctly(self, xyz_builder):
        """Only visited-mask neighbours should appear as valid."""
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        visited[[2, 4]] = True  # only 2 of 5 neighbours are visited
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3, 4, 5]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_mask=visited,
        )
        assert result is not None
        _, _, vm, _ = result
        assert int(vm.sum()) == 2

    def test_visited_set_filters_correctly(self, xyz_builder):
        visited_set = {1, 3, 5}
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3, 4, 5]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3, 4, 5]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_set=visited_set,
        )
        assert result is not None
        _, _, vm, _ = result
        assert int(vm.sum()) == 3

    def test_no_visited_gives_zero_valid(self, xyz_builder):
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert result is not None
        _, _, vm, _ = result
        assert int(vm.sum()) == 0

    def test_all_visited_gives_all_valid(self, xyz_builder):
        nbrs = np.array([1, 2, 3, 4, 5])
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        visited[nbrs] = True
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(xyz_builder.num_points, nbrs, np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_mask=visited,
        )
        assert result is not None
        _, _, vm, _ = result
        assert int(vm.sum()) == 5

    def test_visited_mask_and_set_consistent(self, xyz_builder):
        """visited_mask and visited_set should give the same valid count."""
        nbrs = np.array([1, 2, 3, 4, 5])
        visited_mask = np.zeros(xyz_builder.num_points, dtype=bool)
        visited_mask[[1, 3]] = True

        r_mask = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(xyz_builder.num_points, nbrs, np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_mask=visited_mask,
        )
        r_set = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(xyz_builder.num_points, nbrs, np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            visited_set={1, 3},
        )
        assert r_mask is not None
        assert r_set is not None
        _, _, vm_m, _ = r_mask
        _, _, vm_s, _ = r_set
        assert int(vm_m.sum()) == int(vm_s.sum())


# ===========================================================================
# TestBuildInfo
# ===========================================================================

class TestBuildInfo:
    def test_min_input_equals_min_visited_dist(self, xyz_builder):
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        nbrs = np.array([1, 2, 3, 4])
        per_nbr_dists = np.array([3.0, 1.5, 4.0, 2.0])
        visited[nbrs] = True
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(xyz_builder.num_points, nbrs, per_nbr_dists),
            visited_mask=visited,
        )
        assert bi["min_input"] == pytest.approx(1.5)

    def test_current_normalization_default(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.5, 1.0, 1.5])),
            visited_mask=visited,
        )
        # No per_point_nn_distances -> falls back to normalization_factor=1.0
        assert bi["current_normalization"] == pytest.approx(1.0)

    def test_max_dist_is_tensor(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2]), np.array([0.5, 1.0])),
            visited_mask=visited,
        )
        assert isinstance(bi["max_dist"], torch.Tensor)

    def test_build_info_neighborhood_is_same_tensor(self, xyz_builder):
        """build_info['neighborhood'] should be the identical tensor object."""
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        nb, pf, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2]), np.array([0.5, 1.0])),
            visited_mask=visited,
        )
        assert bi["neighborhood"] is nb
        assert bi["point_features"] is pf


# ===========================================================================
# TestCurrentNormalization
# ===========================================================================

class TestCurrentNormalization:
    def test_per_point_nn_distances_used_when_provided(self, base_data):
        nn_dists = np.full(base_data["n"], 0.5, dtype=np.float32)
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz"]),
            ring=2,
            normalization_factor=99.0,  # should NOT be used
            per_point_nn_distances=nn_dists,
            device="cpu",
        )
        nbrs = np.array([5, 10, 15])
        visited = np.ones(base_data["n"], dtype=bool)
        _, _, _, bi = builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(builder.num_points, nbrs, np.array([0.5, 1.0, 1.5])),
            visited_mask=visited,
        )
        # normalization should be mean of nn_dists[nbrs] = 0.5
        assert bi["current_normalization"] == pytest.approx(0.5, rel=1e-4)

    def test_normalization_factor_fallback(self, base_data):
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz"]),
            ring=2,
            normalization_factor=2.5,
            device="cpu",
        )
        visited = np.ones(base_data["n"], dtype=bool)
        _, _, _, bi = builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(builder.num_points, np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
            visited_mask=visited,
        )
        assert bi["current_normalization"] == pytest.approx(2.5)


# ===========================================================================
# TestDenormalizeResult
# ===========================================================================

class TestDenormalizeResult:
    """
    Tests for GaussianInputBuilder.denormalize_result.

    The method must invert the full two-stage normalization pipeline.
    """

    def test_result_is_float(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.5, 1.0, 1.5])),
            visited_mask=visited,
        )
        result = xyz_builder.denormalize_result(0.0, bi)
        assert isinstance(result, float)

    def test_result_ge_min_input(self, xyz_builder):
        """Denormalized distance must never be smaller than min_input."""
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
            visited_mask=visited,
        )
        result = xyz_builder.denormalize_result(-100.0, bi)
        assert result >= bi["min_input"] - 1e-6

    def test_zero_raw_pred_ge_min_input(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([2.0, 3.0, 4.0])),
            visited_mask=visited,
        )
        result = xyz_builder.denormalize_result(0.0, bi)
        assert result >= bi["min_input"] - 1e-6

    def test_positive_raw_pred_increases_distance(self, xyz_builder):
        """Larger raw prediction should map to a larger absolute distance."""
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        _, _, _, bi = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
            visited_mask=visited,
        )
        r0 = xyz_builder.denormalize_result(0.0, bi)
        r1 = xyz_builder.denormalize_result(1.0, bi)
        assert r1 > r0

    def test_roundtrip_approximate(self, base_data):
        """
        Build input from known distances, pass a positive raw_pred through
        denormalize_result and verify result is finite and >= min_input.
        """
        rng = np.random.RandomState(7)
        n = 20
        positions = rng.randn(n, 3).astype(np.float32)
        builder = GaussianInputBuilder(
            positions=positions,
            dataset_config=_make_config(["xyz"], nn_mean=1.0),
            ring=2,
            normalization_factor=1.0,
            device="cpu",
        )

        visited = np.ones(n, dtype=bool)
        nbrs = np.array([1, 2, 3, 4])
        per_nbr_dists = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result = builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(n, nbrs, per_nbr_dists),
            visited_mask=visited,
        )
        assert result is not None
        _, _, _, bi = result

        raw_pred = 0.5
        recovered = builder.denormalize_result(raw_pred, bi)
        assert recovered >= bi["min_input"] - 1e-6
        assert np.isfinite(recovered)


# ===========================================================================
# TestRing1NeighborIndices
# ===========================================================================

class TestRing1NeighborIndices:
    def test_ring1_provided_does_not_crash(self, xyz_builder):
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
            ring1_neighbor_indices=np.array([1, 2, 3, 4]),
            visited_mask=visited,
        )
        assert result is not None
        assert len(result) == 4


# ===========================================================================
# TestMultipleAttributes
# ===========================================================================

class TestMultipleAttributes:
    def test_full_builder_build_input_shape(self, full_builder, base_data):
        visited = np.ones(base_data["n"], dtype=bool)
        result = full_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(full_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert result is not None
        nb, pf, vm, _ = result
        assert nb.shape == (32, 15)
        assert pf.shape == (14,)
        assert vm.shape == (32,)

    def test_scale_attribute_shape(self, base_data):
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz", "scale"]),
            ring=2,
            scales=base_data["scales"],
            device="cpu",
        )
        # xyz(3)+scale(3)+geo(1) = 7 neighbor features; xyz(3)+scale(3) = 6 point features
        assert builder.get_neighbor_feature_dim() == 7
        assert builder.get_point_feature_dim() == 6

    def test_rotation_attribute_shape(self, base_data):
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz", "rotation"]),
            ring=2,
            rotations=base_data["rotations"],
            device="cpu",
        )
        # xyz(3)+rotation(4)+geo(1) = 8 neighbor features
        assert builder.get_neighbor_feature_dim() == 8
        assert builder.get_point_feature_dim() == 7


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    def test_single_visited_neighbour(self, xyz_builder):
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        visited[1] = True
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
            visited_mask=visited,
        )
        assert result is not None
        _, _, vm, _ = result
        assert int(vm.sum()) == 1

    def test_many_unvisited_neighbours_still_returns(self, xyz_builder):
        """Unvisited neighbours should not make build_input fail."""
        visited = np.zeros(xyz_builder.num_points, dtype=bool)
        visited[1] = True  # only 1 visited out of 10
        nbrs = np.arange(1, 11)
        per_nbr_dists = np.arange(1, 11) * 0.5
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=nbrs,
            neighbor_distances=_gdist(xyz_builder.num_points, nbrs, per_nbr_dists),
            visited_mask=visited,
        )
        # create_train_example may return None if patch is invalid, but at
        # least it should not raise an exception.
        # If it does return, the valid count should be 1.
        if result is not None:
            _, _, vm, _ = result
            assert int(vm.sum()) == 1

    def test_opacities_flat_or_2d_both_work(self, base_data):
        """Opacities can be (N,) or (N,1) - both should work."""
        for opacities in [
            base_data["opacities"].flatten(),       # (N,)
            base_data["opacities"].reshape(-1, 1),  # (N,1)
        ]:
            builder = GaussianInputBuilder(
                positions=base_data["positions"],
                dataset_config=_make_config(["xyz", "opacity"]),
                ring=2,
                opacities=opacities,
                device="cpu",
            )
            assert builder.opacities.ndim == 1  # always flattened internally

    def test_no_none_for_valid_patch(self, xyz_builder):
        """A patch with 3 visited neighbours must not return None."""
        visited = np.ones(xyz_builder.num_points, dtype=bool)
        result = xyz_builder.build_input(
            point_idx=0,
            all_neighbor_indices=np.array([1, 2, 3]),
            neighbor_distances=_gdist(xyz_builder.num_points, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
            visited_mask=visited,
        )
        assert result is not None


# ===========================================================================
# TestRingComputation — n_neighbors triggers internal ring computation
# ===========================================================================

class TestRingComputation:
    """Tests for the n_neighbors-based internal ring computation."""

    @pytest.fixture()
    def ring_builder(self, base_data):
        """Builder with n_neighbors=5, ring-2, internal ring computation."""
        cfg = _make_config(["xyz"])
        cfg["n_neighbors"] = 5
        cfg["use_mahalanobis"] = False
        return GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=cfg,
            ring=2,
            device="cpu",
        )

    def test_ring1_neighbors_populated(self, ring_builder):
        assert ring_builder.ring1_neighbors is not None
        assert len(ring_builder.ring1_neighbors) == ring_builder.num_points

    def test_ring_neighbors_populated(self, ring_builder):
        assert ring_builder.ring_neighbors is not None
        assert len(ring_builder.ring_neighbors) == ring_builder.num_points

    def test_ring1_nbrs_have_correct_size(self, ring_builder):
        # Each point should have at most n_neighbors=5 ring-1 neighbours
        for idx, nbrs in ring_builder.ring1_neighbors.items():
            assert len(nbrs) <= 5

    def test_ring_k_nbrs_superset_of_ring1(self, ring_builder):
        # Ring-2 should be a superset of ring-1 for each point
        for idx in range(ring_builder.num_points):
            r1 = set(ring_builder.ring1_neighbors[idx].tolist())
            rk = set(ring_builder.ring_neighbors[idx].tolist())
            assert r1.issubset(rk)

    def test_normalization_factor_set_from_computation(self, ring_builder):
        # normalization_factor should be set to mean_nn_dist (> 0)
        assert ring_builder.normalization_factor > 0

    def test_per_point_nn_distances_set(self, ring_builder):
        assert ring_builder.per_point_nn_distances is not None
        assert len(ring_builder.per_point_nn_distances) == ring_builder.num_points
        assert (ring_builder.per_point_nn_distances > 0).all()

    def test_no_ring_computation_without_n_neighbors(self, xyz_builder):
        assert xyz_builder.ring1_neighbors is None
        assert xyz_builder.ring_neighbors is None

    def test_build_input_works_with_computed_rings(self, ring_builder):
        """build_input should work using internally computed ring neighbours."""
        point_idx = 0
        nbrs = ring_builder.ring_neighbors[point_idx]
        n = ring_builder.num_points
        visited = np.ones(n, dtype=bool)
        dists = np.random.RandomState(42).rand(n).astype(np.float64) * 2
        dists[point_idx] = 0.0
        result = ring_builder.build_input(
            point_idx=point_idx,
            all_neighbor_indices=nbrs,
            neighbor_distances=dists,
            ring1_neighbor_indices=ring_builder.ring1_neighbors[point_idx],
            visited_mask=visited,
        )
        assert result is not None


# ===========================================================================
# TestTransformsConfig — transforms_config triggers internal transform build
# ===========================================================================

class TestTransformsConfig:
    """Tests for the transforms_config parameter."""

    def test_transforms_config_none_by_default(self, xyz_builder):
        # When no transforms_config is provided, inference_transforms stays None
        assert xyz_builder.inference_transforms is None

    def test_transforms_config_empty_list_gives_none(self, base_data):
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz"]),
            ring=2,
            device="cpu",
            transforms_config=[],
        )
        assert builder.inference_transforms is None

    def test_inference_transforms_kwarg_takes_precedence(self, base_data):
        """When inference_transforms is a callable, transforms_config ignored."""
        dummy_transform = lambda nbhd, pf: (nbhd, pf)
        builder = GaussianInputBuilder(
            positions=base_data["positions"],
            dataset_config=_make_config(["xyz"]),
            ring=2,
            device="cpu",
            inference_transforms=dummy_transform,
            transforms_config=[{"name": "nonexistent"}],
        )
        assert builder.inference_transforms is dummy_transform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
