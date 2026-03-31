"""
Tests for utils/misc.py — FPS utilities for Gaussian splatting data.

Works with or without CUDA (CPU fallback is used when CUDA is unavailable).
"""

import pytest
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Import utils/misc.py from the project root directly, to avoid confusion
# with DataSets/utils/ (which is a different package).
_misc_path = Path(__file__).resolve().parent.parent.parent / "utils" / "misc.py"
_spec = importlib.util.spec_from_file_location("utils_misc", str(_misc_path))
_misc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_misc)

fps_gs = _misc.fps_gs
fps_downsample_gaussians = _misc.fps_downsample_gaussians
_pack_attributes = _misc._pack_attributes
_to_2d = _misc._to_2d
get_attribute_size = _misc.get_attribute_size


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def point_cloud(rng):
    """100 random 3D points."""
    return rng.randn(100, 3).astype(np.float32)


@pytest.fixture
def gaussian_attrs(rng):
    """Full set of Gaussian attributes for 100 points."""
    N = 100
    return {
        "positions": rng.randn(N, 3).astype(np.float32),
        "scales": rng.randn(N, 3).astype(np.float32),
        "rotations": rng.randn(N, 4).astype(np.float32),
        "opacities": rng.rand(N, 1).astype(np.float32),
        "sh_features": rng.randn(N, 3).astype(np.float32),
        "normals": rng.randn(N, 3).astype(np.float32),
    }


# ── _to_2d ───────────────────────────────────────────────────────────────────

class TestTo2d:
    def test_1d_becomes_column(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_2d(arr)
        assert result.shape == (3, 1)
        assert result.dtype == np.float32

    def test_2d_passthrough(self):
        arr = np.zeros((5, 3), dtype=np.float32)
        result = _to_2d(arr)
        assert result.shape == (5, 3)

    def test_trailing_dim_squeezed(self):
        arr = np.zeros((4, 1, 1), dtype=np.float32)
        result = _to_2d(arr)
        assert result.shape == (4, 1)


# ── _pack_attributes ─────────────────────────────────────────────────────────

class TestPackAttributes:
    def test_xyz_only(self, point_cloud):
        packed = _pack_attributes(["xyz"], point_cloud)
        assert packed.shape == (100, 3)
        np.testing.assert_array_equal(packed, point_cloud)

    def test_xyz_scale(self, gaussian_attrs):
        packed = _pack_attributes(
            ["xyz", "scale"],
            gaussian_attrs["positions"],
            scales=gaussian_attrs["scales"],
        )
        assert packed.shape == (100, 6)  # 3 + 3

    def test_all_attrs(self, gaussian_attrs):
        packed = _pack_attributes(
            ["xyz", "scale", "rotation", "opacity", "sh", "normals"],
            gaussian_attrs["positions"],
            scales=gaussian_attrs["scales"],
            rotations=gaussian_attrs["rotations"],
            opacities=gaussian_attrs["opacities"],
            sh_features=gaussian_attrs["sh_features"],
            normals=gaussian_attrs["normals"],
        )
        # 3 + 3 + 4 + 1 + 3 + 3 = 17
        assert packed.shape == (100, 17)

    def test_missing_required_attr_raises(self, point_cloud):
        with pytest.raises(ValueError, match="scales required"):
            _pack_attributes(["xyz", "scale"], point_cloud)

    def test_unknown_attr_raises(self, point_cloud):
        with pytest.raises(ValueError, match="Unknown attribute"):
            _pack_attributes(["xyz", "foobar"], point_cloud)


# ── get_attribute_size ────────────────────────────────────────────────────────

class TestGetAttributeSize:
    def test_known_sizes(self):
        assert get_attribute_size("xyz") == 3
        assert get_attribute_size("rotation") == 4
        assert get_attribute_size("opacity") == 1

    def test_unknown_defaults_to_1(self):
        assert get_attribute_size("nonexistent") == 1


# ── fps_gs (basic) ───────────────────────────────────────────────────────────

class TestFpsGs:
    def test_basic_downsample(self, point_cloud):
        idx = fps_gs(point_cloud, n=20)
        assert idx.shape == (20,)
        assert idx.dtype == np.int64
        assert len(np.unique(idx)) == 20  # all unique
        assert np.all(idx >= 0) and np.all(idx < 100)

    def test_n_equal_N_returns_all(self, point_cloud):
        idx = fps_gs(point_cloud, n=100)
        assert len(idx) == 100
        np.testing.assert_array_equal(np.sort(idx), np.arange(100))

    def test_n_greater_than_N_returns_all(self, point_cloud):
        idx = fps_gs(point_cloud, n=200)
        assert len(idx) == 100
        np.testing.assert_array_equal(np.sort(idx), np.arange(100))

    def test_multi_attr_fps(self, gaussian_attrs):
        idx = fps_gs(
            gaussian_attrs["positions"],
            n=30,
            attributes=["xyz", "scale"],
            scales=gaussian_attrs["scales"],
        )
        assert idx.shape == (30,)
        assert len(np.unique(idx)) == 30

    def test_single_point(self):
        pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        idx = fps_gs(pos, n=1)
        assert len(idx) == 1
        assert idx[0] == 0


# ── fps_gs with protected_indices ────────────────────────────────────────────

class TestFpsGsProtected:
    def test_protected_always_present(self, point_cloud):
        protected = np.array([0, 10, 50, 99])
        idx = fps_gs(point_cloud, n=20, protected_indices=protected)
        assert idx.shape == (20,)
        for p in protected:
            assert p in idx, f"Protected index {p} missing from output"

    def test_protected_count_equals_n(self, point_cloud):
        """When protected count == n, only protected indices are returned."""
        protected = np.array([5, 15, 25])
        idx = fps_gs(point_cloud, n=3, protected_indices=protected)
        np.testing.assert_array_equal(np.sort(idx), np.sort(protected))

    def test_protected_exceeds_n(self, point_cloud):
        """When there are more protected points than n, return first n."""
        protected = np.array([1, 2, 3, 4, 5])
        idx = fps_gs(point_cloud, n=3, protected_indices=protected)
        assert len(idx) == 3
        for p in idx:
            assert p in protected

    def test_protected_empty_falls_through(self, point_cloud):
        """Empty protected array = standard FPS."""
        idx_no_protect = fps_gs(point_cloud, n=20)
        idx_empty = fps_gs(point_cloud, n=20, protected_indices=np.array([]))
        # Both should return 20 points (content may differ due to seed)
        assert idx_no_protect.shape == idx_empty.shape

    def test_protected_duplicates_handled(self, point_cloud):
        """Duplicate protected indices are de-duplicated."""
        protected = np.array([0, 0, 10, 10, 10])
        idx = fps_gs(point_cloud, n=20, protected_indices=protected)
        assert len(idx) == 20
        assert 0 in idx
        assert 10 in idx

    def test_all_non_protected_taken(self, point_cloud):
        """When n - n_protected > remaining non-protected points, take all."""
        # Protect 90 points, ask for 95 → need 5 from 10 remaining, but
        # let's test the edge: protect 95, ask for 100 → need 5 from 5 remaining → take all
        protected = np.arange(95)
        idx = fps_gs(point_cloud, n=100, protected_indices=protected)
        assert len(idx) == 100  # 95 protected + 5 remaining = 100
        np.testing.assert_array_equal(np.sort(idx), np.arange(100))

    def test_protected_output_sorted(self, point_cloud):
        """Output indices are sorted."""
        protected = np.array([99, 50, 0])
        idx = fps_gs(point_cloud, n=20, protected_indices=protected)
        np.testing.assert_array_equal(idx, np.sort(idx))


# ── fps_downsample_gaussians ─────────────────────────────────────────────────

class TestFpsDownsampleGaussians:
    def test_basic_dict_keys(self, gaussian_attrs):
        result = fps_downsample_gaussians(
            gaussian_attrs["positions"],
            n=30,
            scales=gaussian_attrs["scales"],
            rotations=gaussian_attrs["rotations"],
        )
        assert "indices" in result
        assert "positions" in result
        assert "scales" in result
        assert "rotations" in result
        assert "opacities" not in result  # not provided
        assert result["positions"].shape == (30, 3)
        assert result["scales"].shape == (30, 3)

    def test_positions_match_indices(self, gaussian_attrs):
        result = fps_downsample_gaussians(
            gaussian_attrs["positions"], n=20
        )
        expected = gaussian_attrs["positions"][result["indices"]]
        np.testing.assert_array_equal(result["positions"], expected)

    def test_with_protected(self, gaussian_attrs):
        protected = np.array([0, 50, 99])
        result = fps_downsample_gaussians(
            gaussian_attrs["positions"],
            n=20,
            protected_indices=protected,
        )
        for p in protected:
            assert p in result["indices"]

    def test_all_optional_arrays(self, gaussian_attrs):
        result = fps_downsample_gaussians(
            gaussian_attrs["positions"],
            n=25,
            scales=gaussian_attrs["scales"],
            rotations=gaussian_attrs["rotations"],
            opacities=gaussian_attrs["opacities"],
            sh_features=gaussian_attrs["sh_features"],
            normals=gaussian_attrs["normals"],
        )
        assert result["opacities"].shape[0] == 25
        assert result["sh_features"].shape[0] == 25
        assert result["normals"].shape[0] == 25
