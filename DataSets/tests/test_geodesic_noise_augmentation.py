#!/usr/bin/env python3
"""
Tests for GeodesicNoiseAugmentation transform.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_transformation import GeodesicNoiseAugmentation


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def single_example():
    """Single (max_neighbors, entry_size) neighborhood with known geodesics."""
    max_neighbors = 32
    entry_size = 12  # xyz(3) + normals(3) + opacity(1) + rotation(4) + geodesic(1)
    point_feature_size = 11

    neighborhood = torch.randn(max_neighbors, entry_size)
    # Set geodesic distances (last col) to increasing positive values
    neighborhood[:, -1] = torch.linspace(0.1, 1.0, max_neighbors)
    # Mask last 5 entries
    neighborhood[-5:, -1] = -10.0

    point_features = torch.randn(point_feature_size)
    return neighborhood, point_features


@pytest.fixture
def batch_example():
    """Batch of (bsize, max_neighbors, entry_size) with known geodesics."""
    bsize = 4
    max_neighbors = 32
    entry_size = 12
    point_feature_size = 11

    neighborhood = torch.randn(bsize, max_neighbors, entry_size)
    for i in range(bsize):
        neighborhood[i, :, -1] = torch.linspace(0.1, 1.0, max_neighbors)
        neighborhood[i, -5:, -1] = -10.0

    point_features = torch.randn(bsize, point_feature_size)
    return neighborhood, point_features


# ── Shape preservation ───────────────────────────────────────────────────


class TestGeodesicNoiseShape:
    def test_single_shape_preserved(self, single_example):
        neighborhood, point_features = single_example
        t = GeodesicNoiseAugmentation(max_noise=0.1, p=1.0)
        out_n, out_p = t(neighborhood.clone(), point_features.clone())
        assert out_n.shape == neighborhood.shape
        assert out_p.shape == point_features.shape

    def test_batch_shape_preserved(self, batch_example):
        neighborhood, point_features = batch_example
        t = GeodesicNoiseAugmentation(max_noise=0.1, p=1.0)
        out_n, out_p = t(neighborhood.clone(), point_features.clone())
        assert out_n.shape == neighborhood.shape
        assert out_p.shape == point_features.shape


# ── Masked entries unchanged ─────────────────────────────────────────────


class TestGeodesicNoiseMasking:
    def test_masked_entries_unchanged(self, single_example):
        """Masked geodesic values (-10) must never be modified."""
        neighborhood, point_features = single_example
        t = GeodesicNoiseAugmentation(max_noise=0.5, p=1.0)

        orig_geo = neighborhood[:, -1].clone()
        masked_idx = orig_geo == -10.0

        out_n, _ = t(neighborhood.clone(), point_features.clone())

        assert torch.all(out_n[masked_idx, -1] == -10.0), \
            "Masked entries were modified"

    def test_masked_entries_batch(self, batch_example):
        neighborhood, point_features = batch_example
        t = GeodesicNoiseAugmentation(max_noise=0.5, p=1.0)
        orig_geo = neighborhood[:, :, -1].clone()
        masked = orig_geo == -10.0

        out_n, _ = t(neighborhood.clone(), point_features.clone())
        assert torch.all(out_n[:, :, -1][masked] == -10.0)


# ── Non-geodesic features unchanged ─────────────────────────────────────


class TestGeodesicNoiseNonGeoUnchanged:
    def test_non_geo_features_unchanged(self, single_example):
        """Only the last column (geodesic) should change."""
        neighborhood, point_features = single_example
        t = GeodesicNoiseAugmentation(max_noise=0.5, p=1.0)
        orig = neighborhood.clone()

        out_n, out_p = t(neighborhood.clone(), point_features.clone())

        # All columns except last should be identical
        assert torch.allclose(out_n[:, :-1], orig[:, :-1]), \
            "Non-geodesic features were modified"
        # point_features should be unchanged
        assert torch.allclose(out_p, point_features)


# ── Noise magnitude ─────────────────────────────────────────────────────


class TestGeodesicNoiseMagnitude:
    def test_additive_offset_within_bounds(self, single_example):
        """Offset should be bounded by max_noise * geo_range."""
        neighborhood, point_features = single_example
        max_noise = 0.2
        t = GeodesicNoiseAugmentation(max_noise=max_noise, p=1.0)

        orig_geo = neighborhood[:, -1].clone()
        valid = orig_geo != -10.0
        geo_range = orig_geo[valid].max() - orig_geo[valid].min()

        out_n, _ = t(neighborhood.clone(), point_features.clone())
        new_geo = out_n[:, -1]

        # All valid entries should shift by the same offset (additive per-example)
        offsets = new_geo[valid] - orig_geo[valid]
        # All offsets should be (approximately) equal
        assert torch.allclose(offsets, offsets[0].expand_as(offsets), atol=1e-6), \
            f"Offsets are not uniform: std={offsets.std()}"
        # Magnitude bounded by max_noise * geo_range
        assert offsets[0].abs() <= max_noise * geo_range + 1e-6, \
            f"Offset {offsets[0]} exceeds bound {max_noise * geo_range}"

    def test_uniform_shift_across_neighbors(self, single_example):
        """All valid neighbors should receive the same additive offset."""
        neighborhood, point_features = single_example
        t = GeodesicNoiseAugmentation(max_noise=0.3, p=1.0)

        np.random.seed(42)
        torch.manual_seed(42)
        orig = neighborhood[:, -1].clone()
        valid = orig != -10.0

        out_n, _ = t(neighborhood.clone(), point_features.clone())
        diffs = out_n[:, -1][valid] - orig[valid]
        # All diffs should be identical (same offset applied to all)
        assert diffs.std() < 1e-6, f"Offsets have std={diffs.std()}, expected ~0"
        assert diffs.abs().sum() > 0, "No noise applied at all"


# ── Probability gating ──────────────────────────────────────────────────


class TestGeodesicNoiseProbability:
    def test_p_zero_no_change(self, single_example):
        """With p=0 the transform should be a no-op."""
        neighborhood, point_features = single_example
        t = GeodesicNoiseAugmentation(max_noise=0.5, p=0.0)
        orig = neighborhood.clone()
        out_n, _ = t(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_n, orig)

    def test_p_one_always_applies(self, batch_example):
        """With p=1 every example should be modified (statistically)."""
        neighborhood, point_features = batch_example
        t = GeodesicNoiseAugmentation(max_noise=0.5, p=1.0)
        orig = neighborhood.clone()
        out_n, _ = t(neighborhood.clone(), point_features.clone())
        # At least one batch entry should differ
        valid = orig[:, :, -1] != -10.0
        changed = (out_n[:, :, -1] != orig[:, :, -1]) & valid
        assert changed.any(), "p=1.0 but nothing changed"


# ── Constructor validation ───────────────────────────────────────────────


class TestGeodesicNoiseInit:
    def test_rejects_zero_noise(self):
        with pytest.raises(AssertionError):
            GeodesicNoiseAugmentation(max_noise=0.0)

    def test_rejects_noise_ge_one(self):
        with pytest.raises(AssertionError):
            GeodesicNoiseAugmentation(max_noise=1.0)

    def test_rejects_negative_noise(self):
        with pytest.raises(AssertionError):
            GeodesicNoiseAugmentation(max_noise=-0.1)


# ── All-masked neighborhood ─────────────────────────────────────────────


class TestGeodesicNoiseEdgeCases:
    def test_all_masked(self):
        """If every neighbour is masked, the transform should be a no-op."""
        max_neighbors = 16
        entry_size = 8
        n = torch.randn(max_neighbors, entry_size)
        n[:, -1] = -10.0  # all masked
        pf = torch.randn(7)

        t = GeodesicNoiseAugmentation(max_noise=0.3, p=1.0)
        out_n, _ = t(n.clone(), pf.clone())
        assert torch.allclose(out_n, n)

    def test_single_valid_neighbor(self):
        """With exactly one valid neighbour, geo_range falls back to abs(geo)."""
        max_neighbors = 8
        entry_size = 4
        n = torch.randn(max_neighbors, entry_size)
        n[:, -1] = -10.0
        n[0, -1] = 0.5  # only this one is valid
        pf = torch.randn(3)

        t = GeodesicNoiseAugmentation(max_noise=0.3, p=1.0)
        out_n, _ = t(n.clone(), pf.clone())
        # Masked entries still -10
        assert torch.all(out_n[1:, -1] == -10.0)
        # Valid entry may have changed (geo_range = max - min = 0,
        # falls back to abs(0.5) = 0.5, offset up to 0.3*0.5 = 0.15)
        assert out_n[0, -1] != -10.0
        assert abs(out_n[0, -1].item() - 0.5) <= 0.15 + 1e-6


# ── Registry integration ────────────────────────────────────────────────


class TestGeodesicNoiseRegistry:
    def test_registered_in_const(self):
        """GeodesicNoiseAugmentation must be in TRANSFORM_REGISTRY."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from models.const import TRANSFORM_REGISTRY
        assert "GeodesicNoiseAugmentation" in TRANSFORM_REGISTRY

    def test_build_from_config(self):
        """build_transforms should instantiate from a YAML-style config."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from models.const import build_transforms
        cfg = [{"name": "GeodesicNoiseAugmentation", "max_noise": 0.15, "p": 0.8}]
        t = build_transforms(cfg, attributes=["xyz", "normals", "opacity", "rotation"])
        assert t is not None
        # Verify it's callable
        n = torch.randn(16, 12)
        n[:, -1] = torch.linspace(0.1, 1.0, 16)
        pf = torch.randn(11)
        out_n, out_p = t(n, pf)
        assert out_n.shape == n.shape
