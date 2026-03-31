#!/usr/bin/env python3
"""Tests for load_utils.py - Gaussian data loading and processing."""

import sys
from pathlib import Path
import numpy as np
import pytest
from io import StringIO

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.load_utils import GaussianDataCPU


class TestGaussianDataCPU:
    """Tests for GaussianDataCPU class."""
    
    def setup_method(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.n_points = 100
        
        # Sample positions
        self.xyz = np.random.randn(self.n_points, 3).astype(np.float32)
        
        # Sample scales (log space)
        self.scales = np.random.randn(self.n_points, 3).astype(np.float32)
        
        # Sample rotations (not normalized)
        self.rotations = np.random.randn(self.n_points, 4).astype(np.float32)
        
        # Sample opacities (logit space)
        self.opacities = np.random.randn(self.n_points, 1).astype(np.float32)
        
        # Sample SH features
        self.features_dc = np.random.randn(self.n_points, 3, 1).astype(np.float32)
        self.features_rest = np.random.randn(self.n_points, 3, 15).astype(np.float32)  # SH degree 3 = 16 coeffs, -1 DC = 15
        
    def test_initialization(self):
        """Test GaussianDataCPU initialization."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            self.features_dc, self.features_rest
        )
        
        assert gaussian_data._xyz.shape == (self.n_points, 3)
        assert gaussian_data._scaling.shape == (self.n_points, 3)
        assert gaussian_data._rotation.shape == (self.n_points, 4)
        assert gaussian_data._opacity.shape == (self.n_points, 1)
        assert gaussian_data._features_dc.shape == (self.n_points, 3, 1)
        assert gaussian_data._features_rest.shape == (self.n_points, 3, 15)
    
    def test_get_xyz(self):
        """Test get_xyz returns unchanged positions."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities
        )
        
        xyz = gaussian_data.get_xyz()
        np.testing.assert_array_equal(xyz, self.xyz)
    
    def test_get_scaling(self):
        """Test get_scaling applies exp activation."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities
        )
        
        scales = gaussian_data.get_scaling()
        expected_scales = np.exp(self.scales)
        
        np.testing.assert_allclose(scales, expected_scales, rtol=1e-6)
        assert np.all(scales > 0), "All scales should be positive after exp"
    
    def test_get_rotation_normalization(self):
        """Test get_rotation normalizes quaternions."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities
        )
        
        rotations = gaussian_data.get_rotation()
        
        # Check normalization
        norms = np.linalg.norm(rotations, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)
    
    def test_get_rotation_positive_first_component(self):
        """Test get_rotation ensures first component is positive."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities
        )
        
        rotations = gaussian_data.get_rotation()
        
        # Check first component is positive
        assert np.all(rotations[:, 0] >= 0), "First component of quaternions should be non-negative"
    
    def test_get_opacity(self):
        """Test get_opacity applies sigmoid activation."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities
        )
        
        opacities = gaussian_data.get_opacity()
        expected_opacities = 1.0 / (1.0 + np.exp(-self.opacities))
        
        np.testing.assert_allclose(opacities, expected_opacities, rtol=1e-6)
        assert np.all((opacities >= 0) & (opacities <= 1)), "Opacities should be in [0, 1]"
    
    def test_get_features_dc(self):
        """Test get_features_dc returns DC component."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            self.features_dc, self.features_rest
        )
        
        features_dc = gaussian_data.get_features_dc()
        
        # Should squeeze last dimension
        assert features_dc.shape == (self.n_points, 3)
        np.testing.assert_array_equal(features_dc, self.features_dc.squeeze(-1))
    
    def test_get_features_dc_2d_input(self):
        """Test get_features_dc with 2D input."""
        features_dc_2d = np.random.randn(self.n_points, 3).astype(np.float32)
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            features_dc_2d, None
        )
        
        features_dc = gaussian_data.get_features_dc()
        assert features_dc.shape == (self.n_points, 3)
        np.testing.assert_array_equal(features_dc, features_dc_2d)
    
    def test_get_features_rest(self):
        """Test get_features_rest returns rest coefficients."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            self.features_dc, self.features_rest
        )
        
        features_rest = gaussian_data.get_features_rest()
        assert features_rest.shape == (self.n_points, 3, 15)
        np.testing.assert_array_equal(features_rest, self.features_rest)
    
    def test_get_features_with_rest(self):
        """Test get_features concatenates DC and rest."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            self.features_dc, self.features_rest
        )
        
        features = gaussian_data.get_features()
        
        # Should concatenate DC (1) + rest (15) = 16 coefficients
        assert features.shape == (self.n_points, 3, 16)
        
        # Check DC component
        np.testing.assert_array_equal(features[:, :, 0:1], self.features_dc)
        
        # Check rest components
        np.testing.assert_array_equal(features[:, :, 1:], self.features_rest)
    
    def test_get_features_dc_only(self):
        """Test get_features with DC only (no rest)."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            self.features_dc, None
        )
        
        features = gaussian_data.get_features()
        assert features.shape == (self.n_points, 3, 1)
        np.testing.assert_array_equal(features, self.features_dc)
    
    def test_no_sh_features(self):
        """Test when no SH features are provided."""
        gaussian_data = GaussianDataCPU(
            self.xyz, self.scales, self.rotations, self.opacities,
            None, None
        )
        
        assert gaussian_data.get_features_dc() is None
        assert gaussian_data.get_features_rest() is None
        assert gaussian_data.get_features() is None


class TestActivationFunctions:
    """Tests for activation function properties."""
    
    def test_exp_activation_properties(self):
        """Test exp activation produces valid scales."""
        scales_logspace = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        gaussian_data = GaussianDataCPU(
            np.zeros((2, 3)), scales_logspace, np.zeros((2, 4)), np.zeros((2, 1))
        )
        
        scales = gaussian_data.get_scaling()
        
        # Check positivity
        assert np.all(scales > 0)
        
        # Check specific values
        np.testing.assert_allclose(scales[0], np.exp(scales_logspace[0]), rtol=1e-6)
        np.testing.assert_allclose(scales[1], np.exp(scales_logspace[1]), rtol=1e-6)
    
    def test_sigmoid_activation_properties(self):
        """Test sigmoid activation produces valid opacities."""
        opacities_logit = np.array([[-5.0], [0.0], [5.0]], dtype=np.float32)
        gaussian_data = GaussianDataCPU(
            np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 4)), opacities_logit
        )
        
        opacities = gaussian_data.get_opacity()
        
        # Check range [0, 1]
        assert np.all(opacities >= 0)
        assert np.all(opacities <= 1)
        
        # Check sigmoid properties
        assert opacities[0] < 0.01, "Large negative logit should give low opacity"
        np.testing.assert_allclose(opacities[1], 0.5, rtol=1e-6, atol=1e-6)
        assert opacities[2] > 0.99, "Large positive logit should give high opacity"
    
    def test_quaternion_normalization_properties(self):
        """Test quaternion normalization produces unit quaternions."""
        # Test various scales
        rotations = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Already normalized
            [2.0, 0.0, 0.0, 0.0],  # Scaled up
            [0.5, 0.5, 0.5, 0.5],  # Non-unit
            [-1.0, 0.0, 0.0, 0.0],  # Negative first component
        ], dtype=np.float32)
        
        gaussian_data = GaussianDataCPU(
            np.zeros((4, 3)), np.zeros((4, 3)), rotations, np.zeros((4, 1))
        )
        
        normalized_rots = gaussian_data.get_rotation()
        
        # Check all are unit quaternions
        norms = np.linalg.norm(normalized_rots, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)
        
        # Check all have positive first component
        assert np.all(normalized_rots[:, 0] >= 0)
        
        # Check last one was flipped
        np.testing.assert_allclose(normalized_rots[3], [1.0, 0.0, 0.0, 0.0], rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_rotations(self):
        """Test handling of zero quaternions (edge case)."""
        rotations = np.zeros((10, 4), dtype=np.float32)
        gaussian_data = GaussianDataCPU(
            np.zeros((10, 3)), np.zeros((10, 3)), rotations, np.zeros((10, 1))
        )
        
        # Should handle division by near-zero gracefully
        normalized_rots = gaussian_data.get_rotation()
        
        # Result might be undefined, but shouldn't crash
        assert normalized_rots.shape == (10, 4)
    
    def test_extreme_opacity_values(self):
        """Test sigmoid with extreme values."""
        opacities = np.array([[-100.0], [-10.0], [0.0], [10.0], [100.0]], dtype=np.float32)
        gaussian_data = GaussianDataCPU(
            np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 4)), opacities
        )
        
        result = gaussian_data.get_opacity()
        
        # Should not overflow/underflow
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0) & (result <= 1))
    
    def test_extreme_scale_values(self):
        """Test exp activation with extreme values."""
        scales = np.array([[-20.0, -10.0, 0.0], [10.0, 20.0, 30.0]], dtype=np.float32)
        gaussian_data = GaussianDataCPU(
            np.zeros((2, 3)), scales, np.zeros((2, 4)), np.zeros((2, 1))
        )
        
        result = gaussian_data.get_scaling()
        
        # Should not overflow/underflow
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
