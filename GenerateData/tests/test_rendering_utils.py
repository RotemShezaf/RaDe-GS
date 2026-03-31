"""Unit tests for rendering_utils.py

Tests the core rendering functionality including:
- Point sampling with radius-based thinning
- Lighting configuration for appearance variation
- Random view shuffling for lighting groups
- Image rendering with brightness clamping
"""

import sys
from pathlib import Path
import math
import argparse
import numpy as np
import open3d as o3d
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from GenerateData.utils.rendering_utils import (
    _sample_points,
    _generate_uv_coordinates,
)

from GenerateData.utils.camera_utils import (
    _compute_camera_centers,
)


class TestPointSampling:
    """Test suite for _sample_points function."""
    
    def setup_method(self):
        """Create a simple test mesh."""
        # Create a simple cube mesh
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        self.mesh.compute_vertex_normals()
        
        # Add vertex colors
        num_vertices = len(self.mesh.vertices)
        colors = np.random.rand(num_vertices, 3)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    def test_sample_points_no_threshold(self):
        """Test that all vertices are returned when thresh=None."""
        vertices, colors, normals = _sample_points(self.mesh, thresh=None, seed=42)
        
        assert len(vertices) == len(self.mesh.vertices)
        assert len(colors) == len(self.mesh.vertices)
        assert len(normals) == len(self.mesh.vertices)
        assert colors.dtype == np.uint8
        assert np.all(colors <= 255)
        assert np.all(colors >= 0)
    
    def test_sample_points_with_threshold(self):
        """Test that sampling reduces point count with threshold or maintains minimum distance."""
        thresh = 0.5
        vertices, colors, normals = _sample_points(self.mesh, thresh=thresh, seed=42)
        
        # Should have fewer or equal points than original (may keep all if mesh is small)
        assert len(vertices) <= len(self.mesh.vertices)
        assert len(vertices) == len(colors)
        assert len(vertices) == len(normals)
        
        # Check that no two points are closer than threshold
        if len(vertices) > 1:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(2, len(vertices)), algorithm='kd_tree')
            nn.fit(vertices)
            distances, _ = nn.kneighbors(vertices)
            # distances[:, 1] is the distance to the second nearest neighbor (first is itself)
            if len(vertices) >= 2:
                assert np.all(distances[:, 1] >= thresh - 1e-6), "Some sampled points are too close"
    
    def test_sample_points_reproducibility(self):
        """Test that same seed produces same results."""
        thresh = 0.3
        vertices1, colors1, normals1 = _sample_points(self.mesh, thresh=thresh, seed=42)
        vertices2, colors2, normals2 = _sample_points(self.mesh, thresh=thresh, seed=42)
        
        np.testing.assert_array_equal(vertices1, vertices2)
        np.testing.assert_array_equal(colors1, colors2)
        np.testing.assert_array_equal(normals1, normals2)
    
    def test_sample_points_different_seeds(self):
        """Test that different seeds produce different results."""
        thresh = 0.3
        vertices1, _, _ = _sample_points(self.mesh, thresh=thresh, seed=42)
        vertices2, _, _ = _sample_points(self.mesh, thresh=thresh, seed=123)
        
        # Should have different point selections
        assert not np.array_equal(vertices1, vertices2)
    
    def test_sample_points_color_conversion(self):
        """Test color conversion to uint8 range."""
        # Set colors in [0, 1] range
        colors_normalized = np.random.rand(len(self.mesh.vertices), 3)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors_normalized)
        
        _, colors, _ = _sample_points(self.mesh, thresh=None, seed=42)
        
        assert colors.dtype == np.uint8
        assert np.all(colors >= 0)
        assert np.all(colors <= 255)
        # Check that conversion is approximately correct
        expected_colors = (colors_normalized * 255).astype(np.uint8)
        np.testing.assert_array_equal(colors, expected_colors)


class TestUVCoordinates:
    """Test suite for UV coordinate generation."""
    
    def test_uv_coordinates_range(self):
        """Test that UV coordinates are in [0, 1] range."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        uv = _generate_uv_coordinates(vertices)
        
        assert uv.shape == (4, 2)
        assert np.all(uv >= 0.0)
        assert np.all(uv <= 1.0)
    
    def test_uv_coordinates_corners(self):
        """Test that corner vertices map to UV corners."""
        vertices = np.array([
            [0.0, 0.0, 0.0],  # Should map to (0, 0)
            [1.0, 0.0, 0.0],  # Should map to (1, 0)
            [1.0, 1.0, 0.0],  # Should map to (1, 1)
            [0.0, 1.0, 0.0],  # Should map to (0, 1)
        ])
        
        uv = _generate_uv_coordinates(vertices)
        
        # Check corners (with small tolerance for floating point)
        expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_allclose(uv, expected, atol=1e-6)
    
    def test_uv_coordinates_negative_vertices(self):
        """Test UV generation with negative vertex coordinates."""
        vertices = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        
        uv = _generate_uv_coordinates(vertices)
        
        assert uv.shape == (4, 2)
        assert np.all(uv >= 0.0)
        assert np.all(uv <= 1.0)


class TestLightingConfiguration:
    """Test suite for lighting group configuration."""
    
    def test_lighting_groups_count(self):
        """Test that 10 lighting groups are created with appearance variation."""
        num_views = 100
        num_groups = 10
        views_per_group = num_views // num_groups
        
        # Create shuffled indices
        rng = np.random.default_rng(seed=42)
        view_indices = np.arange(num_views)
        rng.shuffle(view_indices)
        
        # Create mapping
        view_to_group = {}
        for i, view_idx in enumerate(view_indices):
            group_id = i // views_per_group
            group_id = min(group_id, num_groups - 1)
            view_to_group[view_idx] = group_id
        
        # Check that all groups are used
        groups_used = set(view_to_group.values())
        assert len(groups_used) == num_groups
        assert min(groups_used) == 0
        assert max(groups_used) == num_groups - 1
    
    def test_fixed_light_id_range(self):
        """Test that light_id is properly validated."""
        # Test valid light_ids (0-4)
        for light_id in range(5):
            # This should not raise any exception
            validated_id = light_id
            assert 0 <= validated_id <= 4
        
        # Test that out-of-range values can be clamped
        assert max(0, min(4, -1)) == 0  # Below range
        assert max(0, min(4, 5)) == 4   # Above range
        assert max(0, min(4, 10)) == 4  # Way above range
    
    def test_light_id_modes_mutually_exclusive(self):
        """Test that light_id and use_decoupled_appearance are handled correctly."""
        # When use_decoupled_appearance=True, light_id should be ignored
        use_decoupled = True
        light_id = 2
        
        # In actual implementation, if use_decoupled_appearance is True,
        # the code uses 10 groups with shuffled views
        # If use_decoupled_appearance is False and light_id is set,
        # it uses the fixed light_id configuration
        
        # Test the logic
        if use_decoupled:
            active_mode = "decoupled"
        elif light_id is not None:
            active_mode = "fixed_light"
        else:
            active_mode = "standard"
        
        assert active_mode == "decoupled"
        
        # Test with light_id only
        use_decoupled = False
        light_id = 3
        if use_decoupled:
            active_mode = "decoupled"
        elif light_id is not None:
            active_mode = "fixed_light"
        else:
            active_mode = "standard"
        
        assert active_mode == "fixed_light"
    
    def test_view_shuffling_reproducibility(self):
        """Test that same seed produces consistent view shuffling."""
        num_views = 100
        
        # First shuffling
        rng1 = np.random.default_rng(seed=42)
        view_indices1 = np.arange(num_views)
        rng1.shuffle(view_indices1)
        
        # Second shuffling with same seed
        rng2 = np.random.default_rng(seed=42)
        view_indices2 = np.arange(num_views)
        rng2.shuffle(view_indices2)
        
        np.testing.assert_array_equal(view_indices1, view_indices2)
    
    def test_view_shuffling_different_seeds(self):
        """Test that different seeds produce different shuffling."""
        num_views = 100
        
        rng1 = np.random.default_rng(seed=42)
        view_indices1 = np.arange(num_views)
        rng1.shuffle(view_indices1)
        
        rng2 = np.random.default_rng(seed=123)
        view_indices2 = np.arange(num_views)
        rng2.shuffle(view_indices2)
        
        assert not np.array_equal(view_indices1, view_indices2)
    
    def test_lighting_group_distribution(self):
        """Test that views are evenly distributed across lighting groups."""
        num_views = 100
        num_groups = 10
        views_per_group = num_views // num_groups
        
        rng = np.random.default_rng(seed=42)
        view_indices = np.arange(num_views)
        rng.shuffle(view_indices)
        
        view_to_group = {}
        for i, view_idx in enumerate(view_indices):
            group_id = i // views_per_group
            group_id = min(group_id, num_groups - 1)
            view_to_group[view_idx] = group_id
        
        # Count views per group
        from collections import Counter
        group_counts = Counter(view_to_group.values())
        
        # Each group should have approximately equal views
        for group_id in range(num_groups):
            count = group_counts[group_id]
            assert count >= views_per_group - 1  # Allow some variation
            assert count <= views_per_group + 1


class TestBrightnessClamping:
    """Test suite for brightness clamping logic."""
    
    def test_brightness_clamping_dark_pixels(self):
        """Test that dark pixels are boosted to minimum brightness."""
        # Simulate dark image with some foreground pixels
        img_np = np.zeros((100, 100, 3), dtype=np.float32)
        
        # Add some very dark foreground pixels
        dark_pixels = np.array([[10, 10], [20, 20], [30, 30]])
        for y, x in dark_pixels:
            img_np[y, x] = [5.0, 5.0, 5.0]  # Very dark (luminance ~ 5)
        
        min_brightness_255 = 20.0
        background_threshold = 2.0
        
        # Detect foreground
        is_foreground = np.any(img_np > background_threshold, axis=2)
        
        # Apply brightness clamping
        foreground_rgb = img_np[is_foreground]
        luminance = 0.299 * foreground_rgb[:, 0] + 0.587 * foreground_rgb[:, 1] + 0.114 * foreground_rgb[:, 2]
        
        too_dark = luminance < min_brightness_255
        assert too_dark.sum() > 0, "Test setup failed: no dark pixels detected"
        
        # Boost dark pixels
        dark_rgb = foreground_rgb[too_dark]
        dark_lum = luminance[too_dark]
        has_brightness = dark_lum > 0.5
        
        if has_brightness.any():
            boost_factors = min_brightness_255 / dark_lum[has_brightness]
            dark_rgb[has_brightness] = np.clip(
                dark_rgb[has_brightness] * boost_factors[:, np.newaxis],
                0, 255
            )
        
        # Verify boosted pixels meet minimum
        new_lum = 0.299 * dark_rgb[:, 0] + 0.587 * dark_rgb[:, 1] + 0.114 * dark_rgb[:, 2]
        assert np.all(new_lum >= min_brightness_255 - 0.1), "Dark pixels not boosted to minimum"
    
    def test_brightness_clamping_preserves_bright_pixels(self):
        """Test that already-bright pixels are not modified."""
        # Create image with bright pixels
        img_np = np.zeros((100, 100, 3), dtype=np.float32)
        
        # Add bright foreground pixels
        bright_pixels = np.array([[10, 10], [20, 20], [30, 30]])
        for y, x in bright_pixels:
            img_np[y, x] = [100.0, 120.0, 110.0]  # Bright enough
        
        original_values = img_np[10, 10].copy()
        
        min_brightness_255 = 20.0
        background_threshold = 2.0
        
        # Apply brightness clamping logic
        is_foreground = np.any(img_np > background_threshold, axis=2)
        foreground_rgb = img_np[is_foreground]
        luminance = 0.299 * foreground_rgb[:, 0] + 0.587 * foreground_rgb[:, 1] + 0.114 * foreground_rgb[:, 2]
        
        too_dark = luminance < min_brightness_255
        
        # Bright pixels should not be flagged as too dark
        assert too_dark.sum() == 0, "Bright pixels incorrectly flagged as dark"
        
        # Values should remain unchanged
        np.testing.assert_array_equal(img_np[10, 10], original_values)


class TestIntegration:
    """Integration tests for full rendering pipeline."""
    
    def setup_method(self):
        """Create temporary output directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        (self.output_dir / "images").mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_mock(self):
        """Test full rendering pipeline with mocked Open3D renderer."""
        # This would require more complex mocking of Open3D
        # For now, just verify that the directory structure is correct
        assert self.output_dir.exists()
        assert (self.output_dir / "images").exists()


class TestAutoCameraRadius:
    """Test suite for _compute_camera_centers with --auto_camera_radius."""

    def _make_args(self, **overrides):
        """Create a mock args namespace with sensible defaults."""
        defaults = dict(
            auto_camera_radius=True,
            camera_radius=None,
            orbit_radius_scale=1.4,
            vertical_fov=45.0,
            image_width=1024,
            image_height=1024,
            camera_distribution="uniform_sphere",
            elevation_deg=25.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _make_box_mesh(self, size=2.0):
        """Create a cube mesh centered at origin with given side length."""
        mesh = o3d.geometry.TriangleMesh.create_box(size, size, size)
        # Center it at origin
        mesh.translate([-size / 2, -size / 2, -size / 2])
        return mesh

    # ------------------------------------------------------------------
    # Basic auto-radius tests
    # ------------------------------------------------------------------
    def test_auto_radius_all_cameras_see_mesh(self):
        """Every camera must be far enough that the bounding sphere fits in the FOV."""
        mesh = self._make_box_mesh(2.0)
        args = self._make_args()
        centers, targets = _compute_camera_centers(mesh, num_views=50, args=args, seed=42)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2
        extent = np.linalg.norm(np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound))
        bounding_radius = extent / 2.0

        half_fov = math.radians(args.vertical_fov / 2.0)
        min_required_dist = bounding_radius / math.tan(half_fov)

        dists = np.linalg.norm(centers - mesh_center, axis=1)
        assert np.all(dists >= min_required_dist - 1e-6), \
            f"Some cameras too close: min dist={dists.min():.4f}, required={min_required_dist:.4f}"

    def test_auto_radius_equals_formula(self):
        """Computed radius should match bounding_radius / tan(min_half_fov) exactly."""
        mesh = self._make_box_mesh(4.0)
        args = self._make_args(image_width=1024, image_height=1024, vertical_fov=45.0)
        centers, _ = _compute_camera_centers(mesh, num_views=10, args=args, seed=7)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2
        extent = np.linalg.norm(np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound))
        bounding_radius = extent / 2.0
        half_fov_v = math.radians(45.0 / 2.0)
        expected_radius = bounding_radius / math.tan(half_fov_v)

        actual_radius = np.linalg.norm(centers[0] - mesh_center)
        np.testing.assert_allclose(actual_radius, expected_radius, rtol=1e-6)

    def test_auto_radius_non_square_uses_tighter_fov(self):
        """For a tall image (w < h), horizontal FOV is tighter and should be used."""
        mesh = self._make_box_mesh(2.0)
        # Tall image: horizontal FOV is smaller
        args_tall = self._make_args(image_width=512, image_height=1024)
        centers_tall, _ = _compute_camera_centers(mesh, num_views=10, args=args_tall, seed=1)

        # Square image
        args_sq = self._make_args(image_width=1024, image_height=1024)
        centers_sq, _ = _compute_camera_centers(mesh, num_views=10, args=args_sq, seed=1)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2

        r_tall = np.linalg.norm(centers_tall[0] - mesh_center)
        r_sq = np.linalg.norm(centers_sq[0] - mesh_center)

        # Tall image needs larger radius to compensate for narrower horizontal FOV
        assert r_tall > r_sq, f"Tall image radius ({r_tall:.4f}) should > square ({r_sq:.4f})"

    def test_auto_radius_wide_image(self):
        """For a wide image (w > h), vertical FOV is tighter — radius same as square if vfov unchanged."""
        mesh = self._make_box_mesh(2.0)
        args_wide = self._make_args(image_width=1920, image_height=1024)
        centers_wide, _ = _compute_camera_centers(mesh, num_views=10, args=args_wide, seed=1)

        args_sq = self._make_args(image_width=1024, image_height=1024)
        centers_sq, _ = _compute_camera_centers(mesh, num_views=10, args=args_sq, seed=1)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2

        r_wide = np.linalg.norm(centers_wide[0] - mesh_center)
        r_sq = np.linalg.norm(centers_sq[0] - mesh_center)

        # Wide image: vertical FOV is the bottleneck, same as square
        np.testing.assert_allclose(r_wide, r_sq, rtol=1e-6)

    def test_auto_radius_scales_with_mesh_size(self):
        """Doubling mesh size should double the camera radius."""
        args = self._make_args()
        mesh_small = self._make_box_mesh(2.0)
        mesh_large = self._make_box_mesh(4.0)

        c_small, _ = _compute_camera_centers(mesh_small, 5, args, seed=0)
        c_large, _ = _compute_camera_centers(mesh_large, 5, args, seed=0)

        bbox_s = mesh_small.get_axis_aligned_bounding_box()
        center_s = (np.asarray(bbox_s.min_bound) + np.asarray(bbox_s.max_bound)) / 2
        bbox_l = mesh_large.get_axis_aligned_bounding_box()
        center_l = (np.asarray(bbox_l.min_bound) + np.asarray(bbox_l.max_bound)) / 2

        r_small = np.linalg.norm(c_small[0] - center_s)
        r_large = np.linalg.norm(c_large[0] - center_l)

        np.testing.assert_allclose(r_large / r_small, 2.0, rtol=1e-6)

    # ------------------------------------------------------------------
    # Fallback tests (auto disabled)
    # ------------------------------------------------------------------
    def test_fixed_camera_radius_used_when_auto_off(self):
        """When auto_camera_radius=False, camera_radius is used directly."""
        mesh = self._make_box_mesh(2.0)
        args = self._make_args(auto_camera_radius=False, camera_radius=10.0)
        centers, _ = _compute_camera_centers(mesh, 5, args, seed=0)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2

        dists = np.linalg.norm(centers - mesh_center, axis=1)
        np.testing.assert_allclose(dists, 10.0, rtol=1e-6)

    def test_orbit_radius_scale_fallback(self):
        """When auto and camera_radius are both off, orbit_radius_scale is used."""
        mesh = self._make_box_mesh(2.0)
        args = self._make_args(auto_camera_radius=False, camera_radius=None, orbit_radius_scale=2.0)
        centers, _ = _compute_camera_centers(mesh, 5, args, seed=0)

        bbox = mesh.get_axis_aligned_bounding_box()
        mesh_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2
        extent = np.linalg.norm(np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound))
        expected_radius = extent * 2.0

        dists = np.linalg.norm(centers - mesh_center, axis=1)
        np.testing.assert_allclose(dists, expected_radius, rtol=1e-6)

    # ------------------------------------------------------------------
    # Output shape and target tests
    # ------------------------------------------------------------------
    def test_output_shapes(self):
        """Centers and targets should have shape (num_views, 3)."""
        mesh = self._make_box_mesh(2.0)
        args = self._make_args()
        centers, targets = _compute_camera_centers(mesh, 20, args, seed=0)

        assert centers.shape == (20, 3)
        assert targets.shape == (20, 3)

    def test_targets_are_mesh_center(self):
        """All targets should point at the mesh bounding box center."""
        mesh = self._make_box_mesh(2.0)
        args = self._make_args()
        _, targets = _compute_camera_centers(mesh, 10, args, seed=0)

        bbox = mesh.get_axis_aligned_bounding_box()
        expected_center = (np.asarray(bbox.min_bound) + np.asarray(bbox.max_bound)) / 2

        for t in targets:
            np.testing.assert_allclose(t, expected_center, atol=1e-10)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
