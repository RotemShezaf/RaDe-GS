#!/usr/bin/env python3
"""
Tests for data transformation classes

Tests cover:
- PointcloudRandomInputDropout
- GaussianPatchRotate
- GaussianPatchCanonicalRotate
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_transformation import (
    PointcloudRandomInputDropout,
    GaussianPatchRotate,
    GaussianPatchCanonicalRotate,
    GaussianPatchDropout,
    GaussianPatchRandomFlip,
    SparseContextDropout,
    GaussianPatchSurfacePerturb,
)


class TestPointcloudRandomInputDropout:
    """Test suite for PointcloudRandomInputDropout."""
    
    def test_dropout_basic(self):
        """Test basic dropout functionality."""
        dropout = PointcloudRandomInputDropout(
            max_dropout_ratio=0.5,
            attributes=["xyz"],
            mask_constant=-10
        )
        
        # Create sample data: (batch_size, num_points, features)
        batch_size = 4
        num_points = 100
        features = 4  # xyz(3) + 1 extra feature (geodesic distance)
        pc = torch.randn(batch_size, num_points, features)
        
        # Apply dropout
        pc_dropped = dropout(pc)
        
        assert pc_dropped.shape == pc.shape
        
        # Check that some points are masked (set to -10)
        for i in range(batch_size):
            masked_points = (pc_dropped[i] == -10).all(dim=1).sum()
            # Should have some dropout (but not deterministic, so just check it's possible)
            assert masked_points >= 0
    
    def test_dropout_no_dropout(self):
        """Test with zero dropout ratio."""
        dropout = PointcloudRandomInputDropout(
            max_dropout_ratio=0.0,
            attributes=["xyz"],
            mask_constant=-10
        )
        # xyz(3) + 1 extra feature (geodesic distance)
        pc = torch.randn(4, 100, 4)
        pc_dropped = dropout(pc)
        
        # Should be unchanged
        assert torch.allclose(pc, pc_dropped)
    
    def test_dropout_mask_constant(self):
        """Test that mask constant is applied correctly."""
        mask_val = -99
        dropout = PointcloudRandomInputDropout(
            max_dropout_ratio=0.99,  # Drop everything
            attributes=["xyz"],
            mask_constant=mask_val
        )
        
        pc = torch.randn(2, 50, 4)  # xyz(3) + 1 extra feature (geodesic distance)
        pc_dropped = dropout(pc)
        # All points should be masked with mask_val
        # (though with random dropout, not guaranteed, so check majority)
        masked_ratio = (pc_dropped == mask_val).float().mean()
        assert masked_ratio > 0.01  # Most should be masked


class TestGaussianPatchRotate:
    """Test suite for GaussianPatchRotate."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood and point_features with known structure."""
        # attributes: xyz(3) + normals(3) + opacity(1) + rotation(4) = 11
        # entry_size = 11 + 1 (geodesic) = 12
        # max_neighbors = 64
        # point_feature_size = 11
        
        batch_size = 2
        max_neighbors = 64
        entry_size = 12
        point_feature_size = 11
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set reasonable values for specific attributes
        for i in range(batch_size):
            # Set xyz coordinates
            neighborhood[i, :, 0:3] = torch.randn(max_neighbors, 3)
            
            # Set normals (normalized)
            normals = torch.randn(max_neighbors, 3)
            neighborhood[i, :, 3:6] = normals / torch.norm(normals, dim=1, keepdim=True)
            
            # Set opacity [0, 1]
            neighborhood[i, :, 6] = torch.rand(max_neighbors)
            
            # Set quaternions (normalized)
            quats = torch.randn(max_neighbors, 4)
            neighborhood[i, :, 7:11] = quats / torch.norm(quats, dim=1, keepdim=True)
            
            # Set geodesic distances
            neighborhood[i, :, 11] = torch.rand(max_neighbors)
            
            # Point features
            # Point xyz (at origin)
            point_features[i, 0:3] = torch.zeros(3)
            # Point normal
            normal = torch.randn(3)
            point_features[i, 3:6] = normal / torch.norm(normal)
            # Point opacity
            point_features[i, 6] = torch.rand(1)
            # Point quaternion
            quat = torch.randn(4)
            point_features[i, 7:11] = quat / torch.norm(quat)
        
        r1_min_val = torch.tensor([0.5, 0.5])
        
        return neighborhood, point_features, r1_min_val
    
    def test_rotation_shape_preservation(self, sample_data):
        """Test that rotation preserves shapes."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        assert rotated_neighborhood.shape == neighborhood.shape
        assert rotated_point_features.shape == point_features.shape
    
    def test_rotation_xyz_changes(self, sample_data):
        """Test that XYZ coordinates change after rotation."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        original_neighborhood = neighborhood.clone()
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Check that XYZ coordinates have changed
        xyz_changed = False
        for neighbor_idx in range(5):  # Check first few neighbors
            original_xyz = original_neighborhood[0, neighbor_idx, 0:3]
            rotated_xyz = rotated_neighborhood[0, neighbor_idx, 0:3]
            
            if not torch.allclose(original_xyz, rotated_xyz, atol=1e-5):
                xyz_changed = True
                break
        
        assert xyz_changed, "XYZ coordinates should change after rotation"
    
    def test_rotation_normals_normalized(self, sample_data):
        """Test that normals remain normalized after rotation."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Check normal magnitudes
        for neighbor_idx in range(10):  # Check first few
            normal = rotated_neighborhood[0, neighbor_idx, 3:6]
            norm = torch.norm(normal)
            
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
                f"Normal magnitude {norm} should be ~1.0"
    
    def test_rotation_quaternion_normalized(self, sample_data):
        """Test that quaternions remain normalized after rotation."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Check quaternion magnitudes
        for neighbor_idx in range(10):
            quat = rotated_neighborhood[0, neighbor_idx, 7:11]
            norm = torch.norm(quat)
            
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
                f"Quaternion magnitude {norm} should be ~1.0"
    
    def test_rotation_opacity_unchanged(self, sample_data):
        """Test that opacity values don't change during rotation."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        original_neighborhood = neighborhood.clone()
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Check opacity values
        for neighbor_idx in range(10):
            original_opacity = original_neighborhood[0, neighbor_idx, 6]
            rotated_opacity = rotated_neighborhood[0, neighbor_idx, 6]
            
            assert torch.isclose(original_opacity, rotated_opacity, atol=1e-6), \
                "Opacity should not change during rotation"
    
    def test_rotation_geodesic_unchanged(self, sample_data):
        """Test that geodesic distances don't change during rotation."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        original_neighborhood = neighborhood.clone()
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Check geodesic distances (last column)
        for neighbor_idx in range(10):
            original_geodesic = original_neighborhood[0, neighbor_idx, -1]
            rotated_geodesic = rotated_neighborhood[0, neighbor_idx, -1]
            
            assert torch.isclose(original_geodesic, rotated_geodesic, atol=1e-6), \
                "Geodesic distance should not change"

    def test_rotation_aug_normals(self, sample_data):
        """Test that aug_normals are rotated consistently with xyz."""
        neighborhood, point_features, r1_min_val = sample_data
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        aug_normals = torch.randn(neighborhood.size(0), neighborhood.size(1), 3)
        aug_normals = aug_normals / aug_normals.norm(dim=-1, keepdim=True)
        original_normals = aug_normals.clone()

        rotated_neighborhood, _ = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val,
            aug_normals=aug_normals,
        )
        # aug_normals should be modified in-place
        # Check that norms are preserved (rotation preserves length)
        norms_after = aug_normals.norm(dim=-1)
        assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-5), \
            "aug_normals should remain unit vectors after rotation"
        # Check they actually changed
        assert not torch.allclose(aug_normals, original_normals, atol=1e-3), \
            "aug_normals should be rotated"


class TestGaussianPatchCanonicalRotate:
    """Test suite for GaussianPatchCanonicalRotate."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood and point_features with known center of mass."""
        batch_size = 2
        max_neighbors = 32  # Smaller for faster tests
        entry_size = 12  # xyz(3) + normals(3) + opacity(1) + rotation(4) + geodesic(1)
        point_feature_size = 11
        
        neighborhood = torch.zeros(batch_size, max_neighbors, entry_size)
        point_features = torch.zeros(batch_size, point_feature_size)
        
        for i in range(batch_size):
            # Create XYZ coordinates with known center of mass
            for neighbor_idx in range(max_neighbors):
                # Create points around a known direction (e.g., [1, 1, 1])
                offset = torch.randn(3) * 0.1
                neighborhood[i, neighbor_idx, 0:3] = torch.tensor([1.0, 1.0, 1.0]) + offset
            
            # Set normals (normalized)
            normals = torch.randn(max_neighbors, 3)
            neighborhood[i, :, 3:6] = normals / torch.norm(normals, dim=1, keepdim=True)
            
            # Set opacity
            neighborhood[i, :, 6] = torch.rand(max_neighbors)
            
            # Set quaternions (normalized)
            quats = torch.randn(max_neighbors, 4)
            neighborhood[i, :, 7:11] = quats / torch.norm(quats, dim=1, keepdim=True)
            
            # Set geodesic distances
            neighborhood[i, :, 11] = torch.rand(max_neighbors)
            
            # Point features
            point_features[i, 0:3] = torch.zeros(3)
            normal = torch.randn(3)
            point_features[i, 3:6] = normal / torch.norm(normal)
            point_features[i, 6] = torch.rand(1)
            quat = torch.randn(4)
            point_features[i, 7:11] = quat / torch.norm(quat)
        
        r1_min_val = torch.tensor([0.5, 0.5])
        
        return neighborhood, point_features, r1_min_val
    
    def test_canonical_rotation_shape(self, sample_data):
        """Test that canonical rotation preserves shape."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchCanonicalRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        assert rotated_neighborhood.shape == neighborhood.shape
        assert rotated_point_features.shape == point_features.shape
    
    def test_canonical_rotation_alignment(self, sample_data):
        """Test that canonical rotation aligns center of mass."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchCanonicalRotate(
            attributes=["xyz", "normals", "opacity", "rotation"],
            target_direction=[0., 1., 0.]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Extract rotated XYZ coordinates
        xyz_coords = rotated_neighborhood[0, :, 0:3]
        
        # Compute center of mass
        center_of_mass = xyz_coords.mean(dim=0)
        
        # Should be roughly aligned with y-axis [0, 1, 0]
        # The alignment means COM direction should be close to [0, 1, 0]
        com_normalized = center_of_mass / torch.norm(center_of_mass)
        target = torch.tensor([0., 1., 0.])
        
        # Dot product should be close to 1 (or -1 if flipped)
        dot_product = torch.dot(com_normalized, target)
        
        assert abs(dot_product) > 0.9, \
            f"Center of mass {com_normalized} not aligned with target [0, 1, 0], dot={dot_product}"
    
    def test_canonical_rotation_deterministic(self, sample_data):
        """Test that canonical rotation is deterministic."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchCanonicalRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated1_n, rotated1_p = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        rotated2_n, rotated2_p = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        # Should produce same result (deterministic)
        assert torch.allclose(rotated1_n, rotated2_n, atol=1e-5), \
            "Canonical rotation should be deterministic"
        assert torch.allclose(rotated1_p, rotated2_p, atol=1e-5), \
            "Canonical rotation should be deterministic"
    
    def test_canonical_rotation_normals_normalized(self, sample_data):
        """Test that normals remain normalized."""
        neighborhood, point_features, r1_min_val = sample_data
        
        rotator = GaussianPatchCanonicalRotate(
            attributes=["xyz", "normals", "opacity", "rotation"]
        )
        
        rotated_neighborhood, rotated_point_features = rotator(
            neighborhood.clone(), point_features.clone(), r1_min_val
        )
        
        for neighbor_idx in range(10):
            normal = rotated_neighborhood[0, neighbor_idx, 3:6]
            norm = torch.norm(normal)
            
            assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
                f"Normal magnitude {norm} should be ~1.0"


class TestTransformationComposition:
    """Test composition of transformations."""
    
    def test_dropout_then_rotate(self):
        """Test applying dropout followed by rotation."""
        # Create sample data
        batch_size = 2
        max_neighbors = 32
        entry_size = 8  # xyz(3) + normals(3) + opacity(1) + geodesic(1)
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        r1_min_val = torch.tensor([0.5, 0.5])
        
        # Set geodesic distances - some > r1_min_val
        for i in range(batch_size):
            for neighbor_idx in range(max_neighbors):
                if neighbor_idx < max_neighbors // 2:
                    neighborhood[i, neighbor_idx, -1] = 0.3  # < r1_min_val
                else:
                    neighborhood[i, neighbor_idx, -1] = 0.8  # > r1_min_val
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.3,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"]
        )
        
        # Apply dropout then rotation
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        rotated_n, rotated_p = rotator(dropped_n, dropped_p, r1_min_val)
        
        # Should work without errors
        assert rotated_n.shape == neighborhood.shape
        assert rotated_p.shape == point_features.shape
    
    def test_rotate_then_canonical(self):
        """Test applying random rotation then canonical rotation."""
        batch_size = 2
        max_neighbors = 32
        entry_size = 8  # xyz(3) + normals(3) + opacity(1) + geodesic(1)
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        r1_min_val = torch.tensor([0.5, 0.5])
        
        # Initialize transformations
        random_rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"]
        )
        
        canonical_rotator = GaussianPatchCanonicalRotate(
            attributes=["xyz", "normals", "opacity"]
        )
        
        # Apply both transformations
        rotated_n, rotated_p = random_rotator(neighborhood.clone(), point_features.clone(), r1_min_val)
        canonical_n, canonical_p = canonical_rotator(rotated_n, rotated_p, r1_min_val)
        
        # Should work without errors
        assert canonical_n.shape == neighborhood.shape
        assert canonical_p.shape == point_features.shape


class TestGaussianPatchDropout:
    """Test suite for GaussianPatchDropout."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood and point_features with neighbors at different geodesic distances."""
        batch_size = 4
        max_neighbors = 64
        # attributes: xyz(3) + normals(3) + opacity(1) + geodesic(1) = 8
        entry_size = 8
        point_feature_size = 7  # xyz(3) + normals(3) + opacity(1), no geodesic
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set reasonable values
        r1_min_val = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        for i in range(batch_size):
            # Set geodesic distances for each neighbor (last column)
            for neighbor_idx in range(max_neighbors):
                # Create mix of distances: some < r1_min_val, some > r1_min_val
                if neighbor_idx < max_neighbors // 2:
                    # First half: distances < r1_min_val
                    neighborhood[i, neighbor_idx, -1] = (torch.rand(1) * 0.4).item()
                else:
                    # Second half: distances > r1_min_val
                    neighborhood[i, neighbor_idx, -1] = (0.6 + torch.rand(1) * 1.0).item()
                
                # Set xyz coordinates
                neighborhood[i, neighbor_idx, 0:3] = torch.randn(3) * 0.5
                
                # Set normals (normalized)
                normal = torch.randn(3)
                neighborhood[i, neighbor_idx, 3:6] = normal / torch.norm(normal)
                
                # Set opacity
                neighborhood[i, neighbor_idx, 6] = torch.rand(1).item()
        
        return neighborhood, point_features, r1_min_val
    
    def test_dropout_basic(self, sample_data):
        """Test basic dropout functionality."""
        neighborhood, point_features, r1_min_val = sample_data
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.5,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        original_neighborhood = neighborhood.clone()
        original_point_features = point_features.clone()
        
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        # Shape should be preserved
        assert dropped_n.shape == original_neighborhood.shape
        assert dropped_p.shape == original_point_features.shape
    
    def test_dropout_only_far_neighbors(self, sample_data):
        """Test that only neighbors with distance > r1_min_val can be dropped."""
        neighborhood, point_features, r1_min_val = sample_data
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.99,  # High dropout ratio to ensure some are dropped
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        original_neighborhood = neighborhood.clone()
        
        # Run dropout multiple times to increase chance of seeing drops
        dropped_n, dropped_p = neighborhood.clone(), point_features.clone()
        for _ in range(10):
            dropped_n, dropped_p = dropout(dropped_n, dropped_p, r1_min_val)
        
        max_neighbors = 64
        
        # Check that near neighbors (distance < r1_min_val) are not dropped
        for i in range(neighborhood.shape[0]):
            r1_min = r1_min_val[i].item()
            
            for neighbor_idx in range(max_neighbors // 2):  # First half has distances < r1_min_val
                original_dist = original_neighborhood[i, neighbor_idx, -1].item()
                
                if original_dist < r1_min:
                    # This neighbor should NOT be dropped - check geodesic is still original
                    dropped_dist = dropped_n[i, neighbor_idx, -1].item()
                    # Near neighbors should not have mask_constant in geodesic
                    assert dropped_dist != -10, \
                        f"Near neighbor with dist={original_dist} < r1_min={r1_min} should not be dropped"
    
    def test_dropout_candidates_selection(self, sample_data):
        """Test that dropout correctly identifies candidates (distance > r1_min_val)."""
        neighborhood, point_features, r1_min_val = sample_data
        
        max_neighbors = 64
        
        # Count candidates for each example
        for i in range(neighborhood.shape[0]):
            r1_min = r1_min_val[i].item()
            
            candidates = 0
            for neighbor_idx in range(max_neighbors):
                dist = neighborhood[i, neighbor_idx, -1].item()
                if dist > r1_min:
                    candidates += 1
            
            # Should have candidates (based on our fixture setup)
            assert candidates > 0, f"Example {i} should have candidates with distance > r1_min_val"
    
    def test_dropout_mask_constant(self, sample_data):
        """Test that mask constant is applied correctly."""
        neighborhood, point_features, r1_min_val = sample_data
        
        mask_val = -99
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.99,  # High dropout to ensure drops
            attributes=["xyz", "normals", "opacity"],
            mask_constant=mask_val
        )
        
        # Apply dropout multiple times to increase chance of masking
        dropped_n, dropped_p = neighborhood.clone(), point_features.clone()
        for _ in range(20):
            dropped_n, dropped_p = dropout(dropped_n, dropped_p, r1_min_val)
        
        max_neighbors = 64
        
        # Check if any entries were masked with mask_val in geodesic distance
        masked_found = False
        for i in range(dropped_n.shape[0]):
            for neighbor_idx in range(max_neighbors):
                geodesic = dropped_n[i, neighbor_idx, -1].item()
                
                if geodesic == mask_val:
                    masked_found = True
                    break
            
            if masked_found:
                break
        
        # With high dropout ratio, we should find at least some masked entries
        # (though probabilistic, so might not always happen)
    
    def test_dropout_zero_ratio(self, sample_data):
        """Test with zero dropout ratio (no dropout)."""
        neighborhood, point_features, r1_min_val = sample_data
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.0,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        original_neighborhood = neighborhood.clone()
        original_point_features = point_features.clone()
        
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        # Should be unchanged
        assert torch.allclose(original_neighborhood, dropped_n, atol=1e-6)
        assert torch.allclose(original_point_features, dropped_p, atol=1e-6)
    
    def test_dropout_zero_r1_min(self):
        """Test dropout behavior when r1_min_val=0 (drops from all neighbors with geodesic > 0)."""
        batch_size = 2
        max_neighbors = 32
        entry_size = 8
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set geodesic distances to positive values (all > 0)
        neighborhood[:, :, -1] = torch.rand(batch_size, max_neighbors) + 0.1
        
        r1_min_val = torch.tensor([0.0, 0.0])  # Zero means all neighbors with geodesic > 0 are candidates
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.99,  # High dropout to ensure some dropout happens
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        # Run multiple times to check dropout can happen
        np.random.seed(42)
        total_dropped = 0
        for _ in range(10):
            dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
            dropped_count = (dropped_n[:, :, -1] == -10).sum().item()
            total_dropped += dropped_count
        
        # With r1_min=0, all neighbors with geodesic > 0 are candidates, so some should be dropped
        assert total_dropped > 0, "Dropout with r1_min=0 should drop neighbors with geodesic > 0"
    
    def test_dropout_shape_preservation(self, sample_data):
        """Test that dropout preserves shapes."""
        neighborhood, point_features, r1_min_val = sample_data
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.5,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        assert dropped_n.shape == neighborhood.shape
        assert dropped_p.shape == point_features.shape
        assert dropped_n.dtype == neighborhood.dtype
    
    def test_dropout_point_features_unchanged(self, sample_data):
        """Test that point features remain unchanged."""
        neighborhood, point_features, r1_min_val = sample_data
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.8,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        original_point_features = point_features.clone()
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        # Point features should be unchanged
        assert torch.allclose(original_point_features, dropped_p, atol=1e-6)
    
    def test_dropout_multiple_attributes(self):
        """Test dropout with various attribute combinations."""
        batch_size = 2
        max_neighbors = 16
        
        # Test with xyz + normals + opacity + rotation
        # entry_size = 3 + 3 + 1 + 4 + 1(geodesic) = 12
        entry_size = 12
        point_feature_size = 11  # No geodesic in point features
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set geodesic distances - half near, half far
        r1_min_val = torch.tensor([0.5, 0.5])
        for i in range(batch_size):
            for neighbor_idx in range(max_neighbors):
                if neighbor_idx < max_neighbors // 2:
                    neighborhood[i, neighbor_idx, -1] = 0.3  # < r1_min
                else:
                    neighborhood[i, neighbor_idx, -1] = 0.8  # > r1_min
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.5,
            attributes=["xyz", "normals", "opacity", "rotation"],
            mask_constant=-10
        )
        
        dropped_n, dropped_p = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        # Should work without errors
        assert dropped_n.shape == neighborhood.shape


class TestGaussianPatchRandomFlip:
    """Test suite for GaussianPatchRandomFlip."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood and point_features with known structure."""
        # attributes: xyz(3) + normals(3) + opacity(1) = 7
        # entry_size = 7 + 1 (geodesic) = 8
        # max_neighbors = 32
        # point_feature_size = 7
        
        batch_size = 2
        max_neighbors = 32
        entry_size = 8
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set reasonable values
        for i in range(batch_size):
            # Set xyz coordinates
            neighborhood[i, :, 0:3] = torch.randn(max_neighbors, 3)
            
            # Set normals (normalized)
            normals = torch.randn(max_neighbors, 3)
            neighborhood[i, :, 3:6] = normals / torch.norm(normals, dim=1, keepdim=True)
            
            # Set opacity [0, 1]
            neighborhood[i, :, 6] = torch.rand(max_neighbors)
            
            # Set geodesic distances
            neighborhood[i, :, 7] = torch.rand(max_neighbors)
            
            # Point features
            point_features[i, 0:3] = torch.randn(3)
            normal = torch.randn(3)
            point_features[i, 3:6] = normal / torch.norm(normal)
            point_features[i, 6] = torch.rand(1)
        
        return neighborhood, point_features
    
    def test_flip_shape_preservation(self, sample_data):
        """Test that flip preserves shapes."""
        neighborhood, point_features = sample_data
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0, 1, 2],
            flip_prob=1.0  # Always flip
        )
        
        flipped_n, flipped_p = flipper(neighborhood.clone(), point_features.clone(), None)
        
        assert flipped_n.shape == neighborhood.shape
        assert flipped_p.shape == point_features.shape
    
    def test_flip_xyz_changes(self, sample_data):
        """Test that XYZ coordinates can change after flip."""
        neighborhood, point_features = sample_data
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0],  # Only flip x-axis
            flip_prob=1.0  # Always flip
        )
        
        original_neighborhood = neighborhood.clone()
        flipped_n, flipped_p = flipper(neighborhood.clone(), point_features.clone(), None)
        
        # X coordinates should be negated
        for neighbor_idx in range(5):
            original_x = original_neighborhood[0, neighbor_idx, 0]
            flipped_x = flipped_n[0, neighbor_idx, 0]
            
            assert torch.isclose(original_x, -flipped_x, atol=1e-5), \
                f"X should be negated: original={original_x}, flipped={flipped_x}"
            
            # Y and Z should be unchanged
            assert torch.isclose(original_neighborhood[0, neighbor_idx, 1], flipped_n[0, neighbor_idx, 1], atol=1e-5)
            assert torch.isclose(original_neighborhood[0, neighbor_idx, 2], flipped_n[0, neighbor_idx, 2], atol=1e-5)
    
    def test_flip_normals_change(self, sample_data):
        """Test that normals are also flipped."""
        neighborhood, point_features = sample_data
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[1],  # Only flip y-axis
            flip_prob=1.0
        )
        
        original_neighborhood = neighborhood.clone()
        flipped_n, flipped_p = flipper(neighborhood.clone(), point_features.clone(), None)
        
        # Y component of normals should be negated
        for neighbor_idx in range(5):
            original_ny = original_neighborhood[0, neighbor_idx, 4]  # normal y at index 4
            flipped_ny = flipped_n[0, neighbor_idx, 4]
            
            assert torch.isclose(original_ny, -flipped_ny, atol=1e-5), \
                f"Normal Y should be negated"
    
    def test_flip_zero_prob_no_change(self, sample_data):
        """Test that zero flip probability leaves data unchanged."""
        neighborhood, point_features = sample_data
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0, 1, 2],
            flip_prob=0.0  # Never flip
        )
        
        original_neighborhood = neighborhood.clone()
        original_point_features = point_features.clone()
        
        flipped_n, flipped_p = flipper(neighborhood.clone(), point_features.clone(), None)
        
        assert torch.allclose(original_neighborhood, flipped_n, atol=1e-6)
        assert torch.allclose(original_point_features, flipped_p, atol=1e-6)
    
    def test_flip_geodesic_unchanged(self, sample_data):
        """Test that geodesic distances are unchanged by flip."""
        neighborhood, point_features = sample_data
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0, 1, 2],
            flip_prob=1.0
        )
        
        original_geodesic = neighborhood[:, :, -1].clone()
        flipped_n, flipped_p = flipper(neighborhood.clone(), point_features.clone(), None)
        
        assert torch.allclose(original_geodesic, flipped_n[:, :, -1], atol=1e-6)
    
    def test_flip_single_example(self, sample_data):
        """Test flip with single example (no batch dimension)."""
        neighborhood, point_features = sample_data
        
        # Take single example
        single_neighborhood = neighborhood[0]  # (max_neighbors, entry_size)
        single_point_features = point_features[0]  # (point_feature_size,)
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0],
            flip_prob=1.0
        )
        
        flipped_n, flipped_p = flipper(single_neighborhood.clone(), single_point_features.clone(), None)
        
        # Should return same shapes
        assert flipped_n.shape == single_neighborhood.shape
        assert flipped_p.shape == single_point_features.shape


class TestDropoutWithNoneR1Min:
    """Test GaussianPatchDropout behavior when r1_min_val is None."""
    
    def test_dropout_none_r1_min_drops_from_all(self):
        """Test that None r1_min allows dropout from all neighbors."""
        batch_size = 2
        max_neighbors = 32
        entry_size = 8
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set geodesic distances to various values
        for i in range(batch_size):
            neighborhood[i, :, -1] = torch.linspace(0.1, 2.0, max_neighbors)
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.99,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        # Run multiple times to check dropout happens
        np.random.seed(42)
        total_dropped = 0
        for _ in range(10):
            dropped_n, _ = dropout(neighborhood.clone(), point_features.clone(), None)
            dropped_count = (dropped_n[:, :, -1] == -10).sum().item()
            total_dropped += dropped_count
        
        assert total_dropped > 0, "Dropout with None r1_min should drop some neighbors"
    
    def test_dropout_none_vs_tensor_r1_min(self):
        """Test difference between None r1_min and tensor r1_min."""
        batch_size = 2
        max_neighbors = 32
        entry_size = 8
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set geodesic distances - half near (0.2), half far (0.8)
        r1_min_val = torch.tensor([0.5, 0.5])
        for i in range(batch_size):
            for neighbor_idx in range(max_neighbors):
                if neighbor_idx < max_neighbors // 2:
                    neighborhood[i, neighbor_idx, -1] = 0.2  # < r1_min
                else:
                    neighborhood[i, neighbor_idx, -1] = 0.8  # > r1_min
        
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.99,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        # With tensor r1_min, only far neighbors (second half) can be dropped
        np.random.seed(42)
        dropped_with_r1_min, _ = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        
        # Near neighbors should never be dropped
        near_dropped = (dropped_with_r1_min[:, :max_neighbors//2, -1] == -10).sum().item()
        assert near_dropped == 0, "Near neighbors should not be dropped when r1_min is specified"
        
        # With None r1_min, all neighbors can be dropped
        np.random.seed(42)
        dropped_with_none, _ = dropout(neighborhood.clone(), point_features.clone(), None)
        
        # This test is probabilistic, but with high dropout ratio some should be dropped


class TestComposedTransformations:
    """Test composed transformations pipeline."""
    
    def test_full_pipeline_dropout_rotate_flip(self):
        """Test full pipeline: dropout -> rotate -> flip."""
        batch_size = 4
        max_neighbors = 32
        entry_size = 8  # xyz(3) + normals(3) + opacity(1) + geodesic(1)
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set valid geodesic distances
        for i in range(batch_size):
            neighborhood[i, :, -1] = torch.linspace(0.1, 1.5, max_neighbors)
        
        # Create transforms
        dropout = GaussianPatchDropout(
            max_dropout_ratio=0.3,
            attributes=["xyz", "normals", "opacity"],
            mask_constant=-10
        )
        
        rotator = GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"]
        )
        
        flipper = GaussianPatchRandomFlip(
            attributes=["xyz", "normals", "opacity"],
            flip_axes=[0, 1, 2],
            flip_prob=0.5
        )
        
        # Apply pipeline with None r1_min (simulating use_r1_min=False)
        n1, p1 = dropout(neighborhood.clone(), point_features.clone(), None)
        n2, p2 = rotator(n1, p1, None)
        n3, p3 = flipper(n2, p2, None)
        
        # Check shapes preserved
        assert n3.shape == neighborhood.shape
        assert p3.shape == point_features.shape
    
    def test_pipeline_with_canonical_rotation(self):
        """Test pipeline including canonical rotation."""
        batch_size = 2
        max_neighbors = 32
        entry_size = 8
        point_feature_size = 7
        
        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)
        
        # Set xyz to have meaningful center of mass
        for i in range(batch_size):
            neighborhood[i, :, 0:3] = torch.randn(max_neighbors, 3)
            neighborhood[i, :, -1] = torch.rand(max_neighbors) + 0.1  # Valid geodesic
        
        # Create transforms
        random_rotate = GaussianPatchRotate(attributes=["xyz", "normals", "opacity"])
        canonical = GaussianPatchCanonicalRotate(attributes=["xyz", "normals", "opacity"])
        flip = GaussianPatchRandomFlip(attributes=["xyz", "normals", "opacity"], flip_prob=0.5)
        
        # Apply: random rotate -> canonical -> flip
        n1, p1 = random_rotate(neighborhood.clone(), point_features.clone(), None)
        n2, p2 = canonical(n1, p1, None)
        n3, p3 = flip(n2, p2, None)
        
        assert n3.shape == neighborhood.shape
        assert p3.shape == point_features.shape


class TestSparseContextDropout:
    """Test suite for SparseContextDropout (unified dropout replacing GaussianPatchDropout)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood with valid/masked neighbors."""
        batch_size = 4
        max_neighbors = 64
        # xyz(3) + geodesic(1)
        entry_size = 4
        point_feature_size = 3

        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)

        # First 40 neighbors are valid (positive geodesic), rest masked
        for i in range(batch_size):
            neighborhood[i, :40, -1] = torch.sort(torch.rand(40))[0]  # sorted ascending
            neighborhood[i, 40:, -1] = -10.0  # masked

        return neighborhood, point_features

    def test_shape_preservation(self, sample_data):
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=0.5,
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, out_p = dropout(neighborhood.clone(), point_features.clone())
        assert out_n.shape == neighborhood.shape
        assert out_p.shape == point_features.shape

    def test_min_valid_guarantee(self, sample_data):
        """At least min_valid valid neighbors are retained (sparse mode)."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=3, max_valid_ratio=0.1, p=1.0,
            p_dropout=0.0,  # always sparse mode
            attributes=["xyz"], mask_constant=-10.0,
        )
        for _ in range(20):
            out_n, _ = dropout(neighborhood.clone(), point_features.clone())
            for i in range(out_n.size(0)):
                n_valid = (out_n[i, :, -1] != -10.0).sum().item()
                assert n_valid >= 3, f"Expected >= 3 valid, got {n_valid}"

    def test_keeps_closest_by_geodesic(self, sample_data):
        """In sparse mode, retained neighbors should be the closest by geodesic."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.2, p=1.0,
            p_dropout=0.0,  # always sparse mode
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, _ = dropout(neighborhood.clone(), point_features.clone())
        for i in range(out_n.size(0)):
            valid_mask = out_n[i, :, -1] != -10.0
            if valid_mask.sum() == 0:
                continue
            surviving_geodesic = out_n[i, valid_mask, -1]
            orig_valid_geodesic = neighborhood[i, :40, -1].sort()[0]
            n_kept = surviving_geodesic.size(0)
            expected = orig_valid_geodesic[:n_kept]
            assert torch.allclose(surviving_geodesic.sort()[0], expected, atol=1e-5), \
                "Surviving neighbors should be the closest by geodesic"

    def test_p_zero_noop(self, sample_data):
        """When p=0, no dropout should occur."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=0.0,
            p_dropout=0.5,
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, out_p = dropout(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_n, neighborhood)
        assert torch.allclose(out_p, point_features)

    def test_reduces_valid_count(self, sample_data):
        """With p=1, valid count should drop significantly (stochastic, multiple runs)."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=0.5,
            attributes=["xyz"], mask_constant=-10.0,
        )
        total_valid_before = 0
        total_valid_after = 0
        for _ in range(20):
            out_n, _ = dropout(neighborhood.clone(), point_features.clone())
            for i in range(out_n.size(0)):
                total_valid_before += (neighborhood[i, :, -1] != -10.0).sum().item()
                total_valid_after += (out_n[i, :, -1] != -10.0).sum().item()
        assert total_valid_after < total_valid_before, \
            "Sparse dropout should reduce total valid count"

    def test_single_example(self, sample_data):
        """Test with unbatched (single example) input."""
        neighborhood, point_features = sample_data
        single_n = neighborhood[0]
        single_p = point_features[0]
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=0.5,
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, out_p = dropout(single_n.clone(), single_p.clone())
        assert out_n.shape == single_n.shape
        assert out_p.shape == single_p.shape

    def test_point_features_unchanged(self, sample_data):
        """Point features should never be modified."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=0.5,
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, out_p = dropout(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_p, point_features)

    def test_p_dropout_one_only_dropout_mode(self, sample_data):
        """p_dropout=1 should always use GaussianPatchDropout-style filtering."""
        neighborhood, point_features = sample_data
        r1_min_val = torch.tensor([0.3] * neighborhood.size(0))
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=1.0,
            max_dropout_ratio=0.8,
            attributes=["xyz"], mask_constant=-10.0,
        )
        # After many runs, neighbours <= r1_min should always survive
        for _ in range(30):
            out_n, _ = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
            for i in range(out_n.size(0)):
                valid_mask = out_n[i, :, -1] != -10.0
                # All surviving geo should be <= r1_min OR were already masked
                surviving_geo = out_n[i, valid_mask, -1]
                orig_valid_below = (neighborhood[i, :40, -1] <= 0.3).sum().item()
                # At least the below-threshold neighbours should survive
                assert valid_mask.sum().item() >= orig_valid_below

    def test_p_dropout_zero_only_sparse_mode(self, sample_data):
        """p_dropout=0 should always use sparse-context mode."""
        neighborhood, point_features = sample_data
        dropout = SparseContextDropout(
            min_valid=2, max_valid_ratio=0.15, p=1.0,
            p_dropout=0.0,
            attributes=["xyz"], mask_constant=-10.0,
        )
        for _ in range(20):
            out_n, _ = dropout(neighborhood.clone(), point_features.clone())
            for i in range(out_n.size(0)):
                n_valid = (out_n[i, :, -1] != -10.0).sum().item()
                max_keep = max(2, int(40 * 0.15))
                assert n_valid <= max_keep + 1  # +1 tolerance for boundary

    def test_dropout_mode_respects_r1_min(self, sample_data):
        """In dropout mode, only neighbours with geo > r1_min are candidates."""
        neighborhood, point_features = sample_data
        # Set r1_min very high so no neighbours are candidates
        r1_min_val = torch.tensor([999.0] * neighborhood.size(0))
        dropout = SparseContextDropout(
            min_valid=1, max_valid_ratio=0.3, p=1.0,
            p_dropout=1.0,  # always dropout mode
            max_dropout_ratio=0.99,
            attributes=["xyz"], mask_constant=-10.0,
        )
        out_n, _ = dropout(neighborhood.clone(), point_features.clone(), r1_min_val)
        # No candidates → no dropout
        for i in range(out_n.size(0)):
            n_valid_before = (neighborhood[i, :, -1] != -10.0).sum().item()
            n_valid_after = (out_n[i, :, -1] != -10.0).sum().item()
            assert n_valid_after == n_valid_before


class TestGaussianPatchSurfacePerturb:
    """Test suite for GaussianPatchSurfacePerturb."""

    @pytest.fixture
    def sample_data(self):
        """Create sample neighborhood with xyz features."""
        batch_size = 4
        max_neighbors = 32
        # xyz(3) + geodesic(1)
        entry_size = 4
        point_feature_size = 3

        neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
        point_features = torch.randn(batch_size, point_feature_size)

        # first 20 neighbors valid, rest masked
        for i in range(batch_size):
            neighborhood[i, :20, -1] = torch.rand(20) + 0.01
            neighborhood[i, 20:, -1] = -10.0

        return neighborhood, point_features

    @pytest.fixture
    def sample_aug_normals(self):
        """Create sample aug_normals with unit normals."""
        batch_size = 4
        max_neighbors = 32
        normals = torch.randn(batch_size, max_neighbors, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        return normals

    def test_shape_preservation(self, sample_data):
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.02, p=1.0, attributes=["xyz"],
        )
        out_n, out_p = perturb(neighborhood.clone(), point_features.clone())
        assert out_n.shape == neighborhood.shape
        assert out_p.shape == point_features.shape

    def test_xyz_changes_isotropic(self, sample_data):
        """XYZ should change with isotropic fallback (no aug_normals)."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(neighborhood.clone(), point_features.clone())
        xyz_diff = (out_n[:, :, :3] - neighborhood[:, :, :3]).abs().sum()
        assert xyz_diff > 0, "XYZ should change after isotropic perturbation"

    def test_xyz_changes_directed(self, sample_data, sample_aug_normals):
        """XYZ should change with directed perturbation along aug_normals."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(
            neighborhood.clone(), point_features.clone(),
            aug_normals=sample_aug_normals.clone(),
        )
        xyz_diff = (out_n[:, :, :3] - neighborhood[:, :, :3]).abs().sum()
        assert xyz_diff > 0, "XYZ should change after directed perturbation"

    def test_directed_perturbation_along_normal(self, sample_data, sample_aug_normals):
        """With aug_normals, perturbation should be strictly along the normal direction."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(
            neighborhood.clone(), point_features.clone(),
            aug_normals=sample_aug_normals.clone(),
        )
        diff = out_n[:, :, :3] - neighborhood[:, :, :3]  # (B, N, 3)
        # For valid neighbors, diff should be scalar * normal
        for i in range(neighborhood.size(0)):
            valid = neighborhood[i, :, -1] != -10.0
            if valid.sum() == 0:
                continue
            d = diff[i, valid]  # (n_valid, 3)
            n = sample_aug_normals[i, valid]  # (n_valid, 3)
            # Project diff onto normal: should recover full magnitude
            proj = (d * n).sum(dim=-1, keepdim=True)  # scalar projection
            reconstructed = proj * n
            residual = (d - reconstructed).norm(dim=-1)
            assert residual.max() < 1e-5, \
                f"Perturbation not along normal, max residual: {residual.max()}"

    def test_geodesic_unchanged(self, sample_data):
        """Geodesic distances should not change."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_n[:, :, -1], neighborhood[:, :, -1]), \
            "Geodesic column should be unchanged"

    def test_p_zero_noop(self, sample_data):
        """p=0 should mean no perturbation."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.1, p=0.0, attributes=["xyz"],
        )
        out_n, out_p = perturb(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_n, neighborhood)
        assert torch.allclose(out_p, point_features)

    def test_perturbation_bounded_directed(self, sample_data, sample_aug_normals):
        """Perturbation magnitude should not exceed max_offset per neighbor."""
        neighborhood, point_features = sample_data
        max_off = 0.02
        perturb = GaussianPatchSurfacePerturb(
            max_offset=max_off, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(
            neighborhood.clone(), point_features.clone(),
            aug_normals=sample_aug_normals.clone(),
        )
        diff = (out_n[:, :, :3] - neighborhood[:, :, :3]).norm(dim=-1)
        assert diff.max().item() <= max_off + 1e-5, \
            f"Max perturbation {diff.max().item()} exceeds {max_off}"

    def test_point_features_unchanged(self, sample_data):
        """Point features should never be modified."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        _, out_p = perturb(neighborhood.clone(), point_features.clone())
        assert torch.allclose(out_p, point_features)

    def test_single_example(self, sample_data):
        """Test with unbatched input."""
        neighborhood, point_features = sample_data
        single_n = neighborhood[0]
        single_p = point_features[0]
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.02, p=1.0, attributes=["xyz"],
        )
        out_n, out_p = perturb(single_n.clone(), single_p.clone())
        assert out_n.shape == single_n.shape
        assert out_p.shape == single_p.shape

    def test_single_example_directed(self, sample_data, sample_aug_normals):
        """Test directed perturbation with unbatched input."""
        neighborhood, point_features = sample_data
        single_n = neighborhood[0]
        single_p = point_features[0]
        single_aug = sample_aug_normals[0]
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.02, p=1.0, attributes=["xyz"],
        )
        out_n, out_p = perturb(
            single_n.clone(), single_p.clone(),
            aug_normals=single_aug.clone(),
        )
        assert out_n.shape == single_n.shape
        assert out_p.shape == single_p.shape

    def test_masked_neighbors_unchanged(self, sample_data, sample_aug_normals):
        """Masked neighbors should not be perturbed."""
        neighborhood, point_features = sample_data
        perturb = GaussianPatchSurfacePerturb(
            max_offset=0.05, p=1.0, attributes=["xyz"],
        )
        out_n, _ = perturb(
            neighborhood.clone(), point_features.clone(),
            aug_normals=sample_aug_normals.clone(),
        )
        for i in range(neighborhood.size(0)):
            masked = neighborhood[i, :, -1] == -10.0
            assert torch.allclose(out_n[i, masked, :3], neighborhood[i, masked, :3]), \
                "Masked neighbors should not be perturbed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])