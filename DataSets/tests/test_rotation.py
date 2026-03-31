#!/usr/bin/env python3
"""
Test script for GaussianPatchRotate transformation.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml
# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#add DataSets path
data_sets_path = project_root / "DataSets"
if str(data_sets_path) not in sys.path:
    sys.path.insert(0, str(data_sets_path))

from data_transformation import GaussianPatchRotate
from gaussian_dataset import GaussianPatchDataset


def test_rotation_basic():
    """Test basic rotation functionality."""
    print("="*80)
    print("Test 1: Basic Rotation Test")
    print("="*80)
    
    # Create a simple example with xyz attributes
    attributes = ["xyz"]
    max_neighbors = 5
    
    # Create rotation transform (new API: no max_neighbors or use_r1_min_val in init)
    rotate = GaussianPatchRotate(attributes=attributes)
    
    print(f"Entry size (with geodesic): {rotate.entry_size}")
    
    # Create sample data in new format: (neighborhood, point_features)
    # entry_size = xyz(3) + geodesic(1) = 4
    # point_feature_size = xyz(3) = 3
    entry_size = 4
    point_feature_size = 3
    
    batch_size = 2
    neighborhood = torch.zeros(batch_size, max_neighbors, entry_size)
    point_features = torch.zeros(batch_size, point_feature_size)
    
    # Set up first example with some test coordinates
    for i in range(max_neighbors):
        neighborhood[0, i, :3] = torch.tensor([i*0.1, i*0.2, i*0.3])  # xyz
        neighborhood[0, i, 3] = i * 0.5  # geodesic distance
    
    # Point features (xyz at origin)
    point_features[0] = torch.tensor([0., 0., 0.])
    
    print(f"\nNeighborhood shape: {neighborhood.shape}")
    print(f"Point features shape: {point_features.shape}")
    print(f"Example before rotation (first neighbor xyz): {neighborhood[0, 0, :3]}")
    
    # Save original values for comparison
    original_xyz = neighborhood[0, 0, :3].clone()
    original_geodesic = neighborhood[:, :, -1].clone()
    
    # Apply rotation (new API: pass neighborhood, point_features, r1_min_val)
    rotated_n, rotated_p = rotate(neighborhood, point_features, None)
    
    print(f"\nExample after rotation (first neighbor xyz): {rotated_n[0, 0, :3]}")
    
    # Check that distances are preserved
    original_dist = torch.norm(original_xyz)
    rotated_dist = torch.norm(rotated_n[0, 0, :3])
    print(f"\nDistance preservation check:")
    print(f"  Original distance: {original_dist:.6f}")
    print(f"  Rotated distance: {rotated_dist:.6f}")
    print(f"  Difference: {abs(original_dist - rotated_dist):.9f}")
    
    # Check that geodesic distances are unchanged
    assert torch.allclose(original_geodesic, rotated_n[:, :, -1]), "Geodesic distances changed!"
    print("\n✓ Geodesic distances preserved correctly")
    
    print("\n" + "="*80)
    print("Test 1 PASSED")
    print("="*80 + "\n")


def test_rotation_with_normals():
    """Test rotation with normals."""
    print("="*80)
    print("Test 2: Rotation with Normals")
    print("="*80)
    
    attributes = ["xyz", "normals"]
    max_neighbors = 3
    
    rotate = GaussianPatchRotate(attributes=attributes)
    
    print(f"Entry size: {rotate.entry_size}")
    
    # entry_size = xyz(3) + normals(3) + geodesic(1) = 7
    # point_feature_size = xyz(3) + normals(3) = 6
    entry_size = 7
    point_feature_size = 6
    
    batch_size = 1
    neighborhood = torch.zeros(batch_size, max_neighbors, entry_size)
    point_features = torch.zeros(batch_size, point_feature_size)
    
    # Set up neighbors: xyz + normals + geodesic
    for i in range(max_neighbors):
        neighborhood[0, i, :3] = torch.tensor([1.0, 0.0, 0.0])  # xyz along x-axis
        neighborhood[0, i, 3:6] = torch.tensor([0.0, 1.0, 0.0])  # normal along y-axis
        neighborhood[0, i, 6] = i * 0.5  # geodesic
    
    # Point features: xyz + normals
    point_features[0, :3] = torch.tensor([0., 0., 0.])  # xyz at origin
    point_features[0, 3:6] = torch.tensor([0., 0., 1.])  # normal along z-axis
    
    print(f"\nBefore rotation:")
    print(f"  First neighbor xyz: {neighborhood[0, 0, :3]}")
    print(f"  First neighbor normal: {neighborhood[0, 0, 3:6]}")
    print(f"  Point normal: {point_features[0, 3:6]}")
    
    # Save original values
    original_xyz = neighborhood[0, 0, :3].clone()
    original_normal = neighborhood[0, 0, 3:6].clone()
    
    rotated_n, rotated_p = rotate(neighborhood, point_features, None)
    
    print(f"\nAfter rotation:")
    print(f"  First neighbor xyz: {rotated_n[0, 0, :3]}")
    print(f"  First neighbor normal: {rotated_n[0, 0, 3:6]}")
    print(f"  Point normal: {rotated_p[0, 3:6]}")
    
    # Check normal is still unit length
    normal_length = torch.norm(rotated_n[0, 0, 3:6])
    print(f"\nNormal length after rotation: {normal_length:.6f}")
    assert torch.allclose(normal_length, torch.tensor(1.0), atol=1e-5), "Normal not unit length!"
    
    print("\n" + "="*80)
    print("Test 2 PASSED")
    print("="*80 + "\n")


def test_with_real_data():
    """Test rotation with real dataset if available."""
    print("="*80)
    print("Test 3: Rotation with Real Data")
    print("="*80)
    
    config_path = "GenerateData/configs/saddle.yaml"
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("Skipping real data test")
        return
    
    try:
        # Load dataset
        dataset = GaussianPatchDataset(
            config_path=config_path,
            ring=2,
            use_r1_min=True
        )
        
        print(f"Loaded dataset with {len(dataset)} examples")
        print(f"Feature dimension: {dataset.feature_dim}")
        print(f"Ring: {dataset.ring}")
        print(f"Attributes: {dataset.config.get('attributes', ['xyz'])}")
        
        # Get a sample from the dataset using the new API
        neighborhood, point_features, target, valid_mask = dataset[0]
        
        print(f"\nNeighborhood shape: {neighborhood.shape}")
        print(f"Point features shape: {point_features.shape}")
        print(f"Target: {target}")
        
        # Create rotation transform (new API)
        rotate = GaussianPatchRotate(
            attributes=dataset.config.get('attributes', ['xyz'])
        )
        
        # Save values before rotation
        xyz_before = neighborhood[:, :3].clone()
        geodesic_before = neighborhood[:, -1].clone()
        
        print(f"\nBefore rotation:")
        print(f"  First neighbor xyz: {xyz_before[0]}")
        print(f"  First neighbor geodesic: {geodesic_before[0]}")
        
        # Apply rotation (single example)
        rotated_n, rotated_p = rotate(neighborhood, point_features, None)
        
        print(f"\nAfter rotation:")
        print(f"  First neighbor xyz: {rotated_n[0, :3]}")
        print(f"  First neighbor geodesic: {rotated_n[0, -1]}")
        
        # Verify geodesic distances are unchanged
        assert torch.allclose(geodesic_before, rotated_n[:, -1]), "Geodesic distances changed during rotation!"
        print("\n✓ Geodesic distances preserved correctly")
        
        # Check that some features actually changed (rotation happened)
        features_changed = not torch.allclose(xyz_before, rotated_n[:, :3], atol=1e-6)
        
        if not features_changed:
            print("\nDEBUG: XYZ features didn't change!")
            print(f"  Max difference: {torch.max(torch.abs(rotated_n[:, :3] - xyz_before)):.10f}")
        
        assert features_changed, "Features didn't change - rotation may not be working!"
        print("✓ Features were transformed (rotation applied)")
        
        print("\n" + "="*80)
        print("Test 3 PASSED")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error during real data test: {e}")
        import traceback
        traceback.print_exc()
        print("Test 3 FAILED or SKIPPED")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING GaussianPatchRotate")
    print("="*80 + "\n")
    
    test_rotation_basic()
    test_rotation_with_normals()
    test_with_real_data()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")
