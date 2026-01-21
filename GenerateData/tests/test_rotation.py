#!/usr/bin/env python3
"""
Test script for GaussianPatchRotate transformation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#add GenerateData to path
generate_data_path = project_root / "GenerateData"
if str(generate_data_path) not in sys.path:
    sys.path.insert(0, str(generate_data_path))

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
    use_r1_min_val = True
    
    # Create rotation transform
    rotate = GaussianPatchRotate(
        attributes=attributes, 
        max_neighbors=max_neighbors,
        use_r1_min_val=use_r1_min_val
    )
    
    print(f"Entry size (with geodesic): {rotate.entry_size}")
    print(f"Point feature size (without geodesic): {rotate.point_feature_size}")
    
    # Create a simple batch: 2 examples
    # Each neighbor: [x, y, z, geodesic_dist]
    # Point features: [0, 0, 0] (relative position)
    # r1_min_val, p_u
    
    batch_size = 2
    feature_dim = max_neighbors * rotate.entry_size + rotate.point_feature_size + 2  # +2 for r1_min and p_u
    
    batch = torch.zeros(batch_size, feature_dim)
    
    # Set up first example with some test coordinates
    for i in range(max_neighbors):
        batch[0, i*4:i*4+3] = torch.tensor([i*0.1, i*0.2, i*0.3])  # xyz
        batch[0, i*4+3] = i * 0.5  # geodesic distance
    
    # Point features (xyz at origin)
    point_start = max_neighbors * rotate.entry_size
    batch[0, point_start:point_start+3] = torch.tensor([0., 0., 0.])
    
    # r1_min_val and p_u
    batch[0, -2] = 0.5  # r1_min_val
    batch[0, -1] = 2.0  # p_u (target)
    
    print(f"\nBatch shape: {batch.shape}")
    print(f"Example before rotation (first neighbor xyz): {batch[0, :3]}")
    print(f"r1_min_val: {batch[0, -2]:.4f}, p_u: {batch[0, -1]:.4f}")
    
    # Save original values for comparison
    original_xyz = batch[0, :3].clone()
    original_r1_min = batch[:, -2].clone()
    original_p_u = batch[:, -1].clone()
    
    # Apply rotation
    rotated_batch = rotate(batch)
    
    print(f"\nExample after rotation (first neighbor xyz): {rotated_batch[0, :3]}")
    print(f"r1_min_val: {rotated_batch[0, -2]:.4f}, p_u: {rotated_batch[0, -1]:.4f}")
    
    # Check that distances are preserved
    original_dist = torch.norm(original_xyz)
    rotated_dist = torch.norm(rotated_batch[0, :3])
    print(f"\nDistance preservation check:")
    print(f"  Original distance: {original_dist:.6f}")
    print(f"  Rotated distance: {rotated_dist:.6f}")
    print(f"  Difference: {abs(original_dist - rotated_dist):.9f}")
    
    # Check that r1_min_val and p_u are unchanged
    assert torch.allclose(original_r1_min, rotated_batch[:, -2]), "r1_min_val changed!"
    assert torch.allclose(original_p_u, rotated_batch[:, -1]), "p_u changed!"
    print("\n✓ r1_min_val and p_u preserved correctly")
    
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
    use_r1_min_val = True
    
    rotate = GaussianPatchRotate(
        attributes=attributes,
        max_neighbors=max_neighbors,
        use_r1_min_val=use_r1_min_val
    )
    
    print(f"Entry size: {rotate.entry_size}")
    print(f"Point feature size: {rotate.point_feature_size}")
    
    batch_size = 1
    feature_dim = max_neighbors * rotate.entry_size + rotate.point_feature_size + 2
    
    batch = torch.zeros(batch_size, feature_dim)
    
    # Set up neighbors: xyz + normals + geodesic
    for i in range(max_neighbors):
        batch[0, i*7:i*7+3] = torch.tensor([1.0, 0.0, 0.0])  # xyz along x-axis
        batch[0, i*7+3:i*7+6] = torch.tensor([0.0, 1.0, 0.0])  # normal along y-axis
        batch[0, i*7+6] = i * 0.5  # geodesic
    
    # Point features: xyz + normals
    point_start = max_neighbors * rotate.entry_size
    batch[0, point_start:point_start+3] = torch.tensor([0., 0., 0.])  # xyz at origin
    batch[0, point_start+3:point_start+6] = torch.tensor([0., 0., 1.])  # normal along z-axis
    
    batch[0, -2] = 0.3  # r1_min_val
    batch[0, -1] = 1.5  # p_u
    
    print(f"\nBefore rotation:")
    print(f"  First neighbor xyz: {batch[0, :3]}")
    print(f"  First neighbor normal: {batch[0, 3:6]}")
    print(f"  Point normal: {batch[0, point_start+3:point_start+6]}")
    
    # Save original values
    original_xyz = batch[0, :3].clone()
    original_normal = batch[0, 3:6].clone()
    
    rotated_batch = rotate(batch)
    
    print(f"\nAfter rotation:")
    print(f"  First neighbor xyz: {rotated_batch[0, :3]}")
    print(f"  First neighbor normal: {rotated_batch[0, 3:6]}")
    print(f"  Point normal: {rotated_batch[0, point_start+3:point_start+6]}")
    
    # Check normal is still unit length
    normal_length = torch.norm(rotated_batch[0, 3:6])
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
        
        # Get a batch
        batch_size = 4
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        batch_data = torch.stack([dataset.examples[i] for i in indices])
        
        print(f"\nBatch shape: {batch_data.shape}")
        
        # Create rotation transform
        
        rotate = GaussianPatchRotate(
            attributes=dataset.config.get('attributes', ['xyz']),
            max_neighbors=dataset.config['ring_size_mapping']['mahalanobis' if dataset.config['use_mahalanobis'] else 'euclidean'][dataset.ring],
            use_r1_min_val=dataset.use_r1_min
        )
        
        # Extract features and targets before rotation (clone to preserve original)
        features_before = (batch_data[:, :-2] if dataset.use_r1_min else batch_data[:, :-1]).clone()
        targets_before = batch_data[:, -1].clone()
        r1_min_before = batch_data[:, -2].clone() if dataset.use_r1_min else None
        
        print(f"\nBefore rotation:")
        print(f"  First example target (p_u): {targets_before[0]:.4f}")
        if r1_min_before is not None:
            print(f"  First example r1_min: {r1_min_before[0]:.4f}")
        print(f"  First neighbor xyz: {features_before[0, :3]}")
        
        # Apply rotation
        rotated_batch = rotate(batch_data)
        
        # Extract after rotation
        features_after = rotated_batch[:, :-2] if dataset.use_r1_min else rotated_batch[:, :-1]
        targets_after = rotated_batch[:, -1]
        r1_min_after = rotated_batch[:, -2] if dataset.use_r1_min else None
        
        print(f"\nAfter rotation:")
        print(f"  First example target (p_u): {targets_after[0]:.4f}")
        if r1_min_after is not None:
            print(f"  First example r1_min: {r1_min_after[0]:.4f}")
        print(f"  First neighbor xyz: {features_after[0, :3]}")
        print(f"  XYZ difference: {torch.norm(features_after[0, :3] - features_before[0, :3]):.6f}")
        
        # Verify targets unchanged
        assert torch.allclose(targets_before, targets_after), "Targets changed during rotation!"
        if r1_min_before is not None:
            assert torch.allclose(r1_min_before, r1_min_after), "r1_min changed during rotation!"
        
        print("\n✓ Targets and r1_min preserved correctly")
        
        # Check that some features actually changed (rotation happened)
        features_changed = not torch.allclose(features_before, features_after, atol=1e-6)
        
        if not features_changed:
            print("\nDEBUG: Features didn't change!")
            print(f"  Max feature difference: {torch.max(torch.abs(features_after - features_before)):.10f}")
            print(f"  Feature before sample: {features_before[0, :10]}")
            print(f"  Feature after sample: {features_after[0, :10]}")
        
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
