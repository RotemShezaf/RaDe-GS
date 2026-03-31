"""
Integration tests for GaussianPatchTransformer with dataset.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from GaussianPatchTransformer import GaussianPatchTransformer, create_gaussian_patch_transformer
from utils import get_attribute_dim


class TestModelWithDatasetFormat:
    """Test model works correctly with the new dataset format."""
    
    def test_forward_with_dataset_shapes(self):
        """Test model with shapes matching dataset output."""
        # Simulate dataset output
        batch_size = 8
        max_neighbors = 32
        attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1  # +1 for geodesic
        point_feature_dim = get_attribute_dim(attributes)
        
        # Create tensors matching dataset format
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        targets = torch.rand(batch_size) * 10
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -10:] = False  # Some invalid neighbors
        
        # Create model
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=256,
            encoder_depth=4,
            num_heads=8
        )
        
        # Forward pass
        predictions = model(neighborhood, point_features, valid_mask)
        
        # Compute loss
        loss = model.get_loss(predictions, targets)
        
        assert predictions.shape == (batch_size, 1)
        assert loss.item() >= 0
    
    def test_training_step(self):
        """Test a complete training step with gradients."""
        batch_size = 4
        max_neighbors = 16
        attributes = ["xyz", "opacity", "scale"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        # Create model
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=128,
            encoder_depth=2,
            num_heads=4
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create batch
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        targets = torch.rand(batch_size) * 10
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predictions = model(neighborhood, point_features, valid_mask)
        loss = model.get_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_evaluation_step(self):
        """Test evaluation with metrics computation."""
        batch_size = 4
        max_neighbors = 16
        attributes = ["xyz", "opacity"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=128,
            encoder_depth=2,
            num_heads=4
        )
        
        # Create batch
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        targets = torch.rand(batch_size) * 10
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(neighborhood, point_features, valid_mask)
            metrics = model.get_metrics(predictions, targets)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0


class TestMaskedPooling:
    """Test that masked pooling works correctly."""
    
    def test_pooling_ignores_invalid(self):
        """Test that max pooling correctly ignores invalid tokens."""
        batch_size = 2
        max_neighbors = 8
        attributes = ["xyz"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        
        # All valid
        valid_mask_all = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        output_all = model(neighborhood, point_features, valid_mask_all, return_embeddings=True)
        
        # Some invalid - should still produce valid output
        valid_mask_partial = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask_partial[:, -4:] = False
        output_partial = model(neighborhood, point_features, valid_mask_partial, return_embeddings=True)
        
        # Both should produce valid predictions
        assert output_all['prediction'].shape == (batch_size, 1)
        assert output_partial['prediction'].shape == (batch_size, 1)
        assert not torch.isnan(output_all['prediction']).any()
        assert not torch.isnan(output_partial['prediction']).any()


class TestPointAttributesVariations:
    """Test different point_attributes configurations."""
    
    def test_no_point_token(self):
        """Test model without point token."""
        batch_size = 2
        max_neighbors = 8
        attributes = ["xyz", "opacity", "scale"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            point_attributes=[],  # No point token
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        assert not model.use_point_token
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)
        assert not torch.isnan(predictions).any()
    
    def test_partial_point_attributes(self):
        """Test model with partial point attributes."""
        batch_size = 2
        max_neighbors = 8
        attributes = ["xyz", "opacity", "scale", "rotation"]
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            point_attributes=["xyz", "opacity"],  # Only subset
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        assert model.use_point_token
        assert model.point_encoded_dim == 4  # xyz(3) + opacity(1)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)


class TestNeighborSorting:
    """Test neighbor sorting by Euclidean distance."""
    
    def test_neighbors_get_sorted(self):
        """Test that neighbors are sorted by Euclidean distance."""
        batch_size = 1
        max_neighbors = 4
        attributes = ["xyz"]
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        # Create point at origin
        point_features = torch.zeros(batch_size, 3)
        
        # Create neighbors at known distances
        # neighbor 0: distance 3
        # neighbor 1: distance 1
        # neighbor 2: distance 2
        # neighbor 3: distance 4
        neighborhood = torch.zeros(batch_size, max_neighbors, 4)  # xyz + geodesic
        neighborhood[0, 0, :3] = torch.tensor([3.0, 0.0, 0.0])  # dist 3
        neighborhood[0, 1, :3] = torch.tensor([1.0, 0.0, 0.0])  # dist 1
        neighborhood[0, 2, :3] = torch.tensor([2.0, 0.0, 0.0])  # dist 2
        neighborhood[0, 3, :3] = torch.tensor([4.0, 0.0, 0.0])  # dist 4
        
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
        
        # sorted_indices should be [1, 2, 0, 3] (sorted by distance)
        expected_order = torch.tensor([[1, 2, 0, 3]])
        assert torch.equal(output['sorted_indices'], expected_order)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
