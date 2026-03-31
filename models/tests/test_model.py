"""
Unit tests for GaussianPatchTransformer model.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from GaussianPatchTransformer import (
    GaussianPatchTransformer,
    GeodesicEncoderDecoderTransformer,
    GaussianPatchSplitTransformer,
    SplitEmbeddingEncoderDecoderTransformer,
    create_gaussian_patch_transformer
)
from utils import get_attribute_dim


class TestModelInitialization:
    """Test model initialization with various configurations."""

    def test_default_configuration(self):
        """Test model initialization with default configuration."""
        model = GaussianPatchTransformer()
        assert model.embed_dim == 384
        assert model.max_neighbors == 32
        assert model.use_point_token == True  # Default uses all point attributes
    
    def test_custom_configuration(self):
        """Test model initialization with custom configuration."""
        model = GaussianPatchTransformer(
            attributes=["xyz", "opacity"],
            max_neighbors=16,
            embed_dim=256,
            encoder_depth=4,
            num_heads=8
        )
        assert model.embed_dim == 256
        assert model.max_neighbors == 16
        assert model.attributes == ["xyz", "opacity"]
    
    def test_factory_function(self):
        """Test model creation using factory function."""
        model = create_gaussian_patch_transformer(
            config={'embed_dim': 192, 'max_neighbors': 24}
        )
        assert model.embed_dim == 192
        assert model.max_neighbors == 24
    
    def test_point_attributes_none(self):
        """Test that point_attributes=None uses all attributes."""
        model = GaussianPatchTransformer(
            attributes=["xyz", "opacity", "scale"],
            point_attributes=None
        )
        assert model.point_attributes == ["xyz", "opacity", "scale"]
        assert model.use_point_token == True
    
    def test_point_attributes_empty(self):
        """Test that point_attributes=[] disables point token."""
        model = GaussianPatchTransformer(
            attributes=["xyz", "opacity", "scale"],
            point_attributes=[]
        )
        assert model.point_attributes == []
        assert model.use_point_token == False
        assert model.point_encoder is None
    
    def test_point_attributes_subset(self):
        """Test point_attributes with subset of attributes."""
        model = GaussianPatchTransformer(
            attributes=["xyz", "opacity", "scale", "rotation"],
            point_attributes=["xyz", "opacity"]
        )
        assert model.point_attributes == ["xyz", "opacity"]
        assert model.use_point_token == True
        assert model.point_encoded_dim == 4  # xyz(3) + opacity(1)


class TestForwardPass:
    """Test forward pass with various inputs."""
    
    @pytest.fixture
    def model_setup(self):
        """Create model and dummy data for testing."""
        attributes = ["xyz", "opacity", "scale"]
        max_neighbors = 16
        batch_size = 4
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=128,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -3:] = False  # Mark last 3 as invalid
        
        return {
            'model': model,
            'neighborhood': neighborhood,
            'point_features': point_features,
            'valid_mask': valid_mask,
            'batch_size': batch_size
        }
    
    def test_forward_output_shape(self, model_setup):
        """Test that forward pass produces correct output shape."""
        model = model_setup['model']
        neighborhood = model_setup['neighborhood']
        point_features = model_setup['point_features']
        valid_mask = model_setup['valid_mask']
        batch_size = model_setup['batch_size']
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)
    
    def test_forward_non_negative_output(self, model_setup):
        """Test that predictions are non-negative (due to ReLU)."""
        model = model_setup['model']
        predictions = model(
            model_setup['neighborhood'],
            model_setup['point_features'],
            model_setup['valid_mask']
        )
        
        assert torch.all(predictions >= 0)
    
    def test_forward_with_embeddings(self, model_setup):
        """Test forward pass with return_embeddings=True."""
        model = model_setup['model']
        output = model(
            model_setup['neighborhood'],
            model_setup['point_features'],
            model_setup['valid_mask'],
            return_embeddings=True
        )
        
        assert isinstance(output, dict)
        assert 'prediction' in output
        assert 'neighbor_tokens' in output
        assert 'point_token' in output
        assert 'cls_output' in output
        assert 'max_pooled' in output
        assert 'attention_mask' in output
    
    def test_forward_without_point_token(self):
        """Test forward pass when point token is disabled."""
        attributes = ["xyz", "opacity"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            point_attributes=[],  # No point token
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)
    
    def test_forward_all_neighbors_valid(self):
        """Test forward pass when all neighbors are valid."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)


class TestAttentionMask:
    """Test attention masking functionality."""
    
    def test_attention_mask_shape(self):
        """Test that attention mask has correct shape."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -2:] = False
        
        output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
        
        # With point token: seq_len = 1 (CLS) + 1 (point) + max_neighbors
        expected_seq_len = 2 + max_neighbors
        assert output['attention_mask'].shape == (batch_size, expected_seq_len)
    
    def test_attention_mask_without_point(self):
        """Test attention mask shape without point token."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            point_attributes=[],  # No point token
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
        
        # Without point token: seq_len = 1 (CLS) + max_neighbors
        expected_seq_len = 1 + max_neighbors
        assert output['attention_mask'].shape == (batch_size, expected_seq_len)
    
    def test_attention_mask_values(self):
        """Test that attention mask correctly reflects valid_mask."""
        attributes = ["xyz"]
        max_neighbors = 4
        batch_size = 1
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_feature_dim = get_attribute_dim(attributes) + 1
        point_feature_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
        point_features = torch.randn(batch_size, point_feature_dim)
        valid_mask = torch.tensor([[True, True, False, False]])  # 2 valid, 2 invalid
        
        output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
        attn_mask = output['attention_mask']
        
        # CLS (0) and point (1) should be True
        assert attn_mask[0, 0] == True  # CLS
        assert attn_mask[0, 1] == True  # point


class TestLossAndMetrics:
    """Test loss computation and metrics."""
    
    @pytest.fixture
    def model(self):
        return GaussianPatchTransformer(
            attributes=["xyz", "opacity"],
            max_neighbors=16,
            embed_dim=128
        )
    
    def test_mse_loss(self, model):
        """Test MSE loss computation."""
        predictions = torch.rand(8, 1) * 10
        targets = torch.rand(8, 1) * 10
        
        loss = model.get_loss(predictions, targets, loss_type='mse')
        
        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar
    
    def test_l1_loss(self, model):
        """Test L1 loss computation."""
        predictions = torch.rand(8, 1) * 10
        targets = torch.rand(8, 1) * 10
        
        loss = model.get_loss(predictions, targets, loss_type='l1')
        
        assert loss.item() >= 0
    
    def test_smooth_l1_loss(self, model):
        """Test Smooth L1 loss computation."""
        predictions = torch.rand(8, 1) * 10
        targets = torch.rand(8, 1) * 10
        
        loss = model.get_loss(predictions, targets, loss_type='smooth_l1')
        
        assert loss.item() >= 0
    
    def test_metrics(self, model):
        """Test metrics computation."""
        predictions = torch.rand(16, 1) * 10
        targets = torch.rand(16, 1) * 10
        
        metrics = model.get_metrics(predictions, targets)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'relative_error_pct' in metrics
        assert 'max_error' in metrics
        
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['max_error'] >= 0
    
    def test_targets_1d(self, model):
        """Test that 1D targets are handled correctly."""
        predictions = torch.rand(8, 1) * 10
        targets = torch.rand(8) * 10  # 1D
        
        loss = model.get_loss(predictions, targets)
        
        assert loss.item() >= 0


class TestDifferentAttributeCombinations:
    """Test model with different attribute combinations."""
    
    @pytest.mark.parametrize("attributes", [
        ["xyz"],
        ["xyz", "opacity"],
        ["xyz", "scale"],
        ["xyz", "rotation"],
        ["xyz", "sh"],
        ["xyz", "opacity", "scale"],
        ["xyz", "opacity", "scale", "rotation"],
        ["xyz", "opacity", "scale", "rotation", "sh"]
    ])
    def test_attribute_combinations(self, attributes):
        """Test model with various attribute combinations."""
        batch_size = 2
        max_neighbors = 8
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        
        predictions = model(neighborhood, point_features, valid_mask)
        
        assert predictions.shape == (batch_size, 1)


class TestGradientFlow:
    """Test gradient flow through the model."""
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model."""
        attributes = ["xyz", "opacity"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        model.train()  # Ensure model is in training mode
        
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim, requires_grad=True)
        point_features = torch.randn(batch_size, point_dim, requires_grad=True)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        targets = torch.rand(batch_size, 1) * 10
        
        predictions = model(neighborhood, point_features, valid_mask)
        loss = model.get_loss(predictions, targets)
        loss.backward()
        
        # Check that loss is finite and computable
        assert torch.isfinite(loss), "Loss should be finite"
        
        # Check that at least some model parameters have gradients (not None)
        params_with_grads = sum(1 for p in model.parameters() if p.grad is not None)
        assert params_with_grads > 0, "At least some model parameters should have gradients"
    
    def test_all_parameters_have_gradients(self):
        """Test that all trainable parameters receive gradients."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        
        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        targets = torch.rand(batch_size, 1) * 10
        
        predictions = model(neighborhood, point_features, valid_mask)
        loss = model.get_loss(predictions, targets)
        loss.backward()
        
        has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        assert has_grad == total_params


class TestModelParameters:
    """Test model parameter counting."""
    
    def test_parameter_count(self):
        """Test that larger model has more parameters."""
        model_small = GaussianPatchTransformer(
            attributes=["xyz"],
            max_neighbors=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4
        )
        
        model_large = GaussianPatchTransformer(
            attributes=["xyz", "opacity", "scale"],
            max_neighbors=32,
            embed_dim=256,
            encoder_depth=6,
            num_heads=8
        )
        
        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())
        
        assert params_large > params_small


class TestGeodesicAttention:
    """Test geodesic self-attention support."""

    def _make_data(self, attributes, max_neighbors, batch_size):
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        # Make geodesic distances positive (last column)
        neighborhood[:, :, -1] = torch.abs(neighborhood[:, :, -1])
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -2:] = False
        return neighborhood, point_features, valid_mask

    def test_geodesic_attention_forward(self):
        """Test forward pass with geodesic attention."""
        attributes = ["xyz", "opacity", "scale"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            attention_type="geodesic",
        )

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        predictions = model(neighborhood, point_features, valid_mask)
        assert predictions.shape == (batch_size, 1)
        assert torch.all(predictions >= 0)

    def test_geodesic_attention_with_point_token_disabled(self):
        """Test geodesic attention when point token is disabled."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            point_attributes=[],
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            attention_type="geodesic",
        )

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        predictions = model(neighborhood, point_features, valid_mask)
        assert predictions.shape == (batch_size, 1)

    def test_geodesic_attention_combined_with_spatial_pe(self):
        """Test geodesic attention can combine with spatial positional encoding."""
        attributes = ["xyz", "opacity"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            pos_encoding_type="spatial",
            attention_type="geodesic",
        )

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        predictions = model(neighborhood, point_features, valid_mask)
        assert predictions.shape == (batch_size, 1)

    def test_geodesic_attention_combined_with_relative_bias(self):
        """Test geodesic attention combined with relative_bias (both biases added)."""
        attributes = ["xyz", "opacity"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            pos_encoding_type="relative_bias",
            attention_type="geodesic",
        )

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        predictions = model(neighborhood, point_features, valid_mask)
        assert predictions.shape == (batch_size, 1)

    def test_geodesic_attention_gradient_flow(self):
        """Test that gradients flow through geodesic attention layers."""
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            attention_type="geodesic",
        )
        model.train()

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        targets = torch.rand(batch_size, 1) * 10

        predictions = model(neighborhood, point_features, valid_mask)
        loss = model.get_loss(predictions, targets)
        loss.backward()

        # Check geodesic encoder blocks have gradients (V proj, temperature, output proj)
        for block in model.encoder_blocks:
            for name, p in block.attn.named_parameters():
                assert p.grad is not None, f"No gradient for encoder attn.{name}"

    def test_geodesic_attention_has_fewer_params(self):
        """Test that geodesic model has fewer params (no Q/K projections)."""
        kwargs = dict(
            attributes=["xyz"],
            max_neighbors=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
        )
        model_std = GaussianPatchTransformer(attention_type="standard", **kwargs)
        model_geo = GaussianPatchTransformer(attention_type="geodesic", **kwargs)

        params_std = sum(p.numel() for p in model_std.parameters())
        params_geo = sum(p.numel() for p in model_geo.parameters())
        # Geodesic uses only V projection (no Q, K), so fewer params
        assert params_geo < params_std

    def test_geodesic_attention_return_embeddings(self):
        """Test return_embeddings works with geodesic attention."""
        attributes = ["xyz", "opacity"]
        max_neighbors = 8
        batch_size = 2

        model = GaussianPatchTransformer(
            attributes=attributes,
            max_neighbors=max_neighbors,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            attention_type="geodesic",
        )

        neighborhood, point_features, valid_mask = self._make_data(
            attributes, max_neighbors, batch_size
        )
        output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
        assert isinstance(output, dict)
        assert "prediction" in output
        assert output["prediction"].shape == (batch_size, 1)

    def test_invalid_attention_type_raises(self):
        """Test that invalid attention_type raises ValueError."""
        with pytest.raises(ValueError, match="attention_type"):
            GaussianPatchTransformer(attention_type="invalid")

    def test_factory_with_geodesic_attention(self):
        """Test factory function with geodesic attention."""
        model = create_gaussian_patch_transformer(
            config={"attention_type": "geodesic", "embed_dim": 64, "encoder_depth": 2, "num_heads": 4}
        )
        assert model.attention_type == "geodesic"
        # All encoder blocks should be GeodesicTransformerEncoderBlock
        for block in model.encoder_blocks:
            assert type(block).__name__ == "GeodesicTransformerEncoderBlock"


class TestGeodesicEncoderDecoder:
    """Test GeodesicEncoderDecoderTransformer model."""

    def _make_data(self, attributes, max_neighbors, batch_size):
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        neighborhood[:, :, -1] = torch.abs(neighborhood[:, :, -1])  # geodesic >= 0
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -2:] = False
        return neighborhood, point_features, valid_mask

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        attributes = ["xyz", "opacity", "scale"]
        max_neighbors = 8
        batch_size = 2
        model = GeodesicEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)
        assert torch.all(pred >= 0)

    def test_requires_xyz(self):
        """Test that model raises if xyz not in attributes."""
        with pytest.raises(ValueError, match="xyz"):
            GeodesicEncoderDecoderTransformer(attributes=["opacity", "scale"])

    def test_return_embeddings(self):
        """Test return_embeddings mode."""
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GeodesicEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        out = model(nb, pf, mask, return_embeddings=True)
        assert "prediction" in out
        assert "encoder_output" in out
        assert "decoder_output" in out
        assert out["prediction"].shape == (batch_size, 1)
        assert out["encoder_output"].shape == (batch_size, max_neighbors, 64)

    def test_gradient_flow(self):
        """Test gradients flow through encoder and decoder."""
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GeodesicEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        model.train()
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        targets = torch.rand(batch_size, 1) * 10
        pred = model(nb, pf, mask)
        loss = model.get_loss(pred, targets)
        loss.backward()
        # Encoder and decoder blocks should have gradients
        for block in model.encoder_blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None, f"No grad for encoder.{name}"
        for block in model.decoder_blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None, f"No grad for decoder.{name}"

    def test_loss_and_metrics(self):
        """Test get_loss and get_metrics work."""
        model = GeodesicEncoderDecoderTransformer(
            attributes=["xyz"], embed_dim=64, encoder_depth=1,
            decoder_depth=1, num_heads=4, max_neighbors=6,
        )
        pred = torch.rand(4, 1)
        tgt = torch.rand(4, 1)
        loss = model.get_loss(pred, tgt, loss_type="mse")
        assert loss.shape == ()
        metrics = model.get_metrics(pred, tgt)
        assert "mae" in metrics and "rmse" in metrics

    def test_factory_creates_enc_dec(self):
        """Test factory creates GeodesicEncoderDecoderTransformer."""
        model = create_gaussian_patch_transformer(
            config={"attention_type": "geodesic_enc_dec", "embed_dim": 64,
                    "encoder_depth": 2, "decoder_depth": 2, "num_heads": 4}
        )
        assert model.attention_type == "geodesic_enc_dec"
        assert type(model).__name__ == "GeodesicEncoderDecoderTransformer"

    def test_all_neighbors_valid(self):
        """Test with all neighbors valid (no padding)."""
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GeodesicEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, decoder_depth=1, num_heads=4,
        )
        nb, pf, _ = self._make_data(attributes, max_neighbors, batch_size)
        mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)


class TestGaussianPatchSplitTransformer:
    """Test GaussianPatchSplitTransformer model (interleaved attr/geo tokens)."""

    def _make_data(self, attributes, max_neighbors, batch_size):
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        neighborhood[:, :, -1] = torch.abs(neighborhood[:, :, -1])  # geodesic >= 0
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -2:] = False
        return neighborhood, point_features, valid_mask

    def test_forward_output_shape(self):
        attributes = ["xyz", "opacity", "scale"]
        max_neighbors = 8
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)
        assert torch.all(pred >= 0)

    def test_forward_without_point_token(self):
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, point_attributes=[],
            max_neighbors=max_neighbors, embed_dim=64,
            encoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_return_embeddings(self):
        attributes = ["xyz", "opacity"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        out = model(nb, pf, mask, return_embeddings=True)
        assert isinstance(out, dict)
        assert "prediction" in out
        assert "attr_tokens" in out
        assert "geo_tokens" in out
        assert "cls_output" in out
        assert "encoder_output" in out
        assert "attention_mask" in out
        assert out["prediction"].shape == (batch_size, 1)
        assert out["attr_tokens"].shape == (batch_size, max_neighbors, 64)
        assert out["geo_tokens"].shape == (batch_size, max_neighbors, 64)

    def test_attention_mask_shape(self):
        attributes = ["xyz"]
        max_neighbors = 8
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        out = model(nb, pf, mask, return_embeddings=True)
        # seq_len = CLS + point + 2*N (attr+geo interleaved)
        expected_seq_len = 1 + 1 + 2 * max_neighbors
        assert out["attention_mask"].shape == (batch_size, expected_seq_len)

    def test_attention_mask_values(self):
        """Attribute tokens always attend; geodesic tokens follow valid_mask."""
        attributes = ["xyz"]
        max_neighbors = 4
        batch_size = 1
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, num_heads=4,
        )
        nb, pf, _ = self._make_data(attributes, max_neighbors, batch_size)
        mask = torch.tensor([[True, True, False, False]])
        out = model(nb, pf, mask, return_embeddings=True)
        attn_mask = out["attention_mask"][0]
        # First 2 tokens (CLS + point) are True
        assert attn_mask[0].item() is True
        assert attn_mask[1].item() is True
        # Interleaved: attr_1=True, geo_1=True, attr_2=True, geo_2=True,
        #              attr_3=True, geo_3=False, attr_4=True, geo_4=False
        # Note: sorted by distance so order may change, but attribute tokens are always True
        prefix_len = 2
        for i in range(max_neighbors):
            attr_idx = prefix_len + 2 * i
            assert attn_mask[attr_idx].item() is True  # attribute tokens always valid

    def test_gradient_flow(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
        )
        model.train()
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        targets = torch.rand(batch_size, 1) * 10
        pred = model(nb, pf, mask)
        loss = model.get_loss(pred, targets)
        loss.backward()
        for block in model.encoder_blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None, f"No grad for encoder.{name}"

    def test_type_embedding(self):
        """Verify type embedding has 3 entries."""
        model = GaussianPatchSplitTransformer(embed_dim=64)
        assert model.type_embed.num_embeddings == 3
        assert model.type_embed.embedding_dim == 64

    def test_pool_max_mean(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, num_heads=4, pool="max_mean",
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_spatial_pos_encoding(self):
        attributes = ["xyz", "opacity"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
            pos_encoding_type="spatial",
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_relative_bias_pos_encoding(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, num_heads=4,
            pos_encoding_type="relative_bias",
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_loss_and_metrics(self):
        model = GaussianPatchSplitTransformer(
            attributes=["xyz"], embed_dim=64, encoder_depth=1, num_heads=4,
            max_neighbors=6,
        )
        pred = torch.rand(4, 1)
        tgt = torch.rand(4, 1)
        loss = model.get_loss(pred, tgt, loss_type="mse")
        assert loss.shape == ()
        metrics = model.get_metrics(pred, tgt)
        assert "mae" in metrics and "rmse" in metrics

    def test_factory_creates_split(self):
        model = create_gaussian_patch_transformer(
            config={"attention_type": "split", "embed_dim": 64,
                    "encoder_depth": 2, "num_heads": 4}
        )
        assert model.attention_type == "split"
        assert type(model).__name__ == "GaussianPatchSplitTransformer"

    def test_all_neighbors_valid(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = GaussianPatchSplitTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, num_heads=4,
        )
        nb, pf, _ = self._make_data(attributes, max_neighbors, batch_size)
        mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_different_attribute_combos(self):
        for attrs in [["xyz"], ["xyz", "opacity"], ["xyz", "scale", "normals"]]:
            max_neighbors = 6
            batch_size = 2
            model = GaussianPatchSplitTransformer(
                attributes=attrs, max_neighbors=max_neighbors,
                embed_dim=64, encoder_depth=1, num_heads=4,
            )
            nb, pf, mask = self._make_data(attrs, max_neighbors, batch_size)
            pred = model(nb, pf, mask)
            assert pred.shape == (batch_size, 1)


class TestSplitEmbeddingEncoderDecoder:
    """Test SplitEmbeddingEncoderDecoderTransformer model."""

    def _make_data(self, attributes, max_neighbors, batch_size):
        neighbor_dim = get_attribute_dim(attributes) + 1
        point_dim = get_attribute_dim(attributes)
        neighborhood = torch.randn(batch_size, max_neighbors, neighbor_dim)
        neighborhood[:, :, -1] = torch.abs(neighborhood[:, :, -1])  # geodesic >= 0
        point_features = torch.randn(batch_size, point_dim)
        valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        valid_mask[:, -2:] = False
        return neighborhood, point_features, valid_mask

    def test_forward_output_shape(self):
        attributes = ["xyz", "opacity", "scale"]
        max_neighbors = 8
        batch_size = 2
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)
        assert torch.all(pred >= 0)

    def test_return_embeddings(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        out = model(nb, pf, mask, return_embeddings=True)
        assert "prediction" in out
        assert "encoder_output" in out
        assert "decoder_output" in out
        assert "cls_output" in out
        assert out["prediction"].shape == (batch_size, 1)
        assert out["encoder_output"].shape == (batch_size, max_neighbors, 64)

    def test_gradient_flow(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=2, decoder_depth=2, num_heads=4,
        )
        model.train()
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        targets = torch.rand(batch_size, 1) * 10
        pred = model(nb, pf, mask)
        loss = model.get_loss(pred, targets)
        loss.backward()
        for block in model.encoder_blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None, f"No grad for encoder.{name}"
        for block in model.decoder_blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None, f"No grad for decoder.{name}"

    def test_loss_and_metrics(self):
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=["xyz"], embed_dim=64, encoder_depth=1,
            decoder_depth=1, num_heads=4, max_neighbors=6,
        )
        pred = torch.rand(4, 1)
        tgt = torch.rand(4, 1)
        loss = model.get_loss(pred, tgt, loss_type="mse")
        assert loss.shape == ()
        metrics = model.get_metrics(pred, tgt)
        assert "mae" in metrics and "rmse" in metrics

    def test_factory_creates_split_enc_dec(self):
        model = create_gaussian_patch_transformer(
            config={"attention_type": "split_enc_dec", "embed_dim": 64,
                    "encoder_depth": 2, "decoder_depth": 2, "num_heads": 4}
        )
        assert model.attention_type == "split_enc_dec"
        assert type(model).__name__ == "SplitEmbeddingEncoderDecoderTransformer"

    def test_pool_max_mean(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, decoder_depth=1,
            num_heads=4, pool="max_mean",
        )
        nb, pf, mask = self._make_data(attributes, max_neighbors, batch_size)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_all_neighbors_valid(self):
        attributes = ["xyz"]
        max_neighbors = 6
        batch_size = 2
        model = SplitEmbeddingEncoderDecoderTransformer(
            attributes=attributes, max_neighbors=max_neighbors,
            embed_dim=64, encoder_depth=1, decoder_depth=1, num_heads=4,
        )
        nb, pf, _ = self._make_data(attributes, max_neighbors, batch_size)
        mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
        pred = model(nb, pf, mask)
        assert pred.shape == (batch_size, 1)

    def test_different_encoder_types(self):
        for enc_type in ["conv", "linear", "residual"]:
            model = SplitEmbeddingEncoderDecoderTransformer(
                attributes=["xyz"], max_neighbors=6,
                embed_dim=64, encoder_depth=1, decoder_depth=1,
                num_heads=4, encoder_type=enc_type,
            )
            nb, pf, mask = self._make_data(["xyz"], 6, 2)
            pred = model(nb, pf, mask)
            assert pred.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
