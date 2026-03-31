"""
Unit tests for utility functions in utils.py
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    get_attribute_dim,
    get_atribues_indices,
    get_attribute_indices,
    Attention,
    CrossAttention,
    FeedForward,
    PositionalEncoding,
    sinusoidal,
    SpatialPositionalEncoding,
    compute_geodesic_distance_scores,
    GeodesicSelfAttention,
)


class TestGetAttributeDim:
    """Test attribute dimension calculation."""
    
    def test_single_xyz(self):
        assert get_attribute_dim(["xyz"]) == 3
    
    def test_single_opacity(self):
        assert get_attribute_dim(["opacity"]) == 1
    
    def test_single_scale(self):
        assert get_attribute_dim(["scale"]) == 3
    
    def test_single_rotation(self):
        assert get_attribute_dim(["rotation"]) == 4
    
    def test_single_sh(self):
        assert get_attribute_dim(["sh"]) == 3
    
    def test_multiple_attributes(self):
        assert get_attribute_dim(["xyz", "opacity"]) == 4
        assert get_attribute_dim(["xyz", "scale", "rotation"]) == 10
        assert get_attribute_dim(["xyz", "opacity", "scale", "rotation", "sh"]) == 14
    
    def test_unknown_attribute(self):
        """Unknown attributes are ignored (not contributing to dimension)."""
        # Unknown attributes don't add to dimension
        assert get_attribute_dim(["xyz", "unknown"]) == 3  # Only xyz counts


class TestGetAttributeIndices:
    """Test attribute indices calculation."""
    
    def test_xyz_indices(self):
        all_attrs = ["xyz", "opacity", "scale"]
        indices = get_atribues_indices(["xyz"], all_attrs, include_geodesic=False)
        assert indices == [0, 1, 2]
    
    def test_opacity_indices(self):
        all_attrs = ["xyz", "opacity", "scale"]
        indices = get_atribues_indices(["opacity"], all_attrs, include_geodesic=False)
        assert indices == [3]
    
    def test_multiple_indices(self):
        all_attrs = ["xyz", "opacity", "scale"]
        indices = get_atribues_indices(["xyz", "opacity"], all_attrs, include_geodesic=False)
        assert indices == [0, 1, 2, 3]


class TestAttention:
    """Test Attention module."""
    
    def test_attention_output_shape(self):
        batch_size = 4
        seq_len = 16
        dim = 256
        num_heads = 8
        
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout=0.1,
            proj_dropout=0.1
        )
        
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_attention_with_mask(self):
        batch_size = 4
        seq_len = 16
        dim = 256
        num_heads = 8
        
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True
        )
        
        x = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, -4:] = 0  # Mask out last 4 positions
        
        output = attn(x, mask)
        assert output.shape == (batch_size, seq_len, dim)


class TestCrossAttention:
    """Test CrossAttention module."""
    
    def test_cross_attention_output_shape(self):
        batch_size = 4
        query_len = 8
        context_len = 16
        dim = 256
        num_heads = 8
        
        cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_dropout=0.1,
            proj_dropout=0.1
        )
        
        query = torch.randn(batch_size, query_len, dim)
        context = torch.randn(batch_size, context_len, dim)
        
        output = cross_attn(query, context)
        
        assert output.shape == (batch_size, query_len, dim)


class TestFeedForward:
    """Test FeedForward module."""
    
    def test_feedforward_output_shape(self):
        batch_size = 4
        seq_len = 16
        dim = 256
        mlp_ratio = 4.0
        
        ff = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=0.1
        )
        
        x = torch.randn(batch_size, seq_len, dim)
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, dim)


class TestPositionalEncoding:
    """Test PositionalEncoding module."""
    
    def test_positional_encoding_shape(self):
        seq_len = 32
        embed_dim = 256
        
        pos_enc = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=100
        )
        
        pos_embed = pos_enc(seq_len)
        
        assert pos_embed.shape == (1, seq_len, embed_dim)
    
    def test_positional_encoding_different_lengths(self):
        embed_dim = 128
        
        pos_enc = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=100
        )
        
        for seq_len in [8, 16, 32, 64]:
            pos_embed = pos_enc(seq_len)
            assert pos_embed.shape == (1, seq_len, embed_dim)


class TestSinusoidal:
    """Test sinusoidal positional encoding function."""

    def test_output_shape_1d(self):
        """Scalar positions: output is (N, features, 2)."""
        positions = torch.linspace(-1., 1., 32)
        out = sinusoidal(positions, features=16)
        assert out.shape == (32, 16, 2)

    def test_output_shape_2d(self):
        """2-D positions: output is (B, N, features, 2)."""
        positions = torch.randn(4, 10)
        out = sinusoidal(positions, features=8)
        assert out.shape == (4, 10, 8, 2)

    def test_output_shape_3d(self):
        """3-D XYZ positions: output is (B, N, 3, features, 2)."""
        positions = torch.randn(4, 16, 3)
        out = sinusoidal(positions, features=16)
        assert out.shape == (4, 16, 3, 16, 2)

    def test_output_dtype_float(self):
        """Float input keeps its dtype."""
        positions = torch.randn(8, 3, dtype=torch.float32)
        out = sinusoidal(positions, features=4)
        assert out.dtype == torch.float32

    def test_sin_cos_on_unit_circle(self):
        """sin^2 + cos^2 == 1 for all frequencies."""
        positions = torch.randn(10)
        out = sinusoidal(positions, features=8)  # (10, 8, 2)
        norms = (out[..., 0] ** 2 + out[..., 1] ** 2)  # (10, 8)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_custom_periods(self):
        """Different period base produces different encodings."""
        positions = torch.randn(5)
        out1 = sinusoidal(positions, features=8, periods=10000)
        out2 = sinusoidal(positions, features=8, periods=100)
        assert not torch.allclose(out1, out2)

    def test_different_features(self):
        """Output feature dimension matches the features argument."""
        positions = torch.randn(6)
        for f in [4, 8, 16, 32]:
            out = sinusoidal(positions, features=f)
            assert out.shape[-2] == f


class TestSpatialPositionalEncoding:
    """Test SpatialPositionalEncoding module."""

    def test_output_shape(self):
        """Basic forward pass produces (B, N, embed_dim)."""
        B, N, embed_dim = 4, 32, 128
        spe = SpatialPositionalEncoding(embed_dim=embed_dim)
        rel_xyz = torch.randn(B, N, 3)
        out = spe(rel_xyz)
        assert out.shape == (B, N, embed_dim)

    def test_output_shape_single_point(self):
        """Works for N=1 (single neighbour)."""
        spe = SpatialPositionalEncoding(embed_dim=64)
        rel_xyz = torch.randn(2, 1, 3)
        out = spe(rel_xyz)
        assert out.shape == (2, 1, 64)

    def test_custom_pe_features(self):
        """Custom pe_features changes raw dim but output stays embed_dim."""
        spe = SpatialPositionalEncoding(embed_dim=256, pe_features=8)
        rel_xyz = torch.randn(3, 16, 3)
        out = spe(rel_xyz)
        assert out.shape == (3, 16, 256)

    def test_proj_raw_dim(self):
        """Projection weight has shape (embed_dim, 3*pe_features*2)."""
        pe_features = 12
        embed_dim = 128
        spe = SpatialPositionalEncoding(embed_dim=embed_dim, pe_features=pe_features)
        expected_in = 3 * pe_features * 2
        assert spe.proj.in_features == expected_in
        assert spe.proj.out_features == embed_dim

    def test_different_inputs_different_outputs(self):
        """Different relative positions produce different encodings."""
        spe = SpatialPositionalEncoding(embed_dim=64)
        spe.eval()
        rel_xyz1 = torch.randn(2, 8, 3)
        rel_xyz2 = torch.randn(2, 8, 3)
        with torch.no_grad():
            out1 = spe(rel_xyz1)
            out2 = spe(rel_xyz2)
        assert not torch.allclose(out1, out2)

    def test_zero_position_deterministic(self):
        """Zero relative position always gives the same output regardless of batch."""
        spe = SpatialPositionalEncoding(embed_dim=64)
        spe.eval()
        zeros = torch.zeros(4, 8, 3)
        with torch.no_grad():
            out = spe(zeros)  # (4, 8, 64)
        # Each row in the batch should be identical
        assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])

    def test_gradients_flow(self):
        """Gradients propagate back through the projection layer."""
        spe = SpatialPositionalEncoding(embed_dim=32)
        rel_xyz = torch.randn(2, 8, 3, requires_grad=True)
        # rel_xyz is integer-free so sinusoidal grads are well-defined
        out = spe(rel_xyz)
        loss = out.sum()
        loss.backward()
        assert spe.proj.weight.grad is not None

    def test_batch_independence(self):
        """Each sample in the batch is encoded independently."""
        spe = SpatialPositionalEncoding(embed_dim=64)
        spe.eval()
        rel_xyz = torch.randn(4, 8, 3)
        with torch.no_grad():
            out_batch = spe(rel_xyz)                    # (4, 8, 64)
            out_single = spe(rel_xyz[2:3])              # (1, 8, 64)
        assert torch.allclose(out_batch[2:3], out_single, atol=1e-5)


class TestComputeGeodesicDistanceScores:
    """Tests for compute_geodesic_distance_scores utility function."""

    def test_output_shape(self):
        """Output shape is (B, S, S)."""
        xyz = torch.randn(2, 10, 3)
        gds = compute_geodesic_distance_scores(xyz)
        assert gds.shape == (2, 10, 10)

    def test_self_distance_zero(self):
        """Diagonal entries (self-distance) should be 0."""
        xyz = torch.randn(3, 8, 3)
        gds = compute_geodesic_distance_scores(xyz)
        for b in range(3):
            diag = torch.diagonal(gds[b])
            assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_symmetry(self):
        """GDS matrix should be symmetric (undirected graph)."""
        xyz = torch.randn(2, 12, 3)
        gds = compute_geodesic_distance_scores(xyz)
        assert torch.allclose(gds, gds.transpose(1, 2), atol=1e-5)

    def test_non_negative(self):
        """All GDS values should be non-negative."""
        xyz = torch.randn(2, 8, 3)
        gds = compute_geodesic_distance_scores(xyz)
        assert (gds >= -1e-5).all()

    def test_mask_excludes_tokens(self):
        """Masked tokens should have INF distance to all other tokens."""
        xyz = torch.randn(1, 6, 3)
        mask = torch.tensor([[True, True, True, True, False, False]])
        gds = compute_geodesic_distance_scores(xyz, mask)
        # Distances to/from masked tokens should be INF (except self)
        assert (gds[0, :4, 4] >= 1e5).all()
        assert (gds[0, :4, 5] >= 1e5).all()
        # Self-distance of masked tokens is 0
        assert gds[0, 4, 4].item() == 0.0
        assert gds[0, 5, 5].item() == 0.0

    def test_collocated_points_zero_distance(self):
        """Points at the same position should have GDS = 0."""
        xyz = torch.zeros(1, 4, 3)  # All at origin
        gds = compute_geodesic_distance_scores(xyz)
        assert torch.allclose(gds, torch.zeros_like(gds), atol=1e-5)

    def test_batch_independence(self):
        """Each batch element computed independently."""
        xyz = torch.randn(4, 8, 3)
        gds_batch = compute_geodesic_distance_scores(xyz)
        gds_single = compute_geodesic_distance_scores(xyz[2:3])
        assert torch.allclose(gds_batch[2:3], gds_single, atol=1e-5)


class TestGeodesicSelfAttention:
    """Tests for GeodesicSelfAttention module."""

    def test_output_shape(self):
        """Output has same shape as input."""
        gsa = GeodesicSelfAttention(dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        gds = compute_geodesic_distance_scores(torch.randn(2, 10, 3))
        out = gsa(x, gds)
        assert out.shape == (2, 10, 64)

    def test_gradient_flow(self):
        """Gradients flow through V projection and temperature."""
        gsa = GeodesicSelfAttention(dim=32, num_heads=4)
        x = torch.randn(2, 6, 32, requires_grad=True)
        gds = compute_geodesic_distance_scores(torch.randn(2, 6, 3)).detach()
        out = gsa(x, gds)
        out.sum().backward()
        assert x.grad is not None
        assert gsa.v_proj.weight.grad is not None
        assert gsa.log_temperature.grad is not None
        assert gsa.proj.weight.grad is not None

    def test_with_mask(self):
        """Forward pass with masking produces valid output."""
        gsa = GeodesicSelfAttention(dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        xyz = torch.randn(2, 8, 3)
        mask_bool = torch.ones(2, 8, dtype=torch.bool)
        mask_bool[:, -2:] = False
        gds = compute_geodesic_distance_scores(xyz, mask_bool).detach()
        attn_mask = mask_bool.unsqueeze(1).unsqueeze(2).float()
        out = gsa(x, gds, mask=attn_mask)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_with_attn_bias(self):
        """Forward pass with additional attention bias."""
        num_heads = 4
        gsa = GeodesicSelfAttention(dim=64, num_heads=num_heads)
        x = torch.randn(2, 6, 64)
        gds = compute_geodesic_distance_scores(torch.randn(2, 6, 3)).detach()
        bias = torch.randn(2, num_heads, 6, 6)
        out = gsa(x, gds, attn_bias=bias)
        assert out.shape == (2, 6, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
