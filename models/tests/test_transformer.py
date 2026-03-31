"""
Unit tests for transformer components in transformer.py
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformer import (
    GaussianPatchEncoder,
    GaussianPatchLinearEncoder,
    PointFeatureEncoder,
    PointFeatureLinearEncoder,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderBlock,
    TransformerDecoderBlock
)


def test_gaussian_patch_encoder():
    """Test GaussianPatchEncoder (Conv1d-based)."""
    print("Testing GaussianPatchEncoder (Conv1d)...")

    batch_size = 4
    num_neighbors = 32
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
    embed_dim = 256

    encoder = GaussianPatchEncoder(
        attributes=attributes,
        embed_dim=embed_dim,
        include_geodesic=True
    )

    # Calculate feature dim (14 attributes + 1 geodesic)
    feature_dim = 3 + 1 + 3 + 4 + 3 + 1  # xyz + opacity + scale + rotation + sh + geodesic

    neighbor_features = torch.randn(batch_size, num_neighbors, feature_dim)
    output = encoder(neighbor_features)

    expected_shape = (batch_size, num_neighbors, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"  Input shape: {neighbor_features.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ GaussianPatchEncoder (Conv1d) test passed!")


def test_gaussian_patch_linear_encoder():
    """Test GaussianPatchLinearEncoder (Linear-based, original implementation)."""
    print("\nTesting GaussianPatchLinearEncoder (Linear)...")

    batch_size = 4
    num_neighbors = 32
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
    embed_dim = 256

    encoder = GaussianPatchLinearEncoder(
        attributes=attributes,
        embed_dim=embed_dim,
        include_geodesic=True
    )

    feature_dim = 3 + 1 + 3 + 4 + 3 + 1  # xyz + opacity + scale + rotation + sh + geodesic
    neighbor_features = torch.randn(batch_size, num_neighbors, feature_dim)
    output = encoder(neighbor_features)

    expected_shape = (batch_size, num_neighbors, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"  Input shape: {neighbor_features.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ GaussianPatchLinearEncoder (Linear) test passed!")


def test_point_feature_encoder():
    """Test PointFeatureEncoder (Conv1d-based)."""
    print("\nTesting PointFeatureEncoder (Conv1d)...")

    batch_size = 4
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
    embed_dim = 256

    encoder = PointFeatureEncoder(
        attributes=attributes,
        embed_dim=embed_dim
    )

    feature_dim = 3 + 1 + 3 + 4 + 3  # xyz + opacity + scale + rotation + sh
    point_features = torch.randn(batch_size, feature_dim)
    output = encoder(point_features)

    expected_shape = (batch_size, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"  Input shape: {point_features.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ PointFeatureEncoder (Conv1d) test passed!")


def test_point_feature_linear_encoder():
    """Test PointFeatureLinearEncoder (Linear-based, original implementation)."""
    print("\nTesting PointFeatureLinearEncoder (Linear)...")

    batch_size = 4
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
    embed_dim = 256

    encoder = PointFeatureLinearEncoder(
        attributes=attributes,
        embed_dim=embed_dim
    )

    feature_dim = 3 + 1 + 3 + 4 + 3  # xyz + opacity + scale + rotation + sh
    point_features = torch.randn(batch_size, feature_dim)
    output = encoder(point_features)

    expected_shape = (batch_size, embed_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"  Input shape: {point_features.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ PointFeatureLinearEncoder (Linear) test passed!")


def test_transformer_encoder_block():
    """Test TransformerEncoderBlock."""
    print("\nTesting TransformerEncoderBlock...")
    
    batch_size = 4
    seq_len = 32
    embed_dim = 256
    num_heads = 8
    
    block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.1
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = block(x)
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    print(f"  Input/Output shape: {x.shape}")
    print("✓ TransformerEncoderBlock test passed!")


def test_transformer_decoder_block():
    """Test TransformerDecoderBlock."""
    print("\nTesting TransformerDecoderBlock...")
    
    batch_size = 4
    query_len = 8
    context_len = 32
    embed_dim = 256
    num_heads = 8
    
    block = TransformerDecoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.1
    )
    
    query = torch.randn(batch_size, query_len, embed_dim)
    context = torch.randn(batch_size, context_len, embed_dim)
    
    output = block(query, context)
    
    assert output.shape == query.shape, f"Expected shape {query.shape}, got {output.shape}"
    
    print(f"  Query shape: {query.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ TransformerDecoderBlock test passed!")


def test_transformer_encoder():
    """Test TransformerEncoder."""
    print("\nTesting TransformerEncoder...")
    
    batch_size = 4
    seq_len = 32
    embed_dim = 256
    depth = 4
    num_heads = 8
    
    encoder = TransformerEncoder(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path_rate=0.1,
        max_seq_len=100
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test forward pass
    output = encoder(x)
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of blocks: {len(encoder.blocks)}")
    print("✓ TransformerEncoder test passed!")


def test_transformer_decoder():
    """Test TransformerDecoder."""
    print("\nTesting TransformerDecoder...")
    
    batch_size = 4
    query_len = 8
    context_len = 32
    embed_dim = 256
    depth = 3
    num_heads = 8
    
    decoder = TransformerDecoder(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        drop_path_rate=0.1
    )
    
    query = torch.randn(batch_size, query_len, embed_dim)
    context = torch.randn(batch_size, context_len, embed_dim)
    
    output = decoder(query, context)
    
    assert output.shape == query.shape, f"Expected shape {query.shape}, got {output.shape}"
    
    print(f"  Query shape: {query.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of blocks: {len(decoder.blocks)}")
    print("✓ TransformerDecoder test passed!")


def test_transformer_pipeline():
    """Test complete encoder-decoder pipeline (Conv1d encoders)."""
    print("\nTesting complete transformer pipeline (Conv1d encoders)...")

    batch_size = 4
    num_neighbors = 32
    embed_dim = 256
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]

    # Create Conv1d encoders
    neighbor_encoder = GaussianPatchEncoder(
        attributes=attributes,
        embed_dim=embed_dim,
        include_geodesic=True
    )

    point_encoder = PointFeatureEncoder(
        attributes=attributes,
        embed_dim=embed_dim
    )
    
    # Create transformer encoder/decoder
    transformer_encoder = TransformerEncoder(
        embed_dim=embed_dim,
        depth=4,
        num_heads=8
    )
    
    transformer_decoder = TransformerDecoder(
        embed_dim=embed_dim,
        depth=2,
        num_heads=8
    )
    
    # Create dummy data
    neighbor_features = torch.randn(batch_size, num_neighbors, 15)  # 14 attrs + 1 geodesic
    point_features = torch.randn(batch_size, 14)  # 14 attrs, no geodesic
    
    # Process through pipeline
    neighbor_tokens = neighbor_encoder(neighbor_features)
    query_token = point_encoder(point_features).unsqueeze(1)
    
    encoded_neighbors = transformer_encoder(neighbor_tokens)
    decoded_query = transformer_decoder(query_token, encoded_neighbors)
    
    # Check shapes
    assert neighbor_tokens.shape == (batch_size, num_neighbors, embed_dim)
    assert query_token.shape == (batch_size, 1, embed_dim)
    assert encoded_neighbors.shape == (batch_size, num_neighbors, embed_dim)
    assert decoded_query.shape == (batch_size, 1, embed_dim)
    
    print(f"  Neighbor tokens shape: {neighbor_tokens.shape}")
    print(f"  Query token shape: {query_token.shape}")
    print(f"  Encoded neighbors shape: {encoded_neighbors.shape}")
    print(f"  Decoded query shape: {decoded_query.shape}")
    print("✓ Transformer pipeline test passed!")


def test_transformer_pipeline_linear():
    """Test complete encoder-decoder pipeline (Linear encoders)."""
    print("\nTesting complete transformer pipeline (Linear encoders)...")

    batch_size = 4
    num_neighbors = 32
    embed_dim = 256
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]

    neighbor_encoder = GaussianPatchLinearEncoder(
        attributes=attributes,
        embed_dim=embed_dim,
        include_geodesic=True
    )

    point_encoder = PointFeatureLinearEncoder(
        attributes=attributes,
        embed_dim=embed_dim
    )

    transformer_encoder = TransformerEncoder(
        embed_dim=embed_dim,
        depth=4,
        num_heads=8
    )

    transformer_decoder = TransformerDecoder(
        embed_dim=embed_dim,
        depth=2,
        num_heads=8
    )

    neighbor_features = torch.randn(batch_size, num_neighbors, 15)  # 14 attrs + 1 geodesic
    point_features = torch.randn(batch_size, 14)  # 14 attrs, no geodesic

    neighbor_tokens = neighbor_encoder(neighbor_features)
    query_token = point_encoder(point_features).unsqueeze(1)

    encoded_neighbors = transformer_encoder(neighbor_tokens)
    decoded_query = transformer_decoder(query_token, encoded_neighbors)

    assert neighbor_tokens.shape == (batch_size, num_neighbors, embed_dim)
    assert query_token.shape == (batch_size, 1, embed_dim)
    assert encoded_neighbors.shape == (batch_size, num_neighbors, embed_dim)
    assert decoded_query.shape == (batch_size, 1, embed_dim)

    print(f"  Neighbor tokens shape: {neighbor_tokens.shape}")
    print(f"  Query token shape: {query_token.shape}")
    print(f"  Encoded neighbors shape: {encoded_neighbors.shape}")
    print(f"  Decoded query shape: {decoded_query.shape}")
    print("✓ Transformer pipeline (Linear) test passed!")


def run_all_tests():
    """Run all transformer component tests."""
    print("=" * 80)
    print("Running transformer.py tests...")
    print("=" * 80)

    test_gaussian_patch_encoder()
    test_gaussian_patch_linear_encoder()
    test_point_feature_encoder()
    test_point_feature_linear_encoder()
    test_transformer_encoder_block()
    test_transformer_decoder_block()
    test_transformer_encoder()
    test_transformer_decoder()
    test_transformer_pipeline()
    test_transformer_pipeline_linear()
    
    print("\n" + "=" * 80)
    print("All transformer tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
