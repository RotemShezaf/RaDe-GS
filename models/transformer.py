"""
Transformer architecture for Gaussian Patch geodesic distance prediction.

This module contains encoder and decoder components designed for processing
Gaussian splatting patches to predict geodesic distances.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from typing import List, Optional, Tuple

try:
    from .utils import (
        get_attribute_dim,
        Attention,
        CrossAttention,
        FeedForward,
        PositionalEncoding,
        GeodesicSelfAttention,
    )
except ImportError:
    from utils import (
        get_attribute_dim,
        Attention,
        CrossAttention,
        FeedForward,
        PositionalEncoding,
        GeodesicSelfAttention,
    )


class GeodesicEmbedding(nn.Module):
    """
    Small MLP encoder for scalar geodesic distance values.

    Maps a single geodesic distance (dim=1) per neighbor to an
    embed_dim-sized token.

    Input:  (B, N, 1)
    Output: (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, geodesic_distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geodesic_distances: (batch_size, num_neighbors, 1)

        Returns:
            (batch_size, num_neighbors, embed_dim)
        """
        return self.encoder(geodesic_distances)


class GaussianPatchLinearEncoder(nn.Module):
    """
    Linear encoder for neighbor features in a Gaussian patch.

    Processes neighbor Gaussians (with geodesic distances) to generate
    token embeddings for each neighbor using fully-connected layers.
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384,
        include_geodesic: bool = True
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode (e.g., ["xyz", "opacity", "scale"])
            embed_dim: Dimension of output embeddings
            include_geodesic: Whether neighbor features include geodesic distance
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim
        self.include_geodesic = include_geodesic

        # Calculate input dimension
        input_dim = get_attribute_dim(attributes)
        if include_geodesic:
            input_dim += 1  # Add geodesic distance dimension

        # Encoder network: projects each neighbor's features to embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Encode neighbor features.

        Args:
            neighbor_features: Tensor of shape (batch_size, num_neighbors, feature_dim)

        Returns:
            Encoded features of shape (batch_size, num_neighbors, embed_dim)
        """
        B, N, F = neighbor_features.shape

        # Apply encoder to each neighbor
        encoded = self.encoder(neighbor_features)  # (B, N, embed_dim)

        return encoded


class GaussianPatchLinearEncoderResidual(nn.Module):
    """
    Residual MLP encoder for neighbor features in a Gaussian patch.

    Like GaussianPatchLinearEncoder but:
    - No LayerNorm
    - LeakyReLU activations instead of GELU
    - A 128→128→128 residual block between the first and second projection layers

    Input:  (B, N, feature_dim)
    Output: (B, N, embed_dim)
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384,
        include_geodesic: bool = True,
        negative_slope: float = 0.01,
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode
            embed_dim: Dimension of output embeddings
            include_geodesic: Whether neighbor features include geodesic distance
            negative_slope: Negative slope for LeakyReLU
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim
        self.include_geodesic = include_geodesic

        input_dim = get_attribute_dim(attributes)
        if include_geodesic:
            input_dim += 1

        self.fc1 = nn.Linear(input_dim, 128)
        # Residual block: 128 → 128 → 128
        self.res_fc1 = nn.Linear(128, 128)
        self.res_fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embed_dim)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: (batch_size, num_neighbors, feature_dim)

        Returns:
            (batch_size, num_neighbors, embed_dim)
        """
        x = self.act(self.fc1(neighbor_features))   # (B, N, 128)
        residual = x
        x = self.act(self.res_fc1(x))               # (B, N, 128)
        x = self.act(self.res_fc2(x) + residual)    # (B, N, 128) + skip
        x = self.act(self.fc2(x))                   # (B, N, 256)
        return self.fc3(x)                          # (B, N, embed_dim)


class GaussianPatchEncoder(nn.Module):
    """
    Conv1d-based encoder for neighbor features in a Gaussian patch.

    Uses a PointNet-style local aggregation: per-neighbor features are first
    encoded independently, then a global context vector (max-pooled across all
    neighbors) is concatenated back to each neighbor and refined, giving each
    token awareness of the full neighborhood.

    Input:  (B, N, feature_dim)
    Output: (B, N, embed_dim)
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384,
        include_geodesic: bool = True
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode
            embed_dim: Dimension of output embeddings
            include_geodesic: Whether neighbor features include geodesic distance
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim
        self.include_geodesic = include_geodesic

        input_dim = get_attribute_dim(attributes)
        if include_geodesic:
            input_dim += 1

        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(8, 128),  # padding-safe: per-sample norm, 8 groups of 16 channels
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.GroupNorm(16, 512),  # padding-safe: per-sample norm, 16 groups of 32 channels
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv1d(512, embed_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Encode neighbor features with global context injection.

        Args:
            neighbor_features: (batch_size, num_neighbors, feature_dim)

        Returns:
            (batch_size, num_neighbors, embed_dim)
        """
        B, N, _ = neighbor_features.shape
        x = neighbor_features.transpose(1, 2)            # (B, F, N)
        feature = self.first_conv(x)                     # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B, 256, 1)
        feature = torch.cat(
            [feature_global.expand(-1, -1, N), feature], dim=1
        )                                                # (B, 512, N)
        feature = self.second_conv(feature)              # (B, embed_dim, N)
        return feature.transpose(1, 2)                   # (B, N, embed_dim)


class PointFeatureLinearEncoder(nn.Module):
    """
    Linear encoder for the central point features in a Gaussian patch.

    Processes the central point's Gaussian features (without geodesic distance)
    to generate a token embedding using fully-connected layers.
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode
            embed_dim: Dimension of output embeddings
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim

        # Calculate input dimension (no geodesic distance for central point)
        input_dim = get_attribute_dim(attributes)

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Encode central point features.

        Args:
            point_features: Tensor of shape (batch_size, feature_dim)

        Returns:
            Encoded features of shape (batch_size, embed_dim)
        """
        encoded = self.encoder(point_features)  # (B, embed_dim)
        return encoded


class PointFeatureLinearResidualEncoder(nn.Module):
    """
    Residual MLP encoder for the central point features in a Gaussian patch.

    Like PointFeatureLinearEncoder but:
    - No LayerNorm
    - LeakyReLU activations instead of GELU
    - A 128→128→128 residual block between the first and second projection layers

    Input:  (B, feature_dim)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384,
        negative_slope: float = 0.01,
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode
            embed_dim: Dimension of output embeddings
            negative_slope: Negative slope for LeakyReLU
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim

        input_dim = get_attribute_dim(attributes)

        self.fc1 = nn.Linear(input_dim, 128)
        # Residual block: 128 → 128 → 128
        self.res_fc1 = nn.Linear(128, 128)
        self.res_fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embed_dim)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_features: (batch_size, feature_dim)

        Returns:
            (batch_size, embed_dim)
        """
        x = self.act(self.fc1(point_features))  # (B, 128)
        residual = x
        x = self.act(self.res_fc1(x))           # (B, 128)
        x = self.act(self.res_fc2(x) + residual)  # (B, 128) + skip
        x = self.act(self.fc2(x))               # (B, 256)
        return self.fc3(x)                      # (B, embed_dim)


class PointFeatureEncoder(nn.Module):
    """
    Conv1d-based encoder for the central point features in a Gaussian patch.

    Applies two Conv1d stages (treating the single point as a sequence of
    length 1) to produce a compact embedding consistent with the
    GaussianPatchEncoder's conv architecture.

    Input:  (B, feature_dim)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        attributes: List[str],
        embed_dim: int = 384
    ):
        """
        Args:
            attributes: List of Gaussian attributes to encode
            embed_dim: Dimension of output embeddings
        """
        super().__init__()
        self.attributes = attributes
        self.embed_dim = embed_dim

        input_dim = get_attribute_dim(attributes)

        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(8, 128),  # padding-safe: per-sample norm, 8 groups of 16 channels
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, embed_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Encode central point features.

        Args:
            point_features: (batch_size, feature_dim)

        Returns:
            (batch_size, embed_dim)
        """
        x = point_features.unsqueeze(2)   # (B, F, 1)
        x = self.first_conv(x)            # (B, 256, 1)
        x = self.second_conv(x)           # (B, embed_dim, 1)
        return x.squeeze(2)               # (B, embed_dim)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with self-attention and feed-forward network.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            attn_bias: Optional relative-position bias of shape
                       (batch_size, num_heads, seq_len, seq_len) added to
                       attention logits before softmax.
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), mask, attn_bias))
        
        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class GeodesicTransformerEncoderBlock(nn.Module):
    """Transformer encoder block using geodesic self-attention.

    Identical to :class:`TransformerEncoderBlock` except that
    standard Q·K^T attention is replaced by :class:`GeodesicSelfAttention`,
    which derives attention weights from a precomputed geodesic distance
    score (GDS) matrix.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GeodesicSelfAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        gds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor ``(B, S, embed_dim)``.
            mask: Attention mask ``(B, 1, 1, S)``.
            attn_bias: Optional per-head bias ``(B, H, S, S)``.
            gds: Geodesic distance score matrix ``(B, S, S)``.
        """
        x = x + self.drop_path(self.attn(self.norm1(x), gds, mask, attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder blocks for processing neighbor tokens.
    Uses sinusoidal positional encoding based on sequence order.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Args:
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Sinusoidal positional encoding based on sequence order
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process token sequence through transformer encoder.
        
        Args:
            x: Input tokens of shape (batch_size, seq_len, embed_dim)
               Tokens should be ordered by distance from source point.
            mask: Optional attention mask
            
        Returns:
            Encoded tokens of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        
        # Add positional encoding based on sequence order
        pos_encoding = self.pos_embed(seq_len)  # (1, seq_len, embed_dim)
        x = x + pos_encoding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block with self-attention, cross-attention, and feed-forward network.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = Attention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout
        )
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor of shape (batch_size, seq_len_q, embed_dim)
            context: Context tensor from encoder of shape (batch_size, seq_len_kv, embed_dim)
            self_attn_mask: Optional self-attention mask
            cross_attn_mask: Optional cross-attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.drop_path(self.self_attn(self.norm1(x), self_attn_mask))
        
        # Cross-attention with residual connection
        x = x + self.drop_path(self.cross_attn(self.norm2(x), context, cross_attn_mask))
        
        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of transformer decoder blocks for processing query tokens with context from encoder.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension
            depth: Number of transformer decoder blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process query tokens through transformer decoder.
        
        Args:
            x: Query tokens of shape (batch_size, seq_len_q, embed_dim)
            context: Context tokens from encoder of shape (batch_size, seq_len_kv, embed_dim)
            self_attn_mask: Optional self-attention mask
            cross_attn_mask: Optional cross-attention mask
            
        Returns:
            Decoded tokens of shape (batch_size, seq_len_q, embed_dim)
        """
        # Apply transformer decoder blocks
        for block in self.blocks:
            x = block(x, context, self_attn_mask, cross_attn_mask)
        
        x = self.norm(x)
        
        return x
