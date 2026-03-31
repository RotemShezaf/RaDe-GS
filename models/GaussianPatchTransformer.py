"""
GaussianPatchTransformer: Encoder-only Transformer for predicting geodesic distances
on Gaussian splatting patches.

This model processes a Gaussian patch containing:
- Neighborhood features: (batch, max_neighbors, entry_size) with geodesic distances
- Point features: (batch, point_feature_size) without geodesic distance
- Valid mask: (batch, max_neighbors) boolean mask indicating valid neighbors

And predicts the geodesic distance for the central point.

Architecture:
- Encoder-only transformer with CLS token (similar to PointTransformer/BERT)
- Neighbors sorted by Euclidean distance from central point
- Standard sinusoidal positional encoding based on sorted order
- Attention masking for padded/invalid neighbors
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from typing import List, Optional, Dict, Any, Tuple

try:
    from transformer import (
        GaussianPatchEncoder,
        GaussianPatchLinearEncoder,
        GaussianPatchLinearEncoderResidual,
        PointFeatureEncoder,
        PointFeatureLinearEncoder,
        PointFeatureLinearResidualEncoder,
        TransformerEncoderBlock,
        GeodesicTransformerEncoderBlock,
        TransformerDecoderBlock,
        GeodesicEmbedding,
    )
    from utils import get_attribute_dim, get_attributes_indices, PositionalEncoding, SpatialPositionalEncoding, RelativePositionBias, compute_geodesic_distance_scores, ATTRIBUTE_DIMS
except ImportError:
    from .transformer import (
        GaussianPatchEncoder,
        GaussianPatchLinearEncoder,
        GaussianPatchLinearEncoderResidual,
        PointFeatureEncoder,
        PointFeatureLinearEncoder,
        PointFeatureLinearResidualEncoder,
        TransformerEncoderBlock,
        GeodesicTransformerEncoderBlock,
        TransformerDecoderBlock,
        GeodesicEmbedding,
    )
    from .utils import get_attribute_dim, get_attributes_indices, PositionalEncoding, SpatialPositionalEncoding, RelativePositionBias, compute_geodesic_distance_scores, ATTRIBUTE_DIMS


class GaussianPatchTransformer(nn.Module):
    """
    Encoder-only Transformer architecture for geodesic distance prediction on Gaussian patches.
    
    Architecture:
    1. Encode central point features -> point token (optional, based on point_attributes)
    2. Encode neighbor features (with geodesic distances) -> neighbor tokens
    3. Sort neighbors by Euclidean distance from central point
    4. Prepend learnable CLS token
    5. Apply sinusoidal positional encoding based on sorted order
    6. Process sequence through transformer encoder with attention masking
    7. Use CLS token output + max-pooled features (masked) to predict geodesic distance
    """
    
    def __init__(
        self,
        attributes: List[str] = ["xyz"],
        point_attributes: Optional[List[str]] = None,
        max_neighbors: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "max",
        pos_encoding_type: str = "index",
        encoder_type: str = "conv",
        attention_type: str = "standard",
        use_cls_token: bool = True,
    ):
        """
        Initialize GaussianPatchTransformer.
        
        Args:
            attributes: List of Gaussian attributes for neighbors (e.g., ["xyz", "opacity", "scale"])
            point_attributes: List of Gaussian attributes for point. If None, uses all attributes.
                             If empty list [], point token is not used.
            max_neighbors: Maximum number of neighbors in a patch
            embed_dim: Dimension of token embeddings
            encoder_depth: Number of transformer encoder blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            pool: Pooling strategy for neighbor tokens: "max", "mean", or "max_mean".
                  "max_mean" concatenates both, giving the head 3×embed_dim input.
            pos_encoding_type: Positional encoding type.
                  "index"   – sinusoidal encoding of sorted neighbor order (default).
                  "spatial" – sinusoidal encoding of relative 3-D XYZ position;
                               requires "xyz" in attributes.
                  "relative_bias" – learned per-head bias added to attention logits
                               at every layer based on pairwise relative 3-D positions;
                               requires "xyz" in attributes.
            encoder_type: Which encoder implementation to use for both neighbor and point encoders.
                  "conv"     – Conv1d PointNet-style encoder with global context (default).
                  "linear"   – MLP (Linear + LayerNorm + GELU) encoder, original implementation.
                  "residual" – MLP with residual block (no LayerNorm, LeakyReLU).
            attention_type: Type of self-attention mechanism.
                  "standard" – vanilla multi-head self-attention (default).
                  "geodesic" – geodesic self-attention (NeurIPS 2022): replaces
                               Q·K^T with attention weights from a Graph-based
                               Geodesic Distance Score matrix computed on-the-fly
                               from token XYZ positions (K-NN graph + Floyd-Warshall
                               shortest paths).  Requires "xyz" in attributes.
                               Can be combined with any positional encoding type.
            use_cls_token: Whether to prepend a learnable [CLS] token to the sequence.
                  True  – prediction head receives [CLS, pool] features (default).
                  False – no CLS token; prediction head receives only pooled features.
        """
        super().__init__()

        if encoder_type not in ("conv", "linear", "residual"):
            raise ValueError(f"encoder_type must be 'conv', 'linear', or 'residual', got '{encoder_type}'")
        self.encoder_type = encoder_type

        if attention_type not in ("standard", "geodesic"):
            raise ValueError(f"attention_type must be 'standard' or 'geodesic', got '{attention_type}'")
        self.attention_type = attention_type
        
        if pool not in ("max", "mean", "max_mean"):
            raise ValueError(f"pool must be 'max', 'mean', or 'max_mean', got '{pool}'")
        self.pool = pool

        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
        
        # Handle point_attributes
        if point_attributes is None:
            self.point_attributes = attributes  # Use all attributes by default
        else:
            self.point_attributes = point_attributes
        
        self.use_point_token = len(self.point_attributes) > 0
        self.use_cls_token = use_cls_token
        
        # Calculate dimensions
        self.neighbor_feature_dim = get_attribute_dim(attributes) + 1  # +1 for geodesic distance
        self.point_feature_dim = get_attribute_dim(attributes)  # Full point features from dataset
        self.point_encoded_dim = get_attribute_dim(self.point_attributes) if self.use_point_token else 0
        
        # Compute indices for xyz (for computing Euclidean distance to sort neighbors)
        self.xyz_indices = self._compute_attributes_indices(["xyz"])
        
        # Compute indices for extracting point_attributes from full point features
        if self.use_point_token:
            self.point_attr_indices = get_attributes_indices(
                self.point_attributes, 
                attributes, 
                include_geodesic=False
            )
        else:
            self.point_attr_indices = []
        
        # Neighbor encoder: encodes neighbor features with geodesic distances
        NeighborEncoderCls = {
            "conv":     GaussianPatchEncoder,
            "linear":   GaussianPatchLinearEncoder,
            "residual": GaussianPatchLinearEncoderResidual,
        }[encoder_type]
        self.neighbor_encoder = NeighborEncoderCls(
            attributes=attributes,
            embed_dim=embed_dim,
            include_geodesic=True
        )

        # Point encoder: encodes central point features (only if point_attributes is not empty)
        PointEncoderCls = {
            "conv":     PointFeatureEncoder,
            "linear":   PointFeatureLinearEncoder,
            "residual": PointFeatureLinearResidualEncoder,
        }[encoder_type]
        if self.use_point_token:
            self.point_encoder = PointEncoderCls(
                attributes=self.point_attributes,
                embed_dim=embed_dim
            )
        else:
            self.point_encoder = None
        
        # CLS token (learnable) – only when enabled
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        
        # Positional encoding
        if pos_encoding_type not in ("index", "spatial", "relative_bias"):
            raise ValueError(f"pos_encoding_type must be 'index', 'spatial', or 'relative_bias', got '{pos_encoding_type}'")
        self.pos_encoding_type = pos_encoding_type

        # Sequence: [(CLS), (point), neighbor_1, neighbor_2, ..., neighbor_N]
        # Total length: (1 if use_cls_token) + (1 if use_point_token) + max_neighbors
        max_seq_len = (1 if self.use_cls_token else 0) + (1 if self.use_point_token else 0) + max_neighbors
        # Index-based sinusoidal PE (always kept as fallback)
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len + 1  # +1 for safety margin
        )
        # Spatial PE: encodes relative 3-D XYZ of each neighbour (additive, once)
        if pos_encoding_type == "spatial":
            if "xyz" not in attributes:
                raise ValueError("pos_encoding_type='spatial' requires 'xyz' in attributes")
            self.spatial_pos_embed = SpatialPositionalEncoding(embed_dim=embed_dim)
        else:
            self.spatial_pos_embed = None

        # Relative position bias: per-head bias injected at every attention layer
        if pos_encoding_type == "relative_bias":
            if "xyz" not in attributes:
                raise ValueError("pos_encoding_type='relative_bias' requires 'xyz' in attributes")
            self.rel_pos_bias = RelativePositionBias(num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # Transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        if attention_type == "geodesic":
            if "xyz" not in attributes:
                raise ValueError("attention_type='geodesic' requires 'xyz' in attributes")
            BlockClass = GeodesicTransformerEncoderBlock
        else:
            BlockClass = TransformerEncoderBlock
        self.encoder_blocks = nn.ModuleList([
            BlockClass(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i]
            )
            for i in range(encoder_depth)
        ])
        
        # Layer normalization after encoder
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Prediction head input size depends on CLS token and pool strategy:
        #   With CLS:    max/mean -> [cls, pool] = 2D,  max_mean -> [cls, max, mean] = 3D
        #   Without CLS: max/mean -> [pool]      = 1D,  max_mean -> [max, mean]      = 2D
        if self.use_cls_token:
            head_input_dim = embed_dim * 3 if pool == "max_mean" else embed_dim * 2
        else:
            head_input_dim = embed_dim * 2 if pool == "max_mean" else embed_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()  # Geodesic distance is strictly non-negative; Softplus is smooth and bounded below by 0
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
    
    def _compute_attributes_indices(self, target_attributes: List[str]) -> Dict[str, tuple]:
        """
        Compute the indices of specified attributes in the feature vector.
        
        Args:
            target_attributes: List of attribute names to find indices for
            
        Returns:
            Dictionary mapping attribute name to (start_idx, end_idx) tuple
        """
        indices = {}
        current_idx = 0
        
        for attr in self.attributes:
            attr_size = 3 if attr in ["xyz", "scale", "sh", "normals"] else (4 if attr == "rotation" else 1)
            
            if attr in target_attributes:
                indices[attr] = (current_idx, current_idx + attr_size)
            
            current_idx += attr_size
        
        return indices
    
    def _init_weights(self, m):
        """Initialize weights using truncated normal distribution."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _sort_neighbors_by_distance(
        self,
        neighbor_features: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort neighbors by Euclidean distance from point.
        
        Args:
            neighbor_features: (batch, max_neighbors, entry_size)
            point_features: (batch, point_feature_size)
            valid_mask: (batch, max_neighbors) boolean mask
            
        Returns:
            Tuple of:
            - sorted_neighbor_features: (batch, max_neighbors, entry_size)
            - sorted_valid_mask: (batch, max_neighbors)
            - sorted_indices: (batch, max_neighbors)
        """
        batch_size = neighbor_features.shape[0]
        device = neighbor_features.device
        
        if "xyz" in self.xyz_indices:
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            point_xyz = point_features[:, xyz_start:xyz_end]  # (B, 3)
            neighbor_xyz = neighbor_features[:, :, xyz_start:xyz_end]  # (B, N, 3)
            
            # Compute Euclidean distances: (B, N)
            euclidean_distances = torch.norm(
                neighbor_xyz - point_xyz.unsqueeze(1), 
                dim=-1
            )
            
            # Set invalid neighbors to large distance so they sort to end
            large_distance = 1e10
            euclidean_distances = torch.where(
                valid_mask,
                euclidean_distances,
                torch.full_like(euclidean_distances, large_distance)
            )
            
            # Get sorted indices
            sorted_indices = torch.argsort(euclidean_distances, dim=1)  # (B, N)
            
            # Gather sorted neighbor features
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, neighbor_features.shape[-1])
            sorted_neighbor_features = torch.gather(neighbor_features, 1, expanded_indices)
            
            # Gather sorted valid mask
            sorted_valid_mask = torch.gather(valid_mask, 1, sorted_indices)
        else:
            # If xyz not available, use original order
            sorted_neighbor_features = neighbor_features
            sorted_valid_mask = valid_mask
            sorted_indices = torch.arange(self.max_neighbors, device=device).unsqueeze(0).expand(batch_size, -1)
        
        return sorted_neighbor_features, sorted_valid_mask, sorted_indices
    
    def _create_attention_mask(
        self,
        valid_mask: torch.Tensor,
        has_point_token: bool
    ) -> torch.Tensor:
        """
        Create attention mask for transformer.
        
        The sequence is: [(CLS), (point), neighbor_1, ..., neighbor_N]
        CLS and point tokens should always attend to each other and valid neighbors.
        Invalid neighbors should not be attended to.
        
        Args:
            valid_mask: (batch, max_neighbors) boolean mask, True for valid
            has_point_token: Whether point token is included in sequence
            
        Returns:
            attention_mask: (batch, seq_len) boolean mask for attention
                           True = attend, False = don't attend
        """
        batch_size = valid_mask.shape[0]
        device = valid_mask.device
        
        parts = []
        
        # CLS token is always valid (if enabled)
        if self.use_cls_token:
            parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        
        if has_point_token:
            # Point token is always valid
            parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        
        parts.append(valid_mask)
        full_mask = torch.cat(parts, dim=1)
        
        return full_mask
    
    def forward(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for geodesic distance prediction.
        
        Args:
            neighborhood: Neighbor features of shape (batch_size, max_neighbors, entry_size)
                         Each entry contains [attributes..., geodesic_distance]
            point_features: Point features of shape (batch_size, point_feature_dim)
                           Contains only attributes (no geodesic distance)
            valid_mask: Boolean mask of shape (batch_size, max_neighbors)
                       True for valid neighbors, False for masked/padded
            return_embeddings: If True, return intermediate embeddings for analysis
            
        Returns:
            If return_embeddings is False:
                Predicted geodesic distances of shape (batch_size, 1)
            If return_embeddings is True:
                Dict containing predictions and intermediate embeddings
        """
        batch_size = neighborhood.shape[0]
        device = neighborhood.device
        
        # Sort neighbors by Euclidean distance from point
        sorted_neighbors, sorted_valid_mask, sorted_indices = self._sort_neighbors_by_distance(
            neighborhood, point_features, valid_mask
        )
        
        # Encode neighbor features -> neighbor tokens
        neighbor_tokens = self.neighbor_encoder(sorted_neighbors)  # (B, N, embed_dim)
        
        # Build token sequence
        token_parts = []
        
        # Optionally prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
            token_parts.append(cls_tokens)
        
        if self.use_point_token:
            # Extract point attributes for encoding
            point_attrs = point_features[:, self.point_attr_indices]  # (B, point_encoded_dim)
            point_token = self.point_encoder(point_attrs)  # (B, embed_dim)
            point_token = point_token.unsqueeze(1)  # (B, 1, embed_dim)
            token_parts.append(point_token)
        else:
            point_token = None
        
        token_parts.append(neighbor_tokens)
        tokens = torch.cat(token_parts, dim=1)
        
        # Number of prefix tokens (before neighbors) for slicing later
        n_prefix = (1 if self.use_cls_token else 0) + (1 if self.use_point_token else 0)
        
        seq_len = tokens.shape[1]
        
        # Add positional encoding
        attn_bias = None  # will be set for relative_bias mode
        if self.pos_encoding_type == "index":
            pos_encoding = self.pos_embed(seq_len)  # (1, seq_len, embed_dim)
            tokens = tokens + pos_encoding
        elif self.pos_encoding_type == "spatial":
            # spatial: encode relative 3-D XYZ position of each neighbour
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            neighbor_xyz = sorted_neighbors[:, :, xyz_start:xyz_end]  # (B, N, 3)
            point_xyz = point_features[:, xyz_start:xyz_end]  # (B, 3)
            rel_xyz = neighbor_xyz - point_xyz.unsqueeze(1)  # (B, N, 3)
            spatial_pe = self.spatial_pos_embed(rel_xyz)  # (B, N, embed_dim)
            # Prefix tokens (CLS and/or point) get zero positional encoding
            if n_prefix > 0:
                prefix_pe = torch.zeros(batch_size, n_prefix, self.embed_dim, device=device)
                full_pe = torch.cat([prefix_pe, spatial_pe], dim=1)  # (B, seq_len, embed_dim)
            else:
                full_pe = spatial_pe
            tokens = tokens + full_pe
        else:  # relative_bias
            # Build absolute 3-D positions for every token in the sequence.
            # CLS and point token are placed at the central point (rel = 0).
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            neighbor_xyz = sorted_neighbors[:, :, xyz_start:xyz_end]  # (B, N, 3)
            point_xyz = point_features[:, xyz_start:xyz_end]          # (B, 3)
            # Use relative positions (central point = origin)
            rel_neighbor = neighbor_xyz - point_xyz.unsqueeze(1)  # (B, N, 3)
            origin = torch.zeros(batch_size, 1, 3, device=device)
            prefix_positions = [origin] * n_prefix  # CLS and/or point at origin
            if prefix_positions:
                positions = torch.cat(prefix_positions + [rel_neighbor], dim=1)
            else:
                positions = rel_neighbor
            attn_bias = self.rel_pos_bias(positions)  # (B, num_heads, S, S)
        
        # Create attention mask
        attn_mask = self._create_attention_mask(sorted_valid_mask, self.use_point_token)  # (B, seq_len)
        
        # Safety: without CLS token (which was always valid), samples with 0
        # valid neighbors would produce an all-masked attention row, causing
        # softmax(-inf, ..., -inf) = NaN.  Force the first position valid so
        # attention always has at least one key to attend to.
        if not self.use_cls_token:
            no_valid = ~attn_mask.any(dim=1)  # (B,) samples with zero valid
            if no_valid.any():
                attn_mask[no_valid, 0] = True
        
        # Convert to attention matrix mask format for broadcasting
        # Attention expects (B, seq_len, seq_len) or broadcastable shape
        # In attention: mask==0 positions get -inf, so we want True->1, False->0
        # Shape (B, 1, 1, seq_len) broadcasts to (B, num_heads, seq_len, seq_len)
        attn_mask_2d = attn_mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, seq_len)
        
        # Compute Geodesic Distance Scores for geodesic attention
        gds_matrix = None
        if self.attention_type == "geodesic":
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            neighbor_xyz = sorted_neighbors[:, :, xyz_start:xyz_end]  # (B, N, 3)
            point_xyz = point_features[:, xyz_start:xyz_end]          # (B, 3)
            rel_neighbor = neighbor_xyz - point_xyz.unsqueeze(1)      # (B, N, 3)
            origin = torch.zeros(batch_size, 1, 3, device=device)
            prefix_positions = [origin] * n_prefix
            if prefix_positions:
                positions = torch.cat(prefix_positions + [rel_neighbor], dim=1)
            else:
                positions = rel_neighbor
            gds_matrix = compute_geodesic_distance_scores(positions, attn_mask).detach()

        # Process through transformer encoder blocks
        for block in self.encoder_blocks:
            if self.attention_type == "geodesic":
                tokens = block(tokens, attn_mask_2d, attn_bias, gds=gds_matrix)
            else:
                tokens = block(tokens, attn_mask_2d, attn_bias)
        
        # Apply layer normalization
        tokens = self.encoder_norm(tokens)
        
        # Extract CLS output (if present) and neighbor tokens for pooling
        if self.use_cls_token:
            cls_output = tokens[:, 0, :]  # CLS token output: (B, embed_dim)
        else:
            cls_output = None
        
        # Get tokens for pooling (exclude prefix tokens)
        pool_tokens = tokens[:, n_prefix:, :]  # Neighbor outputs: (B, N, embed_dim)
        pool_mask = sorted_valid_mask  # (B, N)
        
        # Compute requested pool(s) over valid neighbor tokens
        any_valid = pool_mask.any(dim=1)  # (B,) – True if at least one valid neighbour
        # Fallback value when all neighbours are invalid: use CLS if available, else zeros
        fallback = cls_output if cls_output is not None else torch.zeros(batch_size, self.embed_dim, device=device)

        if self.pool in ("max", "max_mean"):
            pool_tokens_max = pool_tokens.clone()
            pool_tokens_max[~pool_mask] = float('-inf')
            max_pooled = torch.max(pool_tokens_max, dim=1)[0]  # (B, embed_dim)
            if (~any_valid).any():
                max_pooled[~any_valid] = fallback[~any_valid]

        if self.pool in ("mean", "max_mean"):
            pool_tokens_mean = pool_tokens.clone()
            pool_tokens_mean[~pool_mask] = 0.0
            valid_counts = pool_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pooled = pool_tokens_mean.sum(dim=1) / valid_counts  # (B, embed_dim)
            if (~any_valid).any():
                mean_pooled[~any_valid] = fallback[~any_valid]

        # Build combined features for prediction head
        parts = []
        if cls_output is not None:
            parts.append(cls_output)
        if self.pool == "max":
            parts.append(max_pooled)
        elif self.pool == "mean":
            parts.append(mean_pooled)
        else:  # max_mean
            parts.append(max_pooled)
            parts.append(mean_pooled)
        combined_features = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        
        # Predict geodesic distance
        prediction = self.prediction_head(combined_features)  # (B, 1)
        
        if return_embeddings:
            return {
                'prediction': prediction,
                'neighbor_tokens': neighbor_tokens,
                'point_token': point_token.squeeze(1) if self.use_point_token else None,
                'cls_output': cls_output,
                'max_pooled': max_pooled if self.pool in ("max", "max_mean") else None,
                'encoder_output': tokens,
                'sorted_indices': sorted_indices,
                'attention_mask': attn_mask
            }
        else:
            return prediction
    
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = 'mse',
        huber_delta: float = 1.0,
        blended_mse_lambda: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Predicted geodesic distances of shape (batch_size, 1)
            targets: Ground truth geodesic distances of shape (batch_size,) or (batch_size, 1)
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1', 'huber', 'log_mse', 'log_l1', 'blended_mse')
            huber_delta: Delta for Huber loss (transition point from quadratic to linear)
            blended_mse_lambda: Weight for linear-space MSE in blended_mse (1-lambda used for log-space MSE)
            
        Returns:
            Scalar loss value
        """
        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Compute loss
        if loss_type == 'mse':
            loss = nn.functional.mse_loss(predictions, targets)
        elif loss_type == 'l1':
            loss = nn.functional.l1_loss(predictions, targets)
        elif loss_type == 'smooth_l1':
            loss = nn.functional.smooth_l1_loss(predictions, targets)
        elif loss_type == 'huber':
            # Huber loss: quadratic for |error| <= delta, linear beyond.
            # Small errors train normally while large-error outliers get capped
            # gradients, preventing them from dominating the batch loss.
            loss = nn.functional.huber_loss(predictions, targets, delta=huber_delta)
        elif loss_type == 'log_mse':
            # MSE in log-distance space: penalises relative error equally across scales.
            # log(0) is undefined, so clamp to a small epsilon.
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt  = torch.log(targets.clamp(min=1e-8))
            loss = nn.functional.mse_loss(log_pred, log_tgt)
        elif loss_type == 'log_l1':
            # L1 in log-distance space.
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt  = torch.log(targets.clamp(min=1e-8))
            loss = nn.functional.l1_loss(log_pred, log_tgt)
        elif loss_type == 'blended_mse':
            # Blended MSE: lambda * linear-space MSE + (1-lambda) * log-space MSE.
            # Linear MSE gives strong gradients for large errors while
            # log-space MSE penalises relative errors equally at all
            # distance scales, improving accuracy for near-source points.
            # blended_mse_lambda=0.5 (default) gives equal weight to both.
            mse_linear = nn.functional.mse_loss(predictions, targets)
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt  = torch.log(targets.clamp(min=1e-8))
            mse_log = nn.functional.mse_loss(log_pred, log_tgt)
            loss = blended_mse_lambda * mse_linear + (1.0 - blended_mse_lambda) * mse_log
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def get_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Predicted geodesic distances of shape (batch_size, 1)
            targets: Ground truth geodesic distances of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Dictionary of metrics
        """
        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        with torch.no_grad():
            # Mean Absolute Error
            mae = torch.mean(torch.abs(predictions - targets)).item()

            #Mean Squared Error
            mse = torch.mean((predictions - targets) ** 2).item()
            
            # Root Mean Squared Error
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
            
            # Relative error (percentage)
            relative_error = torch.mean(
                torch.abs(predictions - targets) / (targets + 1e-8)
            ).item() * 100
            
            # Max error
            max_error = torch.max(torch.abs(predictions - targets)).item()
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'relative_error_pct': relative_error,
            'max_error': max_error
        }
    
    def print_model_info(self):
        """Print model architecture information."""
        print("=" * 80)
        print("GaussianPatchTransformer Model Information (Encoder-Only)")
        print("=" * 80)
        print(f"Neighbor attributes: {self.attributes}")
        print(f"Point attributes: {self.point_attributes}")
        print(f"Use point token: {self.use_point_token}")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Neighbor feature dim: {self.neighbor_feature_dim}")
        print(f"Point feature dim (full): {self.point_feature_dim}")
        print(f"Point feature dim (encoded): {self.point_encoded_dim}")
        print()
        print(f"Encoder depth: {self.encoder_depth}")
        print(f"Number of attention heads: {self.num_heads}")
        print(f"Attention type: {self.attention_type}")
        seq_len = 1 + (1 if self.use_point_token else 0) + self.max_neighbors
        print(f"Sequence length: 1 (CLS) + {'1 (point) + ' if self.use_point_token else ''}{self.max_neighbors} (neighbors) = {seq_len}")
        print(f"Positional encoding: Sinusoidal (based on sorted order by Euclidean distance)")
        print()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 80)


class GaussianPatchSplitTransformer(nn.Module):
    """
    Encoder-only Transformer with split attribute/geodesic embeddings.

    Unlike :class:`GaussianPatchTransformer` which encodes all neighbor
    features (attributes + geodesic) into a single token per neighbor,
    this model creates **two tokens per neighbor**:

    * **Attribute token** – embedded from all Gaussian attributes
      (xyz, opacity, scale, …) *excluding* geodesic distance.
    * **Geodesic token** – embedded from the scalar geodesic distance alone.

    The tokens are interleaved in the sequence:

        [CLS, (point?), attr_1, geo_1, attr_2, geo_2, …, attr_N, geo_N]

    A learnable **type embedding** distinguishes attribute tokens from
    geodesic tokens.

    The attention mask is applied **only to geodesic tokens** (attribute
    tokens always attend, even for padded neighbors).
    """

    def __init__(
        self,
        attributes: List[str] = ["xyz"],
        point_attributes: Optional[List[str]] = None,
        max_neighbors: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "max",
        pos_encoding_type: str = "index",
        encoder_type: str = "conv",
        attention_type: str = "split",
    ):
        super().__init__()

        if encoder_type not in ("conv", "linear", "residual"):
            raise ValueError(f"encoder_type must be 'conv', 'linear', or 'residual', got '{encoder_type}'")
        self.encoder_type = encoder_type
        self.attention_type = attention_type

        if pool not in ("max", "mean", "max_mean"):
            raise ValueError(f"pool must be 'max', 'mean', or 'max_mean', got '{pool}'")
        self.pool = pool

        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads

        # Handle point_attributes
        if point_attributes is None:
            self.point_attributes = attributes
        else:
            self.point_attributes = point_attributes
        self.use_point_token = len(self.point_attributes) > 0

        # Dimensions (same as GaussianPatchTransformer for interface compat)
        self.neighbor_feature_dim = get_attribute_dim(attributes) + 1
        self.point_feature_dim = get_attribute_dim(attributes)
        self.point_encoded_dim = get_attribute_dim(self.point_attributes) if self.use_point_token else 0

        # XYZ indices for sorting
        self.xyz_indices = self._compute_attributes_indices(["xyz"])

        # Point attribute indices
        if self.use_point_token:
            self.point_attr_indices = get_attributes_indices(
                self.point_attributes, attributes, include_geodesic=False
            )
        else:
            self.point_attr_indices = []

        # ── Attribute encoder (all attrs *without* geodesic) ──────────
        NeighborEncoderCls = {
            "conv":     GaussianPatchEncoder,
            "linear":   GaussianPatchLinearEncoder,
            "residual": GaussianPatchLinearEncoderResidual,
        }[encoder_type]
        self.attribute_encoder = NeighborEncoderCls(
            attributes=attributes,
            embed_dim=embed_dim,
            include_geodesic=False,
        )

        # ── Geodesic encoder (scalar → embed_dim) ────────────────────
        self.geodesic_encoder = GeodesicEmbedding(embed_dim=embed_dim)

        # ── Point encoder (optional) ─────────────────────────────────
        PointEncoderCls = {
            "conv":     PointFeatureEncoder,
            "linear":   PointFeatureLinearEncoder,
            "residual": PointFeatureLinearResidualEncoder,
        }[encoder_type]
        if self.use_point_token:
            self.point_encoder = PointEncoderCls(attributes=self.point_attributes, embed_dim=embed_dim)
        else:
            self.point_encoder = None

        # ── CLS token ────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Learnable type embeddings (attribute vs geodesic) ────────
        # type 0 = CLS/point, type 1 = attribute token, type 2 = geodesic token
        self.type_embed = nn.Embedding(3, embed_dim)

        # ── Positional encoding ──────────────────────────────────────
        if pos_encoding_type not in ("index", "spatial", "relative_bias"):
            raise ValueError(f"pos_encoding_type must be 'index', 'spatial', or 'relative_bias', got '{pos_encoding_type}'")
        self.pos_encoding_type = pos_encoding_type

        # seq: [CLS, (point?), attr_1, geo_1, …, attr_N, geo_N]
        max_seq_len = 1 + (1 if self.use_point_token else 0) + 2 * max_neighbors
        self.pos_embed = PositionalEncoding(embed_dim=embed_dim, max_len=max_seq_len + 1)

        if pos_encoding_type == "spatial":
            if "xyz" not in attributes:
                raise ValueError("pos_encoding_type='spatial' requires 'xyz' in attributes")
            self.spatial_pos_embed = SpatialPositionalEncoding(embed_dim=embed_dim)
        else:
            self.spatial_pos_embed = None

        if pos_encoding_type == "relative_bias":
            if "xyz" not in attributes:
                raise ValueError("pos_encoding_type='relative_bias' requires 'xyz' in attributes")
            self.rel_pos_bias = RelativePositionBias(num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # ── Transformer encoder blocks (standard attention) ──────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ── Prediction head ──────────────────────────────────────────
        head_input_dim = embed_dim * 3 if pool == "max_mean" else embed_dim * 2
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    def _compute_attributes_indices(self, target_attributes: List[str]) -> Dict[str, tuple]:
        indices = {}
        current_idx = 0
        for attr in self.attributes:
            attr_size = 3 if attr in ["xyz", "scale", "sh", "normals"] else (4 if attr == "rotation" else 1)
            if attr in target_attributes:
                indices[attr] = (current_idx, current_idx + attr_size)
            current_idx += attr_size
        return indices

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _sort_neighbors_by_distance(
        self,
        neighbor_features: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = neighbor_features.shape[0]
        device = neighbor_features.device
        if "xyz" in self.xyz_indices:
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            point_xyz = point_features[:, xyz_start:xyz_end]
            neighbor_xyz = neighbor_features[:, :, xyz_start:xyz_end]
            euclidean_distances = torch.norm(neighbor_xyz - point_xyz.unsqueeze(1), dim=-1)
            euclidean_distances = torch.where(valid_mask, euclidean_distances, torch.full_like(euclidean_distances, 1e10))
            sorted_indices = torch.argsort(euclidean_distances, dim=1)
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, neighbor_features.shape[-1])
            sorted_neighbor_features = torch.gather(neighbor_features, 1, expanded_indices)
            sorted_valid_mask = torch.gather(valid_mask, 1, sorted_indices)
        else:
            sorted_neighbor_features = neighbor_features
            sorted_valid_mask = valid_mask
            sorted_indices = torch.arange(self.max_neighbors, device=device).unsqueeze(0).expand(batch_size, -1)
        return sorted_neighbor_features, sorted_valid_mask, sorted_indices

    # ------------------------------------------------------------------
    def _create_split_attention_mask(
        self, valid_mask: torch.Tensor, has_point_token: bool
    ) -> torch.Tensor:
        """Build attention mask for the interleaved sequence.

        Layout: [CLS, (point?), attr_1, geo_1, attr_2, geo_2, …]

        * CLS / point tokens → always True
        * Attribute tokens   → always True (even for padded neighbors)
        * Geodesic tokens    → True only for valid neighbors

        Returns:
            ``(B, seq_len)`` boolean mask.
        """
        B, N = valid_mask.shape
        device = valid_mask.device

        prefix = [torch.ones(B, 1, dtype=torch.bool, device=device)]  # CLS
        if has_point_token:
            prefix.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        # Interleave: attr_i → True, geo_i → valid_mask[:, i]
        attr_valid = torch.ones(B, N, dtype=torch.bool, device=device)
        # Stack (B, N, 2) then reshape to (B, 2*N)
        interleaved = torch.stack([attr_valid, valid_mask], dim=2).reshape(B, 2 * N)

        full_mask = torch.cat(prefix + [interleaved], dim=1)
        return full_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        B = neighborhood.shape[0]
        N = self.max_neighbors
        device = neighborhood.device

        # Sort neighbors by Euclidean distance
        sorted_neighbors, sorted_valid_mask, sorted_indices = \
            self._sort_neighbors_by_distance(neighborhood, point_features, valid_mask)

        # ── Encode attribute tokens (exclude geodesic = last col) ────
        attr_features = sorted_neighbors[:, :, :-1]        # (B, N, attr_dim)
        attr_tokens = self.attribute_encoder(attr_features)  # (B, N, D)

        # ── Encode geodesic tokens ───────────────────────────────────
        geo_features = sorted_neighbors[:, :, -1:]          # (B, N, 1)
        geo_tokens = self.geodesic_encoder(geo_features)    # (B, N, D)

        # ── Interleave: (B, 2N, D) ──────────────────────────────────
        interleaved = torch.stack([attr_tokens, geo_tokens], dim=2)  # (B, N, 2, D)
        interleaved = interleaved.reshape(B, 2 * N, self.embed_dim)  # (B, 2N, D)

        # ── Build type ids for type embedding ────────────────────────
        # 0 = CLS/point, 1 = attribute, 2 = geodesic
        prefix_len = 1 + (1 if self.use_point_token else 0)
        type_ids_neighbor = torch.tensor([1, 2], device=device).repeat(N)  # (2N,)
        type_ids_prefix = torch.zeros(prefix_len, dtype=torch.long, device=device)
        type_ids = torch.cat([type_ids_prefix, type_ids_neighbor])  # (seq_len,)

        # ── Build full token sequence ────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_point_token:
            point_attrs = point_features[:, self.point_attr_indices]
            point_token = self.point_encoder(point_attrs).unsqueeze(1)
            tokens = torch.cat([cls_tokens, point_token, interleaved], dim=1)
        else:
            point_token = None
            tokens = torch.cat([cls_tokens, interleaved], dim=1)

        seq_len = tokens.shape[1]

        # ── Add type embedding ───────────────────────────────────────
        tokens = tokens + self.type_embed(type_ids).unsqueeze(0)  # broadcast (1, S, D)

        # ── Add positional encoding ──────────────────────────────────
        attn_bias = None
        if self.pos_encoding_type == "index":
            tokens = tokens + self.pos_embed(seq_len)
        elif self.pos_encoding_type == "spatial":
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            neighbor_xyz = sorted_neighbors[:, :, xyz_start:xyz_end]
            point_xyz = point_features[:, xyz_start:xyz_end]
            rel_xyz = neighbor_xyz - point_xyz.unsqueeze(1)         # (B, N, 3)
            spatial_pe = self.spatial_pos_embed(rel_xyz)            # (B, N, D)
            # Duplicate PE for each (attr, geo) pair
            spatial_pe_pair = spatial_pe.unsqueeze(2).expand(-1, -1, 2, -1)  # (B,N,2,D)
            spatial_pe_flat = spatial_pe_pair.reshape(B, 2 * N, self.embed_dim)
            n_prefix = prefix_len
            prefix_pe = torch.zeros(B, n_prefix, self.embed_dim, device=device)
            full_pe = torch.cat([prefix_pe, spatial_pe_flat], dim=1)
            tokens = tokens + full_pe
        else:  # relative_bias
            xyz_start, xyz_end = self.xyz_indices["xyz"]
            neighbor_xyz = sorted_neighbors[:, :, xyz_start:xyz_end]
            point_xyz = point_features[:, xyz_start:xyz_end]
            rel_neighbor = neighbor_xyz - point_xyz.unsqueeze(1)
            # Each (attr_i, geo_i) pair shares the same 3D position
            rel_pair = rel_neighbor.unsqueeze(2).expand(-1, -1, 2, -1).reshape(B, 2 * N, 3)
            origin = torch.zeros(B, 1, 3, device=device)
            if self.use_point_token:
                positions = torch.cat([origin, origin, rel_pair], dim=1)
            else:
                positions = torch.cat([origin, rel_pair], dim=1)
            attn_bias = self.rel_pos_bias(positions)

        # ── Attention mask ───────────────────────────────────────────
        attn_mask = self._create_split_attention_mask(sorted_valid_mask, self.use_point_token)
        attn_mask_2d = attn_mask.unsqueeze(1).unsqueeze(2).float()

        # ── Transformer encoder ──────────────────────────────────────
        for block in self.encoder_blocks:
            tokens = block(tokens, attn_mask_2d, attn_bias)

        tokens = self.encoder_norm(tokens)

        # ── Extract outputs & pool ───────────────────────────────────
        cls_output = tokens[:, 0, :]
        pool_start = prefix_len
        pool_tokens = tokens[:, pool_start:, :]            # (B, 2N, D)

        # Build pool mask: interleaved [attr_valid, geo_valid, …]
        attr_pool_valid = torch.ones(B, N, dtype=torch.bool, device=device)
        pool_mask = torch.stack([attr_pool_valid, sorted_valid_mask], dim=2).reshape(B, 2 * N)

        any_valid = pool_mask.any(dim=1)

        if self.pool in ("max", "max_mean"):
            pt_max = pool_tokens.clone()
            pt_max[~pool_mask] = float("-inf")
            max_pooled = pt_max.max(dim=1).values
            if (~any_valid).any():
                max_pooled[~any_valid] = cls_output[~any_valid]

        if self.pool in ("mean", "max_mean"):
            pt_mean = pool_tokens.clone()
            pt_mean[~pool_mask] = 0.0
            counts = pool_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pooled = pt_mean.sum(dim=1) / counts
            if (~any_valid).any():
                mean_pooled[~any_valid] = cls_output[~any_valid]

        if self.pool == "max":
            combined = torch.cat([cls_output, max_pooled], dim=-1)
        elif self.pool == "mean":
            combined = torch.cat([cls_output, mean_pooled], dim=-1)
        else:
            combined = torch.cat([cls_output, max_pooled, mean_pooled], dim=-1)

        prediction = self.prediction_head(combined)

        if return_embeddings:
            return {
                "prediction": prediction,
                "attr_tokens": attr_tokens,
                "geo_tokens": geo_tokens,
                "point_token": point_token.squeeze(1) if self.use_point_token else None,
                "cls_output": cls_output,
                "encoder_output": tokens,
                "sorted_indices": sorted_indices,
                "attention_mask": attn_mask,
            }
        return prediction

    # ------------------------------------------------------------------
    def get_loss(self, predictions, targets, loss_type="mse", huber_delta=1.0, blended_mse_lambda=0.5):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if loss_type == "mse":
            return nn.functional.mse_loss(predictions, targets)
        elif loss_type == "l1":
            return nn.functional.l1_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return nn.functional.smooth_l1_loss(predictions, targets)
        elif loss_type == "huber":
            return nn.functional.huber_loss(predictions, targets, delta=huber_delta)
        elif loss_type == "log_mse":
            return nn.functional.mse_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "log_l1":
            return nn.functional.l1_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "blended_mse":
            mse_linear = nn.functional.mse_loss(predictions, targets)
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt = torch.log(targets.clamp(min=1e-8))
            mse_log = nn.functional.mse_loss(log_pred, log_tgt)
            return blended_mse_lambda * mse_linear + (1.0 - blended_mse_lambda) * mse_log
        raise ValueError(f"Unknown loss type: {loss_type}")

    def get_metrics(self, predictions, targets):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()
            rmse = mse ** 0.5
            rel = torch.mean(torch.abs(predictions - targets) / (targets + 1e-8)).item() * 100
            mx = torch.max(torch.abs(predictions - targets)).item()
        return {"mae": mae, "mse": mse, "rmse": rmse, "relative_error_pct": rel, "max_error": mx}

    def print_model_info(self):
        print("=" * 80)
        print("GaussianPatchSplitTransformer Model Information")
        print("=" * 80)
        print(f"Attributes: {self.attributes}")
        print(f"Point attributes: {self.point_attributes}")
        print(f"Use point token: {self.use_point_token}")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Encoder depth: {self.encoder_depth}")
        print(f"Attention heads: {self.num_heads}")
        print(f"Pool: {self.pool}")
        print(f"Encoder type: {self.encoder_type}")
        print(f"Pos encoding: {self.pos_encoding_type}")
        seq_len = 1 + (1 if self.use_point_token else 0) + 2 * self.max_neighbors
        print(f"Sequence length: {seq_len} (CLS + {'point + ' if self.use_point_token else ''}{self.max_neighbors} attr + {self.max_neighbors} geo)")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 80)

class SplitEmbeddingEncoderDecoderTransformer(nn.Module):
    """Encoder-Decoder Transformer with split attribute/geodesic embeddings.

    Like :class:`GeodesicEncoderDecoderTransformer` but:

    * **Encoder** processes all Gaussian attributes (not just XYZ) using
      **standard** self-attention (not geodesic self-attention).
    * **Decoder** processes geodesic distances using standard self-attention
      + cross-attention to the encoder.
    * Both encoder and decoder use separate embedding layers.
    * Valid mask applied only on the decoder side (geodesic data may be padded).
    """

    def __init__(
        self,
        attributes: List[str] = ["xyz"],
        point_attributes: Optional[List[str]] = None,
        max_neighbors: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        decoder_depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "max",
        encoder_type: str = "conv",
    ):
        super().__init__()

        if pool not in ("max", "mean", "max_mean"):
            raise ValueError(f"pool must be 'max', 'mean', or 'max_mean', got '{pool}'")

        self.attention_type = "split_enc_dec"
        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.pool = pool
        self.encoder_type = encoder_type

        # Dimensions
        self.neighbor_feature_dim = get_attribute_dim(attributes) + 1
        self.point_feature_dim = get_attribute_dim(attributes)

        # XYZ indices (for sorting)
        current_idx = 0
        self.xyz_start = None
        for attr in attributes:
            size = ATTRIBUTE_DIMS.get(attr, 0)
            if attr == "xyz":
                self.xyz_start = current_idx
            current_idx += size
        if self.xyz_start is not None:
            self.xyz_end = self.xyz_start + 3
        else:
            self.xyz_start = 0
            self.xyz_end = 0

        # ── Attribute encoder (for encoder side — no geodesic) ───────
        NeighborEncoderCls = {
            "conv":     GaussianPatchEncoder,
            "linear":   GaussianPatchLinearEncoder,
            "residual": GaussianPatchLinearEncoderResidual,
        }[encoder_type]
        self.attribute_encoder = NeighborEncoderCls(
            attributes=attributes,
            embed_dim=embed_dim,
            include_geodesic=False,
        )

        # ── Geodesic encoder (for decoder side) ─────────────────────
        self.geodesic_encoder = GeodesicEmbedding(embed_dim=embed_dim)

        # ── CLS token (prepended to decoder) ────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Encoder blocks (standard self-attention on attributes) ──
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr_enc[i],
            )
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ── Decoder blocks (self-attn + cross-attn) ─────────────────
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr_dec[i],
            )
            for i in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # ── Prediction head ──────────────────────────────────────────
        head_input_dim = embed_dim * 3 if pool == "max_mean" else embed_dim * 2
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _sort_by_euclidean(self, neighborhood, point_features, valid_mask):
        if self.xyz_start is not None and self.xyz_end > self.xyz_start:
            point_xyz = point_features[:, self.xyz_start:self.xyz_end]
            neigh_xyz = neighborhood[:, :, self.xyz_start:self.xyz_end]
            dists = (neigh_xyz - point_xyz.unsqueeze(1)).norm(dim=-1)
            dists = torch.where(valid_mask, dists, torch.full_like(dists, 1e10))
            idx = dists.argsort(dim=1)
            expanded = idx.unsqueeze(-1).expand(-1, -1, neighborhood.shape[-1])
            return torch.gather(neighborhood, 1, expanded), torch.gather(valid_mask, 1, idx), idx
        B = neighborhood.shape[0]
        return neighborhood, valid_mask, torch.arange(self.max_neighbors, device=neighborhood.device).unsqueeze(0).expand(B, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        B, N, _ = neighborhood.shape
        device = neighborhood.device

        # Sort neighbours by Euclidean distance
        sorted_nb, sorted_mask, sorted_idx = self._sort_by_euclidean(
            neighborhood, point_features, valid_mask,
        )

        # ── Encoder: all attributes (no geodesic), no mask ───────────
        attr_features = sorted_nb[:, :, :-1]                           # (B, N, attr_dim)
        enc_tokens = self.attribute_encoder(attr_features)             # (B, N, D)

        for block in self.encoder_blocks:
            enc_tokens = block(enc_tokens, mask=None, attn_bias=None)
        enc_tokens = self.encoder_norm(enc_tokens)

        # ── Decoder: geodesic distances → cross-attend to encoder ────
        geo_dists = sorted_nb[:, :, -1:]                               # (B, N, 1)
        dec_tokens = self.geodesic_encoder(geo_dists)                  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                        # (B, 1, D)
        dec_tokens = torch.cat([cls, dec_tokens], dim=1)               # (B, 1+N, D)

        # Decoder self-attention mask (CLS always valid)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        dec_mask = torch.cat([cls_valid, sorted_mask], dim=1)
        dec_mask_4d = dec_mask.unsqueeze(1).unsqueeze(2).float()

        for block in self.decoder_blocks:
            dec_tokens = block(dec_tokens, enc_tokens,
                               self_attn_mask=dec_mask_4d, cross_attn_mask=None)
        dec_tokens = self.decoder_norm(dec_tokens)

        # ── Pool & predict ───────────────────────────────────────────
        cls_out = dec_tokens[:, 0, :]
        pool_tokens = dec_tokens[:, 1:, :]
        any_valid = sorted_mask.any(dim=1)

        if self.pool in ("max", "max_mean"):
            pt_max = pool_tokens.clone()
            pt_max[~sorted_mask] = float("-inf")
            max_pooled = pt_max.max(dim=1).values
            if (~any_valid).any():
                max_pooled[~any_valid] = cls_out[~any_valid]

        if self.pool in ("mean", "max_mean"):
            pt_mean = pool_tokens.clone()
            pt_mean[~sorted_mask] = 0.0
            counts = sorted_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pooled = pt_mean.sum(dim=1) / counts
            if (~any_valid).any():
                mean_pooled[~any_valid] = cls_out[~any_valid]

        if self.pool == "max":
            combined = torch.cat([cls_out, max_pooled], dim=-1)
        elif self.pool == "mean":
            combined = torch.cat([cls_out, mean_pooled], dim=-1)
        else:
            combined = torch.cat([cls_out, max_pooled, mean_pooled], dim=-1)

        prediction = self.prediction_head(combined)

        if return_embeddings:
            return {
                "prediction": prediction,
                "encoder_output": enc_tokens,
                "decoder_output": dec_tokens,
                "cls_output": cls_out,
                "sorted_indices": sorted_idx,
            }
        return prediction

    # ------------------------------------------------------------------
    def get_loss(self, predictions, targets, loss_type="mse", huber_delta=1.0, blended_mse_lambda=0.5):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if loss_type == "mse":
            return nn.functional.mse_loss(predictions, targets)
        elif loss_type == "l1":
            return nn.functional.l1_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return nn.functional.smooth_l1_loss(predictions, targets)
        elif loss_type == "huber":
            return nn.functional.huber_loss(predictions, targets, delta=huber_delta)
        elif loss_type == "log_mse":
            return nn.functional.mse_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "log_l1":
            return nn.functional.l1_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "blended_mse":
            mse_linear = nn.functional.mse_loss(predictions, targets)
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt = torch.log(targets.clamp(min=1e-8))
            mse_log = nn.functional.mse_loss(log_pred, log_tgt)
            return blended_mse_lambda * mse_linear + (1.0 - blended_mse_lambda) * mse_log
        raise ValueError(f"Unknown loss type: {loss_type}")

    def get_metrics(self, predictions, targets):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()
            rmse = mse ** 0.5
            rel = torch.mean(torch.abs(predictions - targets) / (targets + 1e-8)).item() * 100
            mx = torch.max(torch.abs(predictions - targets)).item()
        return {"mae": mae, "mse": mse, "rmse": rmse, "relative_error_pct": rel, "max_error": mx}

    def print_model_info(self):
        print("=" * 80)
        print("SplitEmbeddingEncoderDecoderTransformer Model Information")
        print("=" * 80)
        print(f"Attributes: {self.attributes}")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Encoder depth: {self.encoder_depth}")
        print(f"Decoder depth: {self.decoder_depth}")
        print(f"Attention heads: {self.num_heads}")
        print(f"Pool: {self.pool}")
        print(f"Encoder type: {self.encoder_type}")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 80)


class GeodesicEncoderDecoderTransformer(nn.Module):
    """Encoder-Decoder Transformer for geodesic distance prediction.

    - **Encoder** processes raw XYZ positions using geodesic self-attention
      (no masking — all spatial tokens always valid).
    - **Decoder** processes raw geodesic distances via standard self-attention
      + cross-attention to the encoder output.  The valid mask is applied only
      here (geodesic data may be padded).

    For now, only XYZ and geodesic distance are used from the neighbour
    features; other Gaussian attributes and point features are ignored.
    """

    def __init__(
        self,
        attributes: List[str] = ["xyz"],
        max_neighbors: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        decoder_depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pool: str = "max",
    ):
        super().__init__()

        if "xyz" not in attributes:
            raise ValueError("GeodesicEncoderDecoderTransformer requires 'xyz' in attributes")
        if pool not in ("max", "mean", "max_mean"):
            raise ValueError(f"pool must be 'max', 'mean', or 'max_mean', got '{pool}'")

        self.attention_type = "geodesic_enc_dec"
        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.pool = pool

        # ── Locate XYZ columns in the neighbour feature vector ────────
        current_idx = 0
        self.xyz_start = None
        for attr in attributes:
            size = ATTRIBUTE_DIMS.get(attr, 0)
            if attr == "xyz":
                self.xyz_start = current_idx
            current_idx += size
        self.xyz_end = self.xyz_start + 3

        # ── Token encoders ────────────────────────────────────────────
        self.xyz_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geo_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── Learnable CLS token (prepended to decoder) ───────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Encoder blocks (geodesic self-attention on XYZ) ──────────
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        self.encoder_blocks = nn.ModuleList([
            GeodesicTransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr_enc[i],
            )
            for i in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ── Decoder blocks (standard self-attn + cross-attn) ─────────
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr_dec[i],
            )
            for i in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # ── Prediction head ───────────────────────────────────────────
        head_input_dim = embed_dim * 3 if pool == "max_mean" else embed_dim * 2
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # ── Weight init ──────────────────────────────────────────────
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    def _sort_by_euclidean(self, neighborhood, point_features, valid_mask):
        """Sort neighbours by Euclidean distance from the central point."""
        point_xyz = point_features[:, self.xyz_start:self.xyz_end]    # (B, 3)
        neigh_xyz = neighborhood[:, :, self.xyz_start:self.xyz_end]   # (B, N, 3)
        dists = (neigh_xyz - point_xyz.unsqueeze(1)).norm(dim=-1)     # (B, N)
        dists = torch.where(valid_mask, dists, torch.full_like(dists, 1e10))
        idx = dists.argsort(dim=1)                                    # (B, N)
        expanded = idx.unsqueeze(-1).expand(-1, -1, neighborhood.shape[-1])
        return torch.gather(neighborhood, 1, expanded), torch.gather(valid_mask, 1, idx), idx

    # ------------------------------------------------------------------
    def forward(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        B, N, _ = neighborhood.shape
        device = neighborhood.device

        # Sort neighbours by Euclidean distance
        sorted_nb, sorted_mask, sorted_idx = self._sort_by_euclidean(
            neighborhood, point_features, valid_mask,
        )

        # Extract XYZ (relative to central point) and geodesic distances
        point_xyz = point_features[:, self.xyz_start:self.xyz_end]        # (B, 3)
        rel_xyz = sorted_nb[:, :, self.xyz_start:self.xyz_end] - point_xyz.unsqueeze(1)  # (B, N, 3)
        geo_dists = sorted_nb[:, :, -1:]                                  # (B, N, 1)

        # ── Encoder: XYZ → geodesic self-attention (no mask) ─────────
        enc_tokens = self.xyz_embed(rel_xyz)                              # (B, N, D)
        gds = compute_geodesic_distance_scores(rel_xyz).detach()          # (B, N, N)

        for block in self.encoder_blocks:
            enc_tokens = block(enc_tokens, mask=None, attn_bias=None, gds=gds)
        enc_tokens = self.encoder_norm(enc_tokens)

        # ── Decoder: geodesic distances → cross-attend to encoder ────
        dec_tokens = self.geo_embed(geo_dists)                            # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                           # (B, 1, D)
        dec_tokens = torch.cat([cls, dec_tokens], dim=1)                  # (B, 1+N, D)

        # Decoder self-attention mask (CLS always valid)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        dec_mask = torch.cat([cls_valid, sorted_mask], dim=1)             # (B, 1+N)
        dec_mask_4d = dec_mask.unsqueeze(1).unsqueeze(2).float()          # (B, 1, 1, 1+N)

        for block in self.decoder_blocks:
            dec_tokens = block(dec_tokens, enc_tokens,
                               self_attn_mask=dec_mask_4d, cross_attn_mask=None)
        dec_tokens = self.decoder_norm(dec_tokens)

        # ── Pool & predict ────────────────────────────────────────────
        cls_out = dec_tokens[:, 0, :]                                     # (B, D)
        pool_tokens = dec_tokens[:, 1:, :]                                # (B, N, D)
        any_valid = sorted_mask.any(dim=1)                                # (B,)

        if self.pool in ("max", "max_mean"):
            pt_max = pool_tokens.clone()
            pt_max[~sorted_mask] = float("-inf")
            max_pooled = pt_max.max(dim=1).values
            if (~any_valid).any():
                max_pooled[~any_valid] = cls_out[~any_valid]

        if self.pool in ("mean", "max_mean"):
            pt_mean = pool_tokens.clone()
            pt_mean[~sorted_mask] = 0.0
            counts = sorted_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pooled = pt_mean.sum(dim=1) / counts
            if (~any_valid).any():
                mean_pooled[~any_valid] = cls_out[~any_valid]

        if self.pool == "max":
            combined = torch.cat([cls_out, max_pooled], dim=-1)
        elif self.pool == "mean":
            combined = torch.cat([cls_out, mean_pooled], dim=-1)
        else:
            combined = torch.cat([cls_out, max_pooled, mean_pooled], dim=-1)

        prediction = self.prediction_head(combined)

        if return_embeddings:
            return {
                "prediction": prediction,
                "encoder_output": enc_tokens,
                "decoder_output": dec_tokens,
                "cls_output": cls_out,
                "sorted_indices": sorted_idx,
            }
        return prediction

    # ------------------------------------------------------------------
    def get_loss(self, predictions, targets, loss_type="mse", huber_delta=1.0, blended_mse_lambda=0.5):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        if loss_type == "mse":
            return nn.functional.mse_loss(predictions, targets)
        elif loss_type == "l1":
            return nn.functional.l1_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return nn.functional.smooth_l1_loss(predictions, targets)
        elif loss_type == "huber":
            return nn.functional.huber_loss(predictions, targets, delta=huber_delta)
        elif loss_type == "log_mse":
            return nn.functional.mse_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "log_l1":
            return nn.functional.l1_loss(
                torch.log(predictions.clamp(min=1e-8)),
                torch.log(targets.clamp(min=1e-8)),
            )
        elif loss_type == "blended_mse":
            mse_linear = nn.functional.mse_loss(predictions, targets)
            log_pred = torch.log(predictions.clamp(min=1e-8))
            log_tgt = torch.log(targets.clamp(min=1e-8))
            mse_log = nn.functional.mse_loss(log_pred, log_tgt)
            return blended_mse_lambda * mse_linear + (1.0 - blended_mse_lambda) * mse_log
        raise ValueError(f"Unknown loss type: {loss_type}")

    # ------------------------------------------------------------------
    def get_metrics(self, predictions, targets):
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()
            rmse = mse ** 0.5
            rel = torch.mean(torch.abs(predictions - targets) / (targets + 1e-8)).item() * 100
            mx = torch.max(torch.abs(predictions - targets)).item()
        return {"mae": mae, "mse": mse, "rmse": rmse, "relative_error_pct": rel, "max_error": mx}

    # ------------------------------------------------------------------
    def print_model_info(self):
        print("=" * 80)
        print("GeodesicEncoderDecoderTransformer Model Information")
        print("=" * 80)
        print(f"Attributes: {self.attributes}")
        print(f"Max neighbors: {self.max_neighbors}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Encoder depth: {self.encoder_depth}")
        print(f"Decoder depth: {self.decoder_depth}")
        print(f"Attention heads: {self.num_heads}")
        print(f"Pool: {self.pool}")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 80)


def create_gaussian_patch_transformer(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Factory function to create a GaussianPatchTransformer,
    GeodesicEncoderDecoderTransformer, GaussianPatchSplitTransformer,
    or SplitEmbeddingEncoderDecoderTransformer model.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments override config values
        
    Returns:
        Initialized model (type depends on ``attention_type``)
    """
    # Default configuration
    default_config = {
        'attributes': ["xyz", "opacity", "scale", "rotation", "sh"],
        'point_attributes': None,  # Use all attributes by default
        'max_neighbors': 32,
        'embed_dim': 384,
        'encoder_depth': 6,
        'decoder_depth': 3,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path_rate': 0.1,
        'pool': 'max',
        'pos_encoding_type': 'index',
        'encoder_type': 'conv',   # conv | linear | residual
        'attention_type': 'standard',  # standard | geodesic | geodesic_enc_dec | split | split_enc_dec
        'use_cls_token': True,
    }
    
    # Update with provided config
    if config is not None:
        default_config.update(config)
    
    # Override with kwargs
    default_config.update(kwargs)
    
    attention_type = default_config.get('attention_type', 'standard')

    # Route to the correct model class
    if attention_type == 'geodesic_enc_dec':
        enc_dec_keys = {
            'attributes', 'max_neighbors', 'embed_dim', 'encoder_depth',
            'decoder_depth', 'num_heads', 'mlp_ratio', 'qkv_bias',
            'dropout', 'attn_dropout', 'drop_path_rate', 'pool',
        }
        enc_dec_config = {k: v for k, v in default_config.items() if k in enc_dec_keys}
        model = GeodesicEncoderDecoderTransformer(**enc_dec_config)
    elif attention_type == 'split_enc_dec':
        split_enc_dec_keys = {
            'attributes', 'point_attributes', 'max_neighbors', 'embed_dim',
            'encoder_depth', 'decoder_depth', 'num_heads', 'mlp_ratio',
            'qkv_bias', 'dropout', 'attn_dropout', 'drop_path_rate',
            'pool', 'encoder_type',
        }
        split_config = {k: v for k, v in default_config.items() if k in split_enc_dec_keys}
        model = SplitEmbeddingEncoderDecoderTransformer(**split_config)
    elif attention_type == 'split':
        split_keys = {k for k in default_config if k not in ('decoder_depth', 'use_cls_token')}
        split_config = {k: v for k, v in default_config.items() if k in split_keys}
        model = GaussianPatchSplitTransformer(**split_config)
    else:
        # Standard or geodesic encoder-only model
        enc_only_config = {k: v for k, v in default_config.items() if k != 'decoder_depth'}
        model = GaussianPatchTransformer(**enc_only_config)
    
    return model


if __name__ == "__main__":
    """Test the model with dummy data."""
    
    # Configuration
    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]
    max_neighbors = 32
    batch_size = 8
    
    from utils import get_attribute_dim
    
    neighbor_feature_dim = get_attribute_dim(attributes) + 1  # +1 for geodesic distance
    point_feature_dim = get_attribute_dim(attributes)
    
    print(f"Neighbor feature dim: {neighbor_feature_dim}")
    print(f"Point feature dim: {point_feature_dim}")
    print()
    
    # Create model with all point attributes
    print("=" * 80)
    print("Test 1: Model with all point attributes")
    print("=" * 80)
    model = create_gaussian_patch_transformer(
        attributes=attributes,
        point_attributes=None,  # Use all
        max_neighbors=max_neighbors,
        embed_dim=256,
        encoder_depth=4,
        num_heads=8
    )
    
    model.print_model_info()
    
    # Create dummy data
    neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
    point_features = torch.randn(batch_size, point_feature_dim)
    valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
    # Make some neighbors invalid
    valid_mask[:, -5:] = False
    
    print("\nTesting forward pass...")
    print(f"Neighborhood shape: {neighborhood.shape}")
    print(f"Point features shape: {point_features.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
    
    # Forward pass
    predictions = model(neighborhood, point_features, valid_mask)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3, 0].detach().numpy()}")
    
    # Test with return_embeddings=True
    print("\nTesting with return_embeddings=True...")
    output = model(neighborhood, point_features, valid_mask, return_embeddings=True)
    print("Output keys:", output.keys())
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Create model without point token
    print("\n" + "=" * 80)
    print("Test 2: Model without point token (empty point_attributes)")
    print("=" * 80)
    model_no_point = create_gaussian_patch_transformer(
        attributes=attributes,
        point_attributes=[],  # No point token
        max_neighbors=max_neighbors,
        embed_dim=256,
        encoder_depth=4,
        num_heads=8
    )
    
    model_no_point.print_model_info()
    
    predictions_no_point = model_no_point(neighborhood, point_features, valid_mask)
    print(f"Prediction shape: {predictions_no_point.shape}")
    
    # Create model with subset of point attributes
    print("\n" + "=" * 80)
    print("Test 3: Model with subset of point attributes")
    print("=" * 80)
    model_partial = create_gaussian_patch_transformer(
        attributes=attributes,
        point_attributes=["xyz", "opacity"],
        max_neighbors=max_neighbors,
        embed_dim=256,
        encoder_depth=4,
        num_heads=8
    )
    
    model_partial.print_model_info()
    
    predictions_partial = model_partial(neighborhood, point_features, valid_mask)
    print(f"Prediction shape: {predictions_partial.shape}")
    
    print("\n✓ All tests passed!")
