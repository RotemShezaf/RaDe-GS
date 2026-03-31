"""
Utility operations for Gaussian Patch Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

_SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


# Canonical attribute → dimension mapping.  Used by all dimension/index helpers
# to avoid duplicated if/elif chains.
ATTRIBUTE_DIMS = {
    "xyz": 3,
    "normals": 3,
    "scale": 3,
    "sh": 3,
    "rotation": 4,
    "opacity": 1,
    "euclidean_distances": 1,
    "geodesic_distance": 1,
}


def get_attribute_dim(attributes: List[str]) -> int:
    """
    Calculate the total dimension for a list of Gaussian attributes.
    
    Args:
        attributes: List of attribute names (e.g., ["xyz", "opacity", "scale"])
        
    Returns:
        Total dimension for all attributes
    """
    dim = 0
    for attr in attributes:
        dim += ATTRIBUTE_DIMS.get(attr, 0)
    return dim

def get_attributes_indices(attributes: List[str], all_attributes: List[str], include_geodesic: bool = True) -> List[int]:
    """
    Get the indices for attributes in the standard Gaussian feature order.
    
    Standard order: xyz(3), opacity(1), scale(3), rotation(4), sh(3), [geodesic_distance(1)]
    
    Args:
        attributes: List of attribute names to select
        all_attributes: List of all attributes present in the dataset
        include_geodesic: Whether to include geodesic distance index
        
    Returns:
        List of indices for the specified attributes
    """
    indices = []
    current_idx = 0
    
    # Standard attribute order
    standard_attrs = [
        "xyz", "opacity", "scale", "rotation", "sh",
        "euclidean_distances", "normals",
    ]
    if include_geodesic:
        standard_attrs.append("geodesic_distance")
    
    for attr_name in standard_attrs:
        attr_size = ATTRIBUTE_DIMS[attr_name]
        if attr_name in attributes and attr_name in all_attributes:
            indices.extend(range(current_idx, current_idx + attr_size))
        if attr_name in all_attributes:
            current_idx += attr_size
    
    return indices


# Backward-compatible alias (original had a typo)
get_atribues_indices = get_attributes_indices


def get_attribute_indices(attributes: List[str], include_geodesic: bool = True) -> List[int]:
    """
    Get the indices for attributes in the standard Gaussian feature order.
    
    Standard order: xyz(3), opacity(1), scale(3), rotation(4), sh(3), [geodesic_distance(1)]
    
    Args:
        attributes: List of attribute names
        include_geodesic: Whether to include geodesic distance index
        
    Returns:
        List of indices for the specified attributes
    """
    indices = []
    current_idx = 0
    
    # Standard attribute order
    standard_attrs = [
        "xyz", "opacity", "scale", "rotation", "sh",
        "euclidean_distances", "normals",
    ]
    if include_geodesic:
        standard_attrs.append("geodesic_distance")
    
    for attr_name in standard_attrs:
        attr_size = ATTRIBUTE_DIMS[attr_name]
        if attr_name in attributes:
            indices.extend(range(current_idx, current_idx + attr_size))
        current_idx += attr_size
    
    return indices


def normalize_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """
    Normalize rotation quaternions to unit length.
    
    Args:
        rotation: Tensor of shape (..., 4) containing quaternions
        
    Returns:
        Normalized quaternions of same shape
    """
    return rotation / (torch.norm(rotation, p=2, dim=-1, keepdim=True) + 1e-9)


def extract_neighbor_features(
    features: torch.Tensor,
    max_neighbors: int,
    entry_size: int,
    attributes: List[str]
) -> torch.Tensor:
    """
    Extract neighbor features from flattened patch representation.
    
    Args:
        features: Flattened features of shape (batch_size, total_features)
        max_neighbors: Number of neighbors in the patch
        entry_size: Size of each neighbor entry (including geodesic distance)
        attributes: List of attributes to extract
        
    Returns:
        Neighbor features of shape (batch_size, max_neighbors, feature_dim)
    """
    batch_size = features.shape[0]
    neighbor_data = features[:, :max_neighbors * entry_size]
    neighbor_data = neighbor_data.reshape(batch_size, max_neighbors, entry_size)
    return neighbor_data


def extract_point_features(
    features: torch.Tensor,
    max_neighbors: int,
    neighbor_entry_size: int,
    point_feature_dim: int
) -> torch.Tensor:
    """
    Extract the central point features from flattened patch representation.
    
    Args:
        features: Flattened features of shape (batch_size, total_features)
        max_neighbors: Number of neighbors in the patch
        neighbor_entry_size: Size of each neighbor entry
        point_feature_dim: Dimension of point features (without geodesic distance)
        
    Returns:
        Point features of shape (batch_size, point_feature_dim)
    """
    start_idx = max_neighbors * neighbor_entry_size
    point_data = features[:, start_idx:start_idx + point_feature_dim]
    return point_data


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer architectures.
    Used for encoding the order of neighbors sorted by distance from the source point.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Args:
            embed_dim: Dimension of the embeddings
            max_len: Maximum sequence length
        """
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
        return self.pe[:, :seq_len, :]


def sinusoidal(positions: torch.Tensor, features: int = 16, periods: int = 10000) -> torch.Tensor:
    """Encode ``positions`` using sinusoidal positional encoding.

    Based on the point_pe encoding scheme used for bounded 3-D spaces.

    Args:
        positions: Tensor of arbitrary shape ``(*)``.
        features: Half the number of features per position dimension.
        periods: Base for the logarithmically-spaced frequency grid.

    Returns:
        Positional encoding of shape ``(*positions.shape, features, 2)``
        where the last two dims are (sin, cos).
    """
    dtype = positions.dtype if positions.is_floating_point() else None
    kwargs = dict(device=positions.device, dtype=dtype)
    omega = torch.logspace(0, 1 / features - 1, features, periods, **kwargs)
    fraction = omega * positions.unsqueeze(-1)
    return torch.stack((fraction.sin(), fraction.cos()), dim=-1)


class SpatialPositionalEncoding(nn.Module):
    """3-D spatial positional encoding via sinusoidal encoding of relative XYZ.

    Each neighbour's position is expressed *relative to the central (query) point*
    and encoded with multi-frequency sinusoidal functions, then projected to the
    model embedding dimension.  This provides geometry-aware positional signals
    instead of sequence-order-based encoding.

    Raw sinusoidal output dim = ``3 * pe_features * 2``; a learned linear layer
    then projects this to ``embed_dim``.
    """

    def __init__(self, embed_dim: int, pe_features: int = 16, pe_periods: int = 10000):
        """
        Args:
            embed_dim: Output embedding dimension.
            pe_features: Half the number of sinusoidal features per coordinate
                         (default 16 → 96-dim raw encoding for 3-D input).
            pe_periods: Base period for logspace frequency spacing (default 10000).
        """
        super().__init__()
        self.pe_features = pe_features
        self.pe_periods = pe_periods
        raw_dim = 3 * pe_features * 2  # 3 coords × features × 2 (sin + cos)
        self.proj = nn.Linear(raw_dim, embed_dim)

    def forward(self, rel_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_xyz: Relative 3-D positions of shape ``(B, N, 3)``.

        Returns:
            Positional encodings of shape ``(B, N, embed_dim)``.
        """
        # sinusoidal(rel_xyz) -> (B, N, 3, pe_features, 2)
        pe_raw = sinusoidal(rel_xyz, self.pe_features, self.pe_periods)
        pe_raw = pe_raw.flatten(-3)  # (B, N, 3 * pe_features * 2)
        return self.proj(pe_raw)  # (B, N, embed_dim)


class RelativePositionBias(nn.Module):
    """Learned per-head attention bias from pairwise relative 3-D positions.

    For every pair of tokens (i, j) the module computes a scalar bias per
    attention head that is *added to the raw attention logits* before softmax.
    This keeps geometric relationships alive at every transformer layer,
    unlike additive positional encodings which are added once before layer 1.

    Pipeline:
        rel_xyz  (B, S, S, 3)
        → sinusoidal encoding  (B, S, S, 3 * pe_features * 2)
        → MLP → (B, S, S, num_heads)
        → permute → (B, num_heads, S, S)   [ready to add to attn logits]
    """

    def __init__(self, num_heads: int, pe_features: int = 8, pe_periods: int = 10000):
        """
        Args:
            num_heads: Number of attention heads (one bias scalar per head).
            pe_features: Half the sinusoidal features per coordinate
                         (default 8 → raw dim = 3 * 8 * 2 = 48).
            pe_periods: Base period for logspace frequency spacing.
        """
        super().__init__()
        self.pe_features = pe_features
        self.pe_periods = pe_periods
        raw_dim = 3 * pe_features * 2
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Absolute 3-D positions of each token, shape ``(B, S, 3)``.
                       CLS / point tokens should be assigned the central-point
                       position (i.e. relative displacement = 0 w.r.t. themselves).

        Returns:
            Attention bias of shape ``(B, num_heads, S, S)``.
        """
        # Pairwise relative positions: (B, S, S, 3)
        rel = positions.unsqueeze(2) - positions.unsqueeze(1)
        # Sinusoidal encoding: (B, S, S, 3, pe_features, 2) → flatten → (B, S, S, raw_dim)
        pe = sinusoidal(rel, self.pe_features, self.pe_periods).flatten(-3)
        bias = self.mlp(pe)  # (B, S, S, num_heads)
        return bias.permute(0, 3, 1, 2)  # (B, num_heads, S, S)


def compute_geodesic_distance_scores(
    xyz: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Graph-based Geodesic Distance Scores (Algorithm 1 from the paper).

    Builds a K-NN graph from XYZ positions, sets edge weights to Euclidean
    distance for neighbours (∞ otherwise), then computes all-pairs shortest
    paths via vectorized tropical (min-plus) semiring repeated squaring.

    Vectorized implementation inspired by GeoFormer (VinAI Research):
    instead of O(S) Floyd-Warshall iterations, repeated tropical matrix
    squaring converges in O(log S) iterations, each fully batched.

    Args:
        xyz: Token positions, shape ``(B, S, 3)``.
        mask: Optional boolean validity mask, shape ``(B, S)``.
              ``True`` = valid token, ``False`` = masked / padding.

    Returns:
        GDS matrix of shape ``(B, S, S)``.  Entry ``[b, i, j]`` is the
        shortest-path distance between tokens *i* and *j* in batch element
        *b*.  Disconnected / masked pairs have value ``INF`` (1e6).
    """
    B, S, _ = xyz.shape
    device = xyz.device
    INF = 1e6

    if S <= 1:
        return torch.zeros(B, S, S, device=device, dtype=xyz.dtype)

    K = max(1, int(S ** 0.5))  # Step 1: K = floor(sqrt(N))

    # (1) Pairwise Euclidean distances ──────────────────────────────────
    euc = (xyz.unsqueeze(2) - xyz.unsqueeze(1)).norm(dim=-1)  # (B, S, S)

    if mask is not None:
        inv = ~mask  # (B, S)
        euc.masked_fill_(inv.unsqueeze(1), INF)
        euc.masked_fill_(inv.unsqueeze(2), INF)

    # (2) K-NN graph (exclude self-loops) ──────────────────────────────
    self_mask = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
    k = min(K, S - 1)
    knn_dist, knn_idx = euc.masked_fill(self_mask, INF).topk(
        k, dim=-1, largest=False
    )  # (B, S, k)

    # (3) Build adjacency from K-NN edges ──────────────────────────────
    adj = torch.full((B, S, S), INF, device=device, dtype=xyz.dtype)
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(knn_idx)
    r_idx = torch.arange(S, device=device).view(1, S, 1).expand_as(knn_idx)
    adj[b_idx, r_idx, knn_idx] = knn_dist

    # Symmetric (undirected) graph
    adj = torch.minimum(adj, adj.transpose(1, 2))

    # Self-distance = 0
    diag = torch.arange(S, device=device)
    adj[:, diag, diag] = 0.0

    if mask is not None:
        adj.masked_fill_(inv.unsqueeze(1), INF)
        adj.masked_fill_(inv.unsqueeze(2), INF)
        adj[:, diag, diag] = 0.0

    # (4) All-pairs shortest paths: tropical semiring repeated squaring
    # D^1 = adj (paths using ≤ 1 edge).  After ceil(log2(S)) squarings,
    # D^(2^k) ≥ S gives exact shortest paths (at most S-1 edges).
    dist = adj
    n_iter = max(1, math.ceil(math.log2(S)))
    for _ in range(n_iter):
        # Tropical matmul: C[i,j] = min_k (dist[i,k] + dist[k,j])
        new_dist = (dist.unsqueeze(3) + dist.unsqueeze(1)).min(dim=2).values
        new_dist = torch.minimum(dist, new_dist)
        if torch.equal(new_dist, dist):
            break
        dist = new_dist

    return dist  # (B, S, S)


class GeodesicSelfAttention(nn.Module):
    """Geodesic Self-Attention mechanism.

    Replaces the standard Q·K^T attention with attention weights derived from
    a precomputed *Geodesic Distance Score* (GDS) matrix.

    Formula (per head *h*)::

        head_h = softmax(-GDS / τ_h) · V_h(X)

    where ``τ_h`` is a learnable per-head temperature and ``V_h`` is the
    value projection for head *h*.

    Based on *Geodesic Self-Attention for 3D Point Clouds* (NeurIPS 2022).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Only V projection (Q, K replaced by GDS)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Learnable log-temperature per head; initialised so that
        # exp(log_temp) = sqrt(head_dim), matching the standard attention
        # scaling factor.
        self.log_temperature = nn.Parameter(
            torch.full((num_heads,), math.log(self.head_dim ** 0.5))
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        gds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape ``(B, S, D)``.
            gds: Precomputed GDS matrix of shape ``(B, S, S)``.
            mask: Optional attention mask ``(B, 1, 1, S)`` —
                  ``1.0`` = attend, ``0.0`` = mask out.
            attn_bias: Optional additional per-head bias ``(B, H, S, S)``
                       (e.g. from ``relative_bias`` positional encoding).
        """
        B, S, D = x.shape

        # V projection → (B, num_heads, S, head_dim)
        v = (
            self.v_proj(x)
            .reshape(B, S, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Temperature: always positive via exp()
        temperature = self.log_temperature.exp().view(1, self.num_heads, 1, 1)

        # GDS → attention logits: negate so closer tokens get higher weight
        attn_logits = -gds.unsqueeze(1) / temperature  # (B, H, S, S)

        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        Args:
            dim: Input and output dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    When PyTorch >= 2.0 is available and no ``attn_bias`` is supplied, the
    forward pass delegates to ``F.scaled_dot_product_attention`` which
    automatically selects the fastest available kernel (FlashAttention /
    Memory-Efficient / Math).  When ``attn_bias`` is provided (e.g. for
    relative-position bias), it falls back to a manual implementation so
    that the bias can be injected into the logits.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0
    ):
        """
        Args:
            dim: Input dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_dropout: Dropout probability for attention weights
            proj_dropout: Dropout probability for output projection
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    def _manual_attention(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback path: explicit matmul → softmax → matmul."""
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        return attn @ v

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            mask: Optional attention mask of shape (batch_size, 1, 1, seq_len) or broadcastable.
                  1.0 = attend, 0.0 = mask out.
            attn_bias: Optional per-head bias added to attention logits,
                       shape (batch_size, num_heads, seq_len, seq_len).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)
        
        if attn_bias is not None or not _SDPA_AVAILABLE:
            # Manual path: required when attn_bias is present, or PyTorch < 2.0
            x = self._manual_attention(q, k, v, mask, attn_bias)
        else:
            # SDPA path: automatic kernel selection (Flash / Memory-Efficient / Math)
            # Convert float mask (1=attend, 0=ignore) → additive mask (0 / -inf)
            if mask is not None:
                sdpa_mask = torch.where(mask.bool(), 0.0, float('-inf'))
            else:
                sdpa_mask = None
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for encoder-decoder architectures.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0
    ):
        """
        Args:
            dim: Input dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_dropout: Dropout probability for attention weights
            proj_dropout: Dropout probability for output projection
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor of shape (batch_size, seq_len_q, dim)
            context: Key-Value tensor of shape (batch_size, seq_len_kv, dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, dim)
        """
        B, N_q, C = x.shape
        N_kv = context.shape[1]
        
        # Generate Q from x, K and V from context
        q = self.q(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x
