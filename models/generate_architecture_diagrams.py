"""Generate professional architecture diagrams for GaussianPatchTransformer and
SplitEmbeddingEncoderDecoderTransformer as PNG files.

All arrows connect exactly to component edges.  No overlap / text occlusion.
Valid-mask flow is shown explicitly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════
# Styling constants
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    "input":       "#E8F5E9",
    "encoder":     "#BBDEFB",
    "attention":   "#FFE0B2",
    "norm":        "#F3E5F5",
    "pool":        "#FFF9C4",
    "head":        "#FFCDD2",
    "output":      "#D7CCC8",
    "cls":         "#B2EBF2",
    "pos":         "#C8E6C9",
    "cross_attn":  "#FFCCBC",
    "geodesic_enc":"#DCEDC8",
    "mask":        "#F8BBD0",   # pink for mask flow
    "residual":    "#ECEFF1",
    "block_bg":    "#FAFAFA",
    "head_bg":     "#FFF3E0",
    "enc_bg":      "#E3F2FD",
    "dec_bg":      "#FFF3E0",
}
EDGE   = "#37474F"
FONT   = "DejaVu Sans"
MASK_C = "#AD1457"   # colour for mask arrows / labels


# ═══════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════
class Box:
    """Stores geometry of a drawn box so arrows can snap to its edges."""
    __slots__ = ("cx", "cy", "w", "h")

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    @property
    def top(self):    return self.cy + self.h / 2
    @property
    def bot(self):    return self.cy - self.h / 2
    @property
    def left(self):   return self.cx - self.w / 2
    @property
    def right(self):  return self.cx + self.w / 2

    def edge(self, side):
        """Return (x, y) of a named edge midpoint."""
        if side == "top":    return (self.cx, self.top)
        if side == "bot":    return (self.cx, self.bot)
        if side == "left":   return (self.left, self.cy)
        if side == "right":  return (self.right, self.cy)
        raise ValueError(side)

    def edge_at_x(self, side, x):
        """Return (x, y) where y is the top/bot edge at a given x offset."""
        if side == "top":    return (x, self.top)
        if side == "bot":    return (x, self.bot)
        raise ValueError(side)


def draw_box(ax, cx, cy, w, h, text, color,
             fontsize=9, bold=False, text_color="black", edge_color=EDGE,
             linewidth=1.2, linestyle="-", zorder=2):
    """Draw a rounded rectangle and return its Box geometry."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.10",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontfamily=FONT, fontweight=weight,
            color=text_color, zorder=zorder + 1)
    return Box(cx, cy, w, h)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.3, style="-|>"):
    """Straight arrow between two exact points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0),
                zorder=3)


def connect(ax, src_box, src_side, dst_box, dst_side,
            color=EDGE, lw=1.3, src_x=None, dst_x=None):
    """Connect exact edges of two Box objects."""
    if src_x is not None:
        x1, y1 = src_box.edge_at_x(src_side, src_x)
    else:
        x1, y1 = src_box.edge(src_side)
    if dst_x is not None:
        x2, y2 = dst_box.edge_at_x(dst_side, dst_x)
    else:
        x2, y2 = dst_box.edge(dst_side)
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw)


def side_label(ax, text, x, y, fontsize=7.5, color=MASK_C, ha="left"):
    ax.text(x, y, text, fontsize=fontsize, fontfamily=FONT,
            color=color, ha=ha, va="center", style="italic", zorder=5)


# ═══════════════════════════════════════════════════════════════════════
# Diagram 1 — GaussianPatchTransformer (encoder-only)
# ═══════════════════════════════════════════════════════════════════════
def draw_gaussian_patch_transformer():
    fig, ax = plt.subplots(figsize=(10, 18))
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.5, 19.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(0, 19.0, "GaussianPatchTransformer",
            ha="center", va="center", fontsize=17, fontfamily=FONT, fontweight="bold")
    ax.text(0, 18.55, "(Encoder-Only  ·  Linear Encoding  ·  Index Positional Encoding)",
            ha="center", va="center", fontsize=10, fontfamily=FONT, color="#616161")

    # ── Row 0: Inputs ─────────────────────────────────────────────────
    y = 17.7
    b_neigh = draw_box(ax, -2.5, y, 3.4, 0.60,
                        "Neighborhood\n(B, N, attr_dim + 1)", COLORS["input"], fontsize=8.5)
    b_point = draw_box(ax, 2.5, y, 3.0, 0.60,
                        "Point Features\n(B, point_dim)", COLORS["input"], fontsize=8.5)
    b_mask  = draw_box(ax, 0.0, y - 0.95, 2.0, 0.50,
                        "Valid Mask\n(B, N)", COLORS["mask"], fontsize=8.5)

    # ── Row 1: Sort ───────────────────────────────────────────────────
    y_sort = 15.9
    b_sort = draw_box(ax, 0, y_sort, 4.6, 0.50,
                       "Sort Neighbors by Euclidean Distance", COLORS["pos"], fontsize=9.5)
    connect(ax, b_neigh, "bot", b_sort, "top", src_x=-2.5, dst_x=-1.0)
    connect(ax, b_point, "bot", b_sort, "top", src_x=2.5,  dst_x=1.0)

    # ── Row 2: Encoders ───────────────────────────────────────────────
    y_enc = 14.7
    b_nenc = draw_box(ax, -2.2, y_enc, 3.5, 0.65,
                       "Linear Neighbor Encoder\nLinear → LN → GELU  ×3\n→ (B, N, embed_dim)",
                       COLORS["encoder"], fontsize=7.5)
    b_penc = draw_box(ax, 2.2, y_enc, 3.2, 0.65,
                       "Linear Point Encoder\nLinear → LN → GELU  ×3\n→ (B, embed_dim)",
                       COLORS["encoder"], fontsize=7.5)
    connect(ax, b_sort, "bot", b_nenc, "top", src_x=-1.0, dst_x=-2.2)
    connect(ax, b_sort, "bot", b_penc, "top", src_x=1.0,  dst_x=2.2)

    # ── Row 3: Build token sequence ───────────────────────────────────
    y_seq = 13.5
    b_seq = draw_box(ax, 0, y_seq, 5.0, 0.55,
                      "[CLS]  +  Point Token  +  Neighbor Tokens\n"
                      "(B, 1 + 1 + N, embed_dim)",
                      COLORS["cls"], fontsize=8.5)
    connect(ax, b_nenc, "bot", b_seq, "top", dst_x=-1.2)
    connect(ax, b_penc, "bot", b_seq, "top", dst_x=1.2)

    # ── Row 4: Positional encoding ────────────────────────────────────
    y_pe = 12.5
    b_pe = draw_box(ax, 0, y_pe, 4.4, 0.50,
                     "⊕  Index Positional Encoding (sinusoidal)",
                     COLORS["pos"], fontsize=9)
    connect(ax, b_seq, "bot", b_pe, "top")

    # ── Row 5: Build attention mask from valid mask ───────────────────
    y_am = 11.5
    b_am = draw_box(ax, 0, y_am, 5.0, 0.55,
                     "Build Attention Mask\n"
                     "[True(CLS), True(point), valid_mask₁ … valid_maskₙ]\n→ (B, 1, 1, S)",
                     COLORS["mask"], fontsize=7.5)
    # Mask arrow from input mask box: elbow via right gutter
    gutter_x = 4.7
    arrow(ax, b_mask.right, b_mask.cy,
          gutter_x, b_mask.cy, color=MASK_C, lw=1.1, style="-")
    arrow(ax, gutter_x, b_mask.cy,
          gutter_x, y_am + b_am.h / 2, color=MASK_C, lw=1.1, style="-")
    arrow(ax, gutter_x, y_am + b_am.h / 2,
          b_am.right, y_am + b_am.h / 2, color=MASK_C, lw=1.1, style="-|>")
    side_label(ax, "valid_mask", b_mask.right + 0.15, b_mask.cy + 0.15, fontsize=7)

    # ── Row 6: Transformer Encoder Block  ×L ──────────────────────────
    y_blk = 9.7
    blk_h = 2.4
    bw = 5.2
    outer = FancyBboxPatch(
        (-bw / 2, y_blk - blk_h / 2), bw, blk_h,
        boxstyle="round,pad=0.18",
        facecolor=COLORS["block_bg"], edgecolor=EDGE,
        linewidth=1.6, linestyle="--", zorder=1)
    ax.add_patch(outer)
    ax.text(bw / 2 - 0.2, y_blk + blk_h / 2 - 0.15, "×L",
            fontsize=11, fontfamily=FONT, fontweight="bold",
            ha="right", va="top", color="#D32F2F", zorder=4)

    b_sa = draw_box(ax, 0, y_blk + 0.65, 4.4, 0.45,
                     "LayerNorm → Multi-Head Self-Attention + DropPath",
                     COLORS["attention"], fontsize=7.5)
    b_res1 = draw_box(ax, 0, y_blk + 0.05, 1.8, 0.32,
                       "⊕ Residual", COLORS["residual"], fontsize=7.5)
    b_ff = draw_box(ax, 0, y_blk - 0.55, 3.6, 0.45,
                     "LayerNorm → FeedForward (MLP) + DropPath",
                     COLORS["attention"], fontsize=7.5)
    b_res2 = draw_box(ax, 0, y_blk - 1.05, 1.8, 0.32,
                       "⊕ Residual", COLORS["residual"], fontsize=7.5)

    connect(ax, b_pe, "bot", b_sa, "top")
    connect(ax, b_sa, "bot", b_res1, "top")
    connect(ax, b_res1, "bot", b_ff, "top")
    connect(ax, b_ff, "bot", b_res2, "top")

    # Mask feeds into self-attention (elbow from gutter down)
    arrow(ax, gutter_x, y_am + b_am.h / 2,
          gutter_x, b_sa.cy, color=MASK_C, lw=1.0, style="-")
    arrow(ax, gutter_x, b_sa.cy,
          b_sa.right, b_sa.cy, color=MASK_C, lw=1.0, style="-|>")
    side_label(ax, "attn mask", gutter_x + 0.1, b_sa.cy + 0.18, fontsize=7)

    # ── Row 7: Encoder LayerNorm ──────────────────────────────────────
    y_ln = 7.7
    b_ln = draw_box(ax, 0, y_ln, 2.4, 0.42, "LayerNorm", COLORS["norm"], fontsize=9.5)
    connect(ax, b_res2, "bot", b_ln, "top")

    # ── Row 8: Extract CLS + Masked Pool ──────────────────────────────
    y_ex = 6.8
    b_cls_out = draw_box(ax, -1.8, y_ex, 2.5, 0.50,
                          "CLS Output\n(B, embed_dim)", COLORS["cls"], fontsize=8)
    b_pool = draw_box(ax, 1.8, y_ex, 2.6, 0.50,
                       "Max Pool (masked)\n(B, embed_dim)", COLORS["pool"], fontsize=8)
    connect(ax, b_ln, "bot", b_cls_out, "top", dst_x=-1.8, src_x=-0.5)
    connect(ax, b_ln, "bot", b_pool,    "top", dst_x=1.8,  src_x=0.5)

    # valid mask feeds into masked pooling (continue gutter down)
    arrow(ax, gutter_x, b_sa.cy,
          gutter_x, y_ex, color=MASK_C, lw=0.9, style="-")
    arrow(ax, gutter_x, y_ex,
          b_pool.right, y_ex, color=MASK_C, lw=0.9, style="-|>")
    side_label(ax, "valid mask\nfor pooling", gutter_x + 0.1, y_ex + 0.2, fontsize=6.5)

    # ── Row 9: Concatenate ────────────────────────────────────────────
    y_cat = 5.9
    b_cat = draw_box(ax, 0, y_cat, 3.8, 0.40,
                      "Concatenate → (B, 2 × embed_dim)", COLORS["residual"], fontsize=8.5)
    connect(ax, b_cls_out, "bot", b_cat, "top", dst_x=-0.8)
    connect(ax, b_pool,    "bot", b_cat, "top", dst_x=0.8)

    # ── Row 10: Prediction Head ───────────────────────────────────────
    y_hd = 4.4
    hd_h = 2.0
    head_outer = FancyBboxPatch(
        (-2.3, y_hd - hd_h / 2), 4.6, hd_h,
        boxstyle="round,pad=0.12",
        facecolor=COLORS["head_bg"], edgecolor=EDGE,
        linewidth=1.2, zorder=1)
    ax.add_patch(head_outer)
    ax.text(0, y_hd + hd_h / 2 - 0.13, "Prediction Head",
            fontsize=9.5, fontfamily=FONT, fontweight="bold",
            ha="center", va="top", zorder=4)

    b_h1 = draw_box(ax, 0, y_hd + 0.40, 3.8, 0.35,
                     "Linear(2D→256) → LN → GELU → Dropout", COLORS["head"], fontsize=7.5)
    b_h2 = draw_box(ax, 0, y_hd - 0.10, 3.8, 0.35,
                     "Linear(256→128) → LN → GELU → Dropout", COLORS["head"], fontsize=7.5)
    b_h3 = draw_box(ax, 0, y_hd - 0.60, 2.8, 0.35,
                     "Linear(128→1) → Softplus", COLORS["head"], fontsize=8)

    connect(ax, b_cat, "bot", b_h1, "top")
    connect(ax, b_h1, "bot", b_h2, "top")
    connect(ax, b_h2, "bot", b_h3, "top")

    # ── Row 11: Output ────────────────────────────────────────────────
    y_out = 2.7
    b_out = draw_box(ax, 0, y_out, 3.4, 0.50,
                      "Geodesic Distance  (B, 1)\n≥ 0", COLORS["output"],
                      fontsize=9.5, bold=True)
    connect(ax, b_h3, "bot", b_out, "top")

    # ── Legend ─────────────────────────────────────────────────────────
    ax.text(-5.0, 1.6, "Legend:", fontsize=8.5, fontweight="bold", fontfamily=FONT, va="top")
    legend_items = [
        (COLORS["input"],   "Input"),
        (COLORS["encoder"], "Encoder"),
        (COLORS["cls"],     "CLS / Sequence"),
        (COLORS["pos"],     "Positional Encoding"),
        (COLORS["mask"],    "Valid Mask Flow"),
        (COLORS["attention"],"Attention / FFN"),
        (COLORS["pool"],    "Pooling"),
        (COLORS["head"],    "Prediction Head"),
        (COLORS["output"],  "Output"),
    ]
    for i, (c, label) in enumerate(legend_items):
        yy = 1.2 - i * 0.35
        ax.add_patch(FancyBboxPatch((-5.0, yy - 0.12), 0.4, 0.24,
                                    boxstyle="round,pad=0.04", facecolor=c,
                                    edgecolor=EDGE, linewidth=0.8))
        ax.text(-4.45, yy, label, fontsize=7.5, fontfamily=FONT, va="center")

    fig.savefig("/home/rotem.shezaf/RaDe-GS/models/GaussianPatchTransformer_architecture.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved GaussianPatchTransformer_architecture.png")


# ═══════════════════════════════════════════════════════════════════════
# Diagram 2 — SplitEmbeddingEncoderDecoderTransformer
# ═══════════════════════════════════════════════════════════════════════
def draw_encoder_decoder_transformer():
    fig, ax = plt.subplots(figsize=(13, 22))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-1, 22.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # lane centres
    LX = -3.2   # encoder
    RX =  3.2   # decoder
    CX =  0.0   # shared bottom

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(0, 22.0, "GaussianPatch Encoder-Decoder Transformer",
            ha="center", va="center", fontsize=16, fontweight="bold", fontfamily=FONT)
    ax.text(0, 21.5, "(SplitEmbeddingEncoderDecoderTransformer  ·  Linear Encoding)",
            ha="center", va="center", fontsize=10, fontfamily=FONT, color="#616161")

    # ── Row 0: Inputs ─────────────────────────────────────────────────
    y_in = 20.7
    b_neigh = draw_box(ax, -1.8, y_in, 3.6, 0.60,
                        "Neighborhood\n(B, N, attr_dim + 1)", COLORS["input"], fontsize=8.5)
    b_point = draw_box(ax, 3.0, y_in, 2.8, 0.60,
                        "Point Features\n(B, point_dim)", COLORS["input"], fontsize=8.5)
    y_mk = y_in - 1.0
    b_mask = draw_box(ax, 0, y_mk, 2.2, 0.48,
                       "Valid Mask\n(B, N)", COLORS["mask"], fontsize=8.5)

    # ── Row 1: Sort ───────────────────────────────────────────────────
    y_sort = 18.8
    b_sort = draw_box(ax, 0, y_sort, 4.8, 0.50,
                       "Sort Neighbors by Euclidean Distance", COLORS["pos"], fontsize=9.5)
    connect(ax, b_neigh, "bot", b_sort, "top", dst_x=-1.0)
    connect(ax, b_point, "bot", b_sort, "top", dst_x=1.0)

    # ── Row 2: Split features ─────────────────────────────────────────
    y_sp = 17.8
    b_attr_feat = draw_box(ax, LX, y_sp, 3.5, 0.50,
                            "Attribute Features\n(B, N, attr_dim) — no geodesic",
                            COLORS["input"], fontsize=7.5)
    b_geo_feat  = draw_box(ax, RX, y_sp, 3.0, 0.50,
                            "Geodesic Distances\n(B, N, 1)",
                            COLORS["input"], fontsize=7.5)
    connect(ax, b_sort, "bot", b_attr_feat, "top", src_x=-1.2, dst_x=LX)
    connect(ax, b_sort, "bot", b_geo_feat,  "top", src_x=1.2,  dst_x=RX)

    # ── Row 3: Embedding layers ───────────────────────────────────────
    y_emb = 16.7
    b_attr_enc = draw_box(ax, LX, y_emb, 3.6, 0.65,
                           "Linear Attribute Encoder\nLinear → LN → GELU  ×3\n→ (B, N, embed_dim)",
                           COLORS["encoder"], fontsize=7.5)
    b_geo_enc  = draw_box(ax, RX, y_emb, 3.4, 0.65,
                           "Geodesic Embedding\nLinear(1→128) → GELU → Linear\n→ (B, N, embed_dim)",
                           COLORS["geodesic_enc"], fontsize=7.5)
    connect(ax, b_attr_feat, "bot", b_attr_enc, "top")
    connect(ax, b_geo_feat,  "bot", b_geo_enc,  "top")

    # ═══════════════════════════════════════════════════════════════════
    # ENCODER (left)
    # ═══════════════════════════════════════════════════════════════════
    y_eblk = 14.6
    eblk_h = 2.6
    ebw = 4.4
    enc_outer = FancyBboxPatch(
        (LX - ebw / 2, y_eblk - eblk_h / 2), ebw, eblk_h,
        boxstyle="round,pad=0.18", facecolor=COLORS["enc_bg"],
        edgecolor="#1565C0", linewidth=1.6, linestyle="--", zorder=1)
    ax.add_patch(enc_outer)
    ax.text(LX, y_eblk + eblk_h / 2 - 0.12, "Encoder  ×L_enc",
            fontsize=9.5, fontweight="bold", fontfamily=FONT,
            ha="center", va="top", color="#1565C0", zorder=4)

    b_esa = draw_box(ax, LX, y_eblk + 0.6, 3.8, 0.42,
                      "LayerNorm → Self-Attention (no mask)\n+ DropPath",
                      COLORS["attention"], fontsize=7)
    b_er1 = draw_box(ax, LX, y_eblk + 0.0, 1.6, 0.30,
                      "⊕ Residual", COLORS["residual"], fontsize=7)
    b_eff = draw_box(ax, LX, y_eblk - 0.55, 3.4, 0.42,
                      "LayerNorm → FeedForward + DropPath",
                      COLORS["attention"], fontsize=7)
    b_er2 = draw_box(ax, LX, y_eblk - 1.1, 1.6, 0.30,
                      "⊕ Residual", COLORS["residual"], fontsize=7)

    connect(ax, b_attr_enc, "bot", b_esa, "top")
    connect(ax, b_esa, "bot", b_er1, "top")
    connect(ax, b_er1, "bot", b_eff, "top")
    connect(ax, b_eff, "bot", b_er2, "top")

    y_eln = 12.5
    b_eln = draw_box(ax, LX, y_eln, 2.4, 0.42,
                      "Encoder LayerNorm", COLORS["norm"], fontsize=8.5)
    connect(ax, b_er2, "bot", b_eln, "top")

    # ═══════════════════════════════════════════════════════════════════
    # DECODER (right)
    # ═══════════════════════════════════════════════════════════════════
    # CLS prepend
    y_cls = 15.6
    b_cls_pre = draw_box(ax, RX, y_cls, 3.4, 0.50,
                          "Prepend [CLS] Token\n(B, 1+N, embed_dim)", COLORS["cls"], fontsize=8)
    connect(ax, b_geo_enc, "bot", b_cls_pre, "top")

    # Build decoder mask from valid mask
    y_dmask = 14.7
    b_dmask = draw_box(ax, RX, y_dmask, 3.6, 0.50,
                        "Decoder Attn Mask\n[True(CLS), valid_mask₁…ₙ]\n→ (B, 1, 1, 1+N)",
                        COLORS["mask"], fontsize=7)
    # Route valid mask to decoder mask box via right gutter
    gutter_x = 5.8
    arrow(ax, b_mask.right, b_mask.cy,
          gutter_x, b_mask.cy, color=MASK_C, lw=1.1, style="-")
    arrow(ax, gutter_x, b_mask.cy,
          gutter_x, b_dmask.cy, color=MASK_C, lw=1.1, style="-")
    arrow(ax, gutter_x, b_dmask.cy,
          b_dmask.right, b_dmask.cy, color=MASK_C, lw=1.1, style="-|>")
    side_label(ax, "valid_mask", b_mask.right + 0.15, b_mask.cy + 0.18, fontsize=7)

    connect(ax, b_cls_pre, "bot", b_dmask, "top")

    # Decoder block
    y_dblk = 12.3
    dblk_h = 3.8
    dbw = 4.6
    dec_outer = FancyBboxPatch(
        (RX - dbw / 2, y_dblk - dblk_h / 2), dbw, dblk_h,
        boxstyle="round,pad=0.18", facecolor=COLORS["dec_bg"],
        edgecolor="#E65100", linewidth=1.6, linestyle="--", zorder=1)
    ax.add_patch(dec_outer)
    ax.text(RX, y_dblk + dblk_h / 2 - 0.12, "Decoder  ×L_dec",
            fontsize=9.5, fontweight="bold", fontfamily=FONT,
            ha="center", va="top", color="#E65100", zorder=4)

    b_dsa = draw_box(ax, RX, y_dblk + 1.30, 4.0, 0.42,
                      "LayerNorm → Masked Self-Attention\n+ DropPath",
                      COLORS["attention"], fontsize=7)
    b_dr1 = draw_box(ax, RX, y_dblk + 0.70, 1.6, 0.30,
                      "⊕ Residual", COLORS["residual"], fontsize=7)
    b_dca = draw_box(ax, RX, y_dblk + 0.10, 4.0, 0.42,
                      "LayerNorm → Cross-Attention (Q=dec, KV=enc)\n+ DropPath",
                      COLORS["cross_attn"], fontsize=7)
    b_dr2 = draw_box(ax, RX, y_dblk - 0.50, 1.6, 0.30,
                      "⊕ Residual", COLORS["residual"], fontsize=7)
    b_dff = draw_box(ax, RX, y_dblk - 1.10, 3.4, 0.42,
                      "LayerNorm → FeedForward + DropPath",
                      COLORS["attention"], fontsize=7)
    b_dr3 = draw_box(ax, RX, y_dblk - 1.65, 1.6, 0.30,
                      "⊕ Residual", COLORS["residual"], fontsize=7)

    connect(ax, b_dmask, "bot", b_dsa, "top")
    connect(ax, b_dsa, "bot", b_dr1, "top")
    connect(ax, b_dr1, "bot", b_dca, "top")
    connect(ax, b_dca, "bot", b_dr2, "top")
    connect(ax, b_dr2, "bot", b_dff, "top")
    connect(ax, b_dff, "bot", b_dr3, "top")

    # Mask into masked self-attention (from gutter down to dsa right edge)
    arrow(ax, gutter_x, b_dmask.cy,
          gutter_x, b_dsa.cy, color=MASK_C, lw=1.0, style="-")
    arrow(ax, gutter_x, b_dsa.cy,
          b_dsa.right, b_dsa.cy, color=MASK_C, lw=1.0, style="-|>")
    side_label(ax, "attn\nmask", gutter_x + 0.1, b_dsa.cy + 0.25, fontsize=6.5)

    # Cross-attention KV from encoder
    arrow(ax, b_eln.right, b_eln.cy,
          b_dca.left, b_dca.cy, color="#E65100", lw=1.4)
    ax.text((b_eln.right + b_dca.left) / 2, b_dca.cy + 0.28,
            "KV from Encoder", fontsize=7.5, fontfamily=FONT,
            ha="center", va="center", color="#E65100", fontweight="bold", zorder=5)

    # Decoder LayerNorm
    y_dln = 9.6
    b_dln = draw_box(ax, RX, y_dln, 2.6, 0.42,
                      "Decoder LayerNorm", COLORS["norm"], fontsize=8.5)
    connect(ax, b_dr3, "bot", b_dln, "top")

    # ═══════════════════════════════════════════════════════════════════
    # Pool & Predict (centred)
    # ═══════════════════════════════════════════════════════════════════

    # Extract CLS + Pool
    y_ex = 8.5
    b_cls_o = draw_box(ax, CX - 1.8, y_ex, 2.5, 0.50,
                        "CLS Output\n(B, embed_dim)", COLORS["cls"], fontsize=8)
    b_pool  = draw_box(ax, CX + 1.8, y_ex, 2.6, 0.50,
                        "Max Pool (masked)\n(B, embed_dim)", COLORS["pool"], fontsize=8)
    connect(ax, b_dln, "bot", b_cls_o, "top", src_x=RX - 0.5, dst_x=CX - 1.8)
    connect(ax, b_dln, "bot", b_pool,  "top", src_x=RX + 0.5, dst_x=CX + 1.8)

    # Mask into pooling (continue gutter line down)
    arrow(ax, gutter_x, b_dsa.cy,
          gutter_x, y_ex, color=MASK_C, lw=0.9, style="-")
    arrow(ax, gutter_x, y_ex,
          b_pool.right, y_ex, color=MASK_C, lw=0.9, style="-|>")
    side_label(ax, "valid mask\nfor pooling", gutter_x + 0.1, y_ex + 0.22, fontsize=6.5)

    # Concatenate
    y_cat = 7.5
    b_cat = draw_box(ax, CX, y_cat, 3.8, 0.40,
                      "Concatenate → (B, 2 × embed_dim)", COLORS["residual"], fontsize=8.5)
    connect(ax, b_cls_o, "bot", b_cat, "top", dst_x=CX - 0.8)
    connect(ax, b_pool,  "bot", b_cat, "top", dst_x=CX + 0.8)

    # Prediction Head
    y_hd = 6.0
    hd_h = 2.0
    head_outer = FancyBboxPatch(
        (CX - 2.3, y_hd - hd_h / 2), 4.6, hd_h,
        boxstyle="round,pad=0.12", facecolor=COLORS["head_bg"],
        edgecolor=EDGE, linewidth=1.2, zorder=1)
    ax.add_patch(head_outer)
    ax.text(CX, y_hd + hd_h / 2 - 0.13, "Prediction Head",
            fontsize=9.5, fontfamily=FONT, fontweight="bold",
            ha="center", va="top", zorder=4)

    b_h1 = draw_box(ax, CX, y_hd + 0.40, 3.8, 0.35,
                     "Linear(2D→256) → LN → GELU → Dropout", COLORS["head"], fontsize=7.5)
    b_h2 = draw_box(ax, CX, y_hd - 0.10, 3.8, 0.35,
                     "Linear(256→128) → LN → GELU → Dropout", COLORS["head"], fontsize=7.5)
    b_h3 = draw_box(ax, CX, y_hd - 0.60, 2.8, 0.35,
                     "Linear(128→1) → Softplus", COLORS["head"], fontsize=8)

    connect(ax, b_cat, "bot", b_h1, "top")
    connect(ax, b_h1, "bot", b_h2, "top")
    connect(ax, b_h2, "bot", b_h3, "top")

    # Output
    y_out = 4.2
    b_out = draw_box(ax, CX, y_out, 3.4, 0.50,
                      "Geodesic Distance  (B, 1)\n≥ 0", COLORS["output"],
                      fontsize=9.5, bold=True)
    connect(ax, b_h3, "bot", b_out, "top")

    # ── Legend ─────────────────────────────────────────────────────────
    ax.text(-6.5, 3.2, "Legend:", fontsize=8.5, fontweight="bold", fontfamily=FONT, va="top")
    legend_items = [
        (COLORS["input"],      "Input"),
        (COLORS["encoder"],    "Attribute Encoder"),
        (COLORS["geodesic_enc"],"Geodesic Encoder"),
        (COLORS["cls"],        "CLS / Sequence"),
        (COLORS["mask"],       "Valid Mask Flow"),
        (COLORS["attention"],  "Self-Attn / FFN"),
        (COLORS["cross_attn"], "Cross-Attention"),
        (COLORS["pool"],       "Pooling"),
        (COLORS["head"],       "Prediction Head"),
        (COLORS["output"],     "Output"),
    ]
    for i, (c, label) in enumerate(legend_items):
        yy = 2.8 - i * 0.35
        ax.add_patch(FancyBboxPatch((-6.5, yy - 0.12), 0.4, 0.24,
                                    boxstyle="round,pad=0.04", facecolor=c,
                                    edgecolor=EDGE, linewidth=0.8))
        ax.text(-5.95, yy, label, fontsize=7.5, fontfamily=FONT, va="center")

    fig.savefig("/home/rotem.shezaf/RaDe-GS/models/GaussianPatchEncoderDecoder_architecture.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved GaussianPatchEncoderDecoder_architecture.png")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    draw_gaussian_patch_transformer()
    draw_encoder_decoder_transformer()
    print("Done!")
