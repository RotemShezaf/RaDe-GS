"""
Argument parsing for GaussianPatchTransformer training.

Supports both command-line arguments and YAML config files.
CLI arguments override config file values.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# Defaults – kept in sync with models/configs/default_train.yaml
# ============================================================================
DEFAULTS = {
    # Dataset
    "dataset_config": None,
    "ring": 2,
    "train_split": 0.8,
    "use_r1_min": False,
    # transforms: list of dicts [{"name": "...", ...kwargs}] or list of name strings
    # Use YAML config for transforms with non-default params.
    # See models/const.py for the full registry.
    "transforms": [],
    # Subset controls – cap the number of loaded examples (GaussianPatchSubsetDataset).
    # max_examples takes precedence over subset_fraction.  None = use full dataset.
    "max_examples": None,
    "subset_fraction": None,
    "subset_seed": 42,

    # Model
    "embed_dim": 64,
    "encoder_depth": 6,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "drop_path_rate": 0.1,
    "point_attributes": None,
    # Pooling strategy for aggregating neighbor tokens before the prediction head.
    # "max"      – masked max-pool (default, original behaviour)
    # "mean"     – masked mean-pool
    # "max_mean" – concat max and mean (doubles pool contribution to head)
    "pool": "max",
    # Positional encoding type.
    # "index"          – sinusoidal on sorted-neighbour order (default)
    # "spatial"        – sinusoidal on relative 3-D XYZ position (requires xyz in attributes)
    # "relative_bias"  – learned per-head attn bias from pairwise 3-D positions (every layer)
    "pos_encoding_type": "index",
    # Encoder implementation.
    # "conv"     – Conv1d PointNet-style with global context injection (default)
    # "linear"   – MLP (Linear + LayerNorm + GELU), original implementation
    # "residual" – MLP with residual block, no LayerNorm, LeakyReLU
    "encoder_type": "conv",
    # Attention mechanism.
    # "standard" – vanilla multi-head self-attention (default)
    # "geodesic" – geodesic self-attention (encoder-only): replaces Q·K^T with GDS-based
    #              attention weights from token XYZ positions
    # "geodesic_enc_dec" – encoder-decoder: encoder uses geodesic self-attention on XYZ,
    #                      decoder cross-attends with raw geodesic distances
    # "split" – split embedding encoder-only: interleaved attribute/geodesic tokens with
    #           type embedding, attention mask only on geodesic tokens
    # "split_enc_dec" – split embedding encoder-decoder: encoder on attributes (no mask),
    #                   decoder on geodesic with cross-attention, regular attention
    "attention_type": "standard",
    "decoder_depth": 3,
    # CLS token
    # True  – prepend a learnable [CLS] token to the sequence (default, original behaviour)
    # False – no CLS token; prediction head receives only pooled neighbor features
    "use_cls_token": True,

    # Training
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "warmup_epochs": 0,
    "weight_decay": 1e-4,
    "loss_type": "mse",  # mse | l1 | smooth_l1 | huber | log_mse | log_l1 | blended_mse
    "huber_delta": 1.0,  # delta for Huber loss (transition from quadratic to linear)
    "blended_mse_lambda": 0.5,  # weight for linear-space MSE in blended_mse (1-lambda = log-space MSE weight)
    "gradient_clip_norm": 0.0,  # max gradient norm (0 = disabled)
    "seed": 42,

    # Early stopping
    "early_stopping": True,
    "patience": 10,
    "min_delta": 0.0,

    # Infrastructure
    "save_dir": "./checkpoints",
    "save_every": 10,
    "num_workers": 4,

    # Logging
    "use_wandb": False,
    "wandb_project": "gaussian-patch-transformer",
    "wandb_run_name": None,
    "log_file": None,

    # Resume
    "resume": False,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the ArgumentParser with all training flags."""
    parser = argparse.ArgumentParser(
        description="Train GaussianPatchTransformer for geodesic distance prediction"
    )

    # ── Config file ────────────────────────────────────────────────────
    parser.add_argument(
        "--train_config", type=str, default=None,
        help="Path to training YAML config (CLI args override config values)"
    )

    # ── Dataset ────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset_config", type=str, default=DEFAULTS["dataset_config"],
        help="Path to dataset YAML config (ring sizes, attributes, paths)"
    )
    parser.add_argument("--ring", type=int, default=None, help="Ring number to train on")
    parser.add_argument("--train_split", type=float, default=None, help="Fraction of data for training")
    parser.add_argument("--use_r1_min", action="store_true", default=None,
                        help="Use r1_min for dropout augmentation")
    parser.add_argument(
        "--transforms", type=str, nargs="*", default=None, metavar="TRANSFORM",
        help=(
            "Data augmentation transforms applied to training patches. "
            "Pass one or more transform names (default parameters will be used). "
            "For custom parameters use the YAML 'dataset.transforms' key. "
            "Available: PointcloudRandomInputDropout, GaussianPatchDropout, "
            "GaussianPatchRotate, GaussianPatchRandomFlip, GaussianPatchCanonicalRotate"
        ),
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Cap the number of loaded examples (uses GaussianPatchSubsetDataset). "
             "Takes precedence over --subset_fraction."
    )
    parser.add_argument(
        "--subset_fraction", type=float, default=None,
        help="Fraction of examples to keep (0 < f <= 1). "
             "Ignored when --max_examples is set."
    )
    parser.add_argument(
        "--subset_seed", type=int, default=None,
        help="RNG seed for the random subset (default 42)."
    )

    # ── Model ──────────────────────────────────────────────────────────
    parser.add_argument("--embed_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=None, help="Number of encoder blocks")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=None, help="MLP hidden dim ratio")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--attn_dropout", type=float, default=None, help="Attention dropout rate")
    parser.add_argument("--drop_path_rate", type=float, default=None, help="Drop path (stochastic depth) rate")
    parser.add_argument(
        "--point_attributes", type=str, nargs="*", default=None,
        help="Point attributes (None=all, empty=no point token)"
    )
    parser.add_argument(
        "--pool", type=str, default=None,
        choices=["max", "mean", "max_mean"],
        help="Pooling strategy for neighbor tokens: max (default), mean, or max_mean (both concatenated)"
    )
    parser.add_argument(
        "--pos_encoding_type", type=str, default=None,
        choices=["index", "spatial", "relative_bias"],
        help="Positional encoding: 'index' (sorted-order sinusoidal, default), 'spatial' (relative-XYZ sinusoidal), or 'relative_bias' (learned per-head bias at every layer)"
    )
    parser.add_argument(
        "--encoder_type", type=str, default=None,
        choices=["conv", "linear", "residual"],
        help="Encoder implementation: 'conv' (Conv1d PointNet-style, default), 'linear' (MLP, original), or 'residual' (MLP with residual block, no LayerNorm, LeakyReLU)"
    )
    parser.add_argument(
        "--attention_type", type=str, default=None,
        choices=["standard", "geodesic", "geodesic_enc_dec", "split", "split_enc_dec"],
        help="Attention mechanism: 'standard' (vanilla MHA), 'geodesic' (encoder-only GDS attention), "
             "'geodesic_enc_dec' (encoder-decoder with GDS encoder and geodesic-distance decoder), "
             "'split' (interleaved attr/geo tokens with type embedding), "
             "'split_enc_dec' (encoder on attributes, decoder on geodesic with cross-attention)"
    )
    parser.add_argument(
        "--decoder_depth", type=int, default=None,
        help="Number of decoder blocks (only used with attention_type='geodesic_enc_dec', default 3)"
    )
    parser.add_argument(
        "--use_cls_token", action="store_true", default=None,
        help="Prepend a learnable [CLS] token to the sequence (default)"
    )
    parser.add_argument(
        "--no_cls_token", dest="use_cls_token", action="store_false",
        help="Disable the [CLS] token; prediction head uses only pooled features"
    )

    # ── Training ───────────────────────────────────────────────────────
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=None, help="Minimum LR for cosine schedule")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Linear LR warmup epochs (0 = no warmup)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument(
        "--loss_type", type=str, default=None,
        choices=["mse", "l1", "smooth_l1", "huber", "log_mse", "log_l1", "blended_mse"],
        help="Loss function. 'huber' is quadratic for small errors, linear for large (see --huber_delta). "
             "'blended_mse' blends linear-space MSE and log-space MSE (see --blended_mse_lambda). "
             "log_* variants compute loss in log-distance space (penalises relative error equally across all scales)"
    )
    parser.add_argument(
        "--huber_delta", type=float, default=None,
        help="Delta for Huber loss: errors below delta are squared, above are linear (default 1.0)"
    )
    parser.add_argument(
        "--blended_mse_lambda", type=float, default=None,
        help="Weight for linear-space MSE in blended_mse loss (default 0.5). "
             "Loss = lambda*MSE_linear + (1-lambda)*MSE_log. "
             "0.0 = pure log-space MSE, 1.0 = pure linear MSE."
    )
    parser.add_argument(
        "--gradient_clip_norm", type=float, default=None,
        help="Max gradient norm for gradient clipping (0 = disabled, default 0)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # ── Early stopping ─────────────────────────────────────────────────
    parser.add_argument("--early_stopping", action="store_true", default=None,
                        help="Enable early stopping")
    parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=None, help="Early-stopping min delta")

    # ── Infrastructure ─────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")

    # ── Logging ────────────────────────────────────────────────────────
    parser.add_argument("--use_wandb", action="store_true", default=None, help="Enable W&B logging")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to training log file (logs to console and file)")

    # ── Resume ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--resume", action="store_true", default=None,
        help="Resume training from best_model.pth found in --save_dir"
    )

    return parser


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_train_args(argv=None) -> argparse.Namespace:
    """
    Parse training arguments.

    Resolution order (later overrides earlier):
        1. Built-in DEFAULTS
        2. YAML config from ``--train_config``
        3. Explicit CLI arguments

    Returns:
        ``argparse.Namespace`` with all training parameters resolved.
    """
    parser = _build_parser()
    cli_args = parser.parse_args(argv)

    # Start with defaults
    merged: Dict[str, Any] = dict(DEFAULTS)

    # Layer YAML config on top
    if cli_args.train_config is not None:
        yaml_cfg = _load_yaml_config(cli_args.train_config)
        # Flatten nested sections (model, training, etc.) into top-level keys
        for key, val in yaml_cfg.items():
            if isinstance(val, dict) and key in ("model", "training", "early_stopping_cfg",
                                                  "infrastructure", "logging", "dataset"):
                merged.update(val)
            else:
                merged[key] = val

    # Layer explicit CLI values on top (skip None – means "not set by user")
    for key, val in vars(cli_args).items():
        if key == "train_config":
            continue
        if val is not None:
            merged[key] = val

    # Coerce types: YAML values bypass argparse's type= converters,
    # so e.g. a learning_rate may arrive as a string.
    for action in parser._actions:
        dest = action.dest
        if dest in merged and merged[dest] is not None and action.type is not None:
            val = merged[dest]
            # Skip list/dict values (e.g. transforms from YAML) – they don't
            # need scalar coercion and calling str() on them would break them.
            if isinstance(val, (list, dict)):
                continue
            try:
                merged[dest] = action.type(val)
            except (ValueError, TypeError):
                pass

    # Build final namespace
    ns = argparse.Namespace(**merged)

    # Keep train_config path for reference
    ns.train_config = cli_args.train_config

    return ns


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert a Namespace to a plain dict (for wandb / serialisation)."""
    return {k: v for k, v in vars(args).items()}
