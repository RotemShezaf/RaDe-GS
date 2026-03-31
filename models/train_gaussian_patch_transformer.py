"""
Training script for GaussianPatchTransformer

Trains the GaussianPatchTransformer model on Gaussian patch data
for geodesic distance prediction.

Usage:
    # Minimal – only CLI args
    python models/train_gaussian_patch_transformer.py \
        --dataset_config DataSets/configs/polynomial/saddle_all.yaml --ring 2

    # With training config file
    python models/train_gaussian_patch_transformer.py \
        --train_config models/configs/default_train.yaml \
        --dataset_config DataSets/configs/polynomial/saddle_all.yaml --ring 2

    # Override any value from CLI
    python models/train_gaussian_patch_transformer.py \
        --train_config models/configs/default_train.yaml \
        --dataset_config DataSets/configs/polynomial/saddle_all.yaml \
        --ring 3 --batch_size 64 --use_wandb
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import yaml
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from args import parse_train_args, args_to_dict
from const import build_transforms
from GaussianPatchTransformer import create_gaussian_patch_transformer
from trainer import Trainer, EarlyStopping
from DataSets.gaussian_dataset import (
    GaussianPatchDataset,
    GaussianPatchSubsetDataset,
    CombinedGaussianPatchDataset,
)


# ───────────────────────────── helpers ───────────────────────────────

def setup_logger(log_file: str = None) -> logging.Logger:
    """Create a logger that writes to console and optionally to a file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def collate_fn(batch):
    """Collate (neighborhood, point_features, target, valid_mask) tuples."""
    neighborhoods = torch.stack([item[0] for item in batch])
    point_features = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    valid_masks = torch.stack([item[3] for item in batch])
    return neighborhoods, point_features, targets, valid_masks


def create_dataloaders(args, dataset_config):
    """
    Create train / val DataLoaders from dataset config.

    Returns:
        (train_loader, val_loader)
    """
    attributes = dataset_config.get("attributes", ["xyz"])
    # Exclude internal-only attributes (prefixed with _) from model features;
    # these are kept in the data on disk and handled separately by the dataset.
    attributes = [a for a in attributes if not a.startswith('_')]
    mask_constant = dataset_config.get("mask_constant", -10.0)

    dataset_class_name = dataset_config.get("dataset_class", "GaussianPatchDataset")
    _DATASET_CLASSES = {
        "GaussianPatchDataset": GaussianPatchDataset,
        "GaussianPatchSubsetDataset": GaussianPatchSubsetDataset,
        "CombinedGaussianPatchDataset": CombinedGaussianPatchDataset,
    }

    # Auto-upgrade to subset class when max_examples / subset_fraction is set
    need_subset = getattr(args, "max_examples", None) is not None or \
                  getattr(args, "subset_fraction", None) is not None
    if need_subset and dataset_class_name == "GaussianPatchDataset":
        dataset_class_name = "GaussianPatchSubsetDataset"

    if dataset_class_name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_class '{dataset_class_name}' in config. "
            f"Supported: {list(_DATASET_CLASSES)}"
        )
    DatasetClass = _DATASET_CLASSES[dataset_class_name]

    # GaussianPatchDataset / Subset uses 'config_path'; Combined uses 'config'
    config_kwarg = "config" if DatasetClass is CombinedGaussianPatchDataset else "config_path"

    # Build data augmentation transform from args.transforms
    transform = build_transforms(
        getattr(args, "transforms", None),
        attributes=attributes,
        mask_constant=mask_constant,
    )

    # Extra kwargs for GaussianPatchSubsetDataset
    subset_kwargs = {}
    if DatasetClass is GaussianPatchSubsetDataset:
        if getattr(args, "max_examples", None) is not None:
            subset_kwargs["max_examples"] = args.max_examples
        if getattr(args, "subset_fraction", None) is not None:
            subset_kwargs["subset_fraction"] = args.subset_fraction
        subset_kwargs["subset_seed"] = getattr(args, "subset_seed", 42)

    dataset = DatasetClass(
        **{config_kwarg: args.dataset_config},
        attributes=attributes,
        ring=args.ring,
        use_r1_min=args.use_r1_min,
        transform=transform,
        **subset_kwargs,
    )

    # ── Post-creation subsetting for datasets that don't support it natively ──
    # GaussianPatchSubsetDataset handles subsetting internally.
    # For other dataset classes (e.g. CombinedGaussianPatchDataset), we apply
    # a torch.utils.data.Subset wrapper after creation.
    if need_subset and DatasetClass is not GaussianPatchSubsetDataset:
        full_len = len(dataset)
        max_ex = getattr(args, "max_examples", None)
        frac = getattr(args, "subset_fraction", None)
        if max_ex is not None:
            target_len = min(max_ex, full_len)
        elif frac is not None:
            target_len = max(1, int(full_len * frac))
        else:
            target_len = full_len

        if target_len < full_len:
            subset_seed = getattr(args, "subset_seed", 42)
            gen = torch.Generator().manual_seed(subset_seed)
            indices = torch.randperm(full_len, generator=gen)[:target_len].sort()[0]
            dataset = torch.utils.data.Subset(dataset, indices.tolist())
            print(
                f"Subset: kept {target_len:,} / {full_len:,} examples "
                f"({target_len / full_len:.1%}, seed={subset_seed})"
            )

    total = len(dataset)
    train_size = int(total * args.train_split)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=True, **loader_kwargs)

    return train_loader, val_loader


# ──────────────────────────── main ────────────────────────────────────

def main():
    args = parse_train_args()

    # ── Logger ────────────────────────────────────────────────────────
    logger = setup_logger(args.log_file)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Dataset config ────────────────────────────────────────────────
    if args.dataset_config is None:
        raise ValueError("--dataset_config is required")

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    attributes = dataset_config.get("attributes", ["xyz"])
    attributes = [a for a in attributes if not a.startswith('_')]
    use_mahalanobis = dataset_config.get("use_mahalanobis", False)
    method = "mahalanobis" if use_mahalanobis else "euclidean"
    max_neighbors = dataset_config["ring_size_mapping"][method][args.ring]

    # Resolve point_attributes
    point_attributes = args.point_attributes  # None | [] | list-of-strings

    logger.info(f"\nDataset configuration:")
    logger.info(f"  Config:          {args.dataset_config}")
    logger.info(f"  Attributes:      {attributes}")
    logger.info(f"  Point attrs:     {point_attributes}")
    logger.info(f"  Ring:            {args.ring}")
    logger.info(f"  Max neighbors:   {max_neighbors}")
    transforms_cfg = getattr(args, "transforms", None)
    logger.info(f"  Transforms:      {transforms_cfg if transforms_cfg else 'none'}")
    if getattr(args, "max_examples", None):
        logger.info(f"  Max examples:    {args.max_examples:,}")
    elif getattr(args, "subset_fraction", None):
        logger.info(f"  Subset fraction: {args.subset_fraction}")

    # ── Data loaders ──────────────────────────────────────────────────
    logger.info("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(args, dataset_config)
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("\nCreating model...")
    model = create_gaussian_patch_transformer(
        attributes=attributes,
        point_attributes=point_attributes,
        max_neighbors=max_neighbors,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        pool=args.pool,
        pos_encoding_type=args.pos_encoding_type,
        encoder_type=args.encoder_type,
        attention_type=args.attention_type,
        decoder_depth=args.decoder_depth,
        use_cls_token=args.use_cls_token,
    )
    model.print_model_info()

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = getattr(args, "warmup_epochs", 0) or 0
    cosine_epochs = max(1, args.num_epochs - warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.min_learning_rate,
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    # ── Early stopping (no weight restore – just track best) ──────────
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode="min",
            restore_best_weights=False,
        )

    # ── W&B ───────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args_to_dict(args),       # save full training args
        )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_type=args.loss_type,
        huber_delta=getattr(args, 'huber_delta', 1.0),
        blended_mse_lambda=getattr(args, 'blended_mse_lambda', 0.5),
        gradient_clip_norm=getattr(args, 'gradient_clip_norm', 0.0),
        early_stopping=early_stopping,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        logger=logger,
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    if getattr(args, "resume", False):
        ckpt_path = Path(args.save_dir) / "best_model.pth"
        if ckpt_path.exists():
            logger.info(f"\nResuming training from checkpoint: {ckpt_path}")
            checkpoint = trainer.load_checkpoint(str(ckpt_path))
            start_epoch = trainer.current_epoch + 1
            logger.info(f"  Resumed from epoch {trainer.current_epoch + 1}")
            logger.info(f"  Best val MAE so far: {trainer.best_val_mae:.6f}")
            # Restore early-stopping counter from history if available
            if early_stopping is not None and trainer.history.get("val_mae"):
                val_maes = trainer.history["val_mae"]
                best_mae = min(val_maes)
                early_stopping.best_score = best_mae
                early_stopping.best_epoch = val_maes.index(best_mae)
                # Approximate counter from epochs since last improvement
                last_improvement = max(
                    i for i, v in enumerate(val_maes) if v <= best_mae + early_stopping.min_delta
                )
                early_stopping.counter = len(val_maes) - 1 - last_improvement
                logger.info(
                    f"  Early-stopping counter restored: {early_stopping.counter}/{args.patience}"
                )
        else:
            logger.warning(
                f"--resume requested but no checkpoint found at {ckpt_path}. "
                "Starting from scratch."
            )

    # ── Train ─────────────────────────────────────────────────────────
    if start_epoch >= args.num_epochs:
        logger.info(
            f"\nCheckpoint epoch ({start_epoch}) >= num_epochs ({args.num_epochs}). "
            "Nothing to train – increase --num_epochs to continue."
        )
    else:
        logger.info(f"\nStarting training from epoch {start_epoch + 1}/{args.num_epochs}...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            start_epoch=start_epoch,
        )

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
