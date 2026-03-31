#!/usr/bin/env python3
"""
Example: How to use transforms with GaussianPatchDataset

This script demonstrates various ways to use data augmentation transforms
with the GaussianPatchDataset for training geodesic distance prediction models.
"""

import torch
from pathlib import Path
from gaussian_dataset import GaussianPatchDataset, create_dataloaders
from data_transformation import (
    GaussianPatchDropout,
    GaussianPatchRotate,
    GaussianPatchCanonicalRotate
)


# Example 1: Single transform - Dropout
def example_dropout():
    """Use dropout augmentation to randomly drop far neighbors."""
    print("\n" + "="*80)
    print("Example 1: Using GaussianPatchDropout")
    print("="*80)
    
    # Create dropout transform
    dropout_transform = GaussianPatchDropout(
        max_dropout_ratio=0.3,  # Drop up to 30% of eligible neighbors
        attributes=["xyz", "normals", "opacity"],
        max_neighbors=64,
        use_r1_min_val=True,
        mask_constant=-10
    )
    
    # Create dataset with transform
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=dropout_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform will randomly drop neighbors with geodesic distance > r1_min_val")
    
    # Test loading
    features, target = dataset[0]
    print(f"\nSample shape: features={features.shape}, target={target.shape}")
    

# Example 2: Single transform - Random rotation
def example_rotation():
    """Use random rotation augmentation."""
    print("\n" + "="*80)
    print("Example 2: Using GaussianPatchRotate")
    print("="*80)
    
    # Create rotation transform
    rotation_transform = GaussianPatchRotate(
        attributes=["xyz", "normals", "opacity", "rotation"],
        max_neighbors=64,
        use_r1_min_val=True
    )
    
    # Create dataset with transform
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity", "rotation"],
        ring=2,
        use_r1_min=True,
        transform=rotation_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform will apply random rotation around y-axis")


# Example 3: Single transform - Canonical rotation
def example_canonical_rotation():
    """Use canonical rotation to align patches."""
    print("\n" + "="*80)
    print("Example 3: Using GaussianPatchCanonicalRotate")
    print("="*80)
    
    # Create canonical rotation transform
    canonical_transform = GaussianPatchCanonicalRotate(
        attributes=["xyz", "normals", "opacity"],
        max_neighbors=64,
        use_r1_min_val=True,
        target_direction=[0., 1., 0.]  # Align to y-axis
    )
    
    # Create dataset with transform
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=canonical_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform will canonically rotate patches based on center of mass")


# Example 4: Compose multiple transforms
class ComposeTransforms:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        """
        Args:
            transforms: List of transform objects to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, batch):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            batch = transform(batch)
        return batch


def example_composed_transforms():
    """Combine multiple transforms."""
    print("\n" + "="*80)
    print("Example 4: Composing Multiple Transforms")
    print("="*80)
    
    # Create composed transform: dropout -> rotation
    composed_transform = ComposeTransforms([
        GaussianPatchDropout(
            max_dropout_ratio=0.2,
            attributes=["xyz", "normals", "opacity"],
            max_neighbors=64,
            use_r1_min_val=True,
            mask_constant=-10
        ),
        GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"],
            max_neighbors=64,
            use_r1_min_val=True
        )
    ])
    
    # Create dataset with composed transform
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=composed_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform pipeline: dropout -> random rotation")


# Example 5: Using transforms with DataLoader
def example_with_dataloader():
    """Use transforms with PyTorch DataLoader for training."""
    print("\n" + "="*80)
    print("Example 5: Using Transforms with DataLoader")
    print("="*80)
    
    # Create transform
    train_transform = ComposeTransforms([
        GaussianPatchDropout(
            max_dropout_ratio=0.3,
            attributes=["xyz", "normals", "opacity"],
            max_neighbors=64,
            use_r1_min_val=True,
            mask_constant=-10
        ),
        GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"],
            max_neighbors=64,
            use_r1_min_val=True
        )
    ])
    
    # Create training dataset with augmentation
    train_dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=train_transform  # Augmentation for training
    )
    
    # Create validation dataset WITHOUT augmentation
    val_dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=None  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} (with augmentation)")
    print(f"Val dataset: {len(val_dataset)} (no augmentation)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Training loop example
    print("\nExample training loop:")
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}")
        
        # Training with augmentation
        for batch_idx, (features, targets) in enumerate(train_loader):
            if batch_idx >= 2:  # Just show first 2 batches
                break
            print(f"  Train batch {batch_idx}: features={features.shape}, targets={targets.shape}")
            # Your training code here: forward pass, loss, backward, etc.
        
        # Validation without augmentation
        for batch_idx, (features, targets) in enumerate(val_loader):
            if batch_idx >= 2:
                break
            print(f"  Val batch {batch_idx}: features={features.shape}, targets={targets.shape}")
            # Your validation code here


# Example 6: Custom transform
class CustomNormalization:
    """Example custom transform for additional preprocessing."""
    
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, batch):
        """Normalize features (except last 2 values which are r1_min_val and p_u)."""
        # Normalize only the features, not the target values
        features = batch[:, :-2]
        features = (features - self.mean) / self.std
        batch[:, :-2] = features
        return batch


def example_custom_transform():
    """Use custom transform."""
    print("\n" + "="*80)
    print("Example 6: Custom Transform")
    print("="*80)
    
    # Compose custom transform with built-in transforms
    custom_transform = ComposeTransforms([
        CustomNormalization(mean=0.0, std=1.0),
        GaussianPatchRotate(
            attributes=["xyz", "normals", "opacity"],
            max_neighbors=64,
            use_r1_min_val=True
        )
    ])
    
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=custom_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform pipeline: custom normalization -> random rotation")


# Example 7: Conditional transforms (random apply)
class RandomApply:
    """Randomly apply a transform with given probability."""
    
    def __init__(self, transform, p=0.5):
        """
        Args:
            transform: Transform to apply
            p: Probability of applying the transform (0 to 1)
        """
        self.transform = transform
        self.p = p
    
    def __call__(self, batch):
        """Apply transform with probability p."""
        if torch.rand(1).item() < self.p:
            return self.transform(batch)
        return batch


def example_random_apply():
    """Apply transforms with probability."""
    print("\n" + "="*80)
    print("Example 7: Random Apply Transforms")
    print("="*80)
    
    # Apply dropout 50% of the time, rotation 70% of the time
    probabilistic_transform = ComposeTransforms([
        RandomApply(
            GaussianPatchDropout(
                max_dropout_ratio=0.3,
                attributes=["xyz", "normals", "opacity"],
                max_neighbors=64,
                use_r1_min_val=True,
                mask_constant=-10
            ),
            p=0.5  # Apply 50% of the time
        ),
        RandomApply(
            GaussianPatchRotate(
                attributes=["xyz", "normals", "opacity"],
                max_neighbors=64,
                use_r1_min_val=True
            ),
            p=0.7  # Apply 70% of the time
        )
    ])
    
    dataset = GaussianPatchDataset(
        config_path="path/to/your/config.yaml",
        attributes=["xyz", "normals", "opacity"],
        ring=2,
        use_r1_min=True,
        transform=probabilistic_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print("Transform pipeline: dropout (50%) -> rotation (70%)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GaussianPatchDataset Transform Examples")
    print("="*80)
    print("\nThese examples show how to use transforms for data augmentation.")
    print("Replace 'path/to/your/config.yaml' with your actual config file path.")
    print("\nAvailable transforms:")
    print("  - GaussianPatchDropout: Drop far neighbors based on geodesic distance")
    print("  - GaussianPatchRotate: Random rotation around y-axis")
    print("  - GaussianPatchCanonicalRotate: Canonical rotation based on center of mass")
    print("\nYou can:")
    print("  - Use single transforms")
    print("  - Compose multiple transforms")
    print("  - Create custom transforms")
    print("  - Apply transforms probabilistically")
    
    # Uncomment to run examples (requires valid config file):
    # example_dropout()
    # example_rotation()
    # example_canonical_rotation()
    # example_composed_transforms()
    # example_with_dataloader()
    # example_custom_transform()
    # example_random_apply()
