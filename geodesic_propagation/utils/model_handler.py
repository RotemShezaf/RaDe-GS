"""
Model handler for loading and using geodesic distance prediction models.

This module provides utilities for:
- Loading trained GaussianPatchTransformer models
- Running inference on Gaussian patches
- Managing model configuration
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ModelHandler:
    """
    Handler for loading and using geodesic distance prediction models.
    
    This class manages:
    - Model loading from checkpoints
    - Configuration management
    - Inference on Gaussian patches
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the model handler.
        
        Args:
            model_path: Path to model checkpoint (.pt/.pth file)
            config: Model configuration dict (if not loading from checkpoint)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model: Optional[nn.Module] = None
        self.config: Dict[str, Any] = config or {}
        self.model_path = Path(model_path) if model_path else None
        self._raw_yaml: Optional[Dict[str, Any]] = None  # full companion YAML
        
        if model_path is not None:
            self.load_model(model_path)
    
    def _load_yaml_config(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Search for a companion YAML config for a checkpoint.

        Search order:
        1. <checkpoint_dir>/<checkpoint_stem>.yaml  (e.g. best_model.yaml)
        2. <checkpoint_dir>/<checkpoint_dir_name>.yaml
        3. <project_root>/models/configs/<checkpoint_dir_name>.yaml

        If found, reads the ``model`` section (training YAML format) or the
        flat dict (evaluate YAML format) and returns a flat model-config dict.
        Returns None if no YAML is found.
        """
        import yaml as _yaml

        checkpoint_dir = model_path.parent
        dir_name = checkpoint_dir.name
        # Project root is three levels up: geodesic_propagation/utils/model_handler.py
        project_root = Path(__file__).resolve().parent.parent.parent

        candidates = [
            checkpoint_dir / f"{model_path.stem}.yaml",
            checkpoint_dir / f"{dir_name}.yaml",
            project_root / "models" / "configs" / f"{dir_name}.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                print(f"Loading model config from: {candidate}")
                with open(candidate, 'r') as f:
                    raw = _yaml.safe_load(f)
                self._raw_yaml = raw  # keep full YAML for dataset config etc.
                # Training YAMLs have a nested 'model' section; flatten it.
                cfg = raw.get('model', raw)
                return cfg

        return None

    def _infer_config_from_state_dict(self, sd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer model configuration from state dict key names and tensor shapes.

        This is a best-effort inference used when no config was saved in the
        checkpoint.  It covers the parameters that actually affect the model
        structure (and therefore the state dict layout).
        """
        # ---- embed_dim -------------------------------------------------------
        embed_dim: int = int(sd['cls_token'].shape[2])

        # ---- max_neighbors ---------------------------------------------------
        # pos_embed.pe shape: [1, max_len, embed_dim] where
        #   max_len = (1 + use_point_token + max_neighbors) + 1  (the +1 is a safety margin)
        # We need to know use_point_token first (determined by whether point_encoder weights exist).
        has_point_encoder = any(k.startswith('point_encoder.') for k in sd)
        # use_point_token is True when point_encoder was used during training.
        # (When point_attributes=[] the encoder is omitted entirely.)
        point_token_offset = 1 if has_point_encoder else 0
        # max_len = 1 (CLS) + point_token_offset + max_neighbors + 1 (safety)
        max_neighbors: int = int(sd['pos_embed.pe'].shape[1]) - 1 - 1 - point_token_offset

        # ---- encoder_depth ---------------------------------------------------
        block_indices = {int(k.split('.')[1]) for k in sd if k.startswith('encoder_blocks.')}
        encoder_depth: int = len(block_indices)

        # ---- encoder_type ----------------------------------------------------
        # "linear" encoders use  neighbor_encoder.encoder.*
        # "conv"   encoders use  neighbor_encoder.first_conv.* / second_conv.*
        if any(k.startswith('neighbor_encoder.encoder.') for k in sd):
            encoder_type = 'linear'
        else:
            encoder_type = 'conv'

        # ---- attributes (inferred from neighbor encoder input dim) -----------
        # neighbor input = attribute_dim + 1 (geodesic distance)
        if encoder_type == 'linear':
            nb_input_dim = int(sd['neighbor_encoder.encoder.0.weight'].shape[1])
        else:
            nb_input_dim = int(sd['neighbor_encoder.first_conv.0.weight'].shape[1])
        attribute_dim = nb_input_dim - 1  # subtract geodesic distance channel

        # Map attribute_dim to the canonical attribute list used during training.
        _DIM_TO_ATTRS: Dict[int, List[str]] = {
            3:  ["xyz"],
            4:  ["xyz", "opacity"],
            6:  ["xyz", "scale"],
            7:  ["xyz", "opacity", "scale"],
            11: ["xyz", "opacity", "scale", "rotation"],
            14: ["xyz", "opacity", "scale", "rotation", "sh"],
        }
        attributes = _DIM_TO_ATTRS.get(attribute_dim, ["xyz", "opacity", "scale", "rotation", "sh"])
        if attribute_dim not in _DIM_TO_ATTRS:
            print(f"Warning: cannot map attribute_dim={attribute_dim} to a known attribute list; "
                  f"falling back to all attributes.")

        # ---- pool ------------------------------------------------------------
        # prediction_head.0.weight shape: [hidden, in_features]
        # in_features = N * embed_dim where N is pool multiplier
        head_in = int(sd['prediction_head.0.weight'].shape[1])
        pool_multiplier = head_in // embed_dim
        pool = {1: 'mean', 2: 'max', 3: 'max_mean'}.get(pool_multiplier, 'max')

        # ---- num_heads -------------------------------------------------------
        # A safe heuristic: largest power-of-2 that divides embed_dim and
        # keeps head_dim >= 16.
        num_heads = 8
        while num_heads > 1 and embed_dim // num_heads < 16:
            num_heads //= 2

        config = {
            'attributes': attributes,
            'max_neighbors': max_neighbors,
            'embed_dim': embed_dim,
            'encoder_depth': encoder_depth,
            'num_heads': num_heads,
            'encoder_type': encoder_type,
            'pool': pool,
            'pos_encoding_type': 'index',
            'point_attributes': None if has_point_encoder else [],
        }
        print(f"Inferred config: {config}")
        return config

    def load_model(self, model_path: Union[str, Path]):
        """
        Load a model from a checkpoint file.
        
        Args:
            model_path: Path to the checkpoint file
        """
        from models.GaussianPatchTransformer import (
            GaussianPatchTransformer,
            create_gaussian_patch_transformer
        )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        elif 'model_config' in checkpoint:
            self.config = checkpoint['model_config']
        else:
            # No config saved — always infer structurally-critical parameters
            # (attributes, max_neighbors, embed_dim, encoder_type, pool, encoder_depth)
            # from the state dict, then overlay any values from a companion YAML.
            sd = checkpoint.get('model_state_dict', checkpoint)
            inferred = self._infer_config_from_state_dict(sd)
            yaml_cfg = self._load_yaml_config(model_path)
            if yaml_cfg is not None:
                # YAML provides non-structural training settings; structural
                # keys inferred from state-dict take precedence.
                structural_keys = {'attributes', 'max_neighbors', 'embed_dim',
                                   'encoder_depth', 'encoder_type', 'pool',
                                   'pos_encoding_type', 'num_heads',
                                   'point_attributes'}
                merged = {**yaml_cfg, **{k: v for k, v in inferred.items()
                                         if k in structural_keys}}
                self.config = merged
            else:
                self.config = inferred
        
        # Create model
        self.model = create_gaussian_patch_transformer(config=self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.model_path = model_path
        print(f"Model loaded successfully on {self.device}")
        self.print_model_info()
    
    def create_model(self, config: Dict[str, Any]):
        """
        Create a new model from configuration (for testing/debugging).
        
        Args:
            config: Model configuration dictionary
        """
        from models.GaussianPatchTransformer import create_gaussian_patch_transformer
        
        self.config = config
        self.model = create_gaussian_patch_transformer(config=config)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model created on {self.device}")
    
    def predict(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict geodesic distances for a batch of points.
        
        Args:
            neighborhood: Neighbor features (batch, max_neighbors, entry_size)
            point_features: Point features (batch, point_feature_dim)
            valid_mask: Boolean mask (batch, max_neighbors)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Predicted distances (batch, 1) or dict with predictions and embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Move to device
        neighborhood = neighborhood.to(self.device)
        point_features = point_features.to(self.device)
        valid_mask = valid_mask.to(self.device)
        
        with torch.no_grad():
            output = self.model(
                neighborhood,
                point_features,
                valid_mask,
                return_embeddings=return_embeddings
            )
        
        return output
    
    def predict_single(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> float:
        """
        Predict geodesic distance for a single point.
        
        Args:
            neighborhood: Neighbor features (max_neighbors, entry_size)
            point_features: Point features (point_feature_dim,)
            valid_mask: Boolean mask (max_neighbors,)
            
        Returns:
            Predicted geodesic distance (float)
        """
        # Add batch dimension
        neighborhood = neighborhood.unsqueeze(0)
        point_features = point_features.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
        
        prediction = self.predict(neighborhood, point_features, valid_mask)
        return prediction[0, 0].item()
    
    def get_attributes(self) -> List[str]:
        """Get the list of attributes the model uses."""
        return self.config.get('attributes', ["xyz"])
    
    def get_max_neighbors(self) -> int:
        """Get the maximum number of neighbors the model expects."""
        return self.config.get('max_neighbors', 32)
    
    def get_point_attributes(self) -> Optional[List[str]]:
        """Get the point attributes the model uses."""
        return self.config.get('point_attributes', None)
    
    def get_transforms_config(self) -> Optional[list]:
        """Return the training transforms config from the companion YAML.

        Looks in ``dataset.transforms`` of the full training YAML that was
        loaded alongside the checkpoint.  Returns ``None`` if no YAML was
        found or if it does not contain transforms.
        """
        if self._raw_yaml is None:
            return None
        dataset_cfg = self._raw_yaml.get('dataset', {})
        return dataset_cfg.get('transforms', None)

    def get_dataset_config_path(self) -> Optional[str]:
        """Return the ``dataset.dataset_config`` path from the companion YAML.

        This is the path to the dataset config YAML used during training.
        Returns ``None`` if no companion YAML was loaded or the key is absent.
        """
        if self._raw_yaml is None:
            return None
        dataset_cfg = self._raw_yaml.get('dataset', {})
        return dataset_cfg.get('dataset_config', None)

    def get_ring(self) -> Optional[int]:
        """Return the ``dataset.ring`` value from the companion YAML."""
        if self._raw_yaml is None:
            return None
        dataset_cfg = self._raw_yaml.get('dataset', {})
        return dataset_cfg.get('ring', None)
        return dataset_cfg.get('transforms', None)
    
    def print_model_info(self):
        """Print model information."""
        if self.model is not None:
            self.model.print_model_info()
    
    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        epoch: int = 0,
        optimizer_state: Optional[Dict] = None,
        extra_info: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save the checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            extra_info: Additional information to save
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> ModelHandler:
    """
    Convenience function to load a model handler from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to use
        
    Returns:
        Initialized ModelHandler with loaded model
    """
    return ModelHandler(model_path=checkpoint_path, device=device)


if __name__ == "__main__":
    """Test the model handler."""
    print("Testing ModelHandler...")
    
    # Create handler with a test model
    handler = ModelHandler()
    handler.create_model({
        'attributes': ["xyz", "opacity", "scale", "rotation", "sh"],
        'max_neighbors': 32,
        'embed_dim': 256,
        'encoder_depth': 4,
        'num_heads': 8,
    })
    
    # Test prediction
    from models.utils import get_attribute_dim
    
    attributes = handler.get_attributes()
    max_neighbors = handler.get_max_neighbors()
    
    neighbor_feature_dim = get_attribute_dim(attributes) + 1
    point_feature_dim = get_attribute_dim(attributes)
    
    batch_size = 4
    neighborhood = torch.randn(batch_size, max_neighbors, neighbor_feature_dim)
    point_features = torch.randn(batch_size, point_feature_dim)
    valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
    valid_mask[:, -5:] = False
    
    predictions = handler.predict(neighborhood, point_features, valid_mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:, 0].tolist()}")
    
    print("\n✓ ModelHandler tests passed!")
