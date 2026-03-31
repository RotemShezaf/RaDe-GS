"""
Trainer class for GaussianPatchTransformer with early stopping and batch training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List
import logging


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = False,
        logger: Optional[Callable[[str], None]] = None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        # restore_best_weights accepted for API compat but ignored
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            model: Model (unused, kept for API compat)
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        improved = False
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Trainer class for GaussianPatchTransformer with early stopping support.
    
    Args:
        model: GaussianPatchTransformer model
        optimizer: Optimizer (if None, AdamW is used)
        scheduler: Learning rate scheduler (optional)
        device: Device to use (if None, auto-detected)
        loss_type: Type of loss ('mse', 'l1', 'smooth_l1')
        early_stopping: EarlyStopping instance (optional)
        save_dir: Directory to save checkpoints
        use_wandb: Whether to log to wandb
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        loss_type: str = 'mse',
        huber_delta: float = 1.0,
        blended_mse_lambda: float = 0.5,
        gradient_clip_norm: float = 0.0,
        early_stopping: Optional[EarlyStopping] = None,
        save_dir: Optional[str] = None,
        use_wandb: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.blended_mse_lambda = blended_mse_lambda
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping = early_stopping
        self.save_dir = Path(save_dir) if save_dir else None
        self.use_wandb = use_wandb
        self.logger = logger
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_mae': [],
            'train_mse': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'learning_rate': []
        }
        
        self.current_epoch = 0
        self.best_val_mae = float('inf')
        self.best_val_metrics: Optional[Dict[str, float]] = None
    
    def train_batch(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on a single batch.
        
        Args:
            neighborhood: Neighbor features (batch, max_neighbors, entry_size)
            point_features: Point features (batch, point_feature_dim)
            targets: Target geodesic distances (batch,)
            valid_mask: Valid neighbor mask (batch, max_neighbors)
            
        Returns:
            Dictionary with loss and metrics
        """
        
        
        # Move to device
        neighborhood = neighborhood.to(self.device)
        point_features = point_features.to(self.device)
        targets = targets.to(self.device)
        valid_mask = valid_mask.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(neighborhood, point_features, valid_mask)
        
        # Compute loss
        loss = self.model.get_loss(predictions, targets, loss_type=self.loss_type, huber_delta=self.huber_delta, blended_mse_lambda=self.blended_mse_lambda)
        
        # Backward pass
        loss.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = self.model.get_metrics(predictions, targets)
        
        return {
            'loss': loss.item(),
            'mae': metrics['mae'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'relative_error_pct': metrics['relative_error_pct']
        }
    
    def validate_batch(
        self,
        neighborhood: torch.Tensor,
        point_features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate on a single batch.
        
        Args:
            neighborhood: Neighbor features
            point_features: Point features
            targets: Target geodesic distances
            valid_mask: Valid neighbor mask
            
        Returns:
            Dictionary with loss and metrics
        """
        
        # Move to device
        neighborhood = neighborhood.to(self.device)
        point_features = point_features.to(self.device)
        targets = targets.to(self.device)
        valid_mask = valid_mask.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(neighborhood, point_features, valid_mask)
            loss = self.model.get_loss(predictions, targets, loss_type=self.loss_type, huber_delta=self.huber_delta, blended_mse_lambda=self.blended_mse_lambda)
            metrics = self.model.get_metrics(predictions, targets)
        
        return {
            'loss': loss.item(),
            'mae': metrics['mae'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'relative_error_pct': metrics['relative_error_pct']
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of average metrics
        """
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total_rmse = 0.0
        total_rel_error = 0.0
        num_batches = 0
        
        iterator = tqdm(train_loader, desc="Training") if show_progress else train_loader
        self.model.train()
        for neighborhood, point_features, targets, valid_mask in iterator:
            batch_metrics = self.train_batch(
                neighborhood, point_features, targets, valid_mask
            )
            
            total_loss += batch_metrics['loss']
            total_mae += batch_metrics['mae']
            total_mse += batch_metrics['mse']
            total_rmse += batch_metrics['rmse']
            total_rel_error += batch_metrics['relative_error_pct']
            num_batches += 1
            
            if show_progress:
                iterator.set_postfix({
                    'loss': f"{batch_metrics['loss']:.4f}",
                    'mae': f"{batch_metrics['mae']:.4f}",
                    'mse': f"{batch_metrics['mse']:.4f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'mse': total_mse / num_batches,
            'rmse': total_rmse / num_batches,
            'relative_error_pct': total_rel_error / num_batches
        }
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of average metrics
        """
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total_rmse = 0.0
        total_rel_error = 0.0
        num_batches = 0

        self.model.eval()
        
        iterator = tqdm(val_loader, desc="Validation") if show_progress else val_loader
        self.model.eval()
        for neighborhood, point_features, targets, valid_mask in iterator:
            batch_metrics = self.validate_batch(
                neighborhood, point_features, targets, valid_mask
            )
            
            total_loss += batch_metrics['loss']
            total_mae += batch_metrics['mae']
            total_mse += batch_metrics['mse']
            total_rmse += batch_metrics['rmse']
            total_rel_error += batch_metrics['relative_error_pct']
            num_batches += 1
            
            if show_progress:
                iterator.set_postfix({
                    'loss': f"{batch_metrics['loss']:.4f}",
                    'mae': f"{batch_metrics['mae']:.4f}",
                    'mse': f"{batch_metrics['mse']:.4f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'mse': total_mse / num_batches,
            'rmse': total_rmse / num_batches,
            'relative_error_pct': total_rel_error / num_batches
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        show_progress: bool = True,
        verbose: bool = True,
        start_epoch: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            show_progress: Whether to show progress bars
            verbose: Whether to print epoch summaries
            start_epoch: First epoch index (used when resuming from a checkpoint)

        Returns:
            Training history dictionary
        """
        if self.use_wandb:
            try:
                import wandb
                wandb.watch(self.model, log='all')
            except ImportError:
                self.use_wandb = False

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            if verbose:
                if self.logger:
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"[Training] Epoch {epoch + 1}/{num_epochs}")
                    self.logger.info(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"[Training] Epoch {epoch + 1}/{num_epochs}")
                    print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, show_progress)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, show_progress)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['learning_rate'].append(current_lr)
            
            if verbose:
                if self.logger:
                    self.logger.info(f"\n[Training] EPOCH {epoch + 1} Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, MSE: {train_metrics['mse']:.4f}, Rel Error: {train_metrics['relative_error_pct']:.2f}%")
                    self.logger.info(f" Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, MSE: {val_metrics['mse']:.4f}, Rel Error: {val_metrics['relative_error_pct']:.2f}%")
                else:
                    print(f"\n[Training] EPOCH {epoch + 1} Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, MSE: {train_metrics['mse']:.4f}, Rel Error: {train_metrics['relative_error_pct']:.2f}%")
                    print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, MSE: {val_metrics['mse']:.4f}, Rel Error: {val_metrics['relative_error_pct']:.2f}%")
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/mae': train_metrics['mae'],
                    'train/mse': train_metrics['mse'],
                    'train/relative_error_pct': train_metrics['relative_error_pct'],
                    'val/loss': val_metrics['loss'],
                    'val/mae': val_metrics['mae'],
                    'val/mse': val_metrics['mse'],
                    'val/relative_error_pct': val_metrics['relative_error_pct'],
                    'learning_rate': current_lr
                })
            
            # Save best model (tracked by validation MAE)
            if val_metrics['mae'] < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                self.best_val_metrics = dict(val_metrics)
                if self.save_dir:
                    self._save_checkpoint('best_model.pth', epoch, val_metrics)
                    if verbose:
                        if self.logger:
                            self.logger.info(f"✓ Saved best model (val MAE: {self.best_val_mae:.6f}, val MSE: {val_metrics['mse']:.6f})")
                        else:
                            print(f"✓ Saved best model (val MAE: {self.best_val_mae:.6f}, val MSE: {val_metrics['mse']:.6f})")
            
            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(val_metrics['mae'], self.model, epoch)
                if should_stop:
                    if verbose:
                        if self.logger:
                            self.logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                            self.logger.info(f"  Best epoch: {self.early_stopping.best_epoch + 1}")
                            self.logger.info(f"  Best val MAE: {self.early_stopping.best_score:.4f}")
                        else:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                            print(f"  Best epoch: {self.early_stopping.best_epoch + 1}")
                            print(f"  Best val MAE: {self.early_stopping.best_score:.4f}")
                    break
        
        if verbose:
            if self.logger:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Training complete! Best validation MAE: {self.best_val_mae:.6f}")
                if self.best_val_metrics:
                    self.logger.info(f"  Best val MSE: {self.best_val_metrics['mse']:.6f}, RMSE: {self.best_val_metrics['rmse']:.6f}, Rel Error: {self.best_val_metrics.get('relative_error_pct', 0):.2f}%")
                self.logger.info(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"Training complete! Best validation MAE: {self.best_val_mae:.6f}")
                if self.best_val_metrics:
                    print(f"  Best val MSE: {self.best_val_metrics['mse']:.6f}, RMSE: {self.best_val_metrics['rmse']:.6f}, Rel Error: {self.best_val_metrics.get('relative_error_pct', 0):.2f}%")
                print(f"{'='*60}")
        
        return self.history
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a checkpoint."""
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / filename
            ckpt = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'history': self.history,
                'best_val_mae': self.best_val_mae,
                'best_val_metrics': self.best_val_metrics,
            }
            if self.scheduler is not None:
                ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(ckpt, save_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint and restore full trainer state for resuming training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', self.history)
        # Restore best metrics so we do not over-write a better checkpoint
        self.best_val_mae = checkpoint.get('best_val_mae', checkpoint.get('metrics', {}).get('mae', float('inf')))
        self.best_val_metrics = checkpoint.get('best_val_metrics', checkpoint.get('metrics', None))
        # Restore scheduler state if present
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    
    def predict(
        self,
        dataloader: DataLoader,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate predictions for a dataloader.
        
        Args:
            dataloader: DataLoader to predict on
            show_progress: Whether to show progress bar
            
        Returns:
            Tensor of predictions
        """
        self.model.eval()
        predictions = []
        
        iterator = tqdm(dataloader, desc="Predicting") if show_progress else dataloader
        
        with torch.no_grad():
            for neighborhood, point_features, targets, valid_mask in iterator:
                neighborhood = neighborhood.to(self.device)
                point_features = point_features.to(self.device)
                valid_mask = valid_mask.to(self.device)
                
                preds = self.model(neighborhood, point_features, valid_mask)
                predictions.append(preds.cpu())
        
        return torch.cat(predictions, dim=0)
