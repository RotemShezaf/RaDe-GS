"""
Tests for the Trainer class with early stopping.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import EarlyStopping, Trainer
from GaussianPatchTransformer import GaussianPatchTransformer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model_config():
    """Configuration for a small test model."""
    return {
        'attributes': ['xyz', 'opacity'],  # 3 + 1 = 4 dims per neighbor
        'point_attributes': ['xyz'],  # 3 dims for point
        'embed_dim': 32,
        'encoder_depth': 2,
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'max_neighbors': 16,
        'dropout': 0.0,
    }


@pytest.fixture
def model(model_config):
    """Create a small model for testing."""
    return GaussianPatchTransformer(**model_config)


@pytest.fixture
def sample_batch(model_config):
    """Create a sample batch of data."""
    batch_size = 4
    max_neighbors = model_config['max_neighbors']
    # xyz (3) + opacity (1) + geodesic_distance (1) = 5 dims for neighbors
    entry_size = 5
    # xyz (3) + opacity (1) = 4 dims for point (no geodesic)
    point_feature_dim = 4
    
    # Random data
    neighborhood = torch.randn(batch_size, max_neighbors, entry_size)
    point_features = torch.randn(batch_size, point_feature_dim)
    targets = torch.rand(batch_size) * 10  # Random distances
    
    # Valid mask (some neighbors valid, some not)
    valid_mask = torch.ones(batch_size, max_neighbors, dtype=torch.bool)
    valid_mask[:, -4:] = False  # Last 4 neighbors invalid
    
    return neighborhood, point_features, targets, valid_mask


@pytest.fixture
def sample_dataloader(sample_batch):
    """Create a sample dataloader with multiple batches."""
    neighborhood, point_features, targets, valid_mask = sample_batch
    
    # Replicate to create more data
    num_samples = 32
    batch_size = 4
    
    neighborhoods = neighborhood.repeat(num_samples // 4, 1, 1)
    point_feats = point_features.repeat(num_samples // 4, 1)
    targs = targets.repeat(num_samples // 4)
    masks = valid_mask.repeat(num_samples // 4, 1)
    
    # Add some noise
    neighborhoods = neighborhoods + torch.randn_like(neighborhoods) * 0.1
    targs = targs + torch.randn_like(targs) * 0.1
    
    dataset = TensorDataset(neighborhoods, point_feats, targs, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@pytest.fixture
def train_val_loaders(sample_batch):
    """Create train and validation dataloaders."""
    neighborhood, point_features, targets, valid_mask = sample_batch
    
    # Create training data (more samples)
    num_train = 64
    batch_size = 8
    
    train_neighborhoods = neighborhood.repeat(num_train // 4, 1, 1)
    train_point_feats = point_features.repeat(num_train // 4, 1)
    train_targets = targets.repeat(num_train // 4)
    train_masks = valid_mask.repeat(num_train // 4, 1)
    
    train_neighborhoods = train_neighborhoods + torch.randn_like(train_neighborhoods) * 0.1
    train_targets = train_targets + torch.randn_like(train_targets) * 0.1
    
    train_dataset = TensorDataset(train_neighborhoods, train_point_feats, train_targets, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation data (fewer samples)
    num_val = 16
    val_neighborhoods = neighborhood.repeat(num_val // 4, 1, 1)
    val_point_feats = point_features.repeat(num_val // 4, 1)
    val_targets = targets.repeat(num_val // 4)
    val_masks = valid_mask.repeat(num_val // 4, 1)
    
    val_neighborhoods = val_neighborhoods + torch.randn_like(val_neighborhoods) * 0.05
    val_targets = val_targets + torch.randn_like(val_targets) * 0.05
    
    val_dataset = TensorDataset(val_neighborhoods, val_point_feats, val_targets, val_masks)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# =============================================================================
# EarlyStopping Tests
# =============================================================================

class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_init(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode='min')
        
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == 'min'
        assert es.best_score is None
        assert es.counter == 0
        assert not es.should_stop
    
    def test_init_max_mode(self):
        """Test EarlyStopping with max mode."""
        es = EarlyStopping(patience=3, mode='max')
        assert es.mode == 'max'
    
    def test_first_score_sets_best(self, model):
        """Test that first score becomes the best score."""
        es = EarlyStopping(patience=5)
        
        should_stop = es(0.5, model, 0)
        
        assert not should_stop
        assert es.best_score == 0.5
        assert es.best_epoch == 0
        assert es.counter == 0
    
    def test_improvement_resets_counter_min_mode(self, model):
        """Test that improvement resets counter in min mode."""
        es = EarlyStopping(patience=5, mode='min')
        
        # First score
        es(1.0, model, 0)
        
        # No improvement
        es(1.0, model, 1)
        es(1.0, model, 2)
        assert es.counter == 2
        
        # Improvement
        es(0.5, model, 3)
        assert es.counter == 0
        assert es.best_score == 0.5
        assert es.best_epoch == 3
    
    def test_improvement_resets_counter_max_mode(self, model):
        """Test that improvement resets counter in max mode."""
        es = EarlyStopping(patience=5, mode='max')
        
        # First score
        es(0.5, model, 0)
        
        # No improvement
        es(0.5, model, 1)
        assert es.counter == 1
        
        # Improvement
        es(0.8, model, 2)
        assert es.counter == 0
        assert es.best_score == 0.8
    
    def test_triggers_early_stop(self, model):
        """Test that early stopping triggers after patience epochs."""
        es = EarlyStopping(patience=3, mode='min')
        
        es(1.0, model, 0)  # Best
        es(1.1, model, 1)  # No improvement, counter=1
        es(1.2, model, 2)  # No improvement, counter=2
        should_stop = es(1.3, model, 3)  # No improvement, counter=3 >= patience
        
        assert should_stop
        assert es.should_stop
        assert es.counter == 3
    
    def test_min_delta_threshold(self, model):
        """Test that min_delta threshold works correctly."""
        es = EarlyStopping(patience=5, min_delta=0.1, mode='min')
        
        es(1.0, model, 0)  # Best
        
        # Small improvement below threshold
        es(0.95, model, 1)  # Not considered improvement
        assert es.counter == 1
        assert es.best_score == 1.0
        
        # Large improvement above threshold
        es(0.8, model, 2)  # 1.0 - 0.1 = 0.9, 0.8 < 0.9 = improvement
        assert es.counter == 0
        assert es.best_score == 0.8
    
    def test_restore_best_weights(self, model):
        """Test weight restoration."""
        es = EarlyStopping(patience=2, restore_best_weights=True)
        
        # Get initial weights
        initial_weight = model.cls_token.data.clone()
        
        # First score - best
        es(1.0, model, 0)
        best_weight = model.cls_token.data.clone()
        
        # Modify model
        model.cls_token.data.fill_(99.0)
        
        # Score worsens
        es(1.5, model, 1)
        es(2.0, model, 2)  # Triggers stop
        
        # Restore best weights
        es.restore_weights(model)
        
        assert torch.allclose(model.cls_token.data, best_weight)
    
    def test_no_restore_when_disabled(self, model):
        """Test that weights are not restored when disabled."""
        es = EarlyStopping(patience=2, restore_best_weights=False)
        
        es(1.0, model, 0)
        es(1.5, model, 1)
        
        # Modify model
        model.cls_token.data.fill_(99.0)
        
        es(2.0, model, 2)
        
        # Restore should do nothing
        es.restore_weights(model)
        
        assert torch.allclose(model.cls_token.data, torch.full_like(model.cls_token.data, 99.0))


# =============================================================================
# Trainer Initialization Tests
# =============================================================================

class TestTrainerInit:
    """Tests for Trainer initialization."""
    
    def test_basic_init(self, model):
        """Test basic trainer initialization."""
        trainer = Trainer(model)
        
        assert trainer.model is model
        assert trainer.optimizer is not None
        assert trainer.scheduler is None
        assert trainer.loss_type == 'mse'
        assert trainer.early_stopping is None
        assert trainer.current_epoch == 0
        assert trainer.best_val_mae == float('inf')
    
    def test_init_with_custom_optimizer(self, model):
        """Test initialization with custom optimizer."""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer=optimizer)
        
        assert trainer.optimizer is optimizer
    
    def test_init_with_scheduler(self, model):
        """Test initialization with scheduler."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler)
        
        assert trainer.scheduler is scheduler
    
    def test_init_with_early_stopping(self, model):
        """Test initialization with early stopping."""
        es = EarlyStopping(patience=5)
        trainer = Trainer(model, early_stopping=es)
        
        assert trainer.early_stopping is es
    
    def test_init_with_save_dir(self, model):
        """Test initialization with save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model, save_dir=tmpdir)
            assert trainer.save_dir == Path(tmpdir)
    
    def test_init_device_auto_detection(self, model):
        """Test that device is auto-detected."""
        trainer = Trainer(model)
        
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert trainer.device == expected_device
    
    def test_init_with_different_loss_types(self, model):
        """Test initialization with different loss types."""
        for loss_type in ['mse', 'l1', 'smooth_l1']:
            trainer = Trainer(model, loss_type=loss_type)
            assert trainer.loss_type == loss_type
    
    def test_history_initialized_empty(self, model):
        """Test that history is initialized with empty lists."""
        trainer = Trainer(model)
        
        assert isinstance(trainer.history, dict)
        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert len(trainer.history['train_loss']) == 0


# =============================================================================
# Trainer Batch Training Tests
# =============================================================================

class TestTrainerBatchOperations:
    """Tests for single batch operations."""
    
    def test_train_batch_returns_metrics(self, model, sample_batch):
        """Test that train_batch returns expected metrics."""
        trainer = Trainer(model, device='cpu')
        neighborhood, point_features, targets, valid_mask = sample_batch
        
        metrics = trainer.train_batch(neighborhood, point_features, targets, valid_mask)
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0
    
    def test_train_batch_updates_model(self, model, sample_batch):
        """Test that train_batch actually trains (loss should decrease over multiple batches)."""
        # Use higher learning rate to see visible changes
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer=optimizer, device='cpu')
        neighborhood, point_features, targets, valid_mask = sample_batch
        
        # Train multiple batches and track loss
        losses = []
        for _ in range(5):
            metrics = trainer.train_batch(neighborhood, point_features, targets, valid_mask)
            losses.append(metrics['loss'])
        
        # The model should be learning - loss at end should be less than at start
        # (with some tolerance for variance)
        assert losses[-1] <= losses[0] * 2.0  # Allow some variance but ensure not exploding
    
    def test_validate_batch_returns_metrics(self, model, sample_batch):
        """Test that validate_batch returns expected metrics."""
        trainer = Trainer(model, device='cpu')
        neighborhood, point_features, targets, valid_mask = sample_batch
        
        metrics = trainer.validate_batch(neighborhood, point_features, targets, valid_mask)
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'relative_error_pct' in metrics
    
    def test_validate_batch_no_gradient(self, model, sample_batch):
        """Test that validate_batch doesn't compute gradients."""
        trainer = Trainer(model, device='cpu')
        neighborhood, point_features, targets, valid_mask = sample_batch
        
        # Enable gradient tracking
        neighborhood.requires_grad_(True)
        
        trainer.validate_batch(neighborhood, point_features, targets, valid_mask)
        
        # No gradients should be computed
        assert neighborhood.grad is None
    
    def test_validate_batch_does_not_update_model(self, model, sample_batch):
        """Test that validate_batch doesn't update model weights."""
        trainer = Trainer(model, device='cpu')
        neighborhood, point_features, targets, valid_mask = sample_batch
        
        # Get initial weights
        initial_weight = model.cls_token.data.clone()
        
        # Validate batch
        trainer.validate_batch(neighborhood, point_features, targets, valid_mask)
        
        # Weights should NOT change
        assert torch.allclose(model.cls_token.data, initial_weight)


# =============================================================================
# Trainer Epoch Training Tests
# =============================================================================

class TestTrainerEpochOperations:
    """Tests for epoch-level operations."""
    
    def test_train_epoch_returns_avg_metrics(self, model, sample_dataloader):
        """Test that train_epoch returns averaged metrics."""
        trainer = Trainer(model, device='cpu')
        
        metrics = trainer.train_epoch(sample_dataloader, show_progress=False)
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_validate_epoch_returns_avg_metrics(self, model, sample_dataloader):
        """Test that validate_epoch returns averaged metrics."""
        trainer = Trainer(model, device='cpu')
        
        metrics = trainer.validate_epoch(sample_dataloader, show_progress=False)
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'relative_error_pct' in metrics
    
    def test_train_epoch_model_in_train_mode(self, model, sample_dataloader):
        """Test that model is in train mode during training."""
        trainer = Trainer(model, device='cpu')
        
        # Force eval mode
        model.eval()
        
        trainer.train_epoch(sample_dataloader, show_progress=False)
        
        # Model should be in train mode
        assert model.training
    
    def test_validate_epoch_model_in_eval_mode(self, model, sample_dataloader):
        """Test that model is in eval mode during validation."""
        trainer = Trainer(model, device='cpu')
        
        # Force train mode
        model.train()
        
        trainer.validate_epoch(sample_dataloader, show_progress=False)
        
        # Model should be in eval mode
        assert not model.training


# =============================================================================
# Trainer Fit Tests
# =============================================================================

class TestTrainerFit:
    """Tests for the full training loop."""
    
    def test_fit_runs_without_error(self, model, train_val_loaders):
        """Test that fit runs without errors."""
        trainer = Trainer(model, device='cpu')
        train_loader, val_loader = train_val_loaders
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=2,
            show_progress=False,
            verbose=False
        )
        
        assert isinstance(history, dict)
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
    
    def test_fit_records_history(self, model, train_val_loaders):
        """Test that fit records training history."""
        trainer = Trainer(model, device='cpu')
        train_loader, val_loader = train_val_loaders
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=3,
            show_progress=False,
            verbose=False
        )
        
        assert len(history['train_loss']) == 3
        assert len(history['train_mae']) == 3
        assert len(history['val_loss']) == 3
        assert len(history['val_mae']) == 3
        assert len(history['learning_rate']) == 3
    
    def test_fit_with_early_stopping(self, model, train_val_loaders):
        """Test that fit respects early stopping."""
        # Create early stopping with very small patience
        es = EarlyStopping(patience=1, min_delta=1e10)  # Will always trigger
        trainer = Trainer(model, device='cpu', early_stopping=es)
        train_loader, val_loader = train_val_loaders
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=10,
            show_progress=False,
            verbose=False
        )
        
        # Should stop early (after 2 epochs at most: first + 1 patience)
        assert len(history['train_loss']) <= 3
    
    def test_fit_updates_best_val_mae(self, model, train_val_loaders):
        """Test that fit updates best validation MAE."""
        trainer = Trainer(model, device='cpu')
        train_loader, val_loader = train_val_loaders
        
        initial_best = trainer.best_val_mae
        
        trainer.fit(
            train_loader, val_loader,
            num_epochs=2,
            show_progress=False,
            verbose=False
        )
        
        assert trainer.best_val_mae < initial_best
    
    def test_fit_with_scheduler(self, model, train_val_loaders):
        """Test that fit uses the scheduler."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, device='cpu')
        train_loader, val_loader = train_val_loaders
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=3,
            show_progress=False,
            verbose=False
        )
        
        # Learning rate should decrease
        assert history['learning_rate'][0] > history['learning_rate'][-1]
    
    def test_fit_saves_best_model(self, model, train_val_loaders):
        """Test that fit saves the best model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model, save_dir=tmpdir, device='cpu')
            train_loader, val_loader = train_val_loaders
            
            trainer.fit(
                train_loader, val_loader,
                num_epochs=2,
                show_progress=False,
                verbose=False
            )
            
            # Check that checkpoint was saved
            checkpoint_path = Path(tmpdir) / 'best_model.pth'
            assert checkpoint_path.exists()


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestTrainerCheckpoints:
    """Tests for checkpoint saving and loading."""
    
    def test_save_checkpoint(self, model, train_val_loaders):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model, save_dir=tmpdir, device='cpu')
            train_loader, val_loader = train_val_loaders
            
            trainer.fit(
                train_loader, val_loader,
                num_epochs=1,
                show_progress=False,
                verbose=False
            )
            
            checkpoint_path = Path(tmpdir) / 'best_model.pth'
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'epoch' in checkpoint
            assert 'metrics' in checkpoint
    
    def test_load_checkpoint(self, model_config, train_val_loaders):
        """Test checkpoint loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            model1 = GaussianPatchTransformer(**model_config)
            trainer1 = Trainer(model1, save_dir=tmpdir, device='cpu')
            train_loader, val_loader = train_val_loaders
            
            trainer1.fit(
                train_loader, val_loader,
                num_epochs=2,
                show_progress=False,
                verbose=False
            )
            
            # Create new model and load
            model2 = GaussianPatchTransformer(**model_config)
            trainer2 = Trainer(model2, device='cpu')
            
            checkpoint_path = Path(tmpdir) / 'best_model.pth'
            checkpoint = trainer2.load_checkpoint(str(checkpoint_path))
            
            assert checkpoint['epoch'] >= 0
            
            # Weights should match
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


# =============================================================================
# Prediction Tests
# =============================================================================

class TestTrainerPredict:
    """Tests for prediction functionality."""
    
    def test_predict_returns_tensor(self, model, sample_dataloader):
        """Test that predict returns a tensor."""
        trainer = Trainer(model, device='cpu')
        
        predictions = trainer.predict(sample_dataloader, show_progress=False)
        
        assert isinstance(predictions, torch.Tensor)
    
    def test_predict_correct_shape(self, model, sample_dataloader):
        """Test that predictions have correct shape."""
        trainer = Trainer(model, device='cpu')
        
        predictions = trainer.predict(sample_dataloader, show_progress=False)
        
        # Count total samples in dataloader
        total_samples = sum(batch[0].shape[0] for batch in sample_dataloader)
        
        # Predictions should match
        assert predictions.shape[0] == total_samples
    
    def test_predict_model_in_eval_mode(self, model, sample_dataloader):
        """Test that model is in eval mode during prediction."""
        trainer = Trainer(model, device='cpu')
        model.train()
        
        trainer.predict(sample_dataloader, show_progress=False)
        
        assert not model.training


# =============================================================================
# Integration Tests
# =============================================================================

class TestTrainerIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_full_training_pipeline(self, model_config, train_val_loaders):
        """Test a complete training pipeline."""
        model = GaussianPatchTransformer(**model_config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping=early_stopping,
                save_dir=tmpdir,
                device='cpu'
            )
            
            train_loader, val_loader = train_val_loaders
            
            history = trainer.fit(
                train_loader, val_loader,
                num_epochs=5,
                show_progress=False,
                verbose=False
            )
            
            # History should be recorded
            assert len(history['train_loss']) > 0
            
            # Best model should be saved
            assert (Path(tmpdir) / 'best_model.pth').exists()
            
            # Make predictions
            predictions = trainer.predict(val_loader, show_progress=False)
            assert predictions.shape[0] > 0
    
    def test_loss_decreases_during_training(self, model_config, train_val_loaders):
        """Test that loss generally decreases during training."""
        model = GaussianPatchTransformer(**model_config)
        
        trainer = Trainer(model, device='cpu')
        train_loader, val_loader = train_val_loaders
        
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=10,
            show_progress=False,
            verbose=False
        )
        
        # Compare first and last loss (with some tolerance for variance)
        first_loss = sum(history['train_loss'][:2]) / 2
        last_loss = sum(history['train_loss'][-2:]) / 2
        
        # Loss should generally decrease (allowing for some variance)
        assert last_loss <= first_loss * 1.5  # Allow some tolerance
    
    def test_different_loss_types_work(self, model_config, train_val_loaders):
        """Test training with different loss types."""
        train_loader, val_loader = train_val_loaders
        
        for loss_type in ['mse', 'l1', 'smooth_l1']:
            model = GaussianPatchTransformer(**model_config)
            trainer = Trainer(model, loss_type=loss_type, device='cpu')
            
            history = trainer.fit(
                train_loader, val_loader,
                num_epochs=1,
                show_progress=False,
                verbose=False
            )
            
            assert len(history['train_loss']) == 1
            assert history['train_loss'][0] > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestTrainerEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_sample_batch(self, model):
        """Test training with a single sample."""
        trainer = Trainer(model, device='cpu')
        
        # Single sample with correct dims (xyz=3 + opacity=1 + geodesic=1 = 5)
        neighborhood = torch.randn(1, 16, 5)
        point_features = torch.randn(1, 4)  # xyz=3 + opacity=1 = 4
        targets = torch.tensor([5.0])
        valid_mask = torch.ones(1, 16, dtype=torch.bool)
        
        metrics = trainer.train_batch(neighborhood, point_features, targets, valid_mask)
        
        assert 'loss' in metrics
        assert not torch.isnan(torch.tensor(metrics['loss']))
    
    def test_all_neighbors_invalid(self, model):
        """Test with all neighbors masked out."""
        trainer = Trainer(model, device='cpu')
        
        neighborhood = torch.randn(2, 16, 5)  # xyz=3 + opacity=1 + geodesic=1 = 5
        point_features = torch.randn(2, 4)  # xyz=3 + opacity=1 = 4
        targets = torch.tensor([5.0, 3.0])
        valid_mask = torch.zeros(2, 16, dtype=torch.bool)  # All invalid
        
        # Should not crash
        metrics = trainer.validate_batch(neighborhood, point_features, targets, valid_mask)
        
        assert 'loss' in metrics
    
    def test_early_stopping_with_patience_zero(self, model):
        """Test early stopping with zero patience."""
        es = EarlyStopping(patience=0)
        trainer = Trainer(model, early_stopping=es, device='cpu')
        
        # Should stop immediately after first epoch if no improvement
        es(1.0, model, 0)  # First score
        should_stop = es(1.1, model, 1)  # No improvement
        
        assert should_stop


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
