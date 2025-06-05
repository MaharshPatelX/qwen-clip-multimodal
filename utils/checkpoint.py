import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import asdict


class ModelCheckpoint:
    """Helper class for managing individual model checkpoints."""
    
    def __init__(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state
        self.epoch = epoch
        self.step = step
        self.metrics = metrics or {}
        self.config = config or {}
    
    def save(self, checkpoint_path: Union[str, Path]):
        """Save checkpoint to disk."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "model_state_dict": self.model_state,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "config": self.config
        }
        
        if self.optimizer_state:
            checkpoint_data["optimizer_state_dict"] = self.optimizer_state
        
        if self.scheduler_state:
            checkpoint_data["scheduler_state_dict"] = self.scheduler_state
        
        torch.save(checkpoint_data, checkpoint_path)
    
    @classmethod
    def load(cls, checkpoint_path: Union[str, Path]) -> "ModelCheckpoint":
        """Load checkpoint from disk."""
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        return cls(
            model_state=checkpoint_data["model_state_dict"],
            optimizer_state=checkpoint_data.get("optimizer_state_dict"),
            scheduler_state=checkpoint_data.get("scheduler_state_dict"),
            epoch=checkpoint_data.get("epoch", 0),
            step=checkpoint_data.get("step", 0),
            metrics=checkpoint_data.get("metrics", {}),
            config=checkpoint_data.get("config", {})
        )


class CheckpointManager:
    """
    Manager for handling model checkpoints during training.
    
    Features:
    - Automatic saving at intervals
    - Best model tracking based on metrics
    - Cleanup of old checkpoints
    - Resume from checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_total_limit: int = 3,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        self.best_metric_value = float('-inf') if greater_is_better else float('inf')
        self.best_checkpoint_path = None
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing state if available
        self._load_manager_state()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Evaluation metrics
            config: Training configuration
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        
        # Convert config to dict if needed
        config_dict = {}
        if config is not None:
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            elif hasattr(config, '__dict__'):
                config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
            else:
                config_dict = dict(config) if isinstance(config, dict) else {}
        
        # Create checkpoint
        checkpoint = ModelCheckpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None,
            scheduler_state=scheduler.state_dict() if scheduler else None,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=config_dict
        )
        
        # Save checkpoint
        checkpoint_filename = f"checkpoint-epoch-{epoch}-step-{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        checkpoint.save(checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Check if this is the best checkpoint
        if self.metric_for_best_model in metrics:
            metric_value = metrics[self.metric_for_best_model]
            
            is_better = (
                (self.greater_is_better and metric_value > self.best_metric_value) or
                (not self.greater_is_better and metric_value < self.best_metric_value)
            )
            
            if is_better or is_best:
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_path
                
                # Save as best model
                best_model_path = self.checkpoint_dir / "best_model"
                if best_model_path.exists():
                    shutil.rmtree(best_model_path)
                
                best_model_path.mkdir(exist_ok=True)
                
                # Save model components
                torch.save(model.state_dict(), best_model_path / "pytorch_model.bin")
                
                # Save config
                with open(best_model_path / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Save metrics
                with open(best_model_path / "training_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                self.logger.info(f"New best model saved with {self.metric_for_best_model}={metric_value:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save manager state
        self._save_manager_state()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> ModelCheckpoint:
        """Load a checkpoint from disk."""
        return ModelCheckpoint.load(checkpoint_path)
    
    def load_best_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Load the best checkpoint if available."""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.load_checkpoint(self.best_checkpoint_path)
        return None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint-*.pt"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    def resume_from_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Specific checkpoint path (if None, uses latest)
            
        Returns:
            Dictionary with resume information
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None:
            self.logger.info("No checkpoint found for resuming")
            return {"resumed": False}
        
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint.model_state)
        
        # Load optimizer state
        if optimizer and checkpoint.optimizer_state:
            optimizer.load_state_dict(checkpoint.optimizer_state)
        
        # Load scheduler state
        if scheduler and checkpoint.scheduler_state:
            scheduler.load_state_dict(checkpoint.scheduler_state)
        
        resume_info = {
            "resumed": True,
            "epoch": checkpoint.epoch,
            "step": checkpoint.step,
            "metrics": checkpoint.metrics,
            "config": checkpoint.config
        }
        
        self.logger.info(f"Resumed from epoch {checkpoint.epoch}, step {checkpoint.step}")
        
        return resume_info
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to stay within save_total_limit."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint-*.pt"))
        
        if len(checkpoint_files) <= self.save_total_limit:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.save_total_limit]
        for file_path in files_to_remove:
            file_path.unlink()
            self.logger.debug(f"Removed old checkpoint: {file_path}")
    
    def _save_manager_state(self):
        """Save manager state for persistence."""
        state = {
            "best_metric_value": self.best_metric_value,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better
        }
        
        state_path = self.checkpoint_dir / "checkpoint_manager_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_manager_state(self):
        """Load manager state if available."""
        state_path = self.checkpoint_dir / "checkpoint_manager_state.json"
        
        if not state_path.exists():
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.best_metric_value = state.get("best_metric_value", 
                                             float('-inf') if self.greater_is_better else float('inf'))
            
            best_path_str = state.get("best_checkpoint_path")
            if best_path_str:
                self.best_checkpoint_path = Path(best_path_str)
            
            self.logger.info(f"Loaded checkpoint manager state: best_metric={self.best_metric_value}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint manager state: {e}")


if __name__ == "__main__":
    # Test checkpoint management
    import tempfile
    import torch.nn as nn
    
    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test checkpoint manager
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(tmp_dir, save_total_limit=2)
        
        # Save a few checkpoints
        for epoch in range(3):
            metrics = {"eval_loss": 1.0 - epoch * 0.2, "accuracy": 0.5 + epoch * 0.1}
            manager.save_checkpoint(model, optimizer, epoch=epoch, metrics=metrics)
        
        # Test resume
        resume_info = manager.resume_from_checkpoint(model, optimizer)
        print(f"Resume info: {resume_info}")
        
        print("Checkpoint management working correctly!")