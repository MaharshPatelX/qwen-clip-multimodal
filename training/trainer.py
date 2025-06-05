import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_scheduler
)
from accelerate import Accelerator
import wandb
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm
import json
from pathlib import Path
import time

from .config import ExperimentConfig

try:
    from ..models import MultimodalLLM
    from ..data import MultimodalDataset, InstructionDataset, create_dataloader
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import MultimodalLLM
    from data import MultimodalDataset, InstructionDataset, create_dataloader


class MultimodalTrainer:
    """
    Custom trainer for multimodal LLM with two-stage training:
    1. Vision-language alignment (freeze language model)
    2. End-to-end fine-tuning
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: MultimodalLLM,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer=None,
        compute_metrics=None
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        
        # Setup accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision="bf16" if config.training.bf16 else ("fp16" if config.training.fp16 else "no"),
            log_with=config.training.report_to,
            project_dir=config.logging_dir
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if self.accelerator.is_local_main_process else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        
        # Initialize tracking
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.experiment_name,
                config=config.to_dict()
            )
        
        # Training state
        self.global_step = 0
        self.best_metric = float('inf') if not config.training.greater_is_better else float('-inf')
        self.patience_counter = 0
        
    def train(self):
        """Execute the complete two-stage training process."""
        self.logger.info("Starting multimodal LLM training...")
        
        # Stage 1: Vision-language alignment
        self.logger.info("=" * 50)
        self.logger.info("STAGE 1: Vision-Language Alignment")
        self.logger.info("=" * 50)
        self._train_stage1()
        
        # Skip Stage 2 if stage2_epochs is 0
        if self.config.training.stage2_epochs > 0:
            # Stage 2: End-to-end fine-tuning  
            self.logger.info("=" * 50)
            self.logger.info("STAGE 2: End-to-End Fine-tuning")
            self.logger.info("=" * 50)
            self._train_stage2()
        else:
            self.logger.info("Stage 2 skipped (stage2_epochs = 0)")
        
        # Final evaluation
        if self.eval_dataset is not None:
            self.logger.info("Running final evaluation...")
            final_metrics = self.evaluate()
            self.logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Save final model
        self.save_model(os.path.join(self.config.output_dir, "final_model"))
        
        # Cleanup
        if self.accelerator.is_main_process:
            self.accelerator.end_training()
        
        self.logger.info("Training completed!")
    
    def _train_stage1(self):
        """Stage 1: Train only fusion module while keeping other components frozen."""
        # Freeze vision encoder and language model
        self._freeze_model_components(freeze_fusion=False)
        
        # Setup optimizer for fusion module only
        optimizer = self._create_optimizer(fusion_only=True)
        
        # Create data loaders
        train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = create_dataloader(
                self.eval_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers
            )
        
        # Setup learning rate scheduler
        num_training_steps = len(train_dataloader) * self.config.training.stage1_epochs
        lr_scheduler = self._create_lr_scheduler(optimizer, num_training_steps)
        
        # Prepare for distributed training
        self.model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, lr_scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.training.stage1_epochs):
            self.logger.info(f"Stage 1 - Epoch {epoch + 1}/{self.config.training.stage1_epochs}")
            
            epoch_loss = self._train_epoch(
                train_dataloader, optimizer, lr_scheduler, epoch, "stage1"
            )
            
            # Evaluation
            if eval_dataloader is not None and (epoch + 1) % self.config.training.eval_steps == 0:
                eval_metrics = self._evaluate_epoch(eval_dataloader, epoch, "stage1")
                self._handle_eval_metrics(eval_metrics, epoch)
            
            self.logger.info(f"Stage 1 - Epoch {epoch + 1} completed, average loss: {epoch_loss:.4f}")
        
        self.logger.info("Stage 1 training completed")
    
    def _train_stage2(self):
        """Stage 2: End-to-end fine-tuning of the entire model."""
        # Unfreeze model components for end-to-end training
        self._freeze_model_components(freeze_fusion=False, freeze_language=False)
        
        # Setup optimizer for entire model
        optimizer = self._create_optimizer(fusion_only=False)
        
        # Create data loaders (reuse from stage 1)
        train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = create_dataloader(
                self.eval_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers
            )
        
        # Setup learning rate scheduler
        num_training_steps = len(train_dataloader) * self.config.training.stage2_epochs
        lr_scheduler = self._create_lr_scheduler(optimizer, num_training_steps)
        
        # Prepare for distributed training
        self.model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, lr_scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.training.stage2_epochs):
            self.logger.info(f"Stage 2 - Epoch {epoch + 1}/{self.config.training.stage2_epochs}")
            
            epoch_loss = self._train_epoch(
                train_dataloader, optimizer, lr_scheduler, epoch, "stage2"
            )
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self._evaluate_epoch(eval_dataloader, epoch, "stage2")
                self._handle_eval_metrics(eval_metrics, epoch)
            
            self.logger.info(f"Stage 2 - Epoch {epoch + 1} completed, average loss: {epoch_loss:.4f}")
        
        self.logger.info("Stage 2 training completed")
    
    def _train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler, 
        epoch: int, 
        stage: str
    ) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"{stage.upper()} Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
                    images=batch.get('image'),
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if (step + 1) % self.config.training.logging_steps == 0:
                self.accelerator.log({
                    f"{stage}/loss": loss.item(),
                    f"{stage}/learning_rate": lr_scheduler.get_last_lr()[0],
                    f"{stage}/epoch": epoch,
                    "global_step": self.global_step
                })
            
            self.global_step += 1
            
            # Save checkpoint
            if (step + 1) % self.config.training.save_steps == 0:
                self.save_checkpoint(f"{stage}_step_{self.global_step}")
        
        return total_loss / num_batches
    
    def _evaluate_epoch(self, dataloader: DataLoader, epoch: int, stage: str) -> Dict[str, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                outputs = self.model(
                    images=batch.get('image'),
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = {f"eval_{stage}/loss": avg_loss}
        
        # Log metrics
        self.accelerator.log({
            **metrics,
            f"eval_{stage}/epoch": epoch,
            "global_step": self.global_step
        })
        
        self.model.train()
        return metrics
    
    def _handle_eval_metrics(self, metrics: Dict[str, float], epoch: int):
        """Handle evaluation metrics for early stopping and model saving."""
        metric_key = f"eval_loss"  # Simplified for now
        current_metric = list(metrics.values())[0]  # Get the loss value
        
        # Check if this is the best model
        is_best = False
        if self.config.training.greater_is_better:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Save best model
        if is_best:
            self.save_model(os.path.join(self.config.output_dir, "best_model"))
            self.logger.info(f"New best model saved with {metric_key}: {current_metric:.4f}")
        
        # Early stopping
        if (self.patience_counter >= self.config.training.early_stopping_patience and 
            self.config.training.early_stopping_patience > 0):
            self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
            return True
        
        return False
    
    def _freeze_model_components(self, freeze_fusion=True, freeze_vision=True, freeze_language=True):
        """Freeze/unfreeze different model components."""
        # Vision encoder
        if freeze_vision and self.config.model.freeze_vision:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = True
        
        # Fusion module
        for param in self.model.fusion_module.parameters():
            param.requires_grad = not freeze_fusion
        
        # Language model
        if freeze_language:
            for param in self.model.language_decoder.parameters():
                param.requires_grad = False
        else:
            # If using LoRA, only unfreeze LoRA parameters
            if self.config.model.use_lora:
                for name, param in self.model.language_decoder.model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for param in self.model.language_decoder.parameters():
                    param.requires_grad = True
    
    def _create_optimizer(self, fusion_only: bool = False) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        if fusion_only:
            # Only optimize fusion module parameters
            params = list(self.model.fusion_module.parameters())
        else:
            # Optimize all trainable parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
        
        return optimizer
    
    def _create_lr_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler."""
        num_warmup_steps = (
            self.config.training.warmup_steps if self.config.training.warmup_steps > 0
            else int(num_training_steps * self.config.training.warmup_ratio)
        )
        
        scheduler = get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set."""
        if self.eval_dataset is None:
            self.logger.warning("No evaluation dataset provided")
            return {}
        
        eval_dataloader = create_dataloader(
            self.eval_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        return self._evaluate_epoch(eval_dataloader, 0, "final")
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap model from accelerator
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model components
        unwrapped_model.save_pretrained(output_dir)
        
        # Save configuration
        self.config.save(os.path.join(output_dir, "training_config.json"))
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save accelerator state
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional training state
        training_state = {
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(training_state, f)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        # Load accelerator state
        self.accelerator.load_state(checkpoint_dir)
        
        # Load additional training state
        training_state_file = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(training_state_file):
            with open(training_state_file, 'r') as f:
                training_state = json.load(f)
            
            self.global_step = training_state["global_step"]
            self.best_metric = training_state["best_metric"]
            self.patience_counter = training_state["patience_counter"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_dir}")


if __name__ == "__main__":
    # Test trainer initialization
    from .config import get_debug_config
    
    config = get_debug_config()
    print("Trainer class loaded successfully!")
    print(f"Config experiment name: {config.experiment_name}")
    print(f"Training stages: {config.training.stage1_epochs} + {config.training.stage2_epochs}")