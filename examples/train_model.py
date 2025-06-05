#!/usr/bin/env python3
"""
Example script for training the multimodal LLM.

This script demonstrates how to:
1. Set up training configuration
2. Load and prepare datasets
3. Initialize the model
4. Run two-stage training
5. Evaluate the trained model

Usage:
    python examples/train_model.py --config configs/debug.json
    python examples/train_model.py --config configs/small_scale.json
"""

import argparse
import logging
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import MultimodalLLM
from data import MultimodalDataset, InstructionDataset, create_dataloader
from training import ExperimentConfig, MultimodalTrainer, get_debug_config, get_small_scale_config
from evaluation import MultimodalEvaluator
from transformers import Qwen2Tokenizer, CLIPImageProcessor


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def prepare_datasets(config: ExperimentConfig):
    """Prepare training and validation datasets."""
    logger = logging.getLogger(__name__)
    
    # Initialize tokenizer and image processor
    tokenizer = Qwen2Tokenizer.from_pretrained(config.model.qwen_model_name)
    image_processor = CLIPImageProcessor.from_pretrained(config.model.clip_model_name)
    
    # Add special tokens
    special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Create datasets based on configuration
    if config.data.dataset_type == "custom":
        train_dataset = MultimodalDataset(
            data_path=config.data.train_data_path,
            image_dir=config.data.image_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=config.data.max_length,
            dataset_type=config.data.dataset_type
        )
        
        val_dataset = None
        if config.data.val_data_path:
            val_dataset = MultimodalDataset(
                data_path=config.data.val_data_path,
                image_dir=config.data.image_dir,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_length=config.data.max_length,
                dataset_type=config.data.dataset_type
            )
    
    elif config.data.dataset_type == "instruction":
        train_dataset = InstructionDataset(
            data_path=config.data.train_data_path,
            image_dir=config.data.image_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=config.data.max_length
        )
        
        val_dataset = None
        if config.data.val_data_path:
            val_dataset = InstructionDataset(
                data_path=config.data.val_data_path,
                image_dir=config.data.image_dir,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_length=config.data.max_length
            )
    
    else:
        raise ValueError(f"Unsupported dataset type: {config.data.dataset_type}")
    
    # Apply debug limitations if enabled
    if config.debug:
        if config.max_train_samples and len(train_dataset) > config.max_train_samples:
            train_dataset.data = train_dataset.data[:config.max_train_samples]
        
        if val_dataset and config.max_eval_samples and len(val_dataset) > config.max_eval_samples:
            val_dataset.data = val_dataset.data[:config.max_eval_samples]
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer


def create_model(config: ExperimentConfig, tokenizer):
    """Create and initialize the multimodal model."""
    logger = logging.getLogger(__name__)
    
    model = MultimodalLLM(
        clip_model_name=config.model.clip_model_name,
        qwen_model_name=config.model.qwen_model_name,
        fusion_type=config.model.fusion_type,
        fusion_config=config.model.fusion_config,
        freeze_vision=config.model.freeze_vision,
        use_lora=config.model.use_lora,
        lora_config=config.model.lora_config
    )
    
    # Resize token embeddings to match tokenizer
    model.language_decoder.model.resize_token_embeddings(len(tokenizer))
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {100.0 * trainable_params / total_params:.2f}%")
    
    return model


def train_model(config: ExperimentConfig, model, train_dataset, val_dataset, tokenizer):
    """Run the training process."""
    logger = logging.getLogger(__name__)
    
    # Create trainer
    trainer = MultimodalTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    logger.info("Starting training...")
    
    # Run training
    trainer.train()
    
    logger.info("Training completed!")
    
    return trainer


def evaluate_model(config: ExperimentConfig, model, val_dataset, tokenizer):
    """Evaluate the trained model."""
    logger = logging.getLogger(__name__)
    
    if val_dataset is None:
        logger.warning("No validation dataset provided for evaluation")
        return {}
    
    logger.info("Running model evaluation...")
    
    # Create evaluator
    evaluator = MultimodalEvaluator()
    
    # For demonstration, just run a small evaluation
    # In practice, you'd want to generate predictions for the entire validation set
    
    from inference import MultimodalInferencePipeline
    
    pipeline = MultimodalInferencePipeline(model)
    
    # Sample a few examples for evaluation
    sample_size = min(50, len(val_dataset))
    predictions = []
    references = []
    
    logger.info(f"Evaluating on {sample_size} samples...")
    
    for i in range(sample_size):
        sample = val_dataset[i]
        
        # For this example, we'll use a simple caption prompt
        # In practice, you'd want to handle different task types
        try:
            # Generate prediction
            prediction = pipeline.caption_image(
                image=sample['image'],
                prompt="Describe this image.",
                max_new_tokens=100
            )
            
            # Get reference (decode the labels)
            reference_ids = sample['labels']
            reference = tokenizer.decode(reference_ids, skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append([reference])  # BLEU expects list of references
            
        except Exception as e:
            logger.warning(f"Error evaluating sample {i}: {e}")
            continue
    
    # Compute metrics
    if predictions and references:
        results = evaluator.evaluate_captioning(predictions, references)
        
        logger.info("Evaluation results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save results
        evaluator.save_results(results, f"{config.output_dir}/evaluation_results.json")
        
        return results
    else:
        logger.warning("No valid predictions generated for evaluation")
        return {}


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train multimodal LLM")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Use debug configuration")
    parser.add_argument("--small-scale", action="store_true", help="Use small-scale configuration")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.debug:
        config = ExperimentConfig.load("configs/debug.json")
    elif args.small_scale:
        config = ExperimentConfig.load("configs/small_scale.json")
    else:
        config = ExperimentConfig()
    
    logger.info(f"Using configuration: {config.experiment_name}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Prepare datasets
    train_dataset, val_dataset, tokenizer = prepare_datasets(config)
    
    # Create model
    model = create_model(config, tokenizer)
    
    if not args.eval_only:
        # Train model
        trainer = train_model(config, model, train_dataset, val_dataset, tokenizer)
        
        # Load best model for evaluation
        best_model_path = f"{config.output_dir}/best_model"
        if Path(best_model_path).exists():
            logger.info("Loading best model for evaluation...")
            model = MultimodalLLM.from_pretrained(best_model_path)
    
    # Evaluate model
    if val_dataset is not None:
        evaluation_results = evaluate_model(config, model, val_dataset, tokenizer)
    
    logger.info("Script completed successfully!")


if __name__ == "__main__":
    main()