"""
Training Package

This package contains training utilities for the multimodal LLM including:
- Training configuration classes
- Custom trainer with two-stage training
- Loss functions and optimization utilities
"""

from .config import (
    ModelConfig,
    DataConfig, 
    TrainingConfig,
    GenerationConfig,
    ExperimentConfig,
    get_debug_config,
    get_small_scale_config,
    get_full_scale_config
)
from .trainer import MultimodalTrainer

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainingConfig", 
    "GenerationConfig",
    "ExperimentConfig",
    "get_debug_config",
    "get_small_scale_config", 
    "get_full_scale_config",
    "MultimodalTrainer"
]