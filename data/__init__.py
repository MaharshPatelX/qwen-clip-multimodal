"""
Data Processing Package

This package contains utilities for loading, preprocessing, and managing
multimodal datasets for training the multimodal LLM.

Components:
- datasets: Dataset classes for different data formats
- preprocessing: Data downloading and conversion utilities
- dataloaders: PyTorch DataLoader configurations
"""

from .datasets.multimodal_dataset import (
    MultimodalDataset, 
    InstructionDataset, 
    VQADataset,
    create_dataloader
)
from .preprocessing.data_utils import (
    DataDownloader,
    DatasetConverter, 
    DataValidator
)

__all__ = [
    "MultimodalDataset",
    "InstructionDataset", 
    "VQADataset",
    "create_dataloader",
    "DataDownloader",
    "DatasetConverter",
    "DataValidator"
]