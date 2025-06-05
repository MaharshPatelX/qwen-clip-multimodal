"""
Inference Package

This package contains inference utilities for the multimodal LLM including:
- Inference pipeline for various tasks (captioning, VQA, chat)
- FastAPI-based REST API for model serving
- Batch processing capabilities
"""

from .pipeline import MultimodalInferencePipeline, BatchInferencePipeline

__all__ = [
    "MultimodalInferencePipeline",
    "BatchInferencePipeline"
]