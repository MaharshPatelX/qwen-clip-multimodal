import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import base64
import io

try:
    from ..models import MultimodalLLM
    from ..training.config import GenerationConfig
except ImportError:
    # Fallback for when running directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import MultimodalLLM
    from training.config import GenerationConfig


class MultimodalInferencePipeline:
    """
    Inference pipeline for the multimodal LLM.
    Provides easy-to-use methods for image captioning, VQA, and chat.
    """
    
    def __init__(
        self,
        model: MultimodalLLM,
        device: str = "auto",
        generation_config: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = generation_config or GenerationConfig()
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache for conversation history
        self.conversation_cache = {}
        
    @torch.no_grad()
    def caption_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            prompt: Caption generation prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated caption string
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Format input text
        input_text = f"<image>\n{prompt}"
        
        # Generate caption
        start_time = time.time()
        
        generated_text = self.model.generate(
            images=processed_image,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        inference_time = time.time() - start_time
        
        # Extract just the generated part (remove input)
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        
        self.logger.info(f"Caption generated in {inference_time:.2f}s")
        
        return generated_text.strip()
    
    @torch.no_grad()
    def answer_question(
        self,
        image: Union[str, Image.Image, np.ndarray],
        question: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Answer a question about an image.
        
        Args:
            image: Input image
            question: Question about the image
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated answer string
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Format as QA
        input_text = f"<image>\nQuestion: {question}\nAnswer:"
        
        # Generate answer
        start_time = time.time()
        
        generated_text = self.model.generate(
            images=processed_image,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        inference_time = time.time() - start_time
        
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        
        self.logger.info(f"Answer generated in {inference_time:.2f}s")
        
        return generated_text.strip()
    
    @torch.no_grad()
    def chat(
        self,
        message: str,
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        conversation_id: str = "default",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        reset_conversation: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Multimodal chat interface with conversation memory.
        
        Args:
            message: User message
            image: Optional image input
            conversation_id: ID for conversation tracking
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reset_conversation: Whether to reset conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Reset conversation if requested
        if reset_conversation or conversation_id not in self.conversation_cache:
            self.conversation_cache[conversation_id] = []
        
        # Get conversation history
        history = self.conversation_cache[conversation_id]
        
        # Preprocess image if provided
        processed_image = None
        if image is not None:
            processed_image = self._preprocess_image(image)
        
        # Format conversation input
        if processed_image is not None:
            current_input = f"<image>\n{message}"
        else:
            current_input = message
        
        # Build conversation context
        conversation_text = self._format_conversation(history, current_input)
        
        # Generate response
        start_time = time.time()
        
        response = self.model.generate(
            images=processed_image,
            input_text=conversation_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        inference_time = time.time() - start_time
        
        if isinstance(response, list):
            response = response[0]
        
        response = response.strip()
        
        # Update conversation history
        self.conversation_cache[conversation_id].append({
            "role": "user",
            "content": message,
            "has_image": image is not None
        })
        self.conversation_cache[conversation_id].append({
            "role": "assistant", 
            "content": response,
            "has_image": False
        })
        
        # Keep conversation history manageable
        if len(self.conversation_cache[conversation_id]) > 20:
            self.conversation_cache[conversation_id] = self.conversation_cache[conversation_id][-20:]
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "inference_time": inference_time,
            "has_image": image is not None,
            "turn_count": len(self.conversation_cache[conversation_id]) // 2
        }
    
    @torch.no_grad()
    def batch_caption(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images in batches.
        
        Args:
            images: List of input images
            prompts: Optional list of prompts (one per image)
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated captions
        """
        if prompts is None:
            prompts = ["Describe this image in detail."] * len(images)
        elif len(prompts) == 1:
            prompts = prompts * len(images)
        
        assert len(images) == len(prompts), "Number of images and prompts must match"
        
        captions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            batch_captions = []
            for img, prompt in zip(batch_images, batch_prompts):
                caption = self.caption_image(img, prompt, **kwargs)
                batch_captions.append(caption)
            
            captions.extend(batch_captions)
            
            self.logger.info(f"Processed batch {i // batch_size + 1}/{(len(images) - 1) // batch_size + 1}")
        
        return captions
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Preprocess image input to PIL Image format.
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image, str):
            # File path
            if image.startswith("data:image"):
                # Base64 encoded image
                image_data = image.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # File path
                image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Numpy array
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # Normalize to 0-255 range
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def _format_conversation(self, history: List[Dict], current_input: str) -> str:
        """Format conversation history for model input."""
        conversation_parts = []
        
        # Add previous turns
        for turn in history[-6:]:  # Keep last 6 turns for context
            role = turn["role"]
            content = turn["content"]
            
            if role == "user":
                conversation_parts.append(f"Human: {content}")
            else:
                conversation_parts.append(f"Assistant: {content}")
        
        # Add current input
        conversation_parts.append(f"Human: {current_input}")
        conversation_parts.append("Assistant:")
        
        return "\n".join(conversation_parts)
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for a given ID."""
        return self.conversation_cache.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a given ID."""
        if conversation_id in self.conversation_cache:
            del self.conversation_cache[conversation_id]
    
    def clear_all_conversations(self):
        """Clear all conversation histories."""
        self.conversation_cache.clear()
    
    def benchmark_inference(
        self,
        test_images: List[Union[str, Image.Image]],
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            test_images: List of test images
            num_runs: Number of runs for averaging
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Running inference benchmark with {len(test_images)} images, {num_runs} runs each")
        
        inference_times = []
        
        for _ in range(num_runs):
            for image in test_images:
                start_time = time.time()
                _ = self.caption_image(image, max_new_tokens=50)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
        
        metrics = {
            "avg_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "throughput_fps": 1.0 / np.mean(inference_times)
        }
        
        self.logger.info(f"Benchmark results: {metrics}")
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "clip_model": self.model.clip_model_name,
            "qwen_model": self.model.qwen_model_name,
            "fusion_type": self.model.fusion_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "vision_dim": self.model.vision_dim,
            "language_dim": self.model.language_dim
        }


class BatchInferencePipeline:
    """Optimized pipeline for batch inference on large datasets."""
    
    def __init__(
        self,
        model: MultimodalLLM,
        device: str = "auto",
        batch_size: int = 8
    ):
        self.model = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @torch.no_grad()
    def process_dataset(
        self,
        image_paths: List[str],
        prompts: List[str],
        output_file: str,
        **generation_kwargs
    ):
        """
        Process a large dataset of images with prompts.
        
        Args:
            image_paths: List of image file paths
            prompts: List of prompts (one per image)
            output_file: Output JSON file path
            **generation_kwargs: Generation parameters
        """
        import json
        
        results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            
            batch_results = []
            for img_path, prompt in zip(batch_paths, batch_prompts):
                try:
                    pipeline = MultimodalInferencePipeline(self.model, self.device)
                    caption = pipeline.caption_image(img_path, prompt, **generation_kwargs)
                    
                    batch_results.append({
                        "image_path": img_path,
                        "prompt": prompt,
                        "generated_text": caption,
                        "success": True
                    })
                except Exception as e:
                    batch_results.append({
                        "image_path": img_path,
                        "prompt": prompt,
                        "generated_text": "",
                        "success": False,
                        "error": str(e)
                    })
            
            results.extend(batch_results)
            
            # Save intermediate results
            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self.logger.info(f"Processed {i + self.batch_size}/{len(image_paths)} images")
        
        # Save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"Processing complete: {success_count}/{len(results)} successful")


if __name__ == "__main__":
    # Test inference pipeline
    print("Inference pipeline classes loaded successfully!")
    
    # Create a dummy config for testing
    try:
        from ..training.config import GenerationConfig
    except ImportError:
        from training.config import GenerationConfig
    
    gen_config = GenerationConfig()
    print(f"Generation config: max_tokens={gen_config.max_new_tokens}, temp={gen_config.temperature}")
    
    print("Inference pipeline ready for use!")