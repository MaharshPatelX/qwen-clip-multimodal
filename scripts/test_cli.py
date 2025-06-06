#!/usr/bin/env python3
"""
Command line testing for multimodal model.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_single_image(model_path, image_path, prompt):
    """Test model with a single image and prompt."""
    
    try:
        from models import MultimodalLLM
        from inference import MultimodalInferencePipeline
        from transformers import Qwen2Tokenizer, CLIPImageProcessor
        
        print(f"🔄 Loading model from {model_path}...")
        
        # Load model and processors
        model = MultimodalLLM.from_pretrained(model_path)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Add special tokens
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Create pipeline
        pipeline = MultimodalInferencePipeline(model, tokenizer, image_processor)
        
        print(f"📸 Image: {image_path}")
        print(f"💭 Prompt: {prompt}")
        print("-" * 50)
        
        # Generate response
        response = pipeline.chat(prompt, image=image_path)
        
        print(f"🤖 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test multimodal model")
    parser.add_argument("--model", default="outputs/coco_pretraining/best_model", 
                       help="Path to trained model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--prompt", default="Describe this image in detail.", 
                       help="Prompt for the model")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Make sure you've copied the checkpoint to best_model directory")
        return
    
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    test_single_image(args.model, args.image, args.prompt)

if __name__ == "__main__":
    main()