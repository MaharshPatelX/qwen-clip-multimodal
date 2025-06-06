#!/usr/bin/env python3
"""
Simple test script that loads checkpoint directly.
"""

import torch
import sys
from pathlib import Path
sys.path.append('.')

def simple_test():
    """Simple test by loading checkpoint directly."""
    
    print("🧪 Simple Model Test")
    print("=" * 30)
    
    try:
        from models import MultimodalLLM
        from training.config import ExperimentConfig
        from transformers import Qwen2Tokenizer, CLIPImageProcessor
        from safetensors.torch import load_file
        
        # Load config
        config = ExperimentConfig.load("configs/coco_pretraining.json")
        
        # Load tokenizer and processor
        tokenizer = Qwen2Tokenizer.from_pretrained(config.model.qwen_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(config.model.clip_model_name)
        
        # Add special tokens
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        print("✅ Loaded config and processors")
        
        # Create model with correct parameters
        model = MultimodalLLM(
            clip_model_name=config.model.clip_model_name,
            qwen_model_name=config.model.qwen_model_name,
            fusion_type=config.model.fusion_type,
            fusion_config=config.model.fusion_config,
            freeze_vision=config.model.freeze_vision,
            use_lora=config.model.use_lora,
            lora_config=config.model.lora_config
        )
        
        print("✅ Created model architecture")
        
        # Load checkpoint weights
        checkpoint_file = "outputs/coco_pretraining/checkpoints/stage1_step_153530/model.safetensors"
        state_dict = load_file(checkpoint_file)
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Loaded trained weights")
        
        # Test with image
        from PIL import Image
        
        image_path = "test_data/images/test_image_1.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        image_inputs = image_processor(image, return_tensors="pt")
        
        # Create text input
        prompt = "Describe this image."
        text_inputs = tokenizer(f"<image>\n{prompt}", return_tensors="pt", padding=True)
        
        print(f"📸 Testing with: {image_path}")
        print(f"💭 Prompt: {prompt}")
        
        # Generate response
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"], 
                images=image_inputs["pixel_values"],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("🤖 Response:")
        print(f"   {response}")
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()