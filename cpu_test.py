#!/usr/bin/env python3
"""
CPU-only test script for the model.
"""

import torch
import sys
from pathlib import Path
sys.path.append('.')

def cpu_test():
    """Test model on CPU only to avoid device issues."""
    
    print("🖥️  CPU-Only Model Test")
    print("=" * 30)
    
    try:
        from models import MultimodalLLM
        from training.config import ExperimentConfig
        from transformers import Qwen2Tokenizer, CLIPImageProcessor
        from safetensors.torch import load_file
        
        # Force CPU usage
        torch.cuda.is_available = lambda: False
        device = torch.device("cpu")
        print(f"🖥️  Using device: {device}")
        
        # Load config
        config = ExperimentConfig.load("configs/coco_pretraining.json")
        
        # Load tokenizer and processor
        tokenizer = Qwen2Tokenizer.from_pretrained(config.model.qwen_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(config.model.clip_model_name)
        
        # Add special tokens
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        print("✅ Loaded config and processors")
        
        # Create model with CPU device map
        model = MultimodalLLM(
            clip_model_name=config.model.clip_model_name,
            qwen_model_name=config.model.qwen_model_name,
            fusion_type=config.model.fusion_type,
            fusion_config=config.model.fusion_config,
            freeze_vision=config.model.freeze_vision,
            use_lora=config.model.use_lora,
            lora_config=config.model.lora_config,
            device_map="cpu"
        )
        
        print("✅ Created model architecture")
        
        # Load checkpoint weights
        checkpoint_file = "outputs/coco_pretraining/checkpoints/stage1_step_153530/model.safetensors"
        state_dict = load_file(checkpoint_file, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        # Ensure everything is on CPU
        model = model.to(device)
        
        print("✅ Loaded trained weights")
        
        # Test with image
        from PIL import Image
        
        image_path = "test_data/images/test_image_1.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        image_inputs = image_processor(image, return_tensors="pt")
        
        # Create text input with simple prompt
        prompt = "Describe this image."
        text_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Ensure all inputs are on CPU
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        print(f"📸 Testing with: {image_path}")
        print(f"💭 Prompt: {prompt}")
        print("🔄 Generating response... (this may take a moment on CPU)")
        
        # Generate response
        model.eval()
        with torch.no_grad():
            try:
                # Simple forward pass first
                vision_features = model.vision_encoder(image_inputs["pixel_values"])
                print("✅ Vision encoder working")
                
                # Test text processing
                text_embeddings = model.language_decoder.get_embeddings(text_inputs["input_ids"])
                print("✅ Language decoder working")
                
                # Try a simple caption
                outputs = model.generate(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"], 
                    images=image_inputs["pixel_values"],
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=False,  # Use greedy decoding for consistency
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print("🤖 Response:")
                print(f"   {response}")
                print("\n✅ Test completed successfully!")
                
            except Exception as gen_error:
                print(f"❌ Generation error: {gen_error}")
                # Try just the vision part
                try:
                    vision_features = model.vision_encoder(image_inputs["pixel_values"])
                    print(f"✅ Vision encoder works - features shape: {vision_features.shape}")
                except Exception as vision_error:
                    print(f"❌ Vision encoder error: {vision_error}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cpu_test()