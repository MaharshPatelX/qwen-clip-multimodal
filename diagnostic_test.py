#!/usr/bin/env python3
"""
Diagnostic test to understand model behavior.
"""

import torch
import sys
from pathlib import Path
sys.path.append('.')

def diagnostic_test():
    """Detailed diagnostic test."""
    
    print("🔍 Diagnostic Model Test")
    print("=" * 40)
    
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
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Loaded tokenizer and processor")
        print(f"📊 Vocab size: {len(tokenizer)}")
        print(f"📊 Special tokens: {tokenizer.additional_special_tokens}")
        
        # Create model
        model = MultimodalLLM(
            clip_model_name=config.model.clip_model_name,
            qwen_model_name=config.model.qwen_model_name,
            fusion_type=config.model.fusion_type,
            fusion_config=config.model.fusion_config,
            freeze_vision=config.model.freeze_vision,
            use_lora=config.model.use_lora,
            lora_config=config.model.lora_config
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load checkpoint
        checkpoint_file = "outputs/coco_pretraining/checkpoints/stage1_step_153530/model.safetensors"
        state_dict = load_file(checkpoint_file, device=str(device))
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Model loaded and ready")
        
        # Test image processing
        from PIL import Image
        image_path = "test_data/images/test_image_1.jpg"
        image = Image.open(image_path).convert('RGB')
        
        print(f"\n🖼️  Testing with: {image_path}")
        print(f"📐 Image size: {image.size}")
        
        # Process image
        image_inputs = image_processor(image, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        print(f"📊 Image tensor shape: {image_inputs['pixel_values'].shape}")
        
        # Test vision encoder
        model.eval()
        with torch.no_grad():
            vision_features = model.vision_encoder(image_inputs["pixel_values"])
            print(f"✅ Vision features shape: {vision_features.shape}")
            
            projected_vision = model.fusion_module(vision_features)
            print(f"✅ Projected features shape: {projected_vision.shape}")
        
        # Test different prompt formats
        test_prompts = [
            "cat",  # Single word
            "A cat",  # Simple phrase
            "This is a",  # Incomplete sentence
            "<image>",  # Just image token
            "<image> cat",  # Image + word
            "<image> This is",  # Image + start of sentence
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: '{prompt}' ---")
            
            try:
                # Tokenize
                text_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                print(f"📝 Input tokens: {text_inputs['input_ids'].shape}")
                print(f"📝 Token IDs: {text_inputs['input_ids'][0].tolist()}")
                print(f"📝 Decoded: '{tokenizer.decode(text_inputs['input_ids'][0])}'")
                
                # Generate very conservatively
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                        images=image_inputs["pixel_values"],
                        max_new_tokens=5,  # Very few tokens
                        do_sample=False,   # Greedy
                        num_beams=1       # Single beam
                    )
                
                print(f"📤 Output shape: {outputs.shape}")
                print(f"📤 Output tokens: {outputs[0].tolist()}")
                
                # Decode full output
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"📤 Full response: '{full_response}'")
                
                # Decode only new tokens
                input_length = text_inputs["input_ids"].shape[1]
                if outputs.shape[1] > input_length:
                    new_tokens = outputs[0][input_length:]
                    new_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    print(f"✨ New tokens: '{new_response}'")
                else:
                    print("⚠️  No new tokens generated")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test language model directly (without images)
        print(f"\n🧠 Testing Language Model Directly")
        print("-" * 30)
        
        simple_prompt = "The cat is"
        text_inputs = tokenizer(simple_prompt, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            # Test just the language decoder
            lm_outputs = model.language_decoder.model.generate(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False
            )
        
        lm_response = tokenizer.decode(lm_outputs[0], skip_special_tokens=True)
        print(f"🧠 Language model response: '{lm_response}'")
        
        print("\n" + "=" * 50)
        print("🎯 DIAGNOSIS COMPLETE")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostic_test()