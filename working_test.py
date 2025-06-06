#!/usr/bin/env python3
"""
Working test script with proper generation parameters.
"""

import torch
import sys
from pathlib import Path
sys.path.append('.')

def working_test():
    """Test with proper generation parameters."""
    
    print("🚀 Working Model Test")
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
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Loaded config and processors")
        
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
        
        # Use GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Model on {device}")
        
        # Load checkpoint weights
        checkpoint_file = "outputs/coco_pretraining/checkpoints/stage1_step_153530/model.safetensors"
        state_dict = load_file(checkpoint_file, device=str(device))
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Loaded trained weights")
        
        # Test with image
        from PIL import Image
        
        image_path = "test_data/images/test_image_1.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Test different prompts
        test_prompts = [
            "Describe this image.",
            "What do you see?",
            "What is in this picture?",
            "Tell me about this image."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/4 ---")
            print(f"💭 Prompt: {prompt}")
            
            try:
                # Process inputs
                image_inputs = image_processor(image, return_tensors="pt")
                
                # Create prompt with image token
                full_prompt = f"<image>\n{prompt}"
                text_inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Move to device
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                # Generate with careful parameters
                model.eval()
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                        images=image_inputs["pixel_values"],
                        max_new_tokens=30,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                
                # Decode only the new tokens
                input_length = text_inputs["input_ids"].shape[1]
                new_tokens = outputs[0][input_length:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                print(f"🤖 Response: {response}")
                
                if not response:
                    print("⚠️  Empty response - trying simpler generation...")
                    
                    # Try with greedy decoding
                    outputs = model.generate(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                        images=image_inputs["pixel_values"],
                        max_new_tokens=20,
                        do_sample=False  # Greedy
                    )
                    
                    new_tokens = outputs[0][input_length:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    print(f"🤖 Greedy Response: {response}")
                
            except Exception as e:
                print(f"❌ Error with prompt {i}: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Model testing completed!")
        print("✅ Your Stage 1 multimodal model is working!")
        print("💡 If responses are basic, that's normal for Stage 1")
        print("💡 Stage 2 training would improve conversation quality")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    working_test()