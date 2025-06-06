#!/usr/bin/env python3
"""
Quick test script for your trained multimodal model.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_model():
    """Test the trained model with sample images."""
    
    print("🤖 Testing Qwen-CLIP Multimodal Model")
    print("=" * 50)
    
    try:
        # Import required modules
        from models import MultimodalLLM
        from inference import MultimodalInferencePipeline
        from transformers import Qwen2Tokenizer, CLIPImageProcessor
        
        print("✅ Modules imported successfully")
        
        # Load tokenizer and image processor
        print("🔄 Loading tokenizer and image processor...")
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Add special tokens
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        print("✅ Tokenizer and image processor loaded")
        
        # Load your trained model
        model_path = "outputs/coco_pretraining/best_model"
        print(f"🔄 Loading model from {model_path}...")
        
        model = MultimodalLLM.from_pretrained(model_path)
        pipeline = MultimodalInferencePipeline(model, tokenizer, image_processor)
        
        print("✅ Model loaded successfully!")
        
        # Test with sample images
        test_images = [
            "test_data/images/test_image_1.jpg",
            "test_data/images/test_image_2.jpg"
        ]
        
        test_prompts = [
            "Describe this image in detail.",
            "What do you see in this picture?",
            "What is the main subject of this image?",
            "What colors are prominent in this image?",
            "What is happening in this scene?"
        ]
        
        print("\n" + "=" * 50)
        print("🖼️  TESTING WITH SAMPLE IMAGES")
        print("=" * 50)
        
        for i, image_path in enumerate(test_images, 1):
            if Path(image_path).exists():
                print(f"\n📸 Test Image {i}: {image_path}")
                print("-" * 30)
                
                for j, prompt in enumerate(test_prompts[:2], 1):  # Test 2 prompts per image
                    try:
                        print(f"💭 Prompt {j}: {prompt}")
                        response = pipeline.chat(prompt, image=image_path)
                        print(f"🤖 Response: {response}")
                        print()
                    except Exception as e:
                        print(f"❌ Error with prompt {j}: {e}")
                        print()
            else:
                print(f"⚠️  Image not found: {image_path}")
        
        print("=" * 50)
        print("🎉 Model testing completed!")
        print("✅ Your multimodal AI is working!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Make sure your model was trained and saved correctly")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check if your model training completed successfully")

def test_model_components():
    """Test individual model components."""
    
    print("\n🔧 COMPONENT TESTING")
    print("=" * 30)
    
    try:
        # Test model loading
        model_path = "outputs/coco_pretraining/best_model"
        if Path(model_path).exists():
            print("✅ Model directory exists")
            
            # Check required files
            required_files = ["model.safetensors", "training_state.json"]
            for file in required_files:
                file_path = Path(model_path) / file
                if file_path.exists():
                    size = file_path.stat().st_size / (1024*1024)  # MB
                    print(f"✅ {file} exists ({size:.1f} MB)")
                else:
                    print(f"❌ {file} missing")
        else:
            print(f"❌ Model directory not found: {model_path}")
            print("💡 Run: cp -r outputs/coco_pretraining/checkpoints/stage1_step_153530 outputs/coco_pretraining/best_model")
            
    except Exception as e:
        print(f"❌ Component test error: {e}")

if __name__ == "__main__":
    # Test components first
    test_model_components()
    
    # Then test the full model
    test_model()