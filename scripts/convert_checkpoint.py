#!/usr/bin/env python3
"""
Convert training checkpoint to a loadable model format.
"""

import os
import json
import torch
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def convert_checkpoint_to_model(checkpoint_path, output_path):
    """Convert a training checkpoint to a loadable model."""
    
    print(f"🔄 Converting checkpoint: {checkpoint_path}")
    print(f"📁 Output location: {output_path}")
    
    try:
        from models import MultimodalLLM
        from training.config import ExperimentConfig
        from transformers import Qwen2Tokenizer, CLIPImageProcessor
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load the original training config
        config_path = "configs/coco_pretraining.json"
        if os.path.exists(config_path):
            config = ExperimentConfig.load(config_path)
            print("✅ Loaded training configuration")
        else:
            print("⚠️  Using default configuration")
            config = ExperimentConfig()
        
        # Initialize tokenizer and image processor
        print("🔄 Loading tokenizer and image processor...")
        tokenizer = Qwen2Tokenizer.from_pretrained(config.model.qwen_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(config.model.clip_model_name)
        
        # Add special tokens
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        print("🔄 Creating model architecture...")
        
        # Create the model with correct parameters
        model = MultimodalLLM(
            clip_model_name=config.model.clip_model_name,
            qwen_model_name=config.model.qwen_model_name,
            fusion_type=config.model.fusion_type,
            fusion_config=config.model.fusion_config,
            freeze_vision=config.model.freeze_vision,
            use_lora=config.model.use_lora,
            lora_config=config.model.lora_config
        )
        
        print("✅ Model architecture created")
        
        # Load the checkpoint weights
        checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(checkpoint_file):
            print(f"🔄 Loading weights from {checkpoint_file}...")
            
            # Load the state dict
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file)
            
            # Load weights into model
            model.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded successfully")
            
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_file}")
            return False
        
        # Save the model in the expected format
        print("🔄 Saving converted model...")
        
        # Save model state
        model.save_pretrained(output_path)
        
        # Save tokenizer and image processor
        tokenizer.save_pretrained(output_path)
        image_processor.save_pretrained(output_path)
        
        # Save config
        config_dict = {
            "model_type": "multimodal_llm",
            "clip_model_name": config.model.clip_model_name,
            "qwen_model_name": config.model.qwen_model_name,
            "fusion_type": config.model.fusion_type,
            "fusion_config": config.model.fusion_config
        }
        
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save a simple config.pt for compatibility
        torch.save(config_dict, os.path.join(output_path, "config.pt"))
        
        print("✅ Model conversion completed!")
        print(f"📁 Converted model saved to: {output_path}")
        
        # Verify the conversion worked
        print("🔍 Verifying converted model...")
        try:
            # Check if required files exist
            required_files = ["pytorch_model.bin", "config.json"]
            all_files_exist = True
            for file in required_files:
                file_path = os.path.join(output_path, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / (1024*1024)
                    print(f"✅ {file} exists ({size:.1f} MB)")
                else:
                    print(f"❌ {file} missing")
                    all_files_exist = False
            
            if all_files_exist:
                print("✅ Model verification successful!")
                return True
            else:
                print("❌ Some required files are missing")
                return False
                
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion function."""
    
    print("🚀 Checkpoint to Model Converter")
    print("=" * 40)
    
    # Define paths
    checkpoint_path = "outputs/coco_pretraining/checkpoints/stage1_step_153530"
    output_path = "outputs/coco_pretraining/converted_model"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("💡 Available checkpoints:")
        checkpoint_dir = "outputs/coco_pretraining/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("stage1_step_")]
            for cp in sorted(checkpoints)[-5:]:  # Show last 5
                print(f"   {cp}")
        return
    
    # Convert checkpoint
    success = convert_checkpoint_to_model(checkpoint_path, output_path)
    
    if success:
        print("\n" + "=" * 40)
        print("🎉 Conversion completed successfully!")
        print(f"✅ You can now test your model with:")
        print(f"   python scripts/test_cli.py --model {output_path} --image test_data/images/test_image_1.jpg")
    else:
        print("\n❌ Conversion failed. Check the error messages above.")

if __name__ == "__main__":
    main()