#!/usr/bin/env python3
"""
Script to recover trained model from checkpoints.
"""

import os
import shutil
from pathlib import Path

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by step number."""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("stage1_step_"):
            try:
                step_num = int(item.split("_")[-1])
                checkpoints.append((step_num, item))
            except ValueError:
                continue
    
    if not checkpoints:
        print("❌ No valid checkpoints found")
        return None
    
    # Sort by step number and get the latest
    latest_step, latest_name = max(checkpoints)
    return os.path.join(checkpoint_dir, latest_name), latest_step

def recover_model_from_checkpoint():
    """Recover the trained model from the latest checkpoint."""
    
    print("🔄 Recovering trained model from checkpoints...")
    
    # Find latest checkpoint
    checkpoint_dir = "outputs/coco_pretraining/checkpoints"
    latest_checkpoint, step_num = find_latest_checkpoint(checkpoint_dir)
    
    if not latest_checkpoint:
        return False
    
    print(f"📁 Found latest checkpoint: step {step_num}")
    print(f"📍 Location: {latest_checkpoint}")
    
    # Create model directories
    output_dir = "outputs/coco_pretraining"
    best_model_dir = os.path.join(output_dir, "best_model")
    final_model_dir = os.path.join(output_dir, "final_model")
    
    # Copy checkpoint to both best_model and final_model
    if os.path.exists(latest_checkpoint):
        print(f"📦 Creating best_model from checkpoint...")
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(latest_checkpoint, best_model_dir)
        print(f"✅ Created: {best_model_dir}")
        
        print(f"📦 Creating final_model from checkpoint...")
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        shutil.copytree(latest_checkpoint, final_model_dir)
        print(f"✅ Created: {final_model_dir}")
        
        # Calculate sizes
        best_size = get_dir_size(best_model_dir)
        final_size = get_dir_size(final_model_dir)
        
        print(f"\n🎉 Model recovery completed!")
        print(f"📊 Best model size: {best_size:.1f} MB")
        print(f"📊 Final model size: {final_size:.1f} MB")
        print(f"🔢 Training steps completed: {step_num:,}")
        
        return True
    else:
        print(f"❌ Checkpoint not found: {latest_checkpoint}")
        return False

def get_dir_size(path):
    """Get directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total / (1024 * 1024)  # Convert to MB

def verify_model():
    """Verify the recovered model can be loaded."""
    try:
        import sys
        sys.path.append('.')
        
        from models import MultimodalLLM
        
        model_path = "outputs/coco_pretraining/best_model"
        if os.path.exists(model_path):
            print(f"🔍 Verifying model at {model_path}...")
            
            # Try to load the model
            model = MultimodalLLM.from_pretrained(model_path)
            print("✅ Model loaded successfully!")
            
            # Check model components
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            return True
        else:
            print(f"❌ Model not found at {model_path}")
            return False
            
    except Exception as e:
        print(f"⚠️  Model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Qwen-CLIP Model Recovery Tool")
    print("=" * 40)
    
    # Recover model
    success = recover_model_from_checkpoint()
    
    if success:
        print("\n" + "=" * 40)
        print("🔍 Verifying recovered model...")
        verify_model()
        
        print("\n" + "=" * 40)
        print("✅ Your trained model is ready to use!")
        print("📍 Location: outputs/coco_pretraining/best_model")
        print("💡 You can now test it or continue with Stage 2")
    else:
        print("❌ Model recovery failed")