#!/usr/bin/env python3
"""
Script to download and prepare datasets for multimodal training.

This script downloads:
1. COCO Captions dataset (for pre-training)
2. LLaVA Instruct dataset (for instruction tuning)
"""

import os
import json
import wget
import zipfile
import tarfile
from pathlib import Path
import argparse
from datasets import load_dataset

def create_directories():
    """Create necessary directories for datasets."""
    dirs = [
        "data/coco",
        "data/coco/images/train2014", 
        "data/coco/images/val2014",
        "data/coco/annotations",
        "data/llava",
        "data/processed"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def download_coco_captions():
    """Download COCO Captions dataset."""
    print("\n🔄 Downloading COCO Captions Dataset...")
    
    base_url = "http://images.cocodataset.org"
    annotation_url = "http://images.cocodataset.org/annotations"
    
    downloads = [
        {
            "url": f"{base_url}/zips/train2014.zip",
            "file": "data/coco/train2014.zip",
            "extract_to": "data/coco/images/",
            "description": "COCO Train Images (13GB)"
        },
        {
            "url": f"{base_url}/zips/val2014.zip", 
            "file": "data/coco/val2014.zip",
            "extract_to": "data/coco/images/",
            "description": "COCO Val Images (6GB)"
        },
        {
            "url": f"{annotation_url}/annotations_trainval2014.zip",
            "file": "data/coco/annotations.zip",
            "extract_to": "data/coco/",
            "description": "COCO Annotations (241MB)"
        }
    ]
    
    for item in downloads:
        print(f"\n📥 Downloading {item['description']}...")
        
        if os.path.exists(item['file']):
            print(f"✓ File already exists: {item['file']}")
            continue
            
        try:
            wget.download(item['url'], item['file'])
            print(f"\n✓ Downloaded: {item['file']}")
            
            # Extract files
            print(f"📦 Extracting to {item['extract_to']}...")
            if item['file'].endswith('.zip'):
                with zipfile.ZipFile(item['file'], 'r') as zip_ref:
                    zip_ref.extractall(item['extract_to'])
            
            print(f"✓ Extracted successfully")
            
        except Exception as e:
            print(f"❌ Error downloading {item['url']}: {e}")
            print("💡 You can manually download from: https://cocodataset.org/#download")

def download_llava_instruct():
    """Download LLaVA Instruct dataset from HuggingFace."""
    print("\n🔄 Downloading LLaVA Instruct Dataset...")
    
    try:
        # Download LLaVA instruction data
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        
        # Save to local file
        output_file = "data/llava/llava_instruct_150k.json"
        dataset.to_json(output_file)
        
        print(f"✓ Downloaded LLaVA Instruct dataset: {len(dataset)} samples")
        print(f"✓ Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error downloading LLaVA dataset: {e}")
        print("💡 Make sure you have 'datasets' library installed: pip install datasets")

def convert_coco_to_instruction_format():
    """Convert COCO captions to instruction format."""
    print("\n🔄 Converting COCO Captions to instruction format...")
    
    # Load COCO annotations
    train_ann_file = "data/coco/annotations/captions_train2014.json"
    val_ann_file = "data/coco/annotations/captions_val2014.json"
    
    if not os.path.exists(train_ann_file):
        print(f"❌ COCO annotations not found: {train_ann_file}")
        return
    
    def process_coco_split(ann_file, image_dir, output_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in data['images']}
        
        # Convert to instruction format
        instruction_data = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id in images:
                instruction_data.append({
                    "image": images[image_id],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nDescribe this image in detail."
                        },
                        {
                            "from": "assistant", 
                            "value": ann['caption']
                        }
                    ]
                })
        
        # Save processed data
        with open(output_file, 'w') as f:
            json.dump(instruction_data, f, indent=2)
        
        print(f"✓ Processed {len(instruction_data)} COCO samples → {output_file}")
        return len(instruction_data)
    
    # Process train and val splits
    train_count = process_coco_split(
        train_ann_file,
        "data/coco/images/train2014/",
        "data/processed/coco_train_instructions.json"
    )
    
    val_count = process_coco_split(
        val_ann_file, 
        "data/coco/images/val2014/",
        "data/processed/coco_val_instructions.json"
    )
    
    print(f"✅ COCO conversion complete: {train_count + val_count} total samples")

def create_training_configs():
    """Create training configurations for different scales."""
    print("\n🔄 Creating training configurations...")
    
    # COCO Pre-training config
    coco_config = {
        "experiment_name": "qwen_clip_coco_pretraining",
        "output_dir": "./outputs/coco_pretraining",
        "data": {
            "train_data_path": "data/processed/coco_train_instructions.json",
            "val_data_path": "data/processed/coco_val_instructions.json", 
            "image_dir": "data/coco/images/train2014/",
            "dataset_type": "instruction",
            "max_length": 512,
            "batch_size": 8,
            "num_workers": 4
        },
        "training": {
            "stage1_epochs": 3,
            "stage2_epochs": 2, 
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 4,
            "eval_steps": 1000,
            "save_steps": 2000,
            "logging_steps": 100
        },
        "model": {
            "freeze_vision": True,
            "use_lora": False
        }
    }
    
    # LLaVA Instruction tuning config
    llava_config = {
        "experiment_name": "qwen_clip_llava_instruction",
        "output_dir": "./outputs/llava_instruction",
        "data": {
            "train_data_path": "data/llava/llava_instruct_150k.json",
            "val_data_path": "data/processed/coco_val_instructions.json",
            "image_dir": "data/coco/images/",
            "dataset_type": "instruction", 
            "max_length": 1024,
            "batch_size": 4,
            "num_workers": 4
        },
        "training": {
            "stage1_epochs": 1,
            "stage2_epochs": 2,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 8,
            "eval_steps": 500,
            "save_steps": 1000,
            "logging_steps": 50
        },
        "model": {
            "freeze_vision": False,
            "use_lora": True
        }
    }
    
    # Save configs
    with open("configs/coco_pretraining.json", "w") as f:
        json.dump(coco_config, f, indent=2)
    
    with open("configs/llava_instruction.json", "w") as f:
        json.dump(llava_config, f, indent=2)
    
    print("✓ Created configs/coco_pretraining.json")
    print("✓ Created configs/llava_instruction.json")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare multimodal datasets")
    parser.add_argument("--coco", action="store_true", help="Download COCO Captions dataset")
    parser.add_argument("--llava", action="store_true", help="Download LLaVA Instruct dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--convert", action="store_true", help="Convert datasets to instruction format")
    parser.add_argument("--configs", action="store_true", help="Create training configurations")
    
    args = parser.parse_args()
    
    print("🤖 Qwen-CLIP Multimodal Dataset Preparation")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    if args.all or args.coco:
        download_coco_captions()
    
    if args.all or args.llava:
        download_llava_instruct()
    
    if args.all or args.convert:
        convert_coco_to_instruction_format()
    
    if args.all or args.configs:
        create_training_configs()
    
    print("\n✅ Dataset preparation complete!")
    print("\n📋 Next steps:")
    print("1. Run COCO pre-training: python examples/train_model.py --config configs/coco_pretraining.json")
    print("2. Run LLaVA instruction tuning: python examples/train_model.py --config configs/llava_instruction.json")

if __name__ == "__main__":
    main()