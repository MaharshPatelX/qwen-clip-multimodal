#!/usr/bin/env python3
"""
Script to backup trained models with metadata.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def create_model_backup():
    """Create a comprehensive backup of trained models."""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"model_backups/backup_{timestamp}"
    
    print(f"🔄 Creating model backup: {backup_dir}")
    
    # Create backup directory
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    
    # Backup models
    models_to_backup = [
        ("outputs/coco_pretraining/best_model", "stage1_best_model"),
        ("outputs/coco_pretraining/final_model", "stage1_final_model"),
        ("outputs/llava_instruction/best_model", "stage2_best_model"),
        ("outputs/llava_instruction/final_model", "stage2_final_model"),
    ]
    
    backed_up = []
    for src, dst in models_to_backup:
        if os.path.exists(src):
            dst_path = os.path.join(backup_dir, dst)
            shutil.copytree(src, dst_path)
            size = get_dir_size(dst_path)
            print(f"✅ Backed up {src} → {dst} ({size:.1f} MB)")
            backed_up.append({"source": src, "destination": dst, "size_mb": size})
        else:
            print(f"⚠️  Skipping {src} (not found)")
    
    # Backup configs
    config_backup = os.path.join(backup_dir, "configs")
    if os.path.exists("configs"):
        shutil.copytree("configs", config_backup)
        print(f"✅ Backed up configs")
    
    # Backup key scripts
    scripts_backup = os.path.join(backup_dir, "scripts")
    os.makedirs(scripts_backup, exist_ok=True)
    
    key_files = [
        "examples/train_model.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in key_files:
        if os.path.exists(file):
            shutil.copy2(file, scripts_backup)
    
    # Create backup metadata
    metadata = {
        "backup_timestamp": timestamp,
        "backup_date": datetime.now().isoformat(),
        "models_backed_up": backed_up,
        "total_size_mb": sum(item["size_mb"] for item in backed_up),
        "git_commit": get_git_commit(),
        "training_completed": datetime.now().isoformat()
    }
    
    with open(os.path.join(backup_dir, "backup_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Backup completed!")
    print(f"📁 Location: {backup_dir}")
    print(f"📊 Total size: {metadata['total_size_mb']:.1f} MB")
    print(f"🔢 Models backed up: {len(backed_up)}")
    
    return backup_dir

def get_dir_size(path):
    """Get directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total / (1024 * 1024)  # Convert to MB

def get_git_commit():
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def list_backups():
    """List all available backups."""
    backup_dir = "model_backups"
    if not os.path.exists(backup_dir):
        print("No backups found.")
        return
    
    backups = []
    for item in os.listdir(backup_dir):
        backup_path = os.path.join(backup_dir, item)
        if os.path.isdir(backup_path):
            info_file = os.path.join(backup_path, "backup_info.json")
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    info = json.load(f)
                backups.append((item, info))
    
    if not backups:
        print("No valid backups found.")
        return
    
    print("📦 Available Model Backups:")
    print("-" * 60)
    for name, info in sorted(backups, reverse=True):
        print(f"📁 {name}")
        print(f"   📅 Date: {info['backup_date'][:19]}")
        print(f"   📊 Size: {info['total_size_mb']:.1f} MB")
        print(f"   🔢 Models: {len(info['models_backed_up'])}")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup trained models")
    parser.add_argument("--backup", action="store_true", help="Create new backup")
    parser.add_argument("--list", action="store_true", help="List existing backups")
    
    args = parser.parse_args()
    
    if args.list:
        list_backups()
    elif args.backup:
        create_model_backup()
    else:
        print("Use --backup to create backup or --list to see existing backups")