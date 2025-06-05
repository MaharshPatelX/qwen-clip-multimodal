import os
import json
import requests
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib


class DataDownloader:
    """Utility class for downloading and preparing multimodal datasets."""
    
    def __init__(self, base_dir: str = "./data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_coco_captions(self, year: str = "2017", split: str = "train"):
        """
        Download COCO Captions dataset.
        
        Args:
            year: Dataset year (2014, 2017)
            split: Dataset split (train, val)
        """
        coco_dir = self.base_dir / "coco" / year
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for COCO dataset
        base_url = f"http://images.cocodataset.org/zips"
        annotations_url = f"http://images.cocodataset.org/annotations/annotations_trainval{year}.zip"
        
        if year == "2017":
            images_url = f"{base_url}/{split}{year}.zip"
        else:
            images_url = f"{base_url}/{split}{year}.zip"
        
        # Download annotations
        annotations_file = coco_dir / f"annotations_trainval{year}.zip"
        if not annotations_file.exists():
            self.logger.info(f"Downloading COCO {year} annotations...")
            self._download_file(annotations_url, annotations_file)
        
        # Download images
        images_file = coco_dir / f"{split}{year}.zip"
        if not images_file.exists():
            self.logger.info(f"Downloading COCO {year} {split} images...")
            self._download_file(images_url, images_file)
        
        self.logger.info(f"COCO {year} {split} dataset downloaded to {coco_dir}")
    
    def download_cc3m_subset(self, num_samples: int = 10000):
        """
        Download a subset of Conceptual Captions 3M dataset.
        
        Args:
            num_samples: Number of samples to download
        """
        cc3m_dir = self.base_dir / "cc3m"
        cc3m_dir.mkdir(parents=True, exist_ok=True)
        
        # Download CC3M TSV file (this is a placeholder - actual URL needs to be obtained)
        tsv_url = "https://ai.google.com/research/ConceptualCaptions/download"
        self.logger.info("Note: CC3M dataset requires manual download from Google AI")
        self.logger.info(f"Please download the TSV file to {cc3m_dir / 'cc3m.tsv'}")
        
        # If TSV exists, download images
        tsv_file = cc3m_dir / "cc3m.tsv"
        if tsv_file.exists():
            self._download_cc3m_images(tsv_file, cc3m_dir / "images", num_samples)
    
    def _download_cc3m_images(self, tsv_file: Path, image_dir: Path, num_samples: int):
        """Download CC3M images from URLs in TSV file."""
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Read TSV file
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['caption', 'url'])
        df = df.head(num_samples)
        
        self.logger.info(f"Downloading {len(df)} CC3M images...")
        
        def download_image(row):
            idx, caption, url = row[0], row[1], row[2]
            image_path = image_dir / f"{idx:08d}.jpg"
            
            if image_path.exists():
                return f"Skipped {idx}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Verify it's an image
                img = Image.open(response.content)
                img = img.convert('RGB')
                img.save(image_path, 'JPEG')
                
                return f"Downloaded {idx}"
            except Exception as e:
                return f"Failed {idx}: {e}"
        
        # Download with threading
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_image, row) for row in df.itertuples()]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if "Failed" in result:
                    self.logger.warning(result)
    
    def _download_file(self, url: str, filepath: Path):
        """Download a file from URL with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


class DatasetConverter:
    """Convert datasets to standardized format for training."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def convert_coco_to_standard(self, split: str = "train"):
        """
        Convert COCO dataset to standard format.
        
        Standard format:
        {
            "image": "image_filename",
            "text": "caption text",
            "id": "unique_id",
            "source": "coco"
        }
        """
        # Load COCO annotations
        ann_file = self.input_dir / "annotations" / f"captions_{split}2017.json"
        
        if not ann_file.exists():
            self.logger.error(f"COCO annotations not found: {ann_file}")
            return
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Convert annotations
        converted_data = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in images:
                converted_data.append({
                    "image": images[image_id],
                    "text": ann['caption'],
                    "id": f"coco_{ann['id']}",
                    "source": "coco"
                })
        
        # Save converted data
        output_file = self.output_dir / f"coco_{split}_standard.json"
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        self.logger.info(f"Converted {len(converted_data)} COCO {split} samples to {output_file}")
    
    def convert_cc3m_to_standard(self, tsv_file: str):
        """Convert CC3M dataset to standard format."""
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=['caption', 'url'])
        
        converted_data = []
        for idx, row in df.iterrows():
            converted_data.append({
                "image": f"{idx:08d}.jpg",
                "text": row['caption'],
                "id": f"cc3m_{idx}",
                "source": "cc3m"
            })
        
        # Save converted data
        output_file = self.output_dir / "cc3m_standard.json"
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        self.logger.info(f"Converted {len(converted_data)} CC3M samples to {output_file}")
    
    def create_instruction_dataset(self, caption_files: List[str], output_file: str):
        """
        Create instruction-following dataset from caption data.
        
        Converts captions to instruction-response pairs for better training.
        """
        instruction_templates = [
            "Describe this image in detail.",
            "What do you see in this image?",
            "Can you tell me what's happening in this picture?",
            "Please provide a description of this image.",
            "What is shown in this image?",
            "Describe the contents of this image.",
            "What can you observe in this picture?",
            "Please explain what you see in this image."
        ]
        
        all_data = []
        
        # Load all caption files
        for caption_file in caption_files:
            with open(caption_file, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        
        # Convert to instruction format
        instruction_data = []
        for idx, item in enumerate(all_data):
            # Random instruction template
            import random
            instruction = random.choice(instruction_templates)
            
            conversation = {
                "id": f"instruction_{idx}",
                "image": item["image"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{instruction}"
                    },
                    {
                        "from": "assistant",
                        "value": item["text"]
                    }
                ]
            }
            instruction_data.append(conversation)
        
        # Save instruction dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(instruction_data, f, indent=2)
        
        self.logger.info(f"Created {len(instruction_data)} instruction samples in {output_path}")


class DataValidator:
    """Validate dataset integrity and quality."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, annotation_file: str, image_dir: str) -> Dict[str, Any]:
        """
        Validate dataset by checking:
        1. All referenced images exist
        2. Images can be loaded
        3. Text quality (length, encoding)
        4. Dataset statistics
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        stats = {
            "total_samples": len(data),
            "missing_images": 0,
            "corrupted_images": 0,
            "empty_captions": 0,
            "valid_samples": 0,
            "avg_caption_length": 0,
            "min_caption_length": float('inf'),
            "max_caption_length": 0
        }
        
        caption_lengths = []
        
        for item in tqdm(data, desc="Validating dataset"):
            image_path = Path(image_dir) / item["image"]
            caption = item["text"]
            
            # Check image exists
            if not image_path.exists():
                stats["missing_images"] += 1
                continue
            
            # Check image can be loaded
            try:
                img = Image.open(image_path)
                img.verify()
            except Exception:
                stats["corrupted_images"] += 1
                continue
            
            # Check caption
            if not caption or len(caption.strip()) == 0:
                stats["empty_captions"] += 1
                continue
            
            caption_len = len(caption.split())
            caption_lengths.append(caption_len)
            
            stats["valid_samples"] += 1
        
        # Calculate statistics
        if caption_lengths:
            stats["avg_caption_length"] = sum(caption_lengths) / len(caption_lengths)
            stats["min_caption_length"] = min(caption_lengths)
            stats["max_caption_length"] = max(caption_lengths)
        
        # Calculate percentages
        stats["valid_percentage"] = (stats["valid_samples"] / stats["total_samples"]) * 100
        
        self.logger.info(f"Dataset validation complete:")
        self.logger.info(f"  Total samples: {stats['total_samples']}")
        self.logger.info(f"  Valid samples: {stats['valid_samples']} ({stats['valid_percentage']:.1f}%)")
        self.logger.info(f"  Missing images: {stats['missing_images']}")
        self.logger.info(f"  Corrupted images: {stats['corrupted_images']}")
        self.logger.info(f"  Empty captions: {stats['empty_captions']}")
        self.logger.info(f"  Avg caption length: {stats['avg_caption_length']:.1f} words")
        
        return stats
    
    def create_clean_dataset(self, annotation_file: str, image_dir: str, output_file: str):
        """Create a cleaned version of the dataset with only valid samples."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        clean_data = []
        
        for item in tqdm(data, desc="Cleaning dataset"):
            image_path = Path(image_dir) / item["image"]
            caption = item["text"]
            
            # Validate image
            if not image_path.exists():
                continue
            
            try:
                img = Image.open(image_path)
                img.verify()
            except Exception:
                continue
            
            # Validate caption
            if not caption or len(caption.strip()) == 0:
                continue
            
            # Additional quality checks
            word_count = len(caption.split())
            if word_count < 3 or word_count > 100:  # Filter very short/long captions
                continue
            
            clean_data.append(item)
        
        # Save clean dataset
        with open(output_file, 'w') as f:
            json.dump(clean_data, f, indent=2)
        
        self.logger.info(f"Created clean dataset with {len(clean_data)} samples in {output_file}")
        
        return len(clean_data)


if __name__ == "__main__":
    # Test data utilities
    base_dir = "./data/raw"
    
    # Initialize downloader
    downloader = DataDownloader(base_dir)
    print("DataDownloader initialized")
    
    # Initialize converter
    converter = DatasetConverter("./data/raw", "./data/processed")
    print("DatasetConverter initialized")
    
    # Initialize validator
    validator = DataValidator("./data/processed")
    print("DataValidator initialized")
    
    print("All data utilities loaded successfully!")