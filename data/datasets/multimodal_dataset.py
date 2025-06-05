import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path


class MultimodalDataset(Dataset):
    """
    Base dataset class for multimodal (image + text) data.
    Supports various dataset formats including COCO, CC3M, and custom formats.
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 512,
        image_size: tuple = (224, 224),
        dataset_type: str = "custom"  # "coco", "cc3m", "custom"
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        self.dataset_type = dataset_type
        
        # Load dataset
        self.data = self._load_data()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_data(self) -> List[Dict]:
        """Load dataset based on format."""
        if self.dataset_type == "coco":
            return self._load_coco_data()
        elif self.dataset_type == "cc3m":
            return self._load_cc3m_data()
        elif self.dataset_type == "custom":
            return self._load_custom_data()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _load_coco_data(self) -> List[Dict]:
        """Load COCO Captions dataset."""
        with open(self.data_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Process annotations
        data = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in images:
                data.append({
                    'image_path': os.path.join(self.image_dir, images[image_id]),
                    'caption': ann['caption'],
                    'image_id': image_id
                })
        
        return data
    
    def _load_cc3m_data(self) -> List[Dict]:
        """Load Conceptual Captions 3M dataset."""
        if self.data_path.endswith('.tsv'):
            df = pd.read_csv(self.data_path, sep='\t', header=None, names=['caption', 'url'])
        else:
            df = pd.read_csv(self.data_path)
        
        data = []
        for idx, row in df.iterrows():
            # Assume images are downloaded and named by index
            image_path = os.path.join(self.image_dir, f"{idx:08d}.jpg")
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'caption': row['caption'],
                    'image_id': idx
                })
        
        return data
    
    def _load_custom_data(self) -> List[Dict]:
        """Load custom dataset format."""
        with open(self.data_path, 'r') as f:
            if self.data_path.endswith('.json'):
                data = json.load(f)
            elif self.data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Custom data must be JSON or JSONL format")
        
        # Validate and process data
        processed_data = []
        for item in data:
            if 'image' in item and 'text' in item:
                image_path = os.path.join(self.image_dir, item['image'])
                if os.path.exists(image_path):
                    processed_data.append({
                        'image_path': image_path,
                        'caption': item['text'],
                        'image_id': item.get('id', len(processed_data))
                    })
        
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        item = self.data[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception as e:
            self.logger.warning(f"Error loading image {item['image_path']}: {e}")
            # Return a black image as fallback
            image_tensor = torch.zeros(3, *self.image_size)
        
        # Process text
        text = item['caption']
        
        # Add image placeholder
        formatted_text = f"<image>\n{text}"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone(),  # For causal LM
            'image_id': item['image_id'],
            'text': text
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following multimodal data.
    Supports conversation format with multiple turns.
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 1024,
        image_size: tuple = (224, 224)
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load conversation data
        self.conversations = self._load_conversations()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversation data."""
        with open(self.data_path, 'r') as f:
            if self.data_path.endswith('.json'):
                data = json.load(f)
            elif self.data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Data must be JSON or JSONL format")
        
        return data
    
    def _format_conversation(self, conversations: List[Dict]) -> str:
        """Format conversation turns into a single string."""
        formatted = []
        
        for turn in conversations:
            role = turn.get('from', 'user')
            content = turn.get('value', '')
            
            if role in ['human', 'user']:
                formatted.append(f"Human: {content}")
            elif role in ['assistant', 'gpt']:
                formatted.append(f"Assistant: {content}")
        
        return '\n'.join(formatted)
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single conversation sample."""
        item = self.conversations[idx]
        
        # Load image if present
        image_tensor = None
        if 'image' in item:
            image_path = os.path.join(self.image_dir, item['image'])
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            except Exception as e:
                self.logger.warning(f"Error loading image {image_path}: {e}")
                image_tensor = torch.zeros(3, *self.image_size)
        
        # Format conversation
        conversation_text = self._format_conversation(item['conversations'])
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone(),
            'conversation_id': item.get('id', idx)
        }
        
        if image_tensor is not None:
            result['image'] = image_tensor
        
        return result


class VQADataset(Dataset):
    """
    Dataset for Visual Question Answering tasks.
    """
    
    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        image_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 256,
        image_size: tuple = (224, 224)
    ):
        self.questions_path = questions_path
        self.annotations_path = annotations_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load VQA data
        self.data = self._load_vqa_data()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_vqa_data(self) -> List[Dict]:
        """Load VQA questions and annotations."""
        # Load questions
        with open(self.questions_path, 'r') as f:
            questions_data = json.load(f)
        
        # Load annotations
        annotations = {}
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, 'r') as f:
                annotations_data = json.load(f)
                annotations = {ann['question_id']: ann for ann in annotations_data['annotations']}
        
        # Process data
        data = []
        for question_item in questions_data['questions']:
            question_id = question_item['question_id']
            image_id = question_item['image_id']
            question = question_item['question']
            
            # Get annotation if available
            annotation = annotations.get(question_id, {})
            answer = annotation.get('multiple_choice_answer', '')
            
            # Format image filename (assuming COCO format)
            image_filename = f"COCO_val2014_{image_id:012d}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'question': question,
                    'answer': answer,
                    'question_id': question_id,
                    'image_id': image_id
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single VQA sample."""
        item = self.data[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception as e:
            self.logger.warning(f"Error loading image {item['image_path']}: {e}")
            image_tensor = torch.zeros(3, *self.image_size)
        
        # Format as QA
        question = item['question']
        answer = item['answer']
        
        # Create input and target
        input_text = f"<image>\nQuestion: {question}\nAnswer:"
        target_text = f"<image>\nQuestion: {question}\nAnswer: {answer}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'question_id': item['question_id'],
            'question': question,
            'answer': answer
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None  # Use default collate_fn
    )


if __name__ == "__main__":
    # Test dataset loading
    from transformers import Qwen2Tokenizer, CLIPImageProcessor
    
    # Initialize processors
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Add special tokens
    special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    print("Dataset classes loaded successfully!")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Image processor: {image_processor.__class__.__name__}")