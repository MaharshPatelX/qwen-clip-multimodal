import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import numpy as np


class CLIPVisionEncoder(nn.Module):
    """
    CLIP Vision Encoder wrapper for extracting visual features from images.
    Uses the pre-trained CLIP ViT-B/32 model from OpenAI.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_encoder=True):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Freeze the vision encoder if specified
        if freeze_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
                
        # Get the output dimension (512 for ViT-B/32)
        self.output_dim = self.vision_model.config.hidden_size
        
    def forward(self, images):
        """
        Forward pass through CLIP vision encoder.
        
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224) or preprocessed pixel values
            
        Returns:
            vision_features: Tensor of shape (batch_size, hidden_size)
        """
        if self.freeze_encoder:
            with torch.no_grad():
                vision_outputs = self.vision_model(pixel_values=images)
        else:
            vision_outputs = self.vision_model(pixel_values=images)
            
        # Get the pooled features (CLS token representation)
        vision_features = vision_outputs.pooler_output
        
        return vision_features
    
    def preprocess_images(self, images):
        """
        Preprocess PIL images or numpy arrays for CLIP.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            pixel_values: Tensor ready for CLIP processing
        """
        if isinstance(images, (list, tuple)):
            # Process multiple images
            processed = self.image_processor(images, return_tensors="pt")
        else:
            # Single image
            processed = self.image_processor(images, return_tensors="pt")
            
        return processed["pixel_values"]
    
    def encode_image(self, image_path_or_pil):
        """
        Convenience method to encode a single image from path or PIL Image.
        
        Args:
            image_path_or_pil: Either a file path string or PIL Image
            
        Returns:
            features: Tensor of shape (1, hidden_size)
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil
            
        pixel_values = self.preprocess_images(image)
        
        # Move to same device as model
        if next(self.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()
            
        with torch.no_grad():
            features = self.forward(pixel_values)
            
        return features
    
    def unfreeze_encoder(self):
        """Unfreeze the vision encoder for fine-tuning."""
        self.freeze_encoder = False
        for param in self.vision_model.parameters():
            param.requires_grad = True
            
    def freeze_encoder_layers(self, freeze_layers=True):
        """Freeze/unfreeze the vision encoder."""
        self.freeze_encoder = freeze_layers
        for param in self.vision_model.parameters():
            param.requires_grad = not freeze_layers


if __name__ == "__main__":
    # Test the CLIP encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize encoder
    encoder = CLIPVisionEncoder().to(device)
    print(f"CLIP output dimension: {encoder.output_dim}")
    
    # Test with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    features = encoder(dummy_images)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature range: [{features.min().item():.3f}, {features.max().item():.3f}]")