import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
from PIL import Image
import warnings

from .clip_encoder import CLIPVisionEncoder
from .qwen_decoder import QwenLanguageDecoder
from .fusion_module import VisionLanguageFusion, AttentionFusion, AdaptiveFusion


class MultimodalLLM(nn.Module):
    """
    Complete multimodal language model combining CLIP vision encoder,
    Qwen language decoder, and vision-language fusion module.
    """
    
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        fusion_type="mlp",  # "mlp", "attention", "adaptive"
        fusion_config=None,
        freeze_vision=True,
        use_lora=False,
        lora_config=None,
        device_map="auto"
    ):
        super().__init__()
        
        self.clip_model_name = clip_model_name
        self.qwen_model_name = qwen_model_name
        self.fusion_type = fusion_type
        self.freeze_vision = freeze_vision
        
        # Initialize vision encoder
        self.vision_encoder = CLIPVisionEncoder(
            model_name=clip_model_name,
            freeze_encoder=freeze_vision
        )
        
        # Initialize language decoder
        self.language_decoder = QwenLanguageDecoder(
            model_name=qwen_model_name,
            use_lora=use_lora,
            lora_config=lora_config,
            device_map=device_map
        )
        
        # Get dimensions
        vision_dim = self.vision_encoder.output_dim
        language_dim = self.language_decoder.hidden_size
        
        # Initialize fusion module
        if fusion_config is None:
            fusion_config = {}
            
        if fusion_type == "mlp":
            self.fusion_module = VisionLanguageFusion(
                vision_dim=vision_dim,
                language_dim=language_dim,
                **fusion_config
            )
        elif fusion_type == "attention":
            self.fusion_module = AttentionFusion(
                vision_dim=vision_dim,
                language_dim=language_dim,
                **fusion_config
            )
        elif fusion_type == "adaptive":
            self.fusion_module = AdaptiveFusion(
                vision_dim=vision_dim,
                language_dim=language_dim,
                **fusion_config
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Get special token IDs
        self.image_token_id = self.language_decoder.tokenizer.convert_tokens_to_ids("<image>")
        self.pad_token_id = self.language_decoder.tokenizer.pad_token_id
        
        # Set model dimensions
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        
    def forward(
        self,
        images=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True
    ):
        """
        Forward pass through the multimodal model.
        
        Args:
            images: Preprocessed images (batch_size, 3, 224, 224)
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs with loss, logits, etc.
        """
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)
        
        # Encode images if provided
        vision_features = None
        if images is not None:
            vision_features = self.vision_encoder(images)  # (batch_size, vision_dim)
            
            # Project to language space
            projected_vision = self.fusion_module(vision_features)  # (batch_size, language_dim)
        
        # Process text input and inject vision features
        if input_ids is not None and vision_features is not None:
            # Get text embeddings
            text_embeddings = self.language_decoder.get_embeddings(input_ids)
            
            # Find image token positions
            image_positions = (input_ids == self.image_token_id)
            
            # Replace image tokens with projected vision features
            if image_positions.any():
                # Expand vision features to match sequence positions
                expanded_vision = projected_vision.unsqueeze(1).expand(-1, input_ids.size(1), -1)
                
                # Replace image token embeddings with vision features
                text_embeddings = torch.where(
                    image_positions.unsqueeze(-1).expand_as(text_embeddings),
                    expanded_vision,
                    text_embeddings
                )
            
            # Forward pass through language model with modified embeddings
            outputs = self.language_decoder.model(
                inputs_embeds=text_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=return_dict
            )
        else:
            # Text-only forward pass
            outputs = self.language_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        return outputs
    
    def generate(
        self,
        images=None,
        input_text=None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=50,
        do_sample=True,
        **kwargs
    ):
        """
        Generate text responses given images and/or text input.
        
        Args:
            images: PIL Images or preprocessed tensors
            input_text: Input text string
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            generated_text: Generated text string or list of strings
        """
        self.eval()
        
        with torch.no_grad():
            # Preprocess images if provided
            if images is not None:
                if isinstance(images, (list, tuple)):
                    if isinstance(images[0], Image.Image):
                        # PIL Images
                        pixel_values = self.vision_encoder.preprocess_images(images)
                    else:
                        # Already preprocessed
                        pixel_values = torch.stack(images) if isinstance(images, list) else images
                elif isinstance(images, Image.Image):
                    # Single PIL Image
                    pixel_values = self.vision_encoder.preprocess_images(images)
                else:
                    # Already preprocessed tensor
                    pixel_values = images
                
                # Move to model device
                device = next(self.parameters()).device
                pixel_values = pixel_values.to(device)
                
                # Encode vision features
                vision_features = self.vision_encoder(pixel_values)
                projected_vision = self.fusion_module(vision_features)
            else:
                projected_vision = None
            
            # Prepare text input
            if input_ids is None and input_text is not None:
                # Add image placeholder if not present and we have images
                if images is not None and "<image>" not in input_text:
                    input_text = f"<image>\n{input_text}"
                
                # Tokenize
                encoded = self.language_decoder.encode_text(input_text)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
            
            # Move to device
            if input_ids is not None:
                device = next(self.parameters()).device
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            
            # Prepare inputs with vision features if available
            if projected_vision is not None and input_ids is not None:
                # Get text embeddings
                text_embeddings = self.language_decoder.get_embeddings(input_ids)
                
                # Replace image tokens with vision features
                image_positions = (input_ids == self.image_token_id)
                if image_positions.any():
                    expanded_vision = projected_vision.unsqueeze(1).expand(-1, input_ids.size(1), -1)
                    text_embeddings = torch.where(
                        image_positions.unsqueeze(-1).expand_as(text_embeddings),
                        expanded_vision,
                        text_embeddings
                    )
                
                # Generate with modified embeddings
                generated_ids = self.language_decoder.model.generate(
                    inputs_embeds=text_embeddings,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.pad_token_id,
                    **kwargs
                )
            else:
                # Text-only generation
                generated_ids = self.language_decoder.generate_text(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    **kwargs
                )
            
            # Decode generated text
            if input_ids is not None:
                # Remove input tokens from generated output
                new_tokens = generated_ids[:, input_ids.size(1):]
                generated_text = self.language_decoder.decode_text(new_tokens)
            else:
                generated_text = self.language_decoder.decode_text(generated_ids)
        
        return generated_text
    
    def caption_image(self, image, prompt="Describe this image in detail."):
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image or preprocessed tensor
            prompt: Caption generation prompt
            
        Returns:
            caption: Generated image caption
        """
        input_text = f"<image>\n{prompt}"
        return self.generate(images=image, input_text=input_text, max_new_tokens=256)
    
    def answer_question(self, image, question):
        """
        Answer a question about an image.
        
        Args:
            image: PIL Image or preprocessed tensor
            question: Question about the image
            
        Returns:
            answer: Generated answer
        """
        input_text = f"<image>\nQuestion: {question}\nAnswer:"
        return self.generate(images=image, input_text=input_text, max_new_tokens=128)
    
    def chat(self, image=None, message="", history=None):
        """
        Multimodal chat interface.
        
        Args:
            image: Optional PIL Image
            message: User message
            history: Previous conversation history
            
        Returns:
            response: Model response
            updated_history: Updated conversation history
        """
        if history is None:
            history = []
        
        # Format input with conversation history
        if image is not None:
            current_input = f"<image>\n{message}"
        else:
            current_input = message
        
        # Add conversation context
        if history:
            conversation = "\n".join(history) + f"\nHuman: {current_input}\nAssistant:"
        else:
            conversation = f"Human: {current_input}\nAssistant:"
        
        # Generate response
        response = self.generate(
            images=image,
            input_text=conversation,
            max_new_tokens=256,
            temperature=0.7
        )
        
        # Update history
        history.append(f"Human: {message}")
        history.append(f"Assistant: {response}")
        
        return response, history
    
    def save_pretrained(self, save_directory):
        """Save the model components."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save language model
        self.language_decoder.model.save_pretrained(
            os.path.join(save_directory, "language_model")
        )
        self.language_decoder.tokenizer.save_pretrained(
            os.path.join(save_directory, "language_model")
        )
        
        # Save fusion module
        torch.save(
            self.fusion_module.state_dict(),
            os.path.join(save_directory, "fusion_module.pt")
        )
        
        # Save configuration
        config = {
            "clip_model_name": self.clip_model_name,
            "qwen_model_name": self.qwen_model_name,
            "fusion_type": self.fusion_type,
            "freeze_vision": self.freeze_vision,
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        
    @classmethod
    def from_pretrained(cls, load_directory):
        """Load a saved model."""
        import os
        
        # Load configuration
        config = torch.load(os.path.join(load_directory, "config.pt"))
        
        # Initialize model
        model = cls(**config)
        
        # Load fusion module
        fusion_state = torch.load(os.path.join(load_directory, "fusion_module.pt"))
        model.fusion_module.load_state_dict(fusion_state)
        
        return model


if __name__ == "__main__":
    # Test the complete multimodal model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalLLM(
        fusion_type="mlp",
        freeze_vision=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy inputs
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_text = ["<image> What is in this image?", "<image> Describe what you see."]
    
    # Test forward pass
    print("\nTesting forward pass...")
    encoded = model.language_decoder.encode_text(dummy_text)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(
            images=dummy_images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print(f"Loss: {outputs.loss}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        generated = model.generate(
            images=dummy_images[:1],
            input_text="<image> What do you see in this image?",
            max_new_tokens=50
        )
    
    print(f"Generated text: {generated}")