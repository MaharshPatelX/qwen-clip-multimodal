import torch
import torch.nn as nn
from transformers import (
    Qwen2ForCausalLM, 
    Qwen2Tokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import warnings


class QwenLanguageDecoder(nn.Module):
    """
    Qwen2.5 Language Model wrapper for text generation.
    Supports both full fine-tuning and LoRA efficient training.
    """
    
    def __init__(
        self, 
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        use_lora=False,
        lora_config=None,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map="auto"
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        
        # Add special tokens for multimodal inputs
        special_tokens = ["<image>", "<|im_start|>", "<|im_end|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Load model
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if quantization_config else torch.float32
        )
        
        # Resize token embeddings to accommodate new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA if requested
        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                                  "gate_proj", "up_proj", "down_proj"]
                )
            self.model = get_peft_model(self.model, lora_config)
            
        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None, vision_features=None):
        """
        Forward pass through Qwen model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            vision_features: Optional vision features to inject (batch_size, vision_dim)
            
        Returns:
            Model outputs with loss, logits, etc.
        """
        # If vision features are provided, we'll inject them via the fusion module
        # This is handled in the main multimodal model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_text(
        self, 
        input_ids, 
        attention_mask=None, 
        generation_config=None,
        **kwargs
    ):
        """
        Generate text using the language model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            generation_config: Custom generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            generated_ids: Generated token sequences
        """
        if generation_config is None:
            generation_config = self.generation_config
            
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs
            )
            
        return generated_ids
    
    def encode_text(self, text, max_length=512, return_tensors="pt"):
        """
        Tokenize and encode text input.
        
        Args:
            text: Input text string or list of strings
            max_length: Maximum sequence length
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Encoded inputs ready for model
        """
        if isinstance(text, str):
            text = [text]
            
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def decode_text(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string or list of strings
        """
        if len(token_ids.shape) == 1:
            # Single sequence
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            # Batch of sequences
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_embeddings(self, input_ids):
        """
        Get token embeddings from the model.
        
        Args:
            input_ids: Token IDs
            
        Returns:
            embeddings: Token embeddings (batch_size, seq_len, hidden_size)
        """
        embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings
    
    def prepare_multimodal_input(self, text, image_placeholder="<image>"):
        """
        Prepare text input with image placeholders for multimodal processing.
        
        Args:
            text: Input text with image placeholders
            image_placeholder: Token to use for image positions
            
        Returns:
            processed_text: Text ready for tokenization
        """
        # Format the input with proper special tokens
        if image_placeholder in text:
            # Replace image placeholder with our special token
            processed_text = text.replace(image_placeholder, "<image>")
        else:
            # Add image token at the beginning if not present
            processed_text = f"<image>\n{text}"
            
        return processed_text
    
    def enable_training_mode(self):
        """Enable training mode for the model."""
        self.model.train()
        if self.use_lora:
            # Enable gradient computation for LoRA parameters
            self.model.enable_adapters()
    
    def enable_inference_mode(self):
        """Enable inference mode for the model."""
        self.model.eval()
        if self.use_lora:
            # Merge LoRA weights for faster inference
            self.model.merge_and_unload()


if __name__ == "__main__":
    # Test the Qwen decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize decoder
    decoder = QwenLanguageDecoder()
    print(f"Qwen hidden size: {decoder.hidden_size}")
    print(f"Vocabulary size: {decoder.vocab_size}")
    
    # Test text encoding
    test_text = "What do you see in this image?"
    encoded = decoder.encode_text(test_text)
    print(f"Input text: {test_text}")
    print(f"Encoded shape: {encoded['input_ids'].shape}")
    
    # Test text generation
    with torch.no_grad():
        generated = decoder.generate_text(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_new_tokens=50
        )
    
    decoded = decoder.decode_text(generated[0])
    print(f"Generated text: {decoded}")