import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisionLanguageFusion(nn.Module):
    """
    Fusion module that projects CLIP vision features to Qwen embedding space.
    Uses a multi-layer perceptron with layer normalization and dropout.
    """
    
    def __init__(
        self,
        vision_dim=512,  # CLIP ViT-B/32 output dimension
        language_dim=896,  # Qwen2.5-0.5B hidden dimension
        hidden_dim=1024,  # Hidden dimension for MLP
        num_layers=2,
        dropout=0.1,
        activation="relu"
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build the MLP layers
        layers = []
        
        if num_layers == 1:
            # Single layer projection
            layers.append(nn.Linear(vision_dim, language_dim))
        else:
            # Multi-layer MLP
            # Input layer
            layers.append(nn.Linear(vision_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(dropout))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, language_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, vision_features):
        """
        Project vision features to language embedding space.
        
        Args:
            vision_features: Tensor of shape (batch_size, vision_dim)
            
        Returns:
            projected_features: Tensor of shape (batch_size, language_dim)
        """
        projected_features = self.projection(vision_features)
        return projected_features


class AttentionFusion(nn.Module):
    """
    Alternative fusion module using cross-attention mechanism.
    Allows more sophisticated interaction between vision and language features.
    """
    
    def __init__(
        self,
        vision_dim=512,
        language_dim=896,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_heads = num_heads
        
        # Project vision features to language dimension
        self.vision_proj = nn.Linear(vision_dim, language_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=language_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and feedforward
        self.layer_norm = nn.LayerNorm(language_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(language_dim, language_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim * 4, language_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, vision_features, language_features=None):
        """
        Fuse vision and language features using cross-attention.
        
        Args:
            vision_features: Tensor of shape (batch_size, vision_dim)
            language_features: Optional language context (batch_size, seq_len, language_dim)
            
        Returns:
            fused_features: Tensor of shape (batch_size, language_dim)
        """
        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # (batch_size, language_dim)
        vision_proj = vision_proj.unsqueeze(1)  # (batch_size, 1, language_dim)
        
        if language_features is not None:
            # Cross-attention between vision and language
            attended_features, _ = self.cross_attention(
                query=vision_proj,
                key=language_features,
                value=language_features
            )
        else:
            # Self-attention on vision features
            attended_features, _ = self.cross_attention(
                query=vision_proj,
                key=vision_proj,
                value=vision_proj
            )
        
        # Residual connection and layer norm
        fused = self.layer_norm(vision_proj + attended_features)
        
        # Feedforward network
        output = fused + self.feedforward(fused)
        
        return output.squeeze(1)  # (batch_size, language_dim)


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module that learns to combine vision and language features
    with learnable gating mechanisms.
    """
    
    def __init__(
        self,
        vision_dim=512,
        language_dim=896,
        gate_dim=256
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        
        # Vision feature projection
        self.vision_proj = nn.Linear(vision_dim, language_dim)
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(vision_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, language_dim),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Linear(language_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim)
        )
        
    def forward(self, vision_features):
        """
        Adaptive fusion of vision features.
        
        Args:
            vision_features: Tensor of shape (batch_size, vision_dim)
            
        Returns:
            adapted_features: Tensor of shape (batch_size, language_dim)
        """
        # Project vision features
        projected = self.vision_proj(vision_features)
        
        # Compute gating weights
        gates = self.gate_network(vision_features)
        
        # Apply gating
        gated_features = projected * gates
        
        # Refine features
        refined_features = self.refinement(gated_features)
        
        return refined_features


if __name__ == "__main__":
    # Test the fusion modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4
    vision_dim = 512
    language_dim = 896
    
    # Create dummy vision features
    vision_features = torch.randn(batch_size, vision_dim).to(device)
    
    # Test VisionLanguageFusion
    print("Testing VisionLanguageFusion:")
    fusion_mlp = VisionLanguageFusion(
        vision_dim=vision_dim,
        language_dim=language_dim,
        hidden_dim=1024,
        num_layers=2
    ).to(device)
    
    projected = fusion_mlp(vision_features)
    print(f"Input shape: {vision_features.shape}")
    print(f"Output shape: {projected.shape}")
    print(f"Parameter count: {sum(p.numel() for p in fusion_mlp.parameters())}")
    
    # Test AttentionFusion
    print("\nTesting AttentionFusion:")
    fusion_attn = AttentionFusion(
        vision_dim=vision_dim,
        language_dim=language_dim,
        num_heads=8
    ).to(device)
    
    attended = fusion_attn(vision_features)
    print(f"Output shape: {attended.shape}")
    print(f"Parameter count: {sum(p.numel() for p in fusion_attn.parameters())}")
    
    # Test AdaptiveFusion
    print("\nTesting AdaptiveFusion:")
    fusion_adaptive = AdaptiveFusion(
        vision_dim=vision_dim,
        language_dim=language_dim
    ).to(device)
    
    adapted = fusion_adaptive(vision_features)
    print(f"Output shape: {adapted.shape}")
    print(f"Parameter count: {sum(p.numel() for p in fusion_adaptive.parameters())}")