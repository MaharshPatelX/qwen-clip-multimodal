from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch


@dataclass
class ModelConfig:
    """Configuration for the multimodal model."""
    
    # Model components
    clip_model_name: str = "openai/clip-vit-base-patch32"
    qwen_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct" 
    fusion_type: str = "mlp"  # "mlp", "attention", "adaptive"
    
    # Fusion module config
    fusion_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 1024,
        "num_layers": 2,
        "dropout": 0.1,
        "activation": "relu"
    })
    
    # Training behavior
    freeze_vision: bool = True
    use_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    # LoRA config
    lora_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
    })


@dataclass 
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset paths
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    image_dir: str = ""
    
    # Dataset type
    dataset_type: str = "custom"  # "coco", "cc3m", "custom", "instruction", "vqa"
    
    # Processing parameters
    max_length: int = 512
    image_size: tuple = (224, 224)
    
    # DataLoader parameters
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training stages
    stage1_epochs: int = 3  # Vision-language alignment
    stage2_epochs: int = 2  # End-to-end fine-tuning
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "polynomial"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # If 0, use warmup_ratio
    
    # Training behavior
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = "multimodal_llm_v1"
    run_name: str = ""
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Device and distributed training
    device: str = "auto"  # "auto", "cuda", "cpu"
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    dataloader_pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Debugging
    debug: bool = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.run_name:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.experiment_name}_{timestamp}"
        
        # Auto-detect device if needed
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            experiment_name=config_dict.get("experiment_name", "multimodal_llm_v1"),
            run_name=config_dict.get("run_name", ""),
            output_dir=config_dict.get("output_dir", "./outputs"),
            logging_dir=config_dict.get("logging_dir", "./logs"),
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            device=config_dict.get("device", "auto"),
            local_rank=config_dict.get("local_rank", -1),
            seed=config_dict.get("seed", 42),
            debug=config_dict.get("debug", False),
            max_train_samples=config_dict.get("max_train_samples"),
            max_eval_samples=config_dict.get("max_eval_samples")
        )
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios
def get_debug_config() -> ExperimentConfig:
    """Get configuration for debugging with small datasets."""
    config = ExperimentConfig()
    config.debug = True
    config.max_train_samples = 100
    config.max_eval_samples = 50
    config.data.batch_size = 4
    config.training.stage1_epochs = 1
    config.training.stage2_epochs = 1
    config.training.eval_steps = 10
    config.training.save_steps = 20
    config.training.logging_steps = 5
    return config


def get_small_scale_config() -> ExperimentConfig:
    """Get configuration for small-scale training."""
    config = ExperimentConfig()
    config.experiment_name = "multimodal_llm_small"
    config.data.batch_size = 8
    config.training.gradient_accumulation_steps = 2
    config.training.stage1_epochs = 2
    config.training.stage2_epochs = 1
    config.model.use_lora = True
    config.model.load_in_8bit = True
    return config


def get_full_scale_config() -> ExperimentConfig:
    """Get configuration for full-scale training."""
    config = ExperimentConfig()
    config.experiment_name = "multimodal_llm_full"
    config.data.batch_size = 32
    config.training.gradient_accumulation_steps = 1
    config.training.stage1_epochs = 3
    config.training.stage2_epochs = 2
    config.model.freeze_vision = True
    config.model.use_lora = False
    return config


if __name__ == "__main__":
    # Test configuration classes
    config = ExperimentConfig()
    print("Default configuration:")
    print(f"Model: {config.model.qwen_model_name}")
    print(f"Fusion: {config.model.fusion_type}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Test saving and loading
    config.save("test_config.json")
    loaded_config = ExperimentConfig.load("test_config.json")
    print(f"\nLoaded config experiment name: {loaded_config.experiment_name}")
    
    # Test predefined configs
    debug_config = get_debug_config()
    print(f"Debug config max samples: {debug_config.max_train_samples}")
    
    print("Configuration system working correctly!")