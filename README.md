# 🤖 Multimodal LLM: Image-Aware Language Generation

A lightweight multimodal large language model that combines CLIP vision understanding with Qwen2.5 language generation capabilities for image captioning, visual question answering, and multimodal chat.

## 🏗️ Architecture

- **Vision Encoder**: CLIP ViT-B/32 (512D embeddings)
- **Language Model**: Qwen2.5-0.5B (0.5B parameters, 32K context)
- **Fusion Module**: Learnable MLP projection (512D → 896D)
- **Total Parameters**: ~583M (495M trainable)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd multimodal_llm

# Create virtual environment (using uv - recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Basic Usage

#### Training

```bash
# Debug training (small dataset, 1 epoch each stage)
python examples/train_model.py --debug

# Small-scale training (LoRA, 8-bit quantization)
python examples/train_model.py --small-scale

# Full-scale training
python examples/train_model.py --config configs/full_scale.json

# Resume from checkpoint
python examples/train_model.py --resume ./outputs/checkpoint-epoch-1-step-100
```

#### Inference

```python
from inference import MultimodalInferencePipeline
from models import MultimodalLLM

# Load trained model
model = MultimodalLLM.from_pretrained("./outputs/best_model")
pipeline = MultimodalInferencePipeline(model)

# Image captioning
caption = pipeline.caption_image("path/to/image.jpg", "Describe this image.")

# Visual question answering
answer = pipeline.answer_question("path/to/image.jpg", "What color is the car?")

# Multimodal chat
response = pipeline.chat("What do you see in this image?", image="path/to/image.jpg")
```

#### API Server

```bash
# Start REST API server
python inference/api.py

# Test endpoints
curl -X POST "http://localhost:8000/caption" \
  -F "image=@path/to/image.jpg" \
  -F "prompt=Describe this image"
```

## 📊 Training Configurations

### Debug Configuration (`configs/debug.json`)
- **Purpose**: Quick testing and development
- **Dataset**: 100 training samples, 50 validation samples
- **Training**: 1 epoch per stage
- **Batch Size**: 4
- **Use Case**: Development and debugging

### Small-Scale Configuration (`configs/small_scale.json`)
- **Purpose**: Resource-constrained training
- **Features**: LoRA fine-tuning, 8-bit quantization
- **Training**: 2+1 epochs
- **Batch Size**: 8 (with gradient accumulation)
- **Use Case**: Limited GPU memory (8-16GB)

### Full-Scale Configuration (`configs/full_scale.json`)
- **Purpose**: Production-quality training
- **Training**: 3+2 epochs
- **Batch Size**: 32
- **Use Case**: High-end GPUs (24GB+ VRAM)

## 📁 Project Structure

```
multimodal_llm/
├── configs/                    # Training configurations
│   ├── debug.json             # Debug settings
│   ├── small_scale.json       # Small-scale training
│   └── full_scale.json        # Full-scale training
├── data/                      # Data processing
│   ├── datasets/              # Dataset classes
│   └── preprocessing/         # Data utilities
├── models/                    # Model components
│   ├── clip_encoder.py        # CLIP vision encoder
│   ├── qwen_decoder.py        # Qwen language decoder
│   ├── fusion_module.py       # Vision-language fusion
│   └── multimodal_model.py    # Complete architecture
├── training/                  # Training pipeline
│   ├── config.py              # Configuration classes
│   └── trainer.py             # Two-stage trainer
├── inference/                 # Inference pipeline
│   ├── pipeline.py            # Inference utilities
│   └── api.py                 # REST API server
├── evaluation/                # Evaluation metrics
├── utils/                     # Utilities
│   ├── logging.py             # Logging setup
│   └── checkpoint.py          # Checkpoint management
├── examples/                  # Example scripts
│   └── train_model.py         # Training example
└── test_data/                 # Sample data for testing
```

## 📚 Detailed Usage

### Preparing Your Data

#### Format 1: Instruction Format (Recommended)
```json
[
  {
    "image": "image1.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is in this image?"
      },
      {
        "from": "assistant",
        "value": "The image shows a cat sitting on a windowsill."
      }
    ]
  }
]
```

#### Format 2: Custom Format
```json
[
  {
    "image": "image1.jpg",
    "text": "A cat sitting on a windowsill",
    "prompt": "Describe this image."
  }
]
```

### Training Pipeline

#### Stage 1: Vision-Language Alignment
- **Duration**: 1-3 epochs
- **Strategy**: Freeze language model, train only fusion module
- **Goal**: Align CLIP features with Qwen embedding space

#### Stage 2: End-to-End Fine-tuning
- **Duration**: 1-2 epochs  
- **Strategy**: Unfreeze all parameters (or use LoRA)
- **Goal**: Task-specific performance optimization

### Custom Configuration

Create your own configuration:

```python
from training.config import ExperimentConfig

config = ExperimentConfig()
config.experiment_name = "my_experiment"
config.data.train_data_path = "path/to/train.json"
config.data.image_dir = "path/to/images"
config.data.batch_size = 16
config.training.learning_rate = 1e-4

# Save configuration
config.save("my_config.json")

# Use in training
python examples/train_model.py --config my_config.json
```

### Evaluation

```python
from evaluation import MultimodalEvaluator

evaluator = MultimodalEvaluator()

# Image captioning metrics
predictions = ["A cat on a windowsill", "A dog in the park"]
references = [["A cat sitting on a windowsill"], ["A dog playing in the park"]]
results = evaluator.evaluate_captioning(predictions, references)

print(f"BLEU-4: {results['bleu_4']:.4f}")
print(f"CIDEr: {results['cider']:.4f}")
print(f"ROUGE-L: {results['rouge_l']:.4f}")
```

## 🛠️ Advanced Features

### LoRA Fine-tuning

Enable parameter-efficient training:

```json
{
  "model": {
    "use_lora": true,
    "lora_config": {
      "r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
  }
}
```

### Distributed Training

```bash
# Multi-GPU training
accelerate launch --multi_gpu examples/train_model.py --config configs/full_scale.json

# Custom accelerate config
accelerate config
accelerate launch examples/train_model.py --config configs/full_scale.json
```

### Quantization

```json
{
  "model": {
    "load_in_8bit": true,  // 8-bit quantization
    "load_in_4bit": false  // 4-bit quantization (experimental)
  }
}
```

### Monitoring with Weights & Biases

```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
python examples/train_model.py --config configs/small_scale.json
```

## 📈 Performance Targets

### COCO Captions Benchmark
- **BLEU-4**: >25
- **CIDEr**: >85
- **ROUGE-L**: >50

### VQA Tasks
- **Accuracy**: >60% on VQAv2
- **Inference Speed**: <2 seconds per response

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
export BATCH_SIZE=4

# Use gradient accumulation
export GRADIENT_ACCUMULATION_STEPS=4

# Enable 8-bit quantization
python examples/train_model.py --small-scale
```

#### Slow Training
```bash
# Use mixed precision
export BF16=true

# Reduce max_length
export MAX_LENGTH=256

# Use fewer workers
export NUM_WORKERS=2
```

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

#### Memory Usage
- Use gradient checkpointing for large models
- Enable mixed precision (BF16/FP16)
- Use LoRA for parameter-efficient training

#### Speed Optimization
- Increase batch size if memory allows
- Use gradient accumulation for effective larger batches
- Enable compiled mode (PyTorch 2.0+)

## 🚀 Deployment

### Model Export

```python
# Save for deployment
model.save_pretrained("./deployment_model")

# Load for inference
from models import MultimodalLLM
model = MultimodalLLM.from_pretrained("./deployment_model")
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "inference/api.py"]
```

### Production API

```bash
# Start production server
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn
gunicorn inference.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📋 Requirements

### Hardware Requirements

#### Minimum (Debug)
- **GPU**: 8GB VRAM (RTX 3070, V100)
- **RAM**: 16GB
- **Storage**: 10GB

#### Recommended (Small-Scale)
- **GPU**: 16GB VRAM (RTX 4080, A100)
- **RAM**: 32GB
- **Storage**: 50GB

#### Optimal (Full-Scale)
- **GPU**: 24GB+ VRAM (RTX 4090, A100)
- **RAM**: 64GB+
- **Storage**: 100GB+

### Software Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.7+ (for GPU training)
- **Transformers**: 4.35+

## 🔗 API Reference

### REST API Endpoints

#### POST /caption
Generate image captions.

```bash
curl -X POST "http://localhost:8000/caption" \
  -F "image=@image.jpg" \
  -F "prompt=Describe this image" \
  -F "max_new_tokens=100"
```

#### POST /question
Answer questions about images.

```bash
curl -X POST "http://localhost:8000/question" \
  -F "image=@image.jpg" \
  -F "question=What color is the car?"
```

#### POST /chat
Multimodal chat interface.

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you see?",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "conversation_id": "user123"
  }'
```

#### GET /health
Health check endpoint.

```bash
curl http://localhost:8000/health
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** for vision understanding
- **Qwen Team** for the language model
- **Hugging Face** for model hosting and transformers library
- **PyTorch** and **Accelerate** for training infrastructure

## 📞 Support

For questions and support:

- **Documentation**: See [examples/](examples/) folder
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions

---

**Happy multimodal modeling! 🎉**