# Gemma Fine-tuning

This repository contains code for full parameter fine-tuning of the Gemma-2b-9b model on A100 80GB GPU. The implementation uses DeepSpeed for efficient training and memory optimization.

## Features

- Full parameter fine-tuning of Gemma-2b-9b
- DeepSpeed ZeRO Stage-2 optimization
- Efficient memory management for A100 GPU
- Modular and extensible codebase
- Support for custom datasets and training configurations

## Requirements

- A100 80GB GPU
- Python 3.8+
- CUDA 11.8+

## Project Structure

```
project_root/
├── config/                 # Configuration files
│   ├── deepspeed_config.json
│   └── training_config.py
├── data/                   # Data processing
│   └── data_processor.py
├── training/              # Training logic
│   └── trainer.py
├── utils/                 # Utility functions
│   └── memory_utils.py
├── requirements.txt
└── train.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemma-finetuning.git
cd gemma-finetuning
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Model Configuration

Edit `config/training_config.py` to modify model parameters:

```python
@dataclass
class ModelConfig:
    model_name: str = "google/gemma-2b-9b"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
```

### Training Configuration

Adjust training parameters in `config/training_config.py`:

```python
@dataclass
class TrainingConfig:
    output_dir: str = "./gemma-ft-output"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    # ... other parameters
```

### DeepSpeed Configuration

Modify DeepSpeed settings in `config/deepspeed_config.json`:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
    // ... other settings
}
```

## Usage

1. Start training:
```bash
python train.py
```

2. Monitor training:
- Training logs will be saved in the specified output directory
- DeepSpeed logs will show memory usage and training progress

3. Find the trained model:
- The final model will be saved in `{output_dir}/final_model`

## Customization

### Using Custom Datasets

1. Modify the `DataConfig` in `config/training_config.py`:
```python
@dataclass
class DataConfig:
    dataset_name: str = "your_dataset_name"
    chunk_length: int = 2048
```

2. If needed, extend the `DataProcessor` class in `data/data_processor.py` to handle your dataset format.

### Memory Optimization

Adjust memory-related parameters based on your GPU:

1. Batch size and gradient accumulation in `TrainingConfig`
2. ZeRO stage and offload settings in `deepspeed_config.json`
3. Chunk length for data processing in `DataConfig`

## Monitoring Resources

Monitor GPU memory usage during training:
```bash
nvidia-smi
```

## Troubleshooting

Common issues and solutions:

1. Out of Memory (OOM):
- Reduce batch size
- Increase gradient accumulation steps
- Enable CPU offloading in DeepSpeed config

2. Training Too Slow:
- Increase batch size if memory allows
- Adjust DeepSpeed bucket sizes
- Check CPU offload settings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- Original Gemma model by Google
- DeepSpeed by Microsoft
- Hugging Face Transformers library

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gemma-finetuning,
  author = {Your Name},
  title = {Gemma Fine-tuning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/gemma-finetuning}
}
```