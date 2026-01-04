# SimCLR Implementation for CIFAR-10

A complete, minimal, and runnable PyTorch implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) for self-supervised learning on CIFAR-10.

## Overview

This implementation provides a research prototype to validate the SimCLR framework on CIFAR-10. It includes all essential components for self-supervised learning, evaluation, explainability, and robustness testing.

## Features

- **Self-Supervised Pretraining**: Contrastive learning with NT-Xent loss
- **Data Augmentation**: Custom SimCLR augmentation pipeline (RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur)
- **Model Architecture**: ResNet-18 encoder with MLP projection head
- **Linear Probe Evaluation**: Standard protocol for assessing learned representations
- **Explainability**: Integrated Gradients for pixel-level attribution visualization
- **Domain Shift Testing**: Robustness evaluation on corrupted CIFAR-10 test sets

## Repository Structure

```
SimCLR-implementation/
├── simclr/                    # Main package
│   ├── __init__.py           # Package initialization and exports
│   ├── config.py             # Configuration and hyperparameters
│   ├── data.py               # Data loading and augmentation pipeline
│   ├── models.py             # Model architectures (encoder, projection head, linear probe)
│   ├── loss.py               # NT-Xent contrastive loss function
│   ├── train.py              # Training loop for SimCLR
│   ├── evaluate.py           # Linear probe evaluation
│   ├── explainability.py     # Integrated Gradients implementation
│   └── domain_shift.py       # Domain shift robustness testing
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SimCLR-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load and preprocess CIFAR-10 data
2. Train SimCLR model using contrastive learning
3. Evaluate learned representations with linear probe
4. Generate attribution visualizations using Integrated Gradients
5. Test robustness on corrupted CIFAR-10 test sets

## Configuration

All hyperparameters can be modified in `simclr/config.py`:

- `PROJECTION_DIM`: Dimension of projection head output (default: 128)
- `TEMPERATURE`: Temperature parameter for NT-Xent loss (default: 0.5)
- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Batch size for contrastive learning (default: 256)
- `LEARNING_RATE`: Learning rate (default: 3e-4)
- `LINEAR_PROBE_EPOCHS`: Epochs for linear probe training (default: 50)

## Components

### Data Pipeline (`simclr/data.py`)
- `SimCLRTransform`: Augmentation pipeline generating two views per image
- `SimCLRDataset`: Dataset wrapper for self-supervised pretraining
- `create_data_loaders()`: Creates DataLoaders for training and evaluation

### Model Architecture (`simclr/models.py`)
- `SimCLRModel`: Complete SimCLR model with ResNet-18 encoder and projection head
- `ProjectionHead`: MLP projection head
- `LinearProbe`: Linear classifier for evaluation

### Loss Function (`simclr/loss.py`)
- `nt_xent_loss()`: Normalized Temperature-scaled Cross Entropy Loss

### Training (`simclr/train.py`)
- `train_simclr()`: Training loop for contrastive learning

### Evaluation (`simclr/evaluate.py`)
- `train_linear_probe()`: Train and evaluate linear probe

### Explainability (`simclr/explainability.py`)
- `integrated_gradients()`: Compute Integrated Gradients attribution
- `visualize_attribution()`: Visualize attribution heatmaps

### Domain Shift (`simclr/domain_shift.py`)
- `CorruptedCIFAR10`: Dataset with corruption transformations
- `evaluate_corruptions()`: Evaluate robustness to domain shifts

## Output

The script generates:
- Training progress logs with loss values
- Linear probe accuracy on clean test set
- Attribution heatmap visualization (`attribution_heatmap.png`)
- Robustness results on corrupted test sets

## Notes

- This is a research prototype optimized for clarity and educational value
- The implementation prioritizes conceptual correctness over production optimization
- CIFAR-10 data will be automatically downloaded on first run
- GPU is automatically used if available, otherwise falls back to CPU

## License

This implementation is provided for research and educational purposes.

