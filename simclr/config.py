"""
Configuration module for SimCLR hyperparameters and settings.
"""

import torch


class Config:
    """Configuration class containing all hyperparameters and settings."""

    # SimCLR hyperparameters
    PROJECTION_DIM = 128  # Dimension of projection head output
    TEMPERATURE = 0.5  # Temperature parameter for NT-Xent loss
    EPOCHS = 100  # Number of training epochs
    BATCH_SIZE = 256  # Batch size for contrastive learning
    LEARNING_RATE = 3e-4  # Learning rate
    WEIGHT_DECAY = 1e-4  # Weight decay for optimizer
    NUM_WORKERS = 4  # DataLoader workers

    # Linear probe hyperparameters
    LINEAR_PROBE_EPOCHS = 50
    LINEAR_PROBE_LR = 1e-3

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    DATA_ROOT = './data'
    # Checkpointing
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_PATH = './checkpoints/simclr.pth'
    PROBE_CHECKPOINT_PATH = './checkpoints/probe.pth'
    SAVE_CHECKPOINTS = True  # Whether to save checkpoints after training

    # CIFAR-10 normalization constants
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2470, 0.2435, 0.2616]

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print(f"Using device: {cls.DEVICE}")
        print(f"Projection dimension: {cls.PROJECTION_DIM}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Training epochs: {cls.EPOCHS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Checkpoint path: {cls.CHECKPOINT_PATH}")
