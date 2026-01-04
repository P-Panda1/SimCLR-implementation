"""
SimCLR: Simple Framework for Contrastive Learning of Visual Representations
A complete implementation for self-supervised learning on CIFAR-10.
"""

__version__ = "1.0.0"

from .config import Config
from .models import SimCLRModel, ProjectionHead, LinearProbe
from .data import SimCLRTransform, SimCLRDataset, create_data_loaders
from .loss import nt_xent_loss
from .train import train_simclr
from .evaluate import train_linear_probe
from .explainability import integrated_gradients, visualize_attribution
from .domain_shift import CorruptedCIFAR10, evaluate_corruptions

__all__ = [
    'Config',
    'SimCLRModel',
    'ProjectionHead',
    'LinearProbe',
    'SimCLRTransform',
    'SimCLRDataset',
    'create_data_loaders',
    'nt_xent_loss',
    'train_simclr',
    'train_linear_probe',
    'integrated_gradients',
    'visualize_attribution',
    'CorruptedCIFAR10',
    'evaluate_corruptions',
]

