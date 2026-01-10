"""Checkpoint utilities: save and load model + optimizer states."""

import os
import torch
from .config import Config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer=None, path=None, extra=None):
    """Save model and optimizer state to path. Creates directory if needed."""
    if path is None:
        path = Config.CHECKPOINT_PATH
    ensure_dir(os.path.dirname(path))

    payload = {
        'model_state_dict': model.state_dict(),
        'extra': extra,
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(payload, path)


def load_checkpoint(model, optimizer=None, path=None, map_location=None):
    """Load checkpoint if present. Returns True on success, False otherwise.

    On success, model and optimizer (if provided) are updated in-place.
    """
    if path is None:
        path = Config.CHECKPOINT_PATH
    if map_location is None:
        map_location = Config.DEVICE

    if not os.path.exists(path):
        return False

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return True
