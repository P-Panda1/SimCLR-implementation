"""
Domain shift testing: evaluate robustness to corrupted CIFAR-10 test sets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from .config import Config


class CorruptedCIFAR10(Dataset):
    """
    Dataset wrapper that applies corruption transformations to CIFAR-10 test set.
    Used to evaluate robustness to domain shifts.
    """
    def __init__(self, root, corruption_type='gaussian_noise', severity=1, download=True):
        self.dataset = datasets.CIFAR10(root=root, train=False, download=download)
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Define corruption parameters
        self.corruption_params = {
            'gaussian_noise': {'std': 0.01 * severity},
            'gaussian_blur': {'sigma': 0.5 * severity},
            'brightness': {'factor': 0.1 * severity}
        }
        
        # Standard normalization
        self.normalize = transforms.Normalize(
            mean=Config.CIFAR10_MEAN, 
            std=Config.CIFAR10_STD
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = transforms.ToTensor()(image)  # Convert to tensor [0, 1]
        
        # Apply corruption
        if self.corruption_type == 'gaussian_noise':
            noise = torch.randn_like(image) * self.corruption_params['gaussian_noise']['std']
            image = torch.clamp(image + noise, 0, 1)
        elif self.corruption_type == 'gaussian_blur':
            sigma = self.corruption_params['gaussian_blur']['sigma']
            # Apply Gaussian blur using convolution
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            image = blur(image)
        elif self.corruption_type == 'brightness':
            factor = self.corruption_params['brightness']['factor']
            image = torch.clamp(image + factor, 0, 1)
        
        # Normalize
        image = self.normalize(image)
        
        return image, label


def evaluate_corruptions(probe, root=None):
    """
    Evaluate linear probe on corrupted CIFAR-10 test sets.
    Tests robustness to domain shifts: Gaussian noise, Gaussian blur, brightness.
    
    Args:
        probe: Trained linear probe model
        root: Root directory for CIFAR-10 data (default: Config.DATA_ROOT)
    
    Returns:
        Dictionary mapping corruption types and severities to accuracies
    """
    if root is None:
        root = Config.DATA_ROOT
    
    corruption_types = ['gaussian_noise', 'gaussian_blur', 'brightness']
    severities = [1, 2, 3]
    
    print(f"\n{'='*60}")
    print("Domain Shift Robustness Testing")
    print(f"{'='*60}\n")
    
    results = {}
    
    for corruption_type in corruption_types:
        print(f"\nCorruption Type: {corruption_type}")
        print("-" * 60)
        for severity in severities:
            # Create corrupted dataset
            corrupted_dataset = CorruptedCIFAR10(
                root=root, 
                corruption_type=corruption_type, 
                severity=severity
            )
            corrupted_loader = DataLoader(
                corrupted_dataset, 
                batch_size=Config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=Config.NUM_WORKERS
            )
            
            # Evaluate
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(corrupted_loader, desc=f"Severity {severity}"):
                    images = images.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE)
                    logits = probe(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            results[f"{corruption_type}_s{severity}"] = accuracy
            print(f"Severity {severity}: {accuracy:.2f}%")
    
    return results

