"""
Data pipeline and augmentation modules for SimCLR.
"""

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from .config import Config


class SimCLRTransform:
    """
    SimCLR augmentation pipeline that generates two augmented views of the same image.
    Applies: RandomResizedCrop, ColorJitter, RandomGrayscale, and GaussianBlur.
    """
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.CIFAR10_MEAN, std=Config.CIFAR10_STD)
        ])
    
    def __call__(self, x):
        # Generate two different augmented views of the same image
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class SimCLRDataset(Dataset):
    """
    Dataset wrapper that applies SimCLR augmentations to generate positive pairs.
    Labels are ignored during self-supervised pretraining.
    """
    def __init__(self, root, train=True, transform=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, _ = self.cifar10[idx]  # Ignore label during pretraining
        if self.transform:
            view1, view2 = self.transform(image)
            return view1, view2
        return image


def create_data_loaders(root=None, batch_size=None):
    """
    Create DataLoaders for pretraining (no labels needed) and evaluation.
    
    Args:
        root: Root directory for CIFAR-10 data (default: Config.DATA_ROOT)
        batch_size: Batch size (default: Config.BATCH_SIZE)
    
    Returns:
        train_loader: DataLoader for contrastive pretraining
        eval_train_loader: DataLoader for linear probe training
        eval_test_loader: DataLoader for linear probe evaluation
    """
    if root is None:
        root = Config.DATA_ROOT
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Pretraining dataset (ignores labels)
    train_dataset = SimCLRDataset(
        root=root, 
        train=True, 
        transform=SimCLRTransform(size=32)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Evaluation dataset (with labels for linear probe)
    eval_train_dataset = datasets.CIFAR10(
        root=root, 
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.CIFAR10_MEAN, std=Config.CIFAR10_STD)
        ])
    )
    eval_train_loader = DataLoader(
        eval_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    
    eval_test_dataset = datasets.CIFAR10(
        root=root, 
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.CIFAR10_MEAN, std=Config.CIFAR10_STD)
        ])
    )
    eval_test_loader = DataLoader(
        eval_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, eval_train_loader, eval_test_loader

