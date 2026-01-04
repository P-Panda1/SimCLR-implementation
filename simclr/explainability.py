"""
Explainability module: Integrated Gradients for attribution visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .config import Config


def integrated_gradients(model, input_image, target_class, baseline=None, steps=50):
    """
    Compute Integrated Gradients attribution for a given input and target class.
    
    Integrated Gradients approximates the integral of gradients along the path
    from a baseline (typically zeros or black image) to the input.
    
    Args:
        model: The model to explain (linear probe)
        input_image: Input image tensor [1, C, H, W]
        target_class: Target class index for attribution
        baseline: Baseline image (default: zeros)
        steps: Number of interpolation steps
    
    Returns:
        Attribution map [H, W] (summed across channels)
    """
    model.eval()
    input_image = input_image.clone().detach().requires_grad_(True)
    
    if baseline is None:
        baseline = torch.zeros_like(input_image)
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps, device=input_image.device)
    gradients = []
    
    for alpha in alphas:
        # Interpolated input
        interpolated = baseline + alpha * (input_image - baseline)
        interpolated.requires_grad_(True)
        
        # Forward pass
        output = model(interpolated)
        target_output = output[0, target_class]
        
        # Backward pass
        grad = torch.autograd.grad(target_output, interpolated, create_graph=False)[0]
        gradients.append(grad)
    
    # Average gradients
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Integrated gradients = (input - baseline) * avg_gradients
    attribution = (input_image - baseline) * avg_gradients
    
    # Sum across channels for visualization
    attribution = attribution[0].sum(dim=0).detach().cpu().numpy()
    
    return attribution


def visualize_attribution(image, attribution, save_path='attribution_heatmap.png'):
    """
    Visualize Integrated Gradients attribution as a heatmap overlaid on the image.
    
    Args:
        image: Input image tensor [1, C, H, W]
        attribution: Attribution map [H, W]
        save_path: Path to save the visualization
    """
    # Denormalize image for visualization
    mean = torch.tensor(Config.CIFAR10_MEAN)
    std = torch.tensor(Config.CIFAR10_STD)
    image_denorm = image[0].cpu().clone()
    for t, m, s in zip(image_denorm, mean, std):
        t.mul_(s).add_(m)
    image_denorm = torch.clamp(image_denorm, 0, 1)
    image_np = image_denorm.permute(1, 2, 0).numpy()
    
    # Normalize attribution for visualization
    attribution_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(attribution_norm, cmap='hot', interpolation='nearest')
    axes[1].set_title('Attribution Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    overlay = image_np.copy()
    attribution_overlay = np.expand_dims(attribution_norm, axis=2)
    overlay = 0.5 * overlay + 0.5 * (attribution_overlay * np.array([1, 0, 0]))  # Red overlay
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attribution visualization saved to {save_path}")
    plt.close()

