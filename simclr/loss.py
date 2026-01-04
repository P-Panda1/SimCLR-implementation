"""
Contrastive loss function: NT-Xent (Normalized Temperature-scaled Cross Entropy).
"""

import torch
import torch.nn.functional as F
from .config import Config


def nt_xent_loss(z1, z2, temperature=None):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Computes contrastive loss between two sets of projections (z1, z2).
    Positive pairs: (z1[i], z2[i]) for all i
    Negative pairs: all other combinations in the batch
    
    Args:
        z1: First set of projections [batch_size, projection_dim]
        z2: Second set of projections [batch_size, projection_dim]
        temperature: Temperature scaling parameter (default: Config.TEMPERATURE)
    
    Returns:
        Contrastive loss value
    """
    if temperature is None:
        temperature = Config.TEMPERATURE
    
    batch_size = z1.size(0)
    
    # Normalize projections to unit hypersphere
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate projections: [z1, z2] -> [2*batch_size, projection_dim]
    projections = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix (cosine similarity)
    # similarity_matrix[i, j] = similarity between projection i and projection j
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create labels for positive pairs
    # For sample i in [0, batch_size-1], positive pair is (i, i+batch_size)
    # For sample i in [batch_size, 2*batch_size-1], positive pair is (i, i-batch_size)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z1.device),
        torch.arange(0, batch_size, device=z1.device)
    ])
    
    # Mask out the diagonal (self-similarity)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Compute cross-entropy loss
    # Each row represents one sample, columns are all other samples
    # Label indicates which column is the positive pair
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

