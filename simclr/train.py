"""
Training loop for SimCLR contrastive learning.
"""

import torch.optim as optim
from tqdm import tqdm
from .config import Config
from .loss import nt_xent_loss


def train_simclr(model, train_loader, epochs=None, lr=None, weight_decay=None):
    """
    Train SimCLR model using contrastive learning.
    Logs loss at regular intervals.
    
    Args:
        model: SimCLR model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs (default: Config.EPOCHS)
        lr: Learning rate (default: Config.LEARNING_RATE)
        weight_decay: Weight decay (default: Config.WEIGHT_DECAY)
    
    Returns:
        Trained model
    """
    if epochs is None:
        epochs = Config.EPOCHS
    if lr is None:
        lr = Config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = Config.WEIGHT_DECAY
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    
    print(f"\n{'='*60}")
    print(f"Training SimCLR for {epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for views1, views2 in pbar:
            views1 = views1.to(Config.DEVICE)
            views2 = views2.to(Config.DEVICE)
            
            # Forward pass: get projections for both views
            z1 = model(views1)
            z2 = model(views2)
            
            # Compute contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=Config.TEMPERATURE)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}\n")
    
    print("Training completed!\n")
    return model

