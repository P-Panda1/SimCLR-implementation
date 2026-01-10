"""
Linear probe evaluation for assessing learned representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .config import Config
from .models import LinearProbe


def train_linear_probe(encoder, train_loader, test_loader, epochs=None, lr=None):
    """
    Train linear probe: freeze encoder, train only linear classifier.
    Returns trained probe and test accuracy.

    Args:
        encoder: Frozen encoder model
        train_loader: DataLoader for training
        test_loader: DataLoader for evaluation
        epochs: Number of training epochs (default: Config.LINEAR_PROBE_EPOCHS)
        lr: Learning rate (default: Config.LINEAR_PROBE_LR)

    Returns:
        probe: Trained linear probe model
        test_accuracy: Test accuracy percentage
    """
    if epochs is None:
        epochs = Config.LINEAR_PROBE_EPOCHS
    if lr is None:
        lr = Config.LINEAR_PROBE_LR

    probe = LinearProbe(encoder).to(Config.DEVICE)
    optimizer = optim.Adam(probe.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print(f"Training Linear Probe for {epochs} epochs")
    print(f"{'='*60}\n")

    # Training loop
    probe.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # Forward pass
            logits = probe(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            pbar.set_postfix(
                {'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Evaluation
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            logits = probe(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"\nLinear Probe Test Accuracy: {test_accuracy:.2f}%\n")

    return probe, test_accuracy
