"""
Main entry point for SimCLR training and evaluation pipeline.
"""

import argparse
import torch.nn.functional as F
import torch
from simclr import (
    Config,
    SimCLRModel,
    create_data_loaders,
    train_simclr,
    train_linear_probe,
    integrated_gradients,
    visualize_attribution,
    evaluate_corruptions,
    save_checkpoint,
    load_checkpoint,
)


def main():
    """Main execution function that runs the complete SimCLR pipeline.

    Supports resuming from a saved checkpoint. Use --resume to load
    weights from `Config.CHECKPOINT_PATH` before training.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Load model checkpoint if present')
    args = parser.parse_args()

    # Print configuration
    Config.print_config()
    print()

    # 1. Data preparation
    print("Loading CIFAR-10 datasets...")
    train_loader, eval_train_loader, eval_test_loader = create_data_loaders()
    print("Data loaders created.\n")

    # 2. Model initialization
    print("Initializing SimCLR model...")
    model = SimCLRModel(projection_dim=Config.PROJECTION_DIM).to(Config.DEVICE)
    print(
        f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optionally resume from checkpoint
    if args.resume:
        loaded = load_checkpoint(model, path=Config.CHECKPOINT_PATH)
        if loaded:
            print(f"Loaded checkpoint from {Config.CHECKPOINT_PATH}")
        else:
            print(
                f"No checkpoint found at {Config.CHECKPOINT_PATH}; training from scratch.")

    # 3. Self-supervised pretraining
    model = train_simclr(model, train_loader, epochs=Config.EPOCHS)

    # 4. Linear probe evaluation
    encoder = model.encoder
    probe, clean_accuracy = train_linear_probe(
        encoder, eval_train_loader, eval_test_loader)
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%\n")

    # Save checkpoints if configured
    if Config.SAVE_CHECKPOINTS:
        try:
            save_checkpoint(model, path=Config.CHECKPOINT_PATH)
            print(f"Saved SimCLR checkpoint to {Config.CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Warning: failed to save simclr checkpoint: {e}")
        try:
            save_checkpoint(probe, path=Config.PROBE_CHECKPOINT_PATH)
            print(f"Saved probe checkpoint to {Config.PROBE_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Warning: failed to save probe checkpoint: {e}")

    # 5. Explainability: Integrated Gradients
    print(f"\n{'='*60}")
    print("Explainability: Integrated Gradients")
    print(f"{'='*60}\n")

    # Get a sample image from test set
    sample_image, sample_label = eval_test_loader.dataset[0]
    sample_image = sample_image.unsqueeze(0).to(Config.DEVICE)

    # Get prediction
    probe.eval()
    with torch.no_grad():
        logits = probe(sample_image)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, predicted_class].item()

    print(
        f"Sample Image - True Label: {sample_label}, Predicted: {predicted_class}, Confidence: {confidence:.4f}")

    # Compute Integrated Gradients
    print("Computing Integrated Gradients...")
    attribution = integrated_gradients(
        probe, sample_image, predicted_class, steps=50)

    # Visualize
    visualize_attribution(sample_image, attribution,
                          save_path='attribution_heatmap.png')

    # 6. Domain shift testing
    corruption_results = evaluate_corruptions(probe)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%")
    print("\nCorruption Robustness:")
    for key, acc in corruption_results.items():
        print(f"  {key}: {acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
