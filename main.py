"""
Main entry point for SimCLR training and evaluation pipeline.
"""

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
    evaluate_corruptions
)


def main():
    """Main execution function that runs the complete SimCLR pipeline."""

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

    # 3. Self-supervised pretraining
    model = train_simclr(model, train_loader, epochs=Config.EPOCHS)

    # 4. Linear probe evaluation
    encoder = model.encoder
    probe, clean_accuracy = train_linear_probe(
        encoder, eval_train_loader, eval_test_loader)
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%\n")

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
