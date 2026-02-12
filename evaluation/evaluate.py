"""
Evaluation script for trained cGAN models.

Generates:
- Sample grid per class
- Class variation plot (fixed z, varying class labels)
- Latent interpolation plot (interpolate between two z vectors)
- Discriminator accuracy statistics

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/improved_final.pt \
                                   --config config/improved_config.yaml \
                                   --arch improved
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import yaml

from models.cgan import cGAN
from data.dataloader import get_dataloaders, CIFAR10_CLASSES
from evaluation.visualize import (
    save_image_grid,
    plot_class_variation,
    plot_latent_interpolation,
)


def evaluate(checkpoint_path: str, config_path: str, arch: str, output_dir: str = "results/eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["model"]["architecture"] = arch

    model = cGAN(config, device)
    epoch = model.load_checkpoint(checkpoint_path)
    model.G.eval()
    model.D.eval()
    print(f"Loaded checkpoint from epoch {epoch}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = config["data"]["num_classes"]
    latent_dim  = config["model"]["latent_dim"]

    # ---- Per-class sample grids ----
    print("Generating per-class sample grids...")
    for class_idx in range(num_classes):
        labels = torch.full((64,), class_idx, dtype=torch.long, device=device)
        samples = model.generate(labels)
        class_name = CIFAR10_CLASSES[class_idx]
        save_image_grid(
            samples,
            out_dir / f"class_{class_idx:02d}_{class_name}.png",
            nrow=8,
            title=f"Class: {class_name}",
        )

    # ---- Class variation (fixed z, all classes) ----
    print("Generating class variation plot...")
    plot_class_variation(model, latent_dim, num_classes, device,
                         out_dir / "class_variation.png")

    # ---- Latent interpolation ----
    print("Generating latent interpolation plot...")
    plot_latent_interpolation(model, latent_dim, num_classes, device,
                              out_dir / "latent_interpolation.png")

    # ---- Discriminator accuracy on test set ----
    print("Computing discriminator accuracy on test set...")
    _, test_loader = get_dataloaders(
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
    )

    real_correct = 0
    fake_correct = 0
    total = 0

    with torch.no_grad():
        for real_images, labels in test_loader:
            real_images, labels = real_images.to(device), labels.to(device)
            bs = real_images.size(0)

            real_scores = model.D(real_images, labels)
            z = torch.randn(bs, latent_dim, device=device)
            fake_images = model.G(z, labels)
            fake_scores = model.D(fake_images, labels)

            real_correct += (real_scores > 0).sum().item()
            fake_correct += (fake_scores < 0).sum().item()
            total += bs

    real_acc = real_correct / total
    fake_acc = fake_correct / total
    print(f"\nDiscriminator Accuracy:")
    print(f"  Real images classified as real: {real_acc:.2%}")
    print(f"  Fake images classified as fake: {fake_acc:.2%}")
    print(f"  Overall accuracy: {(real_acc + fake_acc) / 2:.2%}")
    print(f"\nEvaluation complete. Results saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cGAN")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--arch",       default="improved", choices=["improved", "baseline"])
    parser.add_argument("--output_dir", default="results/eval")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.arch, args.output_dir)


if __name__ == "__main__":
    main()
