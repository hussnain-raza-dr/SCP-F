"""
Visualization utilities for cGAN training and evaluation.

Produces all plots required for the presentation:
- Training curves (G/D loss + discriminator accuracy)
- Sample image grids
- Class variation (fixed z, varying class label)
- Latent interpolation (spherical interpolation between two z vectors)
"""

from pathlib import Path
from typing import Union

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from data.dataloader import CIFAR10_CLASSES


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] images back to [0, 1] for display."""
    return (images.clamp(-1, 1) + 1) / 2


def save_image_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 10,
    title: str = None,
):
    """Save a grid of generated images."""
    images = denormalize(images.cpu())
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    np_grid = grid.permute(1, 2, 0).numpy()

    nrows = (len(images) + nrow - 1) // nrow
    fig, ax = plt.subplots(figsize=(nrow * 1.2, nrows * 1.2))
    ax.imshow(np_grid)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: dict, path: Union[str, Path]):
    """
    Plot G/D losses and discriminator accuracy over training epochs.
    This is a key diagnostic for detecting discriminator dominance.
    """
    epochs = range(1, len(history["d_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["d_loss"], label="D Loss", color="#e74c3c", linewidth=2)
    ax1.plot(epochs, history["g_loss"], label="G Loss", color="#3498db", linewidth=2)
    if any(v != 0 for v in history.get("gp", [0])):
        ax1.plot(epochs, history["gp"], label="Gradient Penalty", color="#2ecc71",
                 linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Losses", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Discriminator accuracy
    ax2.plot(epochs, [v * 100 for v in history["real_acc"]],
             label="Real → Real", color="#27ae60", linewidth=2)
    ax2.plot(epochs, [v * 100 for v in history["fake_acc"]],
             label="Fake → Fake", color="#e67e22", linewidth=2)
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Discriminator Accuracy", fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("cGAN Training Dynamics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_variation(
    model,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    path: Union[str, Path],
    n_fixed_z: int = 6,
):
    """
    Class variation plot: fix several z vectors, vary the class label.
    Grid: rows = fixed z vectors, cols = classes.
    Shows that the generator can produce distinct, class-conditioned images.
    """
    model.G.eval()
    with torch.no_grad():
        fixed_zs = torch.randn(n_fixed_z, latent_dim, device=device)
        all_images = []
        for z in fixed_zs:
            z_rep = z.unsqueeze(0).expand(num_classes, -1)
            labels = torch.arange(num_classes, device=device)
            imgs = model.generate(labels, z_rep)
            all_images.append(imgs.cpu())

    # Grid: n_fixed_z rows x num_classes cols
    fig, axes = plt.subplots(n_fixed_z, num_classes, figsize=(num_classes * 1.2, n_fixed_z * 1.2))
    for row_idx, row_imgs in enumerate(all_images):
        for col_idx in range(num_classes):
            img = denormalize(row_imgs[col_idx]).permute(1, 2, 0).numpy()
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    CIFAR10_CLASSES[col_idx][:4], fontsize=7
                )

    plt.suptitle("Class Variation (rows = fixed z, cols = class labels)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_interpolation(
    model,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
    path: Union[str, Path],
    n_steps: int = 10,
    n_pairs: int = 4,
):
    """
    Latent interpolation: smoothly interpolate between two z vectors using
    spherical linear interpolation (slerp). Uses the same class label throughout.
    A smooth interpolation indicates a well-structured latent space.
    """
    def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between unit sphere vectors."""
        z1_n = z1 / z1.norm(dim=-1, keepdim=True)
        z2_n = z2 / z2.norm(dim=-1, keepdim=True)
        dot = (z1_n * z2_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        # Handle near-parallel vectors
        mask = sin_omega.abs() < 1e-6
        interp = torch.where(
            mask,
            z1 + t * (z2 - z1),
            (torch.sin((1 - t) * omega) / sin_omega) * z1 +
            (torch.sin(t * omega) / sin_omega) * z2
        )
        return interp

    model.G.eval()
    steps = [i / (n_steps - 1) for i in range(n_steps)]

    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(n_steps * 1.1, n_pairs * 1.2))

    with torch.no_grad():
        for pair_idx in range(n_pairs):
            class_idx = pair_idx % num_classes
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)
            label = torch.tensor([class_idx], device=device)

            for step_idx, t in enumerate(steps):
                z_interp = slerp(z1, z2, t)
                img = model.generate(label, z_interp)[0].cpu()
                img = denormalize(img).permute(1, 2, 0).numpy()
                axes[pair_idx, step_idx].imshow(img)
                axes[pair_idx, step_idx].axis("off")
                if pair_idx == 0:
                    axes[pair_idx, step_idx].set_title(
                        f"t={t:.1f}", fontsize=7
                    )
            axes[pair_idx, 0].set_ylabel(
                CIFAR10_CLASSES[class_idx], fontsize=8, rotation=90
            )

    plt.suptitle("Latent Space Interpolation (spherical, same class label)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_hyperparameter_comparison(results: dict, path: Union[str, Path]):
    """
    Plot comparison of different hyperparameter configurations.
    results: {config_name: {"g_loss": [...], "d_loss": [...]}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for idx, (config_name, history) in enumerate(results.items()):
        epochs = range(1, len(history["g_loss"]) + 1)
        color = colors[idx % len(colors)]
        ax1.plot(epochs, history["g_loss"], label=config_name, color=color, linewidth=1.5)
        ax2.plot(epochs, history["d_loss"], label=config_name, color=color, linewidth=1.5)

    for ax, title in zip([ax1, ax2], ["Generator Loss", "Discriminator Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hyperparameter Search Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
