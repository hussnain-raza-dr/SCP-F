"""
Training script for conditional GAN (baseline or improved).

Usage:
    # Train improved model (default)
    python training/train.py --config config/improved_config.yaml

    # Train baseline
    python training/train.py --config config/baseline_config.yaml --arch baseline

    # Resume from checkpoint
    python training/train.py --config config/improved_config.yaml --resume checkpoints/epoch_50.pt

Example (Google Colab):
    !python training/train.py --config config/improved_config.yaml --arch improved
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataloader import get_dataloaders
from models.cgan import cGAN
from evaluation.visualize import (
    plot_training_curves,
    save_image_grid,
    plot_class_variation,
    plot_latent_interpolation,
)


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict, arch: str, resume_path: str = None):
    # ---- Setup ----
    seed = config["data"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Override architecture from CLI
    config["model"]["architecture"] = arch

    # ---- Data ----
    train_loader, _ = get_dataloaders(
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=config["data"].get("num_workers", 2),
    )
    print(f"Dataset: {len(train_loader.dataset)} training images, "
          f"{config['data']['num_classes']} classes")

    # ---- Model ----
    model = cGAN(config, device)
    g_params = sum(p.numel() for p in model.G.parameters())
    d_params = sum(p.numel() for p in model.D.parameters())
    print(f"Generator parameters:     {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"G/D parameter ratio:      {g_params/d_params:.2f}")
    if model.G_ema is not None:
        print(f"EMA enabled (decay={model.ema_decay})")

    start_epoch = 0
    if resume_path:
        start_epoch = model.load_checkpoint(resume_path)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    # ---- Output directories ----
    results_dir = Path("results")
    ckpt_dir = Path("checkpoints")
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Metrics log
    history = {
        "d_loss": [], "g_loss": [], "gp": [],
        "real_acc": [], "fake_acc": [],
        "lr_g": [], "lr_d": [],
    }

    num_epochs    = config["training"]["num_epochs"]
    n_critic      = config["training"].get("n_critic", 1)
    num_classes   = config["data"]["num_classes"]
    latent_dim    = config["model"]["latent_dim"]

    # Fixed noise + labels for consistent evaluation samples
    fixed_z = torch.randn(num_classes * 8, latent_dim, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat(8)

    print(f"\n{'='*60}")
    print(f"Training {arch} cGAN for {num_epochs} epochs")
    print(f"Loss: {config['training']['loss_type']}, n_critic={n_critic}")
    print(f"LR_G: {config['training']['lr_g']}, LR_D: {config['training']['lr_d']}")
    if config['training'].get('d_noise_std', 0) > 0:
        print(f"Instance noise: std={config['training']['d_noise_std']}, "
              f"decay over {config['training']['d_noise_decay_epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        d_losses, g_losses, gps, real_accs, fake_accs = [], [], [], [], []

        model.G.train()
        model.D.train()
        model.current_epoch = epoch  # For instance noise decay

        for batch_idx, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # ---- Train Discriminator n_critic times ----
            for _ in range(n_critic):
                d_metrics = model.train_discriminator(real_images, labels)
            d_losses.append(d_metrics["d_loss"])
            gps.append(d_metrics["gp"])
            real_accs.append(d_metrics["real_acc"])
            fake_accs.append(d_metrics["fake_acc"])

            # ---- Train Generator once ----
            g_metrics = model.train_generator(batch_size, labels)
            g_losses.append(g_metrics["g_loss"])

        # Step LR schedulers at end of each epoch
        model.step_schedulers()

        # ---- Epoch summary ----
        avg_d = np.mean(d_losses)
        avg_g = np.mean(g_losses)
        avg_gp = np.mean(gps)
        avg_racc = np.mean(real_accs)
        avg_facc = np.mean(fake_accs)
        elapsed = time.time() - epoch_start

        # Log current learning rates
        cur_lr_g = model.opt_G.param_groups[0]["lr"]
        cur_lr_d = model.opt_D.param_groups[0]["lr"]

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["gp"].append(avg_gp)
        history["real_acc"].append(avg_racc)
        history["fake_acc"].append(avg_facc)
        history["lr_g"].append(cur_lr_g)
        history["lr_d"].append(cur_lr_d)

        noise_info = ""
        if model.d_noise_std > 0:
            noise_info = f" | noise={model._get_noise_std():.4f}"

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"D: {avg_d:.4f} | G: {avg_g:.4f} | GP: {avg_gp:.4f} | "
              f"D_real: {avg_racc:.2%} | D_fake: {avg_facc:.2%} | "
              f"LR_G: {cur_lr_g:.6f}{noise_info} | "
              f"Time: {elapsed:.1f}s")

        # ---- Save samples every 10 epochs ----
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = model.generate(fixed_labels, fixed_z)
            save_image_grid(
                samples,
                results_dir / f"samples_epoch_{epoch+1:04d}.png",
                nrow=num_classes,
                title=f"Epoch {epoch+1}",
            )
            plot_training_curves(history, results_dir / "training_curves.png")

            # Save training history as JSON for later analysis
            history_serializable = {
                k: [float(v) for v in vals] for k, vals in history.items()
            }
            with open(results_dir / "training_history.json", "w") as f:
                json.dump(history_serializable, f, indent=2)

        # ---- Save checkpoint every 25 epochs ----
        if (epoch + 1) % 25 == 0:
            ckpt_path = ckpt_dir / f"{arch}_epoch_{epoch+1:04d}.pt"
            model.save_checkpoint(str(ckpt_path), epoch + 1, history)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ---- Final evaluation visuals ----
    print("\nGenerating final evaluation visualizations...")
    final_samples = model.generate(fixed_labels, fixed_z)
    save_image_grid(final_samples, results_dir / "final_samples.png", nrow=num_classes)

    plot_class_variation(model, latent_dim, num_classes, device,
                         results_dir / "class_variation.png")
    plot_latent_interpolation(model, latent_dim, num_classes, device,
                              results_dir / "latent_interpolation.png")
    plot_training_curves(history, results_dir / "training_curves.png")

    # Save final checkpoint
    model.save_checkpoint(
        str(ckpt_dir / f"{arch}_final.pt"), num_epochs, history
    )

    # Save final training history
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    print(f"\nTraining complete. Results saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train conditional GAN on CIFAR-10")
    parser.add_argument("--config", type=str, default="config/improved_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--arch", type=str, default="improved",
                        choices=["improved", "baseline"],
                        help="Architecture to use")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, arch=args.arch, resume_path=args.resume)


if __name__ == "__main__":
    main()
