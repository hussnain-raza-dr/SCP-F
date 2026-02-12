"""
Systematic hyperparameter search for the improved cGAN.

Strategy: Manual staged search (more efficient than grid search for GANs).
Rationale: GAN training is sensitive to hyperparameter interactions, so we
search in stages, fixing the best value from each stage before proceeding.

Stage 1: Loss type & n_critic (most impactful on stability)
Stage 2: Learning rates (search independently for G and D)
Stage 3: Latent/embedding dimensions (secondary effects)
Stage 4: Gradient penalty lambda (fine-tuning)

Usage:
    python training/hyperparameter_search.py --stage 1 --epochs 30

Results are saved to results/hp_search/ for comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import random
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataloader import get_dataloaders
from models.cgan import cGAN
from evaluation.visualize import plot_hyperparameter_comparison


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BASE_CONFIG = {
    "model": {
        "architecture": "improved",
        "latent_dim": 128,
        "embed_dim": 64,
        "generator_base_channels": 256,
        "discriminator_base_channels": 64,
    },
    "training": {
        "batch_size": 64,
        "num_epochs": 30,
        "lr_g": 0.0001,
        "lr_d": 0.0004,
        "beta1": 0.0,
        "beta2": 0.9,
        "n_critic": 5,
        "gp_lambda": 10.0,
        "loss_type": "wgan-gp",
    },
    "data": {
        "num_classes": 10,
        "image_size": 32,
        "seed": 42,
        "num_workers": 2,
    },
}

# Search spaces for each stage
SEARCH_STAGES = {
    1: {
        "description": "Loss type and critic update ratio",
        "configs": {
            "wgan-gp_n1": {"training.loss_type": "wgan-gp", "training.n_critic": 1,
                            "training.beta1": 0.0, "training.lr_d": 0.0002},
            "wgan-gp_n5": {"training.loss_type": "wgan-gp", "training.n_critic": 5,
                            "training.beta1": 0.0, "training.lr_d": 0.0004},
            "lsgan_n1":   {"training.loss_type": "lsgan",   "training.n_critic": 1,
                            "training.beta1": 0.5, "training.lr_d": 0.0002},
            "vanilla_n1": {"training.loss_type": "vanilla",  "training.n_critic": 1,
                            "training.beta1": 0.5, "training.lr_d": 0.0002},
        }
    },
    2: {
        "description": "Learning rate search (G and D independently)",
        "configs": {
            "lr_1e-4_4e-4": {"training.lr_g": 1e-4, "training.lr_d": 4e-4},
            "lr_2e-4_2e-4": {"training.lr_g": 2e-4, "training.lr_d": 2e-4},
            "lr_5e-5_2e-4": {"training.lr_g": 5e-5, "training.lr_d": 2e-4},
            "lr_1e-4_8e-4": {"training.lr_g": 1e-4, "training.lr_d": 8e-4},
        }
    },
    3: {
        "description": "Latent and embedding dimension",
        "configs": {
            "z100_e50":  {"model.latent_dim": 100, "model.embed_dim": 50},
            "z128_e64":  {"model.latent_dim": 128, "model.embed_dim": 64},
            "z256_e128": {"model.latent_dim": 256, "model.embed_dim": 128},
        }
    },
    4: {
        "description": "Gradient penalty lambda",
        "configs": {
            "gp5":  {"training.gp_lambda": 5.0},
            "gp10": {"training.gp_lambda": 10.0},
            "gp20": {"training.gp_lambda": 20.0},
        }
    },
}


def set_nested(config: dict, key_path: str, value):
    """Set a nested config value using dot notation: 'training.lr_g'."""
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_single_experiment(config: dict, n_epochs: int, device: torch.device) -> dict:
    """Train for n_epochs and return loss history."""
    set_seed(config["data"]["seed"])
    train_loader, _ = get_dataloaders(
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=config["data"].get("num_workers", 2),
    )

    model = cGAN(config, device)
    n_critic = config["training"].get("n_critic", 1)

    history = {"d_loss": [], "g_loss": []}

    for epoch in range(n_epochs):
        d_losses, g_losses = [], []
        for real_images, labels in train_loader:
            real_images, labels = real_images.to(device), labels.to(device)
            batch_size = real_images.size(0)
            for _ in range(n_critic):
                d_m = model.train_discriminator(real_images, labels)
            g_m = model.train_generator(batch_size, labels)
            d_losses.append(d_m["d_loss"])
            g_losses.append(g_m["g_loss"])

        history["d_loss"].append(np.mean(d_losses))
        history["g_loss"].append(np.mean(g_losses))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: D={history['d_loss'][-1]:.4f}, "
                  f"G={history['g_loss'][-1]:.4f}")

    return history


def run_stage(stage: int, n_epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_info = SEARCH_STAGES[stage]
    print(f"\n{'='*60}")
    print(f"Stage {stage}: {stage_info['description']}")
    print(f"Training each config for {n_epochs} epochs on {device}")
    print(f"{'='*60}\n")

    results = {}
    out_dir = Path(f"results/hp_search/stage{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for config_name, overrides in stage_info["configs"].items():
        print(f"Config: {config_name}")
        config = copy.deepcopy(BASE_CONFIG)
        for key_path, value in overrides.items():
            set_nested(config, key_path, value)
        config["training"]["num_epochs"] = n_epochs

        history = run_single_experiment(config, n_epochs, device)
        results[config_name] = history
        print(f"  Final: D={history['d_loss'][-1]:.4f}, G={history['g_loss'][-1]:.4f}\n")

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_hyperparameter_comparison(results, out_dir / "comparison.png")
    print(f"Results saved to {out_dir}/")

    # Print ranking by final G loss (lower is harder for G = D is too strong,
    # we want G loss stable and moderate)
    print("\nConfig ranking by |G_loss final| (lower magnitude = more stable):")
    ranked = sorted(results.items(), key=lambda x: abs(x[1]["g_loss"][-1]))
    for rank, (name, hist) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: G={hist['g_loss'][-1]:.4f}, D={hist['d_loss'][-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for cGAN")
    parser.add_argument("--stage",  type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run_stage(args.stage, args.epochs)


if __name__ == "__main__":
    main()
