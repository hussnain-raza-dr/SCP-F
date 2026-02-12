"""
Loss functions for cGAN training.

Four options are provided:
1. Hinge loss (SN-GAN) — recommended for spectral-normalized models
   - Natural pairing with spectral normalization: no gradient penalty needed
   - Reference: Miyato et al., "Spectral Normalization for GANs" (2018)
   - Also used by BigGAN (Brock et al., 2019)

2. WGAN-GP (Wasserstein GAN with Gradient Penalty)
   - Addresses mode collapse and training instability by optimizing Earth Mover's distance
   - Gradient penalty enforces Lipschitz constraint without weight clipping
   - Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)

3. LSGAN (Least Squares GAN) — alternative stable option
   - Uses MSE loss instead of BCE, avoids vanishing gradients in early training
   - Reference: Mao et al., "Least Squares Generative Adversarial Networks" (2017)

4. Vanilla GAN (BCE) — baseline, known to be unstable
   - Standard minimax game from Goodfellow et al. (2014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---- Hinge Loss (SN-GAN / BigGAN) ------------------------------------------

def hinge_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """Hinge loss for D: penalizes real scores < 1 and fake scores > -1."""
    return (F.relu(1.0 - real_scores).mean() +
            F.relu(1.0 + fake_scores).mean())


def hinge_g_loss(fake_scores: Tensor) -> Tensor:
    """Hinge loss for G: maximize fake scores."""
    return -fake_scores.mean()


# ---- WGAN-GP ---------------------------------------------------------------

def wgan_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """Discriminator loss: maximize real - fake scores."""
    return fake_scores.mean() - real_scores.mean()


def wgan_g_loss(fake_scores: Tensor) -> Tensor:
    """Generator loss: minimize negative fake scores (maximize fake scores)."""
    return -fake_scores.mean()


def gradient_penalty(
    discriminator: nn.Module,
    real_images: Tensor,
    fake_images: Tensor,
    labels: Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tensor:
    """
    Computes the gradient penalty for WGAN-GP.

    Interpolates between real and fake images, computes discriminator output,
    then penalizes the gradient norm if it deviates from 1.

    Mathematical motivation:
        For the Wasserstein distance to be valid, the discriminator (critic)
        must be 1-Lipschitz. Instead of weight clipping (which limits capacity),
        GP adds a soft constraint: (||∇D(x̂)||₂ - 1)² averaged over interpolated samples.
    """
    batch_size = real_images.size(0)

    # Random interpolation coefficient alpha in [0, 1]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    # Discriminator scores on interpolated images
    interp_scores = discriminator(interpolated, labels)

    # Compute gradients of scores w.r.t. interpolated images
    gradients = torch.autograd.grad(
        outputs=interp_scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients and compute L2 norm per sample
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty


# ---- LSGAN -----------------------------------------------------------------

def lsgan_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """LSGAN discriminator: MSE toward 1 for real, 0 for fake."""
    return 0.5 * (F.mse_loss(real_scores, torch.ones_like(real_scores)) +
                  F.mse_loss(fake_scores, torch.zeros_like(fake_scores)))


def lsgan_g_loss(fake_scores: Tensor) -> Tensor:
    """LSGAN generator: MSE toward 1 for fake scores."""
    return 0.5 * F.mse_loss(fake_scores, torch.ones_like(fake_scores))


# ---- Vanilla GAN (BCE) -----------------------------------------------------

def vanilla_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """Standard BCE discriminator loss."""
    real_loss = F.binary_cross_entropy_with_logits(
        real_scores, torch.ones_like(real_scores)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_scores, torch.zeros_like(fake_scores)
    )
    return (real_loss + fake_loss) * 0.5


def vanilla_g_loss(fake_scores: Tensor) -> Tensor:
    """Non-saturating generator loss (log D(G(z)) instead of -log(1-D(G(z))))."""
    return F.binary_cross_entropy_with_logits(
        fake_scores, torch.ones_like(fake_scores)
    )


# ---- Dispatcher ------------------------------------------------------------

class GANLoss:
    """Unified interface for different GAN loss types."""

    VALID_TYPES = ("hinge", "wgan-gp", "lsgan", "vanilla")

    def __init__(self, loss_type: str = "hinge"):
        assert loss_type in self.VALID_TYPES, \
            f"Unknown loss type: {loss_type}. Valid: {self.VALID_TYPES}"
        self.loss_type = loss_type

    def d_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        if self.loss_type == "hinge":
            return hinge_d_loss(real_scores, fake_scores)
        elif self.loss_type == "wgan-gp":
            return wgan_d_loss(real_scores, fake_scores)
        elif self.loss_type == "lsgan":
            return lsgan_d_loss(real_scores, fake_scores)
        else:
            return vanilla_d_loss(real_scores, fake_scores)

    def g_loss(self, fake_scores: Tensor) -> Tensor:
        if self.loss_type == "hinge":
            return hinge_g_loss(fake_scores)
        elif self.loss_type == "wgan-gp":
            return wgan_g_loss(fake_scores)
        elif self.loss_type == "lsgan":
            return lsgan_g_loss(fake_scores)
        else:
            return vanilla_g_loss(fake_scores)
