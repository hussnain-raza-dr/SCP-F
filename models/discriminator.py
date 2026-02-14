"""
Improved Discriminator for conditional image discrimination on CIFAR-10.

Key design choices vs. baseline:
1. Spectral normalization on ALL layers.
   - Why: The main reason the baseline discriminator dominates the generator
     is that without Lipschitz constraints, it can grow its weights freely and
     become arbitrarily powerful. Spectral norm enforces a 1-Lipschitz constraint
     per layer, keeping the discriminator from running away from the generator.
     This is the single most impactful change for training stability.

2. Residual blocks for downsampling (no strided conv directly in residual path).
   - Why: Residual connections allow the gradient to flow directly from the
     discriminator's output back to early layers, preventing vanishing gradients
     and helping the discriminator give more informative feedback to the generator.

3. Projection conditioning (Miyato & Koyama, 2018) instead of concatenation.
   - Why: The projection discriminator's class conditioning is theoretically
     motivated and empirically stronger. The class embedding is projected to
     the final features via inner product, giving a clean conditional signal.

4. No Batch Normalization in the discriminator.
   - Why: For WGAN-GP, using batch norm in the discriminator interferes with the
     gradient penalty computation (the penalty is computed on interpolated samples
     and batch norm introduces batch-level dependencies). We use layer norm or
     no norm instead.

5. Leaky ReLU (slope 0.2) throughout for stable gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from models.generator import SelfAttention  # Shared attention module


class ResBlockDown(nn.Module):
    """
    Residual block with average pooling downsampling.
    All convolutions use spectral normalization.
    No batch norm (required for WGAN-GP).
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(x, 0.2)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        skip = self.shortcut(x)
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)

        return out + skip


class ImprovedDiscriminator(nn.Module):
    """
    Discriminator: 32x32 RGB image + class label -> real/fake score.

    Architecture:
      ResBlockDown 3   -> 64   (32->16)  [first block: no activation at start]
      SelfAttention at 16x16
      ResBlockDown 64  -> 128  (16->8)
      ResBlockDown 128 -> 256  (8->4)
      ResBlockDown 256 -> 256  (4->4, no spatial downsampling)
      GlobalSumPool -> Linear(256, 1)
      Projection conditioning: inner product with class embedding

    Note: Outputs raw logits (no sigmoid) — required for WGAN-GP.
    """
    def __init__(
        self,
        base_channels: int = 64,
        num_classes: int = 10,
        embed_dim: int = 256,
    ):
        super().__init__()

        # First conv (no activation before, as per SN-GAN practice)
        self.first_conv = spectral_norm(nn.Conv2d(3, base_channels, 3, padding=1))

        self.res1 = ResBlockDown(base_channels,      base_channels * 2,  downsample=True)   # 16x16
        self.attn = SelfAttention(base_channels * 2, use_sn=True)
        self.res2 = ResBlockDown(base_channels * 2,  base_channels * 4,  downsample=True)   # 8x8
        self.res3 = ResBlockDown(base_channels * 4,  base_channels * 8,  downsample=True)   # 4x4
        self.res4 = ResBlockDown(base_channels * 8,  base_channels * 8,  downsample=False)  # 4x4

        final_channels = base_channels * 8

        # Linear output for WGAN-GP (no sigmoid)
        self.linear = spectral_norm(nn.Linear(final_channels, 1))

        # Projection conditioning: class embedding dotted with feature vector
        # This is theoretically motivated and avoids simple concatenation
        self.embed = spectral_norm(nn.Embedding(num_classes, final_channels))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        h = self.first_conv(x)
        h = self.res1(h)          # 16x16
        h = self.attn(h)          # Self-attention
        h = self.res2(h)          # 8x8
        h = self.res3(h)          # 4x4
        h = self.res4(h)          # 4x4

        # Global mean pooling — sum pooling with ResBlocks produces feature
        # magnitudes ~450, making hinge margins (±1) meaningless. Mean pooling
        # normalizes by the spatial size (4×4=16), keeping outputs in a
        # reasonable range for hinge loss.
        h = F.leaky_relu(h, 0.2)
        h = h.mean(dim=[2, 3])    # (B, C)

        # Unconditional output
        out = self.linear(h).squeeze(1)

        # Projection conditioning: add inner product of features and class embedding
        class_emb = self.embed(labels)                     # (B, C)
        out = out + (h * class_emb).sum(dim=1)             # (B,)

        return out


# ---- Baseline Discriminator (for baseline training) ----

class BaselineDiscriminator(nn.Module):
    """
    Baseline conditional DCGAN discriminator as specified in the assignment.
    Strided convolutions + BatchNorm + LeakyReLU. Class embedding concatenated.
    """
    def __init__(self, embed_dim: int = 50, num_classes: int = 10):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)

        self.conv_layers = nn.Sequential(
            # 3 -> 64, 32->16
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 128, 16->8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 256, 8->4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 256*4*4 + embed_dim -> 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 + embed_dim, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        features_flat = features.view(x.size(0), -1)
        emb = self.embed(labels)
        combined = torch.cat([features_flat, emb], dim=1)
        return self.classifier(combined).squeeze(1)
