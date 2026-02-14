"""
Improved Generator for conditional image generation on CIFAR-10 (32x32).

Key design choices vs. baseline:
1. Residual blocks instead of plain ConvTranspose2d chains.
   - Why: Residual connections prevent gradient vanishing in deep generators,
     allowing the network to learn identity mappings and refine features
     rather than having to reconstruct them from scratch at each layer.

2. Self-attention at the 8x8 spatial level.
   - Why: Convolutions are local operations. For CIFAR-10, inter-region
     consistency (e.g., coherent object shape) requires long-range dependencies.
     Self-attention lets the generator attend to distant spatial locations.

3. NO spectral normalization in the generator (SN is for D only).
   - Why: Standard in BigGAN/SN-GAN. ConditionalBN handles normalization in G.
     SN on G limits expressivity and causes Tanh saturation at the output layer.

4. Pixel shuffle (sub-pixel convolution) for the final upsampling step.
   - Why: Avoids checkerboard artifacts common with ConvTranspose2d.
     PixelShuffle rearranges channels into spatial resolution cleanly.

5. Class conditioning via projection (not concatenation).
   - Why: Projection conditioning (embedding projected into a scale/shift for
     each residual block) is more expressive than simple concatenation. Used
     in BigGAN and shown to improve class fidelity significantly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm  # Used by discriminator (imports SelfAttention)


class SelfAttention(nn.Module):
    """
    Self-attention module from SAGAN (Zhang et al., 2018).
    Applied at a specific spatial scale to capture global structure.

    Note: no spectral norm here — SN is only for the discriminator.
    When used in D, the D wraps its own attention convs with SN.
    """
    def __init__(self, in_channels: int, use_sn: bool = False):
        super().__init__()
        wrap = spectral_norm if use_sn else (lambda m: m)
        self.query = wrap(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key   = wrap(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = wrap(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))  # Learned mixing weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C//8
        k = self.key(x).view(B, -1, H * W)                      # B, C//8, HW
        attn = F.softmax(torch.bmm(q, k), dim=-1)               # B, HW, HW
        v = self.value(x).view(B, -1, H * W)                    # B, C, HW
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization: learns separate gamma/beta per class.
    This is more expressive than concatenating the embedding to the noise vector,
    as it conditions every layer, not just the input.
    """
    def __init__(self, num_features: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta  = nn.Embedding(num_classes, num_features)
        # Initialize to identity transform
        nn.init.ones_(self.embed_gamma.weight)
        nn.init.zeros_(self.embed_beta.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.embed_gamma(labels).view(-1, out.shape[1], 1, 1)
        beta  = self.embed_beta(labels).view(-1, out.shape[1], 1, 1)
        return gamma * out + beta


class ResBlockUp(nn.Module):
    """
    Residual block with bilinear upsampling + Conv2d (avoids checkerboard artifacts
    that come with ConvTranspose2d).
    Uses Conditional Batch Normalization for class conditioning at every layer.
    """
    def __init__(self, in_channels: int, out_channels: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes, embed_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes, embed_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Shortcut: 1x1 conv to match channels, applied after upsampling
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Main path
        out = F.relu(self.cbn1(x, labels))
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = F.relu(self.cbn2(out, labels))
        out = self.conv2(out)

        # Shortcut path
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.shortcut(skip)

        return out + skip


class ImprovedGenerator(nn.Module):
    """
    Generator: z (latent) + class label -> 32x32 RGB image.

    Architecture:
      Linear -> reshape to 4x4 feature map
      ResBlockUp 256->256  (4->8)
      SelfAttention at 8x8
      ResBlockUp 256->128  (8->16)
      ResBlockUp 128->64   (16->32)
      BN + Conv2d(SN, 64, 3) + Tanh  (no ReLU — centered inputs prevent saturation)
    """
    def __init__(
        self,
        latent_dim: int = 128,
        embed_dim: int = 64,
        base_channels: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Project z to spatial feature map (no SN — ConditionalBN normalizes in res1)
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)

        # Residual upsampling blocks
        self.res1 = ResBlockUp(base_channels,      base_channels,      num_classes, embed_dim)
        self.res2 = ResBlockUp(base_channels,      base_channels // 2, num_classes, embed_dim)
        self.res3 = ResBlockUp(base_channels // 2, base_channels // 4, num_classes, embed_dim)

        # Self-attention after first upsampling (8x8 resolution)
        self.attn = SelfAttention(base_channels)

        # Final output layer: BN → Conv(SN) → Tanh.
        # SN on the final conv is the ONLY SN in G — it structurally prevents
        # Tanh saturation. No ReLU: BN gives centered ~N(0,1) inputs, so the
        # dot product w·x involves cancellations → output stays ~N(0, ||w||²)
        # ≈ N(0,1). With ReLU, all-positive inputs would sum constructively
        # → output magnitude ~sqrt(fan_in) → permanent Tanh saturation.
        self.final_bn = nn.BatchNorm2d(base_channels // 4)
        self.final_conv = spectral_norm(nn.Conv2d(base_channels // 4, 3, 3, padding=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Project and reshape: (B, latent_dim) -> (B, C, 4, 4)
        x = self.fc(z).view(-1, self.base_channels, 4, 4)

        # Residual upsampling: 4->8->16->32
        x = self.res1(x, labels)   # 8x8
        x = self.attn(x)           # Self-attention at 8x8
        x = self.res2(x, labels)   # 16x16
        x = self.res3(x, labels)   # 32x32

        # Output: BN → Conv(SN) → Tanh (no ReLU — keeps inputs centered)
        x = self.final_bn(x)
        x = torch.tanh(self.final_conv(x))
        return x


# ---- Baseline Generator (for comparison / baseline training) ----

class BaselineGenerator(nn.Module):
    """
    Baseline conditional DCGAN generator as specified in the assignment.
    Uses ConvTranspose2d + BatchNorm + ReLU, class embedding concatenated to z.
    """
    def __init__(
        self,
        latent_dim: int = 100,
        embed_dim: int = 50,
        num_classes: int = 10,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        input_dim = latent_dim + embed_dim

        self.net = nn.Sequential(
            # Project: (nz+nembed) -> 256 x 4 x 4
            nn.Linear(input_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.embed(labels)
        x = torch.cat([z, emb], dim=1)
        return self.net(x)
