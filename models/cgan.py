"""
Complete conditional GAN wrapper.
Handles model creation, optimizer setup, and single forward steps for both
the improved and baseline architectures.

v2 improvements:
- Exponential Moving Average (EMA) of generator weights for inference
- Cosine LR scheduling for both G and D
- Instance noise injection on discriminator inputs (decays over training)
"""

import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.generator import ImprovedGenerator, BaselineGenerator
from models.discriminator import ImprovedDiscriminator, BaselineDiscriminator
from training.losses import GANLoss, gradient_penalty, r1_penalty


class cGAN(nn.Module):
    """
    Conditional GAN wrapper: bundles generator, discriminator, optimizers, and
    loss computation into a single convenient object.
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        model_cfg    = config["model"]
        training_cfg = config["training"]
        data_cfg     = config["data"]

        self.loss_type  = training_cfg.get("loss_type", "wgan-gp")
        self.n_critic   = training_cfg.get("n_critic", 5)
        self.gp_lambda  = training_cfg.get("gp_lambda", 10.0)
        self.latent_dim = model_cfg["latent_dim"]
        self.num_classes = data_cfg["num_classes"]

        # Instance noise settings
        self.d_noise_std = training_cfg.get("d_noise_std", 0.0)
        self.d_noise_decay_epochs = training_cfg.get("d_noise_decay_epochs", 100)
        self.current_epoch = 0

        # Lazy gradient penalty: only compute GP every N D steps (StyleGAN2 trick)
        self.gp_every = training_cfg.get("gp_every", 1)
        self.d_step_count = 0

        # R1 gradient penalty (StyleGAN2): penalizes ||∇D(real)||² to prevent
        # D from becoming overconfident. Applied lazily every r1_every steps.
        self.r1_gamma = training_cfg.get("r1_gamma", 0.0)
        self.r1_every = training_cfg.get("r1_every", 16)

        # Build models
        arch = model_cfg.get("architecture", "improved")
        if arch == "improved":
            self.G = ImprovedGenerator(
                latent_dim=model_cfg["latent_dim"],
                embed_dim=model_cfg["embed_dim"],
                base_channels=model_cfg.get("generator_base_channels", 256),
                num_classes=self.num_classes,
            ).to(device)
            self.D = ImprovedDiscriminator(
                base_channels=model_cfg.get("discriminator_base_channels", 64),
                num_classes=self.num_classes,
            ).to(device)
        else:  # baseline
            self.G = BaselineGenerator(
                latent_dim=model_cfg["latent_dim"],
                embed_dim=model_cfg["embed_dim"],
                num_classes=self.num_classes,
            ).to(device)
            self.D = BaselineDiscriminator(
                embed_dim=model_cfg["embed_dim"],
                num_classes=self.num_classes,
            ).to(device)

        # Optimizers — TTUR: separate learning rates for G and D
        self.opt_G = Adam(
            self.G.parameters(),
            lr=training_cfg["lr_g"],
            betas=(training_cfg["beta1"], training_cfg["beta2"]),
        )
        self.opt_D = Adam(
            self.D.parameters(),
            lr=training_cfg["lr_d"],
            betas=(training_cfg["beta1"], training_cfg["beta2"]),
        )

        # LR scheduling
        num_epochs = training_cfg.get("num_epochs", 200)
        lr_schedule = training_cfg.get("lr_schedule", "none")
        if lr_schedule == "cosine":
            self.sched_G = CosineAnnealingLR(self.opt_G, T_max=num_epochs, eta_min=1e-6)
            self.sched_D = CosineAnnealingLR(self.opt_D, T_max=num_epochs, eta_min=1e-6)
        else:
            self.sched_G = None
            self.sched_D = None

        self.criterion = GANLoss(self.loss_type)
        self.grad_clip = training_cfg.get("grad_clip", 0.0)

        # EMA (Exponential Moving Average) of generator weights
        self.ema_decay = training_cfg.get("ema_decay", 0.0)
        if self.ema_decay > 0:
            self.G_ema = copy.deepcopy(self.G)
            self.G_ema.eval()
            for p in self.G_ema.parameters():
                p.requires_grad_(False)
        else:
            self.G_ema = None

    # ------------------------------------------------------------------
    # Instance noise (decays linearly to 0)
    # ------------------------------------------------------------------

    def _get_noise_std(self) -> float:
        """Get current instance noise std (linearly decays to 0)."""
        if self.d_noise_std <= 0:
            return 0.0
        decay = max(0.0, 1.0 - self.current_epoch / self.d_noise_decay_epochs)
        return self.d_noise_std * decay

    def _add_instance_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to images for discriminator input."""
        std = self._get_noise_std()
        if std > 0:
            noise = torch.randn_like(images) * std
            return images + noise
        return images

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA generator weights after each G step."""
        if self.G_ema is None:
            return
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_discriminator(
        self, real_images: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """One discriminator update step. Returns loss components."""
        self.opt_D.zero_grad()
        batch_size = real_images.size(0)

        # Add instance noise to real images
        real_noisy = self._add_instance_noise(real_images)
        real_scores = self.D(real_noisy, labels)

        # Generate fakes (no grad needed for G here)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        with torch.no_grad():
            fake_images = self.G(z, labels)

        # Add instance noise to fake images
        fake_noisy = self._add_instance_noise(fake_images)
        fake_scores = self.D(fake_noisy.detach(), labels)

        # Adversarial loss
        d_loss = self.criterion.d_loss(real_scores, fake_scores)

        # Lazy gradient penalty: only compute every gp_every D steps
        gp = torch.tensor(0.0, device=self.device)
        r1 = torch.tensor(0.0, device=self.device)
        self.d_step_count += 1
        total_d_loss = d_loss

        compute_gp = (self.loss_type == "wgan-gp" and
                      self.d_step_count % self.gp_every == 0)
        if compute_gp:
            gp = gradient_penalty(
                self.D, real_images, fake_images, labels,
                self.device, lambda_gp=self.gp_lambda
            )
            total_d_loss = total_d_loss + gp

        # R1 gradient penalty: penalize large gradients on real data.
        # Applied every r1_every steps for efficiency, WITHOUT scaling by
        # r1_every (the lazy scaling from StyleGAN2 causes destructive
        # gradient spikes with hinge loss — hinge is bounded ~0-2, so a
        # 16x spike overwhelms the adversarial signal and kills D).
        compute_r1 = (self.r1_gamma > 0 and
                      self.d_step_count % self.r1_every == 0)
        if compute_r1:
            r1 = r1_penalty(self.D, real_images, labels)
            total_d_loss = total_d_loss + (self.r1_gamma / 2.0) * r1

        total_d_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
        self.opt_D.step()

        # Discriminator accuracy (for logging)
        with torch.no_grad():
            if self.loss_type == "vanilla":
                real_acc = (torch.sigmoid(real_scores) > 0.5).float().mean().item()
                fake_acc = (torch.sigmoid(fake_scores) < 0.5).float().mean().item()
            else:
                real_acc = (real_scores > 0).float().mean().item()
                fake_acc = (fake_scores < 0).float().mean().item()

        return {
            "d_loss": d_loss.item(),
            "gp": gp.item(),
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }

    def train_generator(self, batch_size: int, labels: torch.Tensor) -> dict:
        """One generator update step. Returns loss."""
        self.opt_G.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.G(z, labels)
        fake_scores = self.D(fake_images, labels)

        g_loss = self.criterion.g_loss(fake_scores)
        g_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
        self.opt_G.step()

        # Update EMA after each G step
        self._update_ema()

        return {"g_loss": g_loss.item()}

    def step_schedulers(self):
        """Step LR schedulers at end of each epoch."""
        if self.sched_G is not None:
            self.sched_G.step()
        if self.sched_D is not None:
            self.sched_D.step()

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, z: torch.Tensor = None,
                 use_ema: bool = True) -> torch.Tensor:
        """Generate images for given labels. Uses EMA generator if available."""
        gen = self.G_ema if (use_ema and self.G_ema is not None) else self.G
        gen.eval()
        if z is None:
            z = torch.randn(len(labels), self.latent_dim, device=self.device)
        images = gen(z, labels)
        self.G.train()
        return images

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        state = {
            "epoch": epoch,
            "G_state": self.G.state_dict(),
            "D_state": self.D.state_dict(),
            "opt_G_state": self.opt_G.state_dict(),
            "opt_D_state": self.opt_D.state_dict(),
            "metrics": metrics,
        }
        if self.G_ema is not None:
            state["G_ema_state"] = self.G_ema.state_dict()
        if self.sched_G is not None:
            state["sched_G_state"] = self.sched_G.state_dict()
            state["sched_D_state"] = self.sched_D.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G_state"])
        self.opt_D.load_state_dict(ckpt["opt_D_state"])
        if self.G_ema is not None and "G_ema_state" in ckpt:
            self.G_ema.load_state_dict(ckpt["G_ema_state"])
        if self.sched_G is not None and "sched_G_state" in ckpt:
            self.sched_G.load_state_dict(ckpt["sched_G_state"])
            self.sched_D.load_state_dict(ckpt["sched_D_state"])
        return ckpt["epoch"]
