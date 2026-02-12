# Conditional DCGAN on CIFAR-10

Conditional Generative Adversarial Network (cGAN) for class-conditioned image generation on the CIFAR-10 dataset.

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA-capable GPU recommended.

> **Note:** To install the GPU version of PyTorch, use:
> ```bash
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

## Training

Train the baseline conditional DCGAN:

```bash
python training/train.py --config config/baseline_config.yaml
```

Resume training from a checkpoint:

```bash
python training/train.py --config config/baseline_config.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

The CIFAR-10 dataset is downloaded automatically on first run.

## Evaluation

Generate samples and compute metrics from a trained checkpoint:

```bash
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint_final.pt --config config/baseline_config.yaml --output results/evaluation
```

This produces:
- **Sample grid** — 10 images per class
- **Latent space interpolation** — smooth transitions between two random z vectors
- **Class variation** — same z vector across all 10 classes
- **Discriminator confusion matrix** — per-class accuracy heatmap (real vs fake)
- **Confidence histogram** — overlapping distribution of D(x) on real vs fake images
- **Per-class generation quality** — bar chart of mean D(G(z)) score per class
- **Pixel statistics** — mean, std, min, max of generated image pixels
- **evaluation_metrics.json** — all metrics saved as JSON for programmatic access

## Configuration

All hyperparameters are controlled via YAML config files (`config/baseline_config.yaml`). This makes it easy to generate multiple configs for hyperparameter tuning.

**Model:** `latent_dim`, `embed_dim`, `num_classes`, `image_channels`, `image_size`, `generator_channels`, `discriminator_channels`, `leaky_relu_slope`, `init_std`

**Training:** `batch_size`, `num_epochs`, `lr_g`, `lr_d`, `beta1`, `beta2`, `weight_decay_g`, `weight_decay_d`, `d_steps`, `g_steps`, `label_smooth_real`, `label_smooth_fake`

**Data augmentation** (all toggleable):
- `horizontal_flip` — random horizontal flip
- `random_rotation` — random rotation by configurable degrees
- `color_jitter` — random brightness, contrast, saturation, hue
- `random_resized_crop` — random crop and resize with configurable scale range

**Evaluation:** `save_every`, `sample_every`, `num_sample_per_class`, `log_every`

## Architecture

**Generator:** Noise z (100-d) + class embedding (50-d) → project to 256×4×4 → ConvTranspose2d upsampling to 32×32×3 (Tanh output).

**Discriminator:** 32×32×3 input → strided Conv2d downsampling to 256×4×4 → flatten → projection discriminator with class embedding.

**Training:** Vanilla GAN loss (BCEWithLogitsLoss), Adam optimizer (β1=0.5, β2=0.999), lr=2e-4, batch size 128.
