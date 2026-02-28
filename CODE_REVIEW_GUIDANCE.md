# Code Review & Guidance: Getting Generation Working

## Diagnosis: What's Going Wrong

After reviewing the full codebase, training logs (200 epochs on Quadro RTX 5000), and generated samples,
the model suffers from **complete Tanh saturation** — every pixel is pushed to exactly -1 or +1,
producing only 8 pure colors (R, G, B, C, M, Y, W, K) instead of natural images.

### Evidence from training logs

| Metric | Epoch 1 | Epoch 50 | Epoch 100 | Epoch 200 |
|--------|---------|----------|-----------|-----------|
| D_loss | 1.17 | 0.90 | 0.64 | 0.38 |
| G_loss | 0.48 | 1.11 | 1.38 | 1.40 |
| D_real accuracy | 69% | 92% | 94% | 93% |
| D_fake accuracy | 82% | 75% | 83% | 95% |

**The discriminator wins completely.** D_loss drops continuously while G_loss climbs,
meaning the generator never learns to produce realistic images. By epoch 200, D classifies
both real and fake correctly >93% of the time — the generator has no useful gradient signal.

### Visual evidence

- **Epoch 1**: Random noise with some color blocks already appearing
- **Epoch 10**: Already dominated by yellow/white blocks — saturation begins immediately
- **Epoch 30**: Mostly white/yellow with sparse color patches — severe mode collapse
- **Epoch 50**: Pure color blocks (magenta, blue, yellow, cyan) — full Tanh saturation
- **Epoch 100–200**: Large flat color regions, zero recognizable structure

The saturation is visible from **epoch 1**, meaning this is an initialization/architecture problem,
not just a training dynamics issue.

---

## Root Causes (Prioritized)

### 1. CRITICAL: Tanh Saturation in Generator Output Layer

**File:** `models/generator.py:149-156`

The current output pipeline is:
```python
self.final_bn = nn.BatchNorm2d(base_channels // 4)  # 64 channels
self.final_conv = spectral_norm(nn.Conv2d(64, 3, 3, padding=1))
# forward: x = torch.tanh(self.final_conv(self.final_bn(x)))
```

**Why this saturates:**
- `final_bn` normalizes to ~N(0,1) per channel (64 channels).
- `final_conv` computes a weighted sum over 64 input channels x 3x3 kernel = 576 terms.
- Even with spectral norm (which only constrains the largest singular value to 1),
  the per-pixel output can still be a sum of hundreds of ~N(0,1) random variables.
- By the Central Limit Theorem, the output magnitude grows with the effective fan-in.
- This pushes Tanh inputs well beyond [-3, +3], causing permanent saturation.
- Once saturated, Tanh gradients are ~0, so the generator can never recover.

**The comment in the code claims "BN gives centered ~N(0,1) inputs, so the dot product
w·x involves cancellations → output stays ~N(0, ||w||²) ≈ N(0,1)". This analysis is wrong:**
the spectral norm constrains ||W||_op ≤ 1, not ||w||² for individual output neurons.
With 64 input channels, individual rows of W can have ||w_i||² >> 1 while ||W||_op = 1.

### 2. CRITICAL: Massive Discriminator-Generator Capacity Imbalance

**From training logs:**
```
Generator parameters:     2,474,692
Discriminator parameters: 9,829,922
G/D parameter ratio:      0.25
```

The discriminator has **4x more parameters** than the generator. Combined with:
- n_critic=2 (D trains twice per G step)
- D-favored TTUR (lr_d = 4x lr_g)
- Spectral norm everywhere in D (good for stability but makes D very efficient)

The discriminator easily overwhelms the generator. The generator never gets a chance
to learn before D becomes too powerful for any gradient signal to be useful.

### 3. HIGH: Orthogonal Initialization + SN Interaction

**File:** `models/generator.py:160-165`

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=1.0)
```

Orthogonal init with gain=1.0 on ALL layers (including the SN-wrapped final conv) is too aggressive.
For a Tanh output layer, the standard recommendation is `gain = nn.init.calculate_gain('tanh')` ≈ 5/3,
but that's for shallow networks. For this architecture, the final conv should use a **much smaller gain**
(e.g., 0.01–0.1) to ensure initial outputs are in the linear regime of Tanh (|x| < 1).

### 4. MEDIUM: Cosine LR Schedule Kills Generator Too Early

**File:** `config/improved_config.yaml:42`

The cosine schedule decays lr_g from 1e-4 → 1e-6 over 200 epochs:
- Epoch 100: lr_g ≈ 5e-5 (half the initial value)
- Epoch 150: lr_g ≈ 1.5e-5
- Epoch 175+: lr_g < 5e-6 (effectively dead)

Since the generator is already losing badly, this schedule removes its last chance to recover.
In the late phase where D might plateau, G's learning rate is too small to exploit that.

### 5. MEDIUM: Instance Noise Decays Too Fast

**File:** `config/improved_config.yaml:45-46`

Instance noise (std=0.1) decays to 0 by epoch 100. After that, the discriminator sees clean
images and its accuracy rapidly climbs from ~83% to ~95%. The noise was the only thing
preventing complete D dominance, and removing it halfway through training is too aggressive.

---

## Recommended Fixes (In Order of Priority)

### Fix 1: Prevent Tanh Saturation — Remove SN from Final Conv + Small Init

**The single most important fix.** The original Option A (scaling init with SN still
present) DOES NOT WORK because spectral norm re-normalizes weights at every forward
pass: `W_sn = W / σ(W)`. Scaling W by 0.1 just gives `0.1W / σ(0.1W) = W / σ(W)` —
the scaling is completely cancelled out.

**The correct fix is to remove SN from the final conv entirely:**

```python
# In __init__:
self.final_conv = nn.Conv2d(base_channels // 4, 3, 3, padding=1)  # No spectral_norm

# In _init_weights:
with torch.no_grad():
    self.final_conv.weight.mul_(0.1)  # Now this actually persists
```

**Why SN on the final conv causes saturation:**
- After BN, each of 64 input channels is ~N(0,1)
- The input vector per pixel has ||x||_2 ≈ sqrt(64) ≈ 8
- SN constrains ||W||_op = 1, so output ≤ ||W||_op * ||x||_2 ≈ 8
- Tanh(8) ≈ 1.0000 — permanent saturation, zero gradient

Without SN, the 0.1x scaling persists and initial pre-Tanh values stay in
the linear regime (|x| < 1). The optimizer can gradually grow weights as needed.

### Fix 2: Rebalance G/D Capacity

**Option A (preferred — less code change):** Reduce D capacity

In `config/improved_config.yaml`:
```yaml
model:
  discriminator_base_channels: 32    # was 64 → reduces D from 9.8M to ~2.5M params
```

**Option B:** Increase G capacity

In `config/improved_config.yaml`:
```yaml
model:
  generator_base_channels: 512    # was 256 → doubles G to ~9.8M params
```

**Option C:** Both — meet in the middle with D=48, G=384

Target a **G/D ratio between 0.5 and 1.5** (currently 0.25).

### Fix 3: Reduce Discriminator Advantage

In `config/improved_config.yaml`:
```yaml
training:
  n_critic: 1          # was 2. D is already dominant, don't give it extra steps
  lr_g: 0.0002         # was 0.0001. Bring closer to D's rate
  lr_d: 0.0002         # was 0.0004. Equal LR is fine with SN+hinge
```

The "BigGAN standard" of D-favored TTUR makes sense when G has comparable capacity.
With G/D = 0.25, the TTUR amplifies an already severe imbalance.

### Fix 4: Fix LR Schedule

**Option A:** Use constant LR for the first phase, then decay
```yaml
training:
  lr_schedule: "none"    # Start with constant LR until generation quality improves
```

**Option B:** Use linear warmup + cosine decay (requires code change in `cgan.py`)
```python
# Warmup for first 10 epochs, then cosine decay
if epoch < warmup_epochs:
    lr = base_lr * (epoch + 1) / warmup_epochs
else:
    lr = ... cosine decay ...
```

### Fix 5: Extend Instance Noise

In `config/improved_config.yaml`:
```yaml
training:
  d_noise_std: 0.15           # was 0.1
  d_noise_decay_epochs: 200   # was 100. Keep noise throughout training
```

---

## Recommended Experiment Plan

### Phase 1: Fix Saturation (1-2 runs, ~7 hours each)

1. **Run A (Critical fix only):** Apply Fix 1 (small-gain final layer, 0.1x init) +
   keep everything else the same. If this alone produces colors beyond the 8 pure values,
   the saturation diagnosis is confirmed.

2. **Run B (All fixes together):** Apply Fixes 1–5 simultaneously. This is the "kitchen sink"
   approach — combine all fixes and see if generation improves to recognizable shapes.

**Config for Run B** (`config/v10_config.yaml`):
```yaml
model:
  architecture: improved
  latent_dim: 128
  embed_dim: 64
  generator_base_channels: 256
  discriminator_base_channels: 32    # Fix 2: reduce D capacity
  use_self_attention: true
  use_residual: true

training:
  batch_size: 128
  num_epochs: 200
  lr_g: 0.0002                       # Fix 3: equal LR
  lr_d: 0.0002                       # Fix 3: equal LR
  beta1: 0.0
  beta2: 0.999
  n_critic: 1                        # Fix 3: no extra D steps
  loss_type: "hinge"
  grad_clip: 0.0
  ema_decay: 0.999
  lr_schedule: "none"                # Fix 4: no decay initially
  d_noise_std: 0.15                  # Fix 5: more noise
  d_noise_decay_epochs: 200          # Fix 5: noise throughout
  r1_gamma: 0.0
  r1_every: 16

data:
  num_classes: 10
  image_size: 32
  seed: 42
  num_workers: 2
```

Plus the code change for Fix 1 (0.1x final conv init).

### Phase 2: Hyperparameter Tuning (3-5 runs)

Once generation shows recognizable objects (even blurry), tune:

1. **G capacity sweep:** base_channels in {256, 384, 512}
2. **LR ratio sweep:** lr_g/lr_d in {1.0, 0.5, 0.25} (keeping lr_d = 2e-4)
3. **Loss function comparison:** hinge vs. WGAN-GP (n_critic=5, gp_lambda=10)
4. **Reintroduce cosine LR:** once baseline works, add cosine decay to sharpen late-training

### Phase 3: Architecture Tweaks (2-3 runs)

1. **Add DiffAugment** (augment both real AND fake images for D).
   This is a 10-line change and consistently helps on small datasets like CIFAR-10.
   Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training" (2020).

2. **Self-attention placement.** Currently at 8x8 in G and 16x16 in D.
   Try 16x16 in G (more spatial context for larger feature maps).

3. **Truncation trick at inference.** After training, sample z from a truncated normal
   (e.g., truncation=0.7) to trade diversity for quality.

---

## Additional Code-Level Issues

### Missing FID/IS Evaluation

The evaluation script (`evaluation/evaluate.py`) only computes discriminator accuracy,
which is **not a reliable metric** for generation quality (the D might be wrong about
what looks "real"). You should add:

- **FID (Frechet Inception Distance):** The standard metric. Use `pytorch-fid` or
  `clean-fid` library. Generate 10K–50K images and compare to CIFAR-10 test set.
  Good CIFAR-10 FID for conditional GAN: 10–30 (excellent < 15).

- **IS (Inception Score):** Measures both quality and diversity.
  Good CIFAR-10 IS: 7–9 (excellent > 8).

Add to `requirements.txt`:
```
pytorch-fid>=0.3.0
```

### Saturation Monitoring

Add a diagnostic to the training loop (`training/train.py`) to catch saturation early:

```python
# After generating samples each epoch, check saturation
with torch.no_grad():
    samples = model.generate(fixed_labels, fixed_z)
    sat_ratio = ((samples.abs() > 0.99).float().mean().item())
    print(f"  Saturation ratio: {sat_ratio:.2%}")  # Should be < 20%
```

If saturation ratio exceeds ~30%, the run is likely doomed — stop and adjust hyperparameters.

### Checkpoint Best Model by FID

Currently checkpoints are saved every 25 epochs regardless of quality.
Once FID is implemented, save the best model by FID score:
```python
if fid < best_fid:
    best_fid = fid
    model.save_checkpoint(str(ckpt_dir / f"{arch}_best_fid.pt"), epoch + 1, history)
```

### DiffAugment Implementation Sketch

A minimal DiffAugment for `data/dataloader.py` or inline in `cgan.py`:

```python
def diff_augment(x, policy='color,translation'):
    """Apply differentiable augmentations to both real and fake images."""
    if 'color' in policy:
        x = x + 0.1 * torch.randn(x.size(0), 3, 1, 1, device=x.device)
    if 'translation' in policy:
        # Random shift by up to 4 pixels
        shift = torch.randint(-4, 5, (x.size(0), 1, 2), device=x.device).float()
        grid = F.affine_grid(
            torch.eye(2, 3, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1) +
            torch.cat([torch.zeros_like(shift), shift], dim=1).view(-1, 2, 3) * 2/32,
            x.size(), align_corners=False
        )
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    return x
```

Apply in `train_discriminator()` to both real_noisy and fake_noisy.

---

## Summary: Priority Action Items

| Priority | Fix | Expected Impact | Effort |
|----------|-----|-----------------|--------|
| P0 | Fix 1: Small-gain final layer (0.1x init) | Eliminates Tanh saturation | 2 lines |
| P0 | Fix 2: D base_channels 64→32 | Rebalances G/D capacity | 1 line (config) |
| P1 | Fix 3: n_critic=1, equal LR | Reduces D advantage | 3 lines (config) |
| P1 | Fix 5: Extend instance noise to 200 epochs | Prevents late-training D dominance | 1 line (config) |
| P2 | Fix 4: Constant LR (no cosine decay) | Lets G learn throughout training | 1 line (config) |
| P2 | Add saturation monitoring | Catch failures early | ~5 lines |
| P3 | Add FID evaluation | Proper quality metric | ~30 lines + dependency |
| P3 | DiffAugment | Better data efficiency | ~20 lines |

**Start with P0 fixes.** If the saturation ratio drops below 30% and you see color gradients
(not just flat pure colors) in the epoch 10 samples, you're on the right track.
Then iterate on P1/P2/P3 fixes with the hyperparameter tuning plan above.
