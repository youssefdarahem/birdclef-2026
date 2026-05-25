import random
import torch
import torch.nn.functional as F

from config import Config


class SpecAugment:
    """Time and frequency masking on log-mel spectrogram (1, n_mels, frames)."""

    def __init__(self, time_mask: int = 40, freq_mask: int = 20, num_masks: int = 2):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.num_masks = num_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        _, n_mels, n_frames = spec.shape
        spec = spec.clone()
        for _ in range(self.num_masks):
            # Frequency masking
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, f0 : f0 + f, :] = 0.0
            # Time masking
            t = random.randint(0, self.time_mask)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, t0 : t0 + t] = 0.0
        return spec


class BackgroundNoiseMixer:
    """Mix a random noise spectrogram at a given SNR range (dB)."""

    def __init__(self, noise_specs: list[torch.Tensor], snr_min: float = 5.0, snr_max: float = 15.0):
        self.noise_specs = noise_specs
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.noise_specs:
            return spec
        noise = random.choice(self.noise_specs)
        # Align frames
        if noise.shape[-1] != spec.shape[-1]:
            if noise.shape[-1] > spec.shape[-1]:
                start = random.randint(0, noise.shape[-1] - spec.shape[-1])
                noise = noise[..., start : start + spec.shape[-1]]
            else:
                noise = F.pad(noise, (0, spec.shape[-1] - noise.shape[-1]))

        snr_db = random.uniform(self.snr_min, self.snr_max)
        signal_power = spec.pow(2).mean().clamp(min=1e-9)
        noise_power  = noise.pow(2).mean().clamp(min=1e-9)
        snr_linear   = 10 ** (snr_db / 10)
        scale = (signal_power / (snr_linear * noise_power)).sqrt()
        return spec + scale * noise


def mixup(
    spec1: torch.Tensor,
    label1: torch.Tensor,
    spec2: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend two (spec, label) pairs. Returns new spec and soft label."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    spec  = lam * spec1  + (1 - lam) * spec2
    label = lam * label1 + (1 - lam) * label2
    return spec, label


class Augmentation:
    """Compose all augmentations. Accepts (spec, label) and returns (spec, label)."""

    def __init__(
        self,
        cfg: Config,
        noise_specs: list[torch.Tensor] | None = None,
        apply_mixup: bool = True,
    ):
        self.spec_aug = SpecAugment(cfg.spec_time_mask, cfg.spec_freq_mask)
        self.noise_mixer = BackgroundNoiseMixer(
            noise_specs or [],
            cfg.noise_snr_db_min,
            cfg.noise_snr_db_max,
        ) if noise_specs else None
        self.apply_mixup = apply_mixup
        self.mixup_alpha = cfg.mixup_alpha
        # Mixup requires a second sample; callers must pass one via batch-level mixup.
        # This class handles per-sample augmentations only.

    def __call__(self, spec: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.spec_aug(spec)
        if self.noise_mixer is not None:
            spec = self.noise_mixer(spec)
        return spec, label


def batch_mixup(
    specs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup across a batch. Shuffles the batch and mixes with lambda ~ Beta(alpha, alpha)."""
    batch_size = specs.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(batch_size, device=specs.device)
    mixed_specs  = lam * specs  + (1 - lam) * specs[perm]
    mixed_labels = lam * labels + (1 - lam) * labels[perm]
    return mixed_specs, mixed_labels
