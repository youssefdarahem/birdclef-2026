"""Test-Time Augmentation (TTA) for inference.

Applies multiple deterministic augmentations to each spectrogram,
runs forward passes for each, then averages the sigmoid probabilities.
This reduces variance and consistently adds +0.5–2% cmAP.

TTA views used:
  1. Original
  2. Time shift right (+1s offset from original start)
  3. Time shift left  (-1s from end, i.e. trim last 1s and pad start)
  4. Frequency roll   (shift all mel bins up by 4 bins, wrap around)

Usage in inference:
    from tta import tta_predict
    probs = tta_predict(model, log_mel_spec)   # (num_classes,)
"""

import torch
import torch.nn.functional as F


def _time_shift(spec: torch.Tensor, shift_frames: int) -> torch.Tensor:
    """Shift spectrogram along time axis (roll, wrapping at edges).
    spec: (1, n_mels, T)
    """
    return torch.roll(spec, shift_frames, dims=-1)


def _freq_shift(spec: torch.Tensor, shift_bins: int) -> torch.Tensor:
    """Shift mel bins up/down by shift_bins (roll, wrapping).
    spec: (1, n_mels, T)
    """
    return torch.roll(spec, shift_bins, dims=-2)


def _time_mask_half(spec: torch.Tensor) -> torch.Tensor:
    """Zero out the first half of the time axis — forces attention to latter half."""
    s = spec.clone()
    half = s.shape[-1] // 2
    s[..., :half] = 0.0
    return s


TTA_TRANSFORMS = [
    lambda x: x,                          # 1. original
    lambda x: _time_shift(x, +50),        # 2. shift +50 frames (~1 s at hop=320)
    lambda x: _time_shift(x, -50),        # 3. shift -50 frames
    lambda x: _freq_shift(x, +4),         # 4. freq shift up 4 mel bins
]


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    spec: torch.Tensor,
    device: torch.device,
    transforms=TTA_TRANSFORMS,
) -> torch.Tensor:
    """Run TTA on a single spectrogram.

    Args:
        model:  trained model (eval mode)
        spec:   (1, n_mels, T) log-mel spectrogram
        device: torch device
        transforms: list of callables, each takes (1, n_mels, T) → (1, n_mels, T)

    Returns:
        probs: (num_classes,) averaged sigmoid probabilities
    """
    all_probs = []
    for tfm in transforms:
        aug = tfm(spec).unsqueeze(0).to(device)   # (1, 1, n_mels, T)
        logits = model(aug)                         # (1, num_classes)
        all_probs.append(torch.sigmoid(logits).squeeze(0).cpu())
    return torch.stack(all_probs).mean(dim=0)       # (num_classes,)


@torch.no_grad()
def tta_predict_batch(
    model: torch.nn.Module,
    specs: torch.Tensor,
    device: torch.device,
    transforms=TTA_TRANSFORMS,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Batched TTA for a stack of spectrograms.

    Args:
        specs: (W, 1, n_mels, T) — W windows from one soundscape file

    Returns:
        probs: (W, num_classes) averaged sigmoid probabilities
    """
    W = specs.shape[0]
    all_view_probs = []

    for tfm in transforms:
        aug_specs = torch.stack([tfm(specs[i]) for i in range(W)])  # (W, 1, n_mels, T)
        view_probs = []
        for i in range(0, W, chunk_size):
            batch  = aug_specs[i : i + chunk_size].to(device)
            logits = model(batch)
            view_probs.append(torch.sigmoid(logits).cpu())
        all_view_probs.append(torch.cat(view_probs, dim=0))   # (W, num_classes)

    return torch.stack(all_view_probs).mean(dim=0)   # (W, num_classes)
