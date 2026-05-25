import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from pathlib import Path

from config import Config


def load_audio(path: Path, cfg: Config) -> torch.Tensor:
    """Load OGG and return mono waveform (1, T) at cfg.sample_rate.

    Backend priority:
      1. soundfile  — fast, no FFmpeg needed (pip install soundfile)
      2. librosa    — pre-installed on Kaggle, handles OGG natively
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)      # (channels, T)
    except Exception:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]           # (1, T)
        waveform = torch.from_numpy(data.astype(np.float32))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        import torchaudio.functional as F_audio
        waveform = F_audio.resample(waveform, sr, cfg.sample_rate)
    return waveform  # (1, T)


def crop_or_pad(waveform: torch.Tensor, target_samples: int, offset: int = -1) -> torch.Tensor:
    """Crop or zero-pad waveform to exactly target_samples.

    offset=-1 means random start; otherwise use the given sample offset.
    """
    length = waveform.shape[-1]
    if length >= target_samples:
        if offset < 0:
            max_start = length - target_samples
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = min(offset, length - target_samples)
        waveform = waveform[..., start : start + target_samples]
    else:
        pad = target_samples - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def build_mel_transform(cfg: Config) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=2.0,
        center=False,       # avoids off-by-one frame (500 frames exact for 5s @ 32kHz hop=320)
    )


_amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)


def waveform_to_logmel(waveform: torch.Tensor, mel_transform: T.MelSpectrogram) -> torch.Tensor:
    """Convert waveform (1, T) → log-mel spectrogram (1, n_mels, frames), normalised."""
    mel = mel_transform(waveform)           # (1, n_mels, frames)
    log_mel = _amplitude_to_db(mel)         # dB scale
    # Per-sample normalisation: zero mean, unit std
    mean = log_mel.mean()
    std = log_mel.std().clamp(min=1e-6)
    log_mel = (log_mel - mean) / std
    return log_mel


def load_clip_as_logmel(
    path: Path,
    cfg: Config,
    mel_transform: T.MelSpectrogram,
    offset_samples: int = -1,
) -> torch.Tensor:
    """Full pipeline: file path → (1, n_mels, frames) log-mel tensor."""
    waveform = load_audio(path, cfg)
    waveform = crop_or_pad(waveform, cfg.clip_samples, offset=offset_samples)
    return waveform_to_logmel(waveform, mel_transform)


def soundscape_windows(
    path: Path,
    cfg: Config,
    mel_transform: T.MelSpectrogram,
) -> list[tuple[int, torch.Tensor]]:
    """Slide non-overlapping 5s windows over a full soundscape file.

    Returns list of (end_second, log_mel_tensor) pairs matching submission row_id format.
    """
    waveform = load_audio(path, cfg)
    total = waveform.shape[-1]
    step = cfg.clip_samples
    results = []
    start = 0
    while start + step <= total:
        chunk = waveform[..., start : start + step]
        log_mel = waveform_to_logmel(chunk, mel_transform)
        end_sec = (start + step) // cfg.sample_rate
        results.append((end_sec, log_mel))
        start += step
    return results
