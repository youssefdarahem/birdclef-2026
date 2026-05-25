import ast
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio.transforms as T

from config import Config, ALL_CLASSES, CLASS_TO_IDX, NUM_CLASSES
from audio_utils import build_mel_transform, load_clip_as_logmel, waveform_to_logmel, load_audio, crop_or_pad


def _parse_label_list(raw) -> list[str]:
    """Parse either a Python-repr list string or semicolon-separated string."""
    if isinstance(raw, list):
        return raw
    raw = str(raw).strip()
    if raw.startswith("["):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return [x.strip() for x in raw.split(";") if x.strip()]


def labels_to_multihot(labels: list[str]) -> torch.Tensor:
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in labels:
        idx = CLASS_TO_IDX.get(lbl)
        if idx is not None:
            vec[idx] = 1.0
    return vec


class BirdDataset(Dataset):
    """Individual species recordings from train_audio/."""

    def __init__(self, df: pd.DataFrame, cfg: Config, mel_transform: T.MelSpectrogram, augment=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mel = mel_transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.cfg.train_audio_dir / row["filename"]

        log_mel = load_clip_as_logmel(path, self.cfg, self.mel, offset_samples=-1)

        primary = [str(row["primary_label"])]
        secondary = _parse_label_list(row.get("secondary_labels", "[]"))
        label = labels_to_multihot(primary + secondary)

        if self.augment is not None:
            log_mel, label = self.augment(log_mel, label)

        return log_mel, label


class SoundscapeDataset(Dataset):
    """5-second windows from train_soundscapes/ with temporal labels."""

    def __init__(self, segments: list[dict], cfg: Config, mel_transform: T.MelSpectrogram, augment=None):
        self.segments = segments
        self.cfg = cfg
        self.mel = mel_transform
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        path = self.cfg.train_soundscapes_dir / seg["filename"]
        offset_samples = seg["start_sample"]

        waveform = load_audio(path, self.cfg)
        waveform = crop_or_pad(waveform, self.cfg.clip_samples, offset=offset_samples)
        log_mel = waveform_to_logmel(waveform, self.mel)

        label = labels_to_multihot(seg["labels"])

        if self.augment is not None:
            log_mel, label = self.augment(log_mel, label)

        return log_mel, label


def _parse_soundscape_labels(csv_path: Path, cfg: Config) -> list[dict]:
    """Convert train_soundscapes_labels.csv rows into segment dicts."""
    df = pd.read_csv(csv_path)
    segments = []
    for _, row in df.iterrows():
        start_str = str(row["start"])      # e.g. "00:00:00"
        h, m, s = start_str.split(":")
        start_sec = int(h) * 3600 + int(m) * 60 + int(s)
        labels = _parse_label_list(row["primary_label"])
        segments.append({
            "filename": row["filename"],
            "start_sample": start_sec * cfg.sample_rate,
            "labels": [str(l) for l in labels],
        })
    return segments


def build_datasets(cfg: Config, augment=None):
    """Return train_bird, val_soundscape, train_soundscape datasets."""
    random.seed(cfg.seed)

    # --- Individual recordings ---
    bird_df = pd.read_csv(cfg.train_csv)
    if cfg.debug:
        bird_df = bird_df.sample(min(cfg.max_debug_samples, len(bird_df)), random_state=cfg.seed)

    mel = build_mel_transform(cfg)

    # --- Soundscape segments ---
    segments = _parse_soundscape_labels(cfg.soundscape_labels_csv, cfg)
    if cfg.debug:
        segments = segments[: cfg.max_debug_samples]

    # Soundscape-level val split (hold out whole files to avoid leakage)
    all_sc_files = list({s["filename"] for s in segments})
    random.shuffle(all_sc_files)
    n_val = max(1, int(len(all_sc_files) * cfg.val_split))
    val_files = set(all_sc_files[:n_val])
    train_sc_files = set(all_sc_files[n_val:])

    train_segs = [s for s in segments if s["filename"] in train_sc_files]
    val_segs   = [s for s in segments if s["filename"] in val_files]

    train_bird = BirdDataset(bird_df, cfg, mel, augment=augment)
    train_sc   = SoundscapeDataset(train_segs, cfg, mel, augment=augment)
    val_sc     = SoundscapeDataset(val_segs,   cfg, mel, augment=None)

    return train_bird, train_sc, val_sc


def build_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Per-sample weight inversely proportional to class frequency (primary label only)."""
    counts = df["primary_label"].value_counts().to_dict()
    weights = [1.0 / counts.get(str(row["primary_label"]), 1) for _, row in df.iterrows()]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg: Config, augment=None):
    """Return train_loader, val_loader."""
    train_bird, train_sc, val_sc = build_datasets(cfg, augment)

    # Weighted sampler for individual recordings (handle class imbalance)
    bird_df = pd.read_csv(cfg.train_csv)
    if cfg.debug:
        bird_df = bird_df.sample(min(cfg.max_debug_samples, len(bird_df)), random_state=cfg.seed)
    sample_weights = build_class_weights(bird_df)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    bird_loader = DataLoader(
        train_bird,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    sc_loader = DataLoader(
        train_sc,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_sc,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return bird_loader, sc_loader, val_loader
