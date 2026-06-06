"""Generate pseudo-labels from a model on all train soundscapes.

Runs sliding-window inference on every .ogg in train_soundscapes/, thresholds
predictions, and writes a CSV in the same format as train_soundscapes_labels.csv.
Windows where no species exceeds the threshold are skipped (silence / low confidence).

The output CSV is merged with the original labels so that:
  - Verified human labels are kept as-is
  - Unlabeled windows with confident predictions get pseudo-labels
  - Already-labeled windows are NOT overwritten by pseudo-labels

Usage
-----
    # Full 3-model ensemble from checkpoints (CUDA-compatible, recommended)
    python src/pseudo_label.py \\
        --checkpoints checkpoints/exp001_baseline/best.pt \\
                      checkpoints/exp005_panns/best.pt \\
                      checkpoints/exp006_ast/best.pt \\
        --weights 0.25 0.35 0.40 \\
        --threshold 0.75 \\
        --output train_soundscapes_pseudo_soft.csv

    # CPU-traced model (use for ensemble_v2_no_tta.pt which has AST — CUDA tracing unsupported)
    python src/pseudo_label.py \\
        --traced checkpoints/ensemble_baseline_panns.pt \\
        --threshold 0.75

    # Hard labels only (old behaviour, no soft_probs column)
    python src/pseudo_label.py \\
        --checkpoints checkpoints/exp005_panns/best.pt \\
        --threshold 0.85 --no-soft

    # Then train with the augmented label set:
    python src/train.py --model efficientnet_b3 --experiment exp008_soft_pseudo \\
        --pseudo-labels train_soundscapes_pseudo_soft.csv \\
        --focal-gamma 2.0 --min-rating 3.5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, ALL_CLASSES, NUM_CLASSES


def _build_mel(cfg: Config):
    return T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max,
        power=2.0, center=False,
    ), T.AmplitudeToDB(stype="power", top_db=80.0)


def _load_wav(path: Path, cfg: Config) -> torch.Tensor:
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
    except Exception:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        wav = torch.from_numpy(data.astype(np.float32))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != cfg.sample_rate:
        import torchaudio.functional as F_a
        wav = F_a.resample(wav, sr, cfg.sample_rate)
    return wav


def _wav_to_logmel(wav: torch.Tensor, mel_tf, db_tf) -> torch.Tensor:
    mel = db_tf(mel_tf(wav))
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def _seconds_to_hms(s: int) -> str:
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _build_predict_fn(args, device) -> "callable[[torch.Tensor], torch.Tensor]":
    """Return a callable (batch) → probs (B, NUM_CLASSES) on CPU."""
    if args.traced:
        model = torch.jit.load(args.traced, map_location=device)
        model.eval()
        print(f"Loaded traced model: {args.traced}")
        # Traced ensemble outputs logit(avg_probs) so sigmoid recovers probabilities
        def predict(batch: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(model(batch.to(device))).cpu()
    else:
        from models import build_model
        weights_raw = args.weights or [1.0] * len(args.checkpoints)
        total = sum(weights_raw)
        weights = [w / total for w in weights_raw]

        members = []
        for ckpt_path, w in zip(args.checkpoints, weights):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg_saved  = ckpt.get("cfg")
            model_name = getattr(cfg_saved, "model_name", "efficientnet_b3")
            m = build_model(model_name, NUM_CLASSES, pretrained=False)
            m.load_state_dict(ckpt["model"])
            m.to(device).eval()
            members.append((m, w))
            cmap = ckpt.get("best_cmap", 0.0)
            print(f"  Loaded {Path(ckpt_path).name}  model={model_name}  cmAP={cmap:.4f}  weight={w:.2f}")

        def predict(batch: torch.Tensor) -> torch.Tensor:
            x = batch.to(device)
            avg = sum(w * torch.sigmoid(m(x)) for m, w in members)
            return avg.cpu()

    return predict


def generate_pseudo_labels(
    args,
    output_path: str,
    threshold: float = 0.85,
    batch_size: int = 32,
    soft: bool = True,
    device: torch.device = None,
):
    cfg        = Config()
    mel_tf, db = _build_mel(cfg)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}  |  Threshold: {threshold}  |  Soft labels: {soft}")
    predict = _build_predict_fn(args, device)

    all_ogg = sorted(cfg.train_soundscapes_dir.glob("*.ogg"))
    print(f"Found {len(all_ogg)} soundscape files")

    existing_labels = {}
    if cfg.soundscape_labels_csv.exists():
        df_existing = pd.read_csv(cfg.soundscape_labels_csv)
        for _, row in df_existing.iterrows():
            h, m, s = str(row["start"]).split(":")
            start_sec = int(h) * 3600 + int(m) * 60 + int(s)
            existing_labels[(str(row["filename"]), start_sec)] = str(row["primary_label"])
        print(f"Existing human labels: {len(existing_labels)} windows")

    pseudo_rows = []
    t0 = time.time()

    with torch.no_grad():
        for fi, ogg_path in enumerate(all_ogg):
            wav   = _load_wav(ogg_path, cfg)
            total = wav.shape[-1]
            start = 0
            specs, start_secs = [], []

            while start + cfg.clip_samples <= total:
                chunk = wav[..., start : start + cfg.clip_samples]
                specs.append(_wav_to_logmel(chunk, mel_tf, db))
                start_secs.append(start // cfg.sample_rate)
                start += cfg.clip_samples

            if not specs:
                continue

            specs_t = torch.stack(specs)   # (W, 1, mels, frames) — stays on CPU until predict()

            all_probs = []
            for i in range(0, len(specs_t), batch_size):
                all_probs.append(predict(specs_t[i : i + batch_size]))
            probs_np = torch.cat(all_probs).numpy()   # (W, NUM_CLASSES)

            fname = ogg_path.name
            for start_sec, row_probs in zip(start_secs, probs_np):
                key = (fname, start_sec)
                if key in existing_labels:
                    continue

                species = [ALL_CLASSES[i] for i, p in enumerate(row_probs) if p >= threshold]
                if not species:
                    continue

                row_data = {
                    "filename":      fname,
                    "start":         _seconds_to_hms(start_sec),
                    "primary_label": ";".join(species),
                }
                if soft:
                    row_data["soft_probs"] = ",".join(f"{p:.4f}" for p in row_probs)
                pseudo_rows.append(row_data)

            if (fi + 1) % 200 == 0 or (fi + 1) == len(all_ogg):
                elapsed = time.time() - t0
                eta     = elapsed / (fi + 1) * (len(all_ogg) - fi - 1)
                print(f"  {fi+1}/{len(all_ogg)} files | "
                      f"{len(pseudo_rows)} pseudo windows | "
                      f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    print(f"\nPseudo-label generation complete: {len(pseudo_rows)} new windows")

    df_pseudo = pd.DataFrame(pseudo_rows)
    if cfg.soundscape_labels_csv.exists():
        df_human  = pd.read_csv(cfg.soundscape_labels_csv)
        df_merged = pd.concat([df_human, df_pseudo], ignore_index=True)
    else:
        df_merged = df_pseudo

    out = Path(output_path)
    if not out.is_absolute():
        out = cfg.data_dir / out
    df_merged.to_csv(out, index=False)

    n_human  = len(existing_labels)
    n_pseudo = len(pseudo_rows)
    print(f"Merged CSV saved → {out}")
    print(f"  Human labels:  {n_human:,} windows")
    print(f"  Pseudo-labels: {n_pseudo:,} windows  (threshold={threshold})")
    print(f"  Total:         {len(df_merged):,} windows")
    print(f"\nNext step:")
    print(f"  python src/train.py --experiment exp008_soft_pseudo \\")
    print(f"      --pseudo-labels {out.name} --focal-gamma 2.0 --min-rating 3.5")


def main():
    parser = argparse.ArgumentParser()

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--traced",       help="CPU-traced .pt ensemble (e.g. ensemble_baseline_panns.pt). "
                                            "Note: ensemble_v2_no_tta.pt (has AST) requires --device cpu.")
    src.add_argument("--checkpoints",  nargs="+", metavar="CKPT",
                                        help="One or more checkpoint best.pt files to ensemble (CUDA-compatible).")

    parser.add_argument("--weights",   nargs="+", type=float, default=None,
                        help="Per-checkpoint weights (only with --checkpoints; default: equal)")
    parser.add_argument("--output",    default="train_soundscapes_pseudo_soft.csv")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Minimum probability to include a species as pseudo-label (default 0.75)")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--no-soft",   action="store_true", dest="no_soft",
                        help="Store hard binary labels instead of soft probability vectors")
    parser.add_argument("--device",    default=None,
                        help="Force device: 'cpu' or 'cuda' (default: auto)")
    args = parser.parse_args()

    if args.checkpoints and args.weights and len(args.weights) != len(args.checkpoints):
        parser.error("--weights must have the same number of values as --checkpoints")

    device = torch.device(args.device) if args.device else None

    generate_pseudo_labels(
        args,
        output_path=args.output,
        threshold=args.threshold,
        batch_size=args.batch_size,
        soft=not args.no_soft,
        device=device,
    )


if __name__ == "__main__":
    main()
