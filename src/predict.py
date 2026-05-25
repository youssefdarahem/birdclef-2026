"""Inference script: generate Kaggle submission CSV.

Usage:
    python src/predict.py --checkpoint checkpoints/exp001_baseline/best.pt
    python src/predict.py --checkpoint checkpoints/exp001_baseline/best.pt --model efficientnet_b3
    python src/predict.py --ensemble checkpoints/exp001_baseline/best.pt checkpoints/exp002_ast/best.pt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, ALL_CLASSES, NUM_CLASSES
from audio_utils import build_mel_transform, soundscape_windows
from models import build_model


@torch.no_grad()
def predict_soundscape(model, path, cfg, mel_transform, device) -> list[dict]:
    """Return list of {row_id: str, class: prob} dicts for one soundscape file."""
    model.eval()
    windows = soundscape_windows(path, cfg, mel_transform)
    rows = []
    stem = path.stem  # e.g. BC2026_Test_0001_S05_20250227_010002

    for end_sec, log_mel in windows:
        spec = log_mel.unsqueeze(0).to(device)      # (1, 1, n_mels, frames)
        logits = model(spec)                         # (1, num_classes)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        row = {"row_id": f"{stem}_{end_sec}"}
        for cls, prob in zip(ALL_CLASSES, probs):
            row[cls] = prob
        rows.append(row)

    return rows


@torch.no_grad()
def predict_soundscape_ensemble(models, path, cfg, mel_transform, device) -> list[dict]:
    """Average predictions from multiple models."""
    for m in models:
        m.eval()
    windows = soundscape_windows(path, cfg, mel_transform)
    rows = []
    stem = path.stem

    for end_sec, log_mel in windows:
        spec = log_mel.unsqueeze(0).to(device)
        avg_probs = torch.zeros(NUM_CLASSES, device=device)
        for m in models:
            logits = m(spec)
            avg_probs += torch.sigmoid(logits).squeeze(0)
        avg_probs /= len(models)
        row = {"row_id": f"{stem}_{end_sec}"}
        for cls, prob in zip(ALL_CLASSES, avg_probs.cpu().tolist()):
            row[cls] = prob
        rows.append(row)

    return rows


def load_model_from_checkpoint(ckpt_path: str, model_name: str, device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_saved = ckpt.get("cfg")
    if cfg_saved is not None and hasattr(cfg_saved, "model_name"):
        model_name = cfg_saved.model_name
    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path")
    parser.add_argument("--ensemble",   nargs="+",    help="Multiple checkpoint paths for ensembling")
    parser.add_argument("--model",      default="efficientnet_b3")
    parser.add_argument("--output",     default=None, help="Output CSV path (auto-named if omitted)")
    parser.add_argument("--from-train", action="store_true",
                        help="Use train_soundscapes/ instead of test_soundscapes/ "
                             "(local format demo — not a real submission)")
    parser.add_argument("--n-files",    type=int, default=None,
                        help="Limit number of soundscape files processed (useful with --from-train)")
    args = parser.parse_args()

    if args.checkpoint is None and not args.ensemble:
        parser.error("Provide --checkpoint or --ensemble")

    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel    = build_mel_transform(cfg)

    if args.ensemble:
        models = [load_model_from_checkpoint(p, args.model, device) for p in args.ensemble]
        predict_fn = lambda path: predict_soundscape_ensemble(models, path, cfg, mel, device)
        ckpt_label = "ensemble"
    else:
        model = load_model_from_checkpoint(args.checkpoint, args.model, device)
        predict_fn = lambda path: predict_soundscape(model, path, cfg, mel, device)
        ckpt_label = Path(args.checkpoint).parent.name

    if args.from_train:
        sound_dir = cfg.train_soundscapes_dir
        print("[demo mode] Using train_soundscapes/ — output is for format verification only")
    else:
        sound_dir = cfg.test_soundscapes_dir

    test_files = sorted(sound_dir.glob("*.ogg"))
    if args.n_files:
        test_files = test_files[: args.n_files]

    if not test_files:
        if args.from_train:
            print(f"No .ogg files found in {sound_dir}.")
        else:
            print(
                f"No test soundscapes found in {sound_dir}.\n"
                "On Kaggle, test soundscapes are injected automatically at runtime.\n"
                "To verify the submission format locally, run with --from-train --n-files 5"
            )
        return

    print(f"Running inference on {len(test_files)} test files...")
    all_rows = []
    for i, path in enumerate(test_files):
        rows = predict_fn(path)
        all_rows.extend(rows)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_files)} files processed, {len(all_rows)} windows so far")

    df = pd.DataFrame(all_rows, columns=["row_id"] + ALL_CLASSES)

    out_path = args.output or str(cfg.submissions_dir / f"submission_{ckpt_label}.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}  ({len(df)} rows × {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
