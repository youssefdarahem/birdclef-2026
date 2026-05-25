"""Export a trained checkpoint to TorchScript (traced).

The traced model is a single .pt file that loads with torch.jit.load()
and requires NO timm, NO soundfile, NO external dependencies at inference time.
Upload this file to Kaggle as your weights dataset.

Usage:
    python src/export.py --checkpoint checkpoints/exp001_baseline/best.pt
    python src/export.py --checkpoint checkpoints/exp001_baseline/best.pt --output checkpoints/exp001_baseline/model_traced.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, NUM_CLASSES
from models import build_model
from audio_utils import build_mel_transform


def export(ckpt_path: str, output_path: str | None = None):
    device = torch.device("cpu")   # always trace on CPU for Kaggle compatibility

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_saved  = ckpt.get("cfg")
    model_name = getattr(cfg_saved, "model_name", "efficientnet_b3")

    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build a representative dummy input (1, 1, n_mels, frames)
    cfg = cfg_saved if cfg_saved is not None else Config()
    mel = build_mel_transform(cfg)
    n_frames = cfg.clip_samples // cfg.hop_length   # ~497 frames for 5s @ hop=320
    dummy = torch.zeros(1, 1, cfg.n_mels, n_frames)

    # Trace — faster and smaller than script for CNNs
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Verify output shape
    with torch.no_grad():
        out = traced(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"

    if output_path is None:
        output_path = str(Path(ckpt_path).parent / "model_traced.pt")

    traced.save(output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Exported → {output_path}  ({size_mb:.1f} MB)")
    print(f"Model:  {model_name}   cmAP: {ckpt.get('best_cmap', 0):.4f}   epoch: {ckpt.get('epoch')}")
    print(f"Input shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(out.shape)}")
    print()
    print("Next steps:")
    print(f"  1. Upload {output_path} to Kaggle → Datasets → New Dataset")
    print("  2. Use notebooks/kaggle_inference.ipynb — set USE_TRACED=True")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/exp001_baseline/best.pt")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
