"""Export an ensemble of checkpoints as a single TorchScript model.

The ensemble simply averages sigmoid probabilities from each member model.
The output is compatible with the kaggle_inference.ipynb notebook — load it
exactly like a single-model traced file.

Usage:
    python src/ensemble_export.py \\
        --checkpoints checkpoints/exp001_baseline/best.pt checkpoints/exp005_panns/best.pt \\
        --output checkpoints/ensemble_baseline_panns.pt

    # Optional weights (default: equal weighting):
    python src/ensemble_export.py \\
        --checkpoints checkpoints/exp001_baseline/best.pt checkpoints/exp005_panns/best.pt \\
        --weights 0.6 0.4 \\
        --output checkpoints/ensemble_weighted.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, NUM_CLASSES
from models import build_model
from audio_utils import build_mel_transform


class EnsembleModel(nn.Module):
    """Weighted average of sigmoid probabilities from multiple models.
    Output: raw logits (sigmoid then logit) — compatible with BCEWithLogitsLoss
    and the Kaggle notebook's torch.sigmoid() call.
    """

    def __init__(self, models: list[nn.Module], weights: list[float]):
        super().__init__()
        self.models  = nn.ModuleList(models)
        total        = sum(weights)
        self.weights = [w / total for w in weights]   # normalize to sum=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_probs = sum(
            w * torch.sigmoid(m(x))
            for m, w in zip(self.models, self.weights)
        )
        # Return logits so the notebook's torch.sigmoid() works correctly
        return torch.logit(avg_probs.clamp(1e-6, 1 - 1e-6))


def load_model_from_checkpoint(ckpt_path: str) -> nn.Module:
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_saved  = ckpt.get("cfg")
    model_name = getattr(cfg_saved, "model_name", "efficientnet_b3")
    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    cmap = ckpt.get("best_cmap", 0.0)
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded {Path(ckpt_path).name}  model={model_name}  cmAP={cmap:.4f}  epoch={epoch}")
    return model, model_name, cmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to best.pt checkpoints to ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Relative weights per checkpoint (default: equal)")
    parser.add_argument("--output", default=None,
                        help="Output path for traced ensemble (default: checkpoints/ensemble.pt)")
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.checkpoints):
        raise ValueError("--weights must have the same number of values as --checkpoints")

    weights = args.weights or [1.0] * len(args.checkpoints)

    print("Loading member models:")
    models, names, cmaps = [], [], []
    for ckpt_path in args.checkpoints:
        m, name, cmap = load_model_from_checkpoint(ckpt_path)
        models.append(m)
        names.append(name)
        cmaps.append(cmap)

    ensemble = EnsembleModel(models, weights)
    ensemble.eval()

    # Dummy input for tracing (use 497 frames — center=False shape)
    cfg = Config()
    n_frames = cfg.clip_samples // cfg.hop_length   # 500; adaptive pooling handles 497 too
    dummy = torch.zeros(1, 1, cfg.n_mels, n_frames)

    print("\nTracing ensemble...")
    with torch.no_grad():
        traced = torch.jit.trace(ensemble, dummy)
        out    = traced(dummy)

    assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"

    if args.output is None:
        args.output = str(Path("checkpoints") / "ensemble.pt")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nEnsemble exported → {args.output}  ({size_mb:.1f} MB)")
    print(f"Members ({len(models)}):")
    for name, cmap, w in zip(names, cmaps, weights):
        norm_w = w / sum(weights)
        print(f"  {name}  cmAP={cmap:.4f}  weight={norm_w:.2f}")
    print(f"\nNext: upload {args.output} to Kaggle as a dataset and update WEIGHTS_DATASET_SLUG.")


if __name__ == "__main__":
    main()
