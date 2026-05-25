# BirdCLEF 2026

Multi-label bioacoustic species detection for the [BirdCLEF 2026 Kaggle competition](https://www.kaggle.com/competitions/birdclef-2026). Given 5-second soundscape windows, predict which of 234 species (birds, frogs, insects, reptiles, mammals) are audible.

**Metric**: class-mean Average Precision (cmAP)  
**Scores so far**: EfficientNet-B3 `0.832` → PANNs CNN14 `0.87`

---

## Setup

```bash
conda create -n birdclef python=3.10
conda activate birdclef
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm transformers soundfile librosa pandas scikit-learn
```

Place the competition data in the project root so the paths match `src/config.py`:

```
birdclef-2026/
  train_audio/          # 11 GB — individual species recordings
  train_soundscapes/    # 5.1 GB — ambient soundscape recordings
  test_soundscapes/     # competition test set
  train.csv
  train_soundscapes_labels.csv
  taxonomy.csv
  sample_submission.csv
```

---

## Running Experiments

### Quick smoke test (verifies the full pipeline in ~2 min)

```bash
python src/train.py --debug
```

### Train a model

```bash
# EfficientNet-B3 baseline
python src/train.py --model efficientnet_b3 --experiment exp001_baseline

# PANNs CNN14 (AudioSet pretrained — weights auto-downloaded ~300 MB)
python src/train.py --model panns --experiment exp005_panns

# Audio Spectrogram Transformer (two-phase fine-tuning)
python src/train.py \
    --model ast \
    --experiment exp006_ast \
    --lr 3e-5 \
    --batch 16 \
    --unfreeze-epoch 6

# SED model (attention-based temporal pooling)
python src/train.py --model sed_efficientnet_b3 --experiment exp004_sed
```

All CLI flags and their defaults:

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b3` | Model name (see models below) |
| `--experiment` | `exp001_baseline` | Name of the checkpoint directory |
| `--epochs` | `30` | Total training epochs |
| `--lr` | `1e-3` | Peak learning rate |
| `--batch` | `32` | Batch size |
| `--unfreeze-epoch` | `None` | Epoch to unfreeze all layers (for AST) |
| `--resume` | `None` | Path to `last.pt` to continue training |
| `--panns-weights` | auto | Path to local CNN14 `.pth` file |
| `--debug` | off | Run on 500 samples for a quick sanity check |

Checkpoints are saved to `checkpoints/<experiment>/`:
- `best.pt` — highest validation cmAP
- `last.pt` — latest epoch (for resuming)

### Analyse performance after training

```bash
python src/analyze.py \
    --checkpoint checkpoints/exp001_baseline/best.pt \
    --save-csv \
    --save-plots
# → checkpoints/exp001_baseline/per_class_metrics.csv
# → checkpoints/exp001_baseline/analysis.png
```

### Generate a local submission CSV

```bash
# From test soundscapes (requires competition data)
python src/predict.py --checkpoint checkpoints/exp001_baseline/best.pt

# Verify format against sample_submission.csv using training soundscapes
python src/predict.py --checkpoint checkpoints/exp001_baseline/best.pt --from-train --n-files 5
```

### Export for Kaggle (no-dependency TorchScript)

```bash
python src/export.py --checkpoint checkpoints/exp001_baseline/best.pt
# → checkpoints/exp001_baseline/model_traced.pt
```

### Build an ensemble

```bash
python src/ensemble_export.py \
    --checkpoints \
        checkpoints/exp001_baseline/best.pt \
        checkpoints/exp005_panns/best.pt \
        checkpoints/exp006_ast/best.pt \
    --output checkpoints/ensemble_v2.pt

# Optional weighted average (if one model is clearly stronger)
python src/ensemble_export.py \
    --checkpoints checkpoints/exp001_baseline/best.pt checkpoints/exp005_panns/best.pt \
    --weights 0.4 0.6 \
    --output checkpoints/ensemble_weighted.pt
```

---

## Kaggle Submission

1. Export a traced model (single model or ensemble) with `src/export.py` / `src/ensemble_export.py`
2. Upload the `.pt` file to **Kaggle → Datasets → New Dataset**
3. Open `notebooks/kaggle_inference.ipynb` and set `WEIGHTS_DATASET_SLUG` to your dataset slug
4. Turn **internet OFF** in the notebook settings
5. Click **Save & Run All (Commit)**

The notebook includes TTA (4 augmentation views) and co-occurrence post-processing out of the box. Set `USE_TTA=False` if the runtime estimate exceeds 75 minutes.

---

## Project Structure

```
src/
  config.py             # All paths and hyperparameters (single source of truth)
  audio_utils.py        # OGG loader, mel spectrogram, sliding window inference
  dataset.py            # BirdDataset + SoundscapeDataset + DataLoader builder
  augmentation.py       # SpecAugment, Mixup, background noise
  train.py              # Training loop (AMP, warmup, cosine LR, checkpointing)
  evaluate.py           # cmAP computation on validation split
  predict.py            # Sliding-window inference → submission CSV
  analyze.py            # Per-class AP, taxonomy breakdown, histogram plots
  export.py             # TorchScript export (single model)
  ensemble_export.py    # TorchScript export (weighted ensemble of N models)
  postprocess.py        # Co-occurrence booster (built from soundscape labels)
  tta.py                # Test-time augmentation (4 views)
  models/
    __init__.py         # build_model() factory
    efficientnet_mel.py # EfficientNet-B3/B4 on log-mel (ImageNet pretrained)
    sed_model.py        # EfficientNet + attention-weighted temporal pooling
    panns_cnn14.py      # CNN14 (AudioSet pretrained, auto-downloaded)
    ast_model.py        # Audio Spectrogram Transformer (AudioSet pretrained)
    temporal_rnn.py     # CNN backbone + GRU temporal head

notebooks/
  01_eda.ipynb          # Class distribution, audio samples, label quality
  02_baseline.ipynb     # End-to-end EfficientNet walkthrough
  03_advanced.ipynb     # AST / ensemble experiments
  kaggle_inference.ipynb  # Self-contained Kaggle submission notebook
```

---

## Adding a New Model

1. Create `src/models/your_model.py` with a class that:
   - Accepts `(B, 1, n_mels, frames)` log-mel input
   - Returns `(B, num_classes)` raw logits (no sigmoid — `BCEWithLogitsLoss` is applied in `train.py`)

2. Register it in `src/models/__init__.py`:
```python
from .your_model import YourModel

def build_model(name, num_classes, pretrained=True, **kwargs):
    ...
    elif name == "yourmodel":
        return YourModel(num_classes=num_classes, pretrained=pretrained)
```

3. Train it:
```bash
python src/train.py --model yourmodel --experiment exp007_yourmodel
```

4. Verify it traces cleanly for Kaggle export:
```bash
python src/export.py --checkpoint checkpoints/exp007_yourmodel/best.pt
```

---

## Modifying Hyperparameters

All defaults live in `src/config.py` as a dataclass. Override on the CLI for one-off runs, or edit `Config` directly for permanent changes.

```python
# src/config.py
@dataclass
class Config:
    batch_size: int = 32        # lower to 16 for AST / large models
    learning_rate: float = 1e-3 # use 3e-5 for transformer fine-tuning
    mixup_alpha: float = 0.4    # 0 to disable Mixup
    bird_soundscape_ratio: float = 0.7  # fraction of batches from individual recordings
    val_split: float = 0.2      # fraction of soundscape files held out for validation
```
