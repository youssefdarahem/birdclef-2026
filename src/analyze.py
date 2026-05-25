"""Performance analysis script.

Loads a trained checkpoint, runs it on the validation split, and produces:
  1. Overall cmAP
  2. Per-class AP table (sorted, with taxonomy info)
  3. cmAP breakdown by taxonomic class (Aves, Amphibia, Insecta, etc.)
  4. Per-class AP histogram
  5. Top-10 best and worst species
  6. Precision/Recall at threshold 0.5
  7. Sample spectrogram visualizations with model predictions

Usage:
    python src/analyze.py --checkpoint checkpoints/exp001_baseline/best.pt
    python src/analyze.py --checkpoint checkpoints/exp001_baseline/best.pt --save-plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, ALL_CLASSES, CLASS_TO_IDX, NUM_CLASSES
from dataset import build_datasets
from audio_utils import build_mel_transform, load_clip_as_logmel
from models import build_model

# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_saved = ckpt.get("cfg")
    model_name = getattr(cfg_saved, "model_name", "efficientnet_b3")
    model = build_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt.get("best_cmap", 0.0), ckpt.get("epoch", "?"), cfg_saved


@torch.no_grad()
def collect_predictions(model, val_dataset, device, batch_size=64):
    """Run model over validation dataset, return (logits, labels) tensors."""
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_logits, all_labels = [], []
    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        logits = model(specs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_per_class_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    preds = (probs >= threshold).astype(int)

    results = []
    for i, cls in enumerate(ALL_CLASSES):
        gt = labels_np[:, i]
        if gt.sum() == 0:
            results.append({"class": cls, "ap": np.nan, "precision": np.nan,
                            "recall": np.nan, "n_positive": 0})
            continue
        ap = average_precision_score(gt, probs[:, i])
        prec = precision_score(gt, preds[:, i], zero_division=0)
        rec  = recall_score(gt, preds[:, i], zero_division=0)
        results.append({"class": cls, "ap": ap, "precision": prec,
                        "recall": rec, "n_positive": int(gt.sum())})

    df = pd.DataFrame(results)
    df["class"] = df["class"].astype(str)
    return df


def merge_taxonomy(metrics_df, taxonomy_csv):
    tax = pd.read_csv(taxonomy_csv)[["primary_label", "scientific_name", "common_name", "class_name"]]
    tax["primary_label"] = tax["primary_label"].astype(str)
    return metrics_df.merge(tax, left_on="class", right_on="primary_label", how="left").drop(columns="primary_label")


def print_summary(metrics_df, best_cmap, epoch):
    valid = metrics_df.dropna(subset=["ap"])
    cmap = valid["ap"].mean()

    print("=" * 70)
    print(f"  Best val cmAP (saved): {best_cmap:.4f}  |  Epoch: {epoch}")
    print(f"  Re-computed cmAP:      {cmap:.4f}  (over {len(valid)} classes with positives)")
    print(f"  Classes with positives in val split: {len(valid)} / {NUM_CLASSES}")
    print("=" * 70)

    print("\n── Per taxonomic class ──────────────────────────────────────────────")
    for tax_cls, grp in metrics_df.groupby("class_name"):
        grp_valid = grp.dropna(subset=["ap"])
        if len(grp_valid) == 0:
            continue
        print(f"  {tax_cls:<12}  cmAP={grp_valid['ap'].mean():.4f}  "
              f"({len(grp_valid)} species)")

    print("\n── Top 10 best species ──────────────────────────────────────────────")
    top10 = metrics_df.dropna(subset=["ap"]).nlargest(10, "ap")
    for _, r in top10.iterrows():
        name = r.get("common_name", r["class"]) or r["class"]
        print(f"  {r['ap']:.3f}  {name:<35}  ({r['class']})  n={r['n_positive']}")

    print("\n── Top 10 worst species (with data) ────────────────────────────────")
    bot10 = metrics_df.dropna(subset=["ap"]).nsmallest(10, "ap")
    for _, r in bot10.iterrows():
        name = r.get("common_name", r["class"]) or r["class"]
        print(f"  {r['ap']:.3f}  {name:<35}  ({r['class']})  n={r['n_positive']}")

    print("\n── AP distribution ─────────────────────────────────────────────────")
    aps = valid["ap"].values
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2
        n = ((aps >= lo) & (aps < hi)).sum()
        bar = "█" * n
        print(f"  [{lo:.1f}-{hi:.1f})  {bar:<50} {n}")
    print()


def plot_analysis(metrics_df, save_dir=None):
    valid = metrics_df.dropna(subset=["ap"]).sort_values("ap", ascending=False)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. AP histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(valid["ap"], bins=20, color="steelblue", edgecolor="white")
    ax1.axvline(valid["ap"].mean(), color="red", linestyle="--",
                label=f"cmAP={valid['ap'].mean():.3f}")
    ax1.set_xlabel("Average Precision")
    ax1.set_ylabel("# species")
    ax1.set_title("Per-class AP distribution")
    ax1.legend()

    # 2. AP by taxonomic class (box plot)
    ax2 = fig.add_subplot(gs[0, 1])
    tax_groups = [grp["ap"].dropna().values
                  for _, grp in valid.groupby("class_name") if len(grp) > 1]
    tax_labels  = [name for name, grp in valid.groupby("class_name") if len(grp) > 1]
    ax2.boxplot(tax_groups, labels=tax_labels, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax2.set_ylabel("Average Precision")
    ax2.set_title("AP by taxonomic class")
    ax2.tick_params(axis="x", rotation=30)

    # 3. Sorted AP bar chart (all species)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(valid)), valid["ap"].values, width=1.0, color="steelblue")
    ax3.axhline(valid["ap"].mean(), color="red", linestyle="--", linewidth=1)
    ax3.set_xlabel("Species (sorted by AP)")
    ax3.set_ylabel("AP")
    ax3.set_title("Per-species AP (sorted)")
    ax3.set_xlim(0, len(valid))

    # 4. Top-15 and bottom-15 species
    ax4 = fig.add_subplot(gs[1, :2])
    top15  = valid.head(15)
    bot15  = valid.tail(15)
    combined = pd.concat([top15, bot15])
    names = (combined.get("common_name", combined["class"])
                     .fillna(combined["class"])
                     .values)
    colors = ["#2ecc71"] * 15 + ["#e74c3c"] * 15
    bars = ax4.barh(range(len(combined)), combined["ap"].values, color=colors)
    ax4.set_yticks(range(len(combined)))
    ax4.set_yticklabels([f"{n[:30]}" for n in names], fontsize=7)
    ax4.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax4.set_xlabel("Average Precision")
    ax4.set_title("Top-15 (green) and Bottom-15 (red) species")

    # 5. Precision vs Recall scatter
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(valid["recall"], valid["precision"],
                          c=valid["ap"], cmap="RdYlGn", alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax5, label="AP")
    ax5.set_xlabel("Recall @ 0.5")
    ax5.set_ylabel("Precision @ 0.5")
    ax5.set_title("Precision vs Recall (threshold=0.5)")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / "analysis.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/exp001_baseline/best.pt")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-csv",   action="store_true",
                        help="Save per-class metrics to a CSV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model, best_cmap, epoch, cfg_saved = load_checkpoint(args.checkpoint, device)

    # Use saved config if available, otherwise defaults
    cfg = cfg_saved if cfg_saved is not None else Config()
    cfg.debug = False  # always full val split for analysis

    _, _, val_dataset = build_datasets(cfg, augment=None)
    print(f"Val samples: {len(val_dataset)}")

    print("Running inference on validation set...")
    logits, labels = collect_predictions(model, val_dataset, device)

    metrics_df = compute_per_class_metrics(logits, labels, threshold=args.threshold)
    metrics_df = merge_taxonomy(metrics_df, cfg.taxonomy_csv)

    print_summary(metrics_df, best_cmap, epoch)

    exp_name = Path(args.checkpoint).parent.name
    if args.save_csv:
        out_csv = Path("checkpoints") / exp_name / "per_class_metrics.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"Metrics saved → {out_csv}")

    save_dir = Path("checkpoints") / exp_name if args.save_plots else None
    plot_analysis(metrics_df, save_dir=save_dir)


if __name__ == "__main__":
    main()
