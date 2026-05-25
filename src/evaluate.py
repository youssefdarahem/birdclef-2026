"""Evaluation utilities: per-class AP and macro cmAP."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from config import ALL_CLASSES, NUM_CLASSES


def compute_cmap(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
) -> dict:
    """Compute class-mean Average Precision (cmAP).

    Args:
        all_logits: (N, num_classes) — raw model logits
        all_labels: (N, num_classes) — binary ground truth

    Returns:
        dict with 'cmap' (float), 'per_class_ap' (list), 'classes_with_data' (int)
    """
    probs  = torch.sigmoid(all_logits).cpu().numpy()
    labels = all_labels.cpu().numpy()

    per_class_ap = []
    for c in range(NUM_CLASSES):
        gt_c = labels[:, c]
        if gt_c.sum() == 0:
            continue  # no positives for this class in the split → skip
        ap = average_precision_score(gt_c, probs[:, c])
        per_class_ap.append(ap)

    cmap = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    return {
        "cmap": cmap,
        "per_class_ap": per_class_ap,
        "classes_with_data": len(per_class_ap),
    }


@torch.no_grad()
def evaluate(model, val_loader, device) -> dict:
    """Run model over val_loader and compute cmAP."""
    model.eval()
    all_logits = []
    all_labels = []

    for specs, labels in val_loader:
        specs  = specs.to(device, non_blocking=True)
        logits = model(specs)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return compute_cmap(all_logits, all_labels)
