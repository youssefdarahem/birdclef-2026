"""Species co-occurrence post-processing.

Builds a co-occurrence matrix from training soundscape labels, then at
inference time boosts predictions for species that frequently co-occur with
already-detected species.

Intuition: if species A is strongly detected, and in training A always appears
alongside species B, we should raise B's score even if the model is uncertain.

Usage
-----
    from postprocess import CooccurrenceBooster
    booster = CooccurrenceBooster(cfg)
    booster.fit()                           # builds matrix from soundscape labels
    probs_boosted = booster.boost(probs)    # probs: (N, num_classes) numpy array
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import Config, ALL_CLASSES, CLASS_TO_IDX, NUM_CLASSES


class CooccurrenceBooster:

    def __init__(self, cfg: Config, alpha: float = 0.15, threshold: float = 0.4):
        """
        alpha:     how strongly to apply the co-occurrence boost (0 = off, 1 = full)
        threshold: detection threshold for "this species is present"
        """
        self.cfg       = cfg
        self.alpha     = alpha
        self.threshold = threshold
        self.cooccur   = None   # (NUM_CLASSES, NUM_CLASSES) conditional probability matrix

    def fit(self):
        """Build co-occurrence matrix from train_soundscapes_labels.csv."""
        df = pd.read_csv(self.cfg.soundscape_labels_csv)

        counts    = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
        marginals = np.zeros(NUM_CLASSES, dtype=np.float32)

        for _, row in df.iterrows():
            raw    = str(row["primary_label"])
            labels = [s.strip() for s in raw.split(";") if s.strip()]
            idxs   = [CLASS_TO_IDX[l] for l in labels if l in CLASS_TO_IDX]
            for i in idxs:
                marginals[i] += 1
                for j in idxs:
                    counts[i, j] += 1

        # P(j | i) = count(i,j) / count(i)  — how often j appears when i appears
        safe_marginals = np.maximum(marginals, 1)
        self.cooccur   = counts / safe_marginals[:, np.newaxis]
        # Zero out the diagonal (trivial self-co-occurrence)
        np.fill_diagonal(self.cooccur, 0.0)

        n_pairs = (self.cooccur > 0.3).sum()
        print(f"Co-occurrence matrix built: {n_pairs} species pairs with P > 0.3")
        return self

    def boost(self, probs: np.ndarray) -> np.ndarray:
        """Apply co-occurrence boost.

        Args:
            probs: (N, NUM_CLASSES) float32 in [0, 1]

        Returns:
            boosted probs: same shape, clipped to [0, 1]
        """
        assert self.cooccur is not None, "Call .fit() first"
        detected = (probs >= self.threshold).astype(np.float32)   # (N, C)
        # For each window, which species are strongly co-occurring with detected ones?
        boost_signal = detected @ self.cooccur          # (N, C) — weighted co-occurrence
        boosted = probs + self.alpha * boost_signal * (1 - probs)  # only boost, never reduce
        return np.clip(boosted, 0.0, 1.0)

    def boost_tensor(self, probs: "torch.Tensor") -> "torch.Tensor":
        import torch
        np_probs  = probs.cpu().numpy()
        boosted   = self.boost(np_probs)
        return torch.from_numpy(boosted).to(probs.device)
