"""Per-label positive-class weighting for imbalanced multi-label BCE.

NIH ChestX-ray14 is severely imbalanced (~54% "No Finding"; mean per-label
positive rate ~5%). Unweighted BCE is minimized by predicting all-negative,
which collapses F1 to 0 and depresses AUROC. `pos_weight[i] = neg_i / pos_i`
rebalances the gradient so positives contribute proportionally.

Returned as a NumPy array (torch-free / testable); the trainer converts it to a
tensor on-device and passes it to ``BCEWithLogitsLoss(pos_weight=...)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pos_weight(train_manifest: pd.DataFrame, cap: float = 20.0) -> np.ndarray:
    """Return pos_weight (L,) = clip(neg/pos, 1, cap) over the TRAIN split.

    cap bounds the weight of ultra-rare labels (e.g. Hernia ~0.2%, raw ratio
    ~500) to avoid destabilizing training; classes with no positives get weight 1.
    """
    label_cols = sorted((c for c in train_manifest.columns if c.startswith("label_")),
                        key=lambda c: int(c.split("_")[1]))
    if not label_cols:
        raise ValueError("no label_* columns found in manifest")
    y = train_manifest[label_cols].to_numpy(dtype=np.float64)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(ratio, 1.0, cap).astype(np.float32)
