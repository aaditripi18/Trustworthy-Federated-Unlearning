"""Explanation-quality / similarity metrics (pure NumPy, reused in M6).

Given two explanation heatmaps (same shape), we quantify agreement three ways,
each capturing a different notion relevant to explanation *fidelity/stability*
after unlearning:

  * cosine_similarity / pearson  — shape agreement of the saliency pattern
    (scale-insensitive: does the model 'look' in the same places?);
  * iou                          — overlap of the *salient regions* after
    thresholding (localization agreement — the clinically meaningful part);
  * mse                          — raw magnitude difference (sensitive to scale).

M6 uses these to measure how much an explanation drifts from the oracle's after
unlearning, and whether preservation keeps it close.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + _EPS))


def iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    """IoU of the thresholded salient regions of two [0,1]-normalized maps."""
    ma, mb = a >= threshold, b >= threshold
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 1.0  # both empty at this threshold -> trivially identical
    return float(np.logical_and(ma, mb).sum() / union)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compare_maps(maps_a: np.ndarray, maps_b: np.ndarray,
                 iou_threshold: float = 0.5) -> dict:
    """Per-sample metrics averaged over a batch. Inputs: (N, gh, gw)."""
    maps_a = np.asarray(maps_a)
    maps_b = np.asarray(maps_b)
    assert maps_a.shape == maps_b.shape, "explanation batches must have equal shape"
    cos, prs, ious, mses = [], [], [], []
    for a, b in zip(maps_a, maps_b):
        cos.append(cosine_similarity(a, b))
        prs.append(pearson(a, b))
        ious.append(iou(a, b, iou_threshold))
        mses.append(mse(a, b))
    return {
        "n": int(len(maps_a)),
        "cosine_similarity": float(np.mean(cos)),
        "pearson": float(np.mean(prs)),
        "iou": float(np.mean(ious)),
        "mse": float(np.mean(mses)),
    }


def preservation_summary(sim_plain_to_oracle: dict, sim_epu_to_oracle: dict) -> dict:
    """Quantify how much EPU improves explanation closeness to the oracle over
    plain unlearning. Inputs are two compare_maps() dicts (vs the oracle).
    Positive gains / mse_reduction => EPU explanations are closer to the oracle."""
    gains = {k: float(sim_epu_to_oracle[k] - sim_plain_to_oracle[k])
             for k in ("cosine_similarity", "pearson", "iou")}
    gains["mse_reduction"] = float(sim_plain_to_oracle["mse"] - sim_epu_to_oracle["mse"])
    gains["epu_closer_to_oracle"] = bool(
        sim_epu_to_oracle["cosine_similarity"] > sim_plain_to_oracle["cosine_similarity"])
    return gains
