"""Multi-label classification metrics.

Prefers scikit-learn (the project standard, in requirements.txt); falls back to
a compact NumPy implementation when sklearn is unavailable, so the metrics are
verifiable in any environment — the same optional-backend pattern used for torch.

All functions take:
  y_true  : (N, L) binary array
  y_score : (N, L) probability/score array (post-sigmoid)
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score)
    _HAVE_SK = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    _HAVE_SK = False


def _auc_binary(yt: np.ndarray, ys: np.ndarray) -> float:
    """ROC-AUC for one class via the Mann-Whitney U rank statistic, with
    tie-corrected (averaged) ranks so tied scores yield AUC=0.5 (matches sklearn)."""
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")  # AUC undefined when a class is all-0 or all-1
    scores = np.concatenate([neg, pos])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0   # average of 1-based ranks i+1..j+1
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ap_binary(yt: np.ndarray, ys: np.ndarray) -> float:
    """Average precision (area under PR curve) for one class."""
    if yt.sum() == 0:
        return float("nan")
    order = np.argsort(-ys, kind="mergesort")
    yt = yt[order]
    tp = np.cumsum(yt)
    precision = tp / (np.arange(len(yt)) + 1)
    recall = tp / yt.sum()
    # sum of precision at each recall increase (sklearn AP convention)
    dr = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * dr).sum())


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Public ROC-AUC for a single binary task (used by the MIA metric).
    Returns NaN if only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if _HAVE_SK:
        return float(roc_auc_score(y_true, np.asarray(y_score)))
    return _auc_binary(y_true, np.asarray(y_score))


def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray,
                       threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    L = y_true.shape[1]
    y_pred = (y_score >= threshold).astype(int)

    if _HAVE_SK:
        per_auc = []
        for i in range(L):
            yt = y_true[:, i]
            per_auc.append(float("nan") if len(np.unique(yt)) < 2
                           else float(roc_auc_score(yt, y_score[:, i])))
        macro_ap = float(average_precision_score(y_true, y_score, average="macro"))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    else:  # pragma: no cover
        per_auc = [_auc_binary(y_true[:, i], y_score[:, i]) for i in range(L)]
        aps = [_ap_binary(y_true[:, i], y_score[:, i]) for i in range(L)]
        macro_ap = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan")
        macro_f1 = _f1_macro(y_true, y_pred)
        micro_f1 = _f1_micro(y_true, y_pred)

    valid = [a for a in per_auc if not np.isnan(a)]
    macro_auc = float(np.mean(valid)) if valid else float("nan")
    return {"per_class_auc": per_auc, "macro_auc": macro_auc,
            "mAP": macro_ap, "macro_f1": macro_f1, "micro_f1": micro_f1}


def _f1_from_counts(tp, fp, fn) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _f1_macro(y_true, y_pred) -> float:  # pragma: no cover
    f1s = []
    for i in range(y_true.shape[1]):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        f1s.append(_f1_from_counts(tp, fp, fn))
    return float(np.mean(f1s))


def _f1_micro(y_true, y_pred) -> float:  # pragma: no cover
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return _f1_from_counts(tp, fp, fn)
