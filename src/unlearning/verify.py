"""Forgetting verification metrics.

Quantifies how well a model has forgotten a client by comparing the *unlearned*
model against the *original* and the *oracle* (retrained-without-client) on three
axes:
  * forget-set loss     — should rise from original toward the oracle's level;
  * membership-inference AUC on the forgotten client's data — should drop from
    original toward the oracle's (a good MIA can no longer tell the forget data
    was trained on);
  * prediction distance to the oracle — the unlearned model's outputs should be
    closer to the oracle's than the original's are.

The pure-metric helpers (bce_per_sample, mia_auc, pred_distance,
closeness_improvement) are torch-free and unit-tested; forgetting_report drives
them over models via the (torch-based) prediction collector.
"""
from __future__ import annotations

import numpy as np

from src.eval.metrics import binary_auc

_EPS = 1e-7


def bce_per_sample(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Mean binary cross-entropy per sample (averaged over labels)."""
    p = np.clip(y_score, _EPS, 1 - _EPS)
    loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return loss.mean(axis=1)


def mia_auc(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> float:
    """Loss-threshold membership inference AUC. Members (trained-on) tend to have
    lower loss, so we score membership by -loss. ~0.5 => indistinguishable."""
    y = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    score = -np.concatenate([member_losses, nonmember_losses])
    return binary_auc(y, score)


def pred_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-sample L2 distance between two prediction matrices."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def closeness_improvement(d_original: float, d_unlearned: float) -> float:
    """Fraction by which the unlearned model is closer to the oracle than the
    original is (>0 means closer; 1.0 means identical to oracle)."""
    return float((d_original - d_unlearned) / d_original) if d_original > 0 else 0.0


def _forget_eval_loader(cfg, dm, forget_client):
    """Deterministic (unshuffled) loader over the forgotten client's images."""
    from src.data.dataset import (CXRDataset, build_transform_import,
                                   make_loader)

    image_size = cfg["data"]["image_size"]
    train_m = dm["manifest"]
    train_m = train_m[train_m.split == "train"].set_index("image_id")
    imgs = dm["partition"]["client_to_images"][forget_client]
    # Reuse the datamodule's image loader (synthetic or NIH) so the forget-set
    # is loaded with the SAME source as training — critical for the real dataset.
    ds = CXRDataset(train_m.loc[imgs].reset_index(), image_size,
                    build_transform_import(image_size, train=False),
                    image_loader=dm["image_loader"])
    # eval loader: deterministic transform + shuffle=False, so eval workers /
    # pin_memory change speed only, never batch contents or order.
    d = cfg["data"]
    return make_loader(ds, d["batch_size"], shuffle=False,
                       num_workers=d.get("eval_num_workers", 0),
                       pin_memory=d.get("pin_memory", False))


def forgetting_report(cfg, dm, models: dict, forget_client: int, device: str) -> dict:
    """models: {"original": m, "unlearned": m, "oracle": m}. Returns a metrics dict."""
    from src.eval.engine import collect_predictions

    forget_loader = _forget_eval_loader(cfg, dm, forget_client)
    nonmember_loader = dm["val"]  # deterministic order (shuffle=False)

    forget_loss, mia, probe = {}, {}, {}
    for name, model in models.items():
        yf, pf = collect_predictions(model, forget_loader, device)
        yv, pv = collect_predictions(model, nonmember_loader, device)
        forget_loss[name] = float(bce_per_sample(yf, pf).mean())
        mia[name] = mia_auc(bce_per_sample(yf, pf), bce_per_sample(yv, pv))
        probe[name] = np.concatenate([pf, pv], axis=0)  # same order across models

    d_orig = pred_distance(probe["original"], probe["oracle"])
    d_unl = pred_distance(probe["unlearned"], probe["oracle"])
    return {
        "forget_client": forget_client,
        "forget_set_loss": forget_loss,
        "mia_auc": mia,
        "pred_distance_to_oracle": {"original": d_orig, "unlearned": d_unl},
        "closeness_improvement_vs_oracle": closeness_improvement(d_orig, d_unl),
        "interpretation": (
            "Good unlearning: mia_auc[unlearned] approaches mia_auc[oracle] and "
            "below mia_auc[original]; closeness_improvement_vs_oracle > 0."),
    }
