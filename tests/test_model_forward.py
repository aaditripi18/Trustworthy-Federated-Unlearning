"""M2 tests: metrics (backend-agnostic) and the ViT forward/overfit (torch-gated).

The metrics tests run everywhere. The model tests require torch + timm and are
skipped automatically when those are absent, so the suite stays green in any
environment while being fully exercised locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import multilabel_metrics  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------- metrics (no torch) ----------------

def test_metrics_perfect_scores():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_score = y_true.astype(float)  # perfect separation
    m = multilabel_metrics(y_true, y_score)
    assert m["macro_auc"] == pytest.approx(1.0)
    assert m["mAP"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_metrics_in_range_and_handles_degenerate_class():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(20, 3))
    y_true[:, 2] = 1  # degenerate class -> AUC NaN, excluded from macro
    y_score = rng.random((20, 3))
    m = multilabel_metrics(y_true, y_score)
    assert 0.0 <= m["macro_auc"] <= 1.0
    assert 0.0 <= m["mAP"] <= 1.0
    assert np.isnan(m["per_class_auc"][2])


# ---------------- model (torch + timm required) ----------------

def _cfg():
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    return cfg


def test_vit_forward_shape():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from src.models.vit import build_model

    cfg = _cfg()
    model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    x = torch.randn(2, cfg["model"]["in_chans"],
                    cfg["data"]["image_size"], cfg["data"]["image_size"])
    out = model(x)
    assert out.shape == (2, cfg["data"]["num_labels"])


def test_overfit_tiny_batch():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from src.models.vit import build_model

    cfg = _cfg()
    torch.manual_seed(0)
    model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    x = torch.randn(4, cfg["model"]["in_chans"],
                    cfg["data"]["image_size"], cfg["data"]["image_size"])
    y = (torch.rand(4, cfg["data"]["num_labels"]) > 0.5).float()
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    init = float(crit(model(x), y).item())
    for _ in range(80):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < 0.5 * init  # the model can memorize a tiny batch
