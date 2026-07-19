"""M5 tests: rollout math + explanation metrics (torch-free), and
attention capture / batch explanation (torch-gated)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.explain.metrics import compare_maps, cosine_similarity, iou, mse  # noqa: E402
from src.explain.rollout import attention_rollout, cls_patch_maps  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------- rollout math (no torch) ----------------

def test_attention_rollout_shapes_and_normalization():
    B, H, N = 2, 2, 5           # 1 cls + 4 patches (2x2 grid)
    rng = np.random.default_rng(0)
    atts = [rng.random((B, H, N, N)) for _ in range(3)]
    roll = attention_rollout(atts)
    assert roll.shape == (B, N, N)
    maps = cls_patch_maps(roll, (2, 2))
    assert maps.shape == (B, 2, 2)
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6


# ---------------- explanation metrics (no torch) ----------------

def test_metrics_identical_vs_different():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert iou(a, a, threshold=0.5) == pytest.approx(1.0)
    assert mse(a, a) == pytest.approx(0.0)

    b = 1.0 - a  # opposite pattern
    assert cosine_similarity(a, b) < cosine_similarity(a, a)
    assert iou(a, b, threshold=0.5) < 1.0


def test_compare_maps_batch():
    A = np.random.default_rng(1).random((4, 3, 3))
    same = compare_maps(A, A)
    assert same["cosine_similarity"] == pytest.approx(1.0)
    assert same["iou"] == pytest.approx(1.0)
    assert same["mse"] == pytest.approx(0.0)
    assert same["n"] == 4


# ---------------- attention capture (torch + timm) ----------------

def test_capture_and_explain_batch():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from src.explain.extract import explain_batch, model_grid
    from src.models.vit import build_model

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    grid = model_grid(model, cfg["data"]["image_size"], cfg["model"]["patch_size"])
    x = torch.randn(2, cfg["model"]["in_chans"],
                    cfg["data"]["image_size"], cfg["data"]["image_size"])
    maps = explain_batch(model, x, grid)
    assert maps.shape == (2, grid[0], grid[1])
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6
