"""M6 tests: EPU registration + preservation summary (torch-free), and the
differentiable explanation loss / end-to-end EPU run (torch-gated)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.explain.metrics import preservation_summary  # noqa: E402
from src.unlearning.methods import available_methods, get_method  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------- registry + preservation math (no torch) ----------------

def test_epu_is_registered():
    assert "epu" in available_methods()
    assert callable(get_method("epu"))


def test_preservation_summary_detects_improvement():
    plain = {"cosine_similarity": 0.5, "pearson": 0.4, "iou": 0.3, "mse": 0.20}
    epu = {"cosine_similarity": 0.8, "pearson": 0.6, "iou": 0.5, "mse": 0.10}
    g = preservation_summary(plain, epu)
    assert g["cosine_similarity"] == pytest.approx(0.3)
    assert g["mse_reduction"] == pytest.approx(0.10)
    assert g["epu_closer_to_oracle"] is True

    g2 = preservation_summary(epu, plain)  # reversed -> EPU worse
    assert g2["epu_closer_to_oracle"] is False


# ---------------- differentiable explanation loss (torch + timm) ----------------

def test_forward_with_maps_is_differentiable():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from src.explain.extract import model_grid
    from src.explain.torch_rollout import (explanation_consistency_loss,
                                           forward_with_maps)
    from src.models.vit import build_model

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    grid = model_grid(model, cfg["data"]["image_size"], cfg["model"]["patch_size"])
    x = torch.randn(2, cfg["model"]["in_chans"],
                    cfg["data"]["image_size"], cfg["data"]["image_size"])

    logits, maps = forward_with_maps(model, x, grid)
    assert logits.shape == (2, cfg["data"]["num_labels"])
    assert maps.shape == (2, grid[0], grid[1])
    ref = maps.detach()
    loss = logits.mean() + explanation_consistency_loss(maps, ref * 0)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


# ---------------- end-to-end EPU (torch + timm) ----------------

def test_epu_end_to_end_runs():
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    from src.federated.client import get_params
    from src.federated.server import run_federated

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    for name in ("model", "federated", "unlearn"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    cfg["data"]["num_clients"] = 3
    cfg["federated"]["num_rounds"] = 1
    cfg["federated"]["local_epochs"] = 1
    cfg["unlearn"]["finetune_rounds"] = 1
    cfg["unlearn"]["forget_client"] = 0
    cfg["unlearn"]["explanation_lambda"] = 1.0

    base = run_federated(cfg)
    out = get_method("epu")(cfg, 0, get_params(base["model"]))
    assert out["method"] == "epu"
    assert "model" in out and out["wall_time_s"] >= 0.0
