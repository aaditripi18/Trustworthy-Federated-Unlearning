"""Gradient-ascent unlearning baseline: registration (torch-free) and behaviour
of the projected ascent + end-to-end runs (torch-gated)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.unlearning.methods import available_methods, get_method  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _cfg():
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    for name in ("model", "federated", "unlearn"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    cfg["data"]["num_clients"] = 3
    cfg["federated"]["num_rounds"] = 1
    cfg["federated"]["local_epochs"] = 1
    cfg["unlearn"]["forget_client"] = 0
    cfg["unlearn"]["finetune_rounds"] = 1
    cfg["unlearn"]["ascent_lr"] = 0.05
    cfg["unlearn"]["ascent_radius"] = 0.02
    return cfg


def test_pga_is_registered():
    assert "pga" in available_methods()
    assert callable(get_method("pga"))


def test_ascent_raises_forget_loss_and_respects_radius():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.data.dataset import build_datamodule
    from src.eval.engine import _as_tensor
    from src.models.vit import build_model
    from src.unlearning.pga import ascent_on_forget_client

    cfg = _cfg()
    dm = build_datamodule(cfg)
    model = build_model(cfg["model"], cfg["data"]["image_size"], cfg["data"]["num_labels"])
    start = [p.detach().clone() for p in model.parameters()]
    loader = dm["clients"][0]
    crit = torch.nn.BCEWithLogitsLoss()

    def forget_loss(m):
        m.eval()
        with torch.no_grad():
            tot = [float(crit(m(_as_tensor(b, "cpu")[0]), _as_tensor(b, "cpu")[1]))
                   for b in loader]
        return sum(tot) / len(tot)

    before = forget_loss(model)
    model, info = ascent_on_forget_client(model, loader, None, "cpu", lr=0.05,
                                          radius_frac=0.02, epochs=3)
    with torch.no_grad():
        moved = float(torch.sqrt(sum(((p - s) ** 2).sum()
                                     for p, s in zip(model.parameters(), start))))
    assert moved <= info["radius"] * (1 + 1e-4)          # projection holds
    assert moved > 0
    assert forget_loss(model) > before                    # ascent, not descent


@pytest.mark.parametrize("lam,rounds", [(0.0, 0), (0.0, 1), (1.0, 1)])
def test_pga_end_to_end_runs(lam, rounds):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.federated.client import get_params
    from src.federated.server import run_federated

    cfg = _cfg()
    cfg["unlearn"]["explanation_lambda"] = lam
    cfg["unlearn"]["finetune_rounds"] = rounds
    base = run_federated(cfg)
    out = get_method("pga")(cfg, 0, get_params(base["model"]))
    assert out["method"] == "pga" and out["kept_clients"] == [1, 2]
    assert out["ascent"]["moved"] <= out["ascent"]["radius"] * (1 + 1e-4)
    assert out["wall_time_s"] >= 0.0


def test_run_single_accepts_extra_baselines():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.eval.experiment import run_single

    cfg = _cfg()
    cfg.update(load_config(REPO_ROOT / "configs" / "explain.yaml"))
    cfg["explain"]["num_probe_images"] = 4
    cfg["federated"]["early_stop_patience"] = 99
    extra = [{"label": "ga_only", "method": "pga", "rounds": 0, "lambda": 0.0},
             {"label": "ga_epu@1", "method": "pga", "rounds": 1, "lambda": 1.0}]
    res = run_single(cfg, 42, [], extra=extra, standard=False)
    assert [r["config"] for r in res] == ["original", "oracle", "ga_only", "ga_epu@1"]
    assert all("expl_cos_to_original" in r and "mia_auc" in r for r in res)


def test_run_single_can_train_original_and_oracle_separately(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.eval.experiment import run_single

    cfg = _cfg()
    cfg.update(load_config(REPO_ROOT / "configs" / "explain.yaml"))
    cfg["explain"]["num_probe_images"] = 4
    cfg["federated"]["early_stop_patience"] = 99
    d = tmp_path / "seed_42"
    a = run_single(cfg, 42, [], ckpt_dir=d, only="original")
    b = run_single(cfg, 42, [], ckpt_dir=d, only="oracle")
    assert [r["config"] for r in a + b] == ["original", "oracle"]
    assert (d / "original.pt").exists() and (d / "oracle.pt").exists()
    res = run_single(cfg, 42, [1.0], reuse_ckpt_dir=d)
    assert {r["config"] for r in res} == {"original", "oracle", "finetune_retain", "epu@1"}
