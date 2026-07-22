"""M4 tests: unlearning registry + forgetting-metric helpers (torch-free),
and an end-to-end oracle/unlearn/report run (torch-gated)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.unlearning.methods import available_methods, get_method  # noqa: E402
from src.unlearning.verify import (bce_per_sample, closeness_improvement,  # noqa: E402
                                    mia_auc, pred_distance)
from src.utils.config import load_config  # noqa: E402


# ---------------- registry (no torch) ----------------

def test_registry_has_method_and_rejects_unknown():
    assert "finetune_retain" in available_methods()
    assert callable(get_method("finetune_retain"))
    with pytest.raises(KeyError):
        get_method("does_not_exist")


# ---------------- forgetting metrics (no torch) ----------------

def test_mia_auc_detects_membership_and_is_symmetric_at_chance():
    # members clearly lower loss than non-members -> MIA easy -> AUC ~1
    members = np.array([0.01, 0.02, 0.03])
    nonmembers = np.array([1.0, 1.2, 0.9])
    assert mia_auc(members, nonmembers) > 0.9
    # identical distributions -> ~chance
    same = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(mia_auc(same, same) - 0.5) < 1e-6 or np.isnan(mia_auc(same, same))


def test_bce_per_sample_and_distance_and_closeness():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    good = np.array([[0.99, 0.01], [0.01, 0.99]])
    bad = np.array([[0.01, 0.99], [0.99, 0.01]])
    assert bce_per_sample(y, good).mean() < bce_per_sample(y, bad).mean()

    a = np.array([[0.0, 0.0]]); b = np.array([[3.0, 4.0]])
    assert pred_distance(a, b) == pytest.approx(5.0)
    # unlearned closer to oracle than original -> positive improvement
    assert closeness_improvement(d_original=1.0, d_unlearned=0.25) == pytest.approx(0.75)
    assert closeness_improvement(0.0, 0.0) == 0.0


# ---------------- end-to-end (torch + timm) ----------------

def test_unlearn_end_to_end_report_structure():
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    from src.data.dataset import build_datamodule
    from src.federated.client import get_params
    from src.federated.server import run_federated
    from src.models.vit import resolve_device
    from src.unlearning.methods import get_method
    from src.unlearning.oracle import retrain_oracle
    from src.unlearning.verify import forgetting_report

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    for name in ("model", "federated", "unlearn"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    cfg["data"]["num_clients"] = 3
    cfg["federated"]["num_rounds"] = 1
    cfg["federated"]["local_epochs"] = 1
    cfg["unlearn"]["finetune_rounds"] = 1
    cfg["unlearn"]["forget_client"] = 0

    base = run_federated(cfg)
    dm = build_datamodule(cfg)
    oracle = retrain_oracle(cfg, 0)
    unlearned = get_method("finetune_retain")(cfg, 0, get_params(base["model"]))
    # The models are built inside run_federated on resolve_device(federated.device);
    # evaluate them on that SAME device (was hardcoded "cpu", which mismatches a
    # cuda-resident model on a GPU machine).
    device = resolve_device(cfg["federated"].get("device", "auto"))
    rep = forgetting_report(
        cfg, dm,
        {"original": base["model"], "unlearned": unlearned["model"], "oracle": oracle["model"]},
        0, device)

    for k in ("forget_set_loss", "mia_auc", "pred_distance_to_oracle",
              "closeness_improvement_vs_oracle"):
        assert k in rep
    assert set(rep["forget_set_loss"]) == {"original", "unlearned", "oracle"}
    assert np.isfinite(rep["pred_distance_to_oracle"]["unlearned"])
