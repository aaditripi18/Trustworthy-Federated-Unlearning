"""M3 tests: FedAvg aggregation (backend-agnostic) + federated loop (torch-gated)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federated.strategy import weighted_average  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------- FedAvg aggregation (no torch) ----------------

def test_fedavg_weighted_average_matches_hand_computation():
    # client A params [1,1] (n=1), client B params [3,3] (n=3)
    # weighted avg = (1*1 + 3*3) / 4 = 2.5
    a = [np.array([1.0, 1.0])]
    b = [np.array([3.0, 3.0])]
    out = weighted_average([a, b], [1, 3])
    assert np.allclose(out[0], [2.5, 2.5])


def test_fedavg_equal_weights_is_plain_mean():
    a = [np.array([[0.0, 2.0]]), np.array([10.0])]
    b = [np.array([[4.0, 0.0]]), np.array([20.0])]
    out = weighted_average([a, b], [1, 1])
    assert np.allclose(out[0], [[2.0, 1.0]])
    assert np.allclose(out[1], [15.0])


# ---------------- federated loop (torch + timm) ----------------

def test_federated_runs_and_updates_global():
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    from src.federated.client import get_params
    from src.federated.server import run_federated

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    cfg.update(load_config(REPO_ROOT / "configs" / "federated.yaml"))
    # keep the smoke run tiny and fast
    cfg["data"]["num_clients"] = 2
    cfg["federated"]["num_rounds"] = 2
    cfg["federated"]["local_epochs"] = 1
    # This test verifies the history feature itself, so enable it explicitly
    # (the production config disables it to save memory on the final run).
    cfg["federated"]["store_history"] = True

    out = run_federated(cfg)
    assert len(out["history"]) == 2                     # per-round history persisted
    assert len(out["history"].rounds[0]["clients"]) == 2
    assert isinstance(out["val_metrics"]["macro_auc"], float)
    # each stored round carries a global-params snapshot for FU replay (M4)
    assert len(out["history"].rounds[0]["global_params"]) == len(get_params(out["model"]))


# ---------------- history round-trip + summary (no torch) ----------------

def test_history_roundtrip_and_summary(tmp_path):
    from src.federated.history import FederatedHistory, history_summary

    h = FederatedHistory()
    gp0 = [np.zeros((2, 2)), np.zeros(3)]
    gp1 = [np.ones((2, 2)), np.ones(3)]
    clients0 = [([np.zeros((2, 2)), np.zeros(3)], 5), ([np.zeros((2, 2)), np.zeros(3)], 7)]
    clients1 = [([np.ones((2, 2)), np.ones(3)], 5), ([np.ones((2, 2)), np.ones(3)], 7)]
    h.add_round(0, gp0, clients0)
    h.add_round(1, gp1, clients1)

    p = tmp_path / "h.pkl"
    h.save(p)
    h2 = FederatedHistory.load(p)
    assert len(h2) == 2 and np.allclose(h2.rounds[1]["global_params"][0], gp1[0])

    s = history_summary(h, {"model": "small_vit"})
    assert s["num_rounds"] == 2
    assert s["num_param_tensors"] == 2 and s["total_parameters"] == 4 + 3
    assert s["rounds"][0]["global_param_drift_from_prev"] is None
    assert s["rounds"][1]["global_param_drift_from_prev"] > 0   # zeros -> ones
    assert s["rounds"][0]["client_sample_counts"] == [5, 7]
