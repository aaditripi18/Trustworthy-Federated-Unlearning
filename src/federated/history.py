"""Per-round federated history.

Stores, for each round, the aggregated global parameters and every client's
locally-updated parameters + sample count. This trajectory is what update-replay
federated unlearning (FedEraser-style, M4) needs to reconstruct an unlearned
model without full retraining.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class FederatedHistory:
    def __init__(self):
        self.rounds: list[dict] = []

    def add_round(self, rnd: int, global_params: list[np.ndarray],
                  client_results: list[tuple[list[np.ndarray], int]]) -> None:
        self.rounds.append({
            "round": rnd,
            "global_params": [a.copy() for a in global_params],
            "clients": [{"params": [a.copy() for a in params], "n": n}
                        for params, n in client_results],
        })

    def __len__(self) -> int:
        return len(self.rounds)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.rounds, f)

    @classmethod
    def load(cls, path: str | Path) -> "FederatedHistory":
        obj = cls()
        with open(path, "rb") as f:
            obj.rounds = pickle.load(f)
        return obj


def history_summary(history: "FederatedHistory", meta: dict) -> dict:
    """Lightweight, human-readable metadata describing a federated run.

    ``meta`` supplies experiment context (aggregation, model, seed, clients,
    local_epochs, lr, timestamp, final metrics). Per-round global-parameter
    drift is computed so one can confirm at a glance that the global model
    actually changed each round.
    """
    rounds_meta = []
    prev_flat = None
    for r in history.rounds:
        flat = np.concatenate([a.ravel() for a in r["global_params"]]).astype(np.float64)
        drift = None if prev_flat is None else float(np.linalg.norm(flat - prev_flat))
        prev_flat = flat
        rounds_meta.append({
            "round": int(r["round"]),
            "num_clients": len(r["clients"]),
            "client_sample_counts": [int(c["n"]) for c in r["clients"]],
            "global_param_drift_from_prev": drift,
        })
    first = history.rounds[0]["global_params"] if history.rounds else []
    return {
        **meta,
        "num_rounds": len(history),
        "num_param_tensors": len(first),
        "total_parameters": int(sum(int(a.size) for a in first)),
        "rounds": rounds_meta,
    }
