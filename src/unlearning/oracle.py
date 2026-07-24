"""Retraining oracle: the gold-standard baseline.

Trains a fresh federated model from scratch over all clients EXCEPT the forgotten
one. This is what a perfectly-unlearned model should resemble; every unlearning
method is judged by how closely it matches this oracle.
"""
from __future__ import annotations

import time


def retrain_oracle(cfg: dict, forget_client: int, logger=None, dm=None,
                   pos_weight=None, lr_schedule=None, on_round_end=None) -> dict:
    from src.federated.server import run_federated

    num_clients = cfg["data"]["num_clients"]
    keep = [c for c in range(num_clients) if c != forget_client]
    if logger:
        logger.info("Oracle: retraining from scratch on clients %s (excluding %d)",
                    keep, forget_client)

    t0 = time.perf_counter()
    # SAME from-scratch regime as Original (pos_weight, schedule, best-checkpoint);
    # only difference is the excluded forget client.
    out = run_federated(cfg, logger=logger, include_clients=keep, dm=dm,
                        pos_weight=pos_weight, lr_schedule=lr_schedule,
                        on_round_end=on_round_end)  # fresh init, full rounds
    elapsed = time.perf_counter() - t0

    out["wall_time_s"] = elapsed
    out["kept_clients"] = keep
    out["forget_client"] = forget_client
    return out
