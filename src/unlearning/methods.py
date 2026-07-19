"""Federated unlearning methods (pluggable registry).

Each method has the signature:
    method(cfg, forget_client, original_params, logger=None) -> dict
returning at least {"model", "global_params", "wall_time_s", "method"}.

Register new algorithms (FedEraser update-replay, gradient ascent, etc.) with
@register("name"); the driver and evaluation pipeline are method-agnostic.
"""
from __future__ import annotations

import time

_REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown unlearning method '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return list(_REGISTRY)


@register("finetune_retain")
def finetune_retain(cfg: dict, forget_client: int, original_params,
                    logger=None, dm=None) -> dict:
    """Warm-started continued FedAvg on the remaining clients only.

    Starts from the trained global model and continues federated training with the
    forgotten client excluded, for a small number of rounds — cheaper than the
    oracle's from-scratch retrain, and drives the model toward the remaining-data
    distribution. An approximate FU baseline; stronger methods plug in alongside.
    """
    from src.federated.server import run_federated

    num_clients = cfg["data"]["num_clients"]
    keep = [c for c in range(num_clients) if c != forget_client]
    rounds = cfg["unlearn"].get("finetune_rounds", 2)
    if logger:
        logger.info("Unlearn[finetune_retain]: %d warm-started rounds on clients %s",
                    rounds, keep)

    t0 = time.perf_counter()
    out = run_federated(cfg, logger=logger, include_clients=keep,
                        init_params=original_params, num_rounds=rounds, dm=dm)
    out["wall_time_s"] = time.perf_counter() - t0
    out["method"] = "finetune_retain"
    out["kept_clients"] = keep
    return out


# Import method modules so their @register decorators run (keep at bottom to
# avoid a circular import at definition time).
from . import epu  # noqa: E402,F401  (registers "epu")
