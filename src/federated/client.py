"""Federated client: parameter (de)serialization and local training.

torch is imported lazily so this module imports without it; the functions are
only called inside a torch-enabled run.
"""
from __future__ import annotations

import numpy as np


def get_params(model) -> list[np.ndarray]:
    """Model state_dict values as an ordered list of NumPy arrays."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_params(model, params: list[np.ndarray]) -> None:
    """Load a parameter list (same order as get_params) back into the model."""
    import torch

    sd = model.state_dict()
    new_sd = {k: torch.as_tensor(p) for k, p in zip(sd.keys(), params)}
    model.load_state_dict(new_sd)


def local_train(model, loader, global_params: list[np.ndarray],
                epochs: int, lr: float, weight_decay: float, device: str,
                logger=None, log_ctx="", log_every: int = 20):
    """Start from the global params, train locally, return (updated_params, n).

    logger/log_ctx/log_every are logging-ONLY and default to no-ops, so existing
    callers and the training math are unchanged.
    """
    import torch

    from src.eval.engine import train_one_epoch

    set_params(model, global_params)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        train_one_epoch(model, loader, optim, criterion, device,
                        logger=logger, log_ctx=log_ctx, epoch=ep + 1, epochs=epochs,
                        log_every=log_every)
    n = len(loader.dataset)
    return get_params(model), n
