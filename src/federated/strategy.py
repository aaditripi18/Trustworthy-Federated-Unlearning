"""FedAvg aggregation.

Torch-agnostic: parameters are represented as an ordered list of NumPy arrays
(one per model tensor), so the aggregation logic is unit-testable without torch
and reused unchanged by the (torch-based) client/server.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def weighted_average(client_params: Sequence[Sequence[np.ndarray]],
                     weights: Sequence[float]) -> list[np.ndarray]:
    """FedAvg: element-wise average of client parameter lists, weighted by the
    number of local examples.

    client_params : list over clients; each is a list of arrays (same shapes/order)
    weights        : per-client weights (e.g., local sample counts)
    """
    if not client_params:
        raise ValueError("no client parameters to aggregate")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    norm = [w / total for w in weights]

    n_layers = len(client_params[0])
    out: list[np.ndarray] = []
    for layer in range(n_layers):
        acc = norm[0] * client_params[0][layer].astype(np.float64)
        for c in range(1, len(client_params)):
            acc = acc + norm[c] * client_params[c][layer].astype(np.float64)
        out.append(acc.astype(client_params[0][layer].dtype))
    return out
