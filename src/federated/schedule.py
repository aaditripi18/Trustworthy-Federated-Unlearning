"""Round-indexed learning-rate schedule for federated training.

FedAvg has one natural global step per round, so the schedule is a pure function
of the (0-indexed) round: a short linear warmup followed by cosine decay to ~0.
This is a scheduling helper only — it does not change the FedAvg algorithm; the
server simply passes the round's LR to the unchanged local Adam step.

Torch-free and deterministic, so it is unit-testable without a GPU.
"""
from __future__ import annotations

import math
from typing import Callable


def make_lr_schedule(base_lr: float, num_rounds: int, warmup_rounds: int = 0,
                     kind: str = "cosine") -> Callable[[int], float]:
    """Return sched(round_idx) -> lr.

    kind="constant" -> base_lr always.
    kind="cosine"   -> linear warmup over `warmup_rounds` (lr scales 1/W..W/W),
                       then cosine decay from base_lr to 0 over the remaining rounds.
    """
    def sched(rnd: int) -> float:
        if kind in (None, "constant"):
            return base_lr
        if warmup_rounds > 0 and rnd < warmup_rounds:
            return base_lr * float(rnd + 1) / float(warmup_rounds)
        if kind == "cosine":
            denom = max(1, num_rounds - warmup_rounds)
            progress = min(1.0, float(rnd - warmup_rounds) / float(denom))
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown lr schedule kind '{kind}'")

    return sched
