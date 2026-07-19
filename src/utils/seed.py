"""Deterministic seeding.

One entry point, ``set_seed``, seeds Python, NumPy, and (if installed) PyTorch,
so every milestone gets reproducible runs from a single call. Torch is imported
lazily and optionally — the scaffold must run before torch is installed (M0),
and become fully deterministic once it is (M2+).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all RNGs in use. Safe to call before torch is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional until M2

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Prefer determinism over speed for reproducible research results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
