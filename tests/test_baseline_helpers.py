"""Torch-free tests for the Stage-1 baseline helpers: LR schedule + pos_weight."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federated.schedule import make_lr_schedule  # noqa: E402
from src.utils.imbalance import compute_pos_weight  # noqa: E402


def test_lr_warmup_then_cosine_decay():
    base, N, W = 0.0002, 25, 3
    s = make_lr_schedule(base, N, W, "cosine")
    # warmup: linear 1/W .. W/W of base
    assert s(0) == pytest.approx(base * 1 / 3)
    assert s(1) == pytest.approx(base * 2 / 3)
    assert s(2) == pytest.approx(base)                 # peak at end of warmup
    # cosine: monotone non-increasing after warmup, ends near 0
    post = [s(r) for r in range(2, N)]
    assert all(post[i] >= post[i + 1] - 1e-12 for i in range(len(post) - 1))
    assert s(N - 1) < base * 0.05
    assert max(s(r) for r in range(N)) <= base + 1e-12  # never exceeds base


def test_lr_constant_mode():
    s = make_lr_schedule(0.001, 10, 0, "constant")
    assert all(s(r) == 0.001 for r in range(10))


def test_pos_weight_neg_over_pos_clipped():
    # label_0: 1/4 positive -> neg/pos = 3 ; label_1: all-negative -> weight 1 ;
    # label_2: ultra-rare (1/100) -> neg/pos=99 but capped to 20.
    n = 100
    df = pd.DataFrame({
        "label_0": [1] * 25 + [0] * 75,
        "label_1": [0] * 100,
        "label_2": [1] * 1 + [0] * 99,
    })
    pw = compute_pos_weight(df, cap=20.0)
    assert pw.dtype == np.float32 and pw.shape == (3,)
    assert pw[0] == pytest.approx(3.0)      # 75/25
    assert pw[1] == pytest.approx(1.0)      # no positives -> weight 1
    assert pw[2] == pytest.approx(20.0)     # 99 clipped to cap
    assert (pw >= 1.0).all()
