"""M0 smoke test.

Validates the scaffold end-to-end: config loads, seeding is deterministic,
logging works, and a trivial forward pass runs. Uses PyTorch when installed
(the real project backend); otherwise falls back to a NumPy linear op so the
scaffold is runnable before torch is available. Both paths are deterministic.

Run from the repo root:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when run directly (python scripts/smoke_test.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _backend() -> str:
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


def deterministic_forward(cfg: dict) -> tuple[tuple[int, int], tuple[int, int], float]:
    """A trivial linear forward pass on a random input. Assumes set_seed was
    already called. Returns (input_shape, output_shape, output_sum)."""
    b = get(cfg, "smoke.batch_size", 4)
    din = get(cfg, "smoke.in_features", 8)
    dout = get(cfg, "smoke.out_features", 3)

    if _backend() == "torch":
        import torch

        x = torch.randn(b, din)
        layer = torch.nn.Linear(din, dout)
        y = layer(x)
        return tuple(x.shape), tuple(y.shape), float(y.sum().item())

    import numpy as np

    x = np.random.randn(b, din)
    w = np.random.randn(din, dout)
    y = x @ w
    return x.shape, y.shape, float(y.sum())


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    log = get_logger("smoke", get(cfg, "log_level", "INFO"))
    seed = get(cfg, "seed", 42)
    backend = _backend()
    log.info("Project=%s | backend=%s | seed=%s", get(cfg, "project"), backend, seed)

    # Run the forward pass twice from the same seed; results must match.
    set_seed(seed)
    in_shape, out_shape, sum_a = deterministic_forward(cfg)
    set_seed(seed)
    _, _, sum_b = deterministic_forward(cfg)

    log.info("Forward pass: input=%s -> output=%s", in_shape, out_shape)
    deterministic = abs(sum_a - sum_b) < 1e-9
    log.info("Determinism check: run1=%.6f run2=%.6f -> %s",
             sum_a, sum_b, "MATCH" if deterministic else "MISMATCH")

    if not deterministic:
        log.error("Non-deterministic output; seeding is broken.")
        print("M0 SMOKE TEST FAILED")
        return 1

    print("M0 SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
