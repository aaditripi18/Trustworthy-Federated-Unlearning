"""M0 tests: config loading, deterministic seeding, and the smoke script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, get  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def test_config_loads_expected_keys():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg["seed"] == 42
    assert get(cfg, "smoke.batch_size") == 4
    assert get(cfg, "missing.key", "default") == "default"


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.allclose(a, b)


def test_smoke_script_runs_and_passes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "smoke_test.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "M0 SMOKE TEST PASSED" in result.stdout
