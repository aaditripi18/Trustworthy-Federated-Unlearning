"""Reproducibility capture: environment, hardware, package versions, seeds.

Writes reproducibility.json so any reported result can be tied to the exact
software/hardware/seed configuration that produced it.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGES = ["numpy", "pandas", "sklearn", "torch", "torchvision",
             "timm", "PIL", "matplotlib", "yaml"]


def _version(mod: str):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _cuda_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        i = 0
        return {"available": True, "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)}
    except Exception:
        return {"available": False}


def _git_commit(repo_root) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def capture_environment(seeds=None, repo_root=None, extra: dict | None = None) -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": {m: _version(m) for m in _PACKAGES},
        "cuda": _cuda_info(),
        "git_commit": _git_commit(repo_root or Path.cwd()),
    }
    if extra:
        env.update(extra)
    return env


def write_reproducibility(path, seeds=None, repo_root=None, extra: dict | None = None) -> dict:
    env = capture_environment(seeds=seeds, repo_root=repo_root, extra=extra)
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    return env
