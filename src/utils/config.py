"""Tiny YAML config loader.

Kept intentionally simple: load a YAML file into a plain dict, with a helper
for safe nested access (``get(cfg, "smoke.batch_size", 4)``). No custom config
objects or schema framework in M0 — that would be over-engineering this early.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested key like 'smoke.batch_size', returning default if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
