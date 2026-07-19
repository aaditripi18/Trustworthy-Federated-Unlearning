"""Reusable explanation extraction over a model + data loader.

Method-registry design (like the unlearning methods) so M6 can compare
explanations from the original / unlearned / oracle models through one interface,
and additional explanation methods can be added without changing callers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rollout import attention_rollout, capture_attentions, cls_patch_maps

_METHODS = {"attention_rollout"}


def model_grid(model, image_size: int, patch_size: int) -> tuple[int, int]:
    pe = getattr(model, "patch_embed", None)
    gs = getattr(pe, "grid_size", None)
    if gs is not None:
        return (int(gs[0]), int(gs[1]))
    g = image_size // patch_size
    return (g, g)


def explain_batch(model, x, grid: tuple[int, int],
                  method: str = "attention_rollout") -> np.ndarray:
    if method not in _METHODS:
        raise KeyError(f"unknown explanation method '{method}'; available: {sorted(_METHODS)}")
    attentions = capture_attentions(model, x)      # torch
    rollout = attention_rollout(attentions)        # numpy
    return cls_patch_maps(rollout, grid)           # (B, gh, gw)


def explain_loader(model, loader, grid, device, method="attention_rollout",
                   max_images=None):
    """Return (maps (N,gh,gw), labels (N,L)) for up to max_images samples."""
    import torch

    model.to(device)
    maps, labels, count = [], [], 0
    for batch in loader:
        x, y = batch
        x = x.to(device) if isinstance(x, torch.Tensor) else \
            torch.as_tensor(np.asarray(x), dtype=torch.float32).to(device)
        maps.append(explain_batch(model, x, grid, method))
        labels.append(np.asarray(y))
        count += len(x)
        if max_images is not None and count >= max_images:
            break
    maps = np.concatenate(maps)
    labels = np.concatenate(labels)
    if max_images is not None:
        maps, labels = maps[:max_images], labels[:max_images]
    return maps, labels


def save_explanations(out_dir, name: str, maps: np.ndarray, meta: dict) -> Path:
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "maps.npy", maps)
    with open(d / "meta.json", "w") as f:
        json.dump({**meta, "shape": list(maps.shape)}, f, indent=2)
    return d


def load_explanations(out_dir, name: str) -> np.ndarray:
    return np.load(Path(out_dir) / name / "maps.npy")
