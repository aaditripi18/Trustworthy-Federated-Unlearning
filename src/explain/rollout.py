"""Attention Rollout for ViT (Abnar & Zuidema, 2020).

Two separable pieces:
  * ``capture_attentions`` (torch) — grabs each block's post-softmax attention
    via a hook on ``attn_drop`` (forcing the non-fused path), which is stable
    across timm versions and needs no architecture change.
  * ``attention_rollout`` / ``cls_patch_maps`` (pure NumPy) — the rollout math,
    so it is unit-testable without torch.

Caveat (documented): raw attention rollout is a *plausibility* explanation, not a
guaranteed-faithful one (cf. Jain & Wallace 2019; Chefer et al. 2021). A
gradient/relevance method can be registered alongside it later; the rollout maps
are sufficient for the M5 pipeline and the M6 before/after comparison.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def capture_attentions(model, x) -> list[np.ndarray]:
    """Return per-block attention matrices (B, heads, N, N) as NumPy arrays."""
    import torch

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no `.blocks`; expected a timm ViT/DeiT.")

    store: list[np.ndarray] = []
    handles = []

    def hook(_module, inputs, _output):
        store.append(inputs[0].detach().cpu().numpy())  # attn_drop input = softmax attn

    for blk in blocks:
        attn = blk.attn
        if hasattr(attn, "fused_attn"):
            attn.fused_attn = False  # force explicit softmax path so attn is materialized
        handles.append(attn.attn_drop.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return store


def attention_rollout(attentions, add_residual: bool = True) -> np.ndarray:
    """Recursively combine per-layer attention (head-averaged, residual-added,
    row-normalized). ``attentions``: list of (B, heads, N, N). Returns (B, N, N)."""
    result = None
    for A in attentions:
        A = np.asarray(A, dtype=np.float64).mean(axis=1)      # average heads -> (B,N,N)
        if add_residual:
            N = A.shape[-1]
            A = A + np.eye(N)[None, :, :]
        A = A / (A.sum(axis=-1, keepdims=True) + _EPS)
        result = A if result is None else np.matmul(A, result)
    return result


def cls_patch_maps(rollout_result: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Extract the CLS->patch attention and reshape to (B, gh, gw), each
    min-max normalized to [0,1]. Works for 1 (ViT) or 2 (DeiT) prefix tokens."""
    gh, gw = grid
    n_patches = gh * gw
    mask = rollout_result[:, 0, -n_patches:]                 # CLS row -> last n_patches tokens
    maps = mask.reshape(-1, gh, gw)
    mn = maps.min(axis=(1, 2), keepdims=True)
    mx = maps.max(axis=(1, 2), keepdims=True)
    return (maps - mn) / (mx - mn + _EPS)
