"""Differentiable attention rollout (torch) for the M6 explanation loss.

Mirrors the NumPy rollout in ``rollout.py`` exactly (head-average, residual,
row-normalize, layer-multiply, CLS->patch, min-max normalize) but in torch so it
is differentiable w.r.t. model parameters. Used only for training the
explanation-preservation regularizer; evaluation still uses the NumPy version.
"""
from __future__ import annotations

_EPS = 1e-8


def forward_with_maps(model, x, grid):
    """Single forward pass returning (logits, rollout_maps).

    rollout_maps is a differentiable (B, gh, gw) tensor built from the attention
    captured during this forward. Enable/disable grad by the caller's context
    (use torch.no_grad() for the frozen reference model).
    """
    attentions = []
    handles = []
    for blk in model.blocks:
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False
        handles.append(blk.attn.attn_drop.register_forward_hook(
            lambda _m, inp, _o: attentions.append(inp[0])))
    try:
        logits = model(x)
    finally:
        for h in handles:
            h.remove()
    return logits, _rollout_cls(attentions, grid)


def _rollout_cls(attentions, grid):
    import torch

    result = None
    for A in attentions:
        A = A.mean(dim=1)                                   # head-average -> (B,N,N)
        N = A.shape[-1]
        A = A + torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A / (A.sum(dim=-1, keepdim=True) + _EPS)
        result = A if result is None else torch.matmul(A, result)

    gh, gw = grid
    n_patches = gh * gw
    mask = result[:, 0, -n_patches:]                        # CLS -> patch tokens
    maps = mask.reshape(-1, gh, gw)
    mn = maps.amin(dim=(1, 2), keepdim=True)
    mx = maps.amax(dim=(1, 2), keepdim=True)
    return (maps - mn) / (mx - mn + _EPS)


def explanation_consistency_loss(current_maps, reference_maps):
    """MSE between current and (detached) reference rollout maps."""
    return ((current_maps - reference_maps) ** 2).mean()
