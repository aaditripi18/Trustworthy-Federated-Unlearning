"""Explanation-Preserving Unlearning (EPU) — the M6 core contribution.

Warm-started federated fine-tune on the remaining clients (as in finetune_retain)
whose local objective is L = L_task + lambda * L_exp, where L_exp keeps each
client's attention-rollout explanation close to the frozen ORIGINAL model's
explanation on the SAME (remaining-client) inputs. This regularizes explanation
drift directly while forgetting proceeds; the forgotten client's data is never
used, so preservation cannot re-teach it (verified via MIA downstream).

Registered as method "epu"; toggled purely by config (unlearn.method).
"""
from __future__ import annotations

import time

from .methods import register


@register("epu")
def epu(cfg: dict, forget_client: int, original_params, logger=None, dm=None,
        pos_weight=None, lr_schedule=None, on_round_end=None) -> dict:
    from src.explain.extract import model_grid
    from src.federated.client import set_params
    from src.federated.server import run_federated
    from src.models.vit import build_model, resolve_device

    num_clients = cfg["data"]["num_clients"]
    keep = [c for c in range(num_clients) if c != forget_client]
    rounds = cfg["unlearn"].get("finetune_rounds", 2)
    lam = cfg["unlearn"].get("explanation_lambda", 1.0)
    device = resolve_device(cfg["unlearn"].get("device", "auto"))
    image_size, n_labels = cfg["data"]["image_size"], cfg["data"]["num_labels"]

    # frozen reference = the original (pre-unlearning) model
    reference = build_model(cfg["model"], image_size, n_labels).to(device)
    set_params(reference, original_params)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    grid = model_grid(reference, image_size, cfg["model"]["patch_size"])

    # constant finetune LR, IDENTICAL to Plain (finetune_retain); the only
    # difference between Plain and EPU is the explanation-consistency term.
    ft_lr = cfg["unlearn"].get("finetune_lr", cfg["federated"]["lr"])

    def local_fn(model, loader, global_params, device,
                 logger=None, log_ctx="", log_every=20):
        return local_train_epu(
            model, reference, loader, global_params, grid,
            epochs=cfg["federated"]["local_epochs"], lr=ft_lr,
            weight_decay=cfg["federated"]["weight_decay"], lam=lam, device=device,
            logger=logger, log_ctx=log_ctx, log_every=log_every, pos_weight=pos_weight)

    if logger:
        logger.info("Unlearn[epu]: %d rounds on clients %s, lambda=%.3f, ft_lr=%.1e",
                    rounds, keep, lam, ft_lr)
    t0 = time.perf_counter()
    out = run_federated(cfg, logger=logger, include_clients=keep,
                        init_params=original_params, num_rounds=rounds,
                        local_fn=local_fn, dm=dm, on_round_end=on_round_end)
    out["wall_time_s"] = time.perf_counter() - t0
    out["method"] = "epu"
    out["kept_clients"] = keep
    out["explanation_lambda"] = lam
    return out


def local_train_epu(model, reference, loader, global_params, grid,
                    epochs, lr, weight_decay, lam, device,
                    logger=None, log_ctx="", log_every=20, pos_weight=None):
    """EPU local update. logger/log_ctx/log_every are logging-ONLY (no effect on
    the loss, optimizer, data order, or RNG); default no-ops preserve behavior."""
    import torch

    from src.eval.engine import _as_tensor, _num_batches
    from src.explain.torch_rollout import (explanation_consistency_loss,
                                           forward_with_maps)
    from src.federated.client import get_params, set_params

    set_params(model, global_params)
    model.to(device)
    model.train()
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)   # same pos_weight as Plain
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    nb = _num_batches(loader)
    for ep in range(epochs):
        for bi, batch in enumerate(loader, start=1):
            x, y = _as_tensor(batch, device)
            logits, cur_maps = forward_with_maps(model, x, grid)
            with torch.no_grad():
                _, ref_maps = forward_with_maps(reference, x, grid)
            loss = criterion(logits, y) + lam * explanation_consistency_loss(
                cur_maps, ref_maps.detach())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if logger is not None and (bi % log_every == 0 or bi == nb or bi == 1):
                logger.info("%s [epu λ=%.3g] | epoch %d/%d | batch %d/%s | loss %.4f",
                            log_ctx, lam, ep + 1, epochs, bi,
                            "?" if nb is None else nb, float(loss.item()))
    return get_params(model), len(loader.dataset)
