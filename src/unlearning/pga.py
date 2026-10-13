"""Gradient-ascent client unlearning (second baseline, after Halimi et al., 2022).

Stage 1 (at the forgotten client): starting from the original global model, take
gradient-ASCENT steps on the forgotten client's own data, i.e. maximise the task
loss on D_u, and project the parameters back onto an L2 ball around the starting
point after every step so the model cannot be driven arbitrarily far.

Stage 2 (remaining clients): the same warm-started FedAvg fine-tune used by
``finetune_retain`` / ``epu``, started from the stage-1 model. With
``explanation_lambda == 0`` this is the plain repair phase (config "ga"); with
``explanation_lambda > 0`` the EPU explanation-consistency term is added, anchored
to the frozen ORIGINAL model exactly as in ``epu`` (config "ga+epu").

Differences from Halimi et al. (stated in the paper): their reference point for
the projection is the average of the other clients' last local models, which this
simulation did not store; we project around the original global model instead.
Unlike finetune_retain and epu, this method READS the forgotten client's data.

Registered as method "pga"; toggled purely by config (unlearn.*).
"""
from __future__ import annotations

import time

from .methods import register


def ascent_on_forget_client(model, loader, pos_weight, device, lr, radius_frac,
                            epochs=1, max_batches=None, grad_clip=1.0,
                            logger=None, log_every=20):
    """Projected gradient ascent on the forget set. Mutates and returns ``model``.

    radius_frac : radius of the L2 ball, as a fraction of ||theta_o||_2.
    Returns (model, info) where info records the radius, the distance moved and
    the first / last ascent-batch loss.
    """
    import torch

    from src.eval.engine import _as_tensor

    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    start = [p.detach().clone() for p in params]
    norm0 = torch.sqrt(sum((p ** 2).sum() for p in start))
    radius = float(radius_frac) * float(norm0)
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32,
                                                        device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optim = torch.optim.SGD(params, lr=lr)

    first = last = None
    steps = 0
    done = False
    for _ in range(epochs):
        for batch in loader:
            x, y = _as_tensor(batch, device)
            loss = criterion(model(x), y)
            optim.zero_grad()
            (-loss).backward()                              # ascent on the task loss
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optim.step()
            with torch.no_grad():                           # project onto the L2 ball
                dist = torch.sqrt(sum(((p - s) ** 2).sum() for p, s in zip(params, start)))
                if float(dist) > radius > 0:
                    scale = radius / float(dist)
                    for p, s in zip(params, start):
                        p.copy_(s + (p - s) * scale)
            steps += 1
            last = float(loss.item())
            first = last if first is None else first
            if logger is not None and (steps % log_every == 0 or steps == 1):
                logger.info("ascent | step %d | forget-batch loss %.4f | dist %.4f / %.4f",
                            steps, last, min(float(dist), radius), radius)
            if max_batches is not None and steps >= max_batches:
                done = True
                break
        if done:
            break
    with torch.no_grad():
        moved = float(torch.sqrt(sum(((p - s) ** 2).sum() for p, s in zip(params, start))))
    return model, {"radius": radius, "moved": moved, "steps": steps,
                   "first_loss": first, "last_loss": last, "param_norm": float(norm0)}


@register("pga")
def pga(cfg: dict, forget_client: int, original_params, logger=None, dm=None,
        pos_weight=None, lr_schedule=None, on_round_end=None) -> dict:
    from src.data.dataset import build_datamodule
    from src.explain.extract import model_grid
    from src.federated.client import get_params, set_params
    from src.federated.server import run_federated
    from src.models.vit import build_model, resolve_device

    from .epu import local_train_epu

    u = cfg["unlearn"]
    num_clients = cfg["data"]["num_clients"]
    keep = [c for c in range(num_clients) if c != forget_client]
    rounds = u.get("finetune_rounds", 2)
    lam = float(u.get("explanation_lambda", 0.0))
    device = resolve_device(u.get("device", "auto"))
    image_size, n_labels = cfg["data"]["image_size"], cfg["data"]["num_labels"]
    if dm is None:
        dm = build_datamodule(cfg)

    t0 = time.perf_counter()
    # ---- stage 1: projected gradient ascent at the forgotten client
    model = build_model(cfg["model"], image_size, n_labels).to(device)
    set_params(model, original_params)
    model, info = ascent_on_forget_client(
        model, dm["clients"][forget_client], pos_weight, device,
        lr=u.get("ascent_lr", 1e-3), radius_frac=u.get("ascent_radius", 0.01),
        epochs=u.get("ascent_epochs", 1), max_batches=u.get("ascent_max_batches"),
        grad_clip=u.get("ascent_grad_clip", 1.0), logger=logger)
    ascended = get_params(model)
    if logger:
        logger.info("Unlearn[pga]: ascent done in %d steps, moved %.4f (radius %.4f, "
                    "||theta_o|| %.2f), forget-batch loss %.4f -> %.4f",
                    info["steps"], info["moved"], info["radius"], info["param_norm"],
                    info["first_loss"], info["last_loss"])

    # ---- stage 2: repair on the remaining clients (plain, or with the EPU term)
    local_fn = None
    if lam > 0:
        reference = build_model(cfg["model"], image_size, n_labels).to(device)
        set_params(reference, original_params)             # anchor = frozen ORIGINAL model
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        grid = model_grid(reference, image_size, cfg["model"]["patch_size"])
        ft_lr = u.get("finetune_lr", cfg["federated"]["lr"])

        def local_fn(m, loader, global_params, device, logger=None, log_ctx="", log_every=20):
            return local_train_epu(
                m, reference, loader, global_params, grid,
                epochs=cfg["federated"]["local_epochs"], lr=ft_lr,
                weight_decay=cfg["federated"]["weight_decay"], lam=lam, device=device,
                logger=logger, log_ctx=log_ctx, log_every=log_every, pos_weight=pos_weight)

    if logger:
        logger.info("Unlearn[pga]: %d repair rounds on clients %s, lambda=%.3f",
                    rounds, keep, lam)
    out = run_federated(cfg, logger=logger, include_clients=keep, init_params=ascended,
                        num_rounds=rounds, local_fn=local_fn, dm=dm, pos_weight=pos_weight,
                        lr_schedule=lr_schedule, on_round_end=on_round_end)
    out["wall_time_s"] = time.perf_counter() - t0
    out["method"] = "pga"
    out["kept_clients"] = keep
    out["explanation_lambda"] = lam
    out["ascent"] = info
    return out
