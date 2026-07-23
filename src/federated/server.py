"""Federated server: the FedAvg simulation loop.

Sequentially trains each client from the current global model, aggregates with
FedAvg, evaluates the global model on the held-out val set each round, and
records the per-round history. A Flower adapter could replace this loop later
without changing client.py / strategy.py / history.py.
"""
from __future__ import annotations

from .client import get_params, local_train, set_params
from .history import FederatedHistory
from .strategy import weighted_average


def run_federated(cfg: dict, logger=None, include_clients=None,
                  init_params=None, num_rounds=None, local_fn=None, dm=None,
                  pos_weight=None, lr_schedule=None, on_round_end=None) -> dict:
    """FedAvg simulation.

    include_clients : optional iterable of client ids to include (others dropped);
                      used by the retrain oracle to exclude the forgotten client.
    init_params     : optional warm-start global parameters (else random init);
                      used by warm-started unlearning methods.
    num_rounds      : optional override of federated.num_rounds.
    local_fn        : optional custom local-update callable
                      (model, loader, global_params, device, **log_kwargs)
                      -> (params, n_examples); defaults to standard BCE local
                      training. EPU (M6) supplies an explanation-regularized local
                      step here. The log_kwargs (logger, log_ctx, log_every) are
                      logging-only and may be ignored by a callable.
    dm              : optional prebuilt datamodule. build_datamodule(cfg) is a pure
                      function of (cfg, seed) using only a locally-seeded RNG, so a
                      reused dm is identical to a freshly built one; passing it in
                      just avoids re-decoding/partitioning on every call.
    pos_weight      : optional (L,) array/tensor forwarded to local BCE loss (the
                      class-imbalance fix). None -> standard BCE (unchanged).
    lr_schedule     : optional callable(round_idx)->lr (warmup+cosine). None ->
                      constant fcfg["lr"] (unchanged). The FedAvg algorithm is not
                      altered; only the per-round local LR is set from this.
    on_round_end    : optional callable(round_idx, model, train_loss, lr)->bool.
                      When provided, the server's built-in per-round val logging is
                      replaced by this callback (which owns evaluation/checkpointing/
                      early-stopping); returning True stops training early. When
                      None, the original behavior is unchanged.
    """
    import torch  # noqa: F401  (ensures torch present; fail fast otherwise)

    from src.data.dataset import build_datamodule
    from src.eval.engine import collect_predictions
    from src.eval.metrics import multilabel_metrics
    from src.models.vit import build_model, resolve_device

    fcfg = cfg["federated"]
    device = resolve_device(fcfg.get("device", "auto"))
    if dm is None:
        dm = build_datamodule(cfg)
    clients = dm["clients"]
    if include_clients is not None:
        include = set(include_clients)
        clients = {c: l for c, l in clients.items() if c in include}
    n_labels = dm["n_labels"]
    image_size = cfg["data"]["image_size"]

    model = build_model(cfg["model"], image_size, n_labels).to(device)
    if init_params is not None:
        set_params(model, init_params)
    global_params = get_params(model)
    history = FederatedHistory()
    val_metrics = None

    rounds = num_rounds if num_rounds is not None else fcfg["num_rounds"]
    log_every = fcfg.get("log_every_batches", 20)   # logging cadence only
    for rnd in range(rounds):
        round_lr = lr_schedule(rnd) if lr_schedule is not None else fcfg["lr"]
        round_losses: list = []
        results = []
        for cid, loader in clients.items():
            ctx = f"round {rnd + 1}/{rounds} | client {cid}"
            if logger:
                try:
                    nb = len(loader)
                except TypeError:
                    nb = "?"
                logger.info("%s | local update start (%d epochs, %s batches)",
                            ctx, fcfg["local_epochs"], nb)
            if local_fn is not None:
                # local_fn accepts logging-only kwargs; EPU forwards them. They
                # never affect the update, so a local_fn ignoring them is fine.
                updated, n = local_fn(model, loader, global_params, device,
                                      logger=logger, log_ctx=ctx, log_every=log_every)
            else:
                updated, n = local_train(
                    model, loader, global_params,
                    epochs=fcfg["local_epochs"], lr=round_lr,
                    weight_decay=fcfg["weight_decay"], device=device,
                    logger=logger, log_ctx=ctx, log_every=log_every,
                    pos_weight=pos_weight, losses_out=round_losses)
            results.append((updated, n))

        global_params = weighted_average([p for p, _ in results],
                                         [n for _, n in results])
        if fcfg.get("store_history", True):
            history.add_round(rnd, global_params, results)

        set_params(model, global_params)
        if on_round_end is not None:
            train_loss = (float(sum(round_losses) / len(round_losses))
                          if round_losses else None)
            if on_round_end(rnd, model, train_loss, round_lr):
                break
        else:
            y_true, y_score = collect_predictions(model, dm["val"], device)
            val_metrics = multilabel_metrics(y_true, y_score, fcfg.get("threshold", 0.5))
            if logger:
                logger.info("round %d | val macro_auc=%.3f mAP=%.3f",
                            rnd + 1, val_metrics["macro_auc"], val_metrics["mAP"])

    set_params(model, global_params)
    return {"model": model, "global_params": global_params,
            "history": history, "val_metrics": val_metrics}
