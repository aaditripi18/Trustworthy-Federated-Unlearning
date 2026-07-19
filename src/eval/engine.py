"""Minimal train/predict loop shared by the training and eval scripts.

torch is imported lazily inside the functions so this module can be imported in
environments without torch (the functions simply aren't callable there).
Batches may be torch tensors (torch DataLoader) or NumPy arrays (NumpyLoader);
both are handled.
"""
from __future__ import annotations

import numpy as np


def _as_tensor(batch, device):
    import torch

    x, y = batch
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    return x.float().to(device), y.float().to(device)


def _num_batches(loader):
    try:
        return len(loader)
    except TypeError:
        return None


def train_one_epoch(model, loader, optimizer, criterion, device,
                    logger=None, log_ctx="", epoch=None, epochs=None,
                    log_every=20) -> float:
    """Train one epoch.

    The logger/log_ctx/epoch/epochs/log_every arguments are logging-ONLY: they do
    not alter the optimization, the data order, or any RNG draw. When logger is
    None (every existing caller/test), behavior is byte-for-byte unchanged.
    """
    model.train()
    total, n = 0.0, 0
    nb = _num_batches(loader)
    for bi, batch in enumerate(loader, start=1):
        x, y = _as_tensor(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        batch_loss = float(loss.item())          # computed once, reused for log + avg
        total += batch_loss * len(x)
        n += len(x)
        if logger is not None and (bi % log_every == 0 or bi == nb or bi == 1):
            logger.info("%s | epoch %s/%s | batch %d/%s | loss %.4f",
                        log_ctx, epoch, epochs, bi, "?" if nb is None else nb, batch_loss)
    return total / max(n, 1)


def collect_predictions(model, loader, device):
    """Return (y_true, y_score) NumPy arrays; y_score is post-sigmoid."""
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = _as_tensor(batch, device)
            ps.append(torch.sigmoid(model(x)).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)
