"""M2 driver: train the centralized ViT baseline and report multi-label metrics.

Run from the repo root:  python scripts/train_central.py
Requires torch + timm (installed via requirements.txt from M2 onward).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _load_cfg() -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    return cfg


def main() -> int:
    log = get_logger("train_central")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; M2 training is a no-op here. "
                    "Run `pip install -r requirements.txt` locally. SKIPPED.")
        print("M2 TRAINING SKIPPED (torch unavailable)")
        return 0

    from src.data.central import build_central_loaders
    from src.eval.engine import collect_predictions, train_one_epoch
    from src.eval.metrics import multilabel_metrics
    from src.models.vit import build_model, resolve_device

    cfg = _load_cfg()
    set_seed(get(cfg, "seed", 42))
    device = resolve_device(get(cfg, "train.device", "auto"))
    log.info("Device=%s", device)

    dm = build_central_loaders(cfg)
    model = build_model(cfg["model"], dm["image_size"], dm["n_labels"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(),
                             lr=get(cfg, "train.lr", 1e-3),
                             weight_decay=get(cfg, "train.weight_decay", 1e-4))

    for epoch in range(1, get(cfg, "train.epochs", 3) + 1):
        loss = train_one_epoch(model, dm["train"], optim, criterion, device)
        log.info("epoch %d | train_loss=%.4f", epoch, loss)

    y_true, y_score = collect_predictions(model, dm["val"], device)
    metrics = multilabel_metrics(y_true, y_score, get(cfg, "train.threshold", 0.5))
    log.info("val metrics: macro_auc=%.3f mAP=%.3f macro_f1=%.3f",
             metrics["macro_auc"], metrics["mAP"], metrics["macro_f1"])

    ckpt_dir = REPO_ROOT / get(cfg, "train.ckpt_dir", "experiments")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_dir / "central_model.pt")
    with open(ckpt_dir / "central_val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved checkpoint + metrics to %s/", ckpt_dir)

    print("M2 CENTRAL BASELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
