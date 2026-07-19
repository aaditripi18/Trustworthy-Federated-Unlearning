"""M2 eval: load the saved central checkpoint and report metrics on the test split.

Run from the repo root:  python scripts/eval.py
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


def main() -> int:
    log = get_logger("eval")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; eval skipped. SKIPPED.")
        print("M2 EVAL SKIPPED (torch unavailable)")
        return 0

    from src.data.central import build_central_loaders
    from src.eval.engine import collect_predictions
    from src.eval.metrics import multilabel_metrics
    from src.models.vit import build_model, resolve_device

    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model.yaml"))
    set_seed(get(cfg, "seed", 42))

    ckpt_path = REPO_ROOT / get(cfg, "train.ckpt_dir", "experiments") / "central_model.pt"
    if not ckpt_path.exists():
        log.error("No checkpoint at %s; run scripts/train_central.py first.", ckpt_path)
        return 1

    device = resolve_device(get(cfg, "train.device", "auto"))
    dm = build_central_loaders(cfg)
    model = build_model(cfg["model"], dm["image_size"], dm["n_labels"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])

    y_true, y_score = collect_predictions(model, dm["test"], device)
    metrics = multilabel_metrics(y_true, y_score, get(cfg, "train.threshold", 0.5))
    log.info("test metrics: %s", json.dumps(metrics, indent=2))
    print("M2 EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
