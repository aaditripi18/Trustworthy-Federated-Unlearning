"""M3 driver: run the FedAvg federated simulation and persist model + history.

Run from the repo root:  python scripts/run_federated.py
Requires torch + timm.
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
    cfg.update(load_config(REPO_ROOT / "configs" / "federated.yaml"))
    return cfg


def main() -> int:
    log = get_logger("run_federated")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; federated run is a no-op here. "
                    "Run `pip install -r requirements.txt` locally. SKIPPED.")
        print("M3 FEDERATED SKIPPED (torch unavailable)")
        return 0

    from src.federated.server import run_federated

    cfg = _load_cfg()
    set_seed(get(cfg, "seed", 42))
    log.info("Federated: %d clients, %d rounds, %d local epoch(s)",
             cfg["data"]["num_clients"], cfg["federated"]["num_rounds"],
             cfg["federated"]["local_epochs"])

    out = run_federated(cfg, logger=log)

    ckpt_dir = REPO_ROOT / get(cfg, "federated.ckpt_dir", "experiments")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["model"].state_dict(), "config": cfg},
               ckpt_dir / "federated_model.pt")
    out["history"].save(ckpt_dir / "federated_history.pkl")
    with open(ckpt_dir / "federated_val_metrics.json", "w") as f:
        json.dump(out["val_metrics"], f, indent=2)

    from datetime import datetime, timezone

    from src.federated.history import history_summary
    meta = {
        "experiment": "M3 federated FedAvg",
        "aggregation": "FedAvg (weighted by client sample count)",
        "model": cfg["model"]["name"],
        "seed": get(cfg, "seed", 42),
        "num_clients": cfg["data"]["num_clients"],
        "local_epochs": cfg["federated"]["local_epochs"],
        "lr": cfg["federated"]["lr"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_val_metrics": out["val_metrics"],
        "source": {"history": "federated_history.pkl", "model": "federated_model.pt"},
    }
    with open(ckpt_dir / "history_summary.json", "w") as f:
        json.dump(history_summary(out["history"], meta), f, indent=2)
    log.info("Saved global model, %d-round history, metrics, and history_summary.json to %s/",
             len(out["history"]), ckpt_dir)

    print("M3 FEDERATED OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
