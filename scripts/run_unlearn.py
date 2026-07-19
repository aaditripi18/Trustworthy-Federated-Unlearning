"""M4 driver: build the retrain oracle, run the unlearning method, and report
forgetting metrics comparing {original, unlearned, oracle}.

Run from the repo root:  python scripts/run_unlearn.py
Requires torch + timm; expects experiments/federated_model.pt from M3.
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
    for name in ("model", "federated", "unlearn"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def main() -> int:
    log = get_logger("run_unlearn")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; M4 is a no-op here. "
                    "Run `pip install -r requirements.txt` locally. SKIPPED.")
        print("M4 UNLEARN SKIPPED (torch unavailable)")
        return 0

    from src.federated.client import get_params
    from src.models.vit import build_model, resolve_device
    from src.unlearning.methods import get_method
    from src.unlearning.oracle import retrain_oracle
    from src.unlearning.verify import forgetting_report
    from src.data.dataset import build_datamodule

    cfg = _load_cfg()
    set_seed(get(cfg, "seed", 42))
    device = resolve_device(get(cfg, "unlearn.device", "auto"))
    forget = cfg["unlearn"]["forget_client"]
    exp = REPO_ROOT / get(cfg, "unlearn.ckpt_dir", "experiments")

    # --- original model (from M3) ---
    ckpt = exp / "federated_model.pt"
    if not ckpt.exists():
        log.error("Missing %s; run scripts/run_federated.py first.", ckpt)
        return 1
    dm = build_datamodule(cfg)
    original = build_model(cfg["model"], cfg["data"]["image_size"], dm["n_labels"]).to(device)
    original.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    original_params = get_params(original)

    # --- oracle (gold standard) ---
    oracle = retrain_oracle(cfg, forget, logger=log)

    # --- unlearning method ---
    method = get_method(cfg["unlearn"]["method"])
    unlearned = method(cfg, forget, original_params, logger=log)

    # --- forgetting verification ---
    report = forgetting_report(
        cfg, dm,
        {"original": original, "unlearned": unlearned["model"], "oracle": oracle["model"]},
        forget, device)
    report["efficiency"] = {
        "oracle_wall_time_s": oracle["wall_time_s"],
        "unlearn_wall_time_s": unlearned["wall_time_s"],
        "speedup_vs_oracle": (oracle["wall_time_s"] / unlearned["wall_time_s"]
                              if unlearned["wall_time_s"] > 0 else None),
        "oracle_rounds": cfg["federated"]["num_rounds"],
        "unlearn_rounds": cfg["unlearn"].get("finetune_rounds", 2),
    }
    report["method"] = unlearned["method"]

    log.info("forget-set loss: %s", report["forget_set_loss"])
    log.info("MIA AUC: %s", report["mia_auc"])
    log.info("closeness improvement vs oracle: %.3f",
             report["closeness_improvement_vs_oracle"])
    log.info("unlearn speedup vs oracle: %.2fx",
             report["efficiency"]["speedup_vs_oracle"] or float("nan"))

    exp.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": unlearned["model"].state_dict(), "config": cfg},
               exp / "unlearned_model.pt")
    torch.save({"state_dict": oracle["model"].state_dict(), "config": cfg},
               exp / "oracle_model.pt")
    with open(exp / "forgetting_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved unlearned_model.pt, oracle_model.pt, forgetting_report.json to %s/", exp)

    print("M4 UNLEARN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
