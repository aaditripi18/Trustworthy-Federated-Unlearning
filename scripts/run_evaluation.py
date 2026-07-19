"""M7 driver: run the full multi-seed, multi-lambda evaluation and write the
IEEE-style tables. Requires torch + timm; uses no saved checkpoints (each seed
retrains its own chain for reproducibility).

Run from the repo root:  python scripts/run_evaluation.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _load_cfg() -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml")
    for name in ("model", "federated", "unlearn", "explain", "eval"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def main() -> int:
    log = get_logger("run_evaluation")
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch not installed; M7 is a no-op here. SKIPPED.")
        print("M7 EVAL SKIPPED (torch unavailable)")
        return 0

    from src.eval.experiment import run_single, write_outputs

    cfg = _load_cfg()
    seeds = cfg["evaluation"]["seeds"]
    lambdas = cfg["evaluation"]["lambdas"]
    out_dir = REPO_ROOT / get(cfg, "evaluation.out_dir", "experiments")

    results = []
    for seed in seeds:
        log.info("=== seed %d ===", seed)
        results.extend(run_single(cfg, seed, lambdas, logger=log))

    meta = {
        "experiment": "M7 evaluation",
        "seeds": seeds, "lambdas": lambdas,
        "model": cfg["model"]["name"], "num_clients": cfg["data"]["num_clients"],
        "forget_client": cfg["unlearn"]["forget_client"],
        "federated_rounds": cfg["federated"]["num_rounds"],
        "finetune_rounds": cfg["unlearn"].get("finetune_rounds", 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    res = write_outputs(out_dir, results, meta)
    log.info("findings: %s", res["findings"]["statement"])
    log.info("Wrote evaluation_results.csv, multi_seed_summary.csv, evaluation_tables.csv, "
             "ablation_lambda.csv, evaluation_summary.json to %s/", out_dir)

    print("M7 EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
