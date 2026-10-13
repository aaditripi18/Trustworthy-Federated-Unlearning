"""One additional seed of the setting-A experiment, in stages that fit short GPU sessions.

The procedure is that of scripts/run_final.py (same configs, same code path through
src.eval.experiment.run_single); only the orchestration differs, so that the two
from-scratch trainings can run as separate jobs:

  SEED_STAGE=original   train the Original model, save seed_<s>/original.pt
  SEED_STAGE=oracle     train the Oracle (client 0 excluded), save seed_<s>/oracle.pt
  SEED_STAGE=A          load both, run Plain and EPU (lambda 0.5 and 1) with the
                        setting-A fine-tuning, evaluate everything

    SEED=45 SEED_STAGE=original python scripts/run_seed.py
    SEED_SMOKE=1 SEED=45 SEED_STAGE=original python scripts/run_seed.py   # tiny check

Outputs go to experiments/new_seeds (override with SEED_OUT).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

SETTING_A = {"finetune_rounds": 2, "finetune_lr": 0.00002, "lambdas": [0.5, 1]}


def _load_cfg() -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "data_nih.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model_nih.yaml"))
    for name in ("federated", "unlearn", "explain", "eval"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def write_timing_csv(exp: Path) -> None:
    """final_results.csv with the training time and rounds of the saved models; run_single
    reads it next to the checkpoint folders when it reuses a checkpoint."""
    rows = [json.loads(p.read_text()) for p in sorted(exp.glob("timing_*.json"))]
    with open(exp / "final_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "config", "wall_time_s", "rounds"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    log = get_logger("run_seed")
    from src.eval.experiment import run_single

    seed = int(os.environ["SEED"])
    stage = os.environ["SEED_STAGE"]
    smoke = bool(os.environ.get("SEED_SMOKE"))
    cfg = _load_cfg()
    exp = Path(os.environ.get("SEED_OUT", REPO_ROOT / "experiments" / "new_seeds"))
    if smoke:
        cfg["data"]["nih"]["subset_size"] = 300
        cfg["federated"]["num_rounds"] = 1
        cfg["federated"]["local_epochs"] = 1
        cfg["federated"]["early_stop_patience"] = 99
        cfg["explain"]["num_probe_images"] = 16
        exp = Path(str(exp) + "_smoke")
    exp.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(exp / f"run_{stage}_{seed}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(fh)
    log.info("=== seed %d, stage %s ===", seed, stage)
    base_dir = exp / f"seed_{seed}"

    if stage in ("original", "oracle"):
        row = run_single(cfg, seed, [], logger=log, ckpt_dir=base_dir, only=stage)[0]
        (exp / f"timing_{stage}_{seed}.json").write_text(json.dumps(row))
        print("SEED STAGE OK ->", base_dir / f"{stage}.pt")
        return 0

    if stage != "A":
        log.error("unknown SEED_STAGE %r", stage)
        return 1
    missing = [n for n in ("original.pt", "oracle.pt") if not (base_dir / n).exists()]
    if missing:
        log.error("stage A needs %s in %s", missing, base_dir)
        return 1
    write_timing_csv(exp)
    cfg["unlearn"]["finetune_rounds"] = 1 if smoke else SETTING_A["finetune_rounds"]
    cfg["unlearn"]["finetune_lr"] = SETTING_A["finetune_lr"]
    cfg["model"]["pretrained"] = False      # every model starts from a saved checkpoint
    res = run_single(cfg, seed, SETTING_A["lambdas"], logger=log,
                     ckpt_dir=exp / "setting_A" / f"seed_{seed}", reuse_ckpt_dir=base_dir)
    import pandas as pd
    out = exp / f"setting_A_seed_{seed}.csv"
    pd.DataFrame(res).to_csv(out, index=False)
    print("SEED STAGE OK ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
