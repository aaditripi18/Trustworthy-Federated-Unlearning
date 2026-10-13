"""Gradient-ascent unlearning baseline on NIH ChestX-ray14 (GPU).

For every seed and every requested fine-tuning setting this evaluates three
models with exactly the evaluation code of the main experiment:

  ga_only   projected gradient ascent on the forgotten client's data, no repair
  ga        ascent, then plain fine-tuning on the remaining clients
  ga_epu@L  ascent, then fine-tuning with the EPU explanation term (lambda = L)

Original and Oracle are loaded from the checkpoints of the main run, so nothing
that is already in the paper is retrained or changed.

    python scripts/run_ga.py                 # setting A, three seeds
    GA_SETTINGS=A,B python scripts/run_ga.py # both settings
    GA_SEEDS=42 python scripts/run_ga.py     # one seed (e.g. one per session)
    GA_SMOKE=1 python scripts/run_ga.py      # tiny end-to-end check, no checkpoints needed
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _load_cfg() -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "data_nih.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model_nih.yaml"))
    for name in ("federated", "unlearn", "explain", "eval", "ga"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def _param_norm(ckpt_path) -> float:
    """L2 norm of the trainable parameters of a saved model."""
    import torch

    from src.models.vit import build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = build_model(dict(c["model"], pretrained=False), c["data"]["image_size"],
                    c["data"]["num_labels"])
    m.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        return float(torch.sqrt(sum((p ** 2).sum() for p in m.parameters())))


def main() -> int:
    log = get_logger("run_ga")
    import pandas as pd
    import torch  # noqa: F401

    from src.eval.experiment import run_single
    from src.utils.repro import write_reproducibility

    cfg = _load_cfg()
    ga = cfg["ga"]
    smoke = bool(os.environ.get("GA_SMOKE"))
    settings = [s.strip() for s in
                os.environ.get("GA_SETTINGS", ",".join(ga["run_settings"])).split(",")]
    seeds = ga["seeds"]
    if os.environ.get("GA_SEEDS"):                       # e.g. GA_SEEDS=42 to run one seed per session
        seeds = [int(s) for s in os.environ["GA_SEEDS"].split(",")]
    exp = REPO_ROOT / ga["out_dir"]
    if smoke:
        cfg["data"]["nih"]["subset_size"] = 300
        cfg["federated"]["num_rounds"] = 1
        cfg["federated"]["local_epochs"] = 1
        cfg["federated"]["early_stop_patience"] = 99
        cfg["explain"]["num_probe_images"] = 16
        ga["ascent_max_batches"] = 3
        seeds, settings = [42], ["A"]
        exp = REPO_ROOT / (ga["out_dir"] + "_smoke")
        log.info("GA SMOKE MODE: subset 300, 1 round, 1 seed -> %s", exp)

    exp.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(exp / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(fh)

    nih = cfg["data"]["nih"]
    if not (REPO_ROOT / nih["csv"]).exists() or not (REPO_ROOT / nih["images_dir"]).exists():
        log.error("NIH data not found at %s / %s. See docs/DATASET_NIH.md.",
                  nih["csv"], nih["images_dir"])
        return 1
    reuse_root = REPO_ROOT / ga["reuse"]
    if not smoke:
        missing = [str(reuse_root / f"seed_{s}" / f) for s in seeds
                   for f in ("original.pt", "oracle.pt")
                   if not (reuse_root / f"seed_{s}" / f).exists()]
        if missing:
            log.error("checkpoints of the main run are required and missing: %s", missing)
            return 1

    rows = []
    for name in settings:
        st = ga["settings"][name]
        for seed in seeds:
            pfile = exp / "partial" / f"setting_{name}_seed_{seed}.json"
            if pfile.exists():
                log.info("=== setting %s, seed %d === (resume: loading completed partial)",
                         name, seed)
                rows.extend(json.loads(pfile.read_text()))
                continue
            log.info("=== setting %s, seed %d ===", name, seed)
            c = copy.deepcopy(cfg)
            u = c["unlearn"]
            u["finetune_rounds"] = 1 if smoke else st["finetune_rounds"]
            u["finetune_lr"] = st["finetune_lr"]
            u["ascent_radius"] = ga["ascent_radius"]
            u["ascent_max_batches"] = ga["ascent_max_batches"]
            u["ascent_epochs"] = ga["ascent_epochs"]
            u["ascent_grad_clip"] = ga["ascent_grad_clip"]
            reuse_dir = None if smoke else reuse_root / f"seed_{seed}"
            if not smoke:
                # Every model here is initialised from a saved checkpoint, so the ImageNet
                # weights are never used; skipping their download changes no result and lets
                # the script run without internet access.
                c["model"]["pretrained"] = False
            # Step size: the L2 ball is reached after `ascent_steps_to_radius` clipped steps.
            # ||theta_o|| is read from the checkpoint; in smoke mode a nominal value is used.
            norm = 50.0 if smoke else _param_norm(reuse_dir / "original.pt")
            u["ascent_lr"] = ga["ascent_radius"] * norm / ga["ascent_steps_to_radius"]
            lam = st["epu_lambda"]
            extra = [
                {"label": "ga_only", "method": "pga", "rounds": 0, "lambda": 0.0},
                {"label": "ga", "method": "pga", "rounds": u["finetune_rounds"], "lambda": 0.0},
                {"label": f"ga_epu@{lam:g}", "method": "pga", "rounds": u["finetune_rounds"],
                 "lambda": float(lam)},
            ]
            res = run_single(c, seed, [], logger=log,
                             ckpt_dir=exp / f"setting_{name}" / f"seed_{seed}",
                             reuse_ckpt_dir=reuse_dir, extra=extra, standard=False)
            for r in res:
                r["setting"] = name
            pfile.parent.mkdir(parents=True, exist_ok=True)
            tmp = pfile.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(
                res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
            tmp.replace(pfile)
            rows.extend(res)

    df = pd.DataFrame(rows)
    df.to_csv(exp / "ga_results.csv", index=False)
    cols = [c for c in ("macro_auc", "mia_auc", "forget_loss", "pred_dist_to_oracle",
                        "expl_cos_to_original", "expl_iou_to_original", "expl_iou_to_oracle",
                        "wall_time_s", "speedup_vs_oracle") if c in df.columns]
    log.info("summary (mean over seeds):\n%s",
             df.groupby(["setting", "config"])[cols].mean().round(4).to_string())
    write_reproducibility(exp / "reproducibility.json", seeds=seeds, repo_root=REPO_ROOT,
                          extra={"experiment": "gradient-ascent baseline", "settings": settings,
                                 "ga": {k: v for k, v in ga.items() if k != "settings"}})
    print("GA BASELINE OK ->", exp / "ga_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
