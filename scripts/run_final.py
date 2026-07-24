"""M8 final experiment driver (NIH ChestX-ray14).

Runs the complete pipeline on the real dataset and produces publication outputs:
  Original FL -> Oracle -> Plain Unlearning -> EPU -> full M7 evaluation
  -> final tables (CSV) -> final figures (PNG+PDF) -> final JSON + reproducibility.

Reuses M0-M7 modules unchanged (via src.eval.experiment). Loads the NIH configs
(data_nih.yaml, model_nih.yaml) so the synthetic default is untouched.

Run from the repo root (after downloading NIH, see docs/DATASET_NIH.md):
    python scripts/run_final.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, get  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

_FINAL_NAMES = {
    "results": "final_results.csv", "summary_csv": "final_multi_seed_summary.csv",
    "tables": "final_tables.csv", "ablation": "final_ablation.csv",
    "summary_json": "final_summary.json",
}


def _load_cfg() -> dict:
    # NIH-specific data + model; the rest of the pipeline config is shared.
    cfg = load_config(REPO_ROOT / "configs" / "data_nih.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model_nih.yaml"))
    for name in ("federated", "unlearn", "explain", "eval"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def main() -> int:
    log = get_logger("run_final")
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch not installed; M8 final run is a no-op here. SKIPPED.")
        print("M8 FINAL SKIPPED (torch unavailable)")
        return 0

    from src.eval.experiment import run_single, write_outputs
    from src.eval.figures import generate_all_figures
    from src.utils.repro import write_reproducibility

    cfg = _load_cfg()

    # Smoke mode (RUN_FINAL_SMOKE=1): exercise the FULL pipeline (Original, Oracle,
    # Plain, EPU) end-to-end on a tiny subset for 1 round / 1 finetune round / 1 seed
    # / 1 lambda, verifying regime propagation before the multi-day run. Writes to a
    # SEPARATE dir so the real experiments/full_experiment/ stays clean.
    if os.environ.get("RUN_FINAL_SMOKE"):
        cfg["data"]["nih"]["subset_size"] = 300
        cfg["federated"]["num_rounds"] = 1
        cfg["federated"]["local_epochs"] = 1
        cfg["federated"]["early_stop_patience"] = 99      # don't early-stop a 1-round smoke
        cfg["unlearn"]["finetune_rounds"] = 1
        cfg["evaluation"]["seeds"] = [42]
        cfg["evaluation"]["lambdas"] = [0.5]
        cfg["explain"]["num_probe_images"] = 16
        cfg["evaluation"]["out_dir"] = "experiments/full_experiment_smoke"
        log.info("RUN_FINAL SMOKE MODE: subset 300, 1 round, 1 finetune round, 1 seed, "
                 "1 lambda -> experiments/full_experiment_smoke")

    exp = REPO_ROOT / get(cfg, "evaluation.out_dir", "experiments")
    seeds = cfg["evaluation"]["seeds"]
    lambdas = cfg["evaluation"]["lambdas"]

    # Persist logs to disk so a killed process (e.g. OOM) still leaves a record;
    # the stream handler alone only lived in the terminal/tmux pane.
    exp.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(exp / "run.log")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(_fh)

    # sanity: fail early with a clear message if NIH files are missing
    nih = cfg["data"]["nih"]
    if not (REPO_ROOT / nih["csv"]).exists() or not (REPO_ROOT / nih["images_dir"]).exists():
        log.error("NIH data not found at %s / %s. See docs/DATASET_NIH.md.",
                  nih["csv"], nih["images_dir"])
        return 1

    log.info("M8 final run on NIH: %d seeds, lambdas=%s, %d clients, %s partition (alpha=%s)",
             len(seeds), lambdas, cfg["data"]["num_clients"],
             cfg["data"]["partition"]["scheme"], cfg["data"]["partition"].get("alpha"))

    # Per-seed checkpointing: each seed is fully seeded and independent, so its
    # results are written atomically the moment it finishes. A later crash resumes
    # by loading completed seeds and only re-running the unfinished ones. The final
    # aggregated outputs are byte-identical to a single uninterrupted run.
    part_dir = exp / "partial"
    part_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        pfile = part_dir / f"seed_{seed}.json"
        if pfile.exists():
            log.info("=== seed %d === (resume: loading completed partial)", seed)
            results.extend(json.loads(pfile.read_text()))
            continue
        log.info("=== seed %d ===", seed)
        seed_results = run_single(cfg, seed, lambdas, logger=log)
        tmp = pfile.with_suffix(".json.tmp")        # atomic: write tmp then replace
        # default handles numpy scalars (np.float64/int64) that json can't natively.
        tmp.write_text(json.dumps(seed_results,
                                  default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        tmp.replace(pfile)
        log.info("seed %d complete; checkpoint written to %s", seed, pfile)
        results.extend(seed_results)

    meta = {
        "experiment": "M8 final (NIH ChestX-ray14)",
        "dataset": "NIH ChestX-ray14",
        "seeds": seeds, "lambdas": lambdas,
        "model": cfg["model"]["name"], "image_size": cfg["data"]["image_size"],
        "num_clients": cfg["data"]["num_clients"],
        "partition": cfg["data"]["partition"],
        "forget_client": cfg["unlearn"]["forget_client"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    res = write_outputs(exp, results, meta, filenames=_FINAL_NAMES)
    figs = generate_all_figures(res["results_df"], exp / "figures")
    write_reproducibility(exp / "reproducibility.json", seeds=seeds, repo_root=REPO_ROOT,
                          extra={"experiment": meta["experiment"], "dataset": meta["dataset"],
                                 "findings": res["findings"]})

    log.info("findings: %s", res["findings"]["statement"])
    log.info("Wrote final_results.csv, final_tables.csv, final_ablation.csv, final_summary.json, "
             "reproducibility.json and figures %s to %s/", figs, exp)
    print("M8 FINAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
