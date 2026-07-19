"""M5 driver: extract attention-rollout explanations for the original, unlearned,
and oracle models on a fixed probe set, save them, and report explanation
similarity (original-vs-oracle, unlearned-vs-oracle) as a demonstration.

Run from the repo root:  python scripts/run_explain.py
Requires torch + timm; uses checkpoints from M3/M4.
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
    for name in ("model", "federated", "unlearn", "explain"):
        cfg.update(load_config(REPO_ROOT / "configs" / f"{name}.yaml"))
    return cfg


def main() -> int:
    log = get_logger("run_explain")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; M5 is a no-op here. SKIPPED.")
        print("M5 EXPLAIN SKIPPED (torch unavailable)")
        return 0

    from src.data.dataset import build_datamodule
    from src.explain.extract import (explain_loader, model_grid,
                                     save_explanations)
    from src.explain.metrics import compare_maps
    from src.models.vit import build_model, resolve_device

    cfg = _load_cfg()
    set_seed(get(cfg, "seed", 42))
    device = resolve_device(get(cfg, "unlearn.device", "auto"))
    exp = REPO_ROOT / "experiments"
    out_dir = REPO_ROOT / get(cfg, "explain.out_dir", "experiments/explanations")
    method = get(cfg, "explain.method", "attention_rollout")
    k = get(cfg, "explain.num_probe_images", 16)

    dm = build_datamodule(cfg)
    grid = None

    ckpts = {"original": exp / "federated_model.pt",
             "unlearned": exp / "unlearned_model.pt",
             "oracle": exp / "oracle_model.pt"}
    if not ckpts["original"].exists():
        log.error("Missing %s; run scripts/run_federated.py first.", ckpts["original"])
        return 1

    maps = {}
    for name, path in ckpts.items():
        if not path.exists():
            log.warning("skip %s: %s not found (run M4).", name, path)
            continue
        model = build_model(cfg["model"], cfg["data"]["image_size"], dm["n_labels"]).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        if grid is None:
            grid = model_grid(model, cfg["data"]["image_size"], cfg["model"]["patch_size"])
        m, _ = explain_loader(model, dm["val"], grid, device, method, max_images=k)
        maps[name] = m
        save_explanations(out_dir, name, m,
                          {"method": method, "grid": list(grid), "probe": "val", "num": int(len(m))})
        log.info("explanations for %s: %s maps saved", name, m.shape)

    report = {"method": method, "grid": list(grid), "num_probe_images": int(k),
              "similarity": {}}
    if "oracle" in maps and "original" in maps:
        report["similarity"]["original_vs_oracle"] = compare_maps(
            maps["original"], maps["oracle"], get(cfg, "explain.iou_threshold", 0.5))
    if "oracle" in maps and "unlearned" in maps:
        report["similarity"]["unlearned_vs_oracle"] = compare_maps(
            maps["unlearned"], maps["oracle"], get(cfg, "explain.iou_threshold", 0.5))
    with open(exp / "explanations_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("similarity report: %s", json.dumps(report["similarity"], indent=2))

    print("M5 EXPLAIN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
