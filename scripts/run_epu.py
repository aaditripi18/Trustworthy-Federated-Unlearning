"""M6 driver: build Original / plain-Unlearned / EPU-Unlearned / Oracle and
report the four-way comparison across diagnostic performance, forgetting,
explanation similarity, and preservation.

Run from the repo root:  python scripts/run_epu.py
Requires torch + timm; uses experiments/federated_model.pt from M3.
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
    log = get_logger("run_epu")
    try:
        import torch
    except ImportError:
        log.warning("torch not installed; M6 is a no-op here. SKIPPED.")
        print("M6 EPU SKIPPED (torch unavailable)")
        return 0

    from src.data.dataset import build_datamodule
    from src.eval.engine import collect_predictions
    from src.eval.metrics import multilabel_metrics
    from src.explain.extract import explain_loader, model_grid
    from src.explain.metrics import compare_maps, preservation_summary
    from src.federated.client import get_params
    from src.models.vit import build_model, resolve_device
    from src.unlearning.methods import get_method
    from src.unlearning.oracle import retrain_oracle
    from src.unlearning.verify import forgetting_report

    cfg = _load_cfg()
    set_seed(get(cfg, "seed", 42))
    device = resolve_device(get(cfg, "unlearn.device", "auto"))
    forget = cfg["unlearn"]["forget_client"]
    exp = REPO_ROOT / "experiments"
    thr = get(cfg, "unlearn.threshold", 0.5)

    ckpt = exp / "federated_model.pt"
    if not ckpt.exists():
        log.error("Missing %s; run scripts/run_federated.py first.", ckpt)
        return 1

    dm = build_datamodule(cfg)

    def fresh(state=None):
        m = build_model(cfg["model"], cfg["data"]["image_size"], dm["n_labels"]).to(device)
        if state is not None:
            m.load_state_dict(state)
        return m

    original = fresh(torch.load(ckpt, map_location=device)["state_dict"])
    orig_params = get_params(original)

    log.info("Building oracle, plain-unlearned, and EPU-unlearned models...")
    oracle = retrain_oracle(cfg, forget, logger=log)
    plain = get_method("finetune_retain")(cfg, forget, orig_params, logger=log)
    epu = get_method("epu")(cfg, forget, orig_params, logger=log)

    models = {"original": original, "unlearned": plain["model"],
              "epu": epu["model"], "oracle": oracle["model"]}

    # 1) diagnostic performance (val)
    diagnostic = {}
    for name, m in models.items():
        yt, ys = collect_predictions(m, dm["val"], device)
        diagnostic[name] = multilabel_metrics(yt, ys, thr)

    # 2) forgetting (vs oracle) for plain and epu
    rep_plain = forgetting_report(cfg, dm, {"original": original, "unlearned": plain["model"],
                                            "oracle": oracle["model"]}, forget, device)
    rep_epu = forgetting_report(cfg, dm, {"original": original, "unlearned": epu["model"],
                                          "oracle": oracle["model"]}, forget, device)
    forgetting = {
        "mia_auc": {"original": rep_plain["mia_auc"]["original"],
                    "unlearned": rep_plain["mia_auc"]["unlearned"],
                    "epu": rep_epu["mia_auc"]["unlearned"],
                    "oracle": rep_plain["mia_auc"]["oracle"]},
        "forget_set_loss": {"original": rep_plain["forget_set_loss"]["original"],
                            "unlearned": rep_plain["forget_set_loss"]["unlearned"],
                            "epu": rep_epu["forget_set_loss"]["unlearned"],
                            "oracle": rep_plain["forget_set_loss"]["oracle"]},
        "pred_distance_to_oracle": {"unlearned": rep_plain["pred_distance_to_oracle"]["unlearned"],
                                    "epu": rep_epu["pred_distance_to_oracle"]["unlearned"]},
    }

    # 3) explanation similarity + 4) preservation
    grid = model_grid(original, cfg["data"]["image_size"], cfg["model"]["patch_size"])
    k = get(cfg, "explain.num_probe_images", 16)
    method = get(cfg, "explain.method", "attention_rollout")
    maps = {name: explain_loader(m, dm["val"], grid, device, method, max_images=k)[0]
            for name, m in models.items()}
    sim_to_oracle = {
        "unlearned": compare_maps(maps["unlearned"], maps["oracle"], get(cfg, "explain.iou_threshold", 0.5)),
        "epu": compare_maps(maps["epu"], maps["oracle"], get(cfg, "explain.iou_threshold", 0.5)),
    }
    sim_to_original = {
        "unlearned": compare_maps(maps["unlearned"], maps["original"]),
        "epu": compare_maps(maps["epu"], maps["original"]),
    }
    preservation = preservation_summary(sim_to_oracle["unlearned"], sim_to_oracle["epu"])

    report = {
        "forget_client": forget,
        "explanation_lambda": epu["explanation_lambda"],
        "chain": ["original", "unlearned", "epu", "oracle"],
        "diagnostic_performance": diagnostic,
        "forgetting": forgetting,
        "explanation_similarity_to_oracle": sim_to_oracle,
        "explanation_similarity_to_original": sim_to_original,
        "preservation_gain_epu_vs_plain": preservation,
        "efficiency": {"oracle_s": oracle["wall_time_s"], "plain_s": plain["wall_time_s"],
                       "epu_s": epu["wall_time_s"]},
    }

    torch.save({"state_dict": epu["model"].state_dict(), "config": cfg}, exp / "epu_model.pt")
    with open(exp / "m6_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("preservation gain (epu vs plain): %s", json.dumps(preservation))
    log.info("Saved epu_model.pt and m6_report.json to %s/", exp)

    print("M6 EPU OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
