"""Stage-1 baseline optimization run: BASE FEDERATED LEARNING ONLY.

Trains the DeiT-Small classifier with the approved improved regime (longer
budget, warmup+cosine LR) and class-imbalance handling (pos_weight BCE), for TWO
independent random seeds, saving separate outputs per seed and an aggregate
(per-seed + mean/std + consistency) report at the end.

NO Oracle, NO Plain Unlearning, NO EPU, NO lambda sweep. Seeds are NOT selected
or tuned on — both are run and reported. The FedAvg algorithm and the
explanation methodology are unchanged.

Run from repo root (needs the NIH data + torch/timm):
    python scripts/train_baseline.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

PREV_BASELINE = {"macro_auc": 0.558, "macro_f1": 0.0, "mAP": 0.06}  # previous run (context)
AGG_METRICS = ["macro_auc", "micro_auc", "mAP", "macro_f1", "precision", "recall"]


def _load_cfg() -> tuple[dict, dict]:
    cfg = load_config(REPO_ROOT / "configs" / "data_nih.yaml")
    cfg.update(load_config(REPO_ROOT / "configs" / "model_nih.yaml"))
    cfg.update(load_config(REPO_ROOT / "configs" / "federated.yaml"))
    bl = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    b = bl["baseline"]
    b["_seeds"] = bl.get("seeds", [42, 43])
    # apply approved regime overrides into the federated config (this run only)
    cfg["federated"]["num_rounds"] = b["num_rounds"]
    cfg["federated"]["local_epochs"] = b["local_epochs"]
    cfg["federated"]["lr"] = b["base_lr"]
    cfg["federated"]["weight_decay"] = b["weight_decay"]
    cfg["federated"]["store_history"] = False       # base-FL only; save host RAM
    return cfg, b


def evaluate_full(model, loader, device, pos_weight_t, threshold: float) -> dict:
    """Full validation pass -> loss + macro/micro AUROC, mAP, F1, precision, recall."""
    import numpy as np
    import torch
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    from src.eval.engine import _as_tensor

    model.eval()
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    ys, logits, tot, nb = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = _as_tensor(batch, device)
            lg = model(x)
            tot += float(crit(lg, y).item()); nb += 1
            ys.append(y.cpu().numpy()); logits.append(lg.cpu().numpy())
    y_true = np.concatenate(ys)
    logit = np.concatenate(logits)
    y_score = 1.0 / (1.0 + np.exp(-logit))
    y_pred = (y_score >= threshold).astype(int)

    per_auc = [roc_auc_score(y_true[:, i], y_score[:, i])
               for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    macro_auc = float(np.mean(per_auc)) if per_auc else float("nan")
    micro_auc = (float(roc_auc_score(y_true.ravel(), y_score.ravel()))
                 if len(np.unique(y_true)) > 1 else float("nan"))
    return {
        "val_loss": tot / max(nb, 1),
        "macro_auc": macro_auc, "micro_auc": micro_auc,
        "mAP": float(average_precision_score(y_true, y_score, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _save_plots(train_hist, val_hist, lr_sched, num_rounds, fig_dir, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    vr = [h["round"] + 1 for h in val_hist]
    tr = [h["round"] + 1 for h in train_hist]
    panels = [
        ("train_loss_vs_rounds", "Training loss", tr, [h["train_loss"] for h in train_hist]),
        ("val_loss_vs_rounds", "Validation loss", vr, [h["val_loss"] for h in val_hist]),
        ("val_macro_auroc_vs_rounds", "Validation macro-AUROC", vr, [h["macro_auc"] for h in val_hist]),
        ("val_micro_auroc_vs_rounds", "Validation micro-AUROC", vr, [h["micro_auc"] for h in val_hist]),
        ("val_map_vs_rounds", "Validation mAP", vr, [h["mAP"] for h in val_hist]),
        ("val_f1_vs_rounds", "Validation macro-F1", vr, [h["macro_f1"] for h in val_hist]),
        ("lr_schedule", "Learning rate", list(range(1, num_rounds + 1)),
         [lr_sched(r) for r in range(num_rounds)]),
    ]
    for name, ylabel, xs, ysv in panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ysv, marker="o", ms=3)
        ax.set_xlabel("Federated round"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs rounds"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=150)
        plt.close(fig)
    log.info("saved %d plots to %s", len(panels), fig_dir)


def _assess(best, val_hist, early_stopped):
    """Heuristic convergence / fit assessment for a single seed."""
    macro = best["macro_auc"]
    last = val_hist[-1]
    late_drop = (last["macro_auc"] < macro - 0.01)
    if macro < 0.68:
        fit = "UNDERFIT"
    elif late_drop:
        fit = "POSSIBLE OVERFIT (val AUROC declined after peak)"
    else:
        fit = "WELL-FIT"
    verdict = "PASS" if macro >= 0.75 else ("BORDERLINE" if macro >= 0.68 else "RETRAIN REQUIRED")
    return {"fit": fit, "verdict": verdict, "peak_round": best["round"] + 1,
            "rounds_run": len(val_hist), "early_stopped": early_stopped, "late_drop": late_drop}


def run_one_seed(cfg, b, seed, out, log) -> dict:
    """Train one seed end-to-end into `out`; return a summary dict for aggregation."""
    import random

    import numpy as np
    import torch

    from src.data.dataset import build_datamodule
    from src.federated.schedule import make_lr_schedule
    from src.federated.server import run_federated
    from src.models.vit import resolve_device
    from src.utils.imbalance import compute_pos_weight
    from src.utils.seed import set_seed

    out.mkdir(parents=True, exist_ok=True)
    seed_fh = logging.FileHandler(out / "run.log")
    seed_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(seed_fh)
    try:
        cfg = dict(cfg); cfg["seed"] = seed
        set_seed(seed)
        device = resolve_device(cfg["federated"].get("device", "auto"))
        threshold = b.get("eval_threshold", 0.5)
        num_rounds = cfg["federated"]["num_rounds"]
        patience = b.get("early_stop_patience", 5)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        log.info("=== seed %d | base-FL only | device=%s ===", seed, device)
        log.info("regime: rounds=%d local_epochs=%d base_lr=%.1e warmup=%d schedule=%s wd=%.1e",
                 num_rounds, cfg["federated"]["local_epochs"], b["base_lr"],
                 b["warmup_rounds"], b["lr_schedule"], cfg["federated"]["weight_decay"])

        dm = build_datamodule(cfg)
        train_m = dm["manifest"][dm["manifest"].split == "train"]
        pos_weight_np = (compute_pos_weight(train_m, cap=b.get("pos_weight_cap", 20.0))
                         if b.get("use_pos_weight", True) else None)
        pos_weight_t = None if pos_weight_np is None else torch.as_tensor(pos_weight_np, device=device)
        if pos_weight_np is not None:
            log.info("seed %d pos_weight: %s", seed, np.round(pos_weight_np, 2).tolist())

        lr_sched = make_lr_schedule(b["base_lr"], num_rounds, b["warmup_rounds"], b["lr_schedule"])
        train_hist: list = []
        val_hist: list = []
        state = {"best_auc": -1.0, "best": None, "no_improve": 0,
                 "t_prev": time.perf_counter(), "t0": time.perf_counter(), "stopped_early": False}

        def on_round_end(rnd, model, train_loss, lr):
            now = time.perf_counter()
            dt = now - state["t_prev"]; state["t_prev"] = now
            m = evaluate_full(model, dm["val"], device, pos_weight_t, threshold)
            gpu_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
            improved = m["macro_auc"] > state["best_auc"] + 1e-5
            if improved:
                state["best_auc"] = m["macro_auc"]
                state["best"] = {"round": rnd, **m}
                state["no_improve"] = 0
                torch.save({"round": rnd, "state_dict": model.state_dict(), "metrics": m,
                            "seed": seed, "model": cfg["model"]["name"]}, out / "best_model.pt")
            else:
                state["no_improve"] += 1
            eta = dt * (num_rounds - (rnd + 1))
            train_hist.append({"round": rnd, "train_loss": train_loss, "lr": lr})
            val_hist.append({"round": rnd, "lr": lr, "time_s": dt, "gpu_gb": gpu_gb, **m})
            log.info(
                "seed %d | round %d/%d | ep=%d | train_loss=%s | val_loss=%.4f | macroAUC=%.4f "
                "microAUC=%.4f mAP=%.4f F1=%.4f P=%.4f R=%.4f | lr=%.2e | gpu=%.1fGB | %.0fs | "
                "eta=%.0fs | best=%.4f%s",
                seed, rnd + 1, num_rounds, cfg["federated"]["local_epochs"],
                "n/a" if train_loss is None else f"{train_loss:.4f}", m["val_loss"],
                m["macro_auc"], m["micro_auc"], m["mAP"], m["macro_f1"], m["precision"], m["recall"],
                lr, gpu_gb, dt, eta, state["best_auc"], "  <== NEW BEST (saved)" if improved else "")
            if state["no_improve"] >= patience:
                state["stopped_early"] = True
                log.info("seed %d EARLY STOP: no val macro-AUROC improvement for %d rounds "
                         "(best=%.4f @round %d).", seed, patience, state["best_auc"],
                         state["best"]["round"] + 1)
                return True
            return False

        result = run_federated(cfg, logger=log, dm=dm, pos_weight=pos_weight_np,
                               lr_schedule=lr_sched, on_round_end=on_round_end)
        total_time = time.perf_counter() - state["t0"]
        final_model = result["model"]

        # ---- save per-seed artifacts ----
        torch.save({"state_dict": final_model.state_dict(), "round": val_hist[-1]["round"],
                    "seed": seed, "model": cfg["model"]["name"]}, out / "final_model.pt")
        torch.save({"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()},
                   out / "rng_state.pt")
        (out / "scheduler_state.json").write_text(json.dumps({
            "kind": b["lr_schedule"], "base_lr": b["base_lr"], "warmup_rounds": b["warmup_rounds"],
            "num_rounds": num_rounds, "last_round": val_hist[-1]["round"],
            "last_lr": val_hist[-1]["lr"]}, indent=2))
        (out / "training_history.json").write_text(json.dumps(train_hist, indent=2))
        (out / "validation_history.json").write_text(json.dumps(val_hist, indent=2))

        n_params = int(sum(p.numel() for p in final_model.parameters()))
        meta = {
            "experiment": "Stage-1 baseline (base-FL only)", "dataset": "NIH ChestX-ray14",
            "seed": seed, "model": cfg["model"]["name"], "num_parameters": n_params,
            "image_size": cfg["data"]["image_size"], "num_clients": cfg["data"]["num_clients"],
            "partition": cfg["data"]["partition"], "regime": {
                "num_rounds": num_rounds, "local_epochs": cfg["federated"]["local_epochs"],
                "base_lr": b["base_lr"], "warmup_rounds": b["warmup_rounds"],
                "lr_schedule": b["lr_schedule"], "weight_decay": cfg["federated"]["weight_decay"]},
            "use_pos_weight": b.get("use_pos_weight", True), "pos_weight_cap": b.get("pos_weight_cap"),
            "pos_weight_values": None if pos_weight_np is None else [float(x) for x in pos_weight_np],
            "device": device,
            "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "total_train_time_s": total_time, "rounds_run": len(val_hist),
            "early_stopped": state["stopped_early"],
            "optimizer_note": ("FedAvg re-instantiates a fresh Adam per client per round, so there is "
                               "no persistent global optimizer state to checkpoint; model + LR schedule "
                               "+ RNG fully determine resumption. Optimizer state is not saved by design."),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out / "experiment_metadata.json").write_text(json.dumps(meta, indent=2))
        _save_plots(train_hist, val_hist, lr_sched, num_rounds, out / "figures", log)

        best = state["best"]; a = _assess(best, val_hist, state["stopped_early"])
        (out / "SEED_REPORT.md").write_text("\n".join([
            f"# Baseline seed {seed}",
            f"- best macro-AUROC {best['macro_auc']:.4f} @round {best['round']+1}; verdict {a['verdict']}; "
            f"fit {a['fit']}; early_stopped {state['stopped_early']}; time {total_time/3600:.2f}h",
        ]))
        log.info("seed %d DONE | best macro-AUROC=%.4f (round %d) | verdict=%s | %.2fh",
                 seed, best["macro_auc"], best["round"] + 1, a["verdict"], total_time / 3600)
        return {"seed": seed, "best": best, "final": val_hist[-1], "assess": a,
                "rounds_run": len(val_hist), "early_stopped": state["stopped_early"],
                "time_h": total_time / 3600}
    finally:
        log.removeHandler(seed_fh)
        seed_fh.close()


def _agg(values):
    import numpy as np
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "spread": float(abs(v[0] - v[1])) if len(v) == 2 else 0.0,
            "values": [float(x) for x in v]}


def main() -> int:
    log = get_logger("train_baseline")
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch not installed; baseline run is a no-op here. SKIPPED.")
        print("BASELINE SKIPPED (torch unavailable)")
        return 0

    cfg, b = _load_cfg()

    # Smoke mode (BASELINE_SMOKE=1): exercise the EXACT pipeline end-to-end on a
    # tiny data subset for 1 round / 1 epoch / 1 seed, to verify every component
    # before the long run. Same code paths, just fast. Writes to a separate dir.
    if os.environ.get("BASELINE_SMOKE"):
        cfg["data"]["nih"]["subset_size"] = 300
        cfg["federated"]["num_rounds"] = 1
        cfg["federated"]["local_epochs"] = 1
        b["num_rounds"] = 1
        b["local_epochs"] = 1
        b["_seeds"] = [b["_seeds"][0]]
        b["early_stop_patience"] = 999
        b["out_dir"] = "experiments/baseline_smoke"
        get_logger("train_baseline").info(
            "SMOKE MODE: subset_size=300, 1 round, 1 local epoch, 1 seed (exact regime otherwise).")

    root = REPO_ROOT / b.get("out_dir", "experiments/baseline")
    root.mkdir(parents=True, exist_ok=True)
    top_fh = logging.FileHandler(root / "run.log")
    top_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    log.addHandler(top_fh)

    nih = cfg["data"]["nih"]
    if not (REPO_ROOT / nih["csv"]).exists() or not (REPO_ROOT / nih["images_dir"]).exists():
        log.error("NIH data not found (%s / %s). See docs/DATASET_NIH.md.", nih["csv"], nih["images_dir"])
        return 1

    seeds = b["_seeds"]
    log.info("=== Stage-1 baseline: %d seeds %s | base-FL only ===", len(seeds), seeds)

    results = []
    for seed in seeds:
        sdir = root / f"seed_{seed}"
        done = sdir / "experiment_metadata.json"
        srep = sdir / "SEED_REPORT.md"
        if done.exists() and (sdir / "validation_history.json").exists():
            # robustness only (not a hyperparameter/scientific change): resume across
            # a crash by loading an already-completed seed instead of recomputing it.
            vh = json.loads((sdir / "validation_history.json").read_text())
            best = max(vh, key=lambda h: (h["macro_auc"] if h["macro_auc"] == h["macro_auc"] else -1))
            a = _assess(best, vh, json.loads(done.read_text()).get("early_stopped", False))
            log.info("seed %d already complete; loaded (best macro-AUROC=%.4f).", seed, best["macro_auc"])
            results.append({"seed": seed, "best": best, "final": vh[-1], "assess": a,
                            "rounds_run": len(vh),
                            "early_stopped": json.loads(done.read_text()).get("early_stopped", False),
                            "time_h": json.loads(done.read_text()).get("total_train_time_s", 0) / 3600})
            continue
        results.append(run_one_seed(cfg, b, seed, sdir, log))

    # ---- aggregate across seeds ----
    agg = {mtr: _agg([r["best"][mtr] for r in results]) for mtr in AGG_METRICS}
    prev = PREV_BASELINE["macro_auc"]
    all_improved = all(r["best"]["macro_auc"] > prev + 0.02 for r in results)
    auc_spread = agg["macro_auc"]["spread"]
    consistent = auc_spread <= 0.03
    (root / "aggregate_metrics.json").write_text(json.dumps(
        {"seeds": seeds, "per_seed_best": {r["seed"]: r["best"] for r in results},
         "aggregate": agg, "improvement_consistent": bool(all_improved and consistent)}, indent=2))

    def row(mtr):
        g = agg[mtr]
        return (f"| {mtr} | " + " | ".join(f"{v:.4f}" for v in g["values"]) +
                f" | {g['mean']:.4f} | {g['std']:.4f} | {g['spread']:.4f} |")

    lines = [
        "# Stage-1 Baseline — 2-Seed Report", "",
        f"- Model: {cfg['model']['name']} | rounds: {cfg['federated']['num_rounds']} | "
        f"local_epochs: {cfg['federated']['local_epochs']} | seeds: {seeds}",
        f"- pos_weight: {b.get('use_pos_weight', True)} (cap {b.get('pos_weight_cap')}) | "
        f"warmup {b['warmup_rounds']} + cosine, base_lr {b['base_lr']}", "",
        "## Per-seed best validation metrics",
        "| metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std(n=2) | |Δ| |",
        "|---|" + "---|" * (len(seeds) + 3),
        *[row(m) for m in AGG_METRICS], "",
        "## Per-seed detail",
        *[f"- seed {r['seed']}: best macro-AUROC {r['best']['macro_auc']:.4f} @round "
          f"{r['best']['round']+1}, verdict {r['assess']['verdict']}, fit {r['assess']['fit']}, "
          f"early_stopped {r['early_stopped']}, {r['time_h']:.2f}h" for r in results], "",
        "## Consistency",
        f"- Both seeds improve over previous baseline ({prev:.3f})? **{all_improved}**",
        f"- macro-AUROC spread |Δ| = {auc_spread:.4f} → **{'CONSISTENT' if consistent else 'INCONSISTENT'}** "
        f"(threshold 0.03).",
        "",
    ]
    if not consistent:
        worse = min(results, key=lambda r: r["best"]["macro_auc"])
        better = max(results, key=lambda r: r["best"]["macro_auc"])
        lines += [
            "## Divergence analysis (seeds disagree)",
            f"- Weaker seed {worse['seed']} ({worse['best']['macro_auc']:.4f}) vs stronger "
            f"{better['seed']} ({better['best']['macro_auc']:.4f}).",
            f"- Weaker seed peaked at round {worse['best']['round']+1}/{worse['rounds_run']}; "
            f"early_stopped={worse['early_stopped']}.",
            "- Inspect the weaker seed's `run.log` and `val_macro_auroc_vs_rounds` plot: check for "
            "early divergence, a val-loss rise (instability), a very different per-label pos_weight "
            "vector (partition skew), or an early plateau. Likely cause = non-IID partition draw "
            "differing across seeds; report the log evidence rather than the number alone.", "",
        ]
    lines += [
        "## Overall verdict",
        f"- {'PASS — proceed to full Oracle/Plain/EPU.' if (all_improved and consistent and agg['macro_auc']['mean'] >= 0.75) else 'REVIEW — see per-seed verdicts and consistency above.'}",
    ]
    (root / "BASELINE_MULTISEED_REPORT.md").write_text("\n".join(lines))
    log.info("MULTI-SEED DONE | macro-AUROC mean=%.4f std=%.4f spread=%.4f | consistent=%s | report=%s",
             agg["macro_auc"]["mean"], agg["macro_auc"]["std"], auc_spread, consistent,
             root / "BASELINE_MULTISEED_REPORT.md")
    print("BASELINE (2-SEED) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
