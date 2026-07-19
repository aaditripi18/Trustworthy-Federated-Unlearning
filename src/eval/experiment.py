"""M7 evaluation framework (separate from training code).

Orchestrates the four-method comparison — Original, plain Unlearned
(finetune_retain), EPU at several lambda values, and the retrain Oracle — across
four axes (diagnostic, forgetting, explanation, efficiency), over multiple seeds,
and builds IEEE-style tables. Reuses M2-M6 modules; adds no algorithm.

Split into:
  * run_single(...)  -> per-seed results (torch; heavy);
  * aggregate / build_tables / write_outputs (pure pandas; unit-tested).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# numeric metric columns produced per evaluated model
METRIC_COLS = [
    "macro_auc", "mAP", "macro_f1", "micro_f1",           # diagnostic
    "forget_loss", "mia_auc", "pred_dist_to_oracle",       # forgetting
    "expl_cos_to_oracle", "expl_pearson_to_oracle",
    "expl_iou_to_oracle", "expl_mse_to_oracle",            # explanation
    "wall_time_s", "rounds", "speedup_vs_oracle",          # efficiency
]


def config_label(method: str, lam) -> str:
    return method if method != "epu" else f"epu@{lam:g}"


# ---------------------------------------------------------------- torch part

def run_single(base_cfg: dict, seed: int, lambdas: list[float], logger=None) -> list[dict]:
    """Run the full chain for one seed and return one result dict per model."""
    import torch  # noqa: F401

    from src.data.dataset import build_datamodule
    from src.eval.engine import collect_predictions
    from src.eval.metrics import multilabel_metrics
    from src.explain.extract import explain_loader, model_grid
    from src.explain.metrics import compare_maps
    from src.federated.client import get_params
    from src.federated.server import run_federated
    from src.models.vit import build_model, resolve_device
    from src.unlearning.methods import get_method
    from src.unlearning.oracle import retrain_oracle
    from src.unlearning.verify import (_forget_eval_loader, bce_per_sample,
                                       mia_auc, pred_distance)
    from src.utils.seed import set_seed

    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    set_seed(seed)
    device = resolve_device(cfg["unlearn"].get("device", "auto"))
    forget = cfg["unlearn"]["forget_client"]
    thr = cfg["unlearn"].get("threshold", 0.5)
    k = cfg["explain"].get("num_probe_images", 16)
    em = cfg["explain"].get("method", "attention_rollout")
    iou_thr = cfg["explain"].get("iou_threshold", 0.5)
    fed_rounds = cfg["federated"]["num_rounds"]
    ft_rounds = cfg["unlearn"].get("finetune_rounds", 2)

    # Build the datamodule ONCE and reuse it for every model in this seed.
    # build_datamodule is a pure function of (cfg, seed) using only a locally
    # seeded RNG, so every consumer sees an identical dm; this only avoids
    # re-decoding images and re-partitioning on each of the ~7 model builds.
    dm = build_datamodule(cfg)
    forget_loader = _forget_eval_loader(cfg, dm, forget)

    # Progress/ETA logging (Opt 5): count the model builds up front so each log
    # line can report "[i/total]" and a running ETA. Purely cosmetic.
    total_stages = 3 + len(lambdas)   # original + plain + oracle + one per lambda
    seed_t0 = time.perf_counter()
    _done = 0

    def _tick(name):
        nonlocal _done
        _done += 1
        if logger:
            el = time.perf_counter() - seed_t0
            eta = el / _done * (total_stages - _done)
            logger.info("seed %d | stage %d/%d done (%s) | elapsed %.1fs | eta %.1fs",
                        seed, _done, total_stages, name, el, eta)

    # base (original), oracle, plain, epu@lambda
    t0 = time.perf_counter()
    base = run_federated(cfg, logger=logger, dm=dm)
    base_t = time.perf_counter() - t0
    orig_params = get_params(base["model"])
    _tick("original")
    oracle = retrain_oracle(cfg, forget, logger=logger, dm=dm)
    _tick("oracle")
    plain = get_method("finetune_retain")(cfg, forget, orig_params, logger=logger, dm=dm)
    _tick("finetune_retain")
    epus = {}
    for lam in lambdas:
        cfg["unlearn"]["explanation_lambda"] = lam
        epus[lam] = get_method("epu")(cfg, forget, orig_params, logger=logger, dm=dm)
        _tick(f"epu@{lam:g}")

    grid = model_grid(base["model"], cfg["data"]["image_size"], cfg["model"]["patch_size"])
    def maps_of(m):
        return explain_loader(m, dm["val"], grid, device, em, max_images=k)[0]

    # oracle reference probes
    _, o_pf = collect_predictions(oracle["model"], forget_loader, device)
    _, o_pv = collect_predictions(oracle["model"], dm["val"], device)
    oracle_probe = np.concatenate([o_pf, o_pv])
    oracle_maps = maps_of(oracle["model"])

    def evaluate(model, method, lam, wall_time, rounds):
        yv_true, pv = collect_predictions(model, dm["val"], device)      # val: labels + preds
        diag = multilabel_metrics(yv_true, pv, thr)
        yf_true, pf = collect_predictions(model, forget_loader, device)  # forget: labels + preds
        member_loss = bce_per_sample(yf_true, pf)          # trained-on (for original) samples
        nonmember_loss = bce_per_sample(yv_true, pv)       # held-out val samples
        row = {"seed": seed, "method": method, "lambda": lam,
               "config": config_label(method, lam),
               "macro_auc": diag["macro_auc"], "mAP": diag["mAP"],
               "macro_f1": diag["macro_f1"], "micro_f1": diag["micro_f1"],
               "forget_loss": float(member_loss.mean()),
               "mia_auc": mia_auc(member_loss, nonmember_loss)}
        probe = np.concatenate([pf, pv])
        if method == "oracle":
            row.update(pred_dist_to_oracle=0.0, expl_cos_to_oracle=1.0,
                       expl_pearson_to_oracle=1.0, expl_iou_to_oracle=1.0,
                       expl_mse_to_oracle=0.0)
        else:
            cm = compare_maps(maps_of(model), oracle_maps, iou_thr)
            row.update(pred_dist_to_oracle=pred_distance(probe, oracle_probe),
                       expl_cos_to_oracle=cm["cosine_similarity"],
                       expl_pearson_to_oracle=cm["pearson"],
                       expl_iou_to_oracle=cm["iou"], expl_mse_to_oracle=cm["mse"])
        row["wall_time_s"] = float(wall_time)
        row["rounds"] = int(rounds)
        row["speedup_vs_oracle"] = float(oracle["wall_time_s"] / wall_time) if wall_time > 0 else None
        return row

    results = [
        evaluate(base["model"], "original", None, base_t, fed_rounds),
        evaluate(plain["model"], "finetune_retain", None, plain["wall_time_s"], ft_rounds),
        evaluate(oracle["model"], "oracle", None, oracle["wall_time_s"], fed_rounds),
    ]
    for lam in lambdas:
        results.append(evaluate(epus[lam]["model"], "epu", lam, epus[lam]["wall_time_s"], ft_rounds))
    return results


# ---------------------------------------------------------------- pure part

def results_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std across seeds, grouped by config."""
    num = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby("config", dropna=False)[num].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return g.reset_index()


def _fmt(m, s) -> str:
    if pd.isna(m):
        return "n/a"
    return f"{m:.3f}±{0.0 if pd.isna(s) else s:.3f}"


def build_tables(df: pd.DataFrame) -> dict:
    summ = aggregate(df).set_index("config")

    def cell(cfg, metric):
        return _fmt(summ.loc[cfg, f"{metric}_mean"], summ.loc[cfg, f"{metric}_std"])

    order = [c for c in ["original", "finetune_retain", *sorted(
        [c for c in summ.index if c.startswith("epu@")],
        key=lambda x: float(x.split("@")[1])), "oracle"] if c in summ.index]

    comparison = pd.DataFrame({
        "config": order,
        "macro_auc": [cell(c, "macro_auc") for c in order],
        "mia_auc": [cell(c, "mia_auc") for c in order],
        "forget_loss": [cell(c, "forget_loss") for c in order],
        "expl_cos_to_oracle": [cell(c, "expl_cos_to_oracle") for c in order],
        "pred_dist_to_oracle": [cell(c, "pred_dist_to_oracle") for c in order],
        "wall_time_s": [cell(c, "wall_time_s") for c in order],
        "speedup_vs_oracle": [cell(c, "speedup_vs_oracle") for c in order],
    })

    epu_cfgs = [c for c in order if c.startswith("epu@")]
    plain_cos = summ.loc["finetune_retain", "expl_cos_to_oracle_mean"] \
        if "finetune_retain" in summ.index else np.nan
    ablation = pd.DataFrame({
        "config": epu_cfgs,
        "lambda": [float(c.split("@")[1]) for c in epu_cfgs],
        "expl_cos_to_oracle": [cell(c, "expl_cos_to_oracle") for c in epu_cfgs],
        "preservation_gain_vs_plain": [
            round(summ.loc[c, "expl_cos_to_oracle_mean"] - plain_cos, 4) for c in epu_cfgs],
        "mia_auc": [cell(c, "mia_auc") for c in epu_cfgs],
        "macro_auc": [cell(c, "macro_auc") for c in epu_cfgs],
    })

    tradeoff = pd.DataFrame({
        "config": order,
        "forgetting_pred_dist_to_oracle": [cell(c, "pred_dist_to_oracle") for c in order],
        "forgetting_mia_auc": [cell(c, "mia_auc") for c in order],
        "explanation_cos_to_oracle": [cell(c, "expl_cos_to_oracle") for c in order],
    })

    oracle_cmp = pd.DataFrame({
        "config": [c for c in order if c != "oracle"],
        "pred_dist_to_oracle": [cell(c, "pred_dist_to_oracle") for c in order if c != "oracle"],
        "expl_cos_to_oracle": [cell(c, "expl_cos_to_oracle") for c in order if c != "oracle"],
        "expl_iou_to_oracle": [cell(c, "expl_iou_to_oracle") for c in order if c != "oracle"],
    })
    return {"comparison": comparison, "ablation_lambda": ablation,
            "tradeoff": tradeoff, "oracle_comparison": oracle_cmp}


def honest_findings(df: pd.DataFrame) -> dict:
    """Objective, data-driven conclusions (no massaging)."""
    summ = aggregate(df).set_index("config")
    epu = [c for c in summ.index if c.startswith("epu@")]
    plain_cos = summ.loc["finetune_retain", "expl_cos_to_oracle_mean"] \
        if "finetune_retain" in summ.index else float("nan")
    best_epu = max(epu, key=lambda c: summ.loc[c, "expl_cos_to_oracle_mean"]) if epu else None
    best_cos = summ.loc[best_epu, "expl_cos_to_oracle_mean"] if best_epu else float("nan")
    any_epu_beats_plain = bool(best_epu is not None and best_cos > plain_cos)
    return {
        "metric": "explanation cosine similarity to oracle (higher = closer)",
        "plain_unlearning_cos_to_oracle": None if pd.isna(plain_cos) else round(float(plain_cos), 4),
        "best_epu_config": best_epu,
        "best_epu_cos_to_oracle": None if pd.isna(best_cos) else round(float(best_cos), 4),
        "any_epu_beats_plain_on_explanation": any_epu_beats_plain,
        "statement": (
            f"Best EPU ({best_epu}) {'>' if any_epu_beats_plain else '<='} plain unlearning "
            f"on explanation closeness to oracle. Reported as measured; not tuned to favor EPU."),
    }


_DEFAULT_NAMES = {
    "results": "evaluation_results.csv", "summary_csv": "multi_seed_summary.csv",
    "tables": "evaluation_tables.csv", "ablation": "ablation_lambda.csv",
    "summary_json": "evaluation_summary.json",
}


def write_outputs(out_dir, results: list[dict], meta: dict, filenames: dict | None = None) -> dict:
    names = dict(_DEFAULT_NAMES)
    if filenames:
        names.update(filenames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = results_dataframe(results)
    df.to_csv(out / names["results"], index=False)
    aggregate(df).to_csv(out / names["summary_csv"], index=False)
    tables = build_tables(df)
    tables["comparison"].to_csv(out / names["tables"], index=False)
    tables["ablation_lambda"].to_csv(out / names["ablation"], index=False)
    findings = honest_findings(df)
    summary = {**meta,
               "tables": {k: v.to_dict(orient="records") for k, v in tables.items()},
               "findings": findings}
    with open(out / names["summary_json"], "w") as f:
        json.dump(summary, f, indent=2)
    return {"tables": tables, "findings": findings, "results_df": df}
