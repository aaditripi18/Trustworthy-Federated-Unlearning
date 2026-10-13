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
    "macro_auc", "micro_auc", "mAP", "macro_f1", "micro_f1",  # diagnostic
    "precision", "recall",
    "forget_loss", "mia_auc", "pred_dist_to_oracle",       # forgetting
    "expl_cos_to_oracle", "expl_pearson_to_oracle",
    "expl_iou_to_oracle", "expl_mse_to_oracle",            # explanation vs Oracle
    "expl_cos_to_original", "expl_pearson_to_original",
    "expl_iou_to_original", "expl_mse_to_original",        # explanation stability vs Original
    "wall_time_s", "rounds", "speedup_vs_oracle",          # efficiency
]


def config_label(method: str, lam) -> str:
    return method if method != "epu" else f"epu@{lam:g}"


# ---------------------------------------------------------------- torch part

def run_single(base_cfg: dict, seed: int, lambdas: list[float], logger=None,
               ckpt_dir=None, reuse_ckpt_dir=None, extra=None, standard=True,
               only=None) -> list[dict]:
    """Run the full chain for one seed and return one result dict per model.

    ckpt_dir : optional directory for per-seed model checkpoints. When set, each
               finalized model (Original/Oracle/Plain/EPU) is atomically saved
               right after it is produced. Artifact-only: the saves read
               state_dicts and serialize them; they consume no RNG and mutate
               nothing, so training, evaluation, and all outputs are unchanged.
               No-op when None.
    extra    : optional list of additional unlearning runs, each a dict
               {"label", "method", "rounds", "lambda"} (e.g. the gradient-ascent
               baseline "pga"). They are evaluated with exactly the same code as
               Plain / EPU and reported under config == label.
    standard : when False, Plain and EPU are not re-run (their results already
               exist); only Original, Oracle and the ``extra`` runs are evaluated.
               Default True leaves the original behaviour unchanged.
    only     : "original" or "oracle" trains and saves just that model and returns its
               timing row (seed, config, wall_time_s, rounds) without evaluation, so
               the two from-scratch trainings can run as separate jobs. A later call
               with ``reuse_ckpt_dir`` then loads both. Default None: unchanged.
    """
    import torch  # noqa: F401

    from src.data.dataset import build_datamodule
    from src.eval.engine import collect_predictions
    from src.eval.metrics import multilabel_metrics
    from src.explain.extract import explain_loader, model_grid
    from src.explain.metrics import compare_maps
    from src.federated.client import get_params, set_params
    from src.federated.schedule import make_lr_schedule
    from src.federated.server import run_federated
    from src.models.vit import build_model, resolve_device
    from src.unlearning.methods import get_method
    from src.unlearning.oracle import retrain_oracle
    from src.unlearning.verify import (_forget_eval_loader, bce_per_sample,
                                       mia_auc, pred_distance)
    from src.utils.imbalance import compute_pos_weight
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

    # ---- frozen baseline regime: ONE pos_weight (all methods), two schedules ----
    fcfg = cfg["federated"]
    patience = fcfg.get("early_stop_patience")
    pos_weight = (compute_pos_weight(dm["manifest"][dm["manifest"].split == "train"],
                                     cap=fcfg.get("pos_weight_cap", 20.0))
                  if fcfg.get("use_pos_weight", False) else None)
    scratch_sched = make_lr_schedule(fcfg["lr"], fed_rounds, fcfg.get("warmup_rounds", 0),
                                     fcfg.get("lr_schedule", "constant"))          # Original/Oracle
    ft_lr = cfg["unlearn"].get("finetune_lr", fcfg["lr"])
    ft_sched = make_lr_schedule(ft_lr, ft_rounds, 0, "constant")                    # Plain/EPU

    def _val_metrics(model):
        yt, ps = collect_predictions(model, dm["val"], device)   # VAL: selection only
        return multilabel_metrics(yt, ps, thr)

    def make_cb(name, total, track_best):
        """Per-round val diagnostics logger; best-checkpoint + early-stop if track_best."""
        st = {"best": -1.0, "best_params": None, "no_improve": 0, "stopped": False, "rounds_run": 0}

        def cb(rnd, model, train_loss, lr):
            st["rounds_run"] = rnd + 1
            m = _val_metrics(model)
            tag = ""
            if track_best:
                if m["macro_auc"] > st["best"] + 1e-5:
                    st["best"] = m["macro_auc"]; st["best_params"] = get_params(model)
                    st["no_improve"] = 0; tag = "  <== NEW BEST (checkpoint)"
                else:
                    st["no_improve"] += 1
            if logger:
                logger.info(
                    "%s | round %d/%d | val macroAUC=%.4f microAUC=%.4f mAP=%.4f F1=%.4f "
                    "P=%.4f R=%.4f | lr=%.2e | train_loss=%s%s", name, rnd + 1, total,
                    m["macro_auc"], m["micro_auc"], m["mAP"], m["macro_f1"], m["precision"],
                    m["recall"], lr, "n/a" if train_loss is None else f"{train_loss:.4f}", tag)
            if track_best and patience and st["no_improve"] >= patience:
                st["stopped"] = True
                if logger:
                    logger.info("%s | EARLY STOP (best val macroAUC=%.4f @round %d)",
                                name, st["best"], rnd + 1 - st["no_improve"])
                return True
            return False
        return cb, st

    # Progress/ETA logging: count model builds up front for "[i/total]" + ETA.
    total_stages = 2 + ((1 + len(lambdas)) if standard else 0) + len(extra or [])
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

    def _save_ckpt(model, name):
        """Atomic checkpoint write (tmp -> replace). Artifact-only: reads the
        finalized model's state_dict and serializes it — consumes no RNG and
        mutates nothing. No-op when ckpt_dir is None."""
        if ckpt_dir is None:
            return
        d = Path(ckpt_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": model.state_dict(), "seed": seed,
                   "model": cfg["model"]["name"], "config": cfg}
        tmp = d / (name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(d / name)                                # atomic rename
        if logger:
            logger.info("saved checkpoint %s", d / name)

    def _load_saved(name, cfgname):
        """Reuse a saved base/oracle checkpoint instead of retraining.
        ORCHESTRATION ONLY: weights are loaded verbatim; no training, no RNG use,
        no change to the EPU/Plain algorithm or loss. Returns (model, wall_time,
        rounds), or None to fall back to training."""
        if reuse_ckpt_dir is None:
            return None
        p = Path(reuse_ckpt_dir) / name
        if not p.exists():
            return None
        ck = torch.load(p, map_location=device)
        m = build_model(cfg["model"], cfg["data"]["image_size"],
                        cfg["data"]["num_labels"]).to(device)
        m.load_state_dict(ck["state_dict"])
        wt, rd = float("nan"), -1                            # carry forward source timing if present
        src = Path(reuse_ckpt_dir).parent / "final_results.csv"
        if src.exists():
            import csv as _csv
            for r in _csv.DictReader(open(src)):
                if int(r["seed"]) == seed and r["config"] == cfgname:
                    wt, rd = float(r["wall_time_s"]), int(r["rounds"]); break
        if logger:
            logger.info("REUSED %s from %s (skipped training)", name, reuse_ckpt_dir)
        return m, wt, rd

    # --- Original: from-scratch regime, best-checkpoint on val (or reuse saved) ---
    _r = _load_saved("original.pt", "original")
    if _r is not None:
        base = {"model": _r[0]}; base_t, base_rounds = _r[1], _r[2]
        orig_params = get_params(base["model"])
    elif only != "oracle":
        base_cb, base_st = make_cb("original", fed_rounds, track_best=True)
        t0 = time.perf_counter()
        base = run_federated(cfg, logger=logger, dm=dm, pos_weight=pos_weight,
                             lr_schedule=scratch_sched, on_round_end=base_cb)
        base_t = time.perf_counter() - t0
        if base_st["best_params"] is not None:
            set_params(base["model"], base_st["best_params"])   # use BEST, not final
        orig_params = get_params(base["model"])                 # warm-start source for Plain/EPU
        base_rounds = base_st["rounds_run"]
        _save_ckpt(base["model"], "original.pt")                # == the evaluated Original
    _tick("original")
    if only == "original":
        return [{"seed": seed, "config": "original", "wall_time_s": float(base_t),
                 "rounds": int(base_rounds)}]

    # --- Oracle: IDENTICAL regime, only the forget client is excluded (or reuse saved) ---
    _ro = _load_saved("oracle.pt", "oracle")
    if _ro is not None:
        oracle = {"model": _ro[0], "wall_time_s": _ro[1]}; oracle_rounds = _ro[2]
    else:
        oracle_cb, oracle_st = make_cb("oracle", fed_rounds, track_best=True)
        oracle = retrain_oracle(cfg, forget, logger=logger, dm=dm, pos_weight=pos_weight,
                                lr_schedule=scratch_sched, on_round_end=oracle_cb)
        if oracle_st["best_params"] is not None:
            set_params(oracle["model"], oracle_st["best_params"])
        oracle_rounds = oracle_st["rounds_run"]
        _save_ckpt(oracle["model"], "oracle.pt")            # == the evaluated Oracle
    _tick("oracle")
    if only == "oracle":
        return [{"seed": seed, "config": "oracle", "wall_time_s": float(oracle["wall_time_s"]),
                 "rounds": int(oracle_rounds)}]

    plain, plain_st, epus, epu_sts = None, None, {}, {}
    if standard:
        # --- Plain: warm-start finetune (final model = operating point) ---
        plain_cb, plain_st = make_cb("finetune_retain", ft_rounds, track_best=False)
        plain = get_method("finetune_retain")(cfg, forget, orig_params, logger=logger, dm=dm,
                                              pos_weight=pos_weight, lr_schedule=ft_sched,
                                              on_round_end=plain_cb)
        _save_ckpt(plain["model"], "plain.pt")                  # == the evaluated Plain
        _tick("finetune_retain")

        # --- EPU per lambda: identical to Plain except the explanation loss ---
        epus, epu_sts = {}, {}
        for lam in lambdas:
            cfg["unlearn"]["explanation_lambda"] = lam
            epu_cb, epu_sts[lam] = make_cb(f"epu@{lam:g}", ft_rounds, track_best=False)
            epus[lam] = get_method("epu")(cfg, forget, orig_params, logger=logger, dm=dm,
                                          pos_weight=pos_weight, lr_schedule=ft_sched,
                                          on_round_end=epu_cb)
            _save_ckpt(epus[lam]["model"], f"epu_lambda{lam}.pt")  # == the evaluated EPU@lam
            _tick(f"epu@{lam:g}")

    # --- additional unlearning baselines (same evaluation path as Plain / EPU) ---
    extras = []
    for spec in (extra or []):
        saved = (cfg["unlearn"].get("explanation_lambda"), cfg["unlearn"].get("finetune_rounds"))
        cfg["unlearn"]["explanation_lambda"] = spec.get("lambda", 0.0)
        cfg["unlearn"]["finetune_rounds"] = spec["rounds"]
        x_cb, x_st = make_cb(spec["label"], spec["rounds"], track_best=False)
        x_out = get_method(spec["method"])(cfg, forget, orig_params, logger=logger, dm=dm,
                                           pos_weight=pos_weight, lr_schedule=ft_sched,
                                           on_round_end=x_cb)
        cfg["unlearn"]["explanation_lambda"], cfg["unlearn"]["finetune_rounds"] = saved
        _save_ckpt(x_out["model"], f"{spec['label']}.pt")
        extras.append((spec, x_out, x_st))
        _tick(spec["label"])

    grid = model_grid(base["model"], cfg["data"]["image_size"], cfg["model"]["patch_size"])
    def maps_of(m):
        return explain_loader(m, dm["test"], grid, device, em, max_images=k)[0]  # probes on TEST

    # oracle reference probes (held-out TEST)
    _, o_pf = collect_predictions(oracle["model"], forget_loader, device)
    _, o_pv = collect_predictions(oracle["model"], dm["test"], device)
    oracle_probe = np.concatenate([o_pf, o_pv])
    oracle_maps = maps_of(oracle["model"])
    original_maps = maps_of(base["model"])                   # reference for explanation STABILITY

    def evaluate(model, method, lam, wall_time, rounds):
        yv_true, pv = collect_predictions(model, dm["test"], device)     # TEST: reporting split
        diag = multilabel_metrics(yv_true, pv, thr)
        yf_true, pf = collect_predictions(model, forget_loader, device)  # forget: labels + preds
        member_loss = bce_per_sample(yf_true, pf)          # forgotten-client samples
        nonmember_loss = bce_per_sample(yv_true, pv)       # held-out TEST samples (non-member)
        row = {"seed": seed, "method": method, "lambda": lam,
               "config": config_label(method, lam),
               "macro_auc": diag["macro_auc"], "micro_auc": diag["micro_auc"],
               "mAP": diag["mAP"], "macro_f1": diag["macro_f1"], "micro_f1": diag["micro_f1"],
               "precision": diag["precision"], "recall": diag["recall"],
               "forget_loss": float(member_loss.mean()),
               "mia_auc": mia_auc(member_loss, nonmember_loss)}
        probe = np.concatenate([pf, pv])
        m_maps = maps_of(model)                              # this model's maps, computed once
        # --- explanation closeness to the ORACLE (existing secondary metric) ---
        if method == "oracle":
            row.update(pred_dist_to_oracle=0.0, expl_cos_to_oracle=1.0,
                       expl_pearson_to_oracle=1.0, expl_iou_to_oracle=1.0,
                       expl_mse_to_oracle=0.0)
        else:
            cm = compare_maps(m_maps, oracle_maps, iou_thr)
            row.update(pred_dist_to_oracle=pred_distance(probe, oracle_probe),
                       expl_cos_to_oracle=cm["cosine_similarity"],
                       expl_pearson_to_oracle=cm["pearson"],
                       expl_iou_to_oracle=cm["iou"], expl_mse_to_oracle=cm["mse"])
        # --- explanation STABILITY relative to the ORIGINAL (EXP-A0 primary metric) ---
        if method == "original":
            row.update(expl_cos_to_original=1.0, expl_pearson_to_original=1.0,
                       expl_iou_to_original=1.0, expl_mse_to_original=0.0)
        else:
            co = compare_maps(m_maps, original_maps, iou_thr)
            row.update(expl_cos_to_original=co["cosine_similarity"],
                       expl_pearson_to_original=co["pearson"],
                       expl_iou_to_original=co["iou"], expl_mse_to_original=co["mse"])
        row["wall_time_s"] = float(wall_time)
        row["rounds"] = int(rounds)
        row["speedup_vs_oracle"] = float(oracle["wall_time_s"] / wall_time) if wall_time > 0 else None
        return row

    results = [evaluate(base["model"], "original", None, base_t, base_rounds)]
    if standard:
        results.append(evaluate(plain["model"], "finetune_retain", None, plain["wall_time_s"],
                                plain_st["rounds_run"]))
    results.append(evaluate(oracle["model"], "oracle", None, oracle["wall_time_s"], oracle_rounds))
    if standard:
        for lam in lambdas:
            results.append(evaluate(epus[lam]["model"], "epu", lam, epus[lam]["wall_time_s"],
                                    epu_sts[lam]["rounds_run"]))
    for spec, x_out, x_st in extras:
        row = evaluate(x_out["model"], spec["method"], spec.get("lambda", 0.0),
                       x_out["wall_time_s"], x_st["rounds_run"])
        row["config"] = spec["label"]
        results.append(row)
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
        "micro_auc": [cell(c, "micro_auc") for c in order],
        "mAP": [cell(c, "mAP") for c in order],
        "macro_f1": [cell(c, "macro_f1") for c in order],
        "precision": [cell(c, "precision") for c in order],
        "recall": [cell(c, "recall") for c in order],
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
