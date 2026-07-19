"""Publication figures from the evaluation results (isolated from training).

Consumes the tidy results DataFrame produced by src/eval/experiment.py and
renders four figures, each saved as PNG and PDF:
  * lambda ablation curve (explanation closeness to oracle vs lambda),
  * forgetting vs explanation trade-off,
  * oracle comparison (per-config explanation closeness),
  * runtime comparison.

Uses the non-interactive Agg backend (no GUI). Reuses experiment.aggregate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiment import aggregate  # noqa: E402


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _ordered(summ):
    epu = sorted([c for c in summ.index if c.startswith("epu@")],
                 key=lambda x: float(x.split("@")[1]))
    return [c for c in ["original", "finetune_retain", *epu, "oracle"] if c in summ.index]


def fig_lambda_ablation(summ, out_dir):
    epu = sorted([c for c in summ.index if c.startswith("epu@")],
                 key=lambda x: float(x.split("@")[1]))
    lams = [float(c.split("@")[1]) for c in epu]
    y = [summ.loc[c, "expl_cos_to_oracle_mean"] for c in epu]
    e = [summ.loc[c, "expl_cos_to_oracle_std"] for c in epu]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(lams, y, yerr=e, marker="o", capsize=3, label="EPU")
    if "finetune_retain" in summ.index:
        ax.axhline(summ.loc["finetune_retain", "expl_cos_to_oracle_mean"],
                   ls="--", color="gray", label="plain unlearning")
    ax.set_xlabel(r"explanation-loss weight $\lambda$")
    ax.set_ylabel("explanation cosine to oracle")
    ax.set_title("Lambda ablation")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_lambda_ablation")


def fig_tradeoff(summ, out_dir):
    order = _ordered(summ)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for c in order:
        x = summ.loc[c, "pred_dist_to_oracle_mean"]      # forgetting: closer to oracle = smaller
        yv = summ.loc[c, "expl_cos_to_oracle_mean"]       # explanation: closer to oracle = larger
        ax.scatter(x, yv, s=40)
        ax.annotate(c, (x, yv), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("prediction distance to oracle (lower = better forgetting)")
    ax.set_ylabel("explanation cosine to oracle (higher = better)")
    ax.set_title("Forgetting vs. explanation trade-off")
    _save(fig, out_dir, "fig_tradeoff")


def fig_oracle_comparison(summ, out_dir):
    order = [c for c in _ordered(summ) if c != "oracle"]
    y = [summ.loc[c, "expl_cos_to_oracle_mean"] for c in order]
    e = [summ.loc[c, "expl_cos_to_oracle_std"] for c in order]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(len(order)), y, yerr=e, capsize=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("explanation cosine to oracle")
    ax.set_title("Oracle comparison (explanation)")
    _save(fig, out_dir, "fig_oracle_comparison")


def fig_runtime(summ, out_dir):
    order = _ordered(summ)
    y = [summ.loc[c, "wall_time_s_mean"] for c in order]
    e = [summ.loc[c, "wall_time_s_std"] for c in order]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(len(order)), y, yerr=np.nan_to_num(e), capsize=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("Runtime comparison")
    _save(fig, out_dir, "fig_runtime")


def generate_all_figures(results_df, out_dir) -> list[str]:
    summ = aggregate(results_df).set_index("config")
    fig_lambda_ablation(summ, out_dir)
    fig_tradeoff(summ, out_dir)
    fig_oracle_comparison(summ, out_dir)
    fig_runtime(summ, out_dir)
    return ["fig_lambda_ablation", "fig_tradeoff", "fig_oracle_comparison", "fig_runtime"]
