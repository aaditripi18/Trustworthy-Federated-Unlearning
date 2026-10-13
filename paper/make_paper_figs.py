"""Regenerate every data figure of the paper from the measured run artifacts.

Nothing is hand-typed: all points, bars and error bars are computed from
ec2_final_run/tfu/experiments/{full_experiment,exp_a1}/final_results.csv and the
per-round validation lines of the two run.log files. Figures are sized for a
single IEEE column (3.5 in) and use Times so they match the body text.

    python paper/make_paper_figs.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "ec2_final_run/tfu/experiments"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600,
})
COL = 3.5  # IEEE single-column width (inches)
# colour-blind-safe (Okabe-Ito) palette
BLUE, ORANGE, GREEN, GREY, RED, PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#6b6b6b", "#D55E00", "#CC79A7")

A = pd.read_csv(EXP / "full_experiment" / "final_results.csv")   # Setting A
B = pd.read_csv(EXP / "exp_a1" / "final_results.csv")            # Setting B
LAM = [1, 10, 30, 100]
SEEDS = [42, 43, 44]


def ms(df, cfg, col):
    v = df.loc[df.config == cfg, col]
    return v.mean(), v.std(ddof=1)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 3
def parse_val_curves(log_path):
    """{seed: {config: [val macro-AUROC per round]}} from a run.log."""
    out, seed = {}, None
    rx = re.compile(r"\| (\S+) \| round (\d+)/\d+ \| val macroAUC=([0-9.]+)")
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = re.search(r"=== seed (\d+) ===", line)
        if m:
            seed = int(m.group(1)); out.setdefault(seed, {})
            continue
        m = rx.search(line)
        if m and seed is not None:
            out[seed].setdefault(m.group(1), []).append(float(m.group(3)))
    return out


def fig_training():
    ca = parse_val_curves(EXP / "full_experiment" / "run.log")
    cb = parse_val_curves(EXP / "exp_a1" / "run.log")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        for cfg, c, ls in (("original", BLUE, "-"), ("oracle", GREEN, "--")):
            y = ca[s][cfg]
            ax.plot(range(1, len(y) + 1), y, ls, color=c, lw=0.9, alpha=0.55 + 0.2 * i,
                    label={"original": "Original (5 clients)", "oracle": "Oracle (4 clients)"}[cfg]
                    if i == 0 else None)
            b = int(np.argmax(y))
            ax.plot(b + 1, y[b], "o", color=c, ms=2.6)
    ax.set_xticks(range(2, 13, 2))
    ax.set_xlabel("FedAvg round (from scratch)")
    ax.set_ylabel("validation macro-AUROC")
    ax.set_title("(a) federated training", pad=3)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)

    ax = axes[1]
    def mean_curve(curves, cfg):
        return np.mean([curves[s][cfg] for s in SEEDS], axis=0)
    ya = mean_curve(ca, "finetune_retain")
    yb = mean_curve(cb, "finetune_retain")
    ye = mean_curve(cb, "epu@100")
    ax.plot(range(1, len(ya) + 1), ya, "s-", color=GREY, label="Plain, setting A")
    ax.plot(range(1, len(yb) + 1), yb, "o-", color=ORANGE, label="Plain, setting B")
    ax.plot(range(1, len(ye) + 1), ye, "^--", color=BLUE, label=r"EPU $\lambda$=100, B")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("unlearning round")
    ax.set_title("(b) unlearning", pad=3)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    fig.subplots_adjust(left=0.135, right=0.995, top=0.9, bottom=0.21, wspace=0.07)
    save(fig, "fig_training")


# ------------------------------------------------------------------ Fig. 4
def fig_dose_response():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
    x = np.arange(len(LAM))
    for ax, col, lab, tag in zip(
            axes, ["expl_cos_to_original", "expl_iou_to_original"],
            ["(a) cosine to Original", "(b) IoU to Original"], ["", ""]):
        pm, ps = ms(B, "finetune_retain", col)
        om, osd = ms(B, "oracle", col)
        ax.axhspan(pm - ps, pm + ps, color=GREY, alpha=0.16, lw=0)
        ax.axhline(pm, color=GREY, ls="--", lw=0.9, label="Plain")
        ax.axhspan(om - osd, om + osd, color=GREEN, alpha=0.13, lw=0)
        ax.axhline(om, color=GREEN, ls=":", lw=0.9, label="Oracle")
        m = [ms(B, f"epu@{l}", col)[0] for l in LAM]
        s = [ms(B, f"epu@{l}", col)[1] for l in LAM]
        ax.errorbar(x, m, yerr=s, marker="o", color=BLUE, capsize=2, capthick=0.7,
                    elinewidth=0.7, label="EPU")
        ax.set_xticks(x); ax.set_xticklabels([str(l) for l in LAM])
        ax.set_xlabel(r"explanation weight $\lambda$")
        ax.set_ylabel(lab)
    axes[0].set_ylim(0.95, 1.002)
    axes[1].set_ylim(0.45, 0.93)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, handlelength=1.8,
               bbox_to_anchor=(0.56, 1.01))
    fig.subplots_adjust(left=0.14, right=0.975, top=0.88, bottom=0.235, wspace=0.42)
    save(fig, "fig_dose_response")


# ------------------------------------------------------------------ Fig. 5
def fig_paired_gain():
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    w = 0.25
    col = "expl_cos_to_original"
    for i, sd in enumerate(SEEDS):
        p = B[(B.seed == sd) & (B.config == "finetune_retain")][col].iloc[0]
        g = [B[(B.seed == sd) & (B.config == f"epu@{l}")][col].iloc[0] - p for l in LAM]
        ax.bar(np.arange(len(LAM)) + (i - 1) * w, g, w, label=f"seed {sd}",
               color=[BLUE, ORANGE, GREEN][i], ec="white", lw=0.3)
    sd_plain = B.loc[B.config == "finetune_retain", col].std(ddof=1)
    ax.axhline(sd_plain, color=RED, ls=":", lw=1.0,
               label=f"Plain across-seed s.d. ({sd_plain:.4f})")
    ax.set_xticks(range(len(LAM)))
    ax.set_xticklabels([rf"$\lambda$ = {l}" for l in LAM])
    ax.set_ylabel("paired gain in cosine\nto Original (EPU $-$ Plain)")
    ax.set_ylim(0, 0.041)
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=1.0, handlelength=1.4)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.185, right=0.995, top=0.97, bottom=0.16)
    save(fig, "fig_paired_gain")


# ------------------------------------------------------------------ Fig. 6
def fig_two_references():
    order = ["oracle", "finetune_retain", "epu@1", "epu@10", "epu@30", "epu@100"]
    names = ["Oracle", "Plain", "EPU\n1", "EPU\n10", "EPU\n30", "EPU\n100"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True)
    x = np.arange(len(order))
    for ax, ref, tag in zip(axes, ["original", "oracle"],
                            ["(a) reference: Original", "(b) reference: Oracle"]):
        cfgs = order if ref == "original" else ["original"] + order[1:]
        labs = names if ref == "original" else ["Orig."] + names[1:]
        m = [ms(B, c, f"expl_iou_to_{ref}")[0] for c in cfgs]
        s = [ms(B, c, f"expl_iou_to_{ref}")[1] for c in cfgs]
        cols = [GREEN if c == "oracle" else PURPLE if c == "original" else
                GREY if c == "finetune_retain" else BLUE for c in cfgs]
        ax.bar(x, m, yerr=s, color=cols, ec="white", lw=0.3, capsize=1.8,
               error_kw={"elinewidth": 0.7, "capthick": 0.7})
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
        ax.set_title(tag, pad=3)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("attention IoU to reference")
    axes[0].set_ylim(0.4, 0.93)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.885, bottom=0.2, wspace=0.07)
    save(fig, "fig_two_references")


# ------------------------------------------------------------------ Fig. 7
def fig_cost():
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    rows = [("Original", A, "original", PURPLE), ("Oracle", A, "oracle", GREEN),
            ("Plain (A)", A, "finetune_retain", GREY), ("EPU (A)", A, "epu@1", BLUE),
            ("Plain (B)", B, "finetune_retain", GREY), ("EPU (B)", B, "epu@100", BLUE)]
    m = [ms(df, c, "wall_time_s")[0] / 3600 for _, df, c, _ in rows]
    s = [ms(df, c, "wall_time_s")[1] / 3600 for _, df, c, _ in rows]
    x = np.arange(len(rows))
    ax.bar(x, m, yerr=s, color=[r[3] for r in rows], ec="white", lw=0.3, capsize=1.8,
           error_kw={"elinewidth": 0.7, "capthick": 0.7})
    for xi, (lab, df, c, _), mi, si in zip(x, rows, m, s):
        if lab in ("Original", "Oracle"):
            continue
        sp = ms(df, c, "speedup_vs_oracle")[0]
        ax.text(xi, mi + si + 0.12, f"{sp:.1f}$\\times$", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylabel("wall-clock time (h)")
    ax.set_ylim(0, 6.3)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.13)
    save(fig, "fig_cost")


if __name__ == "__main__":
    fig_training()
    fig_dose_response()
    fig_paired_gain()
    fig_two_references()
    fig_cost()
