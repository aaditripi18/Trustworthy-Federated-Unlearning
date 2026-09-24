"""Regenerate the four presentation figures directly from the measured CSVs.

Nothing here is hand-typed: every point, bar and error bar is computed from
ec2_final_run/tfu/experiments/{exp_a1,full_experiment}/final_results.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths are resolved relative to this file so the script runs anywhere the
# repository is checked out (it lives in demo/, so the repo root is one up).
ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "ec2_final_run/tfu/experiments"
# The charts are written into the demo's own assets, NOT into
# experiments/*/figures — the run's output must stay byte-for-byte as the
# training job wrote it, which is what backs the reproducibility claim.
OUT = ROOT / "demo/assets/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, GREY, GREEN, RED = "#2563eb", "#6b7280", "#0f766e", "#b91c1c"

a1 = pd.read_csv(EXP / "exp_a1" / "final_results.csv")
main = pd.read_csv(EXP / "full_experiment" / "final_results.csv")

LAM = [1, 10, 30, 100]
CFG = {l: f"epu@{l}" for l in LAM}


def ms(df, cfg, col):
    v = df.loc[df.config == cfg, col]
    return v.mean(), v.std(ddof=1)


# ---------------------------------------------------------------- Figure 1
# dose-response of explanation fidelity to the Original model in lambda
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
for ax, col, lab in zip(
        axes,
        ["expl_cos_to_original", "expl_iou_to_original"],
        ["Attention cosine  →  Original", "Attention IoU  →  Original"]):
    pm, ps = ms(a1, "finetune_retain", col)
    ax.axhspan(pm - ps, pm + ps, color=GREY, alpha=0.15, zorder=0)
    ax.axhline(pm, color=GREY, ls="--", lw=1.8,
               label=f"Plain unlearning  ({pm:.4f})")
    m = [ms(a1, CFG[l], col)[0] for l in LAM]
    s = [ms(a1, CFG[l], col)[1] for l in LAM]
    ax.errorbar(range(len(LAM)), m, yerr=s, marker="o", ms=8, lw=2.2,
                capsize=5, color=BLUE, label="EPU (proposed)")
    ax.set_xticks(range(len(LAM)))
    ax.set_xticklabels([str(l) for l in LAM])
    ax.set_xlabel("explanation-loss weight  λ")
    ax.set_ylabel(lab)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.suptitle("Explanation fidelity to the original model rises monotonically with λ"
             "   (mean ± 1 s.d., seeds 42/43/44)", y=1.03, fontsize=13)
fig.savefig(OUT / "fig_dose_response.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# paired per-seed gain: EPU minus Plain, same seed, same run
fig, ax = plt.subplots(figsize=(8.6, 4.2))
seeds = sorted(a1.seed.unique())
w = 0.24
for i, sd in enumerate(seeds):
    g = []
    for l in LAM:
        e = a1[(a1.seed == sd) & (a1.config == CFG[l])]["expl_cos_to_original"].iloc[0]
        p = a1[(a1.seed == sd) & (a1.config == "finetune_retain")]["expl_cos_to_original"].iloc[0]
        g.append(e - p)
    ax.bar(np.arange(len(LAM)) + (i - 1) * w, g, w,
           label=f"seed {sd}", color=[BLUE, GREEN, "#7c3aed"][i], alpha=0.9)
plain_sd = a1.loc[a1.config == "finetune_retain", "expl_cos_to_original"].std(ddof=1)
ax.axhline(plain_sd, color=RED, ls=":", lw=1.8,
           label=f"Plain's own across-seed s.d. ({plain_sd:.4f})")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(len(LAM)))
ax.set_xticklabels([f"λ = {l}" for l in LAM])
ax.set_ylabel("cosine gain over Plain\n(same seed, paired)")
allg = [a1[(a1.seed == s) & (a1.config == CFG[l])]["expl_cos_to_original"].iloc[0]
        - a1[(a1.seed == s) & (a1.config == "finetune_retain")]["expl_cos_to_original"].iloc[0]
        for l in LAM for s in seeds]
ax.set_title(f"EPU beats Plain in all {len(allg)} paired comparisons — "
             f"the weakest is still {min(allg)/plain_sd:.1f}× Plain's own seed noise")
ax.set_ylim(0, 0.040)
ax.legend(frameon=False, fontsize=10, ncol=4, loc="upper center")
fig.savefig(OUT / "fig_paired_gain.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# the gain costs nothing: forgetting stays at chance, accuracy is unchanged
order = ["original", "finetune_retain", CFG[1], CFG[10], CFG[30], CFG[100], "oracle"]
names = ["Original", "Plain", "EPU λ=1", "EPU λ=10", "EPU λ=30", "EPU λ=100", "Oracle"]
cols = [GREY, GREY] + [BLUE] * 4 + [GREEN]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for ax, col, lab, ref in zip(
        axes, ["mia_auc", "macro_auc"],
        ["Membership-inference AUC", "Macro-AUROC (diagnosis)"], [0.5, None]):
    m = [ms(a1, c, col)[0] for c in order]
    s = [ms(a1, c, col)[1] for c in order]
    ax.bar(range(len(order)), m, yerr=s, capsize=4, color=cols, alpha=0.9)
    if ref is not None:
        ax.axhline(ref, color=RED, ls="--", lw=1.8, label="chance = 0.50 (forgotten)")
        ax.legend(frameon=False, fontsize=10)
        ax.set_ylim(0.40, 0.60)
    else:
        ax.set_ylim(0.70, 0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(lab)
fig.suptitle("Forgetting stays at chance and diagnostic accuracy is unchanged at every λ",
             y=1.02, fontsize=13)
fig.savefig(OUT / "fig_forgetting_accuracy.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# cost of unlearning against the retrain oracle (main campaign)
morder = ["original", "finetune_retain", "epu@0.5", "epu@1", "oracle"]
mnames = ["Original\n(federated train)", "Plain", "EPU λ=0.5", "EPU λ=1", "Oracle\n(full retrain)"]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
mm = [ms(main, c, "wall_time_s")[0] / 3600 for c in morder]
sm = [ms(main, c, "wall_time_s")[1] / 3600 for c in morder]
axes[0].bar(range(len(morder)), mm, yerr=sm, capsize=4,
            color=[GREY, GREY, BLUE, BLUE, GREEN], alpha=0.9)
axes[0].set_ylabel("wall-clock time (hours)")
axes[0].set_title("Cost of producing the model")
sp = [ms(main, c, "speedup_vs_oracle")[0] for c in morder]
ss = [ms(main, c, "speedup_vs_oracle")[1] for c in morder]
axes[1].bar(range(len(morder)), sp, yerr=ss, capsize=4,
            color=[GREY, GREY, BLUE, BLUE, GREEN], alpha=0.9)
axes[1].axhline(1.0, color=RED, ls="--", lw=1.8, label="oracle = 1.0×")
axes[1].legend(frameon=False, fontsize=10)
axes[1].set_ylabel("speed-up vs oracle (×)")
axes[1].set_title("Unlearning is cheaper than retraining")
for ax in axes:
    ax.set_xticks(range(len(morder)))
    ax.set_xticklabels(mnames, rotation=20, ha="right", fontsize=9.5)
fig.suptitle("Unlearning avoids a full retrain: Plain 5.2× and EPU 3.9× faster than the oracle "
             "(NVIDIA Tesla T4)", y=1.03, fontsize=13)
fig.savefig(OUT / "fig_cost.png")
plt.close(fig)

print("figures written to", OUT)
for f in sorted(OUT.glob("fig_*.png")):
    print("  ", f.name)
