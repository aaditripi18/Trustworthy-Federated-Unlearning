"""Check every number quoted in the running text of the paper against the run CSVs/logs.

Tables and plots are generated from the same files by build_paper.py and
make_paper_figs.py; this script covers the numbers typed into the prose.

    python paper/verify_numbers.py        # exits non-zero on any mismatch
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

EXP = Path(__file__).resolve().parent.parent / "ec2_final_run/tfu/experiments"
A = pd.read_csv(EXP / "full_experiment" / "final_results.csv")
B = pd.read_csv(EXP / "exp_a1" / "final_results.csv")
LAM = [1, 10, 30, 100]
fails = 0


def m(df, cfg, col):
    return df.loc[df.config == cfg, col].mean()


def sd(df, cfg, col):
    return df.loc[df.config == cfg, col].std(ddof=1)


def diff(df, cfg, col, base="finetune_retain"):
    a = df[df.config == cfg].set_index("seed")[col]
    b = df[df.config == base].set_index("seed")[col]
    return a - b


def check(name, value, quoted, nd):
    """The quoted figure must equal the measured value rounded to nd decimals."""
    global fails
    ok = round(float(value), nd) == round(float(quoted), nd)
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: measured {value:.5f}, paper {quoted}")


def within(name, values, lo, hi):
    global fails
    ok = all(lo <= v <= hi for v in values)
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {[round(float(v), 4) for v in values]} in [{lo}, {hi}]")


# --- baseline
check("Original macro-AUROC", m(A, "original", "macro_auc"), 0.805, 3)
check("Original macro-AUROC s.d.", sd(A, "original", "macro_auc"), 0.007, 3)
check("Original mAP", m(A, "original", "mAP"), 0.206, 3)
check("Original precision", m(A, "original", "precision"), 0.148, 3)
check("Original recall", m(A, "original", "recall"), 0.552, 3)
check("Oracle macro-AUROC", m(A, "oracle", "macro_auc"), 0.802, 3)
check("Oracle - Original (points)", 100 * (m(A, "original", "macro_auc") - m(A, "oracle", "macro_auc")), 0.3, 1)
# --- utility and cost
check("A Plain macro-AUROC", m(A, "finetune_retain", "macro_auc"), 0.806, 3)
check("A EPU macro-AUROC", m(A, "epu@1", "macro_auc"), 0.806, 3)
check("A Plain mAP", m(A, "finetune_retain", "mAP"), 0.212, 3)
check("A Plain hours", m(A, "finetune_retain", "wall_time_s") / 3600, 0.68, 2)
check("Oracle hours", m(A, "oracle", "wall_time_s") / 3600, 3.50, 2)
check("A EPU hours", m(A, "epu@0.5", "wall_time_s") / 3600, 0.91, 2)
check("A Plain speed-up", m(A, "finetune_retain", "speedup_vs_oracle"), 5.2, 1)
check("A EPU speed-up", m(A, "epu@1", "speedup_vs_oracle"), 3.9, 1)
check("B Plain speed-up", m(B, "finetune_retain", "speedup_vs_oracle"), 1.9, 1)
check("B EPU speed-up", m(B, "epu@100", "speedup_vs_oracle"), 1.6, 1)
check("B Plain macro-AUROC", m(B, "finetune_retain", "macro_auc"), 0.789, 3)
within("B EPU macro-AUROC", [m(B, f"epu@{l}", "macro_auc") for l in LAM], 0.7865, 0.7895)
allb = ["finetune_retain"] + [f"epu@{l}" for l in LAM]
within("B below Original (points)", [100 * (m(B, "original", "macro_auc") - m(B, c, "macro_auc")) for c in allb], 1.55, 1.85)
within("B below Oracle (points)", [100 * (m(B, "oracle", "macro_auc") - m(B, c, "macro_auc")) for c in allb], 1.25, 1.55)
within("B recall", [m(B, c, "recall") for c in ["finetune_retain"] + [f"epu@{l}" for l in LAM]], 0.48, 0.495)
within("B paired macro-AUROC diff", [diff(B, f"epu@{l}", "macro_auc").mean() for l in LAM], -0.002, 0.002)
within("EPU extra cost ratio", [m(A, "epu@1", "wall_time_s") / m(A, "finetune_retain", "wall_time_s"),
                                m(B, "epu@100", "wall_time_s") / m(B, "finetune_retain", "wall_time_s")], 1.24, 1.34)
# --- forgetting
check("Original MIA", m(A, "original", "mia_auc"), 0.506, 3)
check("Original MIA s.d.", sd(A, "original", "mia_auc"), 0.029, 3)
check("Oracle MIA", m(A, "oracle", "mia_auc"), 0.499, 3)
check("A Plain MIA", m(A, "finetune_retain", "mia_auc"), 0.505, 3)
within("B MIA", [m(B, c, "mia_auc") for c in ["finetune_retain"] + [f"epu@{l}" for l in LAM]], 0.5015, 0.5035)
within("paired MIA diff", [diff(B, f"epu@{l}", "mia_auc").mean() for l in LAM]
       + [diff(A, c, "mia_auc").mean() for c in ("epu@0.5", "epu@1")], -0.001, 0.001)
check("forget loss Original", m(A, "original", "forget_loss"), 0.354, 3)
check("forget loss Oracle", m(A, "oracle", "forget_loss"), 0.350, 3)
check("forget loss Plain A", m(A, "finetune_retain", "forget_loss"), 0.338, 3)
check("forget loss Plain B", m(B, "finetune_retain", "forget_loss"), 0.317, 3)
check("dist Original", m(A, "original", "pred_dist_to_oracle"), 0.326, 3)
check("dist Plain A", m(A, "finetune_retain", "pred_dist_to_oracle"), 0.417, 3)
check("dist Plain B", m(B, "finetune_retain", "pred_dist_to_oracle"), 0.558, 3)
d = diff(B, "epu@100", "pred_dist_to_oracle")
check("dist paired EPU100", d.mean(), 0.023, 3)
check("dist paired EPU100 s.d.", d.std(ddof=1), 0.011, 3)
check("dist paired p", stats.ttest_rel(B[B.config == "epu@100"].pred_dist_to_oracle,
                                       B[B.config == "finetune_retain"].pred_dist_to_oracle).pvalue, 0.06, 2)
# --- stability
check("Plain cos to Original", m(B, "finetune_retain", "expl_cos_to_original"), 0.976, 3)
check("Plain IoU to Original", m(B, "finetune_retain", "expl_iou_to_original"), 0.603, 3)
check("EPU100 cos", m(B, "epu@100", "expl_cos_to_original"), 0.9987, 4)
check("EPU100 Pearson", m(B, "epu@100", "expl_pearson_to_original"), 0.996, 3)
check("EPU100 IoU", m(B, "epu@100", "expl_iou_to_original"), 0.871, 3)
check("MSE ratio Plain/EPU100", m(B, "finetune_retain", "expl_mse_to_original") / m(B, "epu@100", "expl_mse_to_original"), 15, 0)
check("Oracle cos to Original", m(B, "oracle", "expl_cos_to_original"), 0.961, 3)
check("Oracle IoU to Original", m(B, "oracle", "expl_iou_to_original"), 0.532, 3)
g = {l: diff(B, f"epu@{l}", "expl_cos_to_original") for l in LAM}
sp = sd(B, "finetune_retain", "expl_cos_to_original")
check("Plain cos s.d.", sp, 0.0068, 4)
check("paired cos gain, lambda 1", g[1].mean(), 0.016, 3)
check("paired cos gain, lambda 100", g[100].mean(), 0.023, 3)
check("gain / s.d., lambda 1", g[1].mean() / sp, 2.4, 1)
check("gain / s.d., lambda 100", g[100].mean() / sp, 3.3, 1)
check("smallest gain / s.d.", min(x.min() for x in g.values()) / sp, 1.7, 1)
within("all 12 paired gains positive", [v for x in g.values() for v in x], 1e-9, 1)
check("paired IoU gain, lambda 1", diff(B, "epu@1", "expl_iou_to_original").mean(), 0.133, 3)
check("paired IoU gain, lambda 100", diff(B, "epu@100", "expl_iou_to_original").mean(), 0.268, 3)
pv = [stats.ttest_rel(B[B.config == f"epu@{l}"][c], B[B.config == "finetune_retain"][c]).pvalue
      for l in LAM for c in ("expl_cos_to_original", "expl_iou_to_original")]
within("stability paired p-values", pv, 0.0055, 0.0325)
# --- stability, setting A (computed post hoc; see compute_setting_a_stability.py)
SA_ = pd.read_csv(Path(__file__).resolve().parent / "setting_a_stability.csv")
check("A: 15 validated rows", len(SA_) * int(SA_.matches_log.all()) * int((SA_.n_probe == 96).all()), 15, 0)
check("A Plain cos", m(SA_, "finetune_retain", "expl_cos_to_original"), 0.988, 3)
check("A Plain IoU", m(SA_, "finetune_retain", "expl_iou_to_original"), 0.701, 3)
check("A EPU0.5 IoU", m(SA_, "epu@0.5", "expl_iou_to_original"), 0.798, 3)
check("A EPU1 IoU", m(SA_, "epu@1", "expl_iou_to_original"), 0.820, 3)
check("A EPU0.5 cos", m(SA_, "epu@0.5", "expl_cos_to_original"), 0.996, 3)
check("A EPU1 cos", m(SA_, "epu@1", "expl_cos_to_original"), 0.997, 3)
check("A paired IoU gain 0.5", diff(SA_, "epu@0.5", "expl_iou_to_original").mean(), 0.097, 3)
check("A paired IoU gain 1", diff(SA_, "epu@1", "expl_iou_to_original").mean(), 0.119, 3)
within("A all 6 paired gains positive", [v for c in ("epu@0.5", "epu@1") for col in
       ("expl_cos_to_original", "expl_iou_to_original") for v in diff(SA_, c, col)], 1e-9, 1)
for c, pq in (("epu@0.5", 0.012), ("epu@1", 0.003)):
    check(f"A paired IoU p, {c}", stats.ttest_rel(
        SA_[SA_.config == c].sort_values("seed").expl_iou_to_original,
        SA_[SA_.config == "finetune_retain"].sort_values("seed").expl_iou_to_original).pvalue, pq, 3)
# --- stronger membership attacks (compute_forgetting_attacks.py)
K = pd.read_csv(Path(__file__).resolve().parent / "forgetting_attacks.csv")
ko = K[K.config == "original"].sort_values("seed")
ku = K[~K.config.isin(["original", "oracle"])]
ke = ku[ku.config.str.startswith("epu")]
check("attack: Original positive AUC", ko.positive_auc.mean(), 0.605, 3)
check("attack: Original positive AUC s.d.", ko.positive_auc.std(ddof=1), 0.042, 3)
for v, qd in zip(ko.positive_auc, (0.644, 0.561, 0.608)):
    check("attack: Original positive AUC per seed", v, qd, 3)
for v, qd in zip(list(ko.positive_auc_lo) + list(ko.positive_auc_hi), (0.60, 0.49, 0.56, 0.69, 0.63, 0.66)):
    check("attack: Original interval bound", v, qd, 2)
check("attack: Original calibrated AUC", ko.calibrated_auc.mean(), 0.514, 3)
check("attack: Original TPR at 5% FPR", ko.positive_tpr_at_5fpr.mean(), 0.136, 3)
check("attack: Original same-finding AUC", ko.within_finding_auc.mean(), 0.550, 3)
within("attack: loss AUC on the subset", K.groupby(["setting", "config"]).loss_auc.mean(), 0.49, 0.51)
within("attack: members per seed", ko.n_members, 292, 1000)
within("attack: non-members per seed", ko.n_nonmembers, 601, 820)
within("attack: findings used", ko.n_findings_used, 5, 8)
check("attack: comparisons", len(ku), 24, 0)
check("attack: all below Original", int((ku.pos_minus_original < 0).sum()), 24, 0)
check("attack: decrease excludes zero", int((ku.pos_minus_original_hi < 0).sum()), 8, 0)
ga = ku.groupby(["setting", "config"]).positive_auc.mean()
within("attack: setting A positive AUC", ga["A"], 0.545, 0.575)
within("attack: setting B positive AUC", ga["B"], 0.525, 0.555)
within("attack: unlearned same-finding AUC", ku.groupby(["setting", "config"]).within_finding_auc.mean(), 0.505, 0.535)
gd = ke.groupby(["setting", "config"]).pos_minus_plain.mean()
check("attack: EPU-Plain smallest mean", gd.min(), -0.017, 3)
check("attack: EPU-Plain largest mean", gd.max(), 0.010, 3)
check("attack: EPU-Plain interval includes zero", int(((ke.pos_minus_plain_lo <= 0) & (ke.pos_minus_plain_hi >= 0)).sum()), 15, 0)
check("attack: EPU lower", int((ke.pos_minus_plain_hi < 0).sum()), 2, 0)
check("attack: EPU higher", int((ke.pos_minus_plain_lo > 0).sum()), 1, 0)
check("attack: the one higher case", ke[ke.pos_minus_plain_lo > 0].pos_minus_plain.max(), 0.009, 3)
# --- oracle fidelity
cfgs = [(A, c) for c in ("original", "finetune_retain", "epu@0.5", "epu@1")] + \
       [(B, c) for c in ["finetune_retain"] + [f"epu@{l}" for l in LAM]]
within("cos to Oracle", [m(df, c, "expl_cos_to_oracle") for df, c in cfgs], 0.9575, 0.9645)
within("IoU to Oracle", [m(df, c, "expl_iou_to_oracle") for df, c in cfgs], 0.50, 0.54)
d = diff(B, "epu@100", "expl_iou_to_oracle")
check("IoU-to-Oracle paired EPU100", d.mean(), 0.029, 3)
check("IoU-to-Oracle paired s.d.", d.std(ddof=1), 0.033, 3)
check("IoU-to-Oracle paired p", stats.ttest_rel(B[B.config == "epu@100"].expl_iou_to_oracle,
                                                B[B.config == "finetune_retain"].expl_iou_to_oracle).pvalue, 0.26, 2)
# --- training log: stopping rounds and the forgotten client's share
seed, stop, best, bat = None, {}, {}, {}
for line in open(EXP / "full_experiment" / "run.log", encoding="utf-8", errors="replace"):
    mm = re.search(r"=== seed (\d+) ===", line)
    if mm:
        seed = int(mm.group(1)); bat[seed] = {}
    mm = re.search(r"round 1/30 \| client (\d) \| local update start \(\d+ epochs, (\d+) batches", line)
    if mm:
        bat[seed].setdefault(int(mm.group(1)), int(mm.group(2)))
    mm = re.search(r"\| (original|oracle) \| EARLY STOP \(best val macroAUC=[0-9.]+ @round (\d+)\)", line)
    if mm:
        best[(seed, mm.group(1))] = int(mm.group(2))
within("rounds run (Original, Oracle)", list(A[A.config.isin(["original", "oracle"])].rounds), 8, 12)
within("best round", list(best.values()), 1, 7)
within("forgotten client's share (%)", [100 * b[0] / sum(b.values()) for b in bat.values()], 7, 36.5)

print("\n%d mismatch(es)" % fails)
sys.exit(1 if fails else 0)
