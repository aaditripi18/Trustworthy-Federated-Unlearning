"""Stronger membership-inference evaluation on the saved checkpoints (no retraining).

Members      : training images of the forgotten client (client 0).
Non-members  : images of the held-out test split.
Both are restricted to the first NIH archive (images 00000001_000 .. 00001335_006),
the same restriction for both groups, and members are capped at MAX_MEMBERS.

Three attacks are run against every model of both unlearning settings:

  loss        global loss threshold (the attack used in the original run):
              score = -BCE(x)
  calibrated  difficulty-calibrated loss with the retrained Oracle as reference
              model: score = BCE_oracle(x) - BCE_model(x). The Oracle saw neither
              the members nor the non-members, so it removes the part of the loss
              that is due to how hard an image is (and to the different label mix
              of client 0 and the test split) rather than to membership.
  positive    the calibrated attack restricted to the positive findings of images
              that carry at least one finding.
  within      control for label mix: the positive attack computed inside each
              finding (members and non-members with the same finding) and averaged.

For each attack: AUC and TPR at 5 % FPR with 95 % intervals from a bootstrap that
resamples patients, not images. Members and non-members share no patient.

    .venv/Scripts/python.exe paper/compute_forgetting_attacks.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.nih import NihImageLoader, load_nih_manifest  # noqa: E402
from src.data.partition import partition_by_patient  # noqa: E402
from src.data.synthetic import split_by_patient  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.eval.metrics import binary_auc  # noqa: E402
from src.models.vit import build_model  # noqa: E402

EXP = ROOT / "ec2_final_run/tfu/experiments"
SAMPLE = ROOT / "ec2_final_run/demo_sample"
IMG = SAMPLE / "images"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "forgetting_cache"
OUT = HERE / "forgetting_attacks.csv"
SEEDS = [42, 43, 44]
MAX_MEMBERS, N_BOOT, EPS = 1000, 1000, 1e-7
MODELS = [  # (setting, config, checkpoint relative to EXP/<run>/seed_<s>/)
    ("-", "original", "full_experiment", "original.pt"),
    ("-", "oracle", "full_experiment", "oracle.pt"),
    ("A", "finetune_retain", "full_experiment", "plain.pt"),
    ("A", "epu@0.5", "full_experiment", "epu_lambda0.5.pt"),
    ("A", "epu@1", "full_experiment", "epu_lambda1.0.pt"),
    ("B", "finetune_retain", "exp_a1", "plain.pt"),
    ("B", "epu@1", "exp_a1", "epu_lambda1.pt"),
    ("B", "epu@10", "exp_a1", "epu_lambda10.pt"),
    ("B", "epu@30", "exp_a1", "epu_lambda30.pt"),
    ("B", "epu@100", "exp_a1", "epu_lambda100.pt"),
]


def predict(ckpt, x):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = build_model(dict(cfg["model"], pretrained=False), cfg["data"]["image_size"],
                    cfg["data"]["num_labels"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            out.append(torch.sigmoid(m(x[i:i + 32])).numpy())
    return np.concatenate(out)


def bce(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))          # (N, L), unweighted


def tpr_at_fpr(member, score, fpr=0.05):
    thr = np.quantile(score[member == 0], 1 - fpr)
    return float((score[member == 1] > thr).mean())


def fast_auc(member, score):
    """Rank-based AUC (ties get the average rank)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = member.sum(); n0 = len(member) - n1
    return float((r[member == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def patient_resamples(member, pid, rng, n=N_BOOT):
    """Bootstrap index sets that resample PATIENTS (all images of a drawn patient are
    kept together), separately for members and non-members."""
    groups = {}
    for i, (mb, p) in enumerate(zip(member, pid)):
        groups.setdefault((int(mb), p), []).append(i)
    g1 = [v for (mb, _), v in groups.items() if mb == 1]
    g0 = [v for (mb, _), v in groups.items() if mb == 0]
    out = []
    for _ in range(n):
        pick = [g1[j] for j in rng.integers(0, len(g1), len(g1))] +                [g0[j] for j in rng.integers(0, len(g0), len(g0))]
        out.append(np.fromiter((i for g in pick for i in g), dtype=int))
    return out


def summarise(member, score, boots, keep=None):
    """AUC and TPR at 5 % FPR with patient-level bootstrap 95 % intervals.
    keep: optional boolean mask restricting the attack to a subset of images."""
    k = np.ones(len(member), bool) if keep is None else keep
    auc = fast_auc(member[k], score[k]); tpr = tpr_at_fpr(member[k], score[k])
    ba, bt = [], []
    for b in boots:
        b = b[k[b]]
        if member[b].min() == member[b].max():
            continue
        ba.append(fast_auc(member[b], score[b])); bt.append(tpr_at_fpr(member[b], score[b]))
    lo, hi = np.percentile(ba, [2.5, 97.5]); tlo, thi = np.percentile(bt, [2.5, 97.5])
    return {"auc": auc, "auc_lo": lo, "auc_hi": hi, "tpr_at_5fpr": tpr,
            "tpr_lo": tlo, "tpr_hi": thi}, np.array(ba)


def main():
    CACHE.mkdir(exist_ok=True)
    base = load_nih_manifest(SAMPLE / "Data_Entry_2017_v2020.csv")
    first_archive = set(sorted(base.image_id)[:4999])
    loader = NihImageLoader(IMG, cache=False)
    tf = build_transform(224, train=False)
    rows = []
    for seed in SEEDS:
        m = split_by_patient(base.copy(), 0.15, 0.15, seed)
        train = m[m.split == "train"]
        part = partition_by_patient(train, 5, 0.5, seed, scheme="dirichlet")
        mem = [i for i in part["client_to_images"][0] if i in first_archive]
        non = [i for i in m[m.split == "test"].image_id if i in first_archive]
        rng = np.random.default_rng(seed)
        if len(mem) > MAX_MEMBERS:
            mem = sorted(rng.choice(mem, MAX_MEMBERS, replace=False).tolist())
        ids = mem + non
        missing = [i for i in ids if not (IMG / i).exists()]
        if missing:
            sys.exit(f"seed {seed}: {len(missing)} images missing from {IMG} (first {missing[0]})")
        member = np.r_[np.ones(len(mem)), np.zeros(len(non))]
        idx = m.set_index("image_id")
        lab = [f"label_{i}" for i in range(14)]
        y = idx.loc[ids][lab].to_numpy(dtype=np.float64)
        print(f"seed {seed}: {len(mem)} members, {len(non)} non-members", flush=True)

        x = None
        losses = {}
        for setting, cfgname, run, fname in MODELS:
            cf = CACHE / f"seed{seed}_{run}_{fname}.npy"
            if cf.exists():
                p = np.load(cf)
            else:
                if x is None:
                    x = torch.from_numpy(np.stack([tf(loader(i, 224)) for i in ids])).float()
                t0 = time.time()
                p = predict(EXP / run / f"seed_{seed}" / fname, x)
                np.save(cf, p)
                print(f"  {run}/{fname}: {time.time() - t0:.0f}s", flush=True)
            assert p.shape == y.shape
            losses[(setting, cfgname)] = bce(y, p)

        pid = idx.loc[ids]["patient_id"].to_numpy()
        assert not set(pid[member == 1]) & set(pid[member == 0]), "patient in both groups"
        boots = patient_resamples(member, pid, rng)
        l_or = losses[("-", "oracle")]
        haspos = y.sum(1) > 0
        pos_score = {}
        for (setting, cfgname), l in losses.items():
            row = {"seed": seed, "setting": setting, "config": cfgname,
                   "n_members": len(mem), "n_nonmembers": len(non),
                   "n_member_patients": len(set(pid[member == 1])),
                   "n_nonmember_patients": len(set(pid[member == 0])),
                   "member_loss": l[member == 1].mean(), "nonmember_loss": l[member == 0].mean()}
            r, _ = summarise(member, -l.mean(1), boots)
            row.update({f"loss_{k}": v for k, v in r.items()})
            if cfgname != "oracle":                      # the reference cannot be attacked with itself
                r, _ = summarise(member, l_or.mean(1) - l.mean(1), boots)
                row.update({f"calibrated_{k}": v for k, v in r.items()})
                d = np.zeros(len(member))
                d[haspos] = ((l_or - l) * y).sum(1)[haspos] / y.sum(1)[haspos]
                r, _ = summarise(member, d, boots, keep=haspos)
                row.update({f"positive_{k}": v for k, v in r.items()})
                row.update({"n_members_with_finding": int((haspos & (member == 1)).sum()),
                            "n_nonmembers_with_finding": int((haspos & (member == 0)).sum())})
                pos_score[(setting, cfgname)] = d
                # control for label mix: the same attack inside each finding, i.e. members
                # and non-members that carry the same finding, averaged over findings
                aucs, ws = [], []
                for c in range(y.shape[1]):
                    pos = y[:, c] == 1
                    n1 = int(member[pos].sum()); n0 = int(pos.sum()) - n1
                    if n1 >= 15 and n0 >= 15:
                        aucs.append(fast_auc(member[pos], (l_or[:, c] - l[:, c])[pos]))
                        ws.append(min(n1, n0))
                row.update({"within_finding_auc": float(np.average(aucs, weights=ws)),
                            "n_findings_used": len(aucs), "n_pairs_weight": int(sum(ws))})
            rows.append(row)
        # paired comparison inside the seed: positive-attack AUC of each model minus that of
        # (i) the Original and (ii) Plain of the same setting, on identical resamples
        def paired(a, b):
            da = [fast_auc(member[k], pos_score[a][k]) - fast_auc(member[k], pos_score[b][k])
                  for k in (bb[haspos[bb]] for bb in boots)]
            pt = fast_auc(member[haspos], pos_score[a][haspos]) -                 fast_auc(member[haspos], pos_score[b][haspos])
            lo, hi = np.percentile(da, [2.5, 97.5])
            return pt, lo, hi
        def paired_tpr(a, b):
            """Difference in TPR at 5 % FPR of the positive-finding attack, same resamples."""
            f = lambda k, m: tpr_at_fpr(member[k], pos_score[m][k])
            da = [f(k, a) - f(k, b) for k in (bb[haspos[bb]] for bb in boots)]
            lo, hi = np.percentile(da, [2.5, 97.5])
            return f(haspos, a) - f(haspos, b), lo, hi
        for row in rows[-len(losses):]:
            key = (row["setting"], row["config"])
            if key not in pos_score:
                continue
            if row["config"] != "original":
                row["pos_minus_original"], row["pos_minus_original_lo"],                     row["pos_minus_original_hi"] = paired(key, ("-", "original"))
            if row["config"].startswith("epu"):
                row["pos_minus_plain"], row["pos_minus_plain_lo"], row["pos_minus_plain_hi"] =                     paired(key, (row["setting"], "finetune_retain"))
                row["tpr_minus_plain"], row["tpr_minus_plain_lo"], row["tpr_minus_plain_hi"] =                     paired_tpr(key, (row["setting"], "finetune_retain"))
            print(f"  {row['setting']} {row['config']:16s} loss {row['loss_auc']:.3f} | calibrated "
                  f"{row['calibrated_auc']:.3f} | positive {row['positive_auc']:.3f} "
                  f"[{row['positive_auc_lo']:.3f}, {row['positive_auc_hi']:.3f}] | within "
                  f"{row['within_finding_auc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
