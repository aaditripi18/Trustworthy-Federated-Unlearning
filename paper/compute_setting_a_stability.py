"""Explanation stability for setting A, computed from the saved checkpoints.

The setting-A run (full_experiment) logged similarity to the Oracle only. This
script computes the missing similarity to the Original model for Plain and EPU
(lambda = 0.5, 1) and for the Oracle, on the same probe set the training run
used: the first 96 images of each seed's test split. No retraining.

It uses the project's own evaluation code (src.explain.extract / metrics) and
verifies itself: for every model it also recomputes similarity to the Oracle
and compares it with the value logged in final_results.csv. New numbers are
written only if those recomputed values match the log.

    .venv/Scripts/python.exe paper/compute_setting_a_stability.py
    .venv/Scripts/python.exe paper/compute_setting_a_stability.py --smoke   # wiring test only

The 243 probe images listed in paper/probe_images_needed.txt must be present in
ec2_final_run/demo_sample/images/ (or pass --images DIR).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.nih import NihImageLoader, load_nih_manifest  # noqa: E402
from src.data.synthetic import split_by_patient  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.explain.extract import explain_batch, model_grid  # noqa: E402
from src.explain.metrics import compare_maps  # noqa: E402
from src.models.vit import build_model  # noqa: E402

EXP = ROOT / "ec2_final_run/tfu/experiments/full_experiment"
SAMPLE = ROOT / "ec2_final_run/demo_sample"
OUT = Path(__file__).resolve().parent / "setting_a_stability.csv"   # run folder stays untouched
SEEDS = [42, 43, 44]
K_PROBE, IOU_THR = 96, 0.5
# The log was produced on a GPU; this script runs on CPU. Cosine is smooth, so it must agree
# closely. IoU thresholds each map at 0.5, so a float-level difference can flip a borderline
# patch: one flip in one of 96 images moves the mean by up to ~5e-4, hence the wider bound.
TOL_COS, TOL_IOU = 2e-4, 2e-3
MODELS = {"original": "original.pt", "oracle": "oracle.pt", "finetune_retain": "plain.pt",
          "epu@0.5": "epu_lambda0.5.pt", "epu@1": "epu_lambda1.0.pt"}


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = build_model(dict(cfg["model"], pretrained=False), cfg["data"]["image_size"],
                    cfg["data"]["num_labels"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), cfg


def maps_of(model, cfg, x):
    size = cfg["data"]["image_size"]
    grid = model_grid(model, size, cfg["model"]["patch_size"])
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 16):
            out.append(explain_batch(model, x[i:i + 16], grid))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(SAMPLE / "images"))
    ap.add_argument("--smoke", action="store_true",
                    help="run on whatever test-split images are present; never writes results")
    args = ap.parse_args()
    img_dir = Path(args.images)
    base = load_nih_manifest(SAMPLE / "Data_Entry_2017_v2020.csv")
    logged = pd.read_csv(EXP / "final_results.csv")
    # Pre-flight: every checkpoint present, and exactly one logged row per seed/config.
    problems = []
    need_cols = ("seed", "config", "expl_cos_to_oracle", "expl_iou_to_oracle")
    lost = [c for c in need_cols if c not in logged.columns]
    if lost:
        sys.exit("pre-flight failed: final_results.csv has no column(s) " + ", ".join(lost))
    for seed in SEEDS:
        for cfgname, fname in MODELS.items():
            if not (EXP / f"seed_{seed}" / fname).exists():
                problems.append(f"missing checkpoint: seed_{seed}/{fname}")
            n = int(((logged.seed == seed) & (logged.config == cfgname)).sum())
            if n != 1:
                problems.append(f"final_results.csv has {n} rows for seed {seed}, {cfgname} "
                                "(expected 1)")
    print(f"pre-flight: {len(SEEDS) * len(MODELS)} checkpoints and log rows checked, "
          f"{len(problems)} problem(s)")
    if problems:
        sys.exit("\n".join(["pre-flight failed:"] + ["  - " + p for p in problems]))
    loader = NihImageLoader(img_dir, cache=False)
    tf = build_transform(224, train=False)
    rows, all_ok = [], True
    worst_cos = worst_iou = 0.0

    for seed in SEEDS:
        m = split_by_patient(base.copy(), 0.15, 0.15, seed)
        test_ids = list(m[m.split == "test"].image_id)
        probe = test_ids[:K_PROBE]
        missing = [i for i in probe if not (img_dir / i).exists()]
        if args.smoke:
            probe = [i for i in test_ids if (img_dir / i).exists()][:8]
            if not probe:
                print(f"seed {seed}: no test-split image present, skipped"); continue
        elif missing:
            sys.exit(f"seed {seed}: {len(missing)} of {K_PROBE} probe images are missing from "
                     f"{img_dir} (first: {missing[0]}). See paper/probe_images_needed.txt.")
        x = torch.from_numpy(np.stack([tf(loader(i, 224)) for i in probe])).float()
        maps = {}
        for cfgname, fname in MODELS.items():
            model, cfg = load(EXP / f"seed_{seed}" / fname)
            maps[cfgname] = maps_of(model, cfg, x)
        for cfgname in MODELS:
            to_orig = compare_maps(maps[cfgname], maps["original"], IOU_THR)
            to_orac = compare_maps(maps[cfgname], maps["oracle"], IOU_THR)
            row = {"seed": seed, "config": cfgname, "n_probe": len(probe),
                   "expl_cos_to_original": to_orig["cosine_similarity"],
                   "expl_pearson_to_original": to_orig["pearson"],
                   "expl_iou_to_original": to_orig["iou"],
                   "expl_mse_to_original": to_orig["mse"],
                   "recomputed_cos_to_oracle": to_orac["cosine_similarity"],
                   "recomputed_iou_to_oracle": to_orac["iou"]}
            ref = logged[(logged.seed == seed) & (logged.config == cfgname)].iloc[0]
            d_cos = abs(row["recomputed_cos_to_oracle"] - ref.expl_cos_to_oracle)
            d_iou = abs(row["recomputed_iou_to_oracle"] - ref.expl_iou_to_oracle)
            ok = d_cos < TOL_COS and d_iou < TOL_IOU
            worst_cos, worst_iou = max(worst_cos, d_cos), max(worst_iou, d_iou)
            all_ok &= ok
            row["matches_log"] = ok
            rows.append(row)
            print(f"seed {seed} {cfgname:16s} cos->orig {row['expl_cos_to_original']:.4f} "
                  f"IoU->orig {row['expl_iou_to_original']:.3f} | cos->oracle "
                  f"{row['recomputed_cos_to_oracle']:.4f} (log {ref.expl_cos_to_oracle:.4f}) "
                  f"{'OK' if ok else 'MISMATCH'}")

    print(f"\nlargest deviation from the log: cosine {worst_cos:.2e} (limit {TOL_COS:.0e}), "
          f"IoU {worst_iou:.2e} (limit {TOL_IOU:.0e})")
    if args.smoke:
        print("\nsmoke run finished (wiring only; values are not the paper's metric).")
        return
    if not all_ok:
        sys.exit("\nRecomputed oracle similarities do not match final_results.csv; nothing "
                 "written. Check that the images are the original NIH PNGs.")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\nall recomputed values match the log; wrote", OUT)
    print(df.groupby("config")[["expl_cos_to_original", "expl_iou_to_original"]]
            .agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
