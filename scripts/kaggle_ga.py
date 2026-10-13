"""Run the gradient-ascent baseline inside a Kaggle notebook (GPU T4).

Inputs the notebook must have attached (Add Input):
  1. the public dataset "NIH Chest X-rays" (nih-chest-xrays/data);
  2. your own private dataset made from tfu_kaggle_bundle.zip, which holds this
     repository's code, the label CSV used for the paper, and the Original /
     Oracle checkpoints of the three seeds.

In a notebook cell:

    import glob, runpy
    runpy.run_path(glob.glob("/kaggle/input/**/scripts/kaggle_ga.py", recursive=True)[0],
                   run_name="__main__")

Everything is written to /kaggle/working/ga_baseline (results CSV, log,
checkpoints), which Kaggle keeps as the notebook's output.

Environment variables (optional): GA_SETTINGS=A or A,B ; GA_SEEDS=42 or 42,43,44.
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working"))
DRY = bool(os.environ.get("KAGGLE_GA_DRYRUN"))     # local test: code copy + partition check only
# client sizes (batches of 32) printed by the original training run, per seed
NIH_NAME = re.compile(r"\d{8}_\d{3}\.png")
EXPECTED_BATCHES = {42: [889, 130, 659, 268, 502], 43: [182, 195, 276, 402, 1391],
                    44: [375, 456, 273, 366, 995]}


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    if not hits:
        sys.exit(f"not found under /kaggle/input: {pattern}")
    return Path(hits[0])


def main() -> int:
    # ---- 1. code: copy the bundled repository to a writable place
    src_repo = find_one("scripts/run_ga.py").parents[1]
    repo = WORK / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(src_repo, repo, ignore=shutil.ignore_patterns("*.pt", "*.png"))
    print("code:", src_repo, "->", repo, flush=True)

    # ---- 2. data: label CSV of the paper + one flat folder of links to the images
    nih = repo / "data" / "nih"
    (nih / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy(find_one("Data_Entry_2017_v2020.csv"), nih / "Data_Entry_2017_v2020.csv")
    # Any folder layout is accepted: every file named like 00000001_000.png is an NIH image.
    pngs = {}
    if not DRY:
        for root, _, files in os.walk(INPUT):
            for f in files:
                if NIH_NAME.fullmatch(f):
                    pngs.setdefault(f, os.path.join(root, f))
        pngs = list(pngs.values())
    if not DRY and len(pngs) < 112120:
        sys.exit(f"expected 112,120 NIH images under /kaggle/input, found {len(pngs)}. "
                 "Attach the dataset nih-chest-xrays/data.")
    for p in pngs:
        dst = nih / "images" / os.path.basename(p)
        if not dst.exists():
            os.symlink(p, dst)
    print("images linked:", len(os.listdir(nih / "images")), flush=True)

    # ---- 3. checkpoints of the main run (Original and Oracle are reused, not retrained)
    reuse = repo / "experiments" / "full_experiment"
    seeds = [int(s) for s in os.environ.get("GA_SEEDS", "42,43,44").split(",")]
    for seed in seeds:
        d = reuse / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        for name in ("original.pt", "oracle.pt"):
            src = find_one(f"seed_{seed}/{name}")           # fails early if a checkpoint is missing
            if not DRY and not (d / name).exists():
                os.symlink(src, d / name)
    shutil.copy(find_one("full_experiment_final_results.csv"), reuse / "final_results.csv")

    # ---- 4. safety check: the client partition must be the one the checkpoints were trained on
    sys.path.insert(0, str(repo))
    from src.data.nih import load_nih_manifest
    from src.data.partition import partition_by_patient
    from src.data.synthetic import split_by_patient
    base = load_nih_manifest(nih / "Data_Entry_2017_v2020.csv")
    for seed in seeds:
        m = split_by_patient(base.copy(), 0.15, 0.15, seed)
        part = partition_by_patient(m[m.split == "train"], 5, 0.5, seed, scheme="dirichlet")
        got = [math.ceil(len(part["client_to_images"][c]) / 32) for c in range(5)]
        if got != EXPECTED_BATCHES[seed]:
            sys.exit(f"seed {seed}: client sizes {got} differ from the original run "
                     f"{EXPECTED_BATCHES[seed]}; refusing to continue.")
    print("partition check passed for seeds", seeds, flush=True)

    if DRY:
        print("dry run finished: bundle is complete and the partition matches")
        return 0
    # ---- 5. run
    out = WORK / "ga_baseline"
    env = dict(os.environ, GA_SEEDS=",".join(map(str, seeds)))
    cfg = repo / "configs" / "ga.yaml"
    cfg.write_text(cfg.read_text().replace("out_dir: experiments/ga_baseline",
                                           f"out_dir: {out.as_posix()}"))
    rc = subprocess.call([sys.executable, "scripts/run_ga.py"], cwd=repo, env=env)
    shutil.rmtree(repo / "data", ignore_errors=True)       # links only; keep the output small
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
