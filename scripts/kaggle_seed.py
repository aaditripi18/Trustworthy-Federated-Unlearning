"""Run one additional seed of the setting-A experiment inside Kaggle notebooks (GPU T4 x2).

Two sessions per seed, each using both GPUs:

  TFU_STAGE=base  Original on GPU 0 and Oracle on GPU 1, trained side by side.
                  Inputs: the code bundle and nih-chest-xrays/data.
  TFU_STAGE=A     Plain / EPU on GPU 0 and the gradient-ascent baseline on GPU 1.
                  Inputs: as above plus the output of the base session.

    import glob, os, runpy
    os.environ.update(TFU_SEED="45", TFU_STAGE="base")
    runpy.run_path(glob.glob("/kaggle/input/**/scripts/kaggle_seed.py", recursive=True)[0],
                   run_name="__main__")

Everything is written to /kaggle/working/new_seeds.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_ga as kg  # noqa: E402  (shared setup: code copy, image links, checkpoints)

NAME = "new_seeds"


def run_pair(repo: Path, jobs: list[tuple[str, dict]]) -> int:
    """Start one process per GPU and wait for all of them."""
    procs = []
    for gpu, (script, extra_env) in enumerate(jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **extra_env)
        procs.append(subprocess.Popen([sys.executable, script], cwd=repo, env=env))
    return max(p.wait() for p in procs)


def main() -> int:
    seed = int(os.environ["TFU_SEED"])
    stage = os.environ["TFU_STAGE"]
    out = kg.WORK / NAME
    out.mkdir(parents=True, exist_ok=True)
    common = {"SEED": str(seed), "SEED_OUT": out.as_posix()}

    if stage == "base":
        repo = kg.setup([seed], reuse_name=NAME, need_checkpoints=False)
        if kg.DRY:
            return 0
        rc = run_pair(repo, [("scripts/run_seed.py", dict(common, SEED_STAGE="original")),
                             ("scripts/run_seed.py", dict(common, SEED_STAGE="oracle"))])
    elif stage == "A":
        repo = kg.setup([seed], reuse_name=NAME, need_checkpoints=True)
        if kg.DRY:
            return 0
        # bring the base session's checkpoints and timings into the output folder
        (out / f"seed_{seed}").mkdir(exist_ok=True)
        for name in ("original.pt", "oracle.pt"):
            dst = out / f"seed_{seed}" / name
            if not dst.exists():
                os.symlink(os.path.realpath(repo / "experiments" / NAME / f"seed_{seed}" / name), dst)
        for p in glob.glob(str(kg.INPUT / "**" / f"timing_*_{seed}.json"), recursive=True):
            shutil.copy(p, out)
        from run_seed import write_timing_csv
        write_timing_csv(out)
        cfg = repo / "configs" / "ga.yaml"
        cfg.write_text(cfg.read_text()
                       .replace("out_dir: experiments/ga_baseline",
                                f"out_dir: {(out / 'ga_baseline').as_posix()}")
                       .replace("reuse: experiments/full_experiment", f"reuse: {out.as_posix()}"))
        rc = run_pair(repo, [("scripts/run_seed.py", dict(common, SEED_STAGE="A")),
                             ("scripts/run_ga.py", {"GA_SEEDS": str(seed), "GA_SETTINGS": "A"})])
        for name in ("original.pt", "oracle.pt"):          # links to the input; already saved there
            (out / f"seed_{seed}" / name).unlink(missing_ok=True)
    else:
        sys.exit(f"unknown TFU_STAGE {stage!r}")
    shutil.rmtree(repo / "data", ignore_errors=True)       # links only; keep the output small
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
