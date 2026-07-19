"""M8 tests: reproducibility capture, figure generation, and image-loader
propagation (compatibility) — all torch-free."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.experiment import results_dataframe  # noqa: E402
from src.utils.repro import capture_environment, write_reproducibility  # noqa: E402


# ---------------- reproducibility ----------------

def test_capture_environment_has_required_fields():
    env = capture_environment(seeds=[42, 43], repo_root=REPO_ROOT)
    for key in ("timestamp_utc", "python", "platform", "hostname", "cpu_count",
                "packages", "cuda", "seeds"):
        assert key in env
    assert env["seeds"] == [42, 43]
    assert "numpy" in env["packages"]


def test_write_reproducibility_creates_file(tmp_path):
    p = tmp_path / "reproducibility.json"
    write_reproducibility(p, seeds=[1], repo_root=REPO_ROOT)
    assert p.exists()
    import json
    assert "packages" in json.load(open(p))


# ---------------- figures ----------------

def _synthetic_results():
    rows = []
    for s in (42, 43):
        for method, lam, cos in [("original", None, 0.3), ("finetune_retain", None, 0.5),
                                 ("oracle", None, 1.0), ("epu", 0.1, 0.6), ("epu", 1.0, 0.4)]:
            rows.append({
                "seed": s, "method": method, "lambda": lam,
                "config": method if method != "epu" else f"epu@{lam:g}",
                "macro_auc": 0.6, "mAP": 0.5, "macro_f1": 0.0, "micro_f1": 0.0,
                "forget_loss": 0.5, "mia_auc": 0.55, "pred_dist_to_oracle": 1.0 - cos,
                "expl_cos_to_oracle": cos, "expl_pearson_to_oracle": cos,
                "expl_iou_to_oracle": cos, "expl_mse_to_oracle": 1.0 - cos,
                "wall_time_s": 2.0, "rounds": 2, "speedup_vs_oracle": 3.0})
    return rows


def test_generate_all_figures_writes_png_and_pdf(tmp_path):
    from src.eval.figures import generate_all_figures  # imports matplotlib
    df = results_dataframe(_synthetic_results())
    names = generate_all_figures(df, tmp_path)
    assert set(names) == {"fig_lambda_ablation", "fig_tradeoff",
                          "fig_oracle_comparison", "fig_runtime"}
    for n in names:
        assert (tmp_path / f"{n}.png").exists()
        assert (tmp_path / f"{n}.pdf").exists()


# ---------------- image-loader propagation (compatibility) ----------------

def test_datamodule_and_forget_loader_use_same_image_loader(tmp_path):
    from PIL import Image
    import pandas as pd

    from src.data.dataset import build_datamodule
    from src.unlearning.verify import _forget_eval_loader

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for p in range(10):
        for k in range(2):
            name = f"{p:08d}_{k:03d}.png"
            Image.fromarray((rng.random((16, 16)) * 255).astype("uint8")).save(images_dir / name)
            rows.append({"Image Index": name, "Finding Labels": "Nodule", "Patient ID": p})
    csv = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    cfg = {"seed": 42, "data": {
        "source": "nih", "num_labels": 14, "image_size": 16,
        "val_fraction": 0.2, "test_fraction": 0.2, "num_clients": 2, "batch_size": 4,
        "nih": {"csv": str(csv), "images_dir": str(images_dir), "subset_size": None},
        "partition": {"scheme": "iid", "alpha": 0.5}},
        "unlearn": {"forget_client": 0}}
    dm = build_datamodule(cfg)
    assert "image_loader" in dm
    fl = _forget_eval_loader(cfg, dm, 0)
    # the forget loader must use the NIH loader (not the synthetic default)
    assert fl.dataset.image_loader is dm["image_loader"]
