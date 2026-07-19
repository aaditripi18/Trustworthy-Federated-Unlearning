"""M8 tests: NIH manifest parsing, image loading, partition schemes, and the
full datamodule on a *fake* NIH dataset (no torch, no real images)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_datamodule  # noqa: E402
from src.data.nih import NIH_LABELS, NihImageLoader, load_nih_manifest  # noqa: E402
from src.data.partition import partition_by_patient, partition_stats  # noqa: E402
from src.data.synthetic import split_by_patient  # noqa: E402


def _fake_nih(tmp: Path, n_patients: int = 12) -> tuple[Path, Path]:
    """Write a tiny Data_Entry-style CSV and matching fake PNGs."""
    from PIL import Image

    images_dir = tmp / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    findings_pool = ["No Finding", "Cardiomegaly", "Effusion|Infiltration",
                     "Mass", "Nodule|Atelectasis", "Hernia"]
    for p in range(n_patients):
        for k in range(2):
            name = f"{p:08d}_{k:03d}.png"
            Image.fromarray((rng.random((16, 16)) * 255).astype("uint8")).save(images_dir / name)
            rows.append({"Image Index": name,
                         "Finding Labels": findings_pool[(p + k) % len(findings_pool)],
                         "Patient ID": p})
    csv = tmp / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, images_dir


def test_nih_manifest_parsing(tmp_path):
    csv, _ = _fake_nih(tmp_path)
    m = load_nih_manifest(csv)
    assert len(m) == 24 and m["patient_id"].nunique() == 12
    label_cols = [f"label_{i}" for i in range(len(NIH_LABELS))]
    assert all(c in m.columns for c in label_cols)
    # "Cardiomegaly" is index 1; a "No Finding" row is all-zero
    card = m[m.image_id.str.startswith("00000001")]  # patient 1 -> Cardiomegaly/Effusion|Infiltr.
    assert card.iloc[0][label_cols].sum() >= 1


def test_nih_image_loader_shape_and_range(tmp_path):
    _, images_dir = _fake_nih(tmp_path)
    loader = NihImageLoader(images_dir)
    img = loader("00000000_000.png", image_size=24)
    assert img.shape == (24, 24)
    assert img.dtype == np.float32 and 0.0 <= img.min() and img.max() <= 1.0


def test_partition_scheme_iid_vs_dirichlet(tmp_path):
    csv, _ = _fake_nih(tmp_path)
    m = split_by_patient(load_nih_manifest(csv), 0.15, 0.15, 42)
    train = m[m.split == "train"]
    iid = partition_by_patient(train, 3, 0.5, 42, scheme="iid")
    diri = partition_by_patient(train, 3, 0.1, 42, scheme="dirichlet")
    for part in (iid, diri):
        s = partition_stats(train, part)
        assert s["patient_disjoint"] and s["patients_covered"]


def test_build_datamodule_with_nih_source(tmp_path):
    csv, images_dir = _fake_nih(tmp_path)
    cfg = {"seed": 42, "data": {
        "source": "nih", "num_labels": 14, "image_size": 16,
        "val_fraction": 0.15, "test_fraction": 0.15, "num_clients": 2, "batch_size": 4,
        "nih": {"csv": str(csv), "images_dir": str(images_dir), "subset_size": None},
        "partition": {"scheme": "dirichlet", "alpha": 0.5}}}
    dm = build_datamodule(cfg)
    assert dm["n_labels"] == 14 and len(dm["clients"]) == 2
    assert dm["stats"]["patient_disjoint"] and dm["stats"]["patients_covered"]
    xs, ys = next(iter(dm["val"]))            # NumpyLoader (no torch) yields arrays
    assert xs.shape[1:] == (1, 16, 16) and ys.shape[1] == 14
