"""M1 tests: patient-disjoint splits/partition, coverage, determinism, non-IID,
and the end-to-end datamodule + loader contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_datamodule  # noqa: E402
from src.data.partition import partition_by_patient, partition_stats  # noqa: E402
from src.data.synthetic import generate_manifest, split_by_patient  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CFG = load_config(REPO_ROOT / "configs" / "data.yaml")


def _manifest():
    d = CFG["data"]
    m = generate_manifest(d["num_patients"], tuple(d["imgs_per_patient"]),
                          d["num_labels"], CFG["seed"])
    return split_by_patient(m, d["val_fraction"], d["test_fraction"], CFG["seed"])


def test_splits_are_patient_disjoint():
    m = _manifest()
    per_patient_splits = m.groupby("patient_id")["split"].nunique()
    assert (per_patient_splits == 1).all()  # no patient spans two splits


def test_partition_disjoint_and_covers_all_train():
    m = _manifest()
    train = m[m.split == "train"]
    part = partition_by_patient(train, CFG["data"]["num_clients"],
                                CFG["data"]["label_skew_alpha"], CFG["seed"])
    stats = partition_stats(train, part)
    assert stats["patient_disjoint"] and stats["patients_covered"]
    # every train image assigned exactly once
    assigned = [img for imgs in part["client_to_images"].values() for img in imgs]
    assert len(assigned) == len(set(assigned)) == len(train)


def test_partition_is_deterministic():
    m = _manifest()
    train = m[m.split == "train"]
    a = partition_by_patient(train, 5, 0.5, 7)["client_to_patients"]
    b = partition_by_patient(train, 5, 0.5, 7)["client_to_patients"]
    assert a == b


def test_partition_is_non_iid():
    m = _manifest()
    train = m[m.split == "train"]
    part = partition_by_patient(train, CFG["data"]["num_clients"], 0.3, CFG["seed"])
    stats = partition_stats(train, part)
    dists = [list(s["label_counts"].values()) for s in stats["clients"].values()]
    # at least two clients have different label distributions (skew present)
    assert any(dists[i] != dists[j] for i in range(len(dists)) for j in range(i + 1, len(dists)))


def test_datamodule_loader_contract():
    dm = build_datamodule(CFG)
    assert len(dm["clients"]) == CFG["data"]["num_clients"]
    any_client = next(iter(dm["clients"].values()))
    xs, ys = next(iter(any_client))
    assert xs.ndim == 4 and xs.shape[1] == 1                 # (B,1,H,W)
    assert xs.shape[2] == xs.shape[3] == CFG["data"]["image_size"]
    assert ys.shape[1] == CFG["data"]["num_labels"]
    assert np.isfinite(xs).all()
