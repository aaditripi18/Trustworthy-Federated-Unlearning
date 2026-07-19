"""Synthetic, CXR-shaped dataset for fast pipeline bring-up.

Produces a manifest with the same *structure* as a real chest-X-ray dataset —
per-image rows, a ``patient_id`` grouping (multiple images per patient), and
multi-label pathology targets — so every downstream component (partitioning,
splits, loaders) is exercised without needing the real data or torch.

Images are generated deterministically on demand from ``image_id`` (see
``synthetic_image``), so no files are stored. Swap this module for a real NIH
ChestX-ray14 loader later without changing the partition/split/loader code.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def generate_manifest(
    num_patients: int,
    imgs_per_patient: tuple[int, int],
    num_labels: int,
    seed: int,
) -> pd.DataFrame:
    """Return a manifest DataFrame: image_id, patient_id, label_0..label_{L-1}.

    Each patient has a bias toward one 'primary' pathology so that patient-level
    partitioning produces meaningful label skew (non-IID) across clients.
    """
    rng = np.random.default_rng(seed)
    lo, hi = imgs_per_patient
    rows = []
    for p in range(num_patients):
        pid = f"P{p:04d}"
        primary = int(rng.integers(0, num_labels))
        n_imgs = int(rng.integers(lo, hi + 1))
        for k in range(n_imgs):
            labels = (rng.random(num_labels) < 0.15).astype(int)  # sparse co-occurrence
            labels[primary] = 1  # patient's characteristic finding
            rows.append({"image_id": f"{pid}_{k}", "patient_id": pid,
                         **{f"label_{i}": int(labels[i]) for i in range(num_labels)}})
    return pd.DataFrame(rows)


def split_by_patient(
    manifest: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> pd.DataFrame:
    """Add a 'split' column (train/val/test) assigned at the PATIENT level, so
    no patient's images appear in more than one split."""
    rng = np.random.default_rng(seed)
    # np.asarray(...) coerces a possible pandas StringArray from .unique() into a
    # standard object ndarray, which rng.shuffle accepts without warning.
    patients = np.asarray(manifest["patient_id"].unique())
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])

    def which(pid: str) -> str:
        if pid in test_p:
            return "test"
        if pid in val_p:
            return "val"
        return "train"

    out = manifest.copy()
    out["split"] = out["patient_id"].map(which)
    return out


def synthetic_image(image_id: str, image_size: int) -> np.ndarray:
    """Deterministic HxW grayscale image derived from image_id (float32 in [0,1])."""
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    return rng.random((image_size, image_size), dtype=np.float32)
