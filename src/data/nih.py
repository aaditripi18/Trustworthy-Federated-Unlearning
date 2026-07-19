"""NIH ChestX-ray14 dataset adapter (isolated from training).

Parses the official ``Data_Entry_2017.csv`` into the same manifest schema the
pipeline already uses (``image_id``, ``patient_id``, ``label_0``..``label_13``)
and provides a deterministic image loader. Nothing else in the pipeline changes:
splitting, partitioning, datasets, and loaders are reused unchanged.

The 14 findings follow the official fixed order; "No Finding" maps to all-zeros.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
_LABEL_INDEX = {name: i for i, name in enumerate(NIH_LABELS)}


def load_nih_manifest(csv_path, subset_size: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read Data_Entry_2017.csv -> manifest DataFrame.

    subset_size : optional cap on the number of *patients* (kept whole) for faster
                  runs; None uses the full dataset.
    """
    df = pd.read_csv(csv_path)
    img_col = "Image Index"
    lab_col = "Finding Labels"
    pid_col = "Patient ID"

    rows = []
    for _, r in df.iterrows():
        labels = np.zeros(len(NIH_LABELS), dtype=int)
        findings = str(r[lab_col]).split("|")
        for f in findings:
            f = f.strip()
            if f in _LABEL_INDEX:
                labels[_LABEL_INDEX[f]] = 1
            # "No Finding" and unknowns -> leave zeros
        rows.append({"image_id": str(r[img_col]),
                     "patient_id": f"P{int(r[pid_col]):05d}",
                     **{f"label_{i}": int(labels[i]) for i in range(len(NIH_LABELS))}})
    manifest = pd.DataFrame(rows)

    if subset_size is not None:
        rng = np.random.default_rng(seed)
        # np.asarray(...) coerces a possible pandas StringArray from .unique()
        # into a standard ndarray, which rng.shuffle accepts without warning.
        patients = np.asarray(manifest["patient_id"].unique())
        rng.shuffle(patients)
        keep = set(patients[:subset_size])
        manifest = manifest[manifest["patient_id"].isin(keep)].reset_index(drop=True)
    return manifest


# Process-wide decode-once cache: {(dir, image_id, size) -> uint8 (H,W)}. Storing
# the exact convert("L").resize(BILINEAR) result once avoids re-decoding the same
# PNG across FL rounds / methods / seeds. The dir is in the key to avoid
# cross-directory collisions (important for test isolation).
_IMAGE_CACHE: dict = {}


def clear_image_cache() -> None:
    _IMAGE_CACHE.clear()


class NihImageLoader:
    """Deterministic grayscale image loader: path -> (H, W) float32 in [0,1].

    With cache=True, the decoded+resized uint8 image is cached and reused. The
    float conversion (/255.0) still happens on every call, so the returned array
    is bit-identical to the uncached path.
    """

    def __init__(self, images_dir, cache: bool = True):
        self.images_dir = Path(images_dir)
        self.cache = cache

    def __call__(self, image_id: str, image_size: int) -> np.ndarray:
        key = (str(self.images_dir), image_id, image_size)
        arr = _IMAGE_CACHE.get(key) if self.cache else None
        if arr is None:
            from PIL import Image
            with Image.open(self.images_dir / image_id) as im:
                im = im.convert("L").resize((image_size, image_size), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)          # exact decoded pixels
            if self.cache:
                _IMAGE_CACHE[key] = arr
        return arr.astype(np.float32) / 255.0
