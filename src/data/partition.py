"""Patient-disjoint, non-IID federated partitioning.

Splits the TRAIN patients across N clients so that:
  * every patient belongs to exactly one client (no patient leakage), and
  * label distributions differ across clients (non-IID), controlled by a
    Dirichlet concentration ``alpha`` over each patient's primary label.

Partitioning at the *patient* level (not image level) is the correct unit for
chest-X-ray FL and is what makes participant-level unlearning meaningful later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _primary_label(patient_rows: pd.DataFrame, label_cols: list[str]) -> int:
    """The patient's most frequent positive label (ties -> lowest index)."""
    counts = patient_rows[label_cols].sum().to_numpy()
    return int(counts.argmax())


def partition_by_patient(
    train_manifest: pd.DataFrame, num_clients: int, alpha: float, seed: int,
    scheme: str = "dirichlet",
) -> dict:
    """Return a partition dict:
        {"client_to_patients": {cid: [pid,...]},
         "client_to_images":   {cid: [image_id,...]}}

    scheme : "dirichlet" (label-skewed non-IID, controlled by alpha) or
             "iid" (patients assigned uniformly at random across clients).
    Partitioning is always at the patient level (no patient leakage).
    """
    rng = np.random.default_rng(seed)
    label_cols = [c for c in train_manifest.columns if c.startswith("label_")]
    client_to_patients: dict[int, list[str]] = {c: [] for c in range(num_clients)}

    if scheme == "iid":
        patients = train_manifest["patient_id"].unique().tolist()
        rng.shuffle(patients)
        for cid, chunk in enumerate(np.array_split(patients, num_clients)):
            client_to_patients[cid].extend(list(chunk))
    else:  # dirichlet (non-IID)
        # Group patients by primary label, then Dirichlet-allocate each group to
        # clients. Small alpha => each label concentrates on few clients (skew).
        # Primary label per patient is computed once via a vectorized groupby
        # (O(rows)); iterating per-patient filters would be O(patients*rows) and
        # is impractical at real scale. Patient order (first-appearance) and the
        # argmax/tie semantics are preserved, so partitions are unchanged.
        patients = train_manifest["patient_id"].unique().tolist()
        label_sums = train_manifest.groupby("patient_id")[label_cols].sum()
        prim_map = {pid: int(row.argmax()) for pid, row in
                    zip(label_sums.index, label_sums.to_numpy())}
        by_label: dict[int, list[str]] = {}
        for pid in patients:
            by_label.setdefault(prim_map[pid], []).append(pid)
        for _, plist in sorted(by_label.items()):
            rng.shuffle(plist)
            proportions = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(proportions) * len(plist)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(plist, cuts)):
                client_to_patients[cid].extend(list(chunk))

    # Map patients -> their image_ids
    pat_to_imgs = train_manifest.groupby("patient_id")["image_id"].apply(list).to_dict()
    client_to_images = {
        cid: [img for pid in pats for img in pat_to_imgs[pid]]
        for cid, pats in client_to_patients.items()
    }
    return {"client_to_patients": client_to_patients,
            "client_to_images": client_to_images}


def partition_stats(train_manifest: pd.DataFrame, partition: dict) -> dict:
    """Per-client sample counts and label distributions, plus disjointness check."""
    label_cols = [c for c in train_manifest.columns if c.startswith("label_")]
    idx = train_manifest.set_index("image_id")
    stats = {"clients": {}}
    all_patients = []
    for cid, imgs in partition["client_to_images"].items():
        sub = idx.loc[imgs]
        stats["clients"][cid] = {
            "n_images": len(imgs),
            "n_patients": len(partition["client_to_patients"][cid]),
            "label_counts": {c: int(sub[c].sum()) for c in label_cols},
        }
        all_patients.extend(partition["client_to_patients"][cid])
    stats["patient_disjoint"] = len(all_patients) == len(set(all_patients))
    stats["patients_covered"] = len(set(all_patients)) == train_manifest["patient_id"].nunique()
    return stats
