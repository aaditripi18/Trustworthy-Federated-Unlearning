"""Centralized (non-federated) loaders over the full train split.

M2 trains a single central model as the baseline, so it needs the whole train
split rather than the per-client shards from M1. This reuses the M1 synthetic
manifest, patient-level split, transforms, dataset, and loader unchanged.
"""
from __future__ import annotations

from .dataset import (CXRDataset, _manifest_and_loader, build_transform_import,
                      make_loader)
from .synthetic import split_by_patient


def build_central_loaders(cfg: dict) -> dict:
    d = cfg["data"]
    seed = cfg.get("seed", 42)

    manifest, image_loader = _manifest_and_loader(cfg)
    manifest = split_by_patient(manifest, d["val_fraction"], d["test_fraction"], seed)

    splits = {name: manifest[manifest.split == name].copy()
              for name in ("train", "val", "test")}
    t_train = build_transform_import(d["image_size"], train=True)
    t_eval = build_transform_import(d["image_size"], train=False)

    def loader(df, train):
        ds = CXRDataset(df, d["image_size"], t_train if train else t_eval,
                        image_loader=image_loader)
        return make_loader(ds, d["batch_size"], shuffle=train, seed=seed)

    return {
        "train": loader(splits["train"], True),
        "val": loader(splits["val"], False),
        "test": loader(splits["test"], False),
        "n_labels": d["num_labels"],
        "image_size": d["image_size"],
        "image_loader": image_loader,
    }
