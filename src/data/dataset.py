"""Dataset, loader, and the M1 datamodule builder.

``CXRDataset`` is a plain map-style dataset (``__len__``/``__getitem__``) that is
directly compatible with a future ``torch.utils.data.DataLoader`` *and* with the
lightweight ``NumpyLoader`` used here while torch is unavailable. This keeps the
pipeline runnable at M1 without committing to torch.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .partition import partition_by_patient, partition_stats
from .synthetic import generate_manifest, split_by_patient, synthetic_image


class CXRDataset:
    """Yields (image (1,H,W) float32, label (L,) float32) for a manifest subset."""

    def __init__(self, manifest: pd.DataFrame, image_size: int,
                 transform: Callable[[np.ndarray], np.ndarray],
                 image_loader: Callable[[str, int], np.ndarray] = synthetic_image):
        self.manifest = manifest.reset_index(drop=True)
        self.label_cols = [c for c in manifest.columns if c.startswith("label_")]
        self.image_size = image_size
        self.transform = transform
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        row = self.manifest.iloc[i]
        img = self.image_loader(row["image_id"], self.image_size)
        x = self.transform(img)
        y = row[self.label_cols].to_numpy(dtype=np.float32)
        return x, y


class NumpyLoader:
    """Minimal batch iterator (stand-in for torch DataLoader until M2)."""

    def __init__(self, dataset: CXRDataset, batch_size: int, shuffle: bool, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch = [self.dataset[j] for j in idx[start:start + self.batch_size]]
            xs = np.stack([b[0] for b in batch])
            ys = np.stack([b[1] for b in batch])
            yield xs, ys

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_loader(dataset: CXRDataset, batch_size: int, shuffle: bool, seed: int = 0,
                num_workers: int = 0, pin_memory: bool = False):
    """Return a torch DataLoader if torch is installed, else a NumpyLoader.

    num_workers/pin_memory are perf-only: they change neither batch contents nor
    order. num_workers is used ONLY for eval loaders (shuffle=False, deterministic
    transform); train loaders keep num_workers=0 so the augmentation RNG stream is
    unchanged. pin_memory only affects host->device copy speed.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)
    except ImportError:
        return NumpyLoader(dataset, batch_size, shuffle, seed)


def build_datamodule(cfg: dict) -> dict:
    """Build manifest -> patient-level split -> non-IID client partition -> loaders.

    Returns a dict with the manifest, partition, per-client train loaders,
    global val/test loaders, and partition statistics.
    """
    d = cfg["data"]
    seed = cfg.get("seed", 42)

    manifest, image_loader = _manifest_and_loader(cfg)
    manifest = split_by_patient(manifest, d["val_fraction"], d["test_fraction"], seed)
    train_m = manifest[manifest.split == "train"].copy()
    val_m = manifest[manifest.split == "val"].copy()
    test_m = manifest[manifest.split == "test"].copy()

    pcfg = d.get("partition", {})
    scheme = pcfg.get("scheme", "dirichlet")
    alpha = pcfg.get("alpha", d.get("label_skew_alpha", 0.5))
    partition = partition_by_patient(train_m, d["num_clients"], alpha, seed, scheme=scheme)
    stats = partition_stats(train_m, partition)

    t_train = build_transform_import(d["image_size"], train=True)
    t_eval = build_transform_import(d["image_size"], train=False)
    ev_workers = d.get("eval_num_workers", 0)   # eval-only workers (deterministic)
    pin = d.get("pin_memory", False)

    idx = train_m.set_index("image_id")
    client_loaders = {}
    for cid, imgs in partition["client_to_images"].items():
        sub = idx.loc[imgs].reset_index()
        ds = CXRDataset(sub, d["image_size"], t_train, image_loader=image_loader)
        # train: num_workers=0 to keep the augmentation RNG stream identical
        client_loaders[cid] = make_loader(ds, d["batch_size"], shuffle=True, seed=seed + cid,
                                          num_workers=0, pin_memory=pin)

    val_loader = make_loader(CXRDataset(val_m, d["image_size"], t_eval, image_loader=image_loader),
                             d["batch_size"], shuffle=False, num_workers=ev_workers, pin_memory=pin)
    test_loader = make_loader(CXRDataset(test_m, d["image_size"], t_eval, image_loader=image_loader),
                              d["batch_size"], shuffle=False, num_workers=ev_workers, pin_memory=pin)

    return {"manifest": manifest, "partition": partition, "stats": stats,
            "clients": client_loaders, "val": val_loader, "test": test_loader,
            "n_labels": d["num_labels"], "image_loader": image_loader}


def build_transform_import(image_size: int, train: bool):
    # Local import keeps transforms swappable without a top-level dependency cycle.
    from .transforms import build_transform
    return build_transform(image_size, train)


def _manifest_and_loader(cfg: dict):
    """Return (manifest, image_loader) for the configured data source.

    'synthetic' uses the generated manifest + deterministic synthetic images.
    'nih' reads the NIH ChestX-ray14 manifest and loads real PNGs from disk.
    The rest of the pipeline (split, partition, dataset, loaders) is identical.
    """
    d = cfg["data"]
    seed = cfg.get("seed", 42)
    if d["source"] == "synthetic":
        m = generate_manifest(d["num_patients"], tuple(d["imgs_per_patient"]),
                              d["num_labels"], seed)
        return m, synthetic_image
    if d["source"] == "nih":
        from .nih import NihImageLoader, load_nih_manifest
        nd = d["nih"]
        m = load_nih_manifest(nd["csv"], nd.get("subset_size"), seed)
        return m, NihImageLoader(nd["images_dir"], cache=d.get("cache_images", True))
    raise ValueError(f"unknown data.source '{d['source']}' (expected synthetic|nih)")
