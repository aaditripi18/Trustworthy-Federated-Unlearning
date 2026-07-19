"""Optimization-safety tests for the decode-once image cache and the picklable
transform (perf optimizations that must NOT change results).

These prove:
  * direct-vs-cached bit-identical loads;
  * the cache is actually reused (image decoded once);
  * the cache touches no np.random state, so the augmentation RNG stream and the
    produced (flipped/unflipped) batches are unchanged whether caching is on/off;
  * the transform is picklable (required for num_workers>0 DataLoader workers).

All torch-free: they run in the sandbox against tiny fake PNGs.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_datamodule  # noqa: E402
from src.data.nih import NihImageLoader, clear_image_cache  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402


def _fake_images(tmp: Path, n: int = 6, size: int = 16) -> Path:
    from PIL import Image

    images_dir = tmp / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((size, size)) * 255).astype("uint8")
        Image.fromarray(arr).save(images_dir / f"img_{i:03d}.png")
    return images_dir


def _fake_nih_csv(tmp: Path, images_dir: Path) -> Path:
    rows = []
    pool = ["No Finding", "Cardiomegaly", "Effusion|Infiltration", "Mass"]
    for i, png in enumerate(sorted(p.name for p in images_dir.glob("*.png"))):
        rows.append({"Image Index": png, "Finding Labels": pool[i % len(pool)],
                     "Patient ID": i})
    csv = tmp / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_cached_vs_uncached_bit_identical(tmp_path):
    """The cached loader returns arrays byte-identical to the uncached loader."""
    clear_image_cache()
    images_dir = _fake_images(tmp_path)
    cached = NihImageLoader(images_dir, cache=True)
    uncached = NihImageLoader(images_dir, cache=False)
    for i in range(6):
        name = f"img_{i:03d}.png"
        a = uncached(name, image_size=24)
        b = cached(name, image_size=24)
        c = cached(name, image_size=24)          # second hit -> from cache
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b), f"cache changed pixels for {name}"
        assert np.array_equal(b, c), "repeat cache read differs"


def test_cache_is_actually_used(tmp_path):
    """After first load, deleting the file must still return the cached array
    (proves the second call did not re-decode from disk)."""
    clear_image_cache()
    images_dir = _fake_images(tmp_path, n=1)
    loader = NihImageLoader(images_dir, cache=True)
    first = loader("img_000.png", image_size=20)
    (images_dir / "img_000.png").unlink()        # remove the source file
    second = loader("img_000.png", image_size=20)  # must come from cache
    assert np.array_equal(first, second)


def test_cache_does_not_touch_rng(tmp_path):
    """Loading images (cached or not) consumes no np.random state, so an
    interleaved augmentation RNG stream is identical either way."""
    clear_image_cache()
    images_dir = _fake_images(tmp_path)
    cached = NihImageLoader(images_dir, cache=True)
    uncached = NihImageLoader(images_dir, cache=False)
    t = build_transform(24, train=True)          # train transform uses np.random flip

    def run(loader):
        np.random.seed(123)
        out = []
        for i in range(6):
            img = loader(f"img_{i:03d}.png", image_size=24)
            out.append(t(img))                    # flip decision draws from np.random
        return out

    a = run(uncached)
    b = run(cached)
    for x, y in zip(a, b):
        assert np.array_equal(x, y), "augmentation stream diverged with caching"


def test_transform_is_picklable():
    """num_workers>0 requires the transform to survive pickling."""
    for train in (False, True):
        t = build_transform(24, train)
        t2 = pickle.loads(pickle.dumps(t))
        img = np.linspace(0, 255, 24 * 24).reshape(24, 24).astype(np.float32)
        np.random.seed(7); a = t(img)
        np.random.seed(7); b = t2(img)
        assert np.array_equal(a, b)


def test_datamodule_cache_on_off_identical(tmp_path):
    """End-to-end: an NIH datamodule built with cache on vs off yields
    byte-identical val batches (NumpyLoader, deterministic)."""
    clear_image_cache()
    images_dir = _fake_images(tmp_path, n=12)
    csv = _fake_nih_csv(tmp_path, images_dir)

    def cfg_for(cache):
        return {"seed": 42, "data": {
            "source": "nih", "num_labels": 14, "image_size": 16,
            "val_fraction": 0.15, "test_fraction": 0.15, "num_clients": 2,
            "batch_size": 4, "cache_images": cache,
            "eval_num_workers": 0, "pin_memory": False,
            "nih": {"csv": str(csv), "images_dir": str(images_dir), "subset_size": None},
            "partition": {"scheme": "dirichlet", "alpha": 0.5}}}

    clear_image_cache()
    dm_off = build_datamodule(cfg_for(False))
    off = [b for b in dm_off["val"]]
    clear_image_cache()
    dm_on = build_datamodule(cfg_for(True))
    on = [b for b in dm_on["val"]]

    assert len(off) == len(on)
    for (xo, yo), (xn, yn) in zip(off, on):
        assert np.array_equal(xo, xn) and np.array_equal(yo, yn)
