"""Image transforms.

Kept NumPy-based for M1 so the pipeline runs without torch. Output is a
channel-first, normalized float32 array of shape (1, H, W). When torchvision is
adopted (M2), these can be replaced by ``torchvision.transforms`` returning
tensors of the same shape; the dataset/loader contract is unchanged.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Simple fixed normalization for the single-channel (grayscale) CXR setup.
_MEAN, _STD = 0.5, 0.25


class _Transform:
    """Picklable transform (was a closure) so it survives DataLoader workers.
    Identical computation and identical np.random usage as before."""

    def __init__(self, image_size: int, train: bool):
        self.image_size = image_size
        self.train = train

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        if x.shape != (self.image_size, self.image_size):  # defensive; synthetic matches
            x = x[:self.image_size, :self.image_size]
        if self.train and np.random.rand() < 0.5:          # light aug: horizontal flip
            x = x[:, ::-1].copy()
        x = (x - _MEAN) / _STD
        return x[None, :, :]                                # (1, H, W)


def build_transform(image_size: int, train: bool) -> Callable[[np.ndarray], np.ndarray]:
    return _Transform(image_size, train)
