"""CustomDataset layouts + behaviour."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from isegm.data.custom_dataset import CustomDataset


def _write_pair(folder: Path, name: str, size=(64, 64), as_layout_b=False):
    img = np.full((*size, 3), 200, dtype=np.uint8)
    mask = np.zeros(size, dtype=np.uint8)
    mask[20:40, 20:40] = 255
    if as_layout_b:
        cv2.imwrite(str(folder / f'{name}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(folder / f'{name}_mask.png'), mask)
    else:
        (folder / 'images').mkdir(exist_ok=True)
        (folder / 'masks').mkdir(exist_ok=True)
        cv2.imwrite(str(folder / 'images' / f'{name}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(folder / 'masks' / f'{name}.png'), mask)


def test_layout_a_discovers_pairs(tmp_path: Path):
    for n in ('a', 'b', 'c'):
        _write_pair(tmp_path, n)
    ds = CustomDataset(tmp_path, augment=False)
    assert len(ds) == 3


def test_layout_b_flat_with_mask_suffix(tmp_path: Path):
    for n in ('a', 'b'):
        _write_pair(tmp_path, n, as_layout_b=True)
    ds = CustomDataset(tmp_path, augment=False)
    assert len(ds) == 2


def test_returns_uint8_image_and_bool_mask(tmp_path: Path):
    _write_pair(tmp_path, 'a')
    ds = CustomDataset(tmp_path, augment=False)
    img, mask = ds[0]
    assert img.dtype == np.uint8
    assert mask.dtype == bool
    assert img.shape[2] == 3


def test_max_size_downscales_oversized_pairs(tmp_path: Path):
    img = np.full((1000, 1500, 3), 180, dtype=np.uint8)
    mask = np.zeros((1000, 1500), dtype=np.uint8)
    (tmp_path / 'images').mkdir()
    (tmp_path / 'masks').mkdir()
    cv2.imwrite(str(tmp_path / 'images' / 'big.png'),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / 'masks' / 'big.png'), mask)

    ds = CustomDataset(tmp_path, augment=False, max_size=512)
    img2, mask2 = ds[0]
    assert max(img2.shape[:2]) == 512
    assert mask2.shape == img2.shape[:2]


def test_synthetic_dataset_loads(synthetic_dataset_dir: Path):
    ds = CustomDataset(synthetic_dataset_dir, augment=False)
    assert len(ds) == 6
    img, mask = ds[0]
    assert img.shape[:2] == mask.shape
    assert mask.any(), "synthetic mask should have foreground pixels"
