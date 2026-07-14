"""
CustomDataset — loads (image, mask) pairs for fine-tuning.

Supported folder layouts
------------------------
Layout A — two sub-folders:
  root/
    images/   *.jpg *.jpeg *.png *.bmp *.tiff
    masks/    *.png *.bmp  (same stem as images)

Layout B — flat folder with matching stems:
  root/
    foo.jpg   foo_mask.png   (or foo.png)
    bar.png   bar_mask.png

The dataset returns (image_np, gt_mask_np) pairs where
  image_np  : uint8 (H, W, 3) RGB
  gt_mask_np: bool  (H, W)    True = foreground
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class CustomDataset:
    """
    Parameters
    ----------
    root : str | Path
        Folder containing images (and masks).
    augment : bool
        If True, applies random horizontal flip + mild colour jitter.
    max_size : int | None
        If given, images larger than this on their longest side are
        downscaled (aspect-ratio preserving).
    """

    def __init__(
        self,
        root: str | Path,
        augment: bool = True,
        max_size: Optional[int] = 1024,
    ):
        self.root = Path(root)
        self.augment = augment
        self.max_size = max_size
        self.pairs = self._discover_pairs()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        img_dir = self.root / 'images'
        msk_dir = self.root / 'masks'

        if img_dir.is_dir() and msk_dir.is_dir():
            return self._layout_a(img_dir, msk_dir)
        return self._layout_b(self.root)

    def _layout_a(self, img_dir: Path, msk_dir: Path) -> list:
        pairs = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            # Try exact stem match, then stem + '_mask'
            for msk_name in (
                img_path.stem + '.png',
                img_path.stem + '.bmp',
                img_path.stem + img_path.suffix,
            ):
                msk_path = msk_dir / msk_name
                if msk_path.exists():
                    pairs.append((img_path, msk_path))
                    break
        return pairs

    def _layout_b(self, folder: Path) -> list:
        pairs = []
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = img_path.stem
            if stem.endswith('_mask'):
                continue
            # Look for <stem>_mask.<ext> or <stem>_mask.png
            for msk_name in (
                stem + '_mask.png',
                stem + '_mask.bmp',
                stem + '_mask' + img_path.suffix,
            ):
                msk_path = folder / msk_name
                if msk_path.exists():
                    pairs.append((img_path, msk_path))
                    break
        return pairs

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, msk_path = self.pairs[idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise IOError(f"Cannot read image: {img_path}")
        if mask_raw is None:
            raise IOError(f"Cannot read mask: {msk_path}")

        # Resize mask to match image if shapes differ
        if mask_raw.shape != image.shape[:2]:
            mask_raw = cv2.resize(
                mask_raw, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Downscale if needed
        if self.max_size is not None:
            h, w = image.shape[:2]
            longest = max(h, w)
            if longest > self.max_size:
                scale = self.max_size / longest
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_raw = cv2.resize(mask_raw, (new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST)

        gt_mask = mask_raw > 127  # bool

        if self.augment:
            image, gt_mask = self._augment(image, gt_mask)

        return image, gt_mask

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(
        image: np.ndarray,
        gt_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            gt_mask = np.fliplr(gt_mask).copy()

        # Mild colour jitter (brightness + contrast)
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta = random.randint(-20, 20)      # brightness
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return image, gt_mask

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return f"CustomDataset: {len(self.pairs)} pairs in '{self.root}'"

    @staticmethod
    def add_pair(
        root: str | Path,
        image: np.ndarray,
        mask: np.ndarray,
        stem: str,
    ) -> None:
        """
        Save a single (image, mask) pair into *root*/images/ and *root*/masks/.

        Parameters
        ----------
        image : uint8 (H, W, 3) RGB
        mask  : uint8 or bool (H, W)  — non-zero = foreground
        stem  : filename stem (no extension)
        """
        root = Path(root)
        (root / 'images').mkdir(parents=True, exist_ok=True)
        (root / 'masks').mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(root / 'images' / f'{stem}.png'), img_bgr)

        msk = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(root / 'masks' / f'{stem}.png'), msk)
