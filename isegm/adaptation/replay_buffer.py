"""
Bounded session replay buffer for online adaptation.

Each entry holds (image, clicks, accepted_mask) where:
  image        : np.uint8 (H, W, 3)        RGB
  clicks       : list[(y, x, is_positive)] in image coordinates
  accepted_mask: np.bool_  (H, W)          user-accepted segmentation
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class BufferEntry:
    image: np.ndarray
    clicks: list  # list of (y, x, is_positive)
    accepted_mask: np.ndarray

    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


class ReplayBuffer:
    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = capacity
        self._items: deque[BufferEntry] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[BufferEntry]:
        return iter(self._items)

    def add(self, image: np.ndarray, clicks: list, mask: np.ndarray) -> None:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        # Defensive copies so later canvas mutations can't corrupt the buffer.
        self._items.append(
            BufferEntry(
                image=image.copy(),
                clicks=list(clicks),
                accepted_mask=mask_bool.copy(),
            )
        )

    def clear(self) -> None:
        self._items.clear()

    def items(self) -> list[BufferEntry]:
        return list(self._items)
