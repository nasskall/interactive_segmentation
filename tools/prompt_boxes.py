"""
Box prompts from expert masks, identical for every model and script.

Kept free of model imports so that tools/imis_benchmark.py (which must not
import Meta's segment_anything) can share it with tools/box_benchmark.py.
"""

from __future__ import annotations

import zlib

import numpy as np


def gt_box(gt: np.ndarray, name: str, jitter: float, seed: int) -> list[int]:
    """Tight box [x0, y0, x1, y1] around ``gt``; with ``jitter`` > 0 each
    side moves by up to that fraction of the box's width or height. The
    shift depends only on (name, seed), so every model sees the same box."""
    ys, xs = np.nonzero(gt)
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    if jitter > 0:
        rng = np.random.default_rng([zlib.crc32(name.encode()), seed])
        w, h = x1 - x0 + 1, y1 - y0 + 1
        dx0, dx1 = rng.uniform(-jitter, jitter, 2) * w
        dy0, dy1 = rng.uniform(-jitter, jitter, 2) * h
        x0, x1, y0, y1 = x0 + dx0, x1 + dx1, y0 + dy0, y1 + dy1
    H, W = gt.shape
    x0, x1 = sorted((min(max(x0, 0), W - 1), min(max(x1, 0), W - 1)))
    y0, y1 = sorted((min(max(y0, 0), H - 1), min(max(y1, 0), H - 1)))
    return [round(x0), round(y0), round(x1), round(y1)]


def box_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return {'iou': inter / union if union else 1.0, 'dice': 2 * inter / total if total else 1.0}
