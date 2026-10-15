"""Annotation geometry: validation, mask run-length encoding, conversions.

Geometry is stored as JSON in one of these shapes (pixel coordinates of the
item's original image, origin top-left):

    {"type": "tag"}
    {"type": "box", "bbox": [x, y, w, h]}
    {"type": "polygon", "points": [[x, y], ...]}
    {"type": "mask", "rle": {"size": [h, w], "counts": [...]}}

Masks use COCO's uncompressed RLE (column-major runs starting with
background), so COCO export can copy them unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np


class GeometryError(ValueError):
    pass


# ── RLE ───────────────────────────────────────────────────────────────────

def rle_encode(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).flatten(order='F')
    # Run boundaries; the first run is background by convention.
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {'size': [h, w], 'counts': counts}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle['size']
    counts = np.asarray(rle['counts'], dtype=np.int64)
    values = np.arange(counts.size) % 2 == 1
    flat = np.repeat(values, counts)
    return flat.reshape((w, h)).T


# ── conversions ───────────────────────────────────────────────────────────

def mask_to_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]


def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
    """Outline of the largest connected region, lightly simplified."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 1.0, True)
    return [[int(x), int(y)] for x, y in contour[:, 0]]


def mask_to_geometry(mask: np.ndarray, label_type: str) -> dict:
    if not mask.any():
        raise GeometryError('The assist session has no mask to save')
    if label_type == 'mask':
        return {'type': 'mask', 'rle': rle_encode(mask)}
    if label_type == 'box':
        return {'type': 'box', 'bbox': mask_to_box(mask)}
    if label_type == 'polygon':
        points = mask_to_polygon(mask)
        if len(points) < 3:
            raise GeometryError('Mask is too small to outline as a polygon')
        return {'type': 'polygon', 'points': points}
    raise GeometryError(f'A mask cannot be saved as a {label_type!r} label')


# ── validation ────────────────────────────────────────────────────────────

def validate(geometry: dict, label_type: str, width: int, height: int) -> dict:
    """Check ``geometry`` against the class's label type and the image size;
    return it in normalised form."""
    if not isinstance(geometry, dict):
        raise GeometryError('geometry must be an object')
    kind = geometry.get('type')
    if kind != label_type:
        raise GeometryError(f'This class takes {label_type!r} labels, got {kind!r}')

    if kind == 'tag':
        return {'type': 'tag'}

    if kind == 'box':
        bbox = geometry.get('bbox')
        if not (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)):
            raise GeometryError('bbox must be [x, y, width, height]')
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise GeometryError(f'bbox {bbox} is empty or outside the {width}x{height} image')
        return {'type': 'box', 'bbox': [float(v) for v in bbox]}

    if kind == 'polygon':
        points = geometry.get('points')
        if not (isinstance(points, list) and len(points) >= 3 and all(
                isinstance(p, list) and len(p) == 2
                and all(isinstance(v, (int, float)) for v in p) for p in points)):
            raise GeometryError('points must be a list of at least 3 [x, y] pairs')
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
            raise GeometryError(f'polygon has points outside the {width}x{height} image')
        return {'type': 'polygon', 'points': [[float(x), float(y)] for x, y in points]}

    if kind == 'mask':
        rle = geometry.get('rle')
        if not (isinstance(rle, dict) and isinstance(rle.get('counts'), list)
                and all(isinstance(c, int) and c >= 0 for c in rle['counts'])):
            raise GeometryError('rle must be {"size": [h, w], "counts": [...]}')
        if rle.get('size') != [height, width]:
            raise GeometryError(f"rle size {rle.get('size')} does not match the "
                                f'{width}x{height} image')
        if sum(rle['counts']) != width * height:
            raise GeometryError('rle counts do not cover the image')
        if not any(rle['counts'][1::2]):
            raise GeometryError('mask is empty')
        return {'type': 'mask', 'rle': {'size': [height, width], 'counts': rle['counts']}}

    raise GeometryError(f'Unknown geometry type {kind!r}')
