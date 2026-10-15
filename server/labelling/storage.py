"""Image files for labelling items.

Uploads are kept byte-for-byte under ``<data_dir>/items/<item_id>/`` as
``image``, ``before`` or ``after`` plus the original extension. Formats a
browser cannot display are converted to PNG when served.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

ALLOWED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
BROWSER_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                 '.webp': 'image/webp'}


class ImageError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    """RGB uint8 array, or ImageError."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageError('Could not decode image')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extension(filename: str) -> str:
    ext = os.path.splitext(filename or '')[1].lower()
    if ext not in ALLOWED_EXTS:
        raise ImageError(f'Unsupported image type {ext or "(none)"}; '
                         f'use one of {", ".join(sorted(ALLOWED_EXTS))}')
    return ext


class ImageStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def item_dir(self, item_id: int) -> Path:
        return self.root / str(int(item_id))

    def save(self, item_id: int, role: str, data: bytes, ext: str) -> str:
        d = self.item_dir(item_id)
        d.mkdir(parents=True, exist_ok=True)
        name = f'{role}{ext}'
        (d / name).write_bytes(data)
        return name

    def path(self, item_id: int, name: str) -> Path:
        return self.item_dir(item_id) / name

    def load_rgb(self, item_id: int, name: str) -> np.ndarray:
        return decode(self.path(item_id, name).read_bytes())

    def for_browser(self, item_id: int, name: str) -> tuple[bytes, str]:
        p = self.path(item_id, name)
        media = BROWSER_TYPES.get(p.suffix.lower())
        if media:
            return p.read_bytes(), media
        rgb = self.load_rgb(item_id, name)
        ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return buf.tobytes(), 'image/png'

    def delete(self, item_id: int) -> None:
        shutil.rmtree(self.item_dir(item_id), ignore_errors=True)
