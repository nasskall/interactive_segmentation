"""
Shared fixtures for the pytest suite.

Heavy resources (the RITM checkpoint, synthetic dataset) are session-scoped
so the suite finishes in a sensible time on CPU. SAM/SAM2 fixtures only
materialise when both the package and a checkpoint are available — tests
that depend on them are otherwise auto-skipped via pytest.mark fixtures.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope='session')
def device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_lesion_pair(rng: np.random.Generator, size=(256, 256), seed_offset=0):
    """Skin-tone background + elliptical 'lesion' with soft edge."""
    H, W = size
    base = (
        (int(rng.integers(180, 230) + seed_offset) % 256),
        (int(rng.integers(140, 190) + seed_offset) % 256),
        (int(rng.integers(130, 170) + seed_offset) % 256),
    )
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = base[0] + (yy / H) * 18
    img[..., 1] = base[1] + (xx / W) * 12
    img[..., 2] = base[2] + ((yy + xx) / (H + W)) * 8
    img += rng.normal(0.0, 6.0, size=img.shape)

    cy = int(rng.integers(H // 3, 2 * H // 3))
    cx = int(rng.integers(W // 3, 2 * W // 3))
    a, b = int(rng.integers(28, 55)), int(rng.integers(28, 55))
    theta = float(rng.uniform(0, np.pi))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rel_y, rel_x = yy - cy, xx - cx
    xr = cos_t * rel_x + sin_t * rel_y
    yr = -sin_t * rel_x + cos_t * rel_y
    dist = (xr / a) ** 2 + (yr / b) ** 2
    mask = dist < 1.0

    img -= np.exp(-dist)[..., None] * 70.0
    return np.clip(img, 0, 255).astype(np.uint8), mask


@pytest.fixture(scope='session')
def synthetic_pair():
    rng = np.random.default_rng(42)
    return _make_lesion_pair(rng, size=(192, 192))


@pytest.fixture(scope='session')
def synthetic_dataset_dir(tmp_path_factory) -> Path:
    """6-pair synthetic skin-lesion dataset under a tmp dir."""
    out = tmp_path_factory.mktemp('synth_ds')
    (out / 'images').mkdir()
    (out / 'masks').mkdir()
    rng = np.random.default_rng(7)
    for i in range(6):
        img, mask = _make_lesion_pair(rng, size=(192, 192), seed_offset=i * 31)
        cv2.imwrite(str(out / 'images' / f'lesion_{i:02d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / 'masks' / f'lesion_{i:02d}.png'),
                    (mask.astype(np.uint8) * 255))
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def ritm_checkpoint_path(repo_root: Path) -> Path:
    p = repo_root / 'model_weights' / 'best_checkpoint_068.pth'
    if not p.exists():
        pytest.skip(f'RITM checkpoint missing: {p}')
    return p


@pytest.fixture(scope='session')
def ritm_model(ritm_checkpoint_path: Path, device: str):
    """Loaded once per session — copying happens in tests that need a fresh
    model (so adapter injection in one test doesn't bleed into another)."""
    from isegm.inference import utils as iu
    model = iu.load_is_model(str(ritm_checkpoint_path), device=device,
                              cpu_dist_maps=True)
    model._model_type = 'ritm'
    return model


def _fresh_ritm(ckpt: Path, device: str):
    from isegm.inference import utils as iu
    m = iu.load_is_model(str(ckpt), device=device, cpu_dist_maps=True)
    m._model_type = 'ritm'
    return m


@pytest.fixture
def fresh_ritm(ritm_checkpoint_path: Path, device: str):
    return _fresh_ritm(ritm_checkpoint_path, device)


# ---------------------------------------------------------------------------
# Optional SAM / SAM2 models
# ---------------------------------------------------------------------------

def _find_first(repo_root: Path, patterns):
    weights = repo_root / 'model_weights'
    for pat in patterns:
        hits = sorted(weights.glob(pat))
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope='session')
def sam_model(repo_root: Path, device: str):
    pytest.importorskip('segment_anything')
    ckpt = _find_first(repo_root, ['sam_vit_*.pth'])
    if ckpt is None:
        pytest.skip('no sam_vit_*.pth in model_weights/')
    from isegm.inference import utils as iu
    label = ('SAM - ViT-B' if 'vit_b' in ckpt.name
             else 'SAM - ViT-L' if 'vit_l' in ckpt.name
             else 'SAM - ViT-H')
    return iu.load_sam_model(str(ckpt), label, device), ckpt, label


@pytest.fixture(scope='session')
def sam2_model(repo_root: Path, device: str):
    pytest.importorskip('sam2.build_sam')
    ckpt = _find_first(repo_root, [
        'sam2*hiera*tiny*.pt', 'sam2*hiera*small*.pt',
        'sam2*hiera*base*.pt', 'sam2*hiera*large*.pt',
    ])
    if ckpt is None:
        pytest.skip('no sam2*hiera*.pt in model_weights/')
    from isegm.inference import utils as iu
    name = ckpt.name.lower()
    label_map = {'tiny': 'SAM2 - Tiny', 'small': 'SAM2 - Small',
                 'base': 'SAM2 - Base+', 'large': 'SAM2 - Large'}
    label = next((v for k, v in label_map.items() if k in name), None)
    if label is None:
        pytest.skip(f'unknown SAM2 variant: {ckpt.name}')
    return iu.load_sam2_model(str(ckpt), label, device), ckpt, label


# ---------------------------------------------------------------------------
# Tk root (one per session, withdrawn so no window pops up during tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'no Tk display: {exc}')
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass
