"""
Realistic simulated clicks (RClicks) for the click benchmarks.

The robot clicker always hits the centre of the largest error. RClicks
(Antonov et al., NeurIPS 2024; https://github.com/emb-ai/rclicks, commit
e20bf76, MIT, Copyright (c) 2024 embodied-ai-star) samples clicks instead
from a clickability map learned from 475K real clicks: positive/negative is
decided as for the robot, the map is restricted to that error and to
unclicked pixels, cut to a band of its probability mass (``band``, e.g.
(0.9, 1.0) keeps the most likely 10% of the mass, (0, 0.1) the least
likely), and a click is drawn from what is
left. ``RClicker._get_next_click`` follows the authors'
isegm/inference/clicker.py; see THIRD_PARTY_NOTICES.md.

The clickability model runs in tools/rclicks_worker.py, in its own venv, so
none of its dependencies enter this one. The model learned from clicks on
natural images, not from clinicians on dermoscopy: here it measures
robustness to realistic click placement, not dermatologists' behaviour.

Setup (once), beside this repository:
    git clone https://github.com/emb-ai/rclicks.git ../external/rclicks
    git -C ../external/rclicks checkout e20bf7636198545331a9d383b77e42107e97ea09
    # clickability_model.pth from the authors' CLICKABILITY_CHECKPOINTS.zip
    # (Google Drive link in their README) into ../external/rclicks/ckpt/
    python -m venv --system-site-packages ../external/rclicks-venv
    ../external/rclicks-venv/bin/pip install mmengine "mmcv-lite>=2.0.0rc4,<2.2.0" ftfy regex albumentations easydict
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from isegm.inference.clicker import Click, Clicker    # noqa: E402
import rclicks_worker as wire                          # noqa: E402  (numpy + torch only at import)


class RClicksClient:
    """Starts the worker and asks it for clickability maps."""

    def __init__(self, python: Path, rclicks_root: Path, checkpoint: Path):
        cmd = [str(python), str(ROOT / 'tools' / 'rclicks_worker.py'),
               '--rclicks-root', str(rclicks_root), '--checkpoint', str(checkpoint)]
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def click_map(self, image: np.ndarray, gt: np.ndarray, error: np.ndarray) -> np.ndarray:
        wire.write_msg(self.proc.stdin, image=image, gt=gt, error=error)
        reply = wire.read_msg(self.proc.stdout)
        if reply is None:
            raise RuntimeError(f'rclicks worker exited with code {self.proc.poll()}')
        return reply['map']

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


class RClicker(Clicker):
    """Clicker whose click positions are sampled from the clickability map.
    Seeded per object from the expert mask, so every model being compared
    gets the same random stream for the same object."""

    def __init__(self, gt_mask, image, client: RClicksClient, band=(0.0, 1.0), seed=0, **kwargs):
        super().__init__(gt_mask=gt_mask, **kwargs)
        self.image, self.client, self.band = image, client, band
        self.rng = np.random.default_rng([zlib.crc32(np.packbits(self.gt_mask).tobytes()), seed])

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)
        fn_dt = cv2.distanceTransform(np.pad(fn_mask, 1).astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
        fp_dt = cv2.distanceTransform(np.pad(fp_mask, 1).astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
        fn_dt, fp_dt = fn_dt * self.not_clicked_map, fp_dt * self.not_clicked_map
        is_positive = fn_dt.max() > fp_dt.max()
        error, dt = (fn_mask, fn_dt) if is_positive else (fp_mask, fp_dt)

        cmap = self.client.click_map(self.image, self.gt_mask, error)
        cmap = cmap * error.astype(np.float32) * self.not_clicked_map
        flat = np.sort(cmap.ravel())
        cum = flat.cumsum()
        lo = cum.searchsorted(flat.sum() * self.band[0], side='left')
        hi = min(cum.searchsorted(flat.sum() * self.band[1], side='right'), len(flat) - 1)
        cmap = cmap * (cmap >= flat[lo]) * (cmap <= flat[hi])
        probs = cmap.astype(np.float64)
        probs /= probs.sum()
        if np.isnan(probs).any():          # empty band: fall back to the robot's click, as the authors do
            ys, xs = np.where(dt == dt.max())
            y, x = ys[0], xs[0]
        else:
            y, x = np.unravel_index(self.rng.choice(probs.size, p=probs.ravel()), probs.shape)
        return Click(is_positive=is_positive, coords=(int(y), int(x)))
