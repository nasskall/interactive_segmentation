"""
Fine-tuning pipeline for ISModel checkpoints.

Accepts (image, mask) pairs and fine-tunes a loaded model using simulated
RITM-style click sequences.  Designed to run in a background thread; all
progress is reported via callbacks so the UI stays responsive.

Usage
-----
from isegm.training.fine_tune import FineTuner, FineTuneConfig

cfg = FineTuneConfig(epochs=10, lr=5e-5, freeze_backbone=True)
tuner = FineTuner(model, base_checkpoint_path, dataset, cfg,
                  progress_cb=..., log_cb=...)
thread = threading.Thread(target=tuner.run, daemon=True)
thread.start()
# … later …
result = tuner.get_result()   # returns fine-tuned model or None if stopped
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FineTuneConfig:
    def __init__(
        self,
        epochs: int = 10,
        lr: float = 5e-5,
        freeze_backbone: bool = True,
        n_clicks: int = 3,          # click iterations simulated per image per epoch
        n_pos: int = 3,             # positive clicks per iteration
        n_neg: int = 3,             # negative clicks per iteration
        bce_weight: float = 1.0,
        iou_weight: float = 1.0,
    ):
        self.epochs = epochs
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.n_clicks = n_clicks
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight


# ---------------------------------------------------------------------------
# Click simulation
# ---------------------------------------------------------------------------

def _sample_clicks_from_mask(
    gt_mask: np.ndarray,
    prev_pred: Optional[np.ndarray],
    n_pos: int,
    n_neg: int,
) -> tuple[list, list]:
    """
    Sample positive and negative click coordinates.

    On the first iteration (prev_pred is None) samples anywhere inside/outside
    the mask.  On subsequent iterations, prefers the largest error region
    (false negatives for positive clicks, false positives for negative clicks).

    Returns (pos_clicks, neg_clicks) where each element is (row, col).
    """
    gt_bool = gt_mask.astype(bool)

    # --- positive clicks ---
    if prev_pred is not None:
        cand_pos = gt_bool & (prev_pred < 0.5)      # false-negative region
        if cand_pos.sum() == 0:
            cand_pos = gt_bool
    else:
        cand_pos = gt_bool

    pos_clicks = _random_sample(cand_pos, n_pos)

    # --- negative clicks ---
    if prev_pred is not None:
        cand_neg = (~gt_bool) & (prev_pred >= 0.5)  # false-positive region
        if cand_neg.sum() == 0:
            cand_neg = ~gt_bool
    else:
        cand_neg = ~gt_bool

    neg_clicks = _random_sample(cand_neg, n_neg)
    return pos_clicks, neg_clicks


def _random_sample(mask: np.ndarray, n: int) -> list:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    idxs = np.random.choice(len(ys), min(n, len(ys)), replace=False)
    return [(int(ys[i]), int(xs[i])) for i in idxs]


def _build_points_nd(
    pos_clicks: list,
    neg_clicks: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Build points tensor of shape (1, 2*N, 3) expected by ISModel.forward.

    Format matches BasePredictor.get_points_nd:
      - rows 0…N-1  → positive clicks  [row, col, click_index]
      - rows N…2N-1 → negative clicks  [row, col, click_index]
      - padding     → [-1, -1, -1]
    """
    N = max(len(pos_clicks), len(neg_clicks), 1)
    pts = np.full((1, 2 * N, 3), -1.0, dtype=np.float32)
    for i, (r, c) in enumerate(pos_clicks):
        pts[0, i] = [r, c, i]
    for i, (r, c) in enumerate(neg_clicks):
        pts[0, N + i] = [r, c, i]
    return torch.from_numpy(pts).to(device)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _soft_iou_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * gt).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + gt.sum(dim=[2, 3]) - inter
    iou = inter / (union + 1e-6)
    return (1.0 - iou).mean()


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------

class FineTuner:
    """
    Fine-tunes a copy of ``model`` on the supplied dataset.

    Thread-safe: call ``run()`` in a ``threading.Thread``.
    Check ``stop()`` / ``get_result()`` from the main thread.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        base_checkpoint_path: str,
        dataset,                            # isegm.data.custom_dataset.CustomDataset
        config: FineTuneConfig,
        progress_cb: Optional[Callable] = None,   # (epoch, total, loss) → None
        log_cb: Optional[Callable] = None,         # (message: str) → None
    ):
        self.model = model
        self.base_checkpoint_path = base_checkpoint_path
        self.dataset = dataset
        self.config = config
        self.progress_cb = progress_cb
        self.log_cb = log_cb

        self._stop = threading.Event()
        self._result_model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def get_result(self) -> Optional[torch.nn.Module]:
        return self._result_model

    def save_checkpoint(self, output_path: str) -> None:
        """
        Save fine-tuned model to *output_path* in the same format as the
        base checkpoint (config + state_dict).
        """
        if self._result_model is None:
            raise RuntimeError("No result model — training has not completed.")

        base_sd = torch.load(
            self.base_checkpoint_path, map_location='cpu', weights_only=False
        )
        torch.save(
            {
                'config': base_sd['config'],
                'state_dict': self._result_model.state_dict(),
            },
            output_path,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        model = copy.deepcopy(self.model)
        model.train()

        # Freeze / unfreeze parameters
        if cfg.freeze_backbone:
            # Only train maps_transform + prediction head; freeze everything else
            for name, param in model.named_parameters():
                param.requires_grad = any(
                    kw in name for kw in ('maps_transform', 'head', 'cls', 'last_layer',
                                          'aux_head', 'ocr')
                )
        else:
            for param in model.parameters():
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            # Fallback: train everything
            for p in model.parameters():
                p.requires_grad = True
            trainable = list(model.parameters())

        optimizer = Adam(trainable, lr=cfg.lr)
        device = next(model.parameters()).device
        with_prev_mask = getattr(model, 'with_prev_mask', False)

        self._log(f"Fine-tuning started. Dataset: {len(self.dataset)} images, "
                  f"epochs: {cfg.epochs}, LR: {cfg.lr}, "
                  f"freeze_backbone: {cfg.freeze_backbone}")
        self._log(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

        for epoch in range(cfg.epochs):
            if self._stop.is_set():
                self._log("Training stopped by user.")
                break

            epoch_losses = []

            for img_idx in range(len(self.dataset)):
                if self._stop.is_set():
                    break

                try:
                    image_np, gt_mask_np = self.dataset[img_idx]
                except Exception as exc:
                    self._log(f"  [warn] sample {img_idx} failed to load: {exc}")
                    continue

                H, W = gt_mask_np.shape

                # (1, 3, H, W) float in [0, 1]
                image_t = (
                    torch.from_numpy(image_np.transpose(2, 0, 1))
                    .float()
                    .unsqueeze(0)
                    .div(255.0)
                    .to(device)
                )
                gt_t = (
                    torch.from_numpy(gt_mask_np.astype(np.float32))
                    .unsqueeze(0).unsqueeze(0)
                    .to(device)
                )

                prev_pred_np: Optional[np.ndarray] = None
                prev_mask_t = torch.zeros(1, 1, H, W, device=device)

                for _click_iter in range(cfg.n_clicks):
                    pos_clicks, neg_clicks = _sample_clicks_from_mask(
                        gt_mask_np, prev_pred_np, cfg.n_pos, cfg.n_neg
                    )
                    if not pos_clicks:
                        continue

                    points_nd = _build_points_nd(pos_clicks, neg_clicks, device)

                    # Build input tensor
                    if with_prev_mask:
                        inp = torch.cat([image_t, prev_mask_t], dim=1)
                    else:
                        inp = image_t

                    optimizer.zero_grad()
                    outputs = model(inp, points_nd)
                    logits = outputs['instances']           # (1, 1, H, W)

                    bce_loss = F.binary_cross_entropy_with_logits(logits, gt_t)
                    iou_loss = _soft_iou_loss(logits, gt_t)
                    loss = cfg.bce_weight * bce_loss + cfg.iou_weight * iou_loss

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        prob = torch.sigmoid(logits)
                        prev_pred_np = prob.squeeze().cpu().numpy()
                        prev_mask_t = prob.detach()

                    epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self._log(f"Epoch {epoch + 1}/{cfg.epochs}  loss={avg_loss:.4f}")
            if self.progress_cb:
                self.progress_cb(epoch + 1, cfg.epochs, avg_loss)

        model.eval()
        self._result_model = model
        self._log("Training complete.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)
