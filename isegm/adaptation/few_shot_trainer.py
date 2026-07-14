"""
Offline few-shot adapter trainer (Mode A).

Wraps a base model with LoRA, runs a small training loop on a labeled
``(image, mask)`` dataset, and writes adapter-only weights to disk.

Auto-detects CUDA vs CPU and adjusts batch size / image resize accordingly
so a CPU-only machine still finishes in a sensible amount of time.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .lora import (
    collect_lora_state,
    lora_parameters,
    set_lora_trainable,
)
from .presets import attach_adapter, DEFAULT_RANKS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FewShotConfig:
    rank: int | None = None        # None → uses DEFAULT_RANKS[model_type]
    epochs: int = 20
    lr: float = 1e-3
    n_clicks: int = 3              # iterative click rounds per image per epoch
    n_pos: int = 3
    n_neg: int = 3
    bce_weight: float = 1.0
    iou_weight: float = 1.0
    cpu_max_size: int = 512        # downscale images on CPU to keep time sane
    cuda_max_size: int = 1024


# ---------------------------------------------------------------------------
# Click sampling (shared across model families)
# ---------------------------------------------------------------------------

def _sample_click_pixels(
    gt_mask: np.ndarray,
    prev_pred: Optional[np.ndarray],
    n_pos: int,
    n_neg: int,
) -> tuple[list, list]:
    """Return (pos_yx, neg_yx); samples from largest error region after iter 1."""
    gt_bool = gt_mask.astype(bool)

    if prev_pred is not None:
        cand_pos = gt_bool & (prev_pred < 0.5)
        if cand_pos.sum() == 0:
            cand_pos = gt_bool
        cand_neg = (~gt_bool) & (prev_pred >= 0.5)
        if cand_neg.sum() == 0:
            cand_neg = ~gt_bool
    else:
        cand_pos = gt_bool
        cand_neg = ~gt_bool

    return _rand_pixels(cand_pos, n_pos), _rand_pixels(cand_neg, n_neg)


def _rand_pixels(mask: np.ndarray, n: int) -> list:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    idx = np.random.choice(len(ys), min(n, len(ys)), replace=False)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def _build_ritm_points_nd(pos: list, neg: list, device) -> torch.Tensor:
    """Format expected by ISModel.forward — see fine_tune._build_points_nd."""
    N = max(len(pos), len(neg), 1)
    pts = np.full((1, 2 * N, 3), -1.0, dtype=np.float32)
    for i, (r, c) in enumerate(pos):
        pts[0, i] = [r, c, i]
    for i, (r, c) in enumerate(neg):
        pts[0, N + i] = [r, c, i]
    return torch.from_numpy(pts).to(device)


def _soft_iou(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * gt).sum(dim=[2, 3])
    union = p.sum(dim=[2, 3]) + gt.sum(dim=[2, 3]) - inter
    return (1.0 - inter / (union + 1e-6)).mean()


# ---------------------------------------------------------------------------
# Per-family training step
# ---------------------------------------------------------------------------

def _fixed_input_size(model) -> Optional[tuple]:
    """Input size a model is locked to, or None if it accepts any size.

    SimpleClick's ViT carries a fixed-size positional embedding, so it can only
    consume images at its native ``img_size``. RITM is fully convolutional and
    takes whatever it is given.
    """
    if getattr(model, '_model_type', None) != 'simpleclick':
        return None
    return tuple(model.backbone.patch_embed.img_size)


def _step_ritm_family(
    model, image_np, gt_np, cfg: FewShotConfig, device,
) -> list[float]:
    """Click-iterative training step for RITM / SimpleClick (ISModel API)."""
    # The dataset caps images at cfg.cpu_max_size/cuda_max_size, which yields
    # arbitrary sizes. Inference reaches SimpleClick through a fixed-size ZoomIn
    # crop, but this path feeds the model directly, so resize here or the ViT
    # dies on a pos_embed/patch-count mismatch.
    fixed = _fixed_input_size(model)
    if fixed is not None and gt_np.shape[:2] != fixed:
        import cv2
        image_np = cv2.resize(image_np, (fixed[1], fixed[0]),
                              interpolation=cv2.INTER_LINEAR)
        gt_np = cv2.resize(gt_np.astype(np.uint8), (fixed[1], fixed[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)

    H, W = gt_np.shape
    image_t = (torch.from_numpy(image_np.transpose(2, 0, 1))
               .float().unsqueeze(0).div(255.0).to(device))
    gt_t = (torch.from_numpy(gt_np.astype(np.float32))
            .unsqueeze(0).unsqueeze(0).to(device))

    with_prev_mask = getattr(model, 'with_prev_mask', False)
    prev_mask_t = torch.zeros(1, 1, H, W, device=device)
    prev_pred_np: Optional[np.ndarray] = None
    losses: list[float] = []

    for _ in range(cfg.n_clicks):
        pos, neg = _sample_click_pixels(gt_np, prev_pred_np, cfg.n_pos, cfg.n_neg)
        if not pos:
            continue
        points_nd = _build_ritm_points_nd(pos, neg, device)
        inp = torch.cat([image_t, prev_mask_t], dim=1) if with_prev_mask else image_t

        out = model(inp, points_nd)
        logits = out['instances']
        loss = (cfg.bce_weight * F.binary_cross_entropy_with_logits(logits, gt_t)
                + cfg.iou_weight * _soft_iou(logits, gt_t))
        loss.backward()
        losses.append(float(loss.detach()))

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            prev_pred_np = prob.squeeze().cpu().numpy()
            prev_mask_t = prob.detach()
    return losses


def _step_sam(
    sam_predictor, image_np, gt_np, cfg: FewShotConfig, device,
) -> list[float]:
    """Forward through SAM's prompt-encoder + mask-decoder with grads on.

    Accepts either a raw ``segment_anything.SamPredictor`` or our wrapper
    ``SAMInteractivePredictor`` (which holds the real predictor under
    ``.sam_predictor``).
    """
    if hasattr(sam_predictor, 'sam_predictor'):
        sam_predictor = sam_predictor.sam_predictor
    sam = sam_predictor.model
    H, W = gt_np.shape

    # Cache image embedding (no grad — encoder is frozen).
    with torch.no_grad():
        sam_predictor.set_image(image_np)
    image_embeddings = sam_predictor.features

    gt_t = (torch.from_numpy(gt_np.astype(np.float32))
            .unsqueeze(0).unsqueeze(0).to(device))

    losses: list[float] = []
    prev_pred_np: Optional[np.ndarray] = None

    for _ in range(cfg.n_clicks):
        pos, neg = _sample_click_pixels(gt_np, prev_pred_np, cfg.n_pos, cfg.n_neg)
        if not pos:
            continue
        # SAM uses (x, y).
        coords_xy = np.array([[c, r] for r, c in pos] + [[c, r] for r, c in neg],
                             dtype=np.float32)
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        coords_t = sam_predictor.transform.apply_coords(coords_xy, (H, W))
        coords_t = torch.as_tensor(coords_t, dtype=torch.float, device=device)[None, ...]
        labels_t = torch.as_tensor(labels, dtype=torch.int, device=device)[None, ...]

        sparse, dense = sam.prompt_encoder(
            points=(coords_t, labels_t), boxes=None, masks=None,
        )
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        masks = sam.postprocess_masks(low_res, sam_predictor.input_size, sam_predictor.original_size)

        loss = (cfg.bce_weight * F.binary_cross_entropy_with_logits(masks, gt_t)
                + cfg.iou_weight * _soft_iou(masks, gt_t))
        loss.backward()
        losses.append(float(loss.detach()))

        with torch.no_grad():
            prev_pred_np = torch.sigmoid(masks).squeeze().cpu().numpy()
    return losses


def _step_sam2(
    sam2_predictor, image_np, gt_np, cfg: FewShotConfig, device,
) -> list[float]:
    """SAM2 variant — uses sam_prompt_encoder / sam_mask_decoder names.

    Accepts either a raw ``SAM2ImagePredictor`` or our wrapper
    ``SAM2InteractivePredictor`` (which holds the real predictor under
    ``.sam2_predictor``).
    """
    if hasattr(sam2_predictor, 'sam2_predictor'):
        sam2_predictor = sam2_predictor.sam2_predictor
    sam2 = sam2_predictor.model
    H, W = gt_np.shape

    with torch.no_grad():
        sam2_predictor.set_image(image_np)

    feats = sam2_predictor._features
    image_embed = feats['image_embed']
    high_res_feats = feats.get('high_res_feats', None)

    gt_t = (torch.from_numpy(gt_np.astype(np.float32))
            .unsqueeze(0).unsqueeze(0).to(device))

    losses: list[float] = []
    prev_pred_np: Optional[np.ndarray] = None

    for _ in range(cfg.n_clicks):
        pos, neg = _sample_click_pixels(gt_np, prev_pred_np, cfg.n_pos, cfg.n_neg)
        if not pos:
            continue
        coords_xy = np.array([[c, r] for r, c in pos] + [[c, r] for r, c in neg],
                             dtype=np.float32)
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        coords_t = sam2_predictor._transforms.transform_coords(
            torch.as_tensor(coords_xy, device=device), normalize=True, orig_hw=(H, W)
        )
        coords_t = coords_t[None, ...]
        labels_t = torch.as_tensor(labels, dtype=torch.int, device=device)[None, ...]

        sparse, dense = sam2.sam_prompt_encoder(
            points=(coords_t, labels_t), boxes=None, masks=None,
        )
        low_res, _, _, _ = sam2.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        masks = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)

        loss = (cfg.bce_weight * F.binary_cross_entropy_with_logits(masks, gt_t)
                + cfg.iou_weight * _soft_iou(masks, gt_t))
        loss.backward()
        losses.append(float(loss.detach()))

        with torch.no_grad():
            prev_pred_np = torch.sigmoid(masks).squeeze().cpu().numpy()
    return losses


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FewShotTrainer:
    """
    Trains a LoRA adapter on a labeled support set and saves it to disk.

    Designed to run in a daemon thread; check ``stop()`` from the UI.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        dataset,                          # CustomDataset
        config: FewShotConfig,
        device: str,
        sam_predictor=None,               # required for SAM
        sam2_predictor=None,              # required for SAM2
        progress_cb: Optional[Callable] = None,    # (epoch, total, loss) -> None
        log_cb: Optional[Callable] = None,
    ):
        if model_type in ('sam',) and sam_predictor is None:
            raise ValueError("SAM training requires sam_predictor argument")
        if model_type == 'sam2' and sam2_predictor is None:
            raise ValueError("SAM2 training requires sam2_predictor argument")

        self.model = model
        self.model_type = model_type
        self.dataset = dataset
        self.config = config
        self.device = device
        self.sam_predictor = sam_predictor
        self.sam2_predictor = sam2_predictor
        self.progress_cb = progress_cb
        self.log_cb = log_cb

        self._stop = threading.Event()
        self._lora_state: Optional[dict] = None
        self._wrapped_modules: list[str] = []

    def stop(self) -> None:
        self._stop.set()

    def get_adapter_state(self) -> Optional[dict]:
        return self._lora_state

    def get_wrapped_module_names(self) -> list[str]:
        return list(self._wrapped_modules)

    def save(self, path: str) -> None:
        if self._lora_state is None:
            raise RuntimeError("No adapter to save — training has not completed.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_type': self.model_type,
                'rank': self.config.rank or DEFAULT_RANKS[self.model_type],
                'wrapped_modules': self._wrapped_modules,
                'lora_state': self._lora_state,
            },
            path,
        )

    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        device = self.device
        max_size = cfg.cuda_max_size if str(device).startswith('cuda') else cfg.cpu_max_size

        # Adjust dataset's max_size for CPU runs to keep training feasible.
        original_max_size = getattr(self.dataset, 'max_size', None)
        try:
            self.dataset.max_size = min(original_max_size or max_size, max_size)
        except AttributeError:
            pass

        # 1. Inject LoRA on the live model (predictors hold a reference, so
        #    after training they immediately benefit from the adapter).
        lora_cfg, wrapped = attach_adapter(self.model, self.model_type, rank=cfg.rank)
        self._wrapped_modules = wrapped
        if not wrapped:
            self._log("No modules matched LoRA preset; nothing to train.")
            return
        self._log(f"Injected LoRA into {len(wrapped)} modules "
                  f"(rank={lora_cfg.rank}, alpha={lora_cfg.alpha:.1f}).")

        set_lora_trainable(self.model, True)
        params = lora_parameters(self.model)
        n_trainable = sum(p.numel() for p in params)
        self._log(f"Trainable parameters: {n_trainable:,} "
                  f"(device={device}, image_size_cap={max_size}px)")

        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        # Keep BN/Dropout in eval (frozen running stats); LoRA modules have
        # neither, so eval mode is the correct training mode here.
        self.model.eval()

        for epoch in range(cfg.epochs):
            if self._stop.is_set():
                self._log("Training stopped by user.")
                break

            epoch_losses: list[float] = []
            for idx in range(len(self.dataset)):
                if self._stop.is_set():
                    break
                try:
                    image_np, gt_np = self.dataset[idx]
                except Exception as exc:
                    self._log(f"  [warn] sample {idx} failed: {exc}")
                    continue

                optimizer.zero_grad()
                if self.model_type in ('ritm', 'simpleclick'):
                    losses = _step_ritm_family(self.model, image_np, gt_np, cfg, device)
                elif self.model_type == 'sam':
                    losses = _step_sam(self.sam_predictor, image_np, gt_np, cfg, device)
                elif self.model_type == 'sam2':
                    losses = _step_sam2(self.sam2_predictor, image_np, gt_np, cfg, device)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")
                optimizer.step()
                epoch_losses.extend(losses)

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            self._log(f"Epoch {epoch + 1}/{cfg.epochs}  loss={avg:.4f}")
            if self.progress_cb:
                self.progress_cb(epoch + 1, cfg.epochs, avg)

        self._lora_state = collect_lora_state(self.model)
        # Return the model to inference mode (no grads) so downstream
        # predictor calls don't trip on grad-requiring tensors in transforms.
        set_lora_trainable(self.model, False)
        # Restore dataset config in case the same instance is reused.
        try:
            self.dataset.max_size = original_max_size
        except AttributeError:
            pass
        self._log("Few-shot adaptation complete.")

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)
