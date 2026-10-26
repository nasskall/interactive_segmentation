"""
Click-free ("promptless") SAM 2: one learned prompt per class.

SAM 2's mask decoder normally takes the embeddings of clicks or a box. Here
each class gets a few learned prompt tokens instead; given an image, the
decoder runs once with every class's tokens (the image embedding is shared)
and returns one mask per class, with no clicks. The image encoder stays
frozen, so each training image is encoded once and its features cached on
disk; LoRA on the prompt encoder and mask decoder plus the class tokens are
trained to reproduce the masks labellers saved, with BCE + soft IoU.

Training images must be fully labelled: a class absent from an image's
target is learned as "not in this image".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.inference.predictors import get_predictor

from .distill import sam2_features
from .few_shot_trainer import _soft_iou
from .lora import collect_lora_state, load_lora_state, lora_parameters, set_lora_trainable
from .presets import attach_adapter
from .project_training import WORK_MAX_SIDE, Cancelled, _resize

TARGET_MAX_SIDE = 256     # targets / training logits (SAM 2's low-res masks are 256)
N_TOKENS = 4              # learned prompt tokens per class


@dataclass
class Example:
    """One fully labelled image: ``load`` returns (RGB image, bool masks of
    shape (n_classes, H, W)) at the item's full resolution."""
    key: str
    load: Callable[[], tuple[np.ndarray, np.ndarray]]


class ClassPrompts(nn.Module):
    """The learned prompt tokens of every class."""

    def __init__(self, n_classes: int, dim: int = 256, n_tokens: int = N_TOKENS):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_classes, n_tokens, dim) * 0.02)


def class_logits(model, prompts: ClassPrompts, feats: dict, device) -> torch.Tensor:
    """Low-res (256 x 256) mask logits of every class, (n_classes, 256, 256)."""
    n = prompts.tokens.shape[0]
    sparse, dense = model.sam_prompt_encoder(points=None, boxes=None, masks=None)
    sparse = torch.cat([sparse.expand(n, -1, -1), prompts.tokens.to(device)], dim=1)
    low_res, _, _, _ = model.sam_mask_decoder(
        image_embeddings=feats['image_embed'].to(device, torch.float32),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense.expand(n, -1, -1, -1),
        multimask_output=False,
        repeat_image=True,
        high_res_features=[f.to(device, torch.float32) for f in feats['high_res_feats']],
    )
    return low_res[:, 0]


@torch.no_grad()
def predict(model, prompts: ClassPrompts, image: np.ndarray, device) -> np.ndarray:
    """Probability of every class at ``image``'s size, (n_classes, H, W)."""
    h, w = image.shape[:2]
    small, _, _ = _resize(image, np.zeros((h, w), bool), [], WORK_MAX_SIDE)
    feats = sam2_features(get_predictor(model, device=device), small)
    logits = class_logits(model, prompts, feats, device)[None]
    return torch.sigmoid(F.interpolate(logits, size=(h, w), mode='bilinear',
                                       align_corners=False))[0].cpu().numpy()


def _targets(masks: np.ndarray, max_side: int) -> np.ndarray:
    """(n_classes, h, w) uint8 0/1 masks, longest side at most ``max_side``
    (sized as ``_resize`` sizes the image)."""
    h, w = masks.shape[1:]
    s = min(1.0, max_side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.stack([cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
                     for m in masks])


def evaluate(model, prompts: ClassPrompts, examples: list[Example], device,
             should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """IoU of each class over all of ``examples`` (intersection and union
    summed over the images) and their mean."""
    n = prompts.tokens.shape[0]
    inter, union = np.zeros(n), np.zeros(n)
    for ex in examples:
        if should_stop and should_stop():
            raise Cancelled()
        image, masks = ex.load()
        image, _, _ = _resize(image, masks[0], [], WORK_MAX_SIDE)
        masks = _targets(masks, WORK_MAX_SIDE).astype(bool)
        pred = predict(model, prompts, image, device) > 0.5
        inter += np.logical_and(pred, masks).sum(axis=(1, 2))
        union += np.logical_or(pred, masks).sum(axis=(1, 2))
    # A class absent from both prediction and labels everywhere scores 1.
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return {'images': len(examples), 'iou': [float(v) for v in iou], 'miou': float(iou.mean())}


def train(model, n_classes: int, examples: list[Example], steps: int, device, work_dir: Path,
          lr: float = 1e-3, token_lr: float = 1e-2, batch: int = 4, rank: int = 8, seed: int = 0,
          report: Optional[Callable[..., None]] = None,
          should_stop: Optional[Callable[[], bool]] = None) -> tuple[ClassPrompts, dict]:
    """Attach LoRA to ``model`` (SAM 2, changed in place), learn class prompts
    and fit both to ``examples``. Returns the prompts and the payload to save
    (``load`` reads it back)."""
    report = report or (lambda *a, **k: None)
    stop = should_stop or (lambda: False)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    predictor = get_predictor(model, device=device)
    records = []
    for i, ex in enumerate(examples):
        if stop():
            raise Cancelled()
        image, masks = ex.load()
        image, _, _ = _resize(image, masks[0], [], WORK_MAX_SIDE)
        feats = sam2_features(predictor, image)
        f = work_dir / f'{ex.key}.pt'
        torch.save({'image_embed': feats['image_embed'].half().cpu(),
                    'high_res_feats': [t.half().cpu() for t in feats['high_res_feats']],
                    'hw': feats['hw']}, f)
        records.append({'file': f, 'target': _targets(masks, TARGET_MAX_SIDE)})
        report('encoding', i + 1, len(examples))

    _, wrapped = attach_adapter(model, 'sam2', rank=rank)
    set_lora_trainable(model, True)
    prompts = ClassPrompts(n_classes).to(device)
    params = lora_parameters(model)
    opt = torch.optim.AdamW([{'params': params, 'lr': lr},
                             {'params': prompts.parameters(), 'lr': token_lr}], weight_decay=1e-4)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    losses = []
    for step in range(1, steps + 1):
        if stop():
            raise Cancelled()
        opt.zero_grad()
        for _ in range(batch):
            rec = records[rng.integers(len(records))]
            feats = torch.load(rec['file'], weights_only=True)
            logits = class_logits(model, prompts, feats, device)[None]
            t = torch.as_tensor(rec['target'], dtype=torch.float32, device=device)[None]
            logits = F.interpolate(logits, size=t.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logits, t) + _soft_iou(logits, t)
            (loss / batch).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_([*params, *prompts.parameters()], 1.0)
        opt.step()
        sched.step()
        report('training', step, steps, loss=float(np.mean(losses[-20 * batch:])))

    set_lora_trainable(model, False)
    model.eval()
    prompts.requires_grad_(False)
    return prompts, {'model_type': 'sam2', 'kind': 'automatic', 'rank': rank,
                     'wrapped_modules': wrapped, 'lora_state': collect_lora_state(model),
                     'class_tokens': prompts.tokens.detach().cpu()}


def load(model, payload: dict) -> ClassPrompts:
    """Apply a saved promptless adapter to a fresh SAM 2 ``model``; returns
    its class prompts."""
    attach_adapter(model, 'sam2', rank=payload['rank'])
    missing, _ = load_lora_state(model, payload['lora_state'])
    if missing:
        raise ValueError(f'Adapter does not fit this SAM 2 model ({len(missing)} layers missing)')
    set_lora_trainable(model, False)
    model.eval()
    tokens = payload['class_tokens']
    prompts = ClassPrompts(tokens.shape[0], tokens.shape[2], tokens.shape[1])
    prompts.tokens.data.copy_(tokens)
    prompts.requires_grad_(False)
    return prompts
