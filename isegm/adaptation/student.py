"""
Standalone student: a small segmentation network distilled from a project's
automatic SAM 2 model.

A U-Net with an ImageNet-pretrained encoder (``segmentation_models_pytorch``)
predicts one map per class, like ``promptless``, but needs neither SAM 2 nor
clicks, runs about ten times faster on a CPU and can be exported. It learns
from the labels of fully labelled images and, on every other image of the
project, from the teacher's probability maps, so unlabelled images add
training data. Images are resized to a square ``INPUT_SIZE`` (aspect not
kept) with flip augmentation; outputs are resized back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .few_shot_trainer import _soft_iou
from .project_training import Cancelled

ENCODER = 'resnet18'
INPUT_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Example:
    """One training or held-out image: ``load`` returns (RGB image, float
    targets (n_classes, H, W) in [0, 1]) at any resolution: labels as 0 / 1,
    or a teacher's probabilities."""
    key: str
    load: Callable[[], tuple[np.ndarray, np.ndarray]]


def encoder_weights_repo(encoder: str = ENCODER) -> str:
    """``repo@revision`` on the Hugging Face hub of the pretrained encoder
    weights (``segmentation_models_pytorch`` pins the revision)."""
    from segmentation_models_pytorch.encoders import encoders
    settings = encoders[encoder]['pretrained_settings']['imagenet']
    return f"{settings['repo_id']}@{settings['revision']}"


def build(n_classes: int, encoder: str = ENCODER, pretrained: bool = True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder, encoder_weights='imagenet' if pretrained else None,
                    classes=n_classes)


def _resized(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _normalise(images: np.ndarray) -> torch.Tensor:
    """uint8 (..., H, W, 3) RGB -> normalised float (..., 3, H, W)."""
    x = (images.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(np.moveaxis(x, -1, -3).copy())


def _target(t: np.ndarray, size: int) -> np.ndarray:
    """(n_classes, size, size) uint8, 0..255 for 0..1."""
    return np.stack([np.round(cv2.resize(c.astype(np.float32), (size, size),
                                         interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
                     for c in t])


@torch.no_grad()
def predict(model, image: np.ndarray, device, size: int = INPUT_SIZE) -> np.ndarray:
    """Probability of every class at ``image``'s size, (n_classes, H, W)."""
    model.eval()
    h, w = image.shape[:2]
    logits = model(_normalise(_resized(image, size))[None].to(device))
    return torch.sigmoid(F.interpolate(logits, size=(h, w), mode='bilinear',
                                       align_corners=False))[0].cpu().numpy()


def evaluate(model, examples: list[Example], device, size: int = INPUT_SIZE,
             should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Per-class IoU over all ``examples`` together, as ``promptless.evaluate``."""
    inter = union = None
    for ex in examples:
        if should_stop and should_stop():
            raise Cancelled()
        image, target = ex.load()
        pred = predict(model, image, device, size) > 0.5
        target = target > 0.5
        i = np.logical_and(pred, target).sum(axis=(1, 2))
        u = np.logical_or(pred, target).sum(axis=(1, 2))
        inter, union = (i, u) if inter is None else (inter + i, union + u)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return {'images': len(examples), 'iou': [float(v) for v in iou], 'miou': float(iou.mean())}


def train(model, examples: list[Example], steps: int, device, lr: float = 1e-3,
          batch: int = 4, size: int = INPUT_SIZE, seed: int = 0,
          report: Optional[Callable[..., None]] = None,
          should_stop: Optional[Callable[[], bool]] = None) -> None:
    """Fit ``model`` (from ``build``) to ``examples`` in place: BCE + soft IoU,
    AdamW with warm-up and cosine decay, random horizontal / vertical flips.
    Inputs and targets are prepared once and kept in memory at ``size``, as
    uint8 (about 0.5 MB per image and class at 384 px)."""
    report = report or (lambda *a, **k: None)
    stop = should_stop or (lambda: False)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = []
    for i, ex in enumerate(examples):
        if stop():
            raise Cancelled()
        image, target = ex.load()
        data.append((_resized(image, size), _target(target, size)))
        report('preparing', i + 1, len(examples))

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    losses = []
    for step in range(1, steps + 1):
        if stop():
            raise Cancelled()
        picks = [data[k] for k in rng.integers(len(data), size=batch)]
        x = _normalise(np.stack([p[0] for p in picks]))
        t = torch.from_numpy(np.stack([p[1] for p in picks]).astype(np.float32) / 255)
        if rng.random() < 0.5:
            x, t = x.flip(-1), t.flip(-1)
        if rng.random() < 0.5:
            x, t = x.flip(-2), t.flip(-2)
        x, t = x.to(device), t.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, t) + _soft_iou(logits, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        report('training', step, steps, loss=float(np.mean(losses[-20:])))
    model.eval()


def payload(model, classes: list[dict], encoder: str = ENCODER, size: int = INPUT_SIZE) -> dict:
    """What a saved student holds (plain tensors and lists: loads with
    ``torch.load(weights_only=True)``)."""
    return {'kind': 'student', 'arch': 'unet', 'encoder': encoder, 'input_size': size,
            'classes': classes, 'state_dict': {k: v.detach().cpu() for k, v in
                                               model.state_dict().items()}}


def load(saved: dict):
    """A student model from ``payload``; returns (model, input size)."""
    if saved.get('kind') != 'student' or saved.get('arch') != 'unet':
        raise ValueError('Not a U-Net student')
    model = build(len(saved['classes']), saved['encoder'], pretrained=False)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved['input_size']
