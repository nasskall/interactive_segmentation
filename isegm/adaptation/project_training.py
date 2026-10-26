"""
Fine-tuning SAM 2 on one labelling project's recorded clicks.

Every assisted annotation in the web app records the clicks that produced it
and the mask the labeller saved (after any correction). ``train`` fits LoRA
on SAM 2's prompt encoder and mask decoder (frozen image encoder, features
computed once and cached on disk) so that, replaying those clicks, the model
predicts the saved mask after each click. It is the objective of
``tools/distill.py train --target gt``, with the labellers' own clicks in
place of simulated ones.

``evaluate`` scores a model on held-out objects with a simulated user (each
click at the centre of the largest error, as in tools/click_benchmark.py),
taking the saved masks as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from interactive_demo.controller import InteractiveController
from isegm.inference.clicker import Clicker
from isegm.inference.predictors import get_predictor

from .distill import distill_step, sam2_features
from .lora import collect_lora_state, lora_parameters, set_lora_trainable
from .presets import attach_adapter

WORK_MAX_SIDE = 1024      # images are resized to this before encoding (SAM 2 works at 1024)
TARGET_MAX_SIDE = 512     # saved masks are stored at most this large as targets
MAX_CLICKS = 20           # longer click sequences are cut (later clicks are fine-tuning noise)


class Cancelled(Exception):
    pass


@dataclass
class Example:
    """One saved object: ``load`` returns (RGB image, bool mask) at the
    item's full resolution; ``clicks`` are ``(y, x, positive)`` in the same
    pixels, in the order the labeller made them."""
    key: str
    load: Callable[[], tuple[np.ndarray, np.ndarray]]
    clicks: list[tuple[int, int, bool]]


def _resize(image: np.ndarray, mask: np.ndarray, clicks, max_side: int):
    h, w = mask.shape
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return image, mask, list(clicks)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    clicks = [(min(size[1] - 1, int(y * s)), min(size[0] - 1, int(x * s)), p) for y, x, p in clicks]
    return image, mask, clicks


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def evaluate(net, examples: list[Example], device, max_clicks: int = 10,
             should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """mIoU after 1 / 3 / 5 clicks and mean clicks to 85% / 90% IoU (max_clicks
    when never reached) of a simulated user on ``examples``."""
    ctrl = InteractiveController(net, device, predictor_params={'brs_mode': 'NoBRS'},
                                 update_image_callback=lambda *a, **k: None)
    curves = []
    for ex in examples:
        if should_stop and should_stop():
            raise Cancelled()
        image, gt, _ = _resize(*ex.load(), [], WORK_MAX_SIDE)
        ctrl.set_image(image)
        robot = Clicker(gt_mask=gt.astype(np.int32))
        pred, ious = np.zeros_like(gt), []
        for _ in range(max_clicks):
            robot.make_next_click(pred)
            c = robot.clicks_list[-1]
            ctrl.add_click(int(c.coords[1]), int(c.coords[0]), is_positive=bool(c.is_positive))
            pred = ctrl.current_object_prob > ctrl.prob_thresh
            ious.append(_iou(pred, gt))
        ctrl.reset_last_object(update_image=False)
        curves.append(ious)
    curves = np.array(curves)

    def noc(t):
        hit = curves >= t
        return float(np.mean([np.argmax(r) + 1 if r.any() else max_clicks for r in hit]))
    return {'objects': len(examples), 'iou@1': float(curves[:, 0].mean()),
            'iou@3': float(curves[:, 2].mean()), 'iou@5': float(curves[:, 4].mean()),
            'noc@85': noc(0.85), 'noc@90': noc(0.90)}


def train(net, examples: list[Example], steps: int, device, work_dir: Path,
          lr: float = 2e-4, batch: int = 4, rank: int = 8, seed: int = 0,
          report: Optional[Callable[..., None]] = None,
          should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Attach LoRA to ``net`` (a SAM 2 model, changed in place) and fit it to
    ``examples``. Returns the adapter payload, loadable with
    ``InteractiveController.load_adapter``. ``report(phase, done, total,
    loss=None)`` is called as it goes; ``should_stop()`` cancels it."""
    report = report or (lambda *a, **k: None)
    stop = should_stop or (lambda: False)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Encode every image once; the encoder stays frozen.
    predictor = get_predictor(net, device=device)
    records = []
    for i, ex in enumerate(examples):
        if stop():
            raise Cancelled()
        image, mask, clicks = _resize(*ex.load(), ex.clicks[:MAX_CLICKS], WORK_MAX_SIDE)
        feats = sam2_features(predictor, image)
        _, small, _ = _resize(image, mask, [], TARGET_MAX_SIDE)
        f = work_dir / f'{ex.key}.pt'
        torch.save({'image_embed': feats['image_embed'].half().cpu(),
                    'high_res_feats': [t.half().cpu() for t in feats['high_res_feats']],
                    'hw': feats['hw']}, f)
        records.append({'file': f, 'clicks': clicks,
                        'target': np.where(small, 255, 0).astype(np.uint8)})
        report('encoding', i + 1, len(examples))

    _, wrapped = attach_adapter(net, 'sam2', rank=rank)
    set_lora_trainable(net, True)
    params = lora_parameters(net)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
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
            k = int(rng.integers(1, len(rec['clicks']) + 1))
            feats = torch.load(rec['file'], weights_only=True)
            targets = np.broadcast_to(rec['target'][None], (len(rec['clicks']), *rec['target'].shape))
            loss = distill_step(net, feats, rec['clicks'], targets, k, device)
            (loss / batch).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        report('training', step, steps, loss=float(np.mean(losses[-20 * batch:])))

    set_lora_trainable(net, False)
    net.eval()
    return {'model_type': 'sam2', 'rank': rank, 'wrapped_modules': wrapped,
            'lora_state': collect_lora_state(net)}
