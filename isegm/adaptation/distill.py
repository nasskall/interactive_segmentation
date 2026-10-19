"""
Distilling an interactive teacher (e.g. an adapted ensemble) into SAM 2.

The student is a SAM 2 model with LoRA on its prompt encoder and mask
decoder; the image encoder stays frozen, so each image's encoder features
are computed once (``sam2_features``) and reused for every step.

A training record is one simulated interaction on one image: the clicks in
order, as ``(y, x, is_positive)``, and the teacher's probability map after
each click (the target). ``distill_step`` replays the first k-1 clicks
without gradients, feeding each prediction's low-res logits into the next
exactly as ``SAM2InteractivePredictor`` does live (three candidates on the
first click, the best-scoring one kept), then trains the k-th prediction
towards the target after k clicks with BCE + soft IoU. Targets may be soft
(teacher probabilities) or hard (expert masks, for a control run).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .few_shot_trainer import _soft_iou


def _inner(predictor):
    """The raw ``SAM2ImagePredictor`` behind our wrapper, or itself."""
    return getattr(predictor, 'sam2_predictor', predictor)


@torch.no_grad()
def sam2_features(predictor, image: np.ndarray) -> dict:
    """Frozen-encoder features of ``image`` (H x W x 3 RGB uint8), in the form
    ``distill_step`` takes; detach and store as float16 to cache."""
    p = _inner(predictor)
    p.set_image(image)
    return {'image_embed': p._features['image_embed'],
            'high_res_feats': list(p._features['high_res_feats']),
            'hw': tuple(image.shape[:2])}


def _forward(model, feats: dict, clicks, prev_logit: Optional[torch.Tensor], device):
    """One prediction for ``clicks``; returns (logits at image size,
    low-res logits of the kept mask)."""
    h, w = feats['hw']
    coords = torch.tensor([[x * 1024.0 / w, y * 1024.0 / h] for y, x, _ in clicks],
                          dtype=torch.float32, device=device)[None]
    labels = torch.tensor([int(p) for _, _, p in clicks], dtype=torch.int, device=device)[None]
    sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None,
                                             masks=prev_logit)
    multimask = len(clicks) == 1
    low_res, scores, _, _ = model.sam_mask_decoder(
        image_embeddings=feats['image_embed'].to(device, torch.float32),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask,
        repeat_image=False,
        high_res_features=[f.to(device, torch.float32) for f in feats['high_res_feats']],
    )
    if multimask:
        best = int(scores[0].argmax())
        low_res = low_res[:, best:best + 1]
    return F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False), low_res


def distill_step(model, feats: dict, clicks: list, targets: np.ndarray, k: int, device,
                 bce_weight: float = 1.0, iou_weight: float = 1.0) -> torch.Tensor:
    """Loss for the prediction after ``k`` clicks (1-based) against
    ``targets[k - 1]`` (H' x W' in [0, 1], resized to the image if needed).
    Call ``.backward()`` on the result."""
    prev = None
    with torch.no_grad():
        for i in range(1, k):
            _, prev = _forward(model, feats, clicks[:i], prev, device)
    logits, _ = _forward(model, feats, clicks[:k], prev, device)
    t = torch.as_tensor(np.asarray(targets[k - 1], np.float32), device=device)[None, None]
    if t.shape[-2:] != logits.shape[-2:]:
        t = F.interpolate(t, size=logits.shape[-2:], mode='bilinear', align_corners=False)
    return (bce_weight * F.binary_cross_entropy_with_logits(logits, t)
            + iou_weight * _soft_iou(logits, t))
