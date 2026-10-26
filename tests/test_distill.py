"""Distillation into SAM 2: the training forward pass must reproduce the
live predictor, and a few steps must move the student towards the target."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isegm.adaptation import attach_adapter
from isegm.adaptation.distill import _forward, distill_step, sam2_features
from isegm.adaptation.lora import lora_parameters, set_lora_trainable
from isegm.inference.clicker import Click
from isegm.inference.predictors import get_predictor

pytestmark = [pytest.mark.slow, pytest.mark.sam2]


class _Clicks:
    def __init__(self, clicks):
        self.clicks = clicks

    def get_clicks(self, clicks_limit=None):
        return self.clicks


def test_forward_matches_the_live_predictor(sam2_model, synthetic_pair, device):
    net, _, _ = sam2_model
    img, gt = synthetic_pair
    pred = get_predictor(net, device=device)
    pred.set_input_image(img)
    feats = sam2_features(pred, img)
    ys, xs = np.where(gt)
    clicks = [(int(ys.mean()), int(xs.mean()), True), (2, 2, False)]

    prev = None
    for k in (1, 2):
        live = pred.get_prediction(_Clicks([Click(p, (y, x)) for y, x, p in clicks[:k]]))
        with torch.no_grad():
            logits, prev = _forward(net, feats, clicks[:k], prev, device)
        ours = (logits[0, 0] > 0).cpu().numpy()
        agree = (ours == (live > 0.5)).mean()
        assert agree > 0.995, f'click {k}: {agree:.4f}'


def test_distill_steps_reduce_the_loss(sam2_model, synthetic_pair, device):
    import copy
    net = copy.deepcopy(sam2_model[0])
    img, gt = synthetic_pair
    pred = get_predictor(net, device=device)
    feats = sam2_features(pred, img)
    attach_adapter(net, 'sam2')
    set_lora_trainable(net, True)
    opt = torch.optim.Adam(lora_parameters(net), lr=1e-3)
    ys, xs = np.where(gt)
    clicks = [(int(ys.mean()), int(xs.mean()), True)]
    # An odd target the base model would not produce: the left half of the lesion.
    target = gt.astype(np.float32)
    target[:, :int(xs.mean())] = 0
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss = distill_step(net, feats, clicks, target[None], 1, device)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
