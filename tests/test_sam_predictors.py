"""SAM / SAM 2 wrapper behaviour and auto-segment mask selection.

The wrapper and selection tests use stand-in predictors, so they run without
segment_anything, sam2 or any checkpoint. The last test runs the real
automatic mask generators when they are available.
"""

from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest
import torch

from interactive_demo.controller import InteractiveController, pick_auto_mask
from isegm.inference.clicker import Click, Clicker
from isegm.inference.predictors import sam2_predictor, sam_predictor

H, W = 64, 80


class FakePredictor:
    """Mimics SamPredictor / SAM2ImagePredictor: three candidates in
    multimask mode (the middle one scores best), one otherwise."""

    def __init__(self, model):
        self.calls = []

    def set_image(self, image):
        pass

    def predict(self, point_coords, point_labels, mask_input, multimask_output):
        self.calls.append(multimask_output)
        n = 3 if multimask_output else 1
        masks = np.zeros((n, H, W), dtype=bool)
        for i in range(n):
            masks[i, : 10 * (i + 1)] = True
        scores = np.array([0.2, 0.9, 0.5][:n])
        logits = np.stack([np.full((256, 256), float(i)) for i in range(n)])
        return masks, scores, logits


@pytest.fixture(params=['sam', 'sam2'])
def wrapper(request, monkeypatch):
    if request.param == 'sam':
        monkeypatch.setattr(sam_predictor, 'SAM_AVAILABLE', True)
        monkeypatch.setattr(sam_predictor, 'SamPredictor', FakePredictor, raising=False)
        pred = sam_predictor.SAMInteractivePredictor(object(), 'cpu')
    else:
        monkeypatch.setattr(sam2_predictor, 'SAM2_AVAILABLE', True)
        monkeypatch.setattr(sam2_predictor, 'SAM2ImagePredictor', FakePredictor, raising=False)
        monkeypatch.setattr(sam2_predictor, 'torch', torch, raising=False)
        pred = sam2_predictor.SAM2InteractivePredictor(object(), 'cpu')
    pred.set_input_image(np.zeros((H, W, 3), np.uint8))
    return pred


def _inner(pred):
    return getattr(pred, 'sam_predictor', None) or pred.sam2_predictor


def test_first_click_uses_best_of_three_candidates(wrapper):
    clicker = Clicker()
    clicker.add_click(Click(is_positive=True, coords=(5, 5)))
    prob = wrapper.get_prediction(clicker)

    assert _inner(wrapper).calls == [True]
    # Candidate 1 scores highest: the top 20 rows, with its logit kept
    # as the prompt for the next click.
    assert prob[:20].all() and not prob[20:].any()
    assert (wrapper.prev_logit == 1.0).all() and wrapper.prev_logit.shape == (1, 256, 256)


def test_later_clicks_use_a_single_mask(wrapper):
    clicker = Clicker()
    clicker.add_click(Click(is_positive=True, coords=(5, 5)))
    wrapper.get_prediction(clicker)
    clicker.add_click(Click(is_positive=False, coords=(40, 40)))
    wrapper.get_prediction(clicker)

    assert _inner(wrapper).calls == [True, False]


# ---------------------------------------------------------------------------
# Auto-segment mask selection
# ---------------------------------------------------------------------------

def _lesion_scene():
    """Background ring around a central lesion, plus a small lesion part."""
    lesion = np.zeros((H, W), bool)
    cv2.circle(lesion.view(np.uint8), (W // 2, H // 2), 15, 1, -1)
    part = np.zeros((H, W), bool)
    part[H // 2 - 3: H // 2 + 3, W // 2 - 3: W // 2 + 3] = True
    candidates = [
        {'segmentation': ~lesion, 'area': int((~lesion).sum())},
        {'segmentation': lesion, 'area': int(lesion.sum())},
        {'segmentation': part, 'area': int(part.sum())},
    ]
    return candidates, lesion


def test_pick_auto_mask_skips_background():
    candidates, lesion = _lesion_scene()
    assert candidates[0]['area'] > candidates[1]['area']  # background is largest
    assert pick_auto_mask(candidates)['segmentation'] is lesion


def test_pick_auto_mask_keeps_object_touching_one_edge():
    # A lesion cut off by one image edge is still foreground.
    lesion = np.zeros((H, W), bool)
    lesion[:30, 20:50] = True
    rest = ~lesion
    picked = pick_auto_mask([{'segmentation': rest, 'area': int(rest.sum())},
                             {'segmentation': lesion, 'area': int(lesion.sum())}])
    assert picked['segmentation'] is lesion


def test_pick_auto_mask_falls_back_to_largest():
    # Both halves touch much of the border; the larger one wins.
    top = np.zeros((H, W), bool)
    top[: H // 3] = True
    bottom = ~top
    picked = pick_auto_mask([{'segmentation': top, 'area': int(top.sum())},
                             {'segmentation': bottom, 'area': int(bottom.sum())}])
    assert picked['segmentation'] is bottom


def test_auto_segment_selects_lesion_for_sam(monkeypatch):
    candidates, lesion = _lesion_scene()

    class FakeGenerator:
        def __init__(self, model):
            pass

        def generate(self, image):
            return candidates

    monkeypatch.setattr(sam_predictor, 'SAM_AVAILABLE', True)
    monkeypatch.setattr(sam_predictor, 'SamPredictor', FakePredictor, raising=False)
    monkeypatch.setitem(sys.modules, 'segment_anything',
                        types.SimpleNamespace(SamAutomaticMaskGenerator=FakeGenerator))
    net = types.SimpleNamespace(_model_type='sam')
    ctrl = InteractiveController(net, 'cpu', predictor_params={},
                                 update_image_callback=lambda *a, **k: None)
    ctrl.set_image(np.zeros((H, W, 3), np.uint8))

    res = ctrl.auto_segment(mode='auto')
    assert res['ok'] and res['mode_used'] == 'auto_mask'
    np.testing.assert_array_equal(ctrl.result_mask > 0, lesion)


# ---------------------------------------------------------------------------
# Real automatic mask generators
# ---------------------------------------------------------------------------

def _synthetic_lesion():
    """Dark irregular lesion on noisy skin, as in the RITM domain."""
    rng = np.random.default_rng(7)
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - h / 2, xx - w / 2)
    r = 0.22 * h + 6 * np.sin(3 * ang) + 4 * np.sin(5 * ang + 1)
    gt = np.hypot(yy - h / 2, xx - w / 2) < r
    img = np.empty((h, w, 3), np.float32)
    img[:] = (205, 160, 140)
    img[gt] = (95, 60, 50)
    img += rng.normal(0, 7, img.shape)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.clip(img, 0, 255).astype(np.uint8), gt


@pytest.mark.slow
@pytest.mark.parametrize('fixture_name', [
    pytest.param('sam_model', marks=pytest.mark.sam),
    pytest.param('sam2_model', marks=pytest.mark.sam2),
])
def test_real_auto_segment_finds_lesion(fixture_name, request, device):
    model = request.getfixturevalue(fixture_name)[0]
    img, gt = _synthetic_lesion()
    ctrl = InteractiveController(model, device, predictor_params={},
                                 update_image_callback=lambda *a, **k: None)
    ctrl.set_image(img)

    res = ctrl.auto_segment(mode='auto')
    assert res['ok'] and res['mode_used'] == 'auto_mask'
    pred = ctrl.result_mask > 0
    iou = (pred & gt).sum() / (pred | gt).sum()
    assert iou > 0.8, f'IoU with the lesion is {iou:.3f}'
