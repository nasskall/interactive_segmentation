"""Ensemble predictor: fusion maths, click pinning, controller integration."""

from __future__ import annotations

import numpy as np
import pytest

from isegm.inference.clicker import Click
from isegm.inference.predictors.ensemble import (EnsembleConfig, EnsembleNet, click_agreement,
                                                 fuse, pin_clicks, reliability_weights, staple)


def _square(h=40, w=40, y0=10, y1=30, x0=10, x1=30, p=0.95):
    out = np.full((h, w), 1 - p, np.float32)
    out[y0:y1, x0:x1] = p
    return out


GOOD = _square()
GOOD2 = _square(y0=11, y1=31)
BAD = _square(y0=0, y1=40, x0=0, x1=40)          # floods the whole frame
CLICKS = [Click(True, (20, 20)), Click(False, (2, 2))]


def test_click_agreement():
    assert click_agreement(GOOD, CLICKS, 1) == pytest.approx(0.95)
    assert click_agreement(BAD, CLICKS, 1) == pytest.approx(0.5)   # misses the negative
    assert click_agreement(GOOD, [], 1) == 1.0


def test_weighted_fusion_downweights_the_outlier():
    probs = np.stack([GOOD, GOOD2, BAD])
    w = reliability_weights(probs, CLICKS, EnsembleConfig())
    assert w.sum() == pytest.approx(1.0)
    assert w[2] < 0.1 < w[0]
    fused, weights = fuse(probs, CLICKS, EnsembleConfig(fusion='weighted'))
    assert weights == pytest.approx(w.tolist())
    assert fused[2, 2] < 0.5 < fused[20, 20]
    # A plain mean is pulled towards the outlier at the negative click.
    mean, _ = fuse(probs, CLICKS, EnsembleConfig(fusion='mean'))
    assert mean[2, 2] > fused[2, 2]


def test_weighted_conf_and_staple_follow_the_majority():
    probs = np.stack([GOOD, GOOD2, BAD])
    for method in ('weighted_conf', 'staple'):
        fused, _ = fuse(probs, CLICKS, EnsembleConfig(fusion=method))
        assert fused.shape == GOOD.shape and fused.dtype == np.float32
        assert fused[20, 20] > 0.5 and fused[2, 2] < 0.5, method


def test_staple_agrees_with_unanimous_models():
    post = staple(np.stack([GOOD, GOOD, GOOD]))
    np.testing.assert_array_equal(post > 0.5, GOOD > 0.5)


def test_pin_clicks():
    prob = np.full((20, 20), 0.5, np.float32)
    pin_clicks(prob, [Click(True, (5, 5)), Click(False, (15, 15))], 2)
    assert prob[5, 5] == 1 and prob[5, 7] == 1 and prob[7, 7] == 0.5   # disc, not square
    assert prob[15, 15] == 0 and prob[0, 0] == 0.5


def test_config_validation(fresh_ritm):
    with pytest.raises(ValueError):
        EnsembleNet([('only', fresh_ritm)])
    with pytest.raises(ValueError):
        EnsembleNet([('a', fresh_ritm), ('b', fresh_ritm)], EnsembleConfig(fusion='vote'))


@pytest.mark.parametrize('fusion', ['mean', 'weighted', 'staple'])
def test_ensemble_runs_through_the_controller(fresh_ritm, device, synthetic_pair, fusion):
    from interactive_demo.controller import InteractiveController
    net = EnsembleNet([('a', fresh_ritm), ('b', fresh_ritm)], EnsembleConfig(fusion=fusion))
    ctrl = InteractiveController(net, device, predictor_params={'brs_mode': 'NoBRS'},
                                 update_image_callback=lambda *a, **k: None)
    img, gt = synthetic_pair
    ctrl.set_image(img)
    ys, xs = np.where(gt)
    ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
    first = ctrl.current_object_prob.copy()
    ctrl.add_click(2, 2, is_positive=False)
    assert ctrl.current_object_prob.shape == gt.shape
    assert ctrl.current_object_prob[2, 2] == 0            # pinned negative click
    assert set(ctrl.predictor.last_weights) == {'a', 'b'}
    ctrl.undo_click()
    np.testing.assert_allclose(ctrl.current_object_prob, first)
    ctrl.finish_object()
    assert ctrl.object_count == 1


@pytest.mark.slow
def test_ensemble_adapts_each_member_and_rolls_back(ritm_checkpoint_path, device, synthetic_pair):
    import torch
    from conftest import _fresh_ritm
    from interactive_demo.controller import InteractiveController
    from isegm.adaptation import EnsembleAdapter
    from isegm.adaptation.lora import collect_lora_state
    a, b = _fresh_ritm(ritm_checkpoint_path, device), _fresh_ritm(ritm_checkpoint_path, device)
    net = EnsembleNet([('a', a), ('b', b)])
    ctrl = InteractiveController(net, device, predictor_params={'brs_mode': 'NoBRS'},
                                 update_image_callback=lambda *a, **k: None)
    img, gt = synthetic_pair
    ys, xs = np.where(gt)
    ctrl.online_recording = True
    ctrl.set_image(img)
    ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
    ctrl.finish_object()

    adapter = ctrl._ensure_online_adapter()
    assert isinstance(adapter, EnsembleAdapter) and len(adapter.members) == 2
    for _, m in adapter.members:
        m.config.steps, m.config.lr = 2, 1e-2
    pre = [collect_lora_state(a), collect_lora_state(b)]
    rep = ctrl.adapt_now()
    assert set(rep['members']) == {'a', 'b'} and rep['steps'] == 4
    post = [collect_lora_state(a), collect_lora_state(b)]
    for p0, p1 in zip(pre, post):
        assert any(not torch.allclose(p0[k], p1[k]) for k in p0 if not k.endswith('.scale'))

    assert ctrl.rollback_adapter() is True
    for p0, net_ in zip(pre, (a, b)):
        now = collect_lora_state(net_)
        assert all(torch.allclose(now[k], v) for k, v in p0.items())
    assert ctrl.rollback_adapter() is False

    # A net shared by two members is adapted once.
    assert len(EnsembleAdapter(EnsembleNet([('x', a), ('y', a)]), device).members) == 1
