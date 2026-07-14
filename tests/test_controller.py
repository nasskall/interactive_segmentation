"""Controller adapter slot lifecycle + auto_segment dispatch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from interactive_demo.controller import InteractiveController


def _ctrl(model, device):
    return InteractiveController(
        model, device,
        predictor_params={'brs_mode': 'NoBRS'},
        update_image_callback=lambda *a, **k: None,
    )


# ---------------------------------------------------------------------------
# Model swap clears adapter
# ---------------------------------------------------------------------------

def test_model_type_property(fresh_ritm, device):
    ctrl = _ctrl(fresh_ritm, device)
    assert ctrl.model_type == 'ritm'


def test_set_net_clears_adapter_and_buffer(fresh_ritm, device,
                                            ritm_checkpoint_path):
    ctrl = _ctrl(fresh_ritm, device)
    # Populate adapter slot + buffer
    ctrl._ensure_online_adapter()
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    ctrl.replay_buffer.add(img, [(10, 10, True)], np.zeros((64, 64), dtype=bool))
    assert ctrl._online_adapter is not None
    assert len(ctrl.replay_buffer) == 1

    # Swap to a fresh model — slot and buffer must clear.
    from isegm.inference import utils as iu
    new_model = iu.load_is_model(str(ritm_checkpoint_path), device=device,
                                  cpu_dist_maps=True)
    new_model._model_type = 'ritm'
    ctrl.set_net(new_model)

    assert ctrl._online_adapter is None
    assert len(ctrl.replay_buffer) == 0


# ---------------------------------------------------------------------------
# Online adapter slot
# ---------------------------------------------------------------------------

def test_ensure_online_adapter_injects_lora(fresh_ritm, device):
    ctrl = _ctrl(fresh_ritm, device)
    adapter = ctrl._ensure_online_adapter()
    assert adapter._injected
    assert len(adapter._wrapped) > 0


def test_adapt_now_with_empty_buffer_returns_safely(fresh_ritm, device):
    ctrl = _ctrl(fresh_ritm, device)
    rep = ctrl.adapt_now()
    assert rep['steps'] == 0
    assert np.isnan(rep['mean_loss'])


def test_rollback_when_no_adapter_returns_false(fresh_ritm, device):
    ctrl = _ctrl(fresh_ritm, device)
    assert ctrl.rollback_adapter() is False


def test_save_adapter_requires_existing_adapter(fresh_ritm, device, tmp_path):
    ctrl = _ctrl(fresh_ritm, device)
    with pytest.raises(RuntimeError):
        ctrl.save_adapter(str(tmp_path / 'x.pt'))


def test_load_adapter_rejects_wrong_model_type(fresh_ritm, device, tmp_path):
    ctrl = _ctrl(fresh_ritm, device)
    ctrl._ensure_online_adapter()
    state = ctrl._online_adapter.state_dict()
    state['model_type'] = 'sam'   # tamper
    p = tmp_path / 'wrong.pt'
    torch.save(state, p)
    with pytest.raises(ValueError):
        ctrl.load_adapter(str(p))


def test_save_load_roundtrip_preserves_status(fresh_ritm, device, tmp_path):
    ctrl = _ctrl(fresh_ritm, device)
    ctrl._ensure_online_adapter()
    p = tmp_path / 'a.pt'
    ctrl.save_adapter(str(p))
    assert p.exists()
    missing, unexpected = ctrl.load_adapter(str(p))
    assert missing == [] and unexpected == []
    assert 'unsaved' not in ctrl.adapter_status() or ctrl._adapter_path is not None


# ---------------------------------------------------------------------------
# Online recording → buffer capture
# ---------------------------------------------------------------------------

def test_finish_object_pushes_to_buffer_when_recording(
    fresh_ritm, device, synthetic_pair,
):
    ctrl = _ctrl(fresh_ritm, device)
    img, gt = synthetic_pair
    ctrl.set_image(img)
    ctrl.online_recording = True

    ys, xs = np.where(gt)
    ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
    ctrl.finish_object()
    assert len(ctrl.replay_buffer) == 1


def test_finish_object_does_not_push_when_recording_off(
    fresh_ritm, device, synthetic_pair,
):
    ctrl = _ctrl(fresh_ritm, device)
    img, gt = synthetic_pair
    ctrl.set_image(img)
    ctrl.online_recording = False

    ys, xs = np.where(gt)
    ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
    ctrl.finish_object()
    assert len(ctrl.replay_buffer) == 0


# ---------------------------------------------------------------------------
# auto_segment routing
# ---------------------------------------------------------------------------

def test_auto_segment_no_image_returns_failure(fresh_ritm, device):
    ctrl = _ctrl(fresh_ritm, device)
    res = ctrl.auto_segment(mode='auto')
    assert res['ok'] is False and 'no image' in res['reason']


def test_auto_segment_routes_ritm_to_center_click(
    fresh_ritm, device, synthetic_pair,
):
    ctrl = _ctrl(fresh_ritm, device)
    img, _gt = synthetic_pair
    ctrl.set_image(img)
    res = ctrl.auto_segment(mode='auto')
    assert res['ok'] and res['mode_used'] == 'center_click'
    assert len(ctrl.clicker.clicks_list) == 1
    c = ctrl.clicker.clicks_list[0]
    H, W = img.shape[:2]
    assert c.coords == (H // 2, W // 2)
    assert c.is_positive


def test_auto_segment_auto_mask_unsupported_for_ritm(
    fresh_ritm, device, synthetic_pair,
):
    ctrl = _ctrl(fresh_ritm, device)
    img, _ = synthetic_pair
    ctrl.set_image(img)
    res = ctrl.auto_segment(mode='auto_mask')
    assert res['ok'] is False
    assert 'auto_mask not supported' in res['reason']


def test_auto_segment_unknown_mode_returns_failure(
    fresh_ritm, device, synthetic_pair,
):
    ctrl = _ctrl(fresh_ritm, device)
    img, _ = synthetic_pair
    ctrl.set_image(img)
    res = ctrl.auto_segment(mode='nonsense')
    assert res['ok'] is False
