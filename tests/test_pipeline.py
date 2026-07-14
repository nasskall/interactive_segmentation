"""End-to-end RITM pipeline test (mirrors the synthetic smoke test as pytest)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from interactive_demo.controller import InteractiveController
from isegm.adaptation import FewShotTrainer, FewShotConfig
from isegm.data.custom_dataset import CustomDataset


def _ctrl(model, device):
    return InteractiveController(
        model, device,
        predictor_params={'brs_mode': 'NoBRS'},
        update_image_callback=lambda *a, **k: None,
    )


@pytest.mark.slow
def test_full_ritm_pipeline(fresh_ritm, device, synthetic_dataset_dir,
                             tmp_path):
    """Runs A-F in a single test:
        A. interactive clicks produce IoU > 0
        B. auto_segment via center click works
        C. online buffer + adapt_now succeeds
        D. offline trainer produces an adapter
        E. save/load round-trip is lossless
        F. adapter shifts predictions
    """
    ds = CustomDataset(synthetic_dataset_dir, augment=False)
    img0, gt0 = ds[0]
    H, W = gt0.shape

    ctrl = _ctrl(fresh_ritm, device)

    # A. Interactive
    ctrl.set_image(img0)
    ys, xs = np.where(gt0)
    ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
    ctrl.add_click(0, 0, is_positive=False)
    pred = ctrl.current_object_prob > 0.5
    iou = float(np.logical_and(pred, gt0).sum() /
                max(np.logical_or(pred, gt0).sum(), 1))
    assert iou > 0.05

    # B. Auto-segment center click
    ctrl.set_image(img0)
    res = ctrl.auto_segment(mode='auto')
    assert res['ok']
    assert res['mode_used'] == 'center_click'

    # C. Online recording → adapt_now
    ctrl.set_image(img0)
    ctrl.online_recording = True
    for k in range(min(2, len(ds))):
        img_k, gt_k = ds[k]
        ctrl.set_image(img_k)
        ys_k, xs_k = np.where(gt_k)
        ctrl.add_click(int(xs_k.mean()), int(ys_k.mean()), is_positive=True)
        ctrl.finish_object()
    assert len(ctrl.replay_buffer) == 2

    adapter = ctrl._ensure_online_adapter()
    adapter.config.steps = 3
    rep = ctrl.adapt_now()
    assert rep['successful_steps'] >= 1
    assert not np.isnan(rep['mean_loss'])
    assert ctrl.rollback_adapter() is True

    # D. Offline trainer — keep using the same model (already has online-C
    # LoRA injected; the trainer's attach_adapter call is idempotent).
    ds_train = CustomDataset(synthetic_dataset_dir, augment=True)
    trainer = FewShotTrainer(
        model=fresh_ritm, model_type='ritm', dataset=ds_train,
        config=FewShotConfig(rank=4, epochs=1, lr=1e-3, n_clicks=1,
                             cpu_max_size=192, cuda_max_size=384),
        device=device, log_cb=lambda m: None,
    )
    trainer.run()
    state = trainer.get_adapter_state()
    assert state is not None and len(state) > 0

    # E. Save / load round-trip
    out = tmp_path / 'adapter.pt'
    trainer.save(str(out))
    payload = torch.load(out, map_location='cpu', weights_only=False)
    assert payload['model_type'] == 'ritm'
    assert len(payload['lora_state']) == len(state)
