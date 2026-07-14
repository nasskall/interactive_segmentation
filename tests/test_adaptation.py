"""End-to-end tests for FewShotTrainer + OnlineAdapter on RITM."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isegm.adaptation import FewShotTrainer, FewShotConfig, OnlineAdapter, OnlineAdaptConfig, ReplayBuffer
from isegm.adaptation.lora import collect_lora_state, lora_parameters


# ---------------------------------------------------------------------------
# Few-shot trainer
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_few_shot_trainer_runs_and_produces_state(
    fresh_ritm, device, synthetic_dataset_dir,
):
    from isegm.data.custom_dataset import CustomDataset
    ds = CustomDataset(synthetic_dataset_dir, augment=True)
    trainer = FewShotTrainer(
        model=fresh_ritm,
        model_type='ritm',
        dataset=ds,
        config=FewShotConfig(rank=4, epochs=1, lr=1e-3, n_clicks=1,
                             cpu_max_size=192, cuda_max_size=384),
        device=device,
        log_cb=lambda m: None,
    )
    trainer.run()
    state = trainer.get_adapter_state()
    assert state is not None and len(state) > 0
    assert len(trainer.get_wrapped_module_names()) > 0


@pytest.mark.slow
def test_few_shot_trainer_saves_loadable_adapter(
    fresh_ritm, device, synthetic_dataset_dir, tmp_path,
):
    from isegm.data.custom_dataset import CustomDataset
    ds = CustomDataset(synthetic_dataset_dir, augment=True)
    trainer = FewShotTrainer(
        model=fresh_ritm, model_type='ritm', dataset=ds,
        config=FewShotConfig(rank=4, epochs=1, lr=1e-3, n_clicks=1),
        device=device, log_cb=lambda m: None,
    )
    trainer.run()
    out = tmp_path / 'adapter.pt'
    trainer.save(str(out))
    payload = torch.load(out, map_location='cpu', weights_only=False)
    assert payload['model_type'] == 'ritm'
    assert payload['rank'] == 4
    assert len(payload['lora_state']) > 0


def test_few_shot_trainer_save_without_run_raises(fresh_ritm, device,
                                                    synthetic_dataset_dir,
                                                    tmp_path):
    from isegm.data.custom_dataset import CustomDataset
    ds = CustomDataset(synthetic_dataset_dir, augment=False)
    trainer = FewShotTrainer(
        model=fresh_ritm, model_type='ritm', dataset=ds,
        config=FewShotConfig(rank=4, epochs=1),
        device=device, log_cb=lambda m: None,
    )
    with pytest.raises(RuntimeError):
        trainer.save(str(tmp_path / 'x.pt'))


def test_few_shot_trainer_stop_signal(fresh_ritm, device,
                                       synthetic_dataset_dir):
    from isegm.data.custom_dataset import CustomDataset
    ds = CustomDataset(synthetic_dataset_dir, augment=False)
    trainer = FewShotTrainer(
        model=fresh_ritm, model_type='ritm', dataset=ds,
        config=FewShotConfig(rank=4, epochs=10, lr=1e-3, n_clicks=1),
        device=device, log_cb=lambda m: None,
    )
    trainer.stop()
    trainer.run()  # should bail out fast — early stop catches the flag
    # No assertion needed beyond "did not hang"


# ---------------------------------------------------------------------------
# Online adapter
# ---------------------------------------------------------------------------

def test_online_adapter_empty_buffer(fresh_ritm, device):
    adapter = OnlineAdapter(
        model=fresh_ritm, model_type='ritm', device=device,
        config=OnlineAdaptConfig(steps=2),
        log_cb=lambda m: None,
    )
    rep = adapter.adapt(ReplayBuffer(capacity=4))
    assert rep['steps'] == 0
    assert np.isnan(rep['mean_loss'])


@pytest.mark.slow
def test_online_adapter_runs_and_returns_loss(fresh_ritm, device,
                                                synthetic_pair):
    img, gt = synthetic_pair
    buf = ReplayBuffer(capacity=4)
    buf.add(img, [(50, 50, True)], gt)
    buf.add(img, [(60, 60, True)], gt)

    adapter = OnlineAdapter(
        model=fresh_ritm, model_type='ritm', device=device,
        config=OnlineAdaptConfig(steps=3, lr=1e-3),
        log_cb=lambda m: None,
    )
    rep = adapter.adapt(buf)
    assert rep['steps'] == 3
    assert rep.get('successful_steps', 0) >= 1
    assert not np.isnan(rep['mean_loss'])


@pytest.mark.slow
def test_online_adapter_snapshot_and_rollback(fresh_ritm, device,
                                                synthetic_pair):
    img, gt = synthetic_pair
    buf = ReplayBuffer(capacity=4)
    buf.add(img, [(50, 50, True)], gt)

    adapter = OnlineAdapter(
        model=fresh_ritm, model_type='ritm', device=device,
        config=OnlineAdaptConfig(steps=2, lr=1e-2),
        log_cb=lambda m: None,
    )
    adapter.ensure_injected()
    pre = collect_lora_state(adapter.model)
    adapter.adapt(buf)
    post = collect_lora_state(adapter.model)

    # Adaptation should have changed at least one tensor.
    changed = any(
        not torch.allclose(pre[k], post[k])
        for k in pre if not k.endswith('.scale')
    )
    assert changed

    assert adapter.rollback() is True
    after_rollback = collect_lora_state(adapter.model)
    for k, v in pre.items():
        assert torch.allclose(after_rollback[k], v), f'rollback failed on {k}'


def test_online_adapter_reset_zeroes_b(fresh_ritm, device):
    adapter = OnlineAdapter(
        model=fresh_ritm, model_type='ritm', device=device,
        config=OnlineAdaptConfig(),
        log_cb=lambda m: None,
    )
    adapter.ensure_injected()
    # Perturb LoRA params to make sure reset really resets.
    for p in lora_parameters(adapter.model):
        with torch.no_grad():
            p.add_(0.1)
    adapter.reset()
    # After reset, lora_B / lora_up.weight should be zero so the adapter is
    # the identity again.
    from isegm.adaptation.lora import LoRALinear, LoRAConv2d
    for mod in adapter.model.modules():
        if isinstance(mod, LoRALinear):
            assert torch.allclose(mod.lora_B, torch.zeros_like(mod.lora_B))
        elif isinstance(mod, LoRAConv2d):
            assert torch.allclose(
                mod.lora_up.weight, torch.zeros_like(mod.lora_up.weight)
            )
