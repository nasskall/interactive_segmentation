"""LoRA primitives + model-type presets."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from isegm.adaptation.lora import (
    LoRAConfig, LoRALinear, LoRAConv2d,
    inject_lora, remove_lora,
    collect_lora_state, load_lora_state,
    lora_parameters, set_lora_trainable,
)
from isegm.adaptation.presets import (
    DEFAULT_RANKS, build_config, attach_adapter,
)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def test_lora_linear_is_identity_at_init():
    base = nn.Linear(10, 8)
    wrapped = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
    x = torch.randn(3, 10)
    assert torch.allclose(wrapped(x), base(x), atol=1e-6), \
        "B is initialised to zero, so the LoRA delta must be zero at init"


def test_lora_conv2d_is_identity_at_init():
    base = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    wrapped = LoRAConv2d(base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(1, 3, 16, 16)
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)


def test_lora_linear_freezes_base_and_only_lora_is_trainable():
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    assert not wrapped.base.weight.requires_grad
    assert not wrapped.base.bias.requires_grad
    assert wrapped.lora_A.requires_grad
    assert wrapped.lora_B.requires_grad


def test_lora_linear_rejects_zero_rank():
    base = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=0, alpha=1.0, dropout=0.0)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 16),
        nn.Linear(16, 4),
    )


def test_inject_lora_wraps_only_matching_modules():
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5', '6'), include_conv2d=False)
    wrapped = inject_lora(m, cfg)
    assert wrapped == ['5', '6']
    assert isinstance(m[5], LoRALinear)
    assert isinstance(m[6], LoRALinear)
    assert isinstance(m[0], nn.Conv2d)


def test_inject_lora_skips_grouped_convs():
    m = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8))
    cfg = LoRAConfig(rank=2, target_substrings=('0',))
    wrapped = inject_lora(m, cfg)
    assert wrapped == [], "depthwise convs should not be wrapped"


def test_inject_lora_is_idempotent():
    """Calling attach twice with the same config must not corrupt wrappers."""
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5', '6'),
                     include_conv2d=False)
    wrapped1 = inject_lora(m, cfg)
    wrapped2 = inject_lora(m, cfg)
    assert sorted(wrapped1) == sorted(wrapped2)
    # The wrappers' inner modules (base / lora_A / lora_B) must stay
    # unwrapped — otherwise lora_parameters() crashes.
    params = lora_parameters(m)
    assert len(params) == 4   # 2 modules x (A, B)


def test_remove_lora_restores_originals():
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5', '6'))
    inject_lora(m, cfg)
    assert isinstance(m[5], LoRALinear)
    n_removed = remove_lora(m)
    assert n_removed == 2
    assert isinstance(m[5], nn.Linear)
    assert isinstance(m[6], nn.Linear)


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_is_lossless():
    torch.manual_seed(0)
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5', '6'),
                     include_conv2d=False)
    inject_lora(m, cfg)
    set_lora_trainable(m, True)

    # Train one step on random targets so tensors aren't trivially zero/init.
    opt = torch.optim.Adam(lora_parameters(m), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(2, 3, 16, 16)
        y = m(x).sum()
        y.backward()
        opt.step()

    state = collect_lora_state(m)
    # Now zero it out, then load — values should match.
    for p in lora_parameters(m):
        with torch.no_grad():
            p.zero_()
    missing, unexpected = load_lora_state(m, state, strict=True)
    assert missing == [] and unexpected == []
    state2 = collect_lora_state(m)
    for k, v in state.items():
        if k.endswith('.scale'):
            continue
        assert torch.allclose(state2[k], v), f'mismatch on {k}'


def test_load_lora_state_strict_rejects_unexpected():
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5',), include_conv2d=False)
    inject_lora(m, cfg)
    bogus = {'5.lora_A': torch.zeros_like(m[5].lora_A),
             '5.lora_B': torch.zeros_like(m[5].lora_B),
             'phantom.lora_A': torch.zeros(4, 4)}
    with pytest.raises(RuntimeError):
        load_lora_state(m, bogus, strict=True)


def test_lora_parameters_excludes_base():
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5',), include_conv2d=False)
    inject_lora(m, cfg)
    params = lora_parameters(m)
    assert len(params) == 2
    # Both should have requires_grad on (default at construction)
    assert all(p.requires_grad for p in params)


def test_set_lora_trainable_freezes_everything_when_false():
    m = _toy_model()
    cfg = LoRAConfig(rank=4, target_substrings=('5',), include_conv2d=False)
    inject_lora(m, cfg)
    set_lora_trainable(m, False)
    assert all(not p.requires_grad for p in m.parameters())


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mt,rank', [
    ('ritm', 4), ('simpleclick', 4), ('sam', 8), ('sam2', 8),
])
def test_default_ranks(mt, rank):
    assert DEFAULT_RANKS[mt] == rank
    cfg = build_config(mt)
    assert cfg.rank == rank


def test_build_config_unknown_model_raises():
    with pytest.raises(ValueError):
        build_config('not-a-model')


def test_attach_adapter_returns_wrapped_names(fresh_ritm):
    cfg, wrapped = attach_adapter(fresh_ritm, 'ritm', rank=4)
    assert cfg.rank == 4
    assert len(wrapped) >= 5, "RITM preset should wrap multiple modules"
    # Sanity: backbone trunk excluded.
    for name in wrapped:
        assert not name.startswith('feature_extractor.layer1')
