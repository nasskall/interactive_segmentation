"""
Generic LoRA (Low-Rank Adaptation) primitives.

Wraps ``nn.Linear`` and ``nn.Conv2d`` modules with a frozen base weight
plus a trainable low-rank delta ``B @ A * (alpha / rank)``. Adapters are
injected by walking ``named_modules()`` and swapping target leaves in
their parent's ``_modules`` dict.

State dict for an adapter contains only the LoRA tensors, so saved
adapter files are tiny (typically << 50 MB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    # Substring filters applied to the *fully qualified* module name. A
    # module is wrapped only if at least one substring in ``target_substrings``
    # matches AND no substring in ``exclude_substrings`` matches.
    target_substrings: Sequence[str] = field(default_factory=tuple)
    exclude_substrings: Sequence[str] = field(default_factory=tuple)
    # If True, also wrap nn.Conv2d (kernel size 1 or 3). Otherwise Linear only.
    include_conv2d: bool = True


# ---------------------------------------------------------------------------
# Wrapped modules
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """``y = x W^T + b + (x A^T) B^T * scale``  with A, B trainable."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B starts at zero so the adapter is the identity at init.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + delta * self.scale


class LoRAConv2d(nn.Module):
    """
    LoRA for nn.Conv2d. Uses a 1x1 down-projection (in -> rank) followed by
    a ksize up-projection (rank -> out) to approximate the original kernel
    delta with O(rank * (in + out * k * k)) parameters.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Down: 1x1 conv reducing channels to rank. Stride/padding/dilation
        # for the spatial kernel live in the Up branch so the output spatial
        # shape matches the base conv exactly.
        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            groups=1,
        )
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
            groups=1,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return out + delta * self.scale


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _name_matches(name: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if includes and not any(s in name for s in includes):
        return False
    if any(s in name for s in excludes):
        return False
    return True


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace ``root.<qualified_name>`` with ``new_module`` in-place."""
    parts = qualified_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    parent._modules[parts[-1]] = new_module


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """
    Wrap matching Linear/Conv2d modules with LoRA layers, in place.

    Idempotent: if a module is already wrapped, it (and all its descendants
    — `base`, `lora_down`, `lora_up`, etc.) are left untouched. The returned
    list reports *all* LoRA-wrapped modules in the model, not just the ones
    wrapped on this call.
    """
    # Snapshot first; mutating modules while iterating named_modules is unsafe.
    targets: list[tuple[str, nn.Module]] = list(model.named_modules())

    # Names of modules that are already LoRA wrappers. We must skip not only
    # them but also any submodules underneath (their .base / .lora_down /
    # .lora_up children, which would otherwise re-match the target filter
    # and corrupt the wrapper structure).
    lora_prefixes = [name for name, mod in targets
                     if name and isinstance(mod, (LoRALinear, LoRAConv2d))]

    def _under_lora(name: str) -> bool:
        return any(name == p or name.startswith(p + '.') for p in lora_prefixes)

    wrapped: list[str] = list(lora_prefixes)

    for name, mod in targets:
        if not name:
            continue
        if isinstance(mod, (LoRALinear, LoRAConv2d)):
            continue
        if _under_lora(name):
            continue
        if not _name_matches(name, config.target_substrings, config.exclude_substrings):
            continue

        if isinstance(mod, nn.Linear):
            new = LoRALinear(mod, config.rank, config.alpha, config.dropout)
            _set_submodule(model, name, new)
            wrapped.append(name)
        elif isinstance(mod, nn.Conv2d) and config.include_conv2d:
            # Skip depthwise / grouped convs — the rank-reduction trick only
            # makes sense for groups == 1. They still receive zero-grad as
            # part of the frozen base.
            if mod.groups != 1:
                continue
            new = LoRAConv2d(mod, config.rank, config.alpha, config.dropout)
            _set_submodule(model, name, new)
            wrapped.append(name)

    return wrapped


def remove_lora(model: nn.Module) -> int:
    """Restore base modules in place, removing all LoRA wrappers. Returns count."""
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (LoRALinear, LoRAConv2d))]
    for name, wrapper in targets:
        _set_submodule(model, name, wrapper.base)
    return len(targets)


# ---------------------------------------------------------------------------
# Trainability + state I/O
# ---------------------------------------------------------------------------

def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only the LoRA delta parameters (everything else stays frozen)."""
    params: list[nn.Parameter] = []
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            params.extend([mod.lora_A, mod.lora_B])
        elif isinstance(mod, LoRAConv2d):
            params.extend([mod.lora_down.weight, mod.lora_up.weight])
    return params


def set_lora_trainable(model: nn.Module, trainable: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if trainable:
        for p in lora_parameters(model):
            p.requires_grad = True


def collect_lora_state(model: nn.Module) -> dict:
    """Extract LoRA tensors keyed by ``<module_name>.<param>``."""
    state: dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            state[f'{name}.lora_A'] = mod.lora_A.detach().cpu().clone()
            state[f'{name}.lora_B'] = mod.lora_B.detach().cpu().clone()
            state[f'{name}.scale'] = torch.tensor(mod.scale)
        elif isinstance(mod, LoRAConv2d):
            state[f'{name}.lora_down.weight'] = mod.lora_down.weight.detach().cpu().clone()
            state[f'{name}.lora_up.weight'] = mod.lora_up.weight.detach().cpu().clone()
            state[f'{name}.scale'] = torch.tensor(mod.scale)
    return state


def load_lora_state(model: nn.Module, state: dict, strict: bool = False) -> tuple[list[str], list[str]]:
    """
    Load LoRA tensors back into wrapped modules. Returns (missing, unexpected).

    The model must already have LoRA injected with a matching topology
    (run ``inject_lora`` with the same config before loading).
    """
    missing: list[str] = []
    consumed: set = set()

    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            for key, tensor in (('lora_A', mod.lora_A), ('lora_B', mod.lora_B)):
                k = f'{name}.{key}'
                if k in state:
                    with torch.no_grad():
                        tensor.copy_(state[k].to(tensor.device))
                    consumed.add(k)
                else:
                    missing.append(k)
            consumed.add(f'{name}.scale')   # informational, ignore if absent
        elif isinstance(mod, LoRAConv2d):
            for key, tensor in (
                ('lora_down.weight', mod.lora_down.weight),
                ('lora_up.weight',   mod.lora_up.weight),
            ):
                k = f'{name}.{key}'
                if k in state:
                    with torch.no_grad():
                        tensor.copy_(state[k].to(tensor.device))
                    consumed.add(k)
                else:
                    missing.append(k)
            consumed.add(f'{name}.scale')

    unexpected = [k for k in state.keys() if k not in consumed and not k.endswith('.scale')]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"LoRA load failed. Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}; "
            f"Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    return missing, unexpected
