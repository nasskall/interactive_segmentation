"""
Few-shot domain adaptation for foundation segmentation models.

Two modes are exposed:
  - Offline (Mode A): train a LoRA adapter from a labeled (image, mask) set.
  - Online  (Mode B): manually-triggered SGD steps over a session replay
                       buffer of (image, clicks, accepted_mask).

Adapters are saved as small ``.pt`` files containing only the LoRA delta
weights, so the base model checkpoint stays untouched.
"""

from .lora import (
    LoRAConfig,
    LoRALinear,
    LoRAConv2d,
    inject_lora,
    collect_lora_state,
    load_lora_state,
    set_lora_trainable,
    lora_parameters,
    remove_lora,
)
from .replay_buffer import ReplayBuffer, BufferEntry
from .presets import attach_adapter, build_config, DEFAULT_RANKS
from .few_shot_trainer import FewShotTrainer, FewShotConfig
from .online_adapter import OnlineAdapter, OnlineAdaptConfig
