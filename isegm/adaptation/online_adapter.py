"""
Online (manually-triggered) adapter updates from the session replay buffer.

User flow:
  1. Toggle "Online adapt from clicks" → buffer recording on.
  2. Segment images normally; each ``finish_object`` pushes a buffer entry.
  3. Click "Adapt now" → run K SGD steps against the buffer with the same
     LoRA config used by ``few_shot_trainer``. State is snapshotted before
     stepping so "Rollback last" can revert.

No automatic adaptation — every update is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .few_shot_trainer import (
    _sample_click_pixels,
    _build_ritm_points_nd,
    _soft_iou,
    _step_ritm_family,
    _step_sam,
    _step_sam2,
)
from .lora import (
    collect_lora_state,
    load_lora_state,
    lora_parameters,
    set_lora_trainable,
)
from .presets import attach_adapter, DEFAULT_RANKS
from .replay_buffer import BufferEntry, ReplayBuffer


@dataclass
class OnlineAdaptConfig:
    rank: int | None = None
    steps: int = 30                 # SGD steps per "Adapt now" press
    lr: float = 5e-4
    bce_weight: float = 1.0
    iou_weight: float = 1.0
    cpu_max_side: int = 384         # extra-conservative on CPU for live UX
    cuda_max_side: int = 768


class OnlineAdapter:
    """Owns the LoRA-injected model + a stack of pre-update snapshots."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        device: str,
        config: OnlineAdaptConfig,
        log_cb: Optional[Callable] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.config = config
        self.log_cb = log_cb

        self._injected = False
        self._wrapped: list[str] = []
        self._snapshots: list[dict] = []  # stack for rollback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_injected(self) -> None:
        if self._injected:
            return
        cfg, wrapped = attach_adapter(self.model, self.model_type, rank=self.config.rank)
        # Newly created LoRA Parameters default to requires_grad=True; flip
        # them off so inference works until adapt() explicitly turns grads on.
        set_lora_trainable(self.model, False)
        self._wrapped = wrapped
        self._injected = True
        self._log(f"Injected LoRA into {len(wrapped)} modules "
                  f"(rank={cfg.rank}).")

    def reset(self) -> None:
        """Re-init LoRA to identity (clears any adaptation done so far)."""
        from .lora import LoRALinear, LoRAConv2d
        self.ensure_injected()
        with torch.no_grad():
            for mod in self.model.modules():
                if isinstance(mod, LoRALinear):
                    torch.nn.init.kaiming_uniform_(mod.lora_A, a=5 ** 0.5)
                    torch.nn.init.zeros_(mod.lora_B)
                elif isinstance(mod, LoRAConv2d):
                    torch.nn.init.kaiming_uniform_(mod.lora_down.weight, a=5 ** 0.5)
                    torch.nn.init.zeros_(mod.lora_up.weight)
        self._snapshots.clear()
        self._log("Online adapter reset to identity.")

    def rollback(self) -> bool:
        if not self._snapshots:
            self._log("Nothing to rollback.")
            return False
        state = self._snapshots.pop()
        load_lora_state(self.model, state, strict=False)
        self._log("Rolled back to previous adapter state.")
        return True

    def state_dict(self) -> dict:
        return {
            'model_type': self.model_type,
            'rank': self.config.rank or DEFAULT_RANKS[self.model_type],
            'wrapped_modules': self._wrapped,
            'lora_state': collect_lora_state(self.model),
        }

    def load_state_dict(self, payload: dict) -> tuple[list, list]:
        self.ensure_injected()
        return load_lora_state(self.model, payload['lora_state'], strict=False)

    # ------------------------------------------------------------------
    # Adaptation step
    # ------------------------------------------------------------------

    def adapt(self, buffer: ReplayBuffer,
              sam_predictor=None, sam2_predictor=None) -> dict:
        """
        Run ``config.steps`` SGD updates over a random sample of buffer
        entries. Returns a small report dict for the UI.

        Predictors are passed per-call because the controller may rebuild
        them between adaptations (e.g., after each ``reset_predictor``).
        """
        if len(buffer) == 0:
            self._log("Replay buffer is empty — nothing to adapt on.")
            return {'steps': 0, 'mean_loss': float('nan')}
        if self.model_type == 'sam' and sam_predictor is None:
            raise ValueError("SAM online adapt requires a sam_predictor")
        if self.model_type == 'sam2' and sam2_predictor is None:
            raise ValueError("SAM2 online adapt requires a sam2_predictor")

        self.ensure_injected()
        self._snapshots.append(collect_lora_state(self.model))

        set_lora_trainable(self.model, True)
        params = lora_parameters(self.model)
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        # Keep base layers (BN/Dropout) in eval; LoRA params are trainable
        # via requires_grad. See few_shot_trainer for rationale.
        self.model.eval()

        max_side = (self.config.cuda_max_side
                    if str(self.device).startswith('cuda')
                    else self.config.cpu_max_side)

        entries = buffer.items()
        losses: list[float] = []
        step_failures: list[str] = []

        for step in range(self.config.steps):
            entry = entries[np.random.randint(len(entries))]
            image_np, gt_np = _resize_pair(entry.image, entry.accepted_mask, max_side)

            optimizer.zero_grad()
            cfg_for_step = _StepConfig(
                n_clicks=1, n_pos=3, n_neg=3,
                bce_weight=self.config.bce_weight,
                iou_weight=self.config.iou_weight,
            )
            try:
                if self.model_type in ('ritm', 'simpleclick'):
                    step_losses = _step_ritm_family(self.model, image_np, gt_np,
                                                    cfg_for_step, self.device)
                elif self.model_type == 'sam':
                    step_losses = _step_sam(sam_predictor, image_np, gt_np,
                                            cfg_for_step, self.device)
                elif self.model_type == 'sam2':
                    step_losses = _step_sam2(sam2_predictor, image_np, gt_np,
                                             cfg_for_step, self.device)
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")
            except Exception as exc:
                msg = f'step {step}: {type(exc).__name__}: {exc}'
                self._log(f"  [warn] {msg}")
                step_failures.append(msg)
                continue

            optimizer.step()
            losses.extend(step_losses)

        # Flip back to inference mode — downstream transforms (e.g. ZoomIn)
        # call .numpy() on the prediction and that fails for grad-tracking
        # tensors.
        set_lora_trainable(self.model, False)
        mean_loss = float(np.mean(losses)) if losses else float('nan')
        successful = self.config.steps - len(step_failures)
        self._log(f"Online adapt: {successful}/{self.config.steps} steps "
                  f"succeeded, mean loss={mean_loss:.4f}.")
        return {
            'steps': self.config.steps,
            'successful_steps': successful,
            'mean_loss': mean_loss,
            'failures': step_failures,
        }

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _StepConfig:
    """Subset of FewShotConfig fields the per-family steps actually read."""
    n_clicks: int
    n_pos: int
    n_neg: int
    bce_weight: float
    iou_weight: float


def _resize_pair(image: np.ndarray, mask: np.ndarray, max_side: int):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, mask.astype(bool)
    import cv2
    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    image_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return image_r, mask_r
