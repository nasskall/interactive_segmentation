"""
Online (manually-triggered) adapter updates from the session replay buffer.

User flow:
  1. Toggle "Online adapt from clicks" → buffer recording on.
  2. Segment images normally; each ``finish_object`` pushes a buffer entry.
  3. Click "Adapt now" → run K SGD steps against the buffer with the same
     LoRA config used by ``few_shot_trainer``. Each step replays the user's
     own recorded clicks for one object (a random prefix of the sequence,
     see ``few_shot_trainer._click_rounds``) and trains towards the mask
     the user accepted. State is snapshotted before stepping so "Rollback
     last" can revert.

No automatic adaptation — every update is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .few_shot_trainer import (
    _scale_clicks,
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
    # Train on the user's recorded clicks; False samples clicks from the
    # accepted mask instead (the earlier behaviour, kept for comparison).
    use_recorded_clicks: bool = True

    @classmethod
    def for_model(cls, model_type: str) -> 'OnlineAdaptConfig':
        """Defaults tuned for ``model_type`` (see MODEL_DEFAULTS)."""
        return cls(**MODEL_DEFAULTS.get(model_type, {}))


# Per-model overrides of OnlineAdaptConfig, from tools/click_benchmark.py
# --sweep-lr on COCO128-seg people (8 session objects, 12 held out):
#   sam: lr 2e-4 x 30 steps was the only setting no worse than the
#        unadapted model on any measure (NoC@90 7.33 -> 5.33, mIoU@1
#        0.33 -> 0.51); the old 5e-4 x 90 steps made it worse (NoC@90 9.08).
#   ritm: (lesion checkpoint) no setting was a clear win on photos of people.
#        lr 2e-4 is the safe choice: one 30-step press is about neutral
#        (NoC@90 7.42 -> 7.33), three reach the best measured setting
#        (NoC@90 6.50, mIoU@1 0.577 -> 0.547), whereas 5e-4 degraded with
#        repeated presses (mIoU@1 0.48 after 90 steps). Applies to every RITM
#        checkpoint; only the lesion one was swept.
#   sam2: lr 2e-4 x 30 steps gave the best NoC@85/90 (3.08/5.42 -> 2.67/4.50,
#        mIoU@1 0.62 -> 0.72); 5e-4 x 30 reached only NoC@90 4.92.
#   simpleclick: adaptation never clearly helped (sessions are short, about
#        2 clicks per object, so it over-fits); 5e-4 x 30 made it much worse
#        (NoC@90 3.75 -> 6.83). lr 1e-4 x 10 steps is the only neutral setting
#        (NoC@90 3.67), so a press barely changes the model.
MODEL_DEFAULTS: dict[str, dict] = {
    'sam': {'lr': 2e-4, 'steps': 30},
    'sam2': {'lr': 2e-4, 'steps': 30},
    'ritm': {'lr': 2e-4, 'steps': 30},
    'simpleclick': {'lr': 1e-4, 'steps': 10},
}


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
            user_clicks = None
            if self.config.use_recorded_clicks and entry.clicks:
                user_clicks = _scale_clicks(entry.clicks, entry.shape(), gt_np.shape)

            optimizer.zero_grad()
            cfg_for_step = _StepConfig(
                n_clicks=1, n_pos=3, n_neg=3,
                bce_weight=self.config.bce_weight,
                iou_weight=self.config.iou_weight,
            )
            try:
                if self.model_type in ('ritm', 'simpleclick'):
                    step_losses = _step_ritm_family(self.model, image_np, gt_np,
                                                    cfg_for_step, self.device, user_clicks)
                elif self.model_type == 'sam':
                    step_losses = _step_sam(sam_predictor, image_np, gt_np,
                                            cfg_for_step, self.device, user_clicks)
                elif self.model_type == 'sam2':
                    step_losses = _step_sam2(sam2_predictor, image_np, gt_np,
                                             cfg_for_step, self.device, user_clicks)
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


class EnsembleAdapter:
    """One ``OnlineAdapter`` per ensemble member, behind the same interface.

    Each member is trained on the user's own clicks towards the accepted
    mask with its own ``OnlineAdaptConfig.for_model`` defaults, as if it
    were the only model; the members are then fused as before. A press
    adapts every member and a rollback undoes the whole press. A net shared
    by two members is adapted once.
    """

    model_type = 'ensemble'

    def __init__(self, ensemble, device: str, log_cb: Optional[Callable] = None):
        self.members: list[tuple[str, OnlineAdapter]] = []
        seen: set[int] = set()
        for (name, net), mt in zip(ensemble.members, ensemble.member_types):
            if id(net) in seen:
                continue
            seen.add(id(net))
            self.members.append((name, OnlineAdapter(
                model=net, model_type=mt, device=device,
                config=OnlineAdaptConfig.for_model(mt), log_cb=log_cb)))

    @property
    def log_cb(self) -> Optional[Callable]:
        return self.members[0][1].log_cb

    @log_cb.setter
    def log_cb(self, cb: Optional[Callable]) -> None:
        for _, a in self.members:
            a.log_cb = cb

    @property
    def _injected(self) -> bool:
        return all(a._injected for _, a in self.members)

    @property
    def _wrapped(self) -> list[str]:
        return [f'{name}.{w}' for name, a in self.members for w in a._wrapped]

    @property
    def _snapshots(self) -> list:
        # Every member pushes one snapshot per press.
        return self.members[0][1]._snapshots

    def ensure_injected(self) -> None:
        for _, a in self.members:
            a.ensure_injected()

    def reset(self) -> None:
        for _, a in self.members:
            a.reset()

    def rollback(self) -> bool:
        if not self._snapshots:
            return False
        for _, a in self.members:
            a.rollback()
        return True

    def adapt(self, buffer: ReplayBuffer, ensemble_predictor) -> dict:
        """Adapt every member; ``ensemble_predictor`` supplies the SAM / SAM 2
        member predictors. Returns the per-member reports under ``members``
        plus their totals."""
        predictors = {m.name: m.predictor for m in ensemble_predictor.members}
        reports = {}
        for name, a in self.members:
            p = predictors.get(name)
            reports[name] = a.adapt(buffer,
                                    sam_predictor=p if a.model_type == 'sam' else None,
                                    sam2_predictor=p if a.model_type == 'sam2' else None)
        losses = [r['mean_loss'] for r in reports.values() if np.isfinite(r['mean_loss'])]
        return {
            'steps': sum(r['steps'] for r in reports.values()),
            'successful_steps': sum(r.get('successful_steps', 0) for r in reports.values()),
            'mean_loss': float(np.mean(losses)) if losses else float('nan'),
            'failures': [f'{n}: {f}' for n, r in reports.items() for f in r.get('failures', [])],
            'members': reports,
        }

    def state_dict(self) -> dict:
        return {'model_type': 'ensemble',
                'members': {name: a.state_dict() for name, a in self.members}}

    def load_state_dict(self, payload: dict) -> tuple[list, list]:
        missing, unexpected = [], []
        for name, a in self.members:
            if name in payload['members']:
                m, u = a.load_state_dict(payload['members'][name])
                missing += m
                unexpected += u
            else:
                missing.append(name)
        return missing, unexpected


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
