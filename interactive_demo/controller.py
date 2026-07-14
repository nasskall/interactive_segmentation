from typing import Callable, Optional

import torch
import numpy as np
from tkinter import messagebox

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
from isegm.adaptation import (
    OnlineAdapter, OnlineAdaptConfig, ReplayBuffer,
)
from isegm.adaptation.lora import collect_lora_state, load_lora_state


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params

        # ── Adaptation state ────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(capacity=32)
        self.online_recording: bool = False
        self._online_adapter: Optional[OnlineAdapter] = None
        self._adapter_path: Optional[str] = None
        self._adapter_log_cb: Optional[Callable[[str], None]] = None

        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        # Capture for online adaptation buffer before state is reset.
        if self.online_recording and self.image is not None:
            clicks_payload = [
                (int(c.coords[0]), int(c.coords[1]), bool(c.is_positive))
                for c in self.clicker.clicks_list
            ]
            accepted = self.current_object_prob > self.prob_thresh
            self.replay_buffer.add(self.image, clicks_payload, accepted)

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    # ──────────────────────────────────────────────────────────────────────
    # Adaptation API (single active-adapter slot per model_type)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def model_type(self) -> str:
        return getattr(self.net, '_model_type', 'ritm')

    def set_net(self, model) -> None:
        """Swap the underlying network and clear adapter state.
        Called by the UI when the user picks a new model_type or weight file.
        """
        self.net = model
        self._online_adapter = None
        self._adapter_path = None
        self.replay_buffer.clear()
        self.reset_predictor()

    def set_adapter_log_cb(self, cb: Optional[Callable[[str], None]]) -> None:
        self._adapter_log_cb = cb
        if self._online_adapter is not None:
            self._online_adapter.log_cb = cb

    def _ensure_online_adapter(self) -> OnlineAdapter:
        if self._online_adapter is None:
            self._online_adapter = OnlineAdapter(
                model=self.net,
                model_type=self.model_type,
                device=self.device,
                config=OnlineAdaptConfig(),
                log_cb=self._adapter_log_cb,
            )
            self._online_adapter.ensure_injected()
        return self._online_adapter

    def adapt_now(self) -> dict:
        """Manually-triggered online adapt step over the replay buffer."""
        adapter = self._ensure_online_adapter()
        sam_p = self.predictor if self.model_type == 'sam' else None
        sam2_p = self.predictor if self.model_type == 'sam2' else None
        report = adapter.adapt(self.replay_buffer,
                               sam_predictor=sam_p, sam2_predictor=sam2_p)
        # The predictor caches features that are now stale relative to the
        # updated mask decoder; rebuild it on the current image.
        self.reset_predictor()
        return report

    def rollback_adapter(self) -> bool:
        if self._online_adapter is None:
            return False
        ok = self._online_adapter.rollback()
        if ok:
            self.reset_predictor()
        return ok

    def reset_adapter(self) -> None:
        if self._online_adapter is not None:
            self._online_adapter.reset()
            self.reset_predictor()

    def save_adapter(self, path: str) -> None:
        if self._online_adapter is None:
            raise RuntimeError("No active adapter to save.")
        import torch as _torch
        _torch.save(self._online_adapter.state_dict(), path)
        self._adapter_path = path

    def load_adapter(self, path: str) -> tuple[list, list]:
        import torch as _torch
        payload = _torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('model_type') != self.model_type:
            raise ValueError(
                f"Adapter is for model_type={payload.get('model_type')!r}, "
                f"current model_type={self.model_type!r}"
            )
        adapter = self._ensure_online_adapter()
        result = adapter.load_state_dict(payload)
        self._adapter_path = path
        self.reset_predictor()
        return result

    def apply_offline_adapter(self, lora_state: dict) -> None:
        """Called by the offline trainer dialog after a successful run.
        The trainer has already injected LoRA on ``self.net`` in place; we
        just rebuild the predictor and remember the slot is occupied."""
        adapter = self._ensure_online_adapter()
        # Trainer wrote new tensors into the same wrapped modules — sync
        # snapshots so the next "Adapt now" can be rolled back independently.
        adapter._snapshots.clear()
        self.reset_predictor()

    def adapter_status(self) -> str:
        if self._online_adapter is None or not self._online_adapter._injected:
            return 'no adapter'
        wrapped = len(self._online_adapter._wrapped)
        snaps = len(self._online_adapter._snapshots)
        path = self._adapter_path
        if path:
            from os.path import basename
            return f'{basename(path)} ({wrapped} layers, {snaps} snaps)'
        return f'unsaved ({wrapped} layers, {snaps} snaps)'

    # ──────────────────────────────────────────────────────────────────────
    # Auto-segmentation (no interactive prompts)
    # ──────────────────────────────────────────────────────────────────────

    def auto_segment(self, mode: str = 'auto') -> dict:
        """
        Produce a segmentation mask without explicit user clicks.

        mode='auto'         — auto-mask-generator for SAM/SAM2, center
                              click for RITM/SimpleClick.
        mode='center_click' — single positive click at the image centre
                              (works for any model).
        mode='auto_mask'    — SAM/SAM2 only; raises otherwise.

        The result enters ``probs_history`` exactly like a regular click
        prediction, so the user can still refine it before pressing Finish.

        Returns
        -------
        dict with keys: ok (bool), mode_used (str), reason (str | None).
        """
        if self.image is None:
            return {'ok': False, 'mode_used': mode, 'reason': 'no image loaded'}

        mt = self.model_type
        chosen = mode
        if mode == 'auto':
            chosen = 'auto_mask' if mt in ('sam', 'sam2') else 'center_click'

        if chosen == 'center_click':
            self.reset_last_object(update_image=False)
            h, w = self.image.shape[:2]
            self.add_click(w // 2, h // 2, is_positive=True)
            return {'ok': True, 'mode_used': 'center_click', 'reason': None}

        if chosen == 'auto_mask':
            if mt not in ('sam', 'sam2'):
                return {'ok': False, 'mode_used': chosen,
                        'reason': f'auto_mask not supported for model_type={mt!r}'}
            return self._auto_mask_generate(mt)

        return {'ok': False, 'mode_used': chosen, 'reason': f'unknown mode {mode!r}'}

    def _auto_mask_generate(self, mt: str) -> dict:
        """SAM / SAM2 automatic mask generator path. Picks the largest
        returned mask (typical heuristic for a single dominant object)."""
        try:
            if mt == 'sam':
                from segment_anything import SamAutomaticMaskGenerator
                generator = SamAutomaticMaskGenerator(self.net)
            else:  # sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                generator = SAM2AutomaticMaskGenerator(self.net)
        except ImportError as exc:
            return {'ok': False, 'mode_used': 'auto_mask',
                    'reason': f'import failed: {exc}'}

        try:
            results = generator.generate(self.image)
        except Exception as exc:
            return {'ok': False, 'mode_used': 'auto_mask',
                    'reason': f'generator failed: {exc}'}

        if not results:
            return {'ok': False, 'mode_used': 'auto_mask',
                    'reason': 'no objects returned'}

        # Largest mask by area is the standard heuristic for a dominant
        # foreground object (e.g., a single skin lesion).
        best = max(results, key=lambda m: m.get('area', m['segmentation'].sum()))
        prob_map = best['segmentation'].astype(np.float32)

        self.reset_last_object(update_image=False)
        # Mirror what add_click does so probs_history + states stay in sync.
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states(),
        })
        self.probs_history.append((np.zeros_like(prob_map), prob_map))
        self.update_image_callback()

        return {'ok': True, 'mode_used': 'auto_mask', 'reason': None,
                'n_candidates': len(results)}

    # ──────────────────────────────────────────────────────────────────────

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
