"""Per-user segmentation sessions on top of InteractiveController.

All sessions share one network; each has its own controller (clicker,
predictor, mask history). GPU work is serialised by a single lock owned by
the ModelManager, so FastAPI's threadpool can call in safely.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from interactive_demo.controller import InteractiveController
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.inference.predictors.ensemble import FUSIONS, EnsembleConfig, EnsembleNet

ENSEMBLE = 'Ensemble'
MODEL_TYPES = [
    'RITM', 'SimpleClick',
    'SAM - ViT-B', 'SAM - ViT-L', 'SAM - ViT-H',
    'SAM2 - Tiny', 'SAM2 - Small', 'SAM2 - Base+', 'SAM2 - Large',
    ENSEMBLE,
]
SAM_TYPES = {t for t in MODEL_TYPES if t.startswith('SAM')}
CHECKPOINT_EXTS = ('.pth', '.pt')
MAX_ENSEMBLE_MEMBERS = 4


def guess_model_type(checkpoint: str) -> Optional[str]:
    """Model type a checkpoint file name suggests, used to offer ensemble
    members; None if it doesn't look like any supported model."""
    name = checkpoint.lower()
    if name.startswith('sam2'):
        for key, label in (('tiny', 'SAM2 - Tiny'), ('small', 'SAM2 - Small'),
                           ('base', 'SAM2 - Base+'), ('large', 'SAM2 - Large')):
            if key in name:
                return label
        return None
    for key, label in (('vit_b', 'SAM - ViT-B'), ('vit_l', 'SAM - ViT-L'),
                       ('vit_h', 'SAM - ViT-H')):
        if name.startswith('sam_') and key in name:
            return label
    if 'vit' in name and ('cocolvis' in name or 'sbd' in name):
        return 'SimpleClick'
    return 'RITM'

MAX_IMAGE_SIDE = 2048
LIMIT_LONGEST_SIZE = 5000


def predictor_params_for(model_label: str, prob_thresh: float, use_zoom_in: bool) -> dict:
    """Mirrors InteractiveDemoApp._reset_predictor with the desktop defaults."""
    if model_label in SAM_TYPES:
        return {'brs_mode': 'NoBRS'}
    zp = None
    if use_zoom_in:
        zp = {'skip_clicks': -1, 'target_size': (400, 400), 'expansion_ratio': 1.4}
    return {
        'brs_mode': 'NoBRS',
        'prob_thresh': prob_thresh,
        'zoom_in_params': zp,
        'predictor_params': {'net_clicks_limit': None, 'max_size': LIMIT_LONGEST_SIZE},
        'brs_opt_func_params': {'min_iou_diff': 1e-3},
        'lbfgs_params': {'maxfun': 20},
    }


def resize_longest(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    """Same rule as InteractiveDemoApp._load_and_resize_image."""
    h, w = image.shape[:2]
    ratio = w / h
    if w > h:
        nw = min(max_side, w)
        nh = int(nw / ratio)
    else:
        nh = min(max_side, h)
        nw = int(nh * ratio)
    return cv2.resize(image, (nw, nh))


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    pal = rng.integers(64, 256, size=(n, 3), dtype=np.uint8)
    pal[0] = 0
    return pal


_PALETTE = _palette(256)


def encode_png(arr: np.ndarray) -> bytes:
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode('.png', arr)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return buf.tobytes()


@dataclass
class Session:
    id: str
    controller: InteractiveController
    filename: str
    original_size: tuple[int, int]          # (h, w) of the uploaded image
    prob_thresh: float = 0.5
    use_zoom_in: bool = True
    last_used: float = field(default_factory=time.time)

    @property
    def image(self) -> np.ndarray:
        return self.controller.image

    def overlay_png(self) -> bytes:
        """RGBA overlay: finished objects + current object coloured by label,
        transparent elsewhere. The client blends it over the image."""
        mask = self.controller.result_mask
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., :3] = _PALETTE[mask % 256]
        rgba[..., 3] = np.where(mask > 0, 255, 0)
        return encode_png(rgba)

    def full_res_mask_png(self) -> bytes:
        """Label mask resized to the original upload, scaled like the desktop
        app's Save Mask (labels spread over 0..255)."""
        mask = self.controller.result_mask
        h, w = self.original_size
        out = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if 0 < out.max() < 256:
            out = (out * (255 // out.max())).astype(np.uint8)
        return encode_png(out)

    def state(self) -> dict:
        c = self.controller
        return {
            'session_id': self.id,
            'filename': self.filename,
            'width': int(self.image.shape[1]),
            'height': int(self.image.shape[0]),
            'original_width': int(self.original_size[1]),
            'original_height': int(self.original_size[0]),
            'clicks': [
                {'x': int(k.coords[1]), 'y': int(k.coords[0]), 'positive': bool(k.is_positive)}
                for k in c.clicker.clicks_list
            ],
            'box': getattr(c.predictor, 'box', None),
            'supports_box': c.supports_box,
            # Ensemble only: weight each member got for the latest click.
            'ensemble_weights': (dict(c.predictor.last_weights)
                                 if getattr(c.predictor, 'last_weights', None) else None),
            'object_count': int(c.object_count),
            'has_incomplete_object': bool(c.is_incomplete_mask),
            'can_undo': bool(c.states),
            'prob_thresh': self.prob_thresh,
            'use_zoom_in': self.use_zoom_in,
            'overlay': 'data:image/png;base64,' + base64.b64encode(self.overlay_png()).decode(),
        }


class SessionNotFound(KeyError):
    pass


class ModelManager:
    """Owns the shared network, the GPU lock and the session table."""

    def __init__(self, net, device: str, model_label: str, checkpoint: Optional[str],
                 weights_dir: Path, max_sessions: int = 16):
        self.net = net
        self.device = device
        self.model_label = model_label
        self.checkpoint = checkpoint
        self.weights_dir = Path(weights_dir)
        self.max_sessions = max_sessions
        self.lock = threading.RLock()
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        # Loaded networks by (model type, checkpoint), so switching between a
        # single model and ensembles that use it doesn't reload weights.
        self._nets: dict[tuple[str, str], object] = {}
        if checkpoint is not None:
            self._nets[(model_label, checkpoint)] = net
        self.ensemble_members: Optional[list[tuple[str, str]]] = None

    # ── models ─────────────────────────────────────────────────────────────

    def list_checkpoints(self) -> list[str]:
        if not self.weights_dir.is_dir():
            return []
        return sorted(p.name for p in self.weights_dir.iterdir()
                      if p.is_file() and p.suffix in CHECKPOINT_EXTS)

    def model_info(self) -> dict:
        checkpoints = self.list_checkpoints()
        is_ensemble = self.model_label == ENSEMBLE
        current = {'model_type': self.model_label, 'checkpoint': self.checkpoint,
                   'supports_box': (self.model_label in SAM_TYPES
                                    or (is_ensemble and any(t in SAM_TYPES
                                                            for t, _ in self.ensemble_members)))}
        if is_ensemble:
            current.update(
                members=[{'model_type': t, 'checkpoint': c} for t, c in self.ensemble_members],
                fusion=self.net.config.fusion, tta=self.net.config.tta)
        return {
            'model_types': MODEL_TYPES,
            'checkpoints': checkpoints,
            'current': current,
            'device': self.device,
            'ensemble': {
                'fusions': list(FUSIONS),
                'max_members': MAX_ENSEMBLE_MEMBERS,
                # Checkpoints whose file name says which model they are.
                'candidates': [{'model_type': t, 'checkpoint': c} for c in checkpoints
                               if (t := guess_model_type(c)) is not None],
            },
        }

    def _check_checkpoint(self, model_label: str, checkpoint: str) -> None:
        if model_label not in MODEL_TYPES or model_label == ENSEMBLE:
            raise ValueError(f'Unknown model type {model_label!r}')
        # Only files directly inside weights_dir; no path traversal.
        if os.path.basename(checkpoint) != checkpoint or checkpoint not in self.list_checkpoints():
            raise ValueError(f'Unknown checkpoint {checkpoint!r}')

    def _get_net(self, model_label: str, checkpoint: str):
        key = (model_label, checkpoint)
        if key not in self._nets:
            net = utils.load_model_by_type(str(self.weights_dir / checkpoint), model_label,
                                           self.device)
            # Build a predictor before caching: a checkpoint can load while
            # its predictor still fails (e.g. the backend package is missing),
            # and switching anyway would break every session and new upload.
            get_predictor(net, device=self.device,
                          **predictor_params_for(model_label, 0.5, True))
            self._nets[key] = net
        return self._nets[key]

    def _switch(self, net, model_label: str, checkpoint: Optional[str]) -> None:
        self.net = net
        self.model_label = model_label
        self.checkpoint = checkpoint
        for s in self._sessions.values():
            s.controller.predictor_params = predictor_params_for(
                model_label, s.prob_thresh, s.use_zoom_in)
            s.controller.set_net(net)
            s.controller.reset_last_object(update_image=False)

    def load_model(self, model_label: str, checkpoint: str) -> None:
        if model_label == ENSEMBLE:
            raise ValueError('Load an ensemble with its members')
        self._check_checkpoint(model_label, checkpoint)
        with self.lock:
            net = self._get_net(model_label, checkpoint)
            self.ensemble_members = None
            self._switch(net, model_label, checkpoint)

    def load_ensemble(self, members: list[tuple[str, str]], fusion: str = 'weighted',
                      tta: bool = True) -> None:
        """Fuse 2..MAX_ENSEMBLE_MEMBERS models, given as (model type,
        checkpoint) pairs; see isegm.inference.predictors.ensemble."""
        if not 2 <= len(members) <= MAX_ENSEMBLE_MEMBERS:
            raise ValueError(f'An ensemble needs 2 to {MAX_ENSEMBLE_MEMBERS} models')
        if len(set(members)) != len(members):
            raise ValueError('Each ensemble member must be a different model')
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {', '.join(FUSIONS)}")
        for label, ckpt in members:
            self._check_checkpoint(label, ckpt)
        with self.lock:
            nets = [self._get_net(label, ckpt) for label, ckpt in members]
            labels = [label for label, _ in members]
            names = [label if labels.count(label) == 1 else f'{label} ({ckpt})'
                     for label, ckpt in members]
            net = EnsembleNet(list(zip(names, nets)), EnsembleConfig(fusion=fusion, tta=tta))
            get_predictor(net, device=self.device)
            self.ensemble_members = list(members)
            self._switch(net, ENSEMBLE, None)

    # ── sessions ───────────────────────────────────────────────────────────

    def create_session(self, image_rgb: np.ndarray, filename: str) -> Session:
        original_size = image_rgb.shape[:2]
        image = resize_longest(image_rgb)
        with self.lock:
            ctrl = InteractiveController(
                self.net, self.device,
                predictor_params=predictor_params_for(self.model_label, 0.5, True),
                update_image_callback=lambda *a, **k: None,
            )
            ctrl.set_image(image)
            sid = uuid.uuid4().hex
            session = Session(sid, ctrl, filename, original_size)
            self._sessions[sid] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return session

    def get(self, sid: str) -> Session:
        with self.lock:
            s = self._sessions.get(sid)
            if s is None:
                raise SessionNotFound(sid)
            self._sessions.move_to_end(sid)
            s.last_used = time.time()
            return s

    def has_session(self, sid: str) -> bool:
        with self.lock:
            return sid in self._sessions

    def delete(self, sid: str) -> None:
        with self.lock:
            if self._sessions.pop(sid, None) is None:
                raise SessionNotFound(sid)

    def update_settings(self, s: Session, prob_thresh: Optional[float],
                        use_zoom_in: Optional[bool]) -> None:
        with self.lock:
            # The threshold only affects how the current probabilities are
            # binarised, so it applies live without touching the predictor.
            if prob_thresh is not None:
                s.prob_thresh = prob_thresh
                s.controller.prob_thresh = prob_thresh
            if use_zoom_in is not None and use_zoom_in != s.use_zoom_in:
                s.use_zoom_in = use_zoom_in
                # A new predictor can't continue the old click sequence, so
                # the in-progress object is dropped.
                s.controller.reset_predictor(
                    predictor_params_for(self.model_label, s.prob_thresh, s.use_zoom_in))
                s.controller.reset_last_object(update_image=False)
