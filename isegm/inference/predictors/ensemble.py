"""
Ensemble of interactive segmentation models behind the RITM predictor
interface (set_input_image / get_prediction / get_states / set_states).

Every member sees the same clicks (and box); their probability maps are
fused into one. Fusion methods:

  mean           -- plain average.
  weighted       -- global per-model weights, recomputed after every click
                    from two label-free reliability signals:
                      * click agreement: how well the model's mask honours
                        the user's clicks (positives inside, negatives out);
                      * consensus: IoU of the model's mask with the mean of
                        the other members (outliers lose weight).
                    weights = softmax(sharpness * score).
  weighted_conf  -- ``weighted`` times a per-pixel confidence |2p - 1|, so
                    at each pixel the more certain models count more.
  staple         -- soft STAPLE (Warfield et al., 2004): EM estimate of each
                    model's sensitivity/specificity and the consensus.

Test-time augmentation (``tta=True``):
  * RITM / SimpleClick average the prediction with a horizontally flipped
    pass (their predictor's built-in ``with_flip``).
  * SAM / SAM 2 average predictions over the clicks shifted by a few pixels
    in four directions ("prompt jitter"); it only re-runs the light mask
    decoder, not the image encoder.

``enforce_clicks`` finally pins a small disc around each click to the
user's label, so the result always agrees with what the user clicked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FUSIONS = ('mean', 'weighted', 'weighted_conf', 'staple')
SAM_FAMILY = ('sam', 'sam2')


@dataclass
class EnsembleConfig:
    fusion: str = 'staple'      # best on the COCO click benchmark
    tta: bool = False           # no consistent gain there, and slower
    sharpness: float = 8.0      # softmax temperature for the reliability weights
    jitter: float = 0.01        # SAM prompt jitter, fraction of the longest side
    click_radius: int = 3       # disc used to score / pin clicks, in pixels
    enforce_clicks: bool = True
    staple_iters: int = 15


class EnsembleNet:
    """A bundle of loaded models that ``get_predictor`` turns into an
    ``EnsemblePredictor``. ``members`` are ``(name, net)`` pairs; each net
    carries its ``_model_type`` tag like a single model does."""

    _model_type = 'ensemble'

    def __init__(self, members: list[tuple[str, object]], config: Optional[EnsembleConfig] = None):
        if len(members) < 2:
            raise ValueError('An ensemble needs at least two models')
        self.members = members
        self.config = config or EnsembleConfig()
        if self.config.fusion not in FUSIONS:
            raise ValueError(f'fusion must be one of {FUSIONS}')

    @property
    def member_types(self) -> list[str]:
        return [getattr(net, '_model_type', 'ritm') for _, net in self.members]


@dataclass
class _Member:
    name: str
    model_type: str
    predictor: object
    last_prob: Optional[np.ndarray] = field(default=None, repr=False)


class EnsemblePredictor:
    def __init__(self, ensemble: EnsembleNet, device, **predictor_kwargs):
        from . import get_predictor
        self.config = ensemble.config
        self.device = device
        self.members: list[_Member] = []
        for (name, net), mt in zip(ensemble.members, ensemble.member_types):
            if mt in SAM_FAMILY:
                pred = get_predictor(net, device=device)
                pred.soft = True
            else:
                pred = get_predictor(net, device=device, brs_mode='NoBRS',
                                     with_flip=self.config.tta)
            self.members.append(_Member(name, mt, pred))
        self.source_image = None
        self.image_shape = None
        self.box = None
        self.last_weights: dict[str, float] = {}

    # ── interface used by InteractiveController ──────────────────────────

    def set_input_image(self, image):
        self.source_image = image
        self.image_shape = image.shape[:2]
        self.box = None
        for m in self.members:
            m.predictor.set_input_image(image)
            m.last_prob = None

    def reset_prompts(self):
        """Forget clicks, box and previous masks; keep image embeddings."""
        self.box = None
        for m in self.members:
            if hasattr(m.predictor, 'reset_prompts'):
                m.predictor.reset_prompts()
            else:
                # RITM's per-image state is cheap to rebuild.
                m.predictor.set_input_image(self.source_image)
            m.last_prob = None

    @property
    def supports_box(self) -> bool:
        return any(m.model_type in SAM_FAMILY for m in self.members)

    def set_box(self, box):
        """Box prompt for the members that take one (SAM / SAM 2)."""
        if not self.supports_box:
            raise ValueError('Box prompts need a SAM or SAM 2 model in the ensemble')
        self.box = [float(v) for v in box]
        for m in self.members:
            if hasattr(m.predictor, 'set_box'):
                m.predictor.set_box(box)

    def get_states(self):
        return {'members': [m.predictor.get_states() for m in self.members],
                'box': self.box}

    def set_states(self, states):
        for m, s in zip(self.members, states['members']):
            m.predictor.set_states(s)
        self.box = states.get('box')

    def get_prediction(self, clicker, prev_mask=None) -> np.ndarray:
        clicks = clicker.get_clicks()
        probs, names = [], []
        for m in self.members:
            if m.model_type not in SAM_FAMILY and not clicks:
                continue   # RITM needs a click; a box alone is SAM-only
            if m.model_type in SAM_FAMILY:
                p = self._predict_sam(m, clicker)
            else:
                p = m.predictor.get_prediction(clicker, prev_mask=prev_mask)
            m.last_prob = p
            probs.append(np.asarray(p, np.float32))
            names.append(m.name)
        if not probs:
            return np.zeros(self.image_shape, np.float32)

        fused, weights = fuse(np.stack(probs), clicks, self.config)
        self.last_weights = dict(zip(names, weights))
        if self.config.enforce_clicks:
            pin_clicks(fused, clicks, self.config.click_radius)
        return fused

    # ── SAM prompt-jitter TTA ────────────────────────────────────────────

    def _predict_sam(self, m: _Member, clicker) -> np.ndarray:
        pred = m.predictor
        clicks = clicker.get_clicks()
        if not (self.config.tta and clicks):
            return pred.get_prediction(clicker)
        state = pred.get_states()
        h, w = self.image_shape
        d = max(2, int(round(self.config.jitter * max(h, w))))
        outs = []
        for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d)):
            pred.set_states(state)
            shifted = [c.copy(coords=(int(np.clip(c.coords[0] + dy, 0, h - 1)),
                                      int(np.clip(c.coords[1] + dx, 0, w - 1))))
                       for c in clicks]
            outs.append(pred.get_prediction(_FixedClicks(shifted)))
        # The un-jittered run goes last so the member's mask memory
        # (prev_logit) follows the real clicks.
        pred.set_states(state)
        outs.append(pred.get_prediction(clicker))
        return np.mean(outs, axis=0)


class _FixedClicks:
    """Minimal stand-in for a Clicker holding a fixed click list."""

    def __init__(self, clicks):
        self._clicks = clicks
        self.click_indx_offset = 0

    def get_clicks(self, clicks_limit=None):
        return self._clicks[:clicks_limit]


# ── fusion ────────────────────────────────────────────────────────────────

def _disc_mean(prob: np.ndarray, y: int, x: int, r: int) -> float:
    h, w = prob.shape
    return float(prob[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].mean())


def click_agreement(prob: np.ndarray, clicks, r: int) -> float:
    """1 when the mask is fully on at every positive click and off at every
    negative one; 0.5 for a model that is unsure everywhere."""
    if not clicks:
        return 1.0
    vals = []
    for c in clicks:
        v = _disc_mean(prob, int(c.coords[0]), int(c.coords[1]), r)
        vals.append(v if c.is_positive else 1.0 - v)
    return float(np.mean(vals))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def reliability_weights(probs: np.ndarray, clicks, cfg: EnsembleConfig) -> np.ndarray:
    n = len(probs)
    if n == 1:
        return np.ones(1)
    masks = probs > 0.5
    scores = []
    for i in range(n):
        others = np.delete(probs, i, axis=0).mean(axis=0) > 0.5
        scores.append(0.5 * click_agreement(probs[i], clicks, cfg.click_radius)
                      + 0.5 * _iou(masks[i], others))
    z = cfg.sharpness * np.asarray(scores)
    e = np.exp(z - z.max())
    return e / e.sum()


def staple(probs: np.ndarray, iters: int = 15, eps: float = 1e-6) -> np.ndarray:
    """Soft binary STAPLE: posterior probability of foreground per pixel."""
    d = probs.reshape(len(probs), -1).astype(np.float64)
    prior = float(np.clip(d.mean(), eps, 1 - eps))
    sens = np.full(len(d), 0.9)
    spec = np.full(len(d), 0.9)
    for _ in range(iters):
        log_a = np.log(prior) + (d * np.log(sens)[:, None]
                                 + (1 - d) * np.log(1 - sens)[:, None]).sum(0)
        log_b = np.log(1 - prior) + ((1 - d) * np.log(spec)[:, None]
                                     + d * np.log(1 - spec)[:, None]).sum(0)
        post = 1.0 / (1.0 + np.exp(np.clip(log_b - log_a, -50, 50)))
        sens = np.clip((post * d).sum(1) / (post.sum() + eps), eps, 1 - eps)
        spec = np.clip(((1 - post) * (1 - d)).sum(1) / ((1 - post).sum() + eps), eps, 1 - eps)
    return post.reshape(probs.shape[1:]).astype(np.float32)


def fuse(probs: np.ndarray, clicks, cfg: EnsembleConfig) -> tuple[np.ndarray, list[float]]:
    """``probs``: (n_models, H, W) in [0, 1]. Returns the fused map and the
    global weight given to each model (uniform for mean / STAPLE)."""
    n = len(probs)
    if cfg.fusion == 'mean' or n == 1:
        return probs.mean(axis=0), [1.0 / n] * n
    if cfg.fusion == 'staple':
        return staple(probs, cfg.staple_iters), [1.0 / n] * n
    w = reliability_weights(probs, clicks, cfg)
    if cfg.fusion == 'weighted':
        return np.tensordot(w, probs, axes=1).astype(np.float32), w.tolist()
    # weighted_conf
    conf = np.abs(2 * probs - 1) + 1e-3
    pw = w[:, None, None] * conf
    return ((pw * probs).sum(0) / pw.sum(0)).astype(np.float32), w.tolist()


def pin_clicks(prob: np.ndarray, clicks, r: int) -> None:
    """Set a disc of radius ``r`` around each click to its label, in place
    (later clicks win where discs overlap)."""
    h, w = prob.shape
    yy, xx = np.ogrid[:h, :w]
    for c in clicks:
        y, x = int(c.coords[0]), int(c.coords[1])
        y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        disc = (yy[y0:y1] - y) ** 2 + (xx[:, x0:x1] - x) ** 2 <= r * r
        prob[y0:y1, x0:x1][disc] = 1.0 if c.is_positive else 0.0
