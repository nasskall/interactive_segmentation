"""Standalone student network (isegm.adaptation.student): training on labels
and soft targets, scoring, and reloading a saved student."""

from __future__ import annotations

import numpy as np
import pytest
import torch

smp = pytest.importorskip('segmentation_models_pytorch')
from isegm.adaptation import student as st  # noqa: E402

pytestmark = pytest.mark.slow


def _examples(synthetic_pair, n=3, soft=False):
    img, gt = synthetic_pair
    lesion = gt.astype(np.float32) * (0.9 if soft else 1.0)
    target = np.stack([lesion, np.zeros_like(lesion)])
    return [st.Example(f'e{i}', lambda: (img, target)) for i in range(n)]


def test_train_predict_and_reload(synthetic_pair, device, tmp_path):
    torch.manual_seed(0)
    model = st.build(2)
    # Two labelled copies and one soft (teacher-style) copy of the same image.
    examples = _examples(synthetic_pair, 2) + _examples(synthetic_pair, 1, soft=True)
    reports = []
    st.train(model, examples, 60, device, batch=2, size=128,
             report=lambda *a, **k: reports.append(a[0]))
    assert reports.count('preparing') == 3 and reports.count('training') == 60
    img, gt = synthetic_pair
    probs = st.predict(model, img, device, size=128)
    assert probs.shape == (2, *gt.shape)
    scores = st.evaluate(model, examples[:1], device, size=128)
    assert scores['iou'][0] > 0.6, scores
    assert (probs[1] > 0.5).mean() < 0.01

    path = tmp_path / 'student.pt'
    torch.save(st.payload(model, [{'id': 1, 'name': 'lesion'}, {'id': 2, 'name': 'other'}],
                          size=128), path)
    again, size = st.load(torch.load(path, weights_only=True))
    assert size == 128
    assert np.abs(st.predict(again, img, device, size) - probs).max() < 1e-5


def test_training_can_be_cancelled(synthetic_pair, device):
    with pytest.raises(st.Cancelled):
        st.train(st.build(2, pretrained=False), _examples(synthetic_pair, 1), 10, device,
                 should_stop=lambda: True)
