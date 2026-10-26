"""Click-free SAM 2 (isegm.adaptation.promptless): class prompts through the
real decoder, training, scoring and reloading a saved model."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from isegm.adaptation import promptless as pl

pytestmark = [pytest.mark.slow, pytest.mark.sam2]


def _examples(synthetic_pair, n=3):
    img, gt = synthetic_pair
    # Class 0: the lesion; class 1: never present.
    masks = np.stack([gt, np.zeros_like(gt)])
    return [pl.Example(f'e{i}', lambda: (img, masks)) for i in range(n)]


def test_train_predict_and_reload(sam2_model, synthetic_pair, device, tmp_path):
    base = sam2_model[0]
    net = copy.deepcopy(base)
    examples = _examples(synthetic_pair)
    reports = []
    prompts, payload = pl.train(net, 2, examples, 60, device, tmp_path / 'feats', batch=2,
                                report=lambda *a, **k: reports.append(a[0]))
    assert reports.count('encoding') == 3 and reports.count('training') == 60
    img, gt = synthetic_pair
    probs = pl.predict(net, prompts, img, device)
    assert probs.shape == (2, *gt.shape)

    scores = pl.evaluate(net, prompts, examples[:1], device)
    assert scores['images'] == 1 and len(scores['iou']) == 2
    # One image, 60 steps: the lesion is found and the absent class stays empty.
    assert scores['iou'][0] > 0.5, scores
    assert (probs[1] > 0.5).mean() < 0.01

    # The saved payload loads (weights_only) onto a fresh base and predicts the same.
    path = tmp_path / 'auto.pt'
    torch.save(payload, path)
    again = copy.deepcopy(base)
    reloaded = pl.load(again, torch.load(path, weights_only=True))
    assert np.abs(pl.predict(again, reloaded, img, device) - probs).max() < 1e-4

    # Served by the web app's loader, which checks the file's SHA-256 and kind.
    from server.sessions import ModelManager
    _, ckpt, label = sam2_model
    manager = ModelManager(base, device, label, ckpt.name, ckpt.parent)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    served, served_prompts = manager.adapter_net(label, ckpt.name, path, sha, kind='automatic')
    assert np.abs(pl.predict(served, served_prompts, img, device) - probs).max() < 1e-4
    with pytest.raises(ValueError, match='interactive'):
        ModelManager(base, device, label, ckpt.name, ckpt.parent).adapter_net(
            label, ckpt.name, path, sha)


def test_training_can_be_cancelled(sam2_model, synthetic_pair, device, tmp_path):
    net = copy.deepcopy(sam2_model[0])
    with pytest.raises(pl.Cancelled):
        pl.train(net, 2, _examples(synthetic_pair, 1), 10, device, tmp_path,
                 should_stop=lambda: True)
