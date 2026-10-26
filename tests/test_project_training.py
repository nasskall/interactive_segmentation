"""Fine-tuning SAM 2 on recorded clicks (isegm.adaptation.project_training)
and serving the result as a project model."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import torch

from isegm.adaptation import project_training as pt
from server.sessions import ModelManager

pytestmark = [pytest.mark.slow, pytest.mark.sam2]


def _examples(synthetic_pair, n=3):
    img, gt = synthetic_pair
    ys, xs = np.where(gt)
    clicks = [(int(ys.mean()), int(xs.mean()), True), (2, 2, False)]
    return [pt.Example(f'e{i}', lambda: (img, gt), clicks) for i in range(n)]


def test_train_evaluate_and_serve_a_project_model(sam2_model, synthetic_pair, device, tmp_path):
    base, ckpt, label = sam2_model
    net = copy.deepcopy(base)
    examples = _examples(synthetic_pair)
    before = pt.evaluate(net, examples[:1], device, max_clicks=5)
    assert set(before) == {'objects', 'iou@1', 'iou@3', 'iou@5', 'noc@85', 'noc@90'}

    reports = []
    payload = pt.train(net, examples, 4, device, tmp_path / 'feats', batch=2,
                       report=lambda *a, **k: reports.append(a[0]))
    assert reports.count('encoding') == 3 and reports.count('training') == 4
    assert payload['model_type'] == 'sam2' and payload['lora_state']
    after = pt.evaluate(net, examples[:1], device, max_clicks=5)
    assert 0 <= after['iou@1'] <= 1

    # The saved adapter loads onto a fresh base and gives the trained model's masks.
    path = tmp_path / 'adapter.pt'
    torch.save(payload, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manager = ModelManager(base, device, label, ckpt.name, ckpt.parent)
    served = manager.adapter_net(label, ckpt.name, path, sha)
    assert served is not base
    assert manager.adapter_net(label, ckpt.name, path, sha) is served       # cached
    again = pt.evaluate(served, examples[:1], device, max_clicks=5)
    assert again['iou@1'] == pytest.approx(after['iou@1'], abs=1e-4)

    with pytest.raises(ValueError, match='SHA-256'):
        manager.adapter_net(label, ckpt.name, path, '0' * 64)


def test_training_can_be_cancelled(sam2_model, synthetic_pair, device, tmp_path):
    net = copy.deepcopy(sam2_model[0])
    with pytest.raises(pt.Cancelled):
        pt.train(net, _examples(synthetic_pair, 1), 100, device, tmp_path, should_stop=lambda: True)
