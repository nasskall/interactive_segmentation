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


def test_encoder_weights_are_fingerprinted():
    import hashlib

    from huggingface_hub import hf_hub_download
    from server.labelling.training import encoder_weights_sha256
    st.build(1)                       # downloads the weights if not cached yet
    ref = st.encoder_weights_repo()
    repo, _, revision = ref.partition('@')
    assert repo == 'smp-hub/resnet18.imagenet' and revision
    path = hf_hub_download(repo, 'model.safetensors', revision=revision, local_files_only=True)
    with open(path, 'rb') as f:
        assert encoder_weights_sha256(ref) == hashlib.sha256(f.read()).hexdigest()
    assert encoder_weights_sha256('smp-hub/no-such-repo@0') is None


def test_training_can_be_cancelled(synthetic_pair, device):
    with pytest.raises(st.Cancelled):
        st.train(st.build(2, pretrained=False), _examples(synthetic_pair, 1), 10, device,
                 should_stop=lambda: True)


def test_onnx_export_matches_pytorch(synthetic_pair, tmp_path):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnxscript')
    torch.manual_seed(0)
    model = st.build(2, pretrained=False).eval()
    path = tmp_path / 'student.onnx'
    st.export_onnx(model, path, size=128)
    assert [p.name for p in tmp_path.iterdir()] == ['student.onnx']   # weights embedded
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    assert [i.name for i in sess.get_inputs()] == ['image']
    img, _ = synthetic_pair
    # The ONNX model takes the image resized to its input size, RGB 0..255.
    x = st._resized(img, 128).astype(np.float32).transpose(2, 0, 1)[None]
    batch = np.concatenate([x, x[:, :, ::-1].copy()])                  # any batch size
    probs = sess.run(None, {'image': batch})[0]
    assert probs.shape == (2, 2, 128, 128)
    with torch.no_grad():
        exact = torch.sigmoid(model(st._normalise(st._resized(img, 128))[None]))[0].numpy()
    assert np.abs(probs[0] - exact).max() < 1e-4


def test_downloaded_student_onnx(tmp_path):
    """What the download endpoint puts in a student's zip: a single ONNX file
    registered in the model card with its hash and input/output contract."""
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnxscript')
    import hashlib

    from server.labelling.training import student_onnx
    path = tmp_path / 'adapter.pt'
    torch.save(st.payload(st.build(1, pretrained=False), [{'id': 7, 'name': 'lesion'}], size=96),
               path)
    out = tmp_path / 'onnx'
    out.mkdir()
    file, entry = student_onnx(path, out)
    assert entry['file'] == 'model.onnx' and '96 x 96' in entry['input']
    assert entry['sha256'] == hashlib.sha256(file.read_bytes()).hexdigest()
    sess = ort.InferenceSession(str(file), providers=['CPUExecutionProvider'])
    probs = sess.run(None, {'image': np.zeros((1, 3, 96, 96), np.float32)})[0]
    assert probs.shape == (1, 1, 96, 96) and 0 <= probs.min() and probs.max() <= 1
