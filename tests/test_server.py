"""REST API round-trips against the real RITM checkpoint."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402
from server.sessions import ModelManager  # noqa: E402


@pytest.fixture
def client(fresh_ritm, device, ritm_checkpoint_path):
    manager = ModelManager(fresh_ritm, device, 'RITM', ritm_checkpoint_path.name,
                           ritm_checkpoint_path.parent)
    return TestClient(create_app(manager))


def _png(h=120, w=160):
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (40, 40, 160), -1)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    return buf.tobytes()


def _upload(client, h=120, w=160):
    r = client.post('/api/sessions', files={'image': ('lesion one.png', _png(h, w), 'image/png')})
    assert r.status_code == 201, r.text
    return r.json()


def test_models_listing(client):
    info = client.get('/api/models').json()
    assert info['current']['model_type'] == 'RITM'
    assert 'best_checkpoint_068.pth' in info['checkpoints']
    assert 'SAM2 - Tiny' in info['model_types']


def test_click_undo_finish_flow(client):
    s = _upload(client)
    sid = s['session_id']
    assert (s['width'], s['height']) == (160, 120)
    assert s['clicks'] == [] and not s['has_incomplete_object']
    assert s['overlay'].startswith('data:image/png;base64,')

    s = client.post(f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60, 'positive': True}).json()
    assert s['clicks'] == [{'x': 80, 'y': 60, 'positive': True}]
    assert s['has_incomplete_object'] and s['can_undo']

    s = client.post(f'/api/sessions/{sid}/clicks', json={'x': 5, 'y': 5, 'positive': False}).json()
    assert len(s['clicks']) == 2

    s = client.post(f'/api/sessions/{sid}/undo').json()
    assert len(s['clicks']) == 1

    s = client.post(f'/api/sessions/{sid}/finish').json()
    assert s['object_count'] == 1 and s['clicks'] == []

    r = client.get(f'/api/sessions/{sid}/mask')
    assert r.status_code == 200
    assert 'filename="lesionone_mask.png"' in r.headers['content-disposition']
    mask = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (120, 160)


def test_image_is_downscaled_and_mask_restored(client):
    s = _upload(client, h=1500, w=3000)
    assert (s['width'], s['height']) == (2048, 1024)
    assert (s['original_width'], s['original_height']) == (3000, 1500)
    img = cv2.imdecode(np.frombuffer(client.get(f"/api/sessions/{s['session_id']}/image").content,
                                     np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (1024, 2048)
    mask = cv2.imdecode(np.frombuffer(client.get(f"/api/sessions/{s['session_id']}/mask").content,
                                      np.uint8), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1500, 3000)


def test_settings_and_auto_segment(client):
    sid = _upload(client)['session_id']
    s = client.patch(f'/api/sessions/{sid}/settings', json={'prob_thresh': 0.7}).json()
    assert s['prob_thresh'] == 0.7
    s = client.post(f'/api/sessions/{sid}/auto-segment').json()
    assert s['clicks'] == [{'x': 80, 'y': 60, 'positive': True}]
    s = client.patch(f'/api/sessions/{sid}/settings', json={'use_zoom_in': False}).json()
    assert not s['use_zoom_in'] and s['clicks'] == []


def test_errors(client):
    assert client.get('/api/sessions/nope').status_code == 404
    r = client.post('/api/sessions', files={'image': ('x.png', b'not an image', 'image/png')})
    assert r.status_code == 400
    sid = _upload(client)['session_id']
    assert client.post(f'/api/sessions/{sid}/clicks', json={'x': 999, 'y': 1}).status_code == 400
    r = client.post('/api/models/load', json={'model_type': 'RITM', 'checkpoint': '../config.yml'})
    assert r.status_code == 400
    assert client.delete(f'/api/sessions/{sid}').status_code == 204
    assert client.get(f'/api/sessions/{sid}').status_code == 404


def test_failed_load_keeps_previous_model(client, monkeypatch):
    # Reproduces SAM being picked while segment_anything is missing: the
    # checkpoint loads, but building its predictor raises.
    from isegm.inference import utils
    from isegm.inference.predictors import sam_predictor

    class FakeSam:
        _model_type = 'sam'

    monkeypatch.setattr(utils, 'load_model_by_type', lambda *a, **k: FakeSam())
    monkeypatch.setattr(sam_predictor, 'SAM_AVAILABLE', False)

    sid = _upload(client)['session_id']
    r = client.post('/api/models/load',
                    json={'model_type': 'SAM - ViT-B', 'checkpoint': 'best_checkpoint_068.pth'})
    assert r.status_code == 500
    assert 'segment_anything is not installed' in r.json()['detail']
    assert client.get('/api/models').json()['current']['model_type'] == 'RITM'

    # Existing and new sessions keep working on the previous model.
    r = client.post(f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60, 'positive': True})
    assert r.status_code == 200
    new_sid = _upload(client)['session_id']
    r = client.post(f'/api/sessions/{new_sid}/clicks', json={'x': 80, 'y': 60, 'positive': True})
    assert r.status_code == 200


def test_box_prompt_needs_sam(client):
    s = _upload(client)
    assert s['supports_box'] is False and s['box'] is None
    sid = s['session_id']
    r = client.post(f'/api/sessions/{sid}/box', json={'x0': 10, 'y0': 10, 'x1': 50, 'y1': 50})
    assert r.status_code == 422
    r = client.post(f'/api/sessions/{sid}/box', json={'x0': 50, 'y0': 10, 'x1': 10, 'y1': 50})
    assert r.status_code == 400


# ── ensemble ──────────────────────────────────────────────────────────────

RITM_MEMBER = {'model_type': 'RITM', 'checkpoint': 'best_checkpoint_068.pth'}
SAM2_MEMBER = {'model_type': 'SAM2 - Tiny', 'checkpoint': 'sam2.1_hiera_tiny.pt'}


def test_guess_model_type():
    from server.sessions import guess_model_type
    assert guess_model_type('best_checkpoint_068.pth') == 'RITM'
    assert guess_model_type('sam_vit_b_01ec64.pth') == 'SAM - ViT-B'
    assert guess_model_type('sam2.1_hiera_tiny.pt') == 'SAM2 - Tiny'
    assert guess_model_type('sam2.1_hiera_base_plus.pt') == 'SAM2 - Base+'
    assert guess_model_type('cocolvis_vit_base.pth') == 'SimpleClick'


def test_ensemble_loads_reuses_models_and_segments(client, monkeypatch, fresh_ritm,
                                                   ritm_checkpoint_path):
    if not (ritm_checkpoint_path.parent / SAM2_MEMBER['checkpoint']).exists():
        pytest.skip('needs sam2.1_hiera_tiny.pt in model_weights/')
    from isegm.inference import utils
    loads = []
    # Stand in RITM for every member so the test stays fast; the ensemble
    # maths has its own tests in test_ensemble.py.
    monkeypatch.setattr(utils, 'load_model_by_type',
                        lambda path, label, device: loads.append(label) or fresh_ritm)

    info = client.get('/api/models').json()
    assert 'Ensemble' in info['model_types']
    assert RITM_MEMBER in info['ensemble']['candidates']
    assert SAM2_MEMBER in info['ensemble']['candidates']
    assert info['current']['supports_box'] is False

    sid = _upload(client)['session_id']
    r = client.post('/api/models/load', json={'model_type': 'Ensemble', 'fusion': 'mean',
                                              'tta': False,
                                              'members': [RITM_MEMBER, SAM2_MEMBER]})
    assert r.status_code == 200, r.text
    cur = r.json()['current']
    assert cur['model_type'] == 'Ensemble' and cur['checkpoint'] is None
    assert cur['members'] == [RITM_MEMBER, SAM2_MEMBER]
    assert (cur['fusion'], cur['tta'], cur['supports_box']) == ('mean', False, True)
    assert loads == ['SAM2 - Tiny']          # RITM was already loaded

    s = client.post(f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60, 'positive': True}).json()
    assert s['has_incomplete_object']
    assert set(s['ensemble_weights']) == {'RITM', 'SAM2 - Tiny'}

    # Changing fusion re-uses both loaded models; STAPLE without TTA is the default.
    r = client.post('/api/models/load', json={'model_type': 'Ensemble',
                                              'members': [RITM_MEMBER, SAM2_MEMBER]})
    cur = r.json()['current']
    assert (cur['fusion'], cur['tta']) == ('staple', False) and loads == ['SAM2 - Tiny']

    # Back to a single model: no reload, no ensemble weights.
    r = client.post('/api/models/load', json=RITM_MEMBER)
    assert r.json()['current']['model_type'] == 'RITM' and 'members' not in r.json()['current']
    s = client.post(f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60, 'positive': True}).json()
    assert s['ensemble_weights'] is None and loads == ['SAM2 - Tiny']


@pytest.mark.parametrize('body', [
    {'model_type': 'Ensemble', 'members': [RITM_MEMBER]},
    {'model_type': 'Ensemble', 'members': [RITM_MEMBER, RITM_MEMBER]},
    {'model_type': 'Ensemble', 'members': [RITM_MEMBER, SAM2_MEMBER], 'fusion': 'vote'},
    {'model_type': 'Ensemble', 'members': [RITM_MEMBER,
                                           {'model_type': 'RITM', 'checkpoint': '../x.pth'}]},
    {'model_type': 'Ensemble', 'members': [RITM_MEMBER,
                                           {'model_type': 'Ensemble', 'checkpoint': 'x.pth'}]},
    {'model_type': 'Ensemble'},
    {'model_type': 'RITM'},
])
def test_ensemble_load_rejects_bad_requests(client, body):
    r = client.post('/api/models/load', json=body)
    assert r.status_code == 400, r.text
    assert client.get('/api/models').json()['current']['model_type'] == 'RITM'
