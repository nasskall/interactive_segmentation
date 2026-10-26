"""Project models: recorded clicks on assisted annotations, training runs and
their register, and assisted labelling on a project's own model. Training
itself is stubbed here (see test_project_training.py for the real one)."""

from __future__ import annotations

import json
import threading

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402
from server.labelling import geometry, training  # noqa: E402
from server.labelling.db import Database, now  # noqa: E402
from server.sessions import ModelManager  # noqa: E402
from test_labelling import PASSWORD, Api, _class, _items, _png, _project  # noqa: E402

SAM2_CKPT = 'sam2.1_hiera_tiny.pt'


@pytest.fixture
def weights_dir(tmp_path, ritm_checkpoint_path):
    d = tmp_path / 'weights'
    d.mkdir()
    (d / SAM2_CKPT).write_bytes(b'stand-in SAM 2 checkpoint')
    return d


@pytest.fixture
def client(fresh_ritm, device, ritm_checkpoint_path, weights_dir, tmp_path):
    manager = ModelManager(fresh_ritm, device, 'RITM', ritm_checkpoint_path.name, weights_dir)
    return TestClient(create_app(manager, data_dir=tmp_path / 'data'))


@pytest.fixture
def admin(client):
    token = client.app.state.labelling.setup_token
    r = client.post('/api/auth/setup', json={'setup_token': token, 'username': 'admin',
                                             'password': PASSWORD, 'display_name': 'Admin'})
    assert r.status_code == 201, r.text
    return Api(client, r.json()['token'], r.json()['user'])


def _ctx(admin: Api):
    return admin.client.app.state.labelling


def _seed_objects(admin: Api, n_items=6, per_item=2, w=160, h=120) -> tuple[dict, list[dict]]:
    """A segmentation project with ``n_items`` images and ``per_item``
    click-made mask annotations on each, written straight to the database."""
    p = _project(admin, 'instance_segmentation')
    cls = _class(admin, p['id'], label_type='mask')
    items = _items(admin, p['id'], n=n_items, w=w, h=h)
    mask = np.zeros((h, w), bool)
    cv2.circle(mask.view(np.uint8), (w // 2, h // 2), 20, 1, -1)
    geom = json.dumps({'type': 'mask', 'rle': geometry.rle_encode(mask)})
    prompts = json.dumps({'image': 'image', 'objects': [
        {'clicks': [[w // 2, h // 2, True], [2, 2, False]], 'box': None}]})
    conn = _ctx(admin).db.connect()
    with conn:
        for it in items:
            for _ in range(per_item):
                conn.execute(
                    'INSERT INTO annotations (item_id, class_id, labeller_id, geometry, source, '
                    "model, prompts, created_at, updated_at) VALUES (?, ?, ?, ?, 'assisted', "
                    "'{}', ?, ?, ?)", (it['id'], cls['id'], admin.user['id'], geom, prompts,
                                       now(), now()))
    conn.close()
    return p, items


def _stub_fit(job, report, should_stop):
    report('training', job['steps'], job['steps'], loss=0.1)
    job['out_dir'].mkdir(parents=True, exist_ok=True)
    (job['out_dir'] / 'adapter.pt').write_bytes(b'adapter ' + str(job['id']).encode())
    return {'metrics': {'base': {'noc@90': 5.0}, 'trained': {'noc@90': 3.0}},
            'adapter_file': 'adapter.pt', 'job': job}


def _train(admin: Api, p: dict, fit=_stub_fit, **body) -> dict:
    _ctx(admin).training.fit = fit
    m = admin.ok('post', f"/api/projects/{p['id']}/models", 202, json=body)
    _ctx(admin).training.join(30)
    return next(x for x in admin.ok('get', f"/api/projects/{p['id']}/models")['models']
                if x['id'] == m['id'])


# ── recorded clicks ───────────────────────────────────────────────────────

def test_assisted_annotation_records_each_objects_clicks_in_item_pixels(admin):
    p = _project(admin, 'semantic_segmentation')
    cls = _class(admin, p['id'], label_type='mask')
    w, h = 3000, 1500          # downscaled to 2048 x 1024 in the session
    item = admin.ok('post', f"/api/projects/{p['id']}/items", 201,
                    files=[('images', ('big.png', _png(w, h), 'image/png'))])[0]
    s = admin.ok('post', f"/api/items/{item['id']}/assist", 201)
    sid, sw = s['session_id'], s['width']
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 1024, 'y': 512})
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 10, 'y': 10, 'positive': False})
    admin.ok('post', f'/api/sessions/{sid}/finish')
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 300, 'y': 200})
    # Undo and redo the second object's click: only the kept one is recorded.
    admin.ok('post', f'/api/sessions/{sid}/undo')
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 400, 'y': 200})
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'from_session': sid})

    s = w / sw
    assert a['prompts'] == {'image': 'image', 'objects': [
        {'clicks': [[int(1024 * s), int(512 * s), True], [int(10 * s), int(10 * s), False]],
         'box': None},
        {'clicks': [[int(400 * s), int(200 * s), True]], 'box': None}]}
    # The next save starts from an empty prompt list.
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 1024, 'y': 512})
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'from_session': sid})
    assert len(a['prompts']['objects']) == 1
    # Manual labels have no prompts.
    rle = geometry.rle_encode(np.ones((h, w), bool))
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'geometry': {'type': 'mask', 'rle': rle}})
    assert a['prompts'] is None and a['source'] == 'manual'


def test_change_detection_prompts_record_which_image(admin):
    p = _project(admin, 'change_detection')
    cls = _class(admin, p['id'], label_type='polygon')
    item = admin.ok('post', f"/api/projects/{p['id']}/items", 201,
                    files=[('before', ('b.png', _png(), 'image/png')),
                           ('after', ('a.png', _png(), 'image/png'))])[0]
    sid = admin.ok('post', f"/api/items/{item['id']}/assist", 201,
                   params={'image': 'before'})['session_id']
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60})
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'from_session': sid})
    assert a['prompts']['image'] == 'before'
    assert a['prompts']['objects'][0]['clicks'] == [[80, 60, True]]


# ── training data and split ───────────────────────────────────────────────

def test_only_single_click_made_mask_objects_are_trainable(admin):
    p, items = _seed_objects(admin, n_items=2, per_item=1)
    conn = _ctx(admin).db.connect()
    with conn:
        ann = conn.execute('SELECT * FROM annotations ORDER BY id').fetchone()
        for prompts in ({'image': 'image', 'objects': []},
                        {'image': 'image', 'objects': [{'clicks': [], 'box': None}]},
                        {'image': 'image', 'objects': [{'clicks': [[1, 1, True]],
                                                        'box': [0, 0, 5, 5]}]},
                        {'image': 'image', 'objects': [{'clicks': [[1, 1, True]], 'box': None}] * 2},
                        {'image': 'after', 'objects': [{'clicks': [[1, 1, True]], 'box': None}]}):
            conn.execute('INSERT INTO annotations (item_id, class_id, labeller_id, geometry, '
                         "source, prompts, created_at, updated_at) VALUES (?, ?, ?, ?, 'assisted', "
                         '?, ?, ?)', (ann['item_id'], ann['class_id'], ann['labeller_id'],
                                      ann['geometry'], json.dumps(prompts), now(), now()))
        objects = training.trainable_objects(conn, p['id'])
    conn.close()
    assert [o['annotation_id'] for o in objects] == [1, 2]
    # Stored as [x, y, positive]; training takes (y, x, positive).
    assert objects[0]['clicks'] == [(60, 80, True), (2, 2, False)]


def test_split_holds_out_whole_items():
    objects = [{'item_id': i // 3, 'n': i} for i in range(30)]
    train, held, held_items = training.split_items(objects)
    assert len(held_items) == 2
    assert {o['item_id'] for o in held} == set(held_items)
    assert not {o['item_id'] for o in train} & set(held_items)
    assert len(train) + len(held) == 30
    assert training.split_items(objects) == (train, held, held_items)   # seeded


def test_training_needs_enough_objects(admin):
    p, _ = _seed_objects(admin, n_items=2, per_item=2)
    info = admin.ok('get', f"/api/projects/{p['id']}/models")
    assert info['data']['trainable_objects'] == 4 and info['data']['problems']
    assert info['base_models'] == [{'model_type': 'SAM2 - Tiny', 'checkpoint': SAM2_CKPT}]
    r = admin.post(f"/api/projects/{p['id']}/models", json={})
    assert r.status_code == 400 and 'at least 10' in r.json()['detail']


# ── training runs ─────────────────────────────────────────────────────────

def test_training_run_is_registered_with_hashes_and_scores(admin):
    p, items = _seed_objects(admin)
    m = _train(admin, p, steps=40)
    assert m['status'] == 'done', m['error']
    assert (m['base_model_type'], m['base_checkpoint'], m['steps']) == ('SAM2 - Tiny', SAM2_CKPT, 40)
    assert m['base_sha256'] and len(m['adapter_sha256']) == 64
    assert m['metrics'] == {'base': {'noc@90': 5.0}, 'trained': {'noc@90': 3.0}}
    assert m['train_objects'] + 2 * len(m['held_out_items']) == 12
    path = _ctx(admin).training.models_dir / m['adapter_file']
    assert training.file_sha256(path) == m['adapter_sha256']
    actions = [e['action'] for e in admin.ok('get', '/api/audit', params={'action': 'model.'})]
    assert actions[:2] == ['model.train_done', 'model.train']


def test_training_job_gets_the_split_and_default_steps(admin):
    p, _ = _seed_objects(admin)
    seen = {}

    def fit(job, report, should_stop):
        seen.update(job)
        return _stub_fit(job, report, should_stop)
    m = _train(admin, p, fit=fit)
    assert seen['steps'] == training.default_steps(m['train_objects'])
    assert {o['item_id'] for o in seen['held']} == set(m['held_out_items'])
    assert len(seen['train']) == m['train_objects']


def test_failed_and_cancelled_runs(admin):
    p, _ = _seed_objects(admin)

    def broken(job, report, should_stop):
        raise RuntimeError('out of memory')
    m = _train(admin, p, fit=broken)
    assert m['status'] == 'failed' and 'out of memory' in m['error']

    started = threading.Event()

    def slow(job, report, should_stop):
        started.set()
        while not should_stop():
            threading.Event().wait(0.01)
        raise RuntimeError('stopped')
    _ctx(admin).training.fit = slow
    m = admin.ok('post', f"/api/projects/{p['id']}/models", 202, json={})
    started.wait(5)
    # One run at a time.
    assert admin.post(f"/api/projects/{p['id']}/models", json={}).status_code == 409
    assert admin.delete(f"/api/project-models/{m['id']}").status_code == 409
    admin.ok('post', f"/api/project-models/{m['id']}/cancel")
    _ctx(admin).training.join(10)
    m = admin.ok('get', f"/api/projects/{p['id']}/models")['models'][0]
    assert m['status'] == 'cancelled' and m['error'] is None
    assert admin.post(f"/api/project-models/{m['id']}/cancel").status_code == 409


def test_restart_marks_unfinished_runs_failed(tmp_path):
    db = Database(tmp_path / 'labelling.db')
    conn = db.connect()
    with conn:
        conn.execute("INSERT INTO projects (name, task_type, created_at) "
                     "VALUES ('p', 'instance_segmentation', ?)", (now(),))
        conn.execute("INSERT INTO project_models (project_id, status, base_model_type, "
                     "base_checkpoint, steps, train_objects, train_items, held_out_items, "
                     "created_at) VALUES (1, 'running', 'SAM2 - Tiny', 'x.pt', 10, 1, 1, '[]', ?)",
                     (now(),))
    training.TrainingRunner(db, None, tmp_path / 'models', tmp_path, 'cpu')
    row = conn.execute('SELECT status, error FROM project_models').fetchone()
    conn.close()
    assert row['status'] == 'failed' and 'restart' in row['error']


def test_labellers_cannot_see_or_train_models(admin):
    from test_labelling import _labeller
    p, items = _seed_objects(admin)
    lab = _labeller(admin)
    admin.ok('patch', f"/api/items/{items[0]['id']}", json={'assignee_id': lab.user['id']})
    assert lab.get(f"/api/projects/{p['id']}/models").status_code == 403
    assert lab.post(f"/api/projects/{p['id']}/models", json={}).status_code == 403
    assert lab.patch(f"/api/projects/{p['id']}", json={'assist_model_id': None}).status_code == 403


# ── assisting with a project model ────────────────────────────────────────

def test_project_model_assists_its_project_only(admin, ritm_checkpoint_path, device):
    from conftest import _fresh_ritm
    p, items = _seed_objects(admin)
    other = _project(admin, 'semantic_segmentation', name='other')
    other_item = _items(admin, other['id'])[0]
    m = _train(admin, p)
    ctx = _ctx(admin)
    own_net = _fresh_ritm(ritm_checkpoint_path, device)
    loaded = []

    def adapter_net(label, ckpt, path, sha):
        loaded.append((label, ckpt, path.name, sha))
        return own_net
    ctx.manager.adapter_net = adapter_net

    r = admin.patch(f"/api/projects/{p['id']}", json={'assist_model_id': 999})
    assert r.status_code == 400
    assert admin.ok('patch', f"/api/projects/{p['id']}",
                    json={'assist_model_id': m['id']})['assist_model_id'] == m['id']

    s = admin.ok('post', f"/api/items/{items[0]['id']}/assist", 201)
    assert s['model'] == f"Project model #{m['id']} (SAM2 - Tiny)"
    assert loaded == [('SAM2 - Tiny', SAM2_CKPT, 'adapter.pt', m['adapter_sha256'])]
    session = ctx.manager.get(s['session_id'])
    assert session.controller.net is own_net
    # Switching the shared model leaves the project-model session alone.
    ctx.manager._switch(_fresh_ritm(ritm_checkpoint_path, device), 'RITM',
                        ritm_checkpoint_path.name)
    assert session.controller.net is own_net

    admin.ok('post', f"/api/sessions/{s['session_id']}/clicks", json={'x': 80, 'y': 60})
    a = admin.ok('post', f"/api/items/{items[0]['id']}/annotations", 201,
                 json={'class_id': admin.ok('get', f"/api/projects/{p['id']}/classes")[0]['id'],
                       'from_session': s['session_id']})
    assert a['model']['adapter'] == {'project_model_id': m['id'], 'file': m['adapter_file'],
                                     'sha256': m['adapter_sha256']}
    assert a['model']['checkpoint'] == SAM2_CKPT

    # Other projects keep the shared model.
    s = admin.ok('post', f"/api/items/{other_item['id']}/assist", 201)
    assert s['model'] is None
    assert ctx.manager.get(s['session_id']).controller.net is ctx.manager.net

    # A model in use cannot be deleted; once unset it can, with its files.
    assert admin.delete(f"/api/project-models/{m['id']}").status_code == 409
    admin.ok('patch', f"/api/projects/{p['id']}", json={'assist_model_id': None})
    admin.ok('delete', f"/api/project-models/{m['id']}", 204)
    assert not (ctx.training.models_dir / m['adapter_file']).exists()
    assert admin.ok('get', f"/api/projects/{p['id']}/models")['models'] == []


def test_changed_adapter_file_is_refused(admin):
    p, items = _seed_objects(admin)
    m = _train(admin, p)
    admin.ok('patch', f"/api/projects/{p['id']}", json={'assist_model_id': m['id']})
    (_ctx(admin).training.models_dir / m['adapter_file']).write_bytes(b'tampered')
    r = admin.post(f"/api/items/{items[0]['id']}/assist")
    assert r.status_code == 409 and 'SHA-256' in r.json()['detail']
