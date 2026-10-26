"""Automatic (click-free) project models: training data and runs, choosing
one, and auto-label proposals that a labeller accepts one by one. The model
itself is stubbed (see test_promptless.py for the real one)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('fastapi')

from isegm.adaptation import promptless  # noqa: E402
from server.labelling import geometry, training  # noqa: E402
from test_labelling import _class, _items, _labeller, _project  # noqa: E402
from test_project_models import (_ctx, _seed_objects, _stub_fit, _train, admin,  # noqa: E402,F401
                                 client, weights_dir)


def _done(admin, items):
    for it in items:
        admin.ok('patch', f"/api/items/{it['id']}", json={'status': 'done'})


def test_automatic_data_uses_done_single_images_and_segmentation_classes(admin):
    p, items = _seed_objects(admin, n_items=4, per_item=2)
    _class(admin, p['id'], 'vehicle', label_type='polygon')
    _done(admin, items[:3])
    conn = _ctx(admin).db.connect()
    classes, data = training.automatic_data(conn, p['id'])
    summary = training.automatic_summary(conn, p['id'])
    conn.close()
    assert [c['name'] for c in classes] == ['lesion', 'vehicle']
    assert [d['item_id'] for d in data] == [it['id'] for it in items[:3]]
    assert all(len(d['labels']) == 2 and d['labels'][0][0] == 0 for d in data)
    assert summary['done_images'] == 3 and summary['labels'] == 6
    assert 'at least 10 images' in summary['problems'][0]


def test_automatic_training_run(admin):
    p, items = _seed_objects(admin, n_items=10, per_item=1)
    _done(admin, items)
    seen = {}

    def fit(job, report, should_stop):
        seen.update(job)
        return _stub_fit(job, report, should_stop)
    m = _train(admin, p, fit=fit, kind='automatic')
    assert m['status'] == 'done' and m['kind'] == 'automatic'
    assert m['classes'] == [{'id': seen['classes'][0]['id'], 'name': 'lesion'}]
    assert len(m['held_out_items']) == 2 and m['train_items'] == 8
    assert seen['steps'] == training.default_auto_steps(8)
    assert len(seen['train']) == 8 and seen['train'][0]['labels']
    info = admin.ok('get', f"/api/projects/{p['id']}/models")
    assert info['auto_data']['done_images'] == 10 and not info['auto_data']['problems']


def test_models_are_chosen_for_the_matching_use(admin):
    p, items = _seed_objects(admin, n_items=10, per_item=1)
    _done(admin, items)
    interactive = _train(admin, p)
    auto = _train(admin, p, kind='automatic')
    url = f"/api/projects/{p['id']}"
    assert admin.patch(url, json={'assist_model_id': auto['id']}).status_code == 400
    assert admin.patch(url, json={'auto_model_id': interactive['id']}).status_code == 400
    proj = admin.ok('patch', url, json={'auto_model_id': auto['id']})
    assert (proj['auto_model_id'], proj['assist_model_id']) == (auto['id'], None)
    assert admin.delete(f"/api/project-models/{auto['id']}").status_code == 409


@pytest.fixture
def auto_project(admin, monkeypatch):
    """An instance segmentation project with a chosen automatic model whose
    'prediction' is two lesion blobs and nothing of the second class."""
    p, items = _seed_objects(admin, n_items=10, per_item=1)
    vehicle = _class(admin, p['id'], 'vehicle', label_type='polygon')
    _done(admin, items)
    m = _train(admin, p, kind='automatic')
    admin.ok('patch', f"/api/projects/{p['id']}", json={'auto_model_id': m['id']})
    loaded = []

    def adapter_net(label, ckpt, path, sha, kind='interactive'):
        loaded.append(kind)
        return 'net', 'prompts'

    def predict(net, prompts, image, device):
        h, w = image.shape[:2]
        probs = np.zeros((2, h, w), np.float32)
        probs[0, 10:40, 10:40] = 0.9
        probs[0, 60:100, 90:150] = 0.8
        return probs
    monkeypatch.setattr(_ctx(admin).manager, 'adapter_net', adapter_net)
    monkeypatch.setattr(promptless, 'predict', predict)
    new = _items(admin, p['id'])[0]
    return p, m, new, vehicle, loaded


def test_auto_label_proposes_and_saves_only_what_is_accepted(admin, auto_project):
    p, m, item, vehicle, loaded = auto_project
    r = admin.ok('post', f"/api/items/{item['id']}/auto-label")
    assert loaded == ['automatic'] and r['model_id'] == m['id']
    # Instance segmentation: one proposal per region.
    assert len(r['proposals']) == 2
    assert {pr['geometry']['type'] for pr in r['proposals']} == {'mask'}
    assert admin.ok('get', f"/api/items/{item['id']}/annotations") == []

    lesion_id = r['proposals'][0]['class_id']
    url = f"/api/items/{item['id']}/annotations"
    a = admin.ok('post', url, 201, json={'class_id': lesion_id,
                                         'from_proposal': {'set': r['set'], 'index': 1}})
    assert a['source'] == 'assisted' and a['prompts'] is None
    assert a['model']['mode'] == 'automatic'
    assert a['model']['adapter'] == {'project_model_id': m['id'], 'kind': 'automatic',
                                     'file': m['adapter_file'], 'sha256': m['adapter_sha256']}
    assert a['geometry'] == r['proposals'][1]['geometry']
    # Once only; and a mask proposal cannot become a polygon-class label.
    assert admin.post(url, json={'class_id': lesion_id,
                                 'from_proposal': {'set': r['set'], 'index': 1}}).status_code == 409
    assert admin.post(url, json={'class_id': vehicle['id'],
                                 'from_proposal': {'set': r['set'], 'index': 0}}).status_code == 400
    assert admin.post(url, json={'class_id': lesion_id,
                                 'from_proposal': {'set': r['set'], 'index': 5}}).status_code == 400
    assert len(admin.ok('get', url)) == 1
    actions = [e['action'] for e in admin.ok('get', '/api/audit', params={'action': 'item.'})]
    assert 'item.auto_label' in actions


def test_proposals_belong_to_their_item_and_user(admin, auto_project):
    p, m, item, vehicle, _ = auto_project
    other = _items(admin, p['id'])[0]
    lab = _labeller(admin)
    for it in (item, other):
        admin.ok('patch', f"/api/items/{it['id']}", json={'assignee_id': lab.user['id']})
    r = admin.ok('post', f"/api/items/{item['id']}/auto-label")
    ref = {'set': r['set'], 'index': 0}
    cid = r['proposals'][0]['class_id']
    assert admin.post(f"/api/items/{other['id']}/annotations",
                      json={'class_id': cid, 'from_proposal': ref}).status_code == 400
    assert lab.post(f"/api/items/{item['id']}/annotations",
                    json={'class_id': cid, 'from_proposal': ref}).status_code == 400
    # Labellers can auto-label their own items.
    r = lab.ok('post', f"/api/items/{item['id']}/auto-label")
    lab.ok('post', f"/api/items/{item['id']}/annotations", 201,
           json={'class_id': cid, 'from_proposal': {'set': r['set'], 'index': 0}})


def test_semantic_projects_get_one_proposal_per_class(admin, monkeypatch):
    p = _project(admin, 'semantic_segmentation')
    cls = _class(admin, p['id'], label_type='mask')
    items = _items(admin, p['id'], n=10)
    mask = np.zeros((120, 160), bool)
    mask[40:80, 60:100] = True
    admin.ok('post', f"/api/items/{items[0]['id']}/annotations", 201, json={
        'class_id': cls['id'], 'geometry': {'type': 'mask', 'rle': geometry.rle_encode(mask)}})
    conn = _ctx(admin).db.connect()
    with conn:
        conn.execute("UPDATE items SET status = 'done' WHERE project_id = ?", (p['id'],))
    conn.close()
    m = _train(admin, p, kind='automatic')
    admin.ok('patch', f"/api/projects/{p['id']}", json={'auto_model_id': m['id']})
    monkeypatch.setattr(_ctx(admin).manager, 'adapter_net', lambda *a, **k: ('net', 'prompts'))

    def predict(net, prompts, image, device):
        probs = np.zeros((1, *image.shape[:2]), np.float32)
        probs[0, 5:20, 5:20] = probs[0, 50:90, 50:90] = 0.9
        return probs
    monkeypatch.setattr(promptless, 'predict', predict)
    r = admin.ok('post', f"/api/items/{items[0]['id']}/auto-label")
    assert [pr['class_id'] for pr in r['proposals']] == [cls['id']]


def test_auto_label_needs_a_chosen_model(admin):
    p, items = _seed_objects(admin, n_items=1, per_item=1)
    r = admin.post(f"/api/items/{items[0]['id']}/auto-label")
    assert r.status_code == 400 and 'no automatic model' in r.json()['detail']
