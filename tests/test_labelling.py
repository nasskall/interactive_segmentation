"""Labelling projects API: accounts, projects, classes, items, annotations,
and SAM/RITM-assisted labelling (against the real RITM checkpoint)."""

from __future__ import annotations

import csv
import io
import json
import zipfile

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402
from server.labelling import geometry  # noqa: E402
from server.sessions import ModelManager  # noqa: E402

PASSWORD = 'correct horse'


def _png(w=160, h=120, color=(40, 40, 160)):
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 4, color, -1)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


class Api:
    """TestClient wrapper that signs requests as one user."""

    def __init__(self, client: TestClient, token: str, user: dict):
        self.client, self.token, self.user = client, token, user

    def __getattr__(self, method):
        def call(url, **kw):
            headers = {'Authorization': f'Bearer {self.token}', **kw.pop('headers', {})}
            return getattr(self.client, method)(url, headers=headers, **kw)
        return call

    def ok(self, method, url, status=200, **kw):
        r = getattr(self, method)(url, **kw)
        assert r.status_code == status, r.text
        return r.json() if r.content else None


@pytest.fixture
def app_factory(fresh_ritm, device, ritm_checkpoint_path, tmp_path):
    def make():
        manager = ModelManager(fresh_ritm, device, 'RITM', ritm_checkpoint_path.name,
                               ritm_checkpoint_path.parent)
        return create_app(manager, data_dir=tmp_path / 'data')
    return make


@pytest.fixture
def client(app_factory):
    return TestClient(app_factory())


@pytest.fixture
def admin(client):
    token = client.app.state.labelling.setup_token
    r = client.post('/api/auth/setup', json={'setup_token': token, 'username': 'admin',
                                             'password': PASSWORD, 'display_name': 'Admin'})
    assert r.status_code == 201, r.text
    return Api(client, r.json()['token'], r.json()['user'])


def _labeller(admin: Api, username='lab1') -> Api:
    user = admin.ok('post', '/api/users', 201, json={
        'username': username, 'password': PASSWORD, 'display_name': username.title()})
    r = admin.client.post('/api/auth/login', json={'username': username, 'password': PASSWORD})
    assert r.status_code == 200, r.text
    return Api(admin.client, r.json()['token'], user)


def _project(admin: Api, task_type='detection', **kw) -> dict:
    return admin.ok('post', '/api/projects', 201,
                    json={'name': kw.pop('name', task_type), 'task_type': task_type, **kw})


def _class(admin: Api, project_id, name='lesion', color='#FF0000', **kw) -> dict:
    return admin.ok('post', f'/api/projects/{project_id}/classes', 201,
                    json={'name': name, 'color': color, **kw})


def _items(admin: Api, project_id, n=1, w=160, h=120) -> list[dict]:
    files = [('images', (f'img{i}.png', _png(w, h), 'image/png')) for i in range(n)]
    return admin.ok('post', f'/api/projects/{project_id}/items', 201, files=files)


# ── accounts ──────────────────────────────────────────────────────────────

def test_first_admin_needs_the_setup_token(client):
    assert client.get('/api/auth/status').json() == {'needs_setup': True}
    body = {'setup_token': 'guess', 'username': 'admin', 'password': PASSWORD,
            'display_name': 'Admin'}
    assert client.post('/api/auth/setup', json=body).status_code == 403

    body['setup_token'] = client.app.state.labelling.setup_token
    r = client.post('/api/auth/setup', json=body)
    assert r.status_code == 201 and r.json()['user']['role'] == 'admin'
    assert client.get('/api/auth/status').json() == {'needs_setup': False}
    assert client.post('/api/auth/setup', json=body).status_code == 409


def test_login_logout_and_unauthenticated_access(admin):
    c = admin.client
    assert c.get('/api/projects').status_code == 401
    assert c.post('/api/auth/login', json={'username': 'admin', 'password': 'nope'}
                  ).status_code == 401
    assert admin.ok('get', '/api/auth/me')['username'] == 'admin'
    assert admin.post('/api/auth/logout').status_code == 204
    assert admin.get('/api/auth/me').status_code == 401


def test_user_management(admin):
    lab = _labeller(admin)
    assert admin.post('/api/users', json={'username': 'LAB1', 'password': PASSWORD,
                                          'display_name': 'x'}).status_code == 409
    assert lab.get('/api/users').status_code == 403
    assert admin.post('/api/users', json={'username': 'x', 'password': PASSWORD,
                                          'display_name': 'x'}).status_code == 422

    # The only admin cannot demote or deactivate themselves.
    assert admin.patch(f"/api/users/{admin.user['id']}", json={'role': 'labeller'}
                       ).status_code == 409

    # Deactivating signs the user out and blocks login.
    admin.ok('patch', f"/api/users/{lab.user['id']}", json={'active': False})
    assert lab.get('/api/auth/me').status_code == 401
    assert admin.client.post('/api/auth/login', json={'username': 'lab1', 'password': PASSWORD}
                             ).status_code == 401


def test_password_change_signs_user_out(admin):
    lab = _labeller(admin)
    admin.ok('patch', f"/api/users/{lab.user['id']}", json={'password': 'another password'})
    assert lab.get('/api/auth/me').status_code == 401
    assert admin.client.post('/api/auth/login', json={'username': 'lab1',
                                                      'password': 'another password'}
                             ).status_code == 200


# ── projects and classes ──────────────────────────────────────────────────

def test_project_and_class_rules(admin):
    p = _project(admin, 'semantic_segmentation', protocol='Outline every lesion.')
    assert p['allowed_label_types'] == ['mask', 'polygon']
    c = _class(admin, p['id'])
    assert (c['label_type'], c['color'], c['position']) == ('mask', '#ff0000', 0)
    assert _class(admin, p['id'], 'ruler', '#00ff00', label_type='polygon')['position'] == 1

    r = admin.post(f"/api/projects/{p['id']}/classes",
                   json={'name': 'box', 'color': '#000000', 'label_type': 'box'})
    assert r.status_code == 400 and 'mask, polygon' in r.json()['detail']
    assert admin.post(f"/api/projects/{p['id']}/classes",
                      json={'name': 'LESION', 'color': '#000000'}).status_code == 409
    assert admin.post(f"/api/projects/{p['id']}/classes",
                      json={'name': 'x', 'color': 'red'}).status_code == 422
    assert admin.post('/api/projects', json={'name': 'Semantic_Segmentation',
                                             'task_type': 'detection'}).status_code == 409

    admin.ok('patch', f"/api/projects/{p['id']}", json={'protocol': 'v2'})
    admin.ok('patch', f"/api/classes/{c['id']}", json={'color': '#00FFAA', 'name': 'mole'})
    classes = admin.ok('get', f"/api/projects/{p['id']}/classes")
    assert [(k['name'], k['color']) for k in classes] == [('mole', '#00ffaa'),
                                                          ('ruler', '#00ff00')]


def test_data_persists_across_restarts(app_factory, admin):
    p = _project(admin)
    _class(admin, p['id'])
    _items(admin, p['id'])

    client = TestClient(app_factory())
    assert client.get('/api/auth/status').json() == {'needs_setup': False}
    again = Api(client, client.post('/api/auth/login', json={
        'username': 'admin', 'password': PASSWORD}).json()['token'], admin.user)
    assert [x['name'] for x in again.ok('get', '/api/projects')] == ['detection']
    item = again.ok('get', f"/api/projects/{p['id']}/items")[0]
    assert again.get(item['images']['image']).status_code == 200


# ── items ─────────────────────────────────────────────────────────────────

def test_upload_single_images(admin):
    p = _project(admin)
    items = _items(admin, p['id'], n=3, w=90, h=70)
    assert [(i['name'], i['width'], i['height'], i['status']) for i in items] == [
        ('img0', 90, 70, 'todo'), ('img1', 90, 70, 'todo'), ('img2', 90, 70, 'todo')]
    r = admin.get(items[0]['images']['image'])
    assert r.headers['content-type'] == 'image/png'
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape == (70, 90, 3)

    bad = admin.post(f"/api/projects/{p['id']}/items",
                     files=[('images', ('notes.txt', b'hello', 'text/plain'))])
    assert bad.status_code == 400 and 'Unsupported image type' in bad.json()['detail']
    broken = admin.post(f"/api/projects/{p['id']}/items",
                        files=[('images', ('x.png', b'not a png', 'image/png'))])
    assert broken.status_code == 400


def test_tiff_is_served_as_png(admin):
    p = _project(admin)
    ok, tif = cv2.imencode('.tif', np.zeros((30, 40, 3), np.uint8))
    item = admin.ok('post', f"/api/projects/{p['id']}/items", 201,
                    files=[('images', ('scan.tif', tif.tobytes(), 'image/tiff'))])[0]
    assert admin.get(item['images']['image']).headers['content-type'] == 'image/png'


def test_change_detection_pairs(admin):
    p = _project(admin, 'change_detection')
    pair = admin.ok('post', f"/api/projects/{p['id']}/items", 201, data={'name': 'site 4'},
                    files=[('before', ('b.png', _png(), 'image/png')),
                           ('after', ('a.png', _png(color=(0, 200, 0)), 'image/png'))])[0]
    assert pair['kind'] == 'pair' and pair['name'] == 'site 4'
    assert set(pair['images']) == {'before', 'after'}

    r = admin.post(f"/api/projects/{p['id']}/items",
                   files=[('before', ('b.png', _png(), 'image/png')),
                          ('after', ('a.png', _png(w=200), 'image/png'))])
    assert r.status_code == 400 and 'same size' in r.json()['detail']
    r = admin.post(f"/api/projects/{p['id']}/items",
                   files=[('images', ('b.png', _png(), 'image/png'))])
    assert r.status_code == 400


def test_delete_item_removes_files(admin):
    p = _project(admin)
    item = _items(admin, p['id'])[0]
    folder = admin.client.app.state.labelling.images.item_dir(item['id'])
    assert folder.is_dir()
    assert admin.delete(f"/api/items/{item['id']}").status_code == 204
    assert not folder.exists()
    assert admin.get(f"/api/items/{item['id']}").status_code == 404


# ── labeller access ───────────────────────────────────────────────────────

def test_labellers_only_see_their_assigned_work(admin):
    lab, other = _labeller(admin), _labeller(admin, 'lab2')
    p = _project(admin)
    _project(admin, 'classification')
    items = _items(admin, p['id'], n=3)
    assert lab.ok('get', '/api/projects') == []

    admin.ok('post', f"/api/projects/{p['id']}/assign",
             json={'user_id': lab.user['id'], 'item_ids': [items[0]['id'], items[1]['id']]})
    admin.ok('patch', f"/api/items/{items[2]['id']}", json={'assignee_id': other.user['id']})

    assert [x['id'] for x in lab.ok('get', '/api/projects')] == [p['id']]
    assert [x['id'] for x in lab.ok('get', f"/api/projects/{p['id']}/items")] == [
        items[0]['id'], items[1]['id']]
    assert lab.get(f"/api/items/{items[2]['id']}").status_code == 404
    assert lab.get(items[2]['images']['image']).status_code == 404
    assert [x['id'] for x in admin.ok('get', f"/api/projects/{p['id']}/items",
                                      params={'assignee_id': other.user['id']})] == [items[2]['id']]

    # Labellers work on items but do not manage projects.
    assert lab.post(f"/api/projects/{p['id']}/classes",
                    json={'name': 'x', 'color': '#000000'}).status_code == 403
    assert lab.patch(f"/api/items/{items[0]['id']}",
                     json={'assignee_id': other.user['id']}).status_code == 403
    assert lab.ok('patch', f"/api/items/{items[0]['id']}", json={'status': 'done'}
                  )['status'] == 'done'

    # Unassigning hides the item again.
    admin.ok('patch', f"/api/items/{items[0]['id']}", json={'assignee_id': None})
    assert lab.get(f"/api/items/{items[0]['id']}").status_code == 404


def test_assign_rejects_unknown_items_and_users(admin):
    p, q = _project(admin), _project(admin, 'classification')
    mine, theirs = _items(admin, p['id'])[0], _items(admin, q['id'])[0]
    assert admin.post(f"/api/projects/{p['id']}/assign",
                      json={'user_id': admin.user['id'], 'item_ids': [mine['id'], theirs['id']]}
                      ).status_code == 400
    assert admin.post(f"/api/projects/{p['id']}/assign",
                      json={'user_id': 999, 'item_ids': [mine['id']]}).status_code == 400


# ── annotations ───────────────────────────────────────────────────────────

def test_box_and_polygon_validation(admin):
    det = _project(admin)
    car = _class(admin, det['id'], 'car')
    item = _items(admin, det['id'])[0]
    url = f"/api/items/{item['id']}/annotations"

    a = admin.ok('post', url, 201, json={'class_id': car['id'],
                                         'geometry': {'type': 'box', 'bbox': [10, 20, 30, 40]}})
    assert a['geometry'] == {'type': 'box', 'bbox': [10.0, 20.0, 30.0, 40.0]}
    assert a['source'] == 'manual' and a['labeller_id'] == admin.user['id']
    assert admin.ok('get', f"/api/items/{item['id']}")['status'] == 'in_progress'

    for geom in ({'type': 'box', 'bbox': [150, 0, 20, 10]},       # past the right edge
                 {'type': 'box', 'bbox': [0, 0, 0, 10]},          # empty
                 {'type': 'polygon', 'points': [[0, 0], [5, 5], [9, 0]]},  # wrong type
                 {'type': 'box'}):
        r = admin.post(url, json={'class_id': car['id'], 'geometry': geom})
        assert r.status_code == 400, geom

    seg = _project(admin, 'instance_segmentation')
    outline = _class(admin, seg['id'], 'cell', label_type='polygon')
    assert admin.post(url, json={'class_id': outline['id'], 'geometry': {
        'type': 'polygon', 'points': [[0, 0], [5, 5], [9, 0]]}}).status_code == 400  # other project
    seg_item = _items(admin, seg['id'])[0]
    seg_url = f"/api/items/{seg_item['id']}/annotations"
    admin.ok('post', seg_url, 201, json={'class_id': outline['id'], 'geometry': {
        'type': 'polygon', 'points': [[0, 0], [50, 5], [20, 60]]}})
    assert admin.post(seg_url, json={'class_id': outline['id'], 'geometry': {
        'type': 'polygon', 'points': [[0, 0], [500, 5], [20, 60]]}}).status_code == 400


def test_mask_annotations_round_trip(admin):
    p = _project(admin, 'semantic_segmentation')
    cls = _class(admin, p['id'])
    item = _items(admin, p['id'])[0]
    mask = np.zeros((120, 160), bool)
    mask[30:90, 40:120] = True
    rle = geometry.rle_encode(mask)
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'geometry': {'type': 'mask', 'rle': rle}})
    np.testing.assert_array_equal(geometry.rle_decode(a['geometry']['rle']), mask)

    wrong_size = geometry.rle_encode(mask[:100])
    assert admin.post(f"/api/items/{item['id']}/annotations", json={
        'class_id': cls['id'], 'geometry': {'type': 'mask', 'rle': wrong_size}}).status_code == 400
    empty = geometry.rle_encode(np.zeros((120, 160), bool))
    assert admin.post(f"/api/items/{item['id']}/annotations", json={
        'class_id': cls['id'], 'geometry': {'type': 'mask', 'rle': empty}}).status_code == 400


def test_single_label_classification_replaces_the_tag(admin):
    p = _project(admin, 'classification')
    benign, malignant = _class(admin, p['id'], 'benign'), _class(admin, p['id'], 'malignant')
    item = _items(admin, p['id'])[0]
    url = f"/api/items/{item['id']}/annotations"
    admin.ok('post', url, 201, json={'class_id': benign['id']})
    admin.ok('post', url, 201, json={'class_id': malignant['id']})
    assert [a['class_id'] for a in admin.ok('get', url)] == [malignant['id']]
    assert admin.post(url, json={'class_id': malignant['id']}).status_code == 409


def test_multi_label_classification_keeps_all_tags(admin):
    p = _project(admin, 'classification', multi_label=True)
    a, b = _class(admin, p['id'], 'hair'), _class(admin, p['id'], 'ruler')
    item = _items(admin, p['id'])[0]
    url = f"/api/items/{item['id']}/annotations"
    admin.ok('post', url, 201, json={'class_id': a['id']})
    admin.ok('post', url, 201, json={'class_id': b['id']})
    assert sorted(x['class_id'] for x in admin.ok('get', url)) == sorted([a['id'], b['id']])


def test_labellers_change_only_their_own_annotations(admin):
    lab, other = _labeller(admin), _labeller(admin, 'lab2')
    p = _project(admin)
    car, bus = _class(admin, p['id'], 'car'), _class(admin, p['id'], 'bus')
    item = _items(admin, p['id'])[0]
    admin.ok('patch', f"/api/items/{item['id']}", json={'assignee_id': lab.user['id']})
    box = {'type': 'box', 'bbox': [1, 1, 10, 10]}
    mine = lab.ok('post', f"/api/items/{item['id']}/annotations", 201,
                  json={'class_id': car['id'], 'geometry': box})
    by_admin = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                        json={'class_id': car['id'], 'geometry': box})

    assert lab.ok('patch', f"/api/annotations/{mine['id']}",
                  json={'class_id': bus['id']})['class_id'] == bus['id']
    assert lab.patch(f"/api/annotations/{by_admin['id']}",
                     json={'class_id': bus['id']}).status_code == 403
    assert other.delete(f"/api/annotations/{mine['id']}").status_code == 404
    assert lab.delete(f"/api/annotations/{mine['id']}").status_code == 204


def test_deleting_a_used_class_needs_force(admin):
    p = _project(admin)
    car = _class(admin, p['id'], 'car')
    item = _items(admin, p['id'])[0]
    admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
             json={'class_id': car['id'], 'geometry': {'type': 'box', 'bbox': [1, 1, 5, 5]}})
    assert admin.delete(f"/api/classes/{car['id']}").status_code == 409
    assert admin.delete(f"/api/classes/{car['id']}", params={'force': True}).status_code == 204
    assert admin.ok('get', f"/api/items/{item['id']}/annotations") == []


def test_progress(admin):
    lab = _labeller(admin)
    p = _project(admin)
    car = _class(admin, p['id'], 'car')
    items = _items(admin, p['id'], n=3)
    admin.ok('post', f"/api/projects/{p['id']}/assign",
             json={'user_id': lab.user['id'], 'item_ids': [items[0]['id'], items[1]['id']]})
    lab.ok('post', f"/api/items/{items[0]['id']}/annotations", 201,
           json={'class_id': car['id'], 'geometry': {'type': 'box', 'bbox': [1, 1, 5, 5]}})
    lab.ok('patch', f"/api/items/{items[0]['id']}", json={'status': 'done'})

    other = _labeller(admin, 'lab2')
    admin.ok('patch', f"/api/items/{items[2]['id']}", json={'assignee_id': other.user['id']})
    assert len(admin.ok('get', f"/api/projects/{p['id']}/progress")['by_labeller']) == 2

    listed = lab.ok('get', f"/api/projects/{p['id']}/items")
    assert [i['annotation_count'] for i in listed] == [1, 0]

    prog = lab.ok('get', f"/api/projects/{p['id']}/progress")
    assert prog['items'] == 3
    assert prog['by_status'] == {'todo': 2, 'in_progress': 0, 'done': 1}
    assert prog['by_labeller'] == [{'user_id': lab.user['id'], 'display_name': 'Lab1',
                                    'items': 2, 'done': 1, 'annotations': 1}]


# ── assisted labelling ────────────────────────────────────────────────────

def test_assist_saves_mask_box_and_polygon_at_full_resolution(admin):
    p = _project(admin, 'change_detection')
    region = _class(admin, p['id'], 'new building', label_type='mask')
    box = _class(admin, p['id'], 'vehicle', label_type='box')
    outline = _class(admin, p['id'], 'road', label_type='polygon')
    # Larger than the 2048 px session limit, so the session is downscaled.
    w, h = 3000, 1500
    item = admin.ok('post', f"/api/projects/{p['id']}/items", 201,
                    files=[('before', ('b.png', _png(w, h), 'image/png')),
                           ('after', ('a.png', _png(w, h), 'image/png'))])[0]
    url = f"/api/items/{item['id']}/annotations"

    for cls in (region, box, outline):
        s = admin.ok('post', f"/api/items/{item['id']}/assist", 201, params={'image': 'after'})
        assert (s['width'], s['original_width']) == (2048, 3000)
        sid = s['session_id']
        admin.ok('post', f'/api/sessions/{sid}/clicks',
                 json={'x': s['width'] // 2, 'y': s['height'] // 2, 'positive': True})
        a = admin.ok('post', url, 201, json={'class_id': cls['id'], 'from_session': sid})
        assert a['source'] == 'assisted'
        # The session is cleared, ready for the next object.
        assert admin.ok('get', f'/api/sessions/{sid}')['clicks'] == []

        g = a['geometry']
        if cls is region:
            m = geometry.rle_decode(g['rle'])
            assert m.shape == (h, w) and m[h // 2, w // 2]
        elif cls is box:
            x, y, bw, bh = g['bbox']
            assert x < w // 2 < x + bw and y < h // 2 < y + bh
        else:
            assert len(g['points']) >= 3

    # An empty session cannot be saved.
    r = admin.post(url, json={'class_id': region['id'], 'from_session': sid})
    assert r.status_code == 400 and 'no mask' in r.json()['detail']


def test_assist_session_is_bound_to_item_and_user(admin):
    lab = _labeller(admin)
    p = _project(admin, 'semantic_segmentation')
    cls = _class(admin, p['id'])
    first, second = _items(admin, p['id'], n=2)
    admin.ok('patch', f"/api/items/{first['id']}", json={'assignee_id': lab.user['id']})
    sid = admin.ok('post', f"/api/items/{first['id']}/assist", 201)['session_id']
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60})

    r = admin.post(f"/api/items/{second['id']}/annotations",
                   json={'class_id': cls['id'], 'from_session': sid})
    assert r.status_code == 400
    r = lab.post(f"/api/items/{first['id']}/annotations",
                 json={'class_id': cls['id'], 'from_session': sid})
    assert r.status_code == 400
    assert admin.post(f"/api/items/{first['id']}/assist", params={'image': 'after'}
                      ).status_code == 400
    assert lab.post(f"/api/items/{second['id']}/assist").status_code == 404


# ── RLE ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('seed', range(5))
def test_rle_round_trip(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((37, 53)) > 0.6
    mask[0, 0] = seed % 2 == 0   # cover both first-pixel cases
    rle = geometry.rle_encode(mask)
    assert sum(rle['counts']) == mask.size
    np.testing.assert_array_equal(geometry.rle_decode(rle), mask)


def test_rle_matches_pycocotools():
    mask_utils = pytest.importorskip('pycocotools.mask')
    mask = np.zeros((40, 60), bool)
    mask[0, 0] = True
    mask[5:30, 10:50] = True
    ours = geometry.rle_encode(mask)
    theirs = mask_utils.frPyObjects(ours, *ours['size'])
    np.testing.assert_array_equal(mask_utils.decode(theirs).astype(bool), mask)
    assert mask_utils.area(theirs) == mask.sum()


# ── exports ───────────────────────────────────────────────────────────────

def _zip(admin: Api, project_id, **params) -> zipfile.ZipFile:
    r = admin.get(f'/api/projects/{project_id}/export', params=params)
    assert r.status_code == 200, r.text
    assert r.headers['content-type'] == 'application/zip'
    return zipfile.ZipFile(io.BytesIO(r.content))


def _png_array(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def _instance_project(admin: Api):
    """Two classes; item 1 has a mask and a polygon, item 2 has nothing."""
    p = _project(admin, 'instance_segmentation', name='cells')
    cell = _class(admin, p['id'], 'cell', '#ff0000', label_type='mask')
    nuc = _class(admin, p['id'], 'nucleus', '#00ff00', label_type='polygon')
    items = _items(admin, p['id'], n=2)
    mask = np.zeros((120, 160), bool)
    mask[10:50, 20:80] = True
    admin.ok('post', f"/api/items/{items[0]['id']}/annotations", 201, json={
        'class_id': cell['id'], 'geometry': {'type': 'mask', 'rle': geometry.rle_encode(mask)}})
    admin.ok('post', f"/api/items/{items[0]['id']}/annotations", 201, json={
        'class_id': nuc['id'],
        'geometry': {'type': 'polygon', 'points': [[100, 60], [140, 60], [140, 100], [100, 100]]}})
    return p, items, mask


def test_export_coco_loads_in_pycocotools(admin, tmp_path):
    coco_api = pytest.importorskip('pycocotools.coco')
    p, items, mask = _instance_project(admin)
    zf = _zip(admin, p['id'], format='coco')
    assert {'annotations.json', f"images/{items[0]['id']}_img0.png",
            f"images/{items[1]['id']}_img1.png"} <= set(zf.namelist())
    (tmp_path / 'ann.json').write_bytes(zf.read('annotations.json'))
    coco = coco_api.COCO(str(tmp_path / 'ann.json'))
    assert [c['name'] for c in coco.loadCats(coco.getCatIds())] == ['cell', 'nucleus']
    assert sorted(coco.getImgIds()) == sorted(i['id'] for i in items)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[items[0]['id']]))
    assert [a['category_id'] for a in anns] == [1, 2]
    np.testing.assert_array_equal(coco.annToMask(anns[0]).astype(bool), mask)
    assert anns[0]['bbox'] == [20, 10, 60, 40] and anns[0]['area'] == mask.sum()
    assert anns[1]['bbox'] == [100, 60, 40, 40] and anns[1]['area'] == 1600
    assert coco.annToMask(anns[1])[80, 120] == 1

    no_images = _zip(admin, p['id'], format='coco', include_images='false')
    assert no_images.namelist() == ['annotations.json']


def test_export_status_filter(admin):
    p, items, _ = _instance_project(admin)
    admin.ok('patch', f"/api/items/{items[0]['id']}", json={'status': 'done'})
    data = json.loads(_zip(admin, p['id'], format='coco', status='done').read('annotations.json'))
    assert [i['id'] for i in data['images']] == [items[0]['id']]
    assert len(data['annotations']) == 2


def test_export_masks(admin):
    from PIL import Image
    p, items, mask = _instance_project(admin)
    zf = _zip(admin, p['id'], format='masks', include_images='false')
    stem = f"{items[0]['id']}_img0"
    cls_png = Image.open(io.BytesIO(zf.read(f'masks/{stem}.png')))
    assert cls_png.mode == 'P' and cls_png.getpalette()[3:9] == [255, 0, 0, 0, 255, 0]
    cls_map = np.array(cls_png)
    assert cls_map[30, 50] == 1 and cls_map[80, 120] == 2 and cls_map[110, 5] == 0
    assert (cls_map == 1).sum() == mask.sum()
    inst = _png_array(zf.read(f'instances/{stem}.png'))
    assert inst.dtype == np.uint16 and set(np.unique(inst)) == {0, 1, 2}
    empty = np.array(Image.open(io.BytesIO(zf.read(f"masks/{items[1]['id']}_img1.png"))))
    assert not empty.any()
    assert zf.read('classes.csv').decode().splitlines() == [
        'index,name,color', '0,background,#000000', '1,cell,#ff0000', '2,nucleus,#00ff00']


def test_export_yolo_detection_and_segmentation(admin):
    p = _project(admin, 'detection')
    _class(admin, p['id'], 'car')
    person = _class(admin, p['id'], 'person')
    item = _items(admin, p['id'])[0]
    admin.ok('post', f"/api/items/{item['id']}/annotations", 201, json={
        'class_id': person['id'], 'geometry': {'type': 'box', 'bbox': [40, 30, 80, 60]}})
    zf = _zip(admin, p['id'], format='yolo')
    assert f"images/{item['id']}_img0.png" in zf.namelist()
    assert zf.read(f"labels/{item['id']}_img0.txt").decode() == \
        '1 0.500000 0.500000 0.500000 0.500000\n'
    data_yaml = zf.read('data.yaml').decode()
    assert '0: "car"' in data_yaml and '1: "person"' in data_yaml

    p, items, _ = _instance_project(admin)
    zf = _zip(admin, p['id'], format='yolo')
    lines = zf.read(f"labels/{items[0]['id']}_img0.txt").decode().splitlines()
    assert [ln.split()[0] for ln in lines] == ['0', '1']
    xs = [float(v) for v in lines[0].split()[1::2]]
    ys = [float(v) for v in lines[0].split()[2::2]]
    assert min(xs) == pytest.approx(20 / 160, abs=1e-6)
    assert max(ys) == pytest.approx(49 / 120, abs=1e-6)
    assert zf.read(f"labels/{items[1]['id']}_img1.txt") == b''


def test_export_csv(admin):
    p = _project(admin, 'classification', multi_label=True)
    cat = _class(admin, p['id'], 'cat')
    dog = _class(admin, p['id'], 'dog')
    items = _items(admin, p['id'], n=2)
    for c in (cat, dog):
        admin.ok('post', f"/api/items/{items[0]['id']}/annotations", 201, json={'class_id': c['id']})
    r = admin.get(f"/api/projects/{p['id']}/export", params={'format': 'csv'})
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/csv')
    assert 'classification_csv.csv' in r.headers['content-disposition']
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert [(int(x['item_id']), x['class'], x['labeller']) for x in rows] == [
        (items[0]['id'], 'cat', 'Admin'), (items[0]['id'], 'dog', 'Admin'),
        (items[1]['id'], '', '')]
    assert rows[0]['bbox_x'] == '' and rows[0]['files'] == f"{items[0]['id']}_img0.png"


def test_export_change_detection_pairs(admin):
    p = _project(admin, 'change_detection')
    cls = _class(admin, p['id'], 'new building', label_type='box')
    item = admin.ok('post', f"/api/projects/{p['id']}/items", 201, files=[
        ('before', ('site_before.png', _png(), 'image/png')),
        ('after', ('site_after.png', _png(color=(0, 200, 0)), 'image/png'))])[0]
    admin.ok('post', f"/api/items/{item['id']}/annotations", 201, json={
        'class_id': cls['id'], 'geometry': {'type': 'box', 'bbox': [10, 10, 20, 30]}})
    zf = _zip(admin, p['id'], format='coco')
    stem = f"{item['id']}_{item['name']}"
    assert {f'images/{stem}_before.png', f'images/{stem}_after.png'} <= set(zf.namelist())
    image = json.loads(zf.read('annotations.json'))['images'][0]
    assert image['file_name'] == f'{stem}_after.png'
    assert image['before_file_name'] == f'{stem}_before.png'
    from PIL import Image
    masks = np.array(Image.open(io.BytesIO(
        _zip(admin, p['id'], format='masks').read(f'masks/{stem}.png'))))
    assert masks[10:40, 10:30].all() and masks.sum() == 600


def test_export_rules(admin):
    p = _project(admin, 'classification')
    for fmt in ('coco', 'masks', 'yolo'):
        r = admin.get(f"/api/projects/{p['id']}/export", params={'format': fmt})
        assert r.status_code == 400 and 'csv' in r.json()['detail']
    det = _project(admin, 'detection')
    assert admin.get(f"/api/projects/{det['id']}/export",
                     params={'format': 'masks'}).status_code == 400
    assert admin.get(f"/api/projects/{det['id']}/export",
                     params={'format': 'pdf'}).status_code == 422
    lab = _labeller(admin)
    item = _items(admin, det['id'])[0]
    admin.ok('post', f"/api/projects/{det['id']}/assign",
             json={'user_id': lab.user['id'], 'item_ids': [item['id']]})
    assert lab.get(f"/api/projects/{det['id']}/export",
                   params={'format': 'csv'}).status_code == 403
