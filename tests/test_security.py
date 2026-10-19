"""Security controls (ISO 27001 / 42001 support): sign-in on every endpoint,
session ownership, admin-only model switching, the hash-chained audit log,
sign-in throttling, model provenance on assisted annotations, security
headers and the database migration."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip('fastapi')

from server.labelling import audit, auth  # noqa: E402
from server.labelling.db import Database  # noqa: E402
from test_labelling import (PASSWORD, _class, _items, _labeller, _project,  # noqa: E402,F401
                            admin, app_factory, client)


def _png_upload():
    from test_labelling import _png
    return {'image': ('x.png', _png(), 'image/png')}


def _actions(admin, prefix=''):
    return [e['action'] for e in admin.ok('get', '/api/audit', params={'action': prefix})]


# ── access control ────────────────────────────────────────────────────────

def test_segmentation_endpoints_need_sign_in(admin):
    c = admin.client
    assert c.get('/api/models').status_code == 401
    assert c.post('/api/sessions', files=_png_upload()).status_code == 401
    sid = admin.ok('post', '/api/sessions', 201, files=_png_upload())['session_id']
    for method, url in [('get', f'/api/sessions/{sid}'), ('get', f'/api/sessions/{sid}/image'),
                        ('post', f'/api/sessions/{sid}/undo'),
                        ('get', f'/api/sessions/{sid}/mask'),
                        ('delete', f'/api/sessions/{sid}')]:
        assert getattr(c, method)(url).status_code == 401, url


def test_sessions_are_private_to_their_owner(admin):
    lab = _labeller(admin)
    sid = lab.ok('post', '/api/sessions', 201, files=_png_upload())['session_id']
    lab.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60})
    # Not even an admin can read or drive someone else's session.
    assert admin.get(f'/api/sessions/{sid}').status_code == 404
    assert admin.post(f'/api/sessions/{sid}/clicks', json={'x': 1, 'y': 1}).status_code == 404
    assert admin.delete(f'/api/sessions/{sid}').status_code == 404
    assert len(lab.ok('get', f'/api/sessions/{sid}')['clicks']) == 1


def test_only_admins_switch_the_model_and_it_is_logged(admin):
    lab = _labeller(admin)
    ckpt = admin.ok('get', '/api/models')['current']['checkpoint']
    body = {'model_type': 'RITM', 'checkpoint': ckpt}
    assert lab.post('/api/models/load', json=body).status_code == 403
    admin.ok('post', '/api/models/load', json=body)
    entry = admin.ok('get', '/api/audit', params={'action': 'model.'})[0]
    assert entry['username'] == 'admin'
    assert entry['detail']['checkpoint'] == ckpt and len(entry['detail']['sha256']) == 64


# ── audit log ─────────────────────────────────────────────────────────────

def test_audit_log_records_changes_and_verifies(admin):
    lab = _labeller(admin)
    p = _project(admin, 'semantic_segmentation')
    _class(admin, p['id'])
    _items(admin, p['id'])
    assert lab.get('/api/audit').status_code == 403
    admin.ok('patch', f"/api/users/{lab.user['id']}", json={'password': 'a new long password'})
    log = admin.ok('get', '/api/audit')
    actions = [e['action'] for e in log]
    for a in ('auth.setup', 'user.create', 'auth.login', 'project.create', 'class.create',
              'item.upload', 'user.update'):
        assert a in actions, a
    upd = next(e for e in log if e['action'] == 'user.update')
    assert upd['detail'] == {'password_changed': True}          # never the password itself
    assert upd['ip'] == 'testclient'
    assert admin.ok('get', '/api/audit/verify')['ok'] is True


def test_editing_or_deleting_an_audit_row_breaks_the_chain(admin):
    _project(admin)
    _project(admin, name='second')
    db_path = admin.client.app.state.labelling.db.path
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE audit_log SET username = 'someone else' WHERE id = 2")
    bad = admin.ok('get', '/api/audit/verify')
    assert bad == {'ok': False, 'rows': 1, 'first_bad_id': 2}

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE audit_log SET username = 'admin' WHERE id = 2")
    assert admin.ok('get', '/api/audit/verify')['ok']
    with sqlite3.connect(db_path) as conn:
        conn.execute('DELETE FROM audit_log WHERE id = 2')
    assert admin.ok('get', '/api/audit/verify')['first_bad_id'] == 3


def test_concurrent_writers_keep_one_chain(tmp_path):
    import threading
    db = Database(tmp_path / 'l.db')

    def write(n):
        for i in range(n):
            audit.record_now(db, 'test.event', username=f'u{i}', ip='')
    threads = [threading.Thread(target=write, args=(25,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with db.connect() as conn:
        assert audit.verify(conn) == {'ok': True, 'rows': 100, 'first_bad_id': None}


# ── sign-in throttling ────────────────────────────────────────────────────

def test_repeated_failed_sign_ins_lock_the_account(admin):
    c = admin.client
    for _ in range(auth.MAX_FAILURES):
        assert c.post('/api/auth/login', json={'username': 'admin', 'password': 'wrong one'}
                      ).status_code == 401
    r = c.post('/api/auth/login', json={'username': 'admin', 'password': PASSWORD})
    assert r.status_code == 429 and 'Retry-After' in r.headers
    actions = _actions(admin, 'auth.')
    assert actions.count('auth.login_failed') == auth.MAX_FAILURES
    assert 'auth.login_blocked' in actions


def test_throttle_window_and_lockout_expire():
    t = [0.0]
    th = auth.LoginThrottle(max_failures=3, window=60, lockout=100, clock=lambda: t[0])
    assert not th.failed('k') and not th.failed('k')
    t[0] = 61                                   # the first two fall out of the window
    assert not th.failed('k') and th.locked('k') == 0
    assert not th.failed('k') and th.failed('k')
    assert th.locked('k') == pytest.approx(100)
    t[0] += 100
    assert th.locked('k') == 0
    th.failed('k')
    th.succeeded('k')
    assert not th.failed('k') and not th.failed('k')


def test_password_policy(admin):
    r = admin.post('/api/users', json={'username': 'short', 'password': 'elevenchars',
                                       'display_name': 'S'})
    assert r.status_code == 422


# ── AI provenance (ISO 42001) ─────────────────────────────────────────────

def test_assisted_annotations_record_the_model(admin):
    p = _project(admin, 'semantic_segmentation')
    cls = _class(admin, p['id'], label_type='mask')
    item = _items(admin, p['id'])[0]
    sid = admin.ok('post', f"/api/items/{item['id']}/assist", 201)['session_id']
    admin.ok('post', f'/api/sessions/{sid}/clicks', json={'x': 80, 'y': 60})
    a = admin.ok('post', f"/api/items/{item['id']}/annotations", 201,
                 json={'class_id': cls['id'], 'from_session': sid})
    assert a['source'] == 'assisted'
    assert a['model']['model_type'] == 'RITM' and len(a['model']['sha256']) == 64
    r = admin.get(f"/api/projects/{p['id']}/export", params={'format': 'csv'})
    import csv, io
    row = next(csv.DictReader(io.StringIO(r.text)))
    assert row['model'].startswith('RITM ') and row['model_sha256'] == a['model']['sha256']
    assert 'project.export' in _actions(admin, 'project.')
    tag_p = _project(admin, 'classification', name='tags')
    tag = _class(admin, tag_p['id'], 'cat')
    tag_item = _items(admin, tag_p['id'])[0]
    m = admin.ok('post', f"/api/items/{tag_item['id']}/annotations", 201,
                 json={'class_id': tag['id']})
    assert m['source'] == 'manual' and m['model'] is None


def test_old_databases_gain_the_model_column(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE annotations (id INTEGER PRIMARY KEY, item_id INTEGER, '
                     'class_id INTEGER, labeller_id INTEGER, geometry TEXT, source TEXT, '
                     'created_at TEXT, updated_at TEXT)')
    Database(path)
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute('PRAGMA table_info(annotations)')}
    assert 'model' in cols


# ── transport ─────────────────────────────────────────────────────────────

def test_security_headers(admin):
    r = admin.get('/api/auth/me')
    assert r.headers['X-Frame-Options'] == 'DENY'
    assert r.headers['X-Content-Type-Options'] == 'nosniff'
    assert "frame-ancestors 'none'" in r.headers['Content-Security-Policy']
    assert r.headers['Cache-Control'] == 'no-store'
    assert 'Strict-Transport-Security' not in r.headers       # plain HTTP in tests
