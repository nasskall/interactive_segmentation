"""Second sign-in factor: authenticator setup, codes, recovery codes,
throttling, keys at rest, the database upgrade and the console reset."""

from __future__ import annotations

import base64
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('segno')

from server.labelling import audit, auth, totp  # noqa: E402
from server.labelling.db import Database  # noqa: E402
from test_labelling import (PASSWORD, Api, _labeller, finish_sign_in,  # noqa: E402,F401
                            admin, app_factory, client, next_code, sign_in)


def _login(client, username='admin', password=PASSWORD):
    r = client.post('/api/auth/login', json={'username': username, 'password': password})
    assert r.status_code == 200, r.text
    return r.json()


def _actions(admin, prefix):
    return [e['action'] for e in admin.ok('get', '/api/audit', params={'action': prefix})]


def test_password_alone_gives_no_session(admin):
    step = _login(admin.client)
    assert step['second_factor'] == 'code' and 'token' not in step
    # The ticket between the two steps is not a session.
    r = admin.client.get('/api/auth/me', headers={'Authorization': f"Bearer {step['challenge']}"})
    assert r.status_code == 401


def test_first_sign_in_sets_up_the_authenticator(admin):
    users = admin.post('/api/users', json={
        'username': 'lab1', 'password': PASSWORD, 'display_name': 'Lab'})
    assert users.status_code == 201 and users.json()['two_factor'] is False
    step = _login(admin.client, 'lab1')
    assert step['second_factor'] == 'enrol'
    # Activating needs the key from the enrol step first.
    r = admin.client.post('/api/auth/2fa/activate', json={'challenge': step['challenge'],
                                                         'code': '123456'})
    assert r.status_code == 401
    e = admin.client.post('/api/auth/2fa/enrol', json={'challenge': step['challenge']}).json()
    assert e['uri'].startswith('otpauth://totp/') and e['qr'].startswith('data:image/svg+xml')
    key = base64.b32decode(e['secret'].replace(' ', '') + '=' * (-len(e['secret'].replace(' ', '')) % 8))
    assert admin.client.post('/api/auth/2fa/activate', json={
        'challenge': step['challenge'], 'code': '000000'}).status_code == 401
    totp.clock.t += totp.STEP
    done = admin.client.post('/api/auth/2fa/activate', json={
        'challenge': step['challenge'], 'code': totp.code_at(key, totp.current_step())})
    assert done.status_code == 200, done.text
    body = done.json()
    assert body['user']['two_factor'] is True and len(body['recovery_codes']) == 10
    assert len(set(body['recovery_codes'])) == 10
    # The ticket is used up.
    assert admin.client.post('/api/auth/2fa/enrol', json={'challenge': step['challenge']}
                             ).status_code == 401
    assert 'auth.2fa_enrolled' in _actions(admin, 'auth.')


def test_codes_are_checked_and_never_reused(admin):
    c = admin.client
    step = _login(c)
    key_code = next_code(c, 'admin')
    ok = c.post('/api/auth/login/verify', json={'challenge': step['challenge'], 'code': key_code})
    assert ok.status_code == 200 and ok.json()['recovery_codes_left'] == 10
    # The same code again, within its 30 seconds: refused.
    step = _login(c)
    assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                  'code': key_code}).status_code == 401
    # Formatting is forgiven.
    code = next_code(c, 'admin')
    r = c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                               'code': f'{code[:3]} {code[3:]}'})
    assert r.status_code == 200


def test_wrong_codes_count_towards_the_lockout(admin):
    c = admin.client
    step = _login(c)
    for _ in range(auth.MAX_FAILURES - 1):
        assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                      'code': '000000'}).status_code == 401
    # A ticket is void after MAX_CHALLENGE_ATTEMPTS wrong codes ...
    step2 = _login(c)
    r = c.post('/api/auth/login/verify', json={'challenge': step2['challenge'], 'code': '000000'})
    assert r.status_code == 401
    # ... and the account is now locked, password and code alike.
    assert c.post('/api/auth/login', json={'username': 'admin', 'password': PASSWORD}
                  ).status_code == 429
    actions = _actions(admin, 'auth.')
    assert actions.count('auth.2fa_failed') == auth.MAX_FAILURES


def test_a_ticket_expires_after_its_attempts(admin):
    c = admin.client
    step = _login(c)
    with closing(c.app.state.labelling.db.connect()) as conn, conn:
        conn.execute('UPDATE login_challenges SET attempts = ?', (auth.MAX_CHALLENGE_ATTEMPTS - 1,))
    assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                  'code': '000000'}).status_code == 401
    r = c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                               'code': next_code(c, 'admin')})
    assert r.status_code == 401 and 'expired' in r.json()['detail']


def test_recovery_codes_work_once_and_can_be_renewed(app_factory):
    from fastapi.testclient import TestClient
    c = TestClient(app_factory())
    r = c.post('/api/auth/setup', json={'setup_token': c.app.state.labelling.setup_token,
                                        'username': 'admin', 'password': PASSWORD,
                                        'display_name': 'Admin'})
    first = finish_sign_in(c, r)
    codes = first['recovery_codes']
    me = c.get('/api/auth/me', headers={'Authorization': f"Bearer {first['token']}"}).json()
    assert me['recovery_codes_left'] == 10 and me['two_factor'] is True

    step = _login(c)
    r = c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                               'code': codes[0].upper().replace('-', ' ')})
    assert r.status_code == 200 and r.json()['recovery_codes_left'] == 9
    step = _login(c)
    assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                  'code': codes[0]}).status_code == 401
    assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                  'code': 'abcd-efgh'}).status_code == 401

    h = {'Authorization': f"Bearer {r.json()['token']}"}
    assert c.post('/api/auth/2fa/recovery-codes', headers=h, json={'code': codes[1]}
                  ).status_code == 400           # needs the app, not a recovery code
    assert c.get('/api/auth/me', headers=h).status_code == 200   # still signed in
    new = c.post('/api/auth/2fa/recovery-codes', headers=h, json={'code': next_code(c, 'admin')})
    assert new.status_code == 200 and len(new.json()['recovery_codes']) == 10
    assert not set(new.json()['recovery_codes']) & set(codes)
    step = _login(c)
    assert c.post('/api/auth/login/verify', json={'challenge': step['challenge'],
                                                  'code': codes[2]}).status_code == 401


def test_keys_are_stored_encrypted(admin, tmp_path):
    db = admin.client.app.state.labelling.db
    key_file = db.path.parent / 'totp.key'
    assert key_file.exists() and (key_file.stat().st_mode & 0o077) == 0
    with closing(db.connect()) as conn:
        row = conn.execute("SELECT totp_secret FROM users WHERE username = 'admin'").fetchone()
        hashes = [r[0] for r in conn.execute('SELECT code_hash FROM recovery_codes')]
    assert row['totp_secret'].startswith('v1$')
    plain = totp.SecretBox(key_file).open(row['totp_secret'])
    assert base64.b32encode(plain).decode().rstrip('=') not in row['totp_secret'].upper()
    assert all(h.startswith('scrypt$') for h in hashes)
    # Another key file cannot open it.
    with pytest.raises(ValueError):
        totp.SecretBox(tmp_path / 'other.key').open(row['totp_secret'])


def test_upgrade_adds_the_columns_and_signs_everyone_out(tmp_path):
    path = tmp_path / 'old.db'
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
            CREATE TABLE tokens (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL);
            INSERT INTO users VALUES (1, 'old', 'Old', 'x', 'admin', 1, '2026-01-01');
            INSERT INTO tokens VALUES ('h', 1, '2999-01-01');
        """)
    db = Database(path)
    with closing(db.connect()) as conn:
        cols = {r['name'] for r in conn.execute('PRAGMA table_info(users)')}
        assert {'totp_secret', 'totp_last_step'} <= cols
        assert conn.execute('SELECT COUNT(*) FROM tokens').fetchone()[0] == 0
    Database(path)  # a second start changes nothing more


def test_console_reset(admin):
    lab = _labeller(admin)
    data_dir = admin.client.app.state.labelling.db.path.parent
    out = subprocess.run([sys.executable, 'tools/reset_2fa.py', 'lab1', '--data-dir', str(data_dir)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert lab.get('/api/auth/me').status_code == 401          # signed out
    assert _login(admin.client, 'lab1')['second_factor'] == 'enrol'
    with closing(admin.client.app.state.labelling.db.connect()) as conn:
        assert audit.verify(conn)['ok']
        assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE action = 'auth.2fa_reset'"
                            ).fetchone()[0] == 1
