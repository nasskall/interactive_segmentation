"""tools/reset_password.py: setting a password from the server's command line."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from server.labelling import audit, auth
from server.labelling.db import Database, now

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import reset_password  # noqa: E402


@pytest.fixture
def data_dir(tmp_path):
    db = Database(tmp_path / 'labelling.db')
    conn = db.connect()
    with conn:
        conn.execute("INSERT INTO users (username, display_name, password_hash, role, created_at) "
                     "VALUES ('admin', 'Admin', ?, 'admin', ?)",
                     (auth.hash_password('old password!'), now()))
        auth.issue_token(conn, 1)
    conn.close()
    return tmp_path


def _run(monkeypatch, data_dir, username, *answers):
    replies = iter(answers)
    monkeypatch.setattr(reset_password.getpass, 'getpass', lambda prompt='': next(replies))
    monkeypatch.setattr(sys, 'argv', ['reset_password.py', username, '--data-dir', str(data_dir)])
    return reset_password.main()


def _state(data_dir):
    conn = Database(data_dir / 'labelling.db').connect()
    user = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    tokens = conn.execute('SELECT COUNT(*) FROM tokens').fetchone()[0]
    log = [dict(r) for r in conn.execute('SELECT * FROM audit_log')]
    chain = audit.verify(conn)
    conn.close()
    return user, tokens, log, chain


def test_reset_sets_the_password_signs_out_and_is_audited(monkeypatch, data_dir):
    assert _run(monkeypatch, data_dir, 'admin', 'new password 123', 'new password 123') == 0
    user, tokens, log, chain = _state(data_dir)
    assert auth.verify_password('new password 123', user['password_hash'])
    assert tokens == 0
    assert [(r['action'], r['username'], r['target']) for r in log] == \
        [('user.update', '(command line)', 'user:1')]
    assert chain['ok']


@pytest.mark.parametrize('username, answers', [
    ('nobody', ()),                                         # unknown account
    ('admin', ('short',)),                                  # under 12 characters
    ('admin', ('new password 123', 'different 12345')),     # entries differ
])
def test_refusals_change_nothing(monkeypatch, data_dir, username, answers):
    assert _run(monkeypatch, data_dir, username, *answers) == 1
    user, tokens, log, _ = _state(data_dir)
    assert auth.verify_password('old password!', user['password_hash'])
    assert tokens == 1 and log == []
