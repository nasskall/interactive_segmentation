"""Append-only, tamper-evident audit log (ISO 27001 A.8.15 logging).

Every security-relevant event (sign-ins, failed sign-ins, lockouts, account
and role changes, project / item / annotation changes, exports, model
loads) is a row in ``audit_log``. Each row stores the SHA-256 of the
previous row's hash plus its own content, so editing or deleting a row in
the database breaks the chain from that point on; ``verify`` finds the first
broken row. Rows are never updated or deleted by the application.

Successful actions are recorded on the request's own connection, so the
entry commits or rolls back with the change it describes. Failures (wrong
password, lockout) are recorded on a separate connection because the
request itself ends in an error.

Chaining under concurrency: a row is inserted first, which takes SQLite's
write lock for the rest of the transaction, and only then is the previous
row's hash read and the new row's hash filled in. No other connection can
add a row in between, so two rows never chain from the same predecessor.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    at        TEXT NOT NULL,
    user_id   INTEGER,
    username  TEXT,
    action    TEXT NOT NULL,
    target    TEXT NOT NULL DEFAULT '',
    detail    TEXT NOT NULL DEFAULT '{}',
    ip        TEXT NOT NULL DEFAULT '',
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
"""

GENESIS = '0' * 64
# Client address of the current request, set by a router dependency.
client_ip: contextvars.ContextVar[str] = contextvars.ContextVar('client_ip', default='')


def _digest(prev_hash: str, at: str, user_id, username, action: str, target: str,
            detail: str, ip: str) -> str:
    payload = json.dumps([prev_hash, at, user_id, username, action, target, detail, ip],
                         separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def record(conn: sqlite3.Connection, action: str, *, user: Optional[sqlite3.Row] = None,
           username: Optional[str] = None, target: str = '', detail: Optional[dict] = None,
           ip: Optional[str] = None) -> None:
    """Append one event. ``user`` is the acting account (or None for an
    anonymous attempt, with ``username`` as typed); ``ip`` defaults to the
    current request's client address."""
    ip = client_ip.get() if ip is None else ip
    user_id = user['id'] if user is not None else None
    username = user['username'] if user is not None else username
    detail_s = json.dumps(detail or {}, sort_keys=True, ensure_ascii=False)
    at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    cur = conn.execute(
        'INSERT INTO audit_log (at, user_id, username, action, target, detail, ip, '
        "prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, '', '')",
        (at, user_id, username, action, target, detail_s, ip))
    row = conn.execute('SELECT hash FROM audit_log WHERE id < ? ORDER BY id DESC LIMIT 1',
                       (cur.lastrowid,)).fetchone()
    prev = row['hash'] if row else GENESIS
    conn.execute('UPDATE audit_log SET prev_hash = ?, hash = ? WHERE id = ?',
                 (prev, _digest(prev, at, user_id, username, action, target, detail_s, ip),
                  cur.lastrowid))


def record_now(db, action: str, **kwargs) -> None:
    """``record`` on its own connection, committed at once."""
    conn = db.connect()
    try:
        with conn:
            record(conn, action, **kwargs)
    finally:
        conn.close()


def verify(conn: sqlite3.Connection) -> dict:
    """Walk the chain; report the first row whose hash doesn't match."""
    prev, n = GENESIS, 0
    for r in conn.execute('SELECT * FROM audit_log ORDER BY id'):
        expected = _digest(prev, r['at'], r['user_id'], r['username'], r['action'],
                           r['target'], r['detail'], r['ip'])
        if r['prev_hash'] != prev or r['hash'] != expected:
            return {'ok': False, 'rows': n, 'first_bad_id': r['id']}
        prev, n = r['hash'], n + 1
    return {'ok': True, 'rows': n, 'first_bad_id': None}


def row_out(r: sqlite3.Row) -> dict:
    return {'id': r['id'], 'at': r['at'], 'user_id': r['user_id'], 'username': r['username'],
            'action': r['action'], 'target': r['target'], 'detail': json.loads(r['detail']),
            'ip': r['ip']}
