"""Labeller accounts: password hashing and bearer tokens.

Passwords are hashed with scrypt from the standard library. Login returns a
random token; only its SHA-256 is stored, so the database alone cannot be
used to sign in.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

TOKEN_LIFETIME = timedelta(days=7)
MIN_PASSWORD_LENGTH = 8
_SCRYPT = {'n': 2 ** 14, 'r': 8, 'p': 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f'scrypt${salt.hex()}${digest.hex()}'


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split('$')
    except ValueError:
        return False
    if scheme != 'scrypt':
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue_token(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + TOKEN_LIFETIME
    conn.execute('DELETE FROM tokens WHERE expires_at < ?',
                 (datetime.now(timezone.utc).isoformat(),))
    conn.execute('INSERT INTO tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)',
                 (_token_hash(token), user_id, expires.isoformat()))
    return token


def revoke_token(conn: sqlite3.Connection, token: str) -> None:
    conn.execute('DELETE FROM tokens WHERE token_hash = ?', (_token_hash(token),))


def user_for_token(conn: sqlite3.Connection, token: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        'SELECT u.* FROM tokens t JOIN users u ON u.id = t.user_id '
        'WHERE t.token_hash = ? AND t.expires_at > ? AND u.active = 1',
        (_token_hash(token), datetime.now(timezone.utc).isoformat()),
    ).fetchone()
