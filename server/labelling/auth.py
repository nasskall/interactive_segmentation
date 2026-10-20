"""Labeller accounts: password hashing, bearer tokens and sign-in throttling.

Passwords are hashed with scrypt from the standard library. Login returns a
random token; only its SHA-256 is stored, so the database alone cannot be
used to sign in. Tokens expire after TOKEN_LIFETIME (a working day).

``LoginThrottle`` locks a username, and separately a client address, for
LOCKOUT after MAX_FAILURES failed sign-ins within WINDOW, which stops
online password guessing (ISO 27001 A.8.5).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

TOKEN_LIFETIME = timedelta(hours=12)
MIN_PASSWORD_LENGTH = 12
MAX_FAILURES = 5
WINDOW = 15 * 60      # seconds
LOCKOUT = 15 * 60     # seconds
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


class LoginThrottle:
    """In-memory failed sign-in counter per key (username or client IP).
    It resets on restart, which is acceptable for a single lab server."""

    def __init__(self, max_failures: int = MAX_FAILURES, window: float = WINDOW,
                 lockout: float = LOCKOUT, clock=time.monotonic):
        self.max_failures, self.window, self.lockout = max_failures, window, lockout
        self.clock = clock
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked(self, *keys: str) -> float:
        """Seconds until every key is unlocked (0 when none is locked)."""
        t = self.clock()
        with self._lock:
            return max([self._locked_until.get(k, 0) - t for k in keys] + [0])

    def failed(self, *keys: str) -> bool:
        """Count a failure; True when it locks one of the keys."""
        t = self.clock()
        locked = False
        with self._lock:
            for k in keys:
                recent = [f for f in self._failures.get(k, []) if t - f < self.window] + [t]
                if len(recent) >= self.max_failures:
                    self._locked_until[k] = t + self.lockout
                    recent, locked = [], True
                self._failures[k] = recent
        return locked

    def succeeded(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._failures.pop(k, None)
                self._locked_until.pop(k, None)
