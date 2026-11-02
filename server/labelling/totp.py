"""Second sign-in factor: one-time codes from an authenticator app (TOTP).

Codes follow RFC 6238 as authenticator apps implement it: HMAC-SHA1 over the
number of 30-second steps since 1970, six digits. A code is accepted for the
current step and one step either side (clock drift), and never twice: each
account remembers the last step it used, so a code seen over someone's
shoulder cannot be replayed.

Each account's authenticator key is stored encrypted (``SecretBox``) with a
key kept in a separate file next to the database. A copy of the database
alone, such as a backup, then gives no way to generate codes.

Recovery codes (``new_recovery_codes``) stand in for a lost phone. Each one
works once and is stored hashed like a password.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import secrets
import struct
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

STEP = 30                 # seconds per code
DIGITS = 6
DRIFT = 1                 # steps accepted either side of now
ISSUER = 'Interactive Labelling'
RECOVERY_CODES = 10
_RECOVERY_ALPHABET = 'abcdefghjkmnpqrstuvwxyz23456789'   # no 0/o, 1/l/i

clock = time.time         # replaced by tests


# ── codes ─────────────────────────────────────────────────────────────────

def new_secret() -> bytes:
    """A 160-bit key, the size RFC 4226 recommends for HMAC-SHA1."""
    return secrets.token_bytes(20)


def secret_text(secret: bytes) -> str:
    """The key as the base32 text authenticator apps accept, in groups of 4."""
    b32 = base64.b32encode(secret).decode().rstrip('=')
    return ' '.join(b32[i:i + 4] for i in range(0, len(b32), 4))


def code_at(secret: bytes, step: int) -> str:
    mac = hmac.new(secret, struct.pack('>Q', step), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    value = struct.unpack('>I', mac[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10 ** DIGITS).zfill(DIGITS)


def current_step(at: Optional[float] = None) -> int:
    return int((clock() if at is None else at) // STEP)


def matching_step(secret: bytes, code: str, last_step: Optional[int]) -> Optional[int]:
    """The step ``code`` belongs to, if it is valid now and newer than
    ``last_step`` (the step this account last signed in with); else None."""
    code = ''.join(ch for ch in code if ch.isdigit())
    if len(code) != DIGITS:
        return None
    now = current_step()
    for step in range(now - DRIFT, now + DRIFT + 1):
        if last_step is not None and step <= last_step:
            continue
        if hmac.compare_digest(code_at(secret, step), code):
            return step
    return None


def otpauth_uri(secret: bytes, account: str) -> str:
    """The URI an authenticator app reads from the QR code."""
    label = quote(f'{ISSUER}:{account}')
    query = urlencode({'secret': base64.b32encode(secret).decode().rstrip('='),
                       'issuer': ISSUER, 'algorithm': 'SHA1', 'digits': DIGITS,
                       'period': STEP})
    return f'otpauth://totp/{label}?{query}'


def qr_data_url(text: str) -> str:
    """``text`` as a QR code, an SVG data URL the page can show as an image."""
    import segno
    buf = io.BytesIO()
    segno.make(text, error='m').save(buf, kind='svg', scale=6, border=2,
                                     dark='#000000', light='#ffffff', xmldecl=False)
    return 'data:image/svg+xml;base64,' + base64.b64encode(buf.getvalue()).decode()


# ── recovery codes ────────────────────────────────────────────────────────

def new_recovery_codes(n: int = RECOVERY_CODES) -> list[str]:
    """Codes like ``k7rm-2xqa``: 8 characters from 31, about 40 bits each."""
    def one() -> str:
        s = ''.join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(8))
        return f'{s[:4]}-{s[4:]}'
    return [one() for _ in range(n)]


def normalise_recovery_code(code: str) -> str:
    s = ''.join(ch for ch in code.lower() if ch.isalnum())
    return f'{s[:4]}-{s[4:]}' if len(s) == 8 else s


def looks_like_recovery_code(code: str) -> bool:
    return any(ch.isalpha() for ch in code)


# ── keys at rest ──────────────────────────────────────────────────────────

class SecretBox:
    """Encrypts the authenticator keys with a key file of its own.

    Encrypt-then-MAC with HMAC-SHA256 as the pseudo-random function: the key
    stream is HMAC(k, "enc" | nonce), the tag HMAC(k, "mac" | nonce | data).
    Keys are 20 bytes, so one HMAC block covers them. The key file is created
    on first use, readable by its owner only.
    """

    def __init__(self, key_file: Path):
        self.key_file = Path(key_file)
        self._key: Optional[bytes] = None

    def _load(self) -> bytes:
        if self._key is None:
            if not self.key_file.exists():
                self.key_file.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as f:
                    f.write(secrets.token_bytes(32))
            self._key = self.key_file.read_bytes()
            if len(self._key) != 32:
                raise RuntimeError(f'{self.key_file} is not a 32-byte key')
        return self._key

    def seal(self, secret: bytes) -> str:
        if len(secret) > 32:
            raise ValueError('secret too long')
        key, nonce = self._load(), secrets.token_bytes(16)
        stream = hmac.new(key, b'enc' + nonce, hashlib.sha256).digest()
        data = bytes(a ^ b for a, b in zip(secret, stream))
        tag = hmac.new(key, b'mac' + nonce + data, hashlib.sha256).digest()[:16]
        return f'v1${nonce.hex()}${data.hex()}${tag.hex()}'

    def open(self, sealed: str) -> bytes:
        version, nonce_hex, data_hex, tag_hex = sealed.split('$')
        if version != 'v1':
            raise ValueError('unknown key format')
        key, nonce, data = self._load(), bytes.fromhex(nonce_hex), bytes.fromhex(data_hex)
        tag = hmac.new(key, b'mac' + nonce + data, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, bytes.fromhex(tag_hex)):
            raise ValueError(f'authenticator key does not match {self.key_file}')
        stream = hmac.new(key, b'enc' + nonce, hashlib.sha256).digest()
        return bytes(a ^ b for a, b in zip(data, stream))
