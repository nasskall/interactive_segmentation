"""
Set a labelling account's password from the server's command line, e.g. when
the only admin has forgotten theirs. Prompts for the new password without
echoing it, signs the account out everywhere and records the reset in the
audit log. Works while the server is running.

    python tools/reset_password.py <username> [--data-dir data]
"""

from __future__ import annotations

import argparse
import getpass
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.labelling import audit, auth  # noqa: E402
from server.labelling.db import Database  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('username')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = ap.parse_args()
    db_path = args.data_dir / 'labelling.db'
    if not db_path.exists():
        print(f'No database at {db_path}', file=sys.stderr)
        return 1
    db = Database(db_path)
    with closing(db.connect()) as conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (args.username,)).fetchone()
    if user is None:
        print(f'No account named {args.username!r}', file=sys.stderr)
        return 1
    password = getpass.getpass(f'New password for {user["username"]} '
                               f'({auth.MIN_PASSWORD_LENGTH}+ characters): ')
    if len(password) < auth.MIN_PASSWORD_LENGTH:
        print(f'Too short: at least {auth.MIN_PASSWORD_LENGTH} characters', file=sys.stderr)
        return 1
    if getpass.getpass('Again: ') != password:
        print('The two entries differ; nothing changed', file=sys.stderr)
        return 1
    with closing(db.connect()) as conn, conn:
        conn.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                     (auth.hash_password(password), user['id']))
        conn.execute('DELETE FROM tokens WHERE user_id = ?', (user['id'],))
        audit.record(conn, 'user.update', username='(command line)', target=f"user:{user['id']}",
                     detail={'password_changed': True, 'via': 'tools/reset_password.py'},
                     ip='localhost')
    print(f'Password changed for {user["username"]}; sign in again.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
