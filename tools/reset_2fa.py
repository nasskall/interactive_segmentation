"""
Clear an account's second factor from the server console, for someone who
has lost both their authenticator app and their recovery codes. They set up
the app again at their next sign-in. Needs shell access to the server, so it
does not weaken the web sign-in. Run from the app folder:

    .venv/bin/python tools/reset_2fa.py <username>
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.labelling import audit  # noqa: E402
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
    with closing(db.connect()) as conn, conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (args.username,)).fetchone()
        if user is None:
            print(f'No account named {args.username!r}', file=sys.stderr)
            return 1
        conn.execute('UPDATE users SET totp_secret = NULL, totp_last_step = NULL WHERE id = ?',
                     (user['id'],))
        for table in ('recovery_codes', 'tokens', 'login_challenges'):
            conn.execute(f'DELETE FROM {table} WHERE user_id = ?', (user['id'],))
        audit.record(conn, 'auth.2fa_reset', username='(command line)', target=f"user:{user['id']}",
                     detail={'via': 'tools/reset_2fa.py'}, ip='localhost')
    print(f'Second factor cleared for {user["username"]}; they set it up again at their next '
          'sign-in.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
