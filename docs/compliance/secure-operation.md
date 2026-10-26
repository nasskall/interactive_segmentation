# Secure operation

How to deploy and run the platform on the lab network so the controls in
this pack hold. Adopt it as the documented operating procedure
(ISO/IEC 27001 5.37).

## 1. Install

- Install Python packages from `requirements.txt`; SAM and SAM 2 from
  Meta's GitHub at the pinned commits (README). Do not use the PyPI `sam2`
  package.
- Put only checkpoints listed in [model-register.md](model-register.md) in
  `model_weights/`, and check each one:
  `sha256sum model_weights/*` must match the register.
- Build the frontend: `npm ci && npm run build` in `frontend/`.
- Keep `data/` on an **encrypted volume** (BitLocker on Windows, LUKS on
  Linux). It holds the database, the audit log and every uploaded image.

## 2. Run

```bash
python -m server.main --host <lab interface IP> --port 8443 \
    --ssl-certfile /path/cert.pem --ssl-keyfile /path/key.pem
```

- **HTTPS is required** whenever `--host` is not `127.0.0.1`. Use a
  certificate from the lab's internal CA so browsers trust it. Without one,
  the server warns at startup because passwords and tokens would cross the
  network unencrypted.
- Firewall the port so only the labelling subnet (or VPN) can reach it.
- Leave `ISEG_CORS_ORIGINS` unset unless the frontend is served from another
  origin.
- Run the server under a dedicated, unprivileged service account that owns
  `data/` and nothing else.
- Keep uvicorn's stdout (the access log) with the system logs.

On first start the server prints a one-time setup token; use it to create
the first admin (the token changes on every restart until an admin exists).

## 3. Accounts

- One account per person; never share accounts.
- Passwords: at least 12 characters. Hand initial passwords over through a
  different channel from the username; ask users to change theirs.
- **Leavers:** deactivate the account the same day (Users page). This ends
  all of that user's sessions at once and keeps their labels attributable.
- **Quarterly review:** check the Users page and the audit log (`Accounts`
  filter) for unexpected admins or accounts.
- Keep at least two admins so one can recover the other's access (Users
  page, *Reset password*). If no admin can sign in, someone with access to
  the server runs `python tools/reset_password.py <username>`: it prompts
  for the new password, signs the account out everywhere and records the
  reset in the audit log (`user.update` by "(command line)").

## 4. Monitoring

Weekly, an admin opens **Audit log**:

- **Check integrity** must report "Intact".
- Review `Sign-ins` for bursts of `auth.login_failed` or any
  `auth.login_blocked` (possible password guessing).
- Review `project.export` (who took data out) and `user.update` role
  changes.
- Review `model.load` (production model switches; the SHA-256 must be in
  the register).

**Off-host copy of the audit log** (monthly, and before any upgrade):

```bash
sqlite3 data/labelling.db ".mode csv" ".headers on" "SELECT * FROM audit_log" > audit_YYYY-MM.csv
```

Store the copies on another system. Each row's `hash` commits to all
earlier rows, so the last hash in a copy fingerprints the log up to that
point.

## 5. Backups

- Back up `data/` (database and images) daily and `model_weights/` after
  each change, to encrypted media or an encrypted backup target.
- Back up the database consistently while the server runs with
  `sqlite3 data/labelling.db ".backup backup.db"` (the database uses WAL
  mode; copying the file alone can miss recent writes).
- Test a restore every quarter on a separate machine, using `--data-dir` to
  point a throwaway server at the restored copy, and run **Check integrity**
  on it.

## 6. Changes

- **Code:** through pull requests; `python -m pytest` must pass, including
  `tests/test_security.py`.
- **Dependencies:** run `pip-audit` and `npm audit` monthly and before each
  release; patch or document accepted risks.
- **Models:** benchmark on the held-out sets first (`tools/click_benchmark.py`,
  lesion set from `tools/build_lesion_benchmark.py`), add the row to the
  model register, then switch in the app (logged as `model.load`).

## 7. Incidents

If an account may be compromised: deactivate it, copy `data/` and the audit
log aside, run **Check integrity**, and review that account's events and
exports in the audit log. If a model is found to produce systematically
wrong masks: find affected labels by the model's SHA-256 in an export and
notify the dataset owners.

## 8. Decommissioning

Export what must be kept, then wipe the encrypted volume and every backup
per the retention policy.
