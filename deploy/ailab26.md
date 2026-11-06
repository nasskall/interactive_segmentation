# Install on ailab26, behind Apache, with training on a GPU

How this app runs on the lab server, next to the audio-summary app (`/mas/`)
and the others on the same HTTPS site. Values come from
`deploy/server-diagnostics.sh` run on 2026-10-26; rerun it if things move.

```
browser ──HTTPS──▶ Apache (ailab-ssl.conf, /iseg/) ──HTTP──▶ uvicorn 127.0.0.1:8020 ──▶ app (clicks, Auto-label on CPU)
                                                                     │
                                                                     └─ each training ──▶ train_worker (child process) on one GPU
```

- **uvicorn** runs the app as one process on localhost only, under systemd
  (`iseg-api`). Clicks and Auto-label run there, on the CPU (48 cores).
- **Apache** is the only way in: `https://<server>/iseg/` on the existing site,
  like `/mas/`. The app finds its base path in the browser; no app setting.
- **Training** runs one model at a time as a child process of the service
  (`train_worker`), on the GPU set by `ISEG_TRAIN_DEVICE` in the service
  file. No Slurm: a run starts at once and has no time limit.

| Value | Used here | Change it if |
| --- | --- | --- |
| Server address | `192.168.25.110` (DHCP) | it changes: ask for a fixed lease |
| Address for users | `https://192.168.25.110/iseg/` | |
| uvicorn port (localhost) | `8020` | `ss -tln` shows it taken; change it in the service and the Apache lines |
| App folder | `/home/admin_ai/ai_apps/interactive_segmentation` | |
| Service user / name | `admin_ai` / `iseg-api` | |
| Training GPU | `ISEG_TRAIN_DEVICE=cuda:0` | `nvidia-smi` shows another app using that GPU |

Python is
3.14; the GPUs are RTX PRO 6000 Blackwell with CUDA 13.2, which need the
`cu132` build of PyTorch (step 2).

**Before you start**
- Steps 5 and 6 need `sudo` with your password (only restarting `mas-api`
  works without one).
- This repository is public. Don't attach the self-hosted GitHub runner
  `ailab26` to it: a pull request from a fork could run code on the server.
  Updates are by hand (*Running it* below).

## Step 1: Get the code

```bash
cd ~/ai_apps
git clone https://github.com/nasskall/interactive_segmentation.git
cd interactive_segmentation
```

## Step 2: Python environment

```bash
cd ~/ai_apps/interactive_segmentation
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu132
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements-server.txt
.venv/bin/python -c "import torch, sam2, segment_anything, fastapi, segmentation_models_pytorch, onnx; print('ok', torch.__version__)"
```

- `--no-build-isolation` builds SAM 2 against the PyTorch just installed,
  instead of downloading another copy to build with.
- `SAM2_BUILD_CUDA=0` skips SAM 2's optional CUDA extension (there is no
  `nvcc` here); SAM 2 works without it.
- The server doesn't need Tk (`python3-tk`): the app only imports it in the
  desktop app.

Check the GPU that trainings will use:

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

It should print `NVIDIA RTX PRO 6000 ...` and `(12, 0)`. In `nvidia-smi`,
pick a GPU with little memory in use; if it is not GPU 0, change
`ISEG_TRAIN_DEVICE` in `deploy/systemd/iseg-api.service` (step 5) to
`cuda:<n>`.

## Step 3: Model checkpoints and the built frontend

The server has no Node.js, and Meta's download site is blocked there, so both
come from your PC. On the PC, in PowerShell, in the repo folder:

```powershell
.\deploy\pack.ps1
scp iseg-bundle.tar.gz admin_ai@192.168.25.110:~/
```

On the server:

```bash
cd ~/ai_apps/interactive_segmentation
tar -xzf ~/iseg-bundle.tar.gz
(cd model_weights && sha256sum -c SHA256SUMS)
```

Every line must say `OK`. That only shows the copy arrived intact:
`pack.ps1` wrote `SHA256SUMS` from the files on the PC. So check them
against the hashes in `docs/compliance/model-register.md` as well:

```bash
grep -oE '`[^`]+\.pth?`.*`[0-9a-f]{64}`' docs/compliance/model-register.md \
  | sed -E 's/^`([^`]+)`.*`([0-9a-f]{64})`.*/\2  \1/' \
  | (cd model_weights && sha256sum -c)
```

Again every line must say `OK`; a `FAILED` file is not the registered
model, so stop there. Only then remove the bundle (about 890 MB):

```bash
rm ~/iseg-bundle.tar.gz
```

## Step 4: First run by hand

```bash
cd ~/ai_apps/interactive_segmentation
ISEG_TRAIN_DEVICE=cuda:0 \
.venv/bin/python -m server.main --cpu --host 127.0.0.1 --port 8020
```

It prints `Project-model training: local worker process` and a setup
token (ignore it: it changes when the service starts). In a second terminal:

```bash
curl -s http://127.0.0.1:8020/api/auth/status      # {"needs_setup":true}
```

Then stop it with Ctrl+C.

## Step 5: Run it as a service

```bash
cd ~/ai_apps/interactive_segmentation
sudo cp deploy/systemd/iseg-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now iseg-api
systemctl status iseg-api --no-pager
curl -s http://127.0.0.1:8020/api/auth/status
```

Optional, as for `mas-api`: let `admin_ai` restart it without a password.

```bash
echo 'admin_ai ALL=(root) NOPASSWD: /usr/bin/systemctl restart iseg-api, /usr/bin/systemctl status iseg-api' \
  | sudo tee /etc/sudoers.d/iseg-api
sudo chmod 440 /etc/sudoers.d/iseg-api && sudo visudo -c
```

## Step 6: Publish it through Apache

The lines are in `deploy/apache/iseg.conf`. They go into the existing HTTPS
site, next to the `/mas/` lines.

```bash
sudo cp /etc/apache2/sites-available/ailab-ssl.conf /etc/apache2/sites-available/ailab-ssl.conf.bak-$(date +%F)
grep -n '/mas/' /etc/apache2/sites-available/ailab-ssl.conf
sudoedit /etc/apache2/sites-available/ailab-ssl.conf
```

Paste the lines of `deploy/apache/iseg.conf` (everything below its comment
header) right after the two `/mas/` lines, inside
`<VirtualHost 192.168.25.110:443>`. Then:

```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

`reload` keeps the other apps' connections. `mod_proxy_http` and
`mod_headers` are already on.

**The site's own login.** `conf-enabled/ailab-auth.conf` sets up a sign-in
for the site. See whether it covers every path:

```bash
sudo grep -nE 'Location|Require|AuthType|AuthFormProvider' /etc/apache2/conf-enabled/ailab-auth.conf
```

If it does, people sign in to the site first and then to this app (which has
its own accounts, roles and audit log). If `/mas/` is exempted, exempt
`/iseg/` the same way.

## Step 7: The first admin

```bash
journalctl -u iseg-api --no-pager | grep -A1 'setup token' | tail -1
```

Open `https://192.168.25.110/iseg/`. The browser warns about the certificate
(self-signed, and its names don't include `192.168.25.110`); continue. Enter
the setup token, a username and a password of 12 or more characters.

Every account then sets up two-step sign-in: scan the QR code with an
authenticator app (Google or Microsoft Authenticator, Aegis, 2FAS, ...), type
the 6-digit code it shows, and save the 10 recovery codes the page shows once.
From then on, signing in needs the password and a code from the app. Add the
other accounts on the **Users** page; each does the same at their first
sign-in.

## Step 8: Check it end to end

1. Create a project with a mask class, upload a few images, and label with
   **Assist**: the first click on an image takes a few seconds, the next ones
   are quick.
2. `curl -s http://127.0.0.1:8020/api/auth/status` still answers;
   `journalctl -u iseg-api -n 20` shows the requests.
3. Once a project has 10 objects labelled with clicks, train an interactive
   model on the **Models** tab. While it runs:

   ```bash
   pgrep -af server.labelling.train_worker
   nvidia-smi
   ```

   show the worker (run by `admin_ai`) and its memory on the training GPU.
   Cancel it once to check that the worker ends and frees the GPU.

## Running it

### Update to a new version

```bash
cd ~/ai_apps/interactive_segmentation
git pull
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements-server.txt
sudo systemctl restart iseg-api
```

If the frontend changed (`frontend/` in the pull), make a new bundle on the PC
(`.\deploy\pack.ps1`) and unpack it again (step 3). A restart cancels a
training in progress; it shows as interrupted, so restart when the Models tab
shows none running.

### Check everything end to end

`deploy/e2e_check.sh` goes through every screen and control in a real
(headless) browser: accounts, projects of all five task types, classes,
uploads, assignment, every labelling tool, quick segment with every model
and ensemble, training of each kind on the GPU (and a cancel), auto-labelling,
exports, settings and the audit log. It starts its own server instance on
port 8021 with an empty data folder in `/tmp`, so live users and data are not
touched. It needs Playwright, once, in its own environment:

```bash
python3 -m venv ~/iseg-e2e
~/iseg-e2e/bin/pip install playwright
~/iseg-e2e/bin/python -m playwright install chromium
```

Then, from the app folder (about 30 minutes, most of it training):

```bash
deploy/e2e_check.sh                 # or --no-train to skip the GPU trainings
```

It prints PASS / FAIL / SKIP for each check and where the report is; each
failure has a screenshot next to the report.

### Where the logs are

| What | Where |
| --- | --- |
| The app (requests, startup, setup token) | `journalctl -u iseg-api -f` |
| Who did what, in the app | **Audit log** page (admins) |
| One training run | `data/models/logs/project_<p>_model_<m>.log` |
| Apache | `/var/log/apache2/` (the site's error and access logs) |

### Back up

Everything that can't be recreated is in `data/`: the database, the uploaded
images and the trained models.

```bash
cd ~/ai_apps/interactive_segmentation
mkdir -p ~/backups
.venv/bin/python -c "import sqlite3; s = sqlite3.connect('data/labelling.db'); d = sqlite3.connect('$HOME/backups/iseg-$(date +%F).db'); s.backup(d); d.close()"
tar -czf ~/backups/iseg-files-$(date +%F).tar.gz data/items data/models
```

The database copy is consistent while the app runs. The images may be health
data: keep backups as private as the server.

`data/totp.key` encrypts every account's authenticator key in the database;
the commands above leave it out on purpose, so a backup alone cannot be used
to make sign-in codes. Keep one copy of it apart from the backups (for
example on the admin's own encrypted storage). Without it, a restored
database still works, but every account must set up two-step sign-in again
(`tools/reset_2fa.py` below, once per account).

### A forgotten admin password

```bash
cd ~/ai_apps/interactive_segmentation
.venv/bin/python tools/reset_password.py <username>
```

### A lost phone

Someone who lost their phone signs in with one of their recovery codes and
then sets the app up again from a new phone after a reset (below), or makes
new recovery codes on their **Account** page (click their name, top right)
if they still have the app. Someone who has lost both the app and the codes
needs a reset from the server console; they set up the app again at their
next sign-in:

```bash
cd ~/ai_apps/interactive_segmentation
.venv/bin/python tools/reset_2fa.py <username>
```

### Remove it

```bash
sudo systemctl disable --now iseg-api
sudo rm /etc/systemd/system/iseg-api.service /etc/sudoers.d/iseg-api
sudo systemctl daemon-reload
```

Then delete the `/iseg/` lines from `ailab-ssl.conf`, `configtest` and
`reload`. The app folder, with `data/`, stays until you delete it.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `/iseg/` gives **502 / 503 Proxy Error** | the service isn't running | `systemctl status iseg-api`, `journalctl -u iseg-api -n 50` |
| The page loads but stays empty, or API calls go to `/api/...` and 404 | an old frontend build without base-path support | unpack a fresh bundle (step 3) |
| A training fails with `CUDA out of memory` | another app uses the training GPU | `nvidia-smi`; set `ISEG_TRAIN_DEVICE` to a free GPU in `/etc/systemd/system/iseg-api.service`, `daemon-reload`, restart |
| A training log says it runs on `cpu` | the service doesn't see a GPU | rerun step 2's GPU check as `admin_ai` |
| A training log shows `no kernel image is available` | PyTorch without Blackwell support | reinstall PyTorch from the `cu132` index (step 2) |
| `ModuleNotFoundError: sam2` | SAM 2 not installed into `.venv` | rerun the `requirements-server.txt` line of step 2 |
| **413 Request Entity Too Large** on upload | Apache's body limit | keep the `LimitRequestBody` line from `iseg.conf` |
| The audit log shows `127.0.0.1` for everyone | requests reached uvicorn without Apache's headers | keep uvicorn on `127.0.0.1` only (as in the service file) |
| Users can't reach the site after a reboot | the server got a new DHCP address | ask for a fixed lease; update this file and the certificate |
