# Install on Tensor 2, with training on a GPU

How the app runs on the Tensor 2 server. No Slurm: clicks and Auto-label run
in the app's process on the CPU, and each training runs as a child process of
the app on one GPU.

```
browser ──HTTPS──▶ Apache /iseg/ (or uvicorn's own HTTPS) ──▶ uvicorn ──▶ app (clicks, Auto-label on CPU)
                                                                  │
                                                                  └─ each training ──▶ train_worker (child process) on one GPU
```

## Step 0: Survey the server and fill in its values

On Tensor 2, from any folder:

```bash
curl -fsSLO https://raw.githubusercontent.com/nasskall/interactive_segmentation/main/deploy/server-diagnostics.sh
bash server-diagnostics.sh        # read-only; also saved to ~/iseg-diag.txt
```

Its SLURM section will just say "not found" on a server without Slurm; that
is expected. From its output, set these once per shell session (every
command below uses them):

```bash
export ISEG_USER=<user that runs the app>                    # id, under ME / PERMISSIONS
export ISEG_GROUP=$(id -gn $ISEG_USER)                        # that user's main group
export ISEG_APP=/home/$ISEG_USER/ai_apps/interactive_segmentation
export ISEG_GPU=cuda:0                                        # a GPU with little memory used, under GPU
export ISEG_PORT=8020                                         # "free" under LISTENING PORTS
export ISEG_ADDR=<server address>                             # under ADDRESSES
```

| Value | Read it from | Note |
| --- | --- | --- |
| `ISEG_GPU` | GPU: `memory.used` per index | `cuda:<index>`; another app on the same GPU can run it out of memory |
| PyTorch build | GPU: `CUDA Version` | step 2 |
| `ISEG_PORT` | LISTENING PORTS | any free port; uvicorn listens on it on localhost only (option A) |
| Apache or not | APACHE | step 6: option A if an HTTPS site already runs, otherwise B |
| Internet access | NETWORK | if `github.com` and `pypi.org` don't answer 200/30x, ask for a proxy first |

**Before you start:** the repository is public. Don't attach a self-hosted
GitHub runner on Tensor 2 to it: a pull request from a fork could run code on
the server.

## Step 1: Get the code

```bash
mkdir -p "$(dirname "$ISEG_APP")" && cd "$(dirname "$ISEG_APP")"
git clone https://github.com/nasskall/interactive_segmentation.git
cd "$ISEG_APP"
```

## Step 2: Python environment

```bash
cd "$ISEG_APP"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/<cuXXX>
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements-server.txt
.venv/bin/python -c "import torch, sam2, segment_anything, fastapi, segmentation_models_pytorch, onnx; print('ok', torch.__version__)"
```

- `<cuXXX>` is the PyTorch build for the server's CUDA: these versions were
  tested with `cu132` (CUDA 13.2, RTX PRO 6000 Blackwell GPUs, which need
  it). For another CUDA version, take the index URL from the selector
  on [pytorch.org](https://pytorch.org/get-started/locally/); the driver's
  `CUDA Version` (step 0) is the highest it supports.
- If `python3 -m venv` fails, the server needs `python3-venv` (ask the admin).
- `--no-build-isolation` builds SAM 2 against the PyTorch just installed;
  `SAM2_BUILD_CUDA=0` skips its optional CUDA extension, which needs `nvcc`.

Check the GPU that trainings will use:

```bash
nvidia-smi
.venv/bin/python -c "import torch; d = torch.device('$ISEG_GPU'); print(torch.cuda.get_device_name(d), torch.cuda.get_device_capability(d))"
```

It must print the GPU's name, not an error.

## Step 3: Model checkpoints and the built frontend

The checkpoints aren't all in git, and the frontend is built with Node.js. On
your PC, in PowerShell, in the repo folder:

```powershell
.\deploy\pack.ps1
scp iseg-bundle.tar.gz <user>@<server address>:~/
```

On Tensor 2:

```bash
cd "$ISEG_APP"
tar -xzf ~/iseg-bundle.tar.gz
(cd model_weights && sha256sum -c SHA256SUMS)
sed -n '/^## Checkpoints/,/^### /p' docs/compliance/model-register.md \
  | grep -oE '`[^`]+\.pth?`.*`[0-9a-f]{64}`' \
  | sed -E 's/^`([^`]+)`.*`([0-9a-f]{64})`.*/\2  \1/' \
  | (cd model_weights && sha256sum -c)
```

Every line of both checks must say `OK`. The first shows the copy arrived
intact; the second that the files are the registered models. On a `FAILED`
line, stop. Then `rm ~/iseg-bundle.tar.gz`.

## Step 4: First run by hand

```bash
cd "$ISEG_APP"
ISEG_TRAIN_DEVICE=$ISEG_GPU .venv/bin/python -m server.main --cpu --host 127.0.0.1 --port $ISEG_PORT
```

It prints `Project-model training: local worker process` and a setup token
(ignore it: it changes when the service starts). In a second terminal:

```bash
curl -s http://127.0.0.1:$ISEG_PORT/api/auth/status      # {"needs_setup":true}
```

Then stop it with Ctrl+C.

## Step 5: Run it as a service

`deploy/systemd/iseg-api.service` is a template; this fills in this server's
user, group, folder, GPU and port:

```bash
cd "$ISEG_APP"
sed -e "s#@ISEG_USER@#$ISEG_USER#g" -e "s#@ISEG_GROUP@#$ISEG_GROUP#g" \
    -e "s#@ISEG_APP@#$ISEG_APP#g" -e "s#@ISEG_GPU@#$ISEG_GPU#g" \
    -e "s#@ISEG_PORT@#$ISEG_PORT#g" \
    deploy/systemd/iseg-api.service > /tmp/iseg-api.service
grep -E '^(User|Group|WorkingDirectory|Environment=ISEG|ExecStart)' /tmp/iseg-api.service
grep -c '@ISEG_' /tmp/iseg-api.service       # must print 0
```

Check the printed lines (no `@ISEG_` left, no empty value). Then:

```bash
sudo cp /tmp/iseg-api.service /etc/systemd/system/iseg-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now iseg-api
systemctl status iseg-api --no-pager
curl -s http://127.0.0.1:$ISEG_PORT/api/auth/status
```

## Step 6: Publish it

### Option A: behind an existing Apache HTTPS site

Paste the lines of `deploy/apache/iseg.conf` (below its comment header) into
the site's `<VirtualHost ...:443>` block, with `8020` changed to `$ISEG_PORT`
if it differs. Back the site file up first:

```bash
apache2ctl -S                                         # which file holds the :443 site
sudo cp /etc/apache2/sites-available/<site>.conf /etc/apache2/sites-available/<site>.conf.bak-$(date +%F)
sudoedit /etc/apache2/sites-available/<site>.conf
sudo a2enmod proxy_http headers                       # no-op if already on
sudo apache2ctl configtest && sudo systemctl reload apache2
```

The address for users is `https://<server address>/iseg/`.

### Option B: no Apache, uvicorn serves HTTPS itself

Needs a certificate and key in PEM files readable by `$ISEG_USER` (from the
institution's IT, or self-signed for a trial). In
`/etc/systemd/system/iseg-api.service`, change the end of `ExecStart` from
`--host 127.0.0.1 --port <port>` to

```
--host 0.0.0.0 --port <port> --ssl-certfile <cert.pem> --ssl-keyfile <key.pem>
```

then `sudo systemctl daemon-reload && sudo systemctl restart iseg-api`, and
open the port in the firewall if one runs (`sudo ufw allow <port>/tcp`). The
address for users is `https://<server address>:<port>/`. Never serve it on
`0.0.0.0` without the certificate: passwords would cross the network in clear.

## Step 7: The first admin

```bash
journalctl -u iseg-api --no-pager | grep -A1 'setup token' | tail -1
```

Open the address from step 6 (with a self-signed certificate the browser
warns first; continue). Enter the setup token, a username and a password of
12 or more characters.

Every account then sets up two-step sign-in: scan the QR code with an
authenticator app (Google or Microsoft Authenticator, Aegis, 2FAS, ...), type
the 6-digit code it shows, and save the 10 recovery codes the page shows once.
From then on, signing in needs the password and a code from the app. Add the
other accounts on the **Users** page; each does the same at their first
sign-in.

## Step 8: Check it end to end

1. Create a project with a mask class, upload a few images, and label with
   **Assist**.
2. Once a project has 10 objects labelled with clicks, train an interactive
   model on the **Models** tab. While it runs,

   ```bash
   pgrep -af server.labelling.train_worker
   nvidia-smi
   ```

   show the worker (run by `$ISEG_USER`) and its memory on the training GPU.
   Cancel it once to check that the worker ends and frees the GPU.
3. For every screen and control: *Check everything end to end*, below.
## Running it

### Update to a new version

```bash
cd "$ISEG_APP"
git pull
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements-server.txt
sudo systemctl restart iseg-api
```

- If the frontend changed (`frontend/` in the pull), make a new bundle on the
  PC (`.\deploy\pack.ps1`) and unpack it again (step 3).
- If `deploy/systemd/iseg-api.service` changed, redo step 5.
- A restart cancels a training in progress; it shows as interrupted, so
  restart when the Models tab shows none running.

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
ISEG_TRAIN_DEVICE=$ISEG_GPU deploy/e2e_check.sh    # or --no-train to skip the GPU trainings
```

Set `ISEG_E2E_PORT` if 8021 is taken. It prints PASS / FAIL / SKIP for each
check and where the report is; each failure has a screenshot next to the
report.

### Where the logs are

| What | Where |
| --- | --- |
| The app (requests, startup, setup token) | `journalctl -u iseg-api -f` |
| Who did what, in the app | **Audit log** page (admins) |
| One training run | `data/models/logs/project_<p>_model_<m>.log` |
| Apache (option A) | `/var/log/apache2/` |

### Back up

Everything that can't be recreated is in `data/`: the database, the uploaded
images and the trained models.

```bash
cd "$ISEG_APP"
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
cd "$ISEG_APP"
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
cd "$ISEG_APP"
.venv/bin/python tools/reset_2fa.py <username>
```

### Remove it

```bash
sudo systemctl disable --now iseg-api
sudo rm /etc/systemd/system/iseg-api.service
sudo systemctl daemon-reload
```

With option A, also delete the `/iseg/` lines from the Apache site,
`configtest` and `reload`. The app folder, with `data/`, stays until you
delete it.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| A training fails with `CUDA out of memory` | another app uses the training GPU | `nvidia-smi`; set `ISEG_TRAIN_DEVICE` to a free GPU in the service file, `daemon-reload`, restart |
| A training log says it runs on `cpu` | PyTorch doesn't see the GPU | rerun step 2's GPU check as `$ISEG_USER`; reinstall PyTorch for the right `<cuXXX>` |
| A training log shows `no kernel image is available` | PyTorch build too old for the GPU | reinstall PyTorch from a newer `cuXXX` index (step 2) |
| `status=217/USER` in `systemctl status` | wrong `User=` in the service | redo step 5 with the right `ISEG_USER` |
| `/iseg/` gives **502 / 503** (option A) | the service isn't running, or the port differs from Apache's lines | `systemctl status iseg-api`; compare the port in both files |
| **413 Request Entity Too Large** on upload (option A) | Apache's body limit | keep the `LimitRequestBody` line from `iseg.conf` |
| The audit log shows `127.0.0.1` for everyone (option A) | requests reached uvicorn without Apache's headers | keep uvicorn on `127.0.0.1` |
