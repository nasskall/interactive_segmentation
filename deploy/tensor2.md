# Install on Tensor 2, with training on a GPU

The same app as on ailab26 ([ailab26.md](ailab26.md)), on the Tensor 2 server.
No Slurm: clicks and Auto-label run in the app's process on the CPU, and each
training runs as a child process of the app on one GPU.

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

Its SLURM section and the ailab26-specific lines (`admin_ai`, `mas-api`) will
just say "not found"; that is expected. From its output, set these once per
shell session (every command below uses them):

```bash
export ISEG_USER=<user that runs the app>                    # id, under ME / PERMISSIONS
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

- `<cuXXX>` is the PyTorch build for the server's CUDA: `cu132` on ailab26
  (CUDA 13.2). For another CUDA version, take the index URL from the selector
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

The service file in the repo is written for ailab26; this makes Tensor 2's
copy from it, with this server's user, folder, GPU and port:

```bash
cd "$ISEG_APP"
sed -e "s#/home/admin_ai/ai_apps/interactive_segmentation#$ISEG_APP#g" \
    -e "s#admin_ai#$ISEG_USER#g" \
    -e "s#ISEG_TRAIN_DEVICE=cuda:0#ISEG_TRAIN_DEVICE=$ISEG_GPU#" \
    -e "s#8020#$ISEG_PORT#g" \
    deploy/systemd/iseg-api.service > /tmp/iseg-api.service
grep -E '^(User|Group|WorkingDirectory|Environment=ISEG|ExecStart)' /tmp/iseg-api.service
```

Check the printed lines. `Group=` gets the user's name; if `id -gn
$ISEG_USER` prints another group, put that one there. Then:

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

Open the address from step 6, enter the setup token, a username and a
password of 12 or more characters, then set up two-step sign-in with an
authenticator app and save the recovery codes (as on ailab26, step 7 there).
Add the other accounts on the **Users** page.

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
3. For every screen and control: `deploy/e2e_check.sh` (set
   `ISEG_TRAIN_DEVICE=$ISEG_GPU` before it, and `ISEG_E2E_PORT` if 8021 is
   taken); setup as in ailab26.md, *Check everything end to end*.

## Running it

Updates, logs, backups, password and two-step resets, and removal work as on
ailab26 ([ailab26.md](ailab26.md), *Running it*), with `$ISEG_APP` as the app
folder. Two differences:

- After a `git pull` that changed `deploy/systemd/iseg-api.service`, redo
  step 5 rather than copying the file as it is.
- Training logs are only in `data/models/logs/` (there are no Slurm logs).

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
