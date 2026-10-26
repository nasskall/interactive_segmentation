# Install on ailab26, behind Apache, with training as Slurm jobs

How this app runs on the lab server, next to the audio-summary app (`/mas/`)
and the others on the same HTTPS site. Values come from
`deploy/server-diagnostics.sh` run on 2026-10-26; rerun it if things move.

```
browser ──HTTPS──▶ Apache (ailab-ssl.conf, /iseg/) ──HTTP──▶ uvicorn 127.0.0.1:8020 ──▶ app (clicks, Auto-label on CPU)
                                                                     │
                                                                     └─ each training ──▶ srun --gres=gpu:1 ──▶ train_worker on a GPU
```

- **uvicorn** runs the app as one process on localhost only, under systemd
  (`iseg-api`). Clicks and Auto-label run there, on the CPU (48 cores).
- **Apache** is the only way in: `https://<server>/iseg/` on the existing site,
  like `/mas/`. The app finds its base path in the browser; no app setting.
- **Slurm** runs every model training as one job on one GPU, through
  `deploy/slurm_train.sh`; the Models tab shows the job id while it runs.

| Value | Used here | Change it if |
| --- | --- | --- |
| Server address | `192.168.25.110` (DHCP) | it changes: ask for a fixed lease |
| Address for users | `https://192.168.25.110/iseg/` | |
| uvicorn port (localhost) | `8020` | `ss -tln` shows it taken; change it in the service and the Apache lines |
| App folder | `/home/admin_ai/ai_apps/interactive_segmentation` | |
| Service user / name | `admin_ai` / `iseg-api` | |
| Slurm options | `--partition=main --time=4:00:00 --cpus-per-task=4 --mem=32G` | trainings need more than 4 h, or the policy says another partition |

`main` gives a job 1 hour unless it asks for more, hence `--time`. Python is
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

Check the GPU from a Slurm job, as trainings will use it:

```bash
srun --partition=short --gres=gpu:1 --time=0:05:00 \
     .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

It should print `NVIDIA RTX PRO 6000 ...` and `(12, 0)`. It may wait in the
queue first if the GPUs are busy.

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
rm ~/iseg-bundle.tar.gz
```

Every line must say `OK`, and the hashes must match the ones in
`docs/compliance/model-register.md`. The bundle is about 890 MB.

## Step 4: First run by hand

```bash
cd ~/ai_apps/interactive_segmentation
ISEG_TRAIN_LAUNCHER=$PWD/deploy/slurm_train.sh \
ISEG_SLURM_ARGS="--partition=main --time=4:00:00 --cpus-per-task=4 --mem=32G" \
.venv/bin/python -m server.main --cpu --host 127.0.0.1 --port 8020
```

It prints `Project-model training: .../deploy/slurm_train.sh` and a setup
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
the setup token, a username and a password of 12 or more characters. Then add
the other accounts on the **Users** page.

## Step 8: Check it end to end

1. Create a project with a mask class, upload a few images, and label with
   **Assist**: the first click on an image takes a few seconds, the next ones
   are quick.
2. `curl -s http://127.0.0.1:8020/api/auth/status` still answers;
   `journalctl -u iseg-api -n 20` shows the requests.
3. Once a project has 10 objects labelled with clicks, train an interactive
   model on the **Models** tab. While it runs:

   ```bash
   squeue -o '%.8i %.20j %.10u %.2t %.10M %b %R' | grep iseg-
   ```

   shows the job (`iseg-m<model>-<pid>`, run by `admin_ai`), and the Models
   tab shows the same Slurm job id. Cancel it once to check that the job
   leaves the queue.

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

### Where the logs are

| What | Where |
| --- | --- |
| The app (requests, startup, setup token) | `journalctl -u iseg-api -f` |
| Who did what, in the app | **Audit log** page (admins) |
| One training run | `data/models/logs/project_<p>_model_<m>.log` |
| The Slurm job of a run | `data/slurm/<job id>.out` / `.err` |
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

### A forgotten admin password

```bash
cd ~/ai_apps/interactive_segmentation
.venv/bin/python tools/reset_password.py <username>
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
| A training stays at **waiting for the scheduler** | the GPUs are taken | `squeue` shows the reason (`Resources`, `Priority`); it starts when a GPU frees |
| A training fails after exactly 1 hour | no `--time` in `ISEG_SLURM_ARGS` | keep `--time=4:00:00` (or more) in the service file, `daemon-reload`, restart |
| A training log shows `no kernel image is available` | PyTorch without Blackwell support | reinstall PyTorch from the `cu132` index (step 2) |
| `ModuleNotFoundError: sam2` | SAM 2 not installed into `.venv` | rerun the `requirements-server.txt` line of step 2 |
| `slurm_train.sh: Permission denied` | the script lost its execute bit | `chmod +x deploy/slurm_train.sh` |
| **413 Request Entity Too Large** on upload | Apache's body limit | keep the `LimitRequestBody` line from `iseg.conf` |
| The audit log shows `127.0.0.1` for everyone | requests reached uvicorn without Apache's headers | keep uvicorn on `127.0.0.1` only (as in the service file) |
| Users can't reach the site after a reboot | the server got a new DHCP address | ask for a fixed lease; update this file and the certificate |
