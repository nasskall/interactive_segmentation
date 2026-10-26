#!/usr/bin/env bash
# Read-only survey of the lab server before installing the labelling app
# behind Apache, with training as Slurm jobs. It changes nothing, submits no
# Slurm job and never prompts: sudo is only ever used as `sudo -n`, which
# fails quietly instead of asking.
#
#   bash server-diagnostics.sh          # prints, and saves ~/iseg-diag.txt
#
# Before sending the output on, skim APP PROCESSES (command lines), GPU
# (process names) and SLURM (other users' job names) for anything you'd
# rather not share. Lines that look like keys, tokens or passwords are
# already filtered out.
{
sec(){ printf '\n===== %s =====\n' "$*"; }
nosecrets(){ grep -viE 'secret|token|password|passwd|api_?key|private'; }

sec "SYSTEM"
date -Is
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release; uname -r; hostname; hostname -f 2>/dev/null
lscpu | grep -E '^(Model name|CPU\(s\))'; free -h
df -h / /home /opt /srv /var /tmp 2>/dev/null
timedatectl 2>/dev/null | grep -Ei 'synchronized|NTP service|Time zone'

sec "ADDRESSES"
hostname -I 2>/dev/null
ip -4 -brief addr 2>/dev/null
ip route 2>/dev/null | grep -E '^default' | head -3
getent hosts "$(hostname)" 2>/dev/null

sec "ME / PERMISSIONS"
id
sudo -n true 2>/dev/null && echo "sudo: passwordless" || echo "sudo: needs password or not allowed"
sudo -n -l 2>/dev/null | grep -E 'NOPASSWD' | head -10

sec "GPU"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1
nvidia-smi 2>/dev/null | grep -m1 'CUDA Version'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
ls -l /dev/nvidia* 2>/dev/null | head -5
getent group video render 2>/dev/null

sec "PYTHON / NODE / GIT"
which -a python3; python3 --version
ls /usr/bin/python3.[0-9]* 2>/dev/null
python3 -c 'import venv, ensurepip' 2>&1 && echo "venv/ensurepip: OK"
python3 -c 'import tkinter' 2>&1 && echo "tkinter: OK (the app imports tkinter.messagebox)" \
  || echo "tkinter: missing (needs: apt install python3-tk)"
for t in uv conda pipx docker git node npm; do printf '%-6s ' "$t"; command -v "$t" || echo "-"; done
node --version 2>/dev/null; npm --version 2>/dev/null; git --version 2>/dev/null

sec "NETWORK (outbound, for git / pip / model downloads)"
env | grep -i '_proxy' | sed -E 's#//[^@/]*@#//***@#'
for u in https://github.com https://pypi.org https://download.pytorch.org/whl/ \
         https://huggingface.co https://dl.fbaipublicfiles.com; do
  printf '%-40s %s\n' "$u" "$(curl -s -o /dev/null -m 6 -w '%{http_code}' "$u")"
done

sec "LISTENING PORTS"
ss -tlnp 2>/dev/null
for p in 8000 8010 8020 8443 8444; do
  ss -tln | grep -qE ":$p\b" && echo "port $p: in use" || echo "port $p: free"
done
command -v ufw >/dev/null && (sudo -n ufw status 2>/dev/null || echo "ufw: status needs sudo")

sec "RUNNING SERVICES"
systemctl list-units --type=service --state=running --no-pager --plain --no-legend | awk '{print $1}'

sec "EXISTING APPS (how MAS is set up)"
ls -la /home/admin_ai/ai_apps 2>&1
systemctl cat mas-api --no-pager 2>&1 | grep -E '^(\[|User|Group|WorkingDirectory|ExecStart|Environment|EnvironmentFile|Restart)' | nosecrets
grep -hE '^(MAS_GPU_LAUNCHER|MAS_SLURM|MAS_LLM|CUDA|WHISPER_DEVICE|API_)' \
  /home/admin_ai/ai_apps/multilingual_audio_summary/.env 2>/dev/null | nosecrets
systemctl list-units --no-pager --plain --no-legend 'actions.runner*' 2>/dev/null | awk '{print $1, $3, $4}'

sec "APP PROCESSES"
ps -eo user,pid,etime,args | grep -Ei 'uvicorn|gunicorn|fastapi|ollama|slurm|python|node' | grep -v grep \
  | cut -c1-180 | nosecrets

sec "APACHE"
apache2ctl -v 2>&1
ls /etc/apache2/sites-enabled/ /etc/apache2/mods-enabled/ 2>&1 | tr '\n' ' '; echo
grep -vE '^\s*#|^\s*$' /etc/apache2/ports.conf 2>&1
apache2ctl -S 2>&1 | head -40
grep -RnE '^\s*(Listen|<VirtualHost|ServerName|ServerAlias|ProxyPass|ProxyPassReverse|ProxyPreserveHost|ProxyTimeout|Timeout|<Location|Alias|DocumentRoot|RewriteRule|RequestHeader|LimitRequestBody|Include|SSLCertificate)' \
  /etc/apache2/sites-enabled/ 2>&1 | nosecrets

sec "CERTIFICATES (names and expiry; public certificates only, no keys)"
for crt in $(grep -RhoE '^\s*SSLCertificateFile\s+\S+' /etc/apache2/sites-enabled/ 2>/dev/null \
             | awk '{print $2}' | sort -u); do
  echo "-- $crt"
  openssl x509 -in "$crt" -noout -subject -issuer -enddate -ext subjectAltName 2>&1
done

sec "SLURM"
sinfo --version 2>&1
sinfo -o '%P %a %l %D %c %m %G %N' 2>&1
scontrol show partition 2>&1 | grep -oE '(PartitionName|Default|MaxTime|DefaultTime|AllowGroups|AllowAccounts|State|TRES)=[^ ]+'
scontrol show node 2>&1 | grep -oE '(NodeName|CPUTot|RealMemory|Gres|CfgTRES|AllocTRES|State)=[^ ]+'
scontrol show config 2>&1 | grep -E '^(ClusterName|SlurmctldHost|SelectType|SelectTypeParameters|GresTypes|DefMemPerCPU|MaxTime|ProctrackType|SlurmUser)\b'
sacctmgr -n show assoc where user=$USER format=Account,Partition,QOS,GrpTRES,MaxWall 2>&1
squeue -o '%.10P %.20j %.10u %.2t %.10M %b %R' 2>&1 | head -15
sacct -n -X -S now-7days -o jobname%30,partition,alloctres%40,elapsed,state 2>/dev/null \
  | grep -E '^ *mas-' | tail -5

sec "WHERE APPS LIVE"
ls -la /opt /srv /var/www 2>&1
} 2>&1 | tee ~/iseg-diag.txt
