#!/usr/bin/env bash
# End-to-end check through the real UI (tools/e2e_check.py) against a separate
# server instance: same code and model_weights/, but its own port and an empty
# data folder in /tmp, so live users, projects and the audit log are not
# touched. Trainings go to Slurm like ailab26's service when srun exists;
# otherwise they run as local worker processes on the GPU, like the
# no-Slurm service's (ISEG_TRAIN_DEVICE, default cuda:0).
#
#   deploy/e2e_check.sh [--no-train] [--steps 20] [--train-timeout 1800]
#
# Needs a Python with Playwright and its Chromium, outside the app's .venv:
#   python3 -m venv ~/iseg-e2e
#   ~/iseg-e2e/bin/pip install playwright
#   ~/iseg-e2e/bin/python -m playwright install chromium
#
#   ISEG_E2E_PYTHON  that Python      (default ~/iseg-e2e/bin/python)
#   ISEG_E2E_PORT    test port        (default 8021)
#   ISEG_E2E_KEEP=1  keep the data folder afterwards
set -u
APP="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ISEG_E2E_PYTHON:-$HOME/iseg-e2e/bin/python}"
PORT="${ISEG_E2E_PORT:-8021}"
WORK="$(mktemp -d /tmp/iseg-e2e.XXXXXX)"
mkdir -p "$WORK/data" "$WORK/slurm" "$WORK/report"

[ -x "$PY" ] || { echo "No Python at $PY; see the comment at the top of this file."; exit 2; }
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Port $PORT is in use; set ISEG_E2E_PORT to a free one."; exit 2
fi

cd "$APP"
if command -v srun >/dev/null; then
  export ISEG_TRAIN_LAUNCHER="$APP/deploy/slurm_train.sh"
  export ISEG_SLURM_ARGS="${ISEG_SLURM_ARGS:---partition=main --time=1:00:00 --cpus-per-task=4 --mem=32G}"
  export ISEG_SLURM_LOG_DIR="$WORK/slurm"
else
  unset ISEG_TRAIN_LAUNCHER
  export ISEG_TRAIN_DEVICE="${ISEG_TRAIN_DEVICE:-cuda:0}"
fi
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  .venv/bin/python -m server.main --cpu --host 127.0.0.1 --port "$PORT" \
  --data-dir "$WORK/data" > "$WORK/server.log" 2>&1 &
SERVER=$!
cleanup() {
  kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
  if [ "${ISEG_E2E_KEEP:-0}" = 1 ]; then echo "Data kept in $WORK/data"; else rm -rf "$WORK/data"; fi
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  grep -q 'Uvicorn running' "$WORK/server.log" && break
  kill -0 "$SERVER" 2>/dev/null || { cat "$WORK/server.log"; exit 1; }
  sleep 1
done
TOKEN="$(grep -A1 'setup token' "$WORK/server.log" | tail -1 | tr -d ' ')"
echo "Test instance on 127.0.0.1:$PORT, data in $WORK/data, log $WORK/server.log"

"$PY" tools/e2e_check.py --url "http://127.0.0.1:$PORT" --setup-token "$TOKEN" \
  --out "$WORK/report" --data-dir "$WORK/data" "$@"
rc=$?
cp "$WORK/server.log" "$WORK/report/"
echo "Report: $WORK/report/report.txt (screenshots of failures next to it)"
exit $rc
