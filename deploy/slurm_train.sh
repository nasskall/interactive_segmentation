#!/usr/bin/env bash
# Slurm launcher for project-model training: the server runs each training as
# one Slurm job on one GPU when started with
#
#   ISEG_TRAIN_LAUNCHER=deploy/slurm_train.sh
#
# srun stays in the foreground, in the run's own process group, so
# a cancel (SIGTERM to the group) or a stopped server cancels the Slurm job.
# The job's stdout/stderr go to $ISEG_SLURM_LOG_DIR/<job id>.out / .err and are
# replayed afterwards into the run's log in data/models/logs/.
#
#   ISEG_SLURM_ARGS   extra srun options, e.g. "--partition=gpu --time=4:00:00"
#   ISEG_SLURM_LOG_DIR  default: data/slurm
#   ISEG_JOB_NAME / ISEG_JOB_COMMENT  set by the server (model, project, user)
set -u
LOGDIR="${ISEG_SLURM_LOG_DIR:-$PWD/data/slurm}"
mkdir -p "$LOGDIR"
name="${ISEG_JOB_NAME:-iseg-train}-$$"
# ISEG_SLURM_ARGS is split into words on purpose.
# shellcheck disable=SC2086
PYTHONUNBUFFERED=1 srun --unbuffered --gres=gpu:1 ${ISEG_SLURM_ARGS:-} --job-name="$name" \
     --quiet --export=ALL --chdir="$PWD" --comment="${ISEG_JOB_COMMENT:-iseg training}" \
     --output="$LOGDIR/%j.out" --error="$LOGDIR/%j.err" "$@"
rc=$?
# Accounting lags a moment after the job ends; the unique name finds its id.
jid=""
for _ in 1 2 3 4 5; do
  jid=$(sacct -n -X --name="$name" -o jobid 2>/dev/null | tr -d ' ' | tail -1)
  [ -n "$jid" ] && break
  sleep 1
done
if [ -n "$jid" ]; then
  echo "[slurm] job $jid ($name): logs $LOGDIR/$jid.out / $jid.err, exit $rc"
  [ -s "$LOGDIR/$jid.out" ] && cat "$LOGDIR/$jid.out"
  [ -s "$LOGDIR/$jid.err" ] && cat "$LOGDIR/$jid.err" >&2
else
  echo "[slurm] job $name finished with exit $rc (id not found in sacct)"
fi
exit $rc
