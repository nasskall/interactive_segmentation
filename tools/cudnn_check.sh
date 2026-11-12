#!/usr/bin/env bash
# Do RITM and SimpleClick crash on this GPU only with cuDNN? Runs a short
# adapted click benchmark (20 session + 20 test images) per model, with
# cuDNN on and off, and prints how each run ended. Run from the repo root:
#
#   srun --gres=gpu:1 --time=1:00:00 bash tools/cudnn_check.sh
set -u
PY=${PY:-.venv/bin/python}
for model in ritm_coco simpleclick; do
  for cudnn in 1 0; do
    echo "=== $model, cuDNN $([ $cudnn = 1 ] && echo on || echo off)"
    ISEG_CUDNN=$cudnn PYTHONUNBUFFERED=1 "$PY" tools/click_benchmark.py --model $model \
      --data datasets/isic2018_protocol --eval-split test --n-eval 20 > /tmp/cudnn_check.$$ 2>&1
    code=$?
    grep -E "baseline|adapted|Error|error|fault|exception" /tmp/cudnn_check.$$ | tail -n 4
    echo "exit code $code  ($([ $code = 0 ] && echo OK || echo CRASHED))"
  done
done
rm -f /tmp/cudnn_check.$$
