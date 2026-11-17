#!/usr/bin/env bash
# A minute-scale run of the distillation steps of run_isic_benchmarks.sh:
# the adapted SimpleClick + SAM 2 teacher labels 3 training images, a
# student trains for 20 steps (checkpoints picked on 3 val images), and the
# student is scored with clicks and with a box on 3 test images. Writes only
# to a temporary folder. Run from the repo root before submitting the chain:
#
#   srun --gres=gpu:1 --time=1:00:00 bash tools/distill_smoke.sh
set -euo pipefail
PY=${PY:-.venv/bin/python}
DATA=datasets/isic2018_protocol
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

echo "== teacher labels 3 images"
"$PY" tools/distill.py targets --pool $DATA --bench $DATA --out "$T/targets" --limit 3 | tail -n 3
echo "== student, 20 steps"
"$PY" tools/distill.py train --targets "$T/targets" --target accepted --seed 1 --bench $DATA \
  --select-split val --n-eval 3 --steps 20 --eval-every 10 --out "$T/student" | tail -n 3
echo "== student scored with clicks"
"$PY" tools/click_benchmark.py --model sam2 --adapter "$T/student/best.pt" --no-adapt \
  --data $DATA --eval-split test --n-eval 3 --per-image "$T/c.json" | tail -n 2
echo "== student scored with a box"
"$PY" tools/box_benchmark.py --model sam2 --adapter "$T/student/best.pt" \
  --data $DATA --eval-split test --n-eval 3 --per-image "$T/b.json" | tail -n 2
echo "== distillation smoke test passed"
