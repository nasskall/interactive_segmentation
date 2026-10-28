#!/usr/bin/env bash
# All benchmark runs for the ISIC experiments, submitted as a chain of
# Slurm jobs (sbatch, one GPU each). Run from the repository root after
# `bash tools/setup_benchmarks.sh --data --smoke` has set up the code,
# checkpoints and datasets:
#
#   bash tools/run_isic_benchmarks.sh            # submit everything
#   DRY=1 bash tools/run_isic_benchmarks.sh      # print the sbatch commands only
#
# Every model is scored on two test sets:
#   isic   ISIC 2018 Task 1 test (1000); IMIS-Net and likely MedSAM trained on it
#   clean  1000 IMA++ images outside every ISIC 2016-2018 challenge set
# Outputs: experiments/isic/ (targets, students), results/<set>/*.json
# (per-image results), results/<set>/summary*.json (bootstrap CIs).
set -euo pipefail
PY=.venv/bin/python
DATA=datasets/isic2018_protocol
EXP=experiments/isic
LOGS=data/slurm/paper
SEEDS="1 2 3"
SLURM="--partition=main --cpus-per-task=4 --mem=32G"
TESTSETS="isic:$DATA clean:datasets/clean_isic_test"
IMIS=${IMIS_ROOT:-../external/IMIS-Bench}
IMIS_PY=${IMIS_PY:-../external/imis-venv/bin/python}
MEDSAM=${MEDSAM_CKPT:-../external/MedSAM/work_dir/MedSAM/medsam_vit_b.pth}
mkdir -p "$EXP" "$LOGS" results/isic results/clean
[ -n "${DRY:-}" ] || [ -f datasets/clean_isic_test/manifest.json ] || { echo "build datasets/clean_isic_test first"; exit 1; }

submit() {   # submit <name> <time> <dependency or ""> <command...>
  local name=$1 time=$2 dep=$3; shift 3
  local args=(--parsable --job-name="isic-$name" --time="$time" --gres=gpu:1
              --output="$LOGS/%j-$name.out" --chdir="$PWD")
  [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
  # shellcheck disable=SC2086
  if [ -n "${DRY:-}" ]; then echo "sbatch ${args[*]} $SLURM --wrap='$*'" >&2; echo "dry$RANDOM"
  else sbatch "${args[@]}" $SLURM --wrap="PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 $*"; fi
}

# All evaluations of one model on every test set, chained with && .
#   evals <click args or ""> <box: sam|sam2|medsam|imis|""> <file stem> [adapter]
evals() {
  local click=$1 box=$2 stem=$3 adapter=${4:-} cmds=() set data res
  for t in $TESTSETS; do
    set=${t%%:*}; data=${t#*:}; res=results/$set
    if [ -n "$click" ]; then
      cmds+=("$click --data $data --eval-split test --per-image $res/$stem.json")
    fi
    for jit in 0 0.1; do
      case $box in
        sam|sam2) cmds+=("$PY tools/box_benchmark.py --model $box ${adapter:+--adapter $adapter} \
                    --data $data --eval-split test --jitter $jit --per-image $res/box${jit}_$stem.json") ;;
        medsam)   cmds+=("$PY tools/box_benchmark.py --model medsam --checkpoint $MEDSAM \
                    --data $data --eval-split test --jitter $jit --per-image $res/box${jit}_$stem.json") ;;
        imis)     cmds+=("$IMIS_PY tools/imis_benchmark.py --imis-root $IMIS \
                    --checkpoint $IMIS/ckpt/IMISNet-B.pth --box --jitter $jit \
                    --data $data --eval-split test --per-image $res/box${jit}_$stem.json") ;;
      esac
    done
  done
  local IFS='&'; echo "${cmds[*]}" | sed 's/&/ \&\& /g'
}

# 1. Freeze the ISIC 2018 splits (CPU work, but cheap enough to share the GPU queue).
[ -f "$DATA/manifest.json" ] && build="" || \
  build=$(submit build 1:00:00 "" "$PY tools/build_isic_protocol.py --raw datasets/isic2018/raw --out $DATA")

# 2. Teacher (SimpleClick + SAM 2, adapted on the session split) labels train.
targets=$(submit targets 24:00:00 "$build" \
  "$PY tools/distill.py targets --pool $DATA --bench $DATA --out $EXP/sc_sam2")

# 3. Students: three targets x three seeds, best checkpoint picked on val,
#    then scored once on each test set (clicks and boxes).
deps=()
declare -A student_job
for target in teacher accepted gt; do
  for s in $SEEDS; do
    run=$EXP/${target}_s$s
    student_job[${target}_s$s]=$(submit "$target-s$s" 16:00:00 "$targets" \
      "$PY tools/distill.py train --targets $EXP/sc_sam2 --target $target --seed $s \
        --bench $DATA --select-split val --out $run && \
       $(evals "$PY tools/click_benchmark.py --model sam2 --adapter $run/best.pt --no-adapt" \
               sam2 "${target}_s$s" "$run/best.pt")")
    deps+=("${student_job[${target}_s$s]}")
  done
done

# 4. Baselines: every backend as shipped, and after online adaptation on the
#    ISIC 2018 session split (3 presses, as in the app).
for m in sam2 sam simpleclick ritm ritm_coco; do
  box=""; case $m in sam|sam2) box=$m ;; esac
  deps+=("$(submit "base-$m" 12:00:00 "$build" \
    "$(evals "$PY tools/click_benchmark.py --model $m --no-adapt" "$box" "base_$m")")")
  deps+=("$(submit "adapted-$m" 12:00:00 "$build" \
    "$(evals "$PY tools/click_benchmark.py --model $m --session-data $DATA" "" "adapted_$m")")")
done

# 4b. IMIS-Net, unmodified, from its own clone and venv (transformers<5, monai,
#     timm); checkpoint from huggingface.co/1Junlong/IMIS-Net @ c3d2edd.
deps+=("$(submit base-imis 12:00:00 "$build" \
  "$(evals "$IMIS_PY tools/imis_benchmark.py --imis-root $IMIS --checkpoint $IMIS/ckpt/IMISNet-B.pth" \
           imis "base_imis")")")

# 4c. MedSAM: box prompts only (its public checkpoint is box-trained).
deps+=("$(submit base-medsam 8:00:00 "$build" "$(evals "" medsam "base_medsam")")")

# 4d. Realistic clicks (RClicks, tools/rclicks_clicker.py): the whole
#     clickability distribution (0,1) and its least likely tenth (0,0.1), for
#     the models the headline comparison needs. Results: rc<band>_<stem>.json.
RC="--clicker rclicks --rclicks-python ${RCLICKS_PY:-../external/rclicks-venv/bin/python}"
rc_evals() {   # rc_evals <click command> <file stem>
  local cmds=() t band
  for t in $TESTSETS; do
    for band in 0,1 0,0.1; do
      cmds+=("$1 $RC --rclicks-band $band --data ${t#*:} --eval-split test \
              --per-image results/${t%%:*}/rc${band/,/-}_$2.json")
    done
  done
  local IFS='&'; echo "${cmds[*]}" | sed 's/&/ \&\& /g'
}
for m in sam2 simpleclick; do
  deps+=("$(submit "rc-$m" 16:00:00 "$build" \
    "$(rc_evals "$PY tools/click_benchmark.py --model $m --no-adapt" "base_$m")")")
done
deps+=("$(submit rc-imis 16:00:00 "$build" \
  "$(rc_evals "$IMIS_PY tools/imis_benchmark.py --imis-root $IMIS --checkpoint $IMIS/ckpt/IMISNet-B.pth" base_imis)")")
for target in accepted gt; do
  for s in $SEEDS; do
    deps+=("$(submit "rc-$target-s$s" 16:00:00 "${student_job[${target}_s$s]}" \
      "$(rc_evals "$PY tools/click_benchmark.py --model sam2 --adapter $EXP/${target}_s$s/best.pt --no-adapt" \
                  "${target}_s$s")")")
  done
done

# 5. Summaries with bootstrap CIs and the paired comparisons the paper needs:
#    one stats job per test set, covering the whole set and, for the clean
#    set, its subgroups (tools/clean_subgroups.py: one acquisition source
#    dominates it, and most of its lesions are small).
dep=$(IFS=:; echo "${deps[*]}")
summaries() {   # summaries <results dir> <suffix> <extra paper_stats args>
  local res=$1 sfx=$2 opt=$3 cmds=() jit band
  cmds+=("$PY tools/paper_stats.py base=$res/base_sam2.json adapted=$res/adapted_sam2.json \
    'teacher=$res/teacher_s*.json' 'accepted=$res/accepted_s*.json' 'gt=$res/gt_s*.json' \
    simpleclick=$res/base_simpleclick.json sam=$res/base_sam.json imis=$res/base_imis.json \
    ritm=$res/base_ritm.json ritm_coco=$res/base_ritm_coco.json \
    adapted_simpleclick=$res/adapted_simpleclick.json adapted_sam=$res/adapted_sam.json \
    adapted_ritm=$res/adapted_ritm.json adapted_ritm_coco=$res/adapted_ritm_coco.json \
    --compare accepted:base accepted:adapted accepted:teacher gt:accepted accepted:simpleclick accepted:imis \
    $opt --json $res/summary$sfx.json")
  for jit in 0 0.1; do
    cmds+=("$PY tools/paper_stats.py sam=$res/box${jit}_base_sam.json sam2=$res/box${jit}_base_sam2.json \
      medsam=$res/box${jit}_base_medsam.json imis=$res/box${jit}_base_imis.json \
      'accepted=$res/box${jit}_accepted_s*.json' 'gt=$res/box${jit}_gt_s*.json' \
      --compare accepted:sam2 accepted:medsam accepted:imis gt:accepted \
      $opt --json $res/summary_box$jit$sfx.json")
  done
  for band in 0-1 0-0.1; do
    cmds+=("$PY tools/paper_stats.py sam2=$res/rc${band}_base_sam2.json simpleclick=$res/rc${band}_base_simpleclick.json \
      imis=$res/rc${band}_base_imis.json 'accepted=$res/rc${band}_accepted_s*.json' 'gt=$res/rc${band}_gt_s*.json' \
      robot_accepted='$res/accepted_s*.json' \
      --compare accepted:sam2 accepted:imis accepted:simpleclick gt:accepted accepted:robot_accepted \
      $opt --json $res/summary_rc$band$sfx.json")
  done
  local IFS='&'; echo "${cmds[*]}" | sed 's/&/ \&\& /g'
}
for t in $TESTSETS; do
  set=${t%%:*}; data=${t#*:}; res=results/$set
  cmd=$(summaries "$res" "" "")
  if [ "$set" = clean ]; then
    cmd="$PY tools/clean_subgroups.py --data $data && $cmd"
    for sub in dominant_source other_sources lesion_ge5; do
      cmd="$cmd && $(summaries "$res" "_$sub" "--names $data/subgroups/$sub.txt")"
    done
  fi
  submit "stats-$set" 1:00:00 "$dep" "$cmd" > /dev/null
done
echo "submitted; watch with: squeue -u \$USER"
