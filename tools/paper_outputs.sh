#!/usr/bin/env bash
# Build the paper's tables and figures from the results that
# tools/run_isic_benchmarks.sh leaves in results/<set>/ (the summary*.json of
# its stats jobs and the per-image click curves). CPU only, a few seconds:
#
#   bash tools/paper_outputs.sh [results dir] [output dir]
#
# Defaults: results/ and paper/draft/generated/. Every table is a LaTeX
# fragment for \input; a model or comparison without results shows a dash.
set -euo pipefail
RES=${1:-results}
OUT=${2:-paper/draft/generated}
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=.venv/Scripts/python.exe
T="$PY tools/paper_tables.py"
SETS=(--summary "ISIC 2018=$RES/isic/summary.json" --summary "Challenge-disjoint=$RES/clean/summary.json")
declare -A NAME=([isic]="ISIC 2018" [clean]="challenge-disjoint")
mkdir -p "$OUT"

CLICK_ROWS=("base=SAM 2" "adapted=SAM 2, adapted online" "simpleclick=SimpleClick" "sam=SAM"
            "imis=IMIS-Net" "teacher=Student (teacher)" "accepted=Student (accepted)"
            "gt=Student (expert)")
CMP_ROWS=("accepted:base=Accepted vs. SAM 2" "accepted:adapted=Accepted vs. adapted SAM 2"
          "accepted:teacher=Accepted vs. teacher" "gt:accepted=Expert vs. accepted"
          "accepted:simpleclick=Accepted vs. SimpleClick" "accepted:imis=Accepted vs. IMIS-Net")

$T table "${SETS[@]}" --metrics iou@1 noc@90 --rows "${CLICK_ROWS[@]}" --seed-sd \
  --caption "Robot-click results on both test sets: mean IoU after the first click and NoC@90, with 95\% bootstrap confidence intervals below each value and the best value in bold. Student rows average three seeds." \
  --label tab:click --out "$OUT/click.tex"
$T compare "${SETS[@]}" --metrics noc@90 --rows "${CMP_ROWS[@]}" \
  --caption "Paired bootstrap comparisons of NoC@90 (first model minus second, with its 95\% confidence interval and two-sided \$p\$); negative values favor the first model." \
  --label tab:click_cmp --out "$OUT/click_cmp.tex"

# Challenge-disjoint set (results/clean) by subgroup (tools/clean_subgroups.py).
$T table --summary "Dominant source=$RES/clean/summary_dominant_source.json" \
  --summary "Other sources=$RES/clean/summary_other_sources.json" \
  --summary 'Lesions $\geq$5\%='"$RES/clean/summary_lesion_ge5.json" \
  --metrics noc@90 --rows "${CLICK_ROWS[@]}" \
  --caption "NoC@90 on subgroups of the challenge-disjoint test set, with 95\% bootstrap confidence intervals below each value." \
  --label tab:subgroups --out "$OUT/subgroups.tex"

# Box prompts: tight and 10% jitter side by side, per test set.
for set in isic clean; do
  $T table --summary "Tight box=$RES/$set/summary_box0.json" \
    --summary "10% jitter=$RES/$set/summary_box0.1.json" --metrics iou dice \
    --rows "sam=SAM" "sam2=SAM 2" "medsam=MedSAM" "imis=IMIS-Net" \
           "accepted=Student (accepted)" "gt=Student (expert)" \
    --caption "Single box prompt on the ${NAME[$set]} test set: IoU and Dice for the tight box and a box with each side shifted by up to 10\% of its size, with 95\% bootstrap confidence intervals below each value." \
    --label "tab:box_$set" --out "$OUT/box_$set.tex"
done

# Realistic clicks (RClicks): whole distribution and least likely tenth.
for set in isic clean; do
  $T table --summary "RClicks (0, 1)=$RES/$set/summary_rc0-1.json" \
    --summary "RClicks (0, 0.1)=$RES/$set/summary_rc0-0.1.json" --metrics iou@1 noc@90 \
    --rows "sam2=SAM 2" "simpleclick=SimpleClick" "imis=IMIS-Net" \
           "accepted=Student (accepted)" "robot_accepted=Same, robot clicks" \
           "gt=Student (expert)" \
    --caption "Realistic clicks on the ${NAME[$set]} test set, sampled from the clickability model over its whole distribution or its least likely tenth, with 95\% bootstrap confidence intervals below each value. The second student row repeats the robot-click result of the same model." \
    --label "tab:rclicks_$set" --out "$OUT/rclicks_$set.tex"
done

# Main-text summary of both: box IoU and realistic-click NoC@90 in one table
# (the two tables above go to the supplementary material).
for set in isic clean; do
  $T table --summary "Tight box=$RES/$set/summary_box0.json::iou" \
    --summary "Box, 10% jitter=$RES/$set/summary_box0.1.json::iou" \
    --summary "RClicks (0, 1)=$RES/$set/summary_rc0-1.json::noc@90" \
    --summary "RClicks (0, 0.1)=$RES/$set/summary_rc0-0.1.json::noc@90" --metrics iou \
    --rows "sam=SAM" "sam2=SAM 2" "simpleclick=SimpleClick" "medsam=MedSAM" "imis=IMIS-Net" \
           "accepted=Student (accepted)" "gt=Student (expert)" \
    --caption "Robustness on the ${NAME[$set]} test set: IoU for a single tight or jittered box (each side shifted by up to 10\% of its size) and NoC@90 for realistic clicks sampled from the whole clickability distribution or its least likely tenth, with 95\% bootstrap confidence intervals below each value." \
    --label "tab:robust_$set" --out "$OUT/robust_$set.tex"
done

$T curves --panels "ISIC 2018=$RES/isic" "Challenge-disjoint=$RES/clean" \
  --groups "SAM 2=base_sam2.json" "SAM 2, adapted=adapted_sam2.json" \
           "SimpleClick=base_simpleclick.json" "IMIS-Net=base_imis.json" \
           "Student (teacher)=teacher_s*.json" "Student (accepted)=accepted_s*.json" \
           "Student (expert)=gt_s*.json" \
  --out "$OUT/curves.pdf"
echo "tables and figure -> $OUT"
