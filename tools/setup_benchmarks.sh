#!/usr/bin/env bash
# Set up everything the paper benchmarks need on a Linux GPU server, beside
# the app's own venv (deploy/ailab26.md, step 2). Safe to re-run: files that
# exist and verify are kept, anything else is fetched again.
#
#   bash tools/setup_benchmarks.sh            # code, venvs, checkpoints
#   bash tools/setup_benchmarks.sh --data     # + ISIC 2018 (13.8 GB) and the clean IMA++ sets
#   bash tools/setup_benchmarks.sh --smoke    # + a two-image run of every pipeline on the GPU
# Then submit the experiments with tools/run_isic_benchmarks.sh.
#
# Run from the repository root. Layout (EXT defaults to ../external):
#   $EXT/IMIS-Bench   + imis-venv      IMIS-Net, unmodified (tools/imis_benchmark.py)
#   $EXT/rclicks      + rclicks-venv   RClicks clickability model (tools/rclicks_worker.py)
#   $EXT/MedSAM/work_dir/MedSAM/medsam_vit_b.pth                   (tools/box_benchmark.py)
#   model_weights/    SAM ViT-B, SAM 2.1 Tiny, SimpleClick ViT-B, RITM COCO+LVIS
# The side venvs reuse the app venv's packages (PyTorch built for the GPUs)
# through a .pth file and add only what differs. Hashes of files without a
# published one are written to model_weights/SHA256SUMS.benchmarks on first
# download; record them in the model register.
set -euo pipefail
ROOT=$(pwd)
EXT=$(realpath -m "${EXT:-$ROOT/../external}")
VENV=${VENV:-$ROOT/.venv}
PY=$VENV/bin/python
DATA=0 SMOKE=0
for a in "$@"; do
  case $a in --data) DATA=1 ;; --smoke) SMOKE=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
[ -f tools/setup_benchmarks.sh ] || { echo "run from the repository root"; exit 1; }
[ -x "$PY" ] || { echo "no app venv at $VENV (deploy/ailab26.md, step 2)"; exit 1; }
mkdir -p "$EXT" model_weights
HASHES=model_weights/SHA256SUMS.benchmarks
log() { printf '\n== %s\n' "$*"; }

verify() {   # verify <file> <sha256 or -> <size or ->
  local f=$1 sha=$2 size=$3
  [ -f "$f" ] || return 1
  if [ "$sha" != - ]; then [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; return; fi
  if [ "$size" != - ]; then [ "$(stat -c %s "$f")" = "$size" ]; return; fi
  return 0
}

record() {   # keep the hash of a file that has no published one
  grep -q " $1\$" "$HASHES" 2>/dev/null || sha256sum "$1" >> "$HASHES"
}

fetch() {   # fetch <url> <dest> <sha256 or -> <size or ->
  local url=$1 dest=$2 sha=$3 size=$4
  if verify "$dest" "$sha" "$size"; then echo "  ok  $dest"; return; fi
  echo "  get $dest"
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 5 --retry-all-errors -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  verify "$dest" "$sha" "$size" || { echo "  FAILED check: $dest"; mv "$dest" "$dest.bad"; exit 1; }
  [ "$sha" = - ] && record "$dest"
  return 0
}

gdrive() {   # gdrive <file id> <dest> <sha256 or ->  (Google Drive, via gdown)
  local id=$1 dest=$2 sha=$3
  if verify "$dest" "$sha" -; then echo "  ok  $dest"; return; fi
  echo "  get $dest"
  mkdir -p "$(dirname "$dest")"
  "$PY" -m gdown --quiet "$id" -O "$dest.part"
  mv "$dest.part" "$dest"
  verify "$dest" "$sha" - || { echo "  FAILED check: $dest"; mv "$dest" "$dest.bad"; exit 1; }
  [ "$sha" = - ] && record "$dest"
  return 0
}

clone() {   # clone <url> <dir> <commit>
  local url=$1 dir=$2 commit=$3
  [ -d "$dir/.git" ] || git clone -q --filter=blob:none "$url" "$dir"
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout -q "$commit"
  echo "  ok  $dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

side_venv() {   # side_venv <name> <pinned packages...>
  local name=$1; shift
  local v=$EXT/$name
  [ -x "$v/bin/python" ] || "$PY" -m venv "$v"
  local own parent
  own=$("$v/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  parent=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  # Appended after the venv's own site-packages, so pinned versions here win.
  echo "$parent" > "$own/zz_app_venv.pth"
  "$v/bin/python" -m pip install -q --upgrade pip
  "$v/bin/python" -m pip install -q "$@"
  echo "  ok  $v ($*)"
}

log "app venv: extra packages for the benchmark tools"
"$PY" -m pip install -q scikit-image==0.26.0 gdown==6.4.2
"$PY" -c "import torch, segment_anything, sam2, skimage; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"

log "model_weights"
fetch https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
      model_weights/sam_vit_b_01ec64.pth ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912 -
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
      model_weights/sam2.1_hiera_tiny.pt - 156008466
gdrive 1Ouy_Oq_L_pEiGfMqF9LbzYVb70H13-TB model_weights/cocolvis_vit_base.pth -    # SimpleClick model zoo
fetch https://github.com/lucasew/ritm_annotation/releases/download/0.0.1/coco_lvis_h18_itermask.pth \
      model_weights/coco_lvis_h18_itermask.pth 5f69cfce354d1507e3850bfc39ee7057c8dd27b6a4910d1d2dc724916b9ee32b -

log "IMIS-Net (D5, D8)"
IMIS=$EXT/IMIS-Bench
clone https://github.com/uni-medical/IMIS-Bench.git "$IMIS" f4519c0f8f5a6e62e1a2bb9dde52be63c4a4fb30
fetch https://huggingface.co/1Junlong/IMIS-Net/resolve/c3d2eddcc07537bc67cb97003905bf8a4bd072f5/IMISNet-B.pth \
      "$IMIS/ckpt/IMISNet-B.pth" 1dd95d16dc359115463f8101d697f8443cb3b8dc359e3aa38e79f317bf1cb84a -
CLIP=https://huggingface.co/openai/clip-vit-base-patch32/resolve/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
fetch $CLIP/merges.txt             "$IMIS/clip-vit-base-patch32/merges.txt"             f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85 -
fetch $CLIP/special_tokens_map.json "$IMIS/clip-vit-base-patch32/special_tokens_map.json" f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf -
fetch $CLIP/tokenizer.json         "$IMIS/clip-vit-base-patch32/tokenizer.json"         b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842 -
fetch $CLIP/tokenizer_config.json  "$IMIS/clip-vit-base-patch32/tokenizer_config.json"  34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad -
fetch $CLIP/vocab.json             "$IMIS/clip-vit-base-patch32/vocab.json"             5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363 -
side_venv imis-venv monai==1.6.1 transformers==4.57.6

log "MedSAM (D9)"
gdrive 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ "$EXT/MedSAM/work_dir/MedSAM/medsam_vit_b.pth" \
       34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74

log "RClicks (D12)"
RCL=$EXT/rclicks
clone https://github.com/emb-ai/rclicks.git "$RCL" e20bf7636198545331a9d383b77e42107e97ea09
CKPT=$RCL/ckpt/clickability_model.pth
CKPT_SHA=487beaddd351cd7045170234b0e719e111e80f73ab7055c523fde70b21f96960
if verify "$CKPT" $CKPT_SHA -; then echo "  ok  $CKPT"; else
  gdrive 1bOgTrkwbnW4GnIQUr9lIY9zNmGRKvdbB "$RCL/ckpt/CLICKABILITY_CHECKPOINTS.zip" -
  "$PY" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extract('clickability_model.pth', sys.argv[2])" \
        "$RCL/ckpt/CLICKABILITY_CHECKPOINTS.zip" "$RCL/ckpt"
  rm "$RCL/ckpt/CLICKABILITY_CHECKPOINTS.zip"
  if [ -f "$HASHES" ]; then sed -i '\#CLICKABILITY_CHECKPOINTS.zip#d' "$HASHES"; fi
  verify "$CKPT" $CKPT_SHA - || { echo "  FAILED check: $CKPT"; exit 1; }
  echo "  ok  $CKPT"
fi
side_venv rclicks-venv mmengine==0.10.7 mmcv-lite==2.1.0 ftfy==6.3.1 regex==2026.9.10 \
          albumentations==2.0.8 easydict==1.13

if [ $DATA = 1 ]; then
  log "ISIC 2018 Task 1 (tools/build_isic_protocol.py)"
  ISIC=https://isic-challenge-data.s3.amazonaws.com/2018
  RAW=datasets/isic2018/raw
  while read -r f size; do
    fetch "$ISIC/$f" "$RAW/$f" - "$size"
    d=$RAW/${f%.zip}
    [ -d "$d" ] || "$PY" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$RAW/$f" "$RAW"
  done <<'EOF'
ISIC2018_Task1-2_Training_Input.zip 11165358566
ISIC2018_Task1_Training_GroundTruth.zip 27402895
ISIC2018_Task1-2_Validation_Input.zip 239231159
ISIC2018_Task1_Validation_GroundTruth.zip 759706
ISIC2018_Task1-2_Test_Input.zip 2370457338
ISIC2018_Task1_Test_GroundTruth.zip 9680101
EOF
  [ -f datasets/isic2018_protocol/manifest.json ] || \
    "$PY" tools/build_isic_protocol.py --raw "$RAW" --out datasets/isic2018_protocol

  log "IMA++ clean sets (tools/build_clean_isic.py, D10)"
  Z=https://zenodo.org/records/14201693/files
  RAW=datasets/imapp/raw
  while read -r f md5; do
    if [ -f "$RAW/$f" ] && [ "$(md5sum "$RAW/$f" | cut -d' ' -f1)" = "$md5" ]; then echo "  ok  $RAW/$f"; continue; fi
    echo "  get $RAW/$f"; mkdir -p "$RAW"
    curl -fsSL --retry 5 --retry-all-errors -o "$RAW/$f" "$Z/$f?download=1"
    [ "$(md5sum "$RAW/$f" | cut -d' ' -f1)" = "$md5" ] || { echo "  FAILED MD5: $RAW/$f"; exit 1; }
  done <<'EOF'
img_metadata.csv c27fdbcaab1e2cae48d6ad7edd3d64c4
seg_metadata.csv aae21dec31b48e45786da70c4609e45f
segs.zip c5143aed98283ab57de2c86ce50973c8
EOF
  [ -f datasets/clean_isic_multi/manifest.json ] || "$PY" tools/build_clean_isic.py
  # The sets must match the ones built for the paper on 2026-10-28.
  while read -r d n sha; do
    got=$("$PY" -c "import json, hashlib, sys; n = sorted(i['name'] for i in json.load(open(sys.argv[1]))['items']); print(len(n), hashlib.sha256('\n'.join(n).encode()).hexdigest())" "datasets/$d/manifest.json")
    [ "$got" = "$n $sha" ] && echo "  ok  $d ($n images, same as the paper's)" || { echo "  MISMATCH $d: $got"; exit 1; }
  done <<'EOF'
clean_isic_test 1000 1c945009687b018eb94f8aeccbdfe3a9f7a1d1a5c39dcb0691a9dc49df8a2952
clean_isic_multi 787 ed549534cd9527b7f23a39328f91cf8381b2393dd2e989fea2275c1193c26149
EOF
fi

if [ $SMOKE = 1 ]; then
  log "smoke test: two images per pipeline"
  T=$(mktemp -d)
  B=datasets/clean_isic_test
  [ -f $B/manifest.json ] || B=datasets/isic2018_protocol
  [ -f $B/manifest.json ] || { echo "  no benchmark data yet: run with --data first"; exit 1; }
  "$PY" tools/click_benchmark.py --model sam2 --data $B --eval-split test --no-adapt --n-eval 2 --per-image "$T/c.json" | tail -1
  "$PY" tools/box_benchmark.py --model medsam --checkpoint "$EXT/MedSAM/work_dir/MedSAM/medsam_vit_b.pth" \
        --data $B --eval-split test --n-eval 2 --per-image "$T/b.json" | tail -1
  "$EXT/imis-venv/bin/python" tools/imis_benchmark.py --imis-root "$IMIS" --checkpoint "$IMIS/ckpt/IMISNet-B.pth" \
        --data $B --eval-split test --n-eval 2 --per-image "$T/i.json" | tail -1
  "$PY" tools/click_benchmark.py --model sam2 --data $B --eval-split test --no-adapt --n-eval 2 --max-clicks 5 \
        --clicker rclicks --rclicks-python "$EXT/rclicks-venv/bin/python" --rclicks-root "$RCL" \
        --rclicks-checkpoint "$CKPT" --per-image "$T/r.json" | tail -1
  "$PY" tools/paper_stats.py "sam2=$T/c.json" "rclicks=$T/r.json" --n-boot 100 | tail -2
  rm -rf "$T"
fi
log "done"
