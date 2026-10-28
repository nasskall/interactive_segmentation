"""
Freeze the ISIC 2018 Task 1 validation set into a fixed lesion benchmark.

The lesion RITM checkpoint was fine-tuned on the ISIC 2018 training split,
so the validation split (100 dermoscopy images, one expert lesion mask each,
CC0) is held out from every model here.

Images are shrunk to at most --max-side pixels on the long side (some
originals are 6700 px wide, which makes CPU benchmarks crawl); masks are
resized with nearest-neighbour and stored as 0/255 PNGs. A seeded split
assigns each image to:

  session  -- objects the simulated user labels before "Adapt now"
  eval     -- held-out objects every model is scored on
  reserve  -- left untouched, to grow eval later without reshuffling

Run from repo root after downloading and unzipping
ISIC2018_Task1-2_Validation_Input.zip and ISIC2018_Task1_Validation_GroundTruth.zip
from https://challenge.isic-archive.com/data/#2018 into datasets/isic2018_val/raw:
    python tools/build_lesion_benchmark.py
Then: python tools/click_benchmark.py --model sam2 --data datasets/lesion_bench

With --all-split NAME every image goes to that one split, e.g. the
distillation pool from the ISIC 2018 test split:
    python tools/build_lesion_benchmark.py --all-split train         --images datasets/isic2018_test/raw/ISIC2018_Task1-2_Test_Input         --masks datasets/isic2018_test/raw/ISIC2018_Task1_Test_GroundTruth         --out datasets/lesion_distill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets' / 'isic2018_val' / 'raw'


def convert(images: Path, masks: Path, out: Path, max_side: int) -> list[dict]:
    """Shrink every ISIC image in ``images`` (and its ``<name>_segmentation.png``
    in ``masks``) into ``out/images`` and ``out/masks``; one manifest item each,
    without a split."""
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    items = []
    for name in sorted(p.stem for p in images.glob('ISIC_*.jpg')):
        bgr = cv2.imread(str(images / f'{name}.jpg'))
        gt = cv2.imread(str(masks / f'{name}_segmentation.png'), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None or bgr.shape[:2] != gt.shape:
            sys.exit(f'{name}: missing image or mask, or their sizes differ')
        h, w = gt.shape
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            size = (round(w * scale), round(h * scale))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(gt > 127, 255, 0).astype(np.uint8)
        cv2.imwrite(str(out / 'images' / f'{name}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / 'masks' / f'{name}.png'), mask)
        items.append({'name': name, 'orig_size': [w, h], 'size': [bgr.shape[1], bgr.shape[0]],
                      'lesion_frac': round(float((mask > 0).mean()), 4)})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--images', type=Path, default=RAW / 'ISIC2018_Task1-2_Validation_Input')
    ap.add_argument('--masks', type=Path, default=RAW / 'ISIC2018_Task1_Validation_GroundTruth')
    ap.add_argument('--out', type=Path, default=ROOT / 'datasets' / 'lesion_bench')
    ap.add_argument('--max-side', type=int, default=1024)
    ap.add_argument('--n-session', type=int, default=20)
    ap.add_argument('--n-eval', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all-split', default=None, help='put every image in this one split')
    args = ap.parse_args()

    names = sorted(p.stem for p in args.images.glob('ISIC_*.jpg'))
    if args.all_split is None and len(names) < args.n_session + args.n_eval:
        sys.exit(f'Found {len(names)} images in {args.images}; '
                 f'need {args.n_session + args.n_eval}')
    items = convert(args.images, args.masks, args.out, args.max_side)

    order = np.random.default_rng(args.seed).permutation(len(items))
    for rank, i in enumerate(order):
        items[i]['split'] = (args.all_split if args.all_split else
                             'session' if rank < args.n_session else
                             'eval' if rank < args.n_session + args.n_eval else 'reserve')

    manifest = {
        'source': (f'{args.images.name} with masks from {args.masks.name}' if args.all_split else
                   'ISIC 2018 Task 1 validation split (CC0); held out from the lesion RITM, '
                   'which was fine-tuned on the ISIC 2018 training split'),
        'max_side': args.max_side, 'seed': args.seed, 'items': items,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    counts = {s: sum(it['split'] == s for it in items) for s in {it['split'] for it in items}}
    fracs = np.array([it['lesion_frac'] for it in items])
    print(f'{len(items)} images -> {args.out}  {counts}')
    print(f'lesion covers {fracs.min():.1%}..{fracs.max():.1%} of the frame '
          f'(median {np.median(fracs):.1%})')


if __name__ == '__main__':
    main()
