"""
Freeze ISIC 2018 Task 1 into one benchmark folder with the standard splits.

All three official splits go into one folder (images/, masks/, manifest.json,
shrunk as in build_lesion_benchmark.py), so every tool reads the same files:

  session  -- --n-session images drawn (seeded) from the training split: the
              objects a simulated user labels before "Adapt now" (teacher)
  train    -- the rest of the training split: the distillation pool
  val      -- the validation split (100): checkpoint selection only
  test     -- the test split (1000): final numbers only

Run from repo root after unzipping the six Task 1 archives from
https://challenge.isic-archive.com/data/#2018 into one folder:
    python tools/build_isic_protocol.py --raw datasets/isic2018/raw --out datasets/isic2018_protocol
Then, for example:
    python tools/click_benchmark.py --model sam2 --data datasets/isic2018_protocol --eval-split test --no-adapt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from build_lesion_benchmark import convert    # noqa: E402

SPLITS = {   # our split -> (image folder, mask folder) as unzipped
    'train': ('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth'),
    'val': ('ISIC2018_Task1-2_Validation_Input', 'ISIC2018_Task1_Validation_GroundTruth'),
    'test': ('ISIC2018_Task1-2_Test_Input', 'ISIC2018_Task1_Test_GroundTruth'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raw', type=Path, default=ROOT / 'datasets' / 'isic2018' / 'raw')
    ap.add_argument('--out', type=Path, default=ROOT / 'datasets' / 'isic2018_protocol')
    ap.add_argument('--max-side', type=int, default=1024)
    ap.add_argument('--n-session', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    for images, masks in SPLITS.values():
        for d in (images, masks):
            if not (args.raw / d).is_dir():
                sys.exit(f'Missing {args.raw / d}')

    items = []
    for split, (images, masks) in SPLITS.items():
        part = convert(args.raw / images, args.raw / masks, args.out, args.max_side)
        for it in part:
            it['split'] = split
        if split == 'train':
            order = np.random.default_rng(args.seed).permutation(len(part))
            for i in order[:args.n_session]:
                part[i]['split'] = 'session'
        items += part

    manifest = {
        'source': 'ISIC 2018 Task 1, official training/validation/test splits; '
                  'session is a seeded subset of training',
        'max_side': args.max_side, 'seed': args.seed, 'items': items,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    counts = {s: sum(it['split'] == s for it in items) for s in ('session', 'train', 'val', 'test')}
    print(f'{len(items)} images -> {args.out}  {counts}')


if __name__ == '__main__':
    main()
