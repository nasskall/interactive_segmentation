"""
Write subgroup image lists for a benchmark folder, for paper_stats.py --names.

The clean IMA++ test set (build_clean_isic.py) is dominated by one
acquisition source, inferred from the original image size (3024 x 2016,
about 70%; small benign nevi beside coloured calibration stickers), and its
median lesion covers about 2% of the image. Results are therefore also
reported on:

  dominant_source   original size 3024 x 2016 (either orientation)
  other_sources     every other original size
  lesion_ge5        lesion covers at least 5% of the image

Image size is only a proxy for the source; the ISIC metadata of these images
names no institution.

    python tools/clean_subgroups.py --data datasets/clean_isic_test
writes datasets/clean_isic_test/subgroups/<name>.txt, one image name per line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMINANT_SIZE = [2016, 3024]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--min-lesion-frac', type=float, default=0.05)
    args = ap.parse_args()

    items = json.loads((args.data / 'manifest.json').read_text())['items']
    groups = {
        'dominant_source': [i['name'] for i in items if sorted(i['orig_size']) == DOMINANT_SIZE],
        'other_sources': [i['name'] for i in items if sorted(i['orig_size']) != DOMINANT_SIZE],
        'lesion_ge5': [i['name'] for i in items if i['lesion_frac'] >= args.min_lesion_frac],
    }
    out = args.data / 'subgroups'
    out.mkdir(exist_ok=True)
    for name, names in groups.items():
        (out / f'{name}.txt').write_text('\n'.join(names) + '\n')
        print(f'{name}: {len(names)} of {len(items)}')


if __name__ == '__main__':
    main()
