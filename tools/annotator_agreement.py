"""
Agreement between human annotators of the same lesion on the clean IMA++
multi-annotator set (build_clean_isic.py --out-multi), as a reference for how
closely any model can be expected to match a single reference mask.

For every image, the IoU of each pair of distinct annotators is computed and
averaged over the pairs of that image; the summary reports the mean and median
over images with a percentile-bootstrap 95% interval, and the share of images
on which a second annotator reaches IoU 0.85 and 0.90 (the NoC targets).
The same is reported per pair of annotation tools, using the IMA++ codes
T1 (manual polygon), T2 (semi-automated flood fill) and T3 (automated mask,
reviewed and accepted by a human expert), and per subgroup of
clean_subgroups.py. The IoU of each annotator with the STAPLE reference is
reported too; it is optimistic, since the reference is built from the same
masks.

    python tools/annotator_agreement.py
writes results/agreement/summary.json (and per_image.json).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOMINANT_SIZE = [2016, 3024]  # as in clean_subgroups.py
TARGETS = (0.85, 0.90)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def read(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f'cannot read {path}')
    return m > 127


def summarize(values: list[float], n_boot: int, rng) -> dict:
    v = np.asarray(values)
    if not len(v):
        return {'n': 0}
    boot = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {'n': len(v), 'mean': float(v.mean()), 'ci95': [float(lo), float(hi)],
           'median': float(np.median(v))}
    out.update({f'share_ge{t:.2f}': float((v >= t).mean()) for t in TARGETS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, default=ROOT / 'datasets' / 'clean_isic_multi')
    ap.add_argument('--seg-meta', type=Path, default=ROOT / 'datasets' / 'imapp' / 'raw' / 'seg_metadata.csv')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'agreement')
    ap.add_argument('--min-lesion-frac', type=float, default=0.05)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    items = json.loads((args.data / 'manifest.json').read_text())['items']
    # build_clean_isic.py keeps the last mask listed for an annotator, so the last row gives its tool.
    tool = {}
    for s in csv.DictReader(open(args.seg_meta, encoding='utf-8')):
        tool[s['ISIC_id'], s['annotator']] = s['tool']

    per_image = []
    for n, it in enumerate(items, 1):
        name = it['name']
        masks = {a: read(args.data / 'masks_annotators' / f'{name}__{a}.png') for a in it['annotators']}
        ref = read(args.data / 'masks' / f'{name}.png')
        pairs = defaultdict(list)
        for a, b in itertools.combinations(sorted(masks), 2):
            key = '-'.join(sorted((tool[name, a], tool[name, b])))
            pairs[key].append(iou(masks[a], masks[b]))
        per_image.append({
            'name': name,
            'pairwise': float(np.mean([v for vs in pairs.values() for v in vs])),
            'by_tools': {k: float(np.mean(vs)) for k, vs in pairs.items()},
            'vs_reference': float(np.mean([iou(m, ref) for m in masks.values()])),
            'dominant_source': sorted(it['orig_size']) == DOMINANT_SIZE,
            'lesion_ge5': it['lesion_frac'] >= args.min_lesion_frac,
        })
        if n % 200 == 0 or n == len(items):
            print(f'  {n}/{len(items)}', flush=True)

    rng = np.random.default_rng(args.seed)
    groups = {
        'all': per_image,
        'dominant_source': [r for r in per_image if r['dominant_source']],
        'other_sources': [r for r in per_image if not r['dominant_source']],
        'lesion_ge5': [r for r in per_image if r['lesion_ge5']],
    }
    summary = {'pairwise': {g: summarize([r['pairwise'] for r in rs], args.n_boot, rng)
                            for g, rs in groups.items()},
               'vs_reference': summarize([r['vs_reference'] for r in per_image], args.n_boot, rng)}
    keys = sorted({k for r in per_image for k in r['by_tools']})
    summary['by_tools'] = {k: summarize([r['by_tools'][k] for r in per_image if k in r['by_tools']],
                                        args.n_boot, rng) for k in keys}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (args.out / 'per_image.json').write_text(json.dumps(per_image, indent=0))

    def line(label, s):
        if s['n']:
            print(f"{label:<22}{s['n']:>5}  {s['mean']:.3f} ({s['ci95'][0]:.3f}-{s['ci95'][1]:.3f})  "
                  f"median {s['median']:.3f}  >=0.85 {s['share_ge0.85']:.1%}  >=0.90 {s['share_ge0.90']:.1%}")
    print(f"{'':<22}{'n':>5}  mean IoU (95% CI)")
    for g, s in summary['pairwise'].items():
        line(f'pairwise, {g}', s)
    for k, s in summary['by_tools'].items():
        line(f'pairwise, {k}', s)
    line('vs. STAPLE reference', summary['vs_reference'])


if __name__ == '__main__':
    main()
