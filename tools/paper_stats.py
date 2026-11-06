"""
Summarise per-image click curves with bootstrap confidence intervals.

Reads the JSON files that ``click_benchmark.py --per-image`` writes (click
curves: mIoU@1/@5, NoC@85/90) or that box_benchmark.py writes (one box per
object: IoU, Dice); the groups in one call must be of one kind. Each
argument is LABEL=GLOB; the files a glob matches are seeds of one model:
every per-image score is averaged over the seeds first, then a 95%
bootstrap interval is taken over images. The spread between seeds is shown
too.

    python tools/paper_stats.py base=results/sam2_base.json "accepted=results/accepted_s*.json" \\
        "gt=results/gt_s*.json" --compare accepted:base gt:accepted

--compare A:B gives the paired bootstrap of A minus B on the images both
were scored on (lower NoC is better, so a negative NoC difference favours A),
with a two-sided bootstrap p-value.

--names FILE restricts every model to the images listed in FILE (one name
per line), e.g. a subgroup written by tools/clean_subgroups.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np

CLICK_METRICS = ('iou@1', 'iou@5', 'noc@85', 'noc@90')
BOX_METRICS = ('iou', 'dice')


def per_image(curve: list[float], max_clicks: int) -> dict[str, float]:
    c = np.asarray(curve)

    def noc(t):
        hit = np.flatnonzero(c >= t)
        return float(hit[0] + 1) if hit.size else float(max_clicks)
    return {'iou@1': float(c[0]), 'iou@5': float(c[min(4, len(c) - 1)]),
            'noc@85': noc(0.85), 'noc@90': noc(0.90)}


def run_scores(run: dict) -> dict[str, dict[str, float]]:
    if 'curves' in run:
        return {n: per_image(c, run['max_clicks']) for n, c in run['curves'].items()}
    return run['scores']


def load_group(pattern: str, only: set[str] | None = None
               ) -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
    """Seed-averaged scores per image, and each seed's means; only the images
    in ``only`` when given."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'No files match {pattern}')
    runs = [run_scores(json.loads(open(f, encoding='utf-8').read())) for f in files]
    if only is not None:
        runs = [{n: v for n, v in r.items() if n in only} for r in runs]
    names = set.intersection(*(set(r) for r in runs))
    if not names:
        sys.exit(f'{pattern}: no images left' + (' after --names' if only is not None else ''))
    if any(len(r) != len(names) for r in runs):
        print(f'  note: {pattern}: seeds cover different images; using the {len(names)} shared',
              file=sys.stderr)
    metrics = list(runs[0][next(iter(names))])
    avg = {n: {m: float(np.mean([r[n][m] for r in runs])) for m in metrics} for n in names}
    seed_means = [{m: float(np.mean([r[n][m] for n in names])) for m in metrics} for r in runs]
    return avg, seed_means


def bootstrap(values: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    idx = rng.integers(len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('groups', nargs='+', help='LABEL=GLOB of per-image JSON files')
    ap.add_argument('--compare', nargs='*', default=[], help='A:B pairs of labels')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None, help='also write the summary here')
    ap.add_argument('--names', default=None, help='only these images (file, one name per line)')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    only = None
    if args.names:
        with open(args.names, encoding='utf-8') as f:
            only = {line.strip() for line in f if line.strip()}

    groups = {}
    for g in args.groups:
        label, _, pattern = g.partition('=')
        if not pattern:
            sys.exit(f'Expected LABEL=GLOB, got {g}')
        groups[label] = load_group(pattern, only)

    kinds = {tuple(next(iter(avg.values()))) for avg, _ in groups.values()}
    if len(kinds) > 1:
        sys.exit('Mixed click-curve and box files; summarise them in separate calls')
    METRICS = CLICK_METRICS if kinds.pop() == CLICK_METRICS else BOX_METRICS
    summary = {'models': {}, 'comparisons': {}, 'names': args.names}
    print(f'{"model":<14}{"n":>6}{"seeds":>6}  ' + '  '.join(f'{m:>24}' for m in METRICS))
    for label, (avg, seeds) in groups.items():
        row, cells = {}, []
        for m in METRICS:
            v = np.array([avg[n][m] for n in sorted(avg)])
            lo, hi = bootstrap(v, args.n_boot, rng)
            sd = float(np.std([s[m] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0
            row[m] = {'mean': float(v.mean()), 'ci95': [lo, hi], 'seed_sd': sd}
            cells.append(f'{v.mean():.3f} [{lo:.3f},{hi:.3f}] ±{sd:.3f}')
        summary['models'][label] = {'n_images': len(avg), 'n_seeds': len(seeds), **row}
        print(f'{label:<14}{len(avg):>6}{len(seeds):>6}  ' + '  '.join(f'{c:>24}' for c in cells))

    for pair in args.compare:
        a, _, b = pair.partition(':')
        if a not in groups or b not in groups:
            sys.exit(f'--compare {pair}: unknown label')
        (ga, _), (gb, _) = groups[a], groups[b]
        names = sorted(set(ga) & set(gb))
        res = {}
        line = []
        for m in METRICS:
            d = np.array([ga[n][m] - gb[n][m] for n in names])
            idx = rng.integers(len(d), size=(args.n_boot, len(d)))
            means = d[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
            res[m] = {'diff': float(d.mean()), 'ci95': [float(lo), float(hi)], 'p': float(p)}
            line.append(f'{m} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}')
        summary['comparisons'][pair] = {'n_images': len(names), **res}
        print(f'{a} - {b} ({len(names)} images): ' + '  '.join(line))

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
