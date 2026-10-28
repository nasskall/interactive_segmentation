"""
Fill the \\res{key} placeholders of the paper from the summary JSONs that
paper_stats.py --json writes, so that no number is copied by hand.

A key file lists one placeholder per line (blank lines and # comments are
skipped), with paths relative to the results directory:

    clean-acc-noc90   mean   clean/summary.json  accepted     noc@90
    clean-gt-acc-d    diff   clean/summary.json  gt:accepted  noc@90
    clean-gt-acc-ci   ci     clean/summary.json  gt:accepted  noc@90
    clean-gt-acc-p    p      clean/summary.json  gt:accepted  noc@90
    clean-acc-drop    delta  iou  +clean/summary_box0.1.json:accepted -clean/summary_box0.json:accepted

mean is a model's mean; diff, ci and p are a paired comparison (first model
minus second); delta is a signed sum of model means. Numbers follow the
tables: three decimals for IoU and Dice, two for NoC, signed values in math
mode, p as $p<0.001$ or $p=0.012$.

    python tools/paper_numbers.py --keys paper/draft/numbers.txt --results results \\
        --out paper/draft/generated/numbers.tex [--tex paper/draft --strict]

writes one definition per key that resolves. With --tex, every \\res{key}
used in the .tex files of that folder is checked against the key file;
--strict exits non-zero if any key is undefined or unresolved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def digits(metric: str) -> int:
    return 2 if metric.startswith('noc') else 3


def signed(x: float, metric: str) -> str:
    return f'${x:+.{digits(metric)}f}$'


class Results:
    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def summary(self, rel: str) -> dict:
        if rel not in self.cache:
            path = self.root / rel
            if not path.is_file():
                raise KeyError(f'{path} not found')
            self.cache[rel] = json.loads(path.read_text(encoding='utf-8'))
        return self.cache[rel]

    def mean(self, rel: str, model: str, metric: str) -> float:
        models = self.summary(rel)['models']
        if model not in models:
            raise KeyError(f'model {model} not in {rel}')
        return models[model][metric]['mean']

    def comparison(self, rel: str, pair: str, metric: str) -> dict:
        comps = self.summary(rel)['comparisons']
        if pair not in comps:
            raise KeyError(f'comparison {pair} not in {rel}')
        return comps[pair][metric]


def value(res: Results, kind: str, args: list[str]) -> str:
    if kind == 'mean':
        rel, model, metric = args
        return f'{res.mean(rel, model, metric):.{digits(metric)}f}'
    if kind in ('diff', 'ci', 'p'):
        rel, pair, metric = args
        c = res.comparison(rel, pair, metric)
        if kind == 'diff':
            return signed(c['diff'], metric)
        if kind == 'ci':
            return f"{signed(c['ci95'][0], metric)} to {signed(c['ci95'][1], metric)}"
        return '$p<0.001$' if c['p'] < 0.001 else f"$p={c['p']:.3f}$"
    if kind == 'delta':
        metric, terms = args[0], args[1:]
        total = 0.0
        for t in terms:
            sign, (rel, model) = t[0], t[1:].rsplit(':', 1)
            if sign not in '+-':
                raise ValueError(f'delta term {t} must start with + or -')
            total += res.mean(rel, model, metric) * (1 if sign == '+' else -1)
        return signed(total, metric)
    raise ValueError(f'unknown kind {kind}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--keys', type=Path, required=True)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tex', type=Path, default=None, help='folder whose .tex files use \\res{key}')
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()

    res = Results(args.results)
    specs, lines, problems = {}, [], []
    for n, raw in enumerate(args.keys.read_text(encoding='utf-8').splitlines(), 1):
        parts = raw.split('#')[0].split()
        if not parts:
            continue
        if len(parts) < 3:
            sys.exit(f'{args.keys}:{n}: expected KEY KIND ARGS..., got {raw!r}')
        key, kind, rest = parts[0], parts[1], parts[2:]
        if key in specs:
            sys.exit(f'{args.keys}:{n}: key {key} defined twice')
        specs[key] = (kind, rest)
        try:
            lines.append(rf'\expandafter\def\csname res@{key}\endcsname{{{value(res, kind, rest)}}}')
        except (KeyError, ValueError) as e:
            problems.append(f'{key}: {e}')

    if args.tex:
        used = set()
        for f in sorted(args.tex.glob('*.tex')):
            used |= set(re.findall(r'\\res\{([^}#]+)\}', f.read_text(encoding='utf-8')))
        used.discard('key')  # the macro's own documentation in main.tex
        problems += [f'{k}: used in the paper but not in {args.keys}' for k in sorted(used - set(specs))]
        unused = sorted(set(specs) - used)
        if unused:
            print(f'  note: keys not used in the paper: {", ".join(unused)}', file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('% Written by tools/paper_numbers.py; do not edit.\n' + '\n'.join(lines) + '\n',
                        encoding='utf-8', newline='\n')
    print(f'  numbers: {len(lines)} of {len(specs)} keys resolved -> {args.out}')
    for p in problems:
        print(f'  missing: {p}', file=sys.stderr)
    if args.strict and problems:
        sys.exit(1)


if __name__ == '__main__':
    main()
