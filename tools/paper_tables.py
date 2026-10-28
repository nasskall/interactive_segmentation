"""
Turn benchmark results into LaTeX tables and click-curve figures for the paper.

    table    one row per model, a column group per test set; reads the
             summary JSONs that paper_stats.py --json writes
    compare  the paired differences of those summaries, one row per pair
    curves   mean IoU after every click, one panel per results directory;
             reads the per-image JSONs that click_benchmark.py writes

    python tools/paper_tables.py table --summary "ISIC 2018=results/isic/summary.json" \\
        --summary "Challenge-disjoint=results/clean/summary.json" --metrics iou@1 noc@90 \\
        --rows "base=SAM 2" "accepted=Student (accepted)" "gt=Student (expert)" \\
        --caption "..." --label tab:click --out paper/draft/tables/click.tex
    python tools/paper_tables.py compare --summary "Challenge-disjoint=results/clean/summary.json" \\
        --metrics noc@90 --rows "gt:accepted=Expert - accepted" --out ...
    python tools/paper_tables.py curves --panels "ISIC 2018=results/isic" "Challenge-disjoint=results/clean" \\
        --groups "SAM 2=base_sam2.json" "Student (accepted)=accepted_s*.json" --out fig.pdf

Cells show the mean and its 95% bootstrap interval; the best mean of each
column is set in bold (lowest for NoC). A model missing from a summary gets
a dash, so one table can mix click-only and box-capable models.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HEADS = {'iou@1': r'mIoU@1$\uparrow$', 'iou@5': r'mIoU@5$\uparrow$',
         'noc@85': r'NoC@85$\downarrow$', 'noc@90': r'NoC@90$\downarrow$',
         'iou': r'IoU$\uparrow$', 'dice': r'Dice$\uparrow$'}
# Slots 1-7 of the validated categorical palette, each with its own marker so
# identity survives greyscale print and colour-vision deficiency.
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
AUTHOR = 'Athanasios Kallipolitis'


def pairs(items: list[str], what: str) -> list[tuple[str, str]]:
    out = []
    for it in items:
        key, sep, val = it.partition('=')
        if not sep or not val:
            sys.exit(f'Expected NAME=VALUE for {what}, got {it}')
        out.append((key, val))
    return out


def tex(s: str) -> str:
    """Escape plain text; a name that already contains LaTeX passes as is."""
    if '\\' in s:
        return s
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def lower_better(metric: str) -> bool:
    return metric.startswith('noc')


def fmt(x: float, metric: str, signed: bool = False) -> str:
    digits = 2 if lower_better(metric) else 3
    return f'${x:+.{digits}f}$' if signed else f'{x:.{digits}f}'


def cell(value: str, interval: str, inline: bool) -> str:
    """The value over its interval in small type, or both on one line."""
    if inline:
        return rf'{value} {{\scriptsize{interval}}}'
    return rf'\begin{{tabular}}[t]{{@{{}}c@{{}}}}{value}\\{{\scriptsize{interval}}}\end{{tabular}}'


def wrap(body: list[str], cols: str, caption: str | None, label: str | None) -> str:
    lines = [r'\begin{table}[t]', r'\centering']
    if caption:
        lines.append(rf'\caption{{{caption}}}')
    if label:
        lines.append(rf'\label{{{label}}}')
    lines += [r'\setlength{\tabcolsep}{4pt}', rf'\begin{{tabular}}{{{cols}}}', r'\toprule',
              *body, r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def load_summaries(items: list[str]) -> list[tuple[str, dict]]:
    """A missing summary (runs not finished yet) becomes a column of dashes."""
    out = []
    for name, path in pairs(items, '--summary'):
        if Path(path).is_file():
            out.append((name, json.loads(Path(path).read_text(encoding='utf-8'))))
        else:
            print(f'  note: {path} not found; {name} is left empty', file=sys.stderr)
            out.append((name, {'models': {}, 'comparisons': {}}))
    return out


def table(args) -> str:
    sets = load_summaries(args.summary)
    rows = pairs(args.rows, '--rows') if args.rows else \
        [(m, m) for m in sets[0][1]['models']]
    best = {}
    for s, summ in sets:
        for m in args.metrics:
            vals = [summ['models'][r][m]['mean'] for r, _ in rows if r in summ['models']]
            if vals:
                best[s, m] = min(vals) if lower_better(m) else max(vals)

    multi = len(sets) > 1
    head = ['Model'] + [HEADS.get(m, m) for _ in sets for m in args.metrics]
    body = []
    if multi:
        body.append(' & ' + ' & '.join(rf'\multicolumn{{{len(args.metrics)}}}{{c}}{{{tex(s)}}}'
                                       for s, _ in sets) + r' \\')
        start = 2
        body.append(' '.join(rf'\cmidrule(lr){{{start + i * len(args.metrics)}-'
                             rf'{start + (i + 1) * len(args.metrics) - 1}}}'
                             for i in range(len(sets))))
    body += [' & '.join(head) + r' \\', r'\midrule']
    for key, name in rows:
        cells = [tex(name)]
        for s, summ in sets:
            for m in args.metrics:
                r = summ['models'].get(key)
                if r is None:
                    cells.append('--')
                    continue
                v, (lo, hi) = r[m]['mean'], r[m]['ci95']
                mean = fmt(v, m)
                if np.isclose(v, best[s, m]):
                    mean = rf'\textbf{{{mean}}}'
                if args.seed_sd and r['n_seeds'] > 1:
                    mean += rf' {{\scriptsize$\pm${fmt(r[m]["seed_sd"], m)}}}'
                cells.append(cell(mean, f'({fmt(lo, m)}--{fmt(hi, m)})', args.inline))
        body.append(' & '.join(cells) + r' \\')
    return wrap(body, 'l' + 'c' * (len(head) - 1), args.caption, args.label)


def p_text(p: float) -> str:
    return r'$<$0.001' if p < 0.001 else f'{p:.3f}'


def compare(args) -> str:
    sets = load_summaries(args.summary)
    rows = pairs(args.rows, '--rows') if args.rows else \
        [(c, tex(c).replace(':', ' $-$ ')) for c in sets[0][1]['comparisons']]
    multi = len(sets) > 1
    head = ['Comparison'] + [c for _ in sets for m in args.metrics
                             for c in (r'$\Delta$' + HEADS.get(m, m).split('$')[0], '$p$')]
    body = []
    if multi:
        width = 2 * len(args.metrics)
        body.append(' & ' + ' & '.join(rf'\multicolumn{{{width}}}{{c}}{{{tex(s)}}}'
                                       for s, _ in sets) + r' \\')
        body.append(' '.join(rf'\cmidrule(lr){{{2 + i * width}-{1 + (i + 1) * width}}}'
                             for i in range(len(sets))))
    body += [' & '.join(head) + r' \\', r'\midrule']
    for key, name in rows:
        cells = [name if args.rows is None else tex(name)]
        for _, summ in sets:
            c = summ['comparisons'].get(key)
            for m in args.metrics:
                if c is None:
                    cells += ['--', '--']
                    continue
                d, (lo, hi), p = c[m]['diff'], c[m]['ci95'], c[m]['p']
                cells += [cell(fmt(d, m, signed=True),
                               f'({fmt(lo, m, True)}, {fmt(hi, m, True)})', args.inline), p_text(p)]
        body.append(' & '.join(cells) + r' \\')
    return wrap(body, 'l' + 'rr' * len(args.metrics) * len(sets), args.caption, args.label)


def mean_curve(pattern: str, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean IoU per click with a 95% bootstrap band over images; seeds
    (the files a glob matches) are averaged per image first."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'No files match {pattern}')
    runs = [json.loads(Path(f).read_text(encoding='utf-8')) for f in files]
    if any('curves' not in r for r in runs):
        sys.exit(f'{pattern}: not click curves (box results have no curve)')
    k = max(r['max_clicks'] for r in runs)
    names = sorted(set.intersection(*(set(r['curves']) for r in runs)))

    def padded(c):
        return np.pad(np.asarray(c, float), (0, k - len(c)), mode='edge')
    per_image = np.array([np.mean([padded(r['curves'][n]) for r in runs], axis=0) for n in names])
    idx = rng.integers(len(names), size=(n_boot, len(names)))
    boots = per_image[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return per_image.mean(axis=0), lo, hi, len(names)


def curves(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = pairs(args.groups, '--groups')
    if len(groups) > len(COLOURS):
        sys.exit(f'At most {len(COLOURS)} models per figure; split the rest into another figure')
    panels = pairs(args.panels, '--panels')
    rng = np.random.default_rng(args.seed)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'pdf.fonttype': 42})
    legend_rows = -(-len(groups) // args.legend_cols)
    legend_h = 0.18 * legend_rows + 0.05   # inches below the panels
    fig, axes = plt.subplots(1, len(panels), figsize=(args.width, args.height + legend_h),
                             sharey=True, squeeze=False)
    for ax, (title, folder) in zip(axes[0], panels):
        k = 0
        for i, (name, stem) in enumerate(groups):
            pattern = str(Path(folder) / stem)
            if not glob.glob(pattern):
                print(f'  note: {title}: no files for {name} ({pattern})', file=sys.stderr)
                continue
            mean, lo, hi, _ = mean_curve(pattern, args.n_boot, rng)
            x = np.arange(1, len(mean) + 1)
            k = max(k, len(mean))
            if args.band:
                ax.fill_between(x, lo, hi, color=COLOURS[i], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=COLOURS[i], marker=MARKERS[i], markersize=3.5,
                    linewidth=1.2, label=name)
        ax.set_title(f'{title}', fontsize=8)
        ax.set_xlabel('Number of clicks')
        ax.set_xticks(range(1, k + 1))
        ax.grid(axis='y', color='#d9d9d6', linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel('Mean IoU')
    legend = {}   # every model once, in --groups order, whichever panel has it
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(lab, h)
    labels = [name for name, _ in groups if name in legend]
    handles = [legend[name] for name in labels]
    if not labels:
        sys.exit('No click curves found for any model; no figure written')
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), args.legend_cols),
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, legend_h / (args.height + legend_h), 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {'Author': AUTHOR, 'Creator': None, 'Producer': None} if args.out.endswith('.pdf') \
        else {'Author': AUTHOR, 'Software': None}
    fig.savefig(args.out, metadata=meta, dpi=300)
    print(f'  figure -> {args.out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('table', 'compare'):
        p = sub.add_parser(cmd)
        p.add_argument('--summary', action='append', required=True,
                       help='SET=summary JSON from paper_stats.py; repeat for column groups')
        p.add_argument('--metrics', nargs='+', required=True, choices=list(HEADS))
        p.add_argument('--rows', nargs='*', default=None,
                       help='KEY=Display name, in table order (KEY is a model label, '
                            'or A:B for compare); default: every row of the first summary')
        p.add_argument('--caption', default=None)
        p.add_argument('--label', default=None)
        p.add_argument('--out', default=None, help='write the LaTeX here (default: print)')
        p.add_argument('--inline', action='store_true',
                       help='interval beside the value instead of below it (wider)')
        if cmd == 'table':
            p.add_argument('--seed-sd', action='store_true', help='add the spread across seeds')
    p = sub.add_parser('curves')
    p.add_argument('--panels', nargs='+', required=True, help='TITLE=results directory')
    p.add_argument('--groups', nargs='+', required=True,
                   help='Display name=file glob inside each directory (seeds of one model)')
    p.add_argument('--out', required=True, help='.pdf for the paper, .png for a preview')
    p.add_argument('--band', action='store_true', help='shade the 95%% bootstrap band')
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=float, default=4.8, help='inches (LNCS text width is 4.8)')
    p.add_argument('--height', type=float, default=2.1, help='inches, without the legend')
    p.add_argument('--legend-cols', type=int, default=3)
    args = ap.parse_args()

    if args.cmd == 'curves':
        curves(args)
        return
    out = table(args) if args.cmd == 'table' else compare(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out, encoding='utf-8')
        print(f'  table -> {args.out}')
    else:
        print(out)


if __name__ == '__main__':
    main()
