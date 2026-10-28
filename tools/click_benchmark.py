"""
Does online adaptation from user clicks help on real images?

A simulated user (the standard RITM "robot" clicker: each click goes to the
centre of the largest error region, positive in missed object, negative in
false background) labels objects from COCO128-seg:

  1. Baseline   -- score the model on held-out eval objects.
  2. Session    -- with recording on, label the training objects until
                   IoU >= --accept-iou (or --max-clicks), then finish each
                   one, filling the replay buffer exactly as the desktop app.
  3. Adapt      -- press "Adapt now" --presses times.
  4. Adapted    -- score the eval objects again.
  5. Rollback   -- undo every press and score once more (must match 1).

Scores: mean IoU after 1/3/5 clicks and NoC@85/90 (clicks needed to reach
that IoU; --max-clicks when never reached). Lower NoC is better.

With --model ensemble the members (--members) are fused instead, and every
--fusion method is scored with and without test-time augmentation. Add
--adapt to run steps 1-5 on the ensemble instead (one --fusion, default
STAPLE, no TTA):
each member is adapted on the recorded clicks with its own defaults.

Run from repo root, e.g.:
    python tools/click_benchmark.py --model ritm --category 0
    python tools/click_benchmark.py --model ensemble --members ritm,sam2
Data: datasets/coco128-seg (Ultralytics COCO128-seg, YOLO polygon labels),
or a folder with a manifest.json and fixed session/eval split, such as the
lesion set from tools/build_lesion_benchmark.py:
    python tools/click_benchmark.py --model sam2 --data datasets/lesion_bench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isegm.inference import utils as iu                         # noqa: E402
from isegm.inference.clicker import Clicker                     # noqa: E402
from isegm.inference.predictors.ensemble import (EnsembleConfig, EnsembleNet,  # noqa: E402
                                                 FUSIONS)
from interactive_demo.controller import InteractiveController   # noqa: E402

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
WEIGHTS = ROOT / 'model_weights'
CHECKPOINTS = {
    'ritm': ('best_checkpoint_068.pth', None),           # fine-tuned for skin lesions
    'ritm_coco': ('coco_lvis_h18_itermask.pth', None),   # general-purpose RITM (COCO + LVIS)
    'sam': ('sam_vit_b_01ec64.pth', 'SAM - ViT-B'),
    'sam2': ('sam2.1_hiera_tiny.pt', 'SAM2 - Tiny'),
    'simpleclick': ('cocolvis_vit_base.pth', None),
}


# ── data ──────────────────────────────────────────────────────────────────

def load_objects(data_dir: Path, category: int, min_area: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The largest instance of ``category`` in each image, if it covers at
    least ``min_area`` of the frame: (name, RGB image, bool mask)."""
    out = []
    for label in sorted((data_dir / 'labels' / 'train2017').glob('*.txt')):
        img_path = data_dir / 'images' / 'train2017' / f'{label.stem}.jpg'
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        best = None
        for line in label.read_text().splitlines():
            vals = line.split()
            if len(vals) < 7 or int(vals[0]) != category:
                continue
            pts = np.array(vals[1:], np.float64).reshape(-1, 2) * [w, h]
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            if best is None or mask.sum() > best.sum():
                best = mask
        if best is not None and best.sum() >= min_area * h * w:
            out.append((label.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), best.astype(bool)))
    return out


def load_manifest(data_dir: Path) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """A benchmark folder with a fixed split (images/, masks/, manifest.json):
    split name -> [(name, RGB image, bool mask)]."""
    splits: dict[str, list] = {}
    for it in json.loads((data_dir / 'manifest.json').read_text())['items']:
        bgr = cv2.imread(str(data_dir / 'images' / f"{it['name']}.jpg"))
        gt = cv2.imread(str(data_dir / 'masks' / f"{it['name']}.png"), cv2.IMREAD_GRAYSCALE)
        splits.setdefault(it['split'], []).append(
            (it['name'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt > 127))
    return splits


# ── simulated user ────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


CLICK_TIMES: list[float] = []   # seconds per click, image encoding excluded
LAST_CURVES: dict[str, list[float]] = {}   # object name -> IoU after each click, last evaluate()


def click_until(ctrl: InteractiveController, image, gt, max_clicks: int,
                stop_iou: float | None) -> list[float]:
    """Robot-click one object; IoU after each click."""
    ctrl.set_image(image)
    robot = Clicker(gt_mask=gt.astype(np.int32))
    pred = np.zeros_like(gt)
    ious = []
    for _ in range(max_clicks):
        robot.make_next_click(pred)
        c = robot.clicks_list[-1]
        t = time.perf_counter()
        ctrl.add_click(int(c.coords[1]), int(c.coords[0]), is_positive=bool(c.is_positive))
        CLICK_TIMES.append(time.perf_counter() - t)
        pred = ctrl.current_object_prob > ctrl.prob_thresh
        ious.append(iou(pred, gt))
        if stop_iou is not None and ious[-1] >= stop_iou:
            break
    return ious


def evaluate(ctrl, objects, max_clicks: int) -> dict:
    CLICK_TIMES.clear()
    LAST_CURVES.clear()
    curves = []
    for name, image, gt in objects:
        curves.append(click_until(ctrl, image, gt, max_clicks, None))
        LAST_CURVES[name] = curves[-1]
        ctrl.reset_last_object(update_image=False)
    curves = np.array(curves)

    def noc(t):
        hit = curves >= t
        return float(np.mean([np.argmax(r) + 1 if r.any() else max_clicks for r in hit]))
    return {'iou@1': curves[:, 0].mean(), 'iou@3': curves[:, 2].mean(),
            'iou@5': curves[:, 4].mean(), 'noc@85': noc(0.85), 'noc@90': noc(0.90),
            's/click': float(np.mean(CLICK_TIMES))}


def fmt(r: dict) -> str:
    return (f"mIoU@1 {r['iou@1']:.3f}  @3 {r['iou@3']:.3f}  @5 {r['iou@5']:.3f}  "
            f"NoC@85 {r['noc@85']:.2f}  NoC@90 {r['noc@90']:.2f}  {r['s/click']:.2f} s/click")


# ── main ──────────────────────────────────────────────────────────────────

def load_model(kind: str):
    ckpt, label = CHECKPOINTS[kind]
    ckpt = WEIGHTS / ckpt
    if kind in ('ritm', 'ritm_coco'):
        return iu.load_is_model(str(ckpt), device=DEVICE, cpu_dist_maps=True)
    if kind == 'sam':
        return iu.load_sam_model(str(ckpt), label, DEVICE)
    if kind == 'simpleclick':
        return iu.load_simpleclick_model(str(ckpt), DEVICE)
    return iu.load_sam2_model(str(ckpt), label, DEVICE)


def save_curves(args, name: str, stage: str):
    """The last evaluate()'s per-object IoU curves, with what produced them."""
    args.per_image.parent.mkdir(parents=True, exist_ok=True)
    args.per_image.write_text(json.dumps({
        'model': name, 'stage': stage, 'adapter': str(args.adapter) if args.adapter else None,
        'data': str(args.data), 'eval_split': args.eval_split, 'seed': args.seed,
        'max_clicks': args.max_clicks, 'curves': LAST_CURVES}, indent=1))
    print(f'  per-image curves -> {args.per_image}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', choices=[*CHECKPOINTS, 'ensemble'], default='ritm')
    ap.add_argument('--members', default='ritm,sam,sam2', help='ensemble members')
    ap.add_argument('--fusion', default=None,
                    help='ensemble fusion methods (default: all; with --adapt, staple)')
    ap.add_argument('--adapt', action='store_true',
                    help='ensemble: adapt the members instead of comparing fusions')
    ap.add_argument('--data', type=Path, default=ROOT / 'datasets' / 'coco128-seg')
    ap.add_argument('--category', type=int, default=0, help='COCO class index (0 = person)')
    ap.add_argument('--min-area', type=float, default=0.03)
    ap.add_argument('--n-train', type=int, default=None,
                    help='session objects (default 8, or the whole manifest split)')
    ap.add_argument('--n-eval', type=int, default=None,
                    help='eval objects (default 12, or the whole manifest split)')
    ap.add_argument('--max-clicks', type=int, default=10)
    ap.add_argument('--accept-iou', type=float, default=0.90)
    ap.add_argument('--presses', type=int, default=3, help='"Adapt now" presses')
    ap.add_argument('--sampled-clicks', action='store_true',
                    help='train on clicks sampled from the accepted masks instead of '
                         'the recorded ones (the earlier behaviour)')
    ap.add_argument('--sweep-lr', default=None,
                    help='comma-separated learning rates to compare, e.g. 5e-4,2e-4,1e-4; '
                         'replaces the --presses run')
    ap.add_argument('--sweep-steps', default='10,30,90',
                    help='total training steps at which each --sweep-lr is scored')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-split', default='eval',
                    help="manifest split to score on (e.g. 'reserve' for a final, untouched check)")
    ap.add_argument('--adapter', type=Path, default=None,
                    help='LoRA adapter to load first (e.g. a distilled student from tools/distill.py)')
    ap.add_argument('--no-adapt', action='store_true',
                    help='score the eval objects once and stop (no session, no adaptation)')
    ap.add_argument('--per-image', type=Path, default=None,
                    help='write every eval object\'s IoU curve to this JSON (tools/paper_stats.py reads it)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (args.data / 'manifest.json').exists():
        # Fixed split; --n-train/--n-eval only cap it, --category is ignored.
        splits = load_manifest(args.data)
        train = splits['session'][:args.n_train]
        evals = splits[args.eval_split][:args.n_eval]
        source = f'{args.data.name} {args.eval_split}'
    else:
        args.n_train = 8 if args.n_train is None else args.n_train
        args.n_eval = 12 if args.n_eval is None else args.n_eval
        objects = load_objects(args.data, args.category, args.min_area)
        if len(objects) < args.n_train + args.n_eval:
            sys.exit(f'Only {len(objects)} objects of class {args.category}; '
                     f'need {args.n_train + args.n_eval}')
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(objects))
        train = [objects[i] for i in order[:args.n_train]]
        evals = [objects[i] for i in order[args.n_train:args.n_train + args.n_eval]]
        source = f'class {args.category}'

    if args.model == 'ensemble':
        net = EnsembleNet([(m, load_model(m)) for m in args.members.split(',')])
    else:
        net = load_model(args.model)
    ctrl = InteractiveController(net, DEVICE, predictor_params={'brs_mode': 'NoBRS'},
                                 update_image_callback=lambda *a, **k: None)
    if args.adapter is not None:
        ctrl.load_adapter(str(args.adapter))
        print(f'loaded adapter {args.adapter}')

    if args.model == 'ensemble' and args.adapt:
        net.config.fusion, net.config.tta = args.fusion or EnsembleConfig.fusion, False
        if args.sweep_lr:
            sys.exit('--sweep-lr is for single models')
    elif args.model == 'ensemble':
        print(f'ensemble of {args.members} on {DEVICE}; {source}: '
              f'{len(evals)} eval objects')
        for fusion in (args.fusion or ','.join(FUSIONS)).split(','):
            for tta in (False, True):
                # The predictor reads this config object on every click.
                net.config.fusion, net.config.tta = fusion, tta
                ctrl.reset_predictor()
                r = evaluate(ctrl, evals, args.max_clicks)
                print(f'  {fusion:<13} tta={"on " if tta else "off"}  {fmt(r)}', flush=True)
        return

    name = (f'ensemble of {args.members} ({net.config.fusion})'
            if args.model == 'ensemble' else args.model)
    print(f'{name} on {DEVICE}; {source}: '
          f'{len(train)} session objects, {len(evals)} eval objects')
    t = time.time()
    base = evaluate(ctrl, evals, args.max_clicks)
    print(f'  baseline  {fmt(base)}   ({time.time() - t:.0f}s)')
    if args.no_adapt:
        if args.per_image:
            save_curves(args, name, 'baseline')
        return

    ctrl.online_recording = True
    session_clicks = []
    for _, image, gt in train:
        ious = click_until(ctrl, image, gt, args.max_clicks, args.accept_iou)
        session_clicks.append(len(ious))
        ctrl.finish_object()
    ctrl.online_recording = False
    print(f'  session   {len(ctrl.replay_buffer)} objects recorded, '
          f'{np.mean(session_clicks):.1f} clicks each')

    adapter = ctrl._ensure_online_adapter()
    for a in ([m for _, m in adapter.members] if args.model == 'ensemble' else [adapter]):
        a.config.use_recorded_clicks = not args.sampled_clicks
    print(f'  training on {"sampled" if args.sampled_clicks else "recorded"} clicks')

    if args.sweep_lr:
        # Same baseline and session for every setting: train in increments,
        # score at each total, then roll everything back.
        totals = [int(s) for s in args.sweep_steps.split(',')]
        for lr in (float(v) for v in args.sweep_lr.split(',')):
            adapter.config.lr = lr
            done, presses = 0, 0
            for total in totals:
                adapter.config.steps = total - done
                ctrl.adapt_now()
                done, presses = total, presses + 1
                r = evaluate(ctrl, evals, args.max_clicks)
                print(f'  lr {lr:<7g} {total:>3} steps  {fmt(r)}', flush=True)
            for _ in range(presses):
                ctrl.rollback_adapter()
        rolled = evaluate(ctrl, evals, args.max_clicks)
        same = all(abs(rolled[k] - base[k]) < 1e-6 for k in base if k != 's/click')
        print(f'  rollback restores the baseline: {"yes" if same else "NO"}')
        return

    t = time.time()
    for i in range(args.presses):
        rep = ctrl.adapt_now()
        print(f'  adapt #{i + 1}  {rep.get("successful_steps", 0)}/{rep["steps"]} steps, '
              f'loss {rep["mean_loss"]:.4f}' + (f'  failures: {rep["failures"][:2]}'
                                                 if rep.get('failures') else ''))
    print(f'            ({time.time() - t:.0f}s)')
    adapted = evaluate(ctrl, evals, args.max_clicks)
    print(f'  adapted   {fmt(adapted)}')
    if args.per_image:
        save_curves(args, name, 'adapted')

    for _ in range(args.presses):
        ctrl.rollback_adapter()
    rolled = evaluate(ctrl, evals, args.max_clicks)
    print(f'  rollback  {fmt(rolled)}')
    same = all(abs(rolled[k] - base[k]) < 1e-6 for k in base if k != 's/click')
    print(f'  rollback restores the baseline: {"yes" if same else "NO"}')


if __name__ == '__main__':
    main()
