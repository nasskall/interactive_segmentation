"""
Distil an interactive teacher (an adapted ensemble by default) into SAM 2.

Two stages, both resumable:

  targets  -- Build the teacher (``--members``, adapted on the lesion
              benchmark's session split exactly as click_benchmark.py does,
              unless --no-adapt), then let the simulated user label every
              image of the distillation pool with it: each click goes to the
              largest error of the teacher's current mask against the
              expert mask, and the teacher's probability map after each
              click is saved. Expert masks only place the clicks; they are
              never targets here.

  train    -- Train SAM 2 (LoRA on prompt encoder + mask decoder, frozen
              encoder with cached features) to reproduce those maps from
              the same clicks (``--target teacher``), the masks a labeller
              would have accepted from the teacher, with no expert check
              (``--target accepted``), or the expert masks (``--target gt``,
              the control). Every --eval-every steps the
              student is scored on the --select-split of --bench and its
              adapter saved; the best checkpoint by NoC@90 is reported.

Run from repo root, e.g.:
    python tools/build_lesion_benchmark.py --all-split train --images ... --masks ... \\
        --out datasets/lesion_distill
    python tools/distill.py targets --members simpleclick,sam2 --out experiments/distill/sc_sam2
    python tools/distill.py train --targets experiments/distill/sc_sam2 --target teacher
    python tools/distill.py train --targets experiments/distill/sc_sam2 --target gt

A killed train run continues from its latest step checkpoint with the same
arguments plus --resume.

Standard ISIC 2018 protocol (tools/build_isic_protocol.py): the teacher is
adapted on the session split and labels the train split, checkpoints are
picked on val, and only click_benchmark.py touches test:
    python tools/distill.py targets --pool datasets/isic2018_protocol --bench datasets/isic2018_protocol         --out experiments/isic/sc_sam2
    python tools/distill.py train --targets experiments/isic/sc_sam2 --target accepted --seed 1         --bench datasets/isic2018_protocol --select-split val --out experiments/isic/accepted_s1
    python tools/click_benchmark.py --model sam2 --adapter experiments/isic/accepted_s1/best.pt         --data datasets/isic2018_protocol --eval-split test --no-adapt --per-image results/accepted_s1.json
--n-records N trains on a fixed subset of N labelled images (the same
nested subsets for every seed), for label-efficiency curves.

Load a saved student in the desktop app (Load adapter) or score it with
    python tools/click_benchmark.py --model sam2 --adapter <run>/best.pt --data datasets/lesion_bench
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import click_benchmark as cb                                     # noqa: E402
from interactive_demo.controller import InteractiveController    # noqa: E402
from isegm.adaptation import attach_adapter                      # noqa: E402
from isegm.adaptation.distill import distill_step, sam2_features  # noqa: E402
from isegm.adaptation.lora import (collect_lora_state, load_lora_state,  # noqa: E402
                                   lora_parameters, set_lora_trainable)
from isegm.inference.clicker import Clicker                      # noqa: E402
from isegm.inference.predictors import get_predictor             # noqa: E402
from isegm.inference.predictors.ensemble import EnsembleNet      # noqa: E402

BENCH = ROOT / 'datasets' / 'lesion_bench'
TARGET_MAX_SIDE = 512     # teacher maps are stored at most this large


def pool_names(data_dir: Path, split: str) -> list[str]:
    return [it['name'] for it in json.loads((data_dir / 'manifest.json').read_text())['items']
            if it['split'] == split]


def load_item(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(RGB image, bool mask) of one pool image, read on demand so the
    whole pool never sits in memory."""
    bgr = cv2.imread(str(data_dir / 'images' / f'{name}.jpg'))
    gt = cv2.imread(str(data_dir / 'masks' / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt > 127


def make_ctrl(net) -> InteractiveController:
    return InteractiveController(net, cb.DEVICE, predictor_params={'brs_mode': 'NoBRS'},
                                 update_image_callback=lambda *a, **k: None)


# ── stage 1: teacher targets ──────────────────────────────────────────────

def build_teacher(members: list[str], adapt: bool, presses: int, out: Path, bench: Path):
    net = (cb.load_model(members[0]) if len(members) == 1
           else EnsembleNet([(m, cb.load_model(m)) for m in members]))
    ctrl = make_ctrl(net)
    saved = out / 'teacher_adapter.pt'
    if adapt and saved.exists():
        # Resuming: reuse the exact teacher that labelled the earlier images.
        ctrl.load_adapter(str(saved))
        print(f'  teacher adapter loaded from {saved}', flush=True)
    elif adapt:
        session = cb.load_manifest(bench)['session']
        ctrl.online_recording = True
        for _, image, gt in session:
            cb.click_until(ctrl, image, gt, 10, 0.90)
            ctrl.finish_object()
        ctrl.online_recording = False
        for i in range(presses):
            rep = ctrl.adapt_now()
            print(f'  teacher adapt #{i + 1}: loss {rep["mean_loss"]:.4f}', flush=True)
        torch.save(ctrl._online_adapter.state_dict(), saved)
    return ctrl


def label_with_teacher(ctrl, image, gt, max_clicks: int, stop_iou: float):
    """Simulated user driven by the teacher's own masks; returns clicks
    ``[(y, x, positive)]``, the teacher's maps after each click, and IoUs."""
    ctrl.set_image(image)
    robot = Clicker(gt_mask=gt.astype(np.int32))
    pred = np.zeros_like(gt)
    clicks, maps, ious = [], [], []
    for _ in range(max_clicks):
        robot.make_next_click(pred)
        c = robot.clicks_list[-1]
        y, x = int(c.coords[0]), int(c.coords[1])
        ctrl.add_click(x, y, is_positive=bool(c.is_positive))
        prob = np.asarray(ctrl.current_object_prob, np.float32)
        pred = prob > ctrl.prob_thresh
        clicks.append((y, x, bool(c.is_positive)))
        maps.append(prob)
        ious.append(cb.iou(pred, gt))
        if ious[-1] >= stop_iou:
            break
    ctrl.reset_last_object(update_image=False)
    return clicks, maps, ious


def cmd_targets(args):
    out = args.out
    (out / 'records').mkdir(parents=True, exist_ok=True)
    members = args.members.split(',')
    (out / 'config.json').write_text(json.dumps(
        {'members': members, 'adapt': not args.no_adapt, 'presses': args.presses, 'bench': str(args.bench),
         'pool': str(args.pool), 'split': args.split, 'max_clicks': args.max_clicks, 'stop_iou': args.stop_iou},
        indent=1))
    pool = pool_names(args.pool, args.split)[:args.limit]
    todo = [n for n in pool if not (out / 'records' / f'{n}.npz').exists()]
    print(f'teacher {members} ({"adapted" if not args.no_adapt else "base"}); '
          f'{len(pool)} pool images, {len(todo)} to label', flush=True)
    if not todo:
        return
    ctrl = build_teacher(members, not args.no_adapt, args.presses, out, args.bench)
    t0, all_ious = time.time(), []
    for n, name in enumerate(todo, 1):
        image, gt = load_item(args.pool, name)
        clicks, maps, ious = label_with_teacher(ctrl, image, gt, args.max_clicks, args.stop_iou)
        h, w = gt.shape
        s = min(1.0, TARGET_MAX_SIDE / max(h, w))
        size = (round(w * s), round(h * s))
        small = np.stack([cv2.resize(m, size, interpolation=cv2.INTER_AREA) for m in maps])
        np.savez_compressed(out / 'records' / f'{name}.npz',
                            clicks=np.array(clicks, np.int32),
                            maps=np.round(small * 255).astype(np.uint8),
                            ious=np.array(ious, np.float32), hw=np.array([h, w]))
        all_ious.append(ious[-1])
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0
            print(f'  {n}/{len(todo)}  final IoU {np.mean(all_ious):.3f}  '
                  f'{el / n:.1f} s/image, {el / n * (len(todo) - n) / 60:.0f} min left',
                  flush=True)


# ── stage 2: train the student ────────────────────────────────────────────

def load_records(targets: Path, names: list[str]) -> list[dict]:
    """The clicks of every labelled pool image; targets are read per step
    (``record_targets``) so memory stays flat however large the pool."""
    recs = []
    for name in names:
        f = targets / 'records' / f'{name}.npz'
        if f.exists():
            clicks = [(int(y), int(x), bool(p)) for y, x, p in np.load(f)['clicks']]
            recs.append({'name': name, 'clicks': clicks, 'file': f})
    return recs


def record_targets(rec: dict, kind: str, pool_dir: Path) -> np.ndarray:
    """uint8 (0..255) target after each click: the teacher's maps; the
    accepted mask (the teacher's last map thresholded, i.e. what a labeller
    using the teacher would have saved, with no expert check); or the expert
    mask. The last two are one mask broadcast over the clicks."""
    maps = np.load(rec['file'])['maps']
    if kind == 'teacher':
        return maps
    if kind == 'accepted':
        m = np.where(maps[-1] > 127, 255, 0).astype(np.uint8)
        return np.broadcast_to(m[None], maps.shape)
    hh, ww = maps.shape[1:]
    _, gt = load_item(pool_dir, rec['name'])
    m = cv2.resize(gt.astype(np.uint8) * 255, (ww, hh), interpolation=cv2.INTER_NEAREST)
    return np.broadcast_to(m[None], maps.shape)


def cached_features(predictor, rec: dict, cache: Path, pool_dir: Path) -> dict:
    f = cache / f"{rec['name']}.pt"
    if f.exists():
        return torch.load(f, weights_only=True)
    feats = sam2_features(predictor, load_item(pool_dir, rec['name'])[0])
    store = {'image_embed': feats['image_embed'].half().cpu(),
             'high_res_feats': [t.half().cpu() for t in feats['high_res_feats']],
             'hw': feats['hw']}
    torch.save(store, f)
    return store


def evaluate_student(net, evals) -> dict:
    set_lora_trainable(net, False)
    net.eval()
    # Plain floats, so saved adapters load with torch.load(weights_only=True).
    r = {k: float(v) for k, v in cb.evaluate(make_ctrl(net), evals, 10).items()}
    set_lora_trainable(net, True)
    return r


def resume(ckpt: Path, net, opt, sched, rng, recs, args, out: Path) -> tuple[int, dict]:
    """Restore a train run from its latest step checkpoint. Returns (step, best).
    Checkpoints from before --resume carry no optimizer or RNG state: AdamW
    then restarts its moments, and the sampler replays its draws so the run
    sees the same records it would have seen uninterrupted."""
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    step = payload['distill']['step']
    if payload['distill']['target'] != args.target:
        raise SystemExit(f"{ckpt} was trained on --target {payload['distill']['target']}")
    load_lora_state(net, payload['lora_state'])
    state = payload.get('resume')
    if state:
        opt.load_state_dict(state['opt'])
        rng.bit_generator.state = state['rng']
    else:
        print('  no optimizer state in checkpoint: AdamW restarts its moments', flush=True)
        for _ in range(step * args.batch):
            rec = recs[rng.integers(len(recs))]
            rng.integers(1, len(rec['clicks']) + 1)
    for _ in range(step):
        sched.step()
    best = {}
    for line in (out / 'log.jsonl').read_text().splitlines():
        r = json.loads(line)
        if best.get('noc@90', 99) > r['noc@90'] or (
                best.get('noc@90') == r['noc@90'] and r['iou@1'] > best['iou@1']):
            best = r
    print(f'resumed from {ckpt.name} (step {step}); best so far: step {best["step"]}  '
          f'{cb.fmt(best)}', flush=True)
    return step, best


def cmd_train(args):
    out = args.out or args.targets / f'student_{args.target}'
    out.mkdir(parents=True, exist_ok=True)
    cache = args.targets / 'feats_sam2'
    cache.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = json.loads((args.targets / 'config.json').read_text())
    pool_dir = Path(cfg['pool'])
    recs = load_records(args.targets, pool_names(pool_dir, cfg.get('split', 'train')))
    if args.n_records is not None:
        # One fixed order (not --seed), so smaller subsets nest inside larger ones.
        order = np.random.default_rng(0).permutation(len(recs))
        recs = [recs[i] for i in sorted(order[:args.n_records])]
    evals = cb.load_manifest(args.bench)[args.select_split][:args.n_eval]
    print(f'student SAM 2 on {len(recs)} records, target={args.target}, '
          f'{args.steps} steps x batch {args.batch}, lr {args.lr}, rank {args.rank}', flush=True)

    net = cb.load_model('sam2')
    predictor = get_predictor(net, device=cb.DEVICE)
    for i, rec in enumerate(recs):               # fill the feature cache once
        if not (cache / f"{rec['name']}.pt").exists():
            cached_features(predictor, rec, cache, pool_dir)
        if (i + 1) % 100 == 0:
            print(f'  cached features {i + 1}/{len(recs)}', flush=True)

    _, wrapped = attach_adapter(net, 'sam2', rank=args.rank)
    set_lora_trainable(net, True)
    params = lora_parameters(net)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, args.steps)
                                                                    / args.steps)))
    log = open(out / 'log.jsonl', 'a')

    def checkpoint(step: int, best: dict) -> dict:
        r = evaluate_student(net, evals)
        payload = {'model_type': 'sam2', 'rank': args.rank, 'wrapped_modules': wrapped,
                   'lora_state': collect_lora_state(net),
                   'distill': {'step': step, 'target': args.target,
                               'teacher': str(args.targets), 'eval': r},
                   # Training state for --resume; ignored when the adapter is loaded.
                   'resume': {'opt': opt.state_dict(), 'rng': rng.bit_generator.state}}
        torch.save(payload, out / f'step{step:05d}.pt')
        log.write(json.dumps({'step': step, **r}) + '\n')
        log.flush()
        print(f'  step {step:5d}  {cb.fmt(r)}', flush=True)
        if best.get('noc@90', 99) > r['noc@90'] or (
                best.get('noc@90') == r['noc@90'] and r['iou@1'] > best['iou@1']):
            torch.save(payload, out / 'best.pt')
            return {**r, 'step': step}
        return best

    start = 0
    ckpts = sorted(out.glob('step*.pt'))
    if args.resume and ckpts:
        start, best = resume(ckpts[-1], net, opt, sched, rng, recs, args, out)
    else:
        best = checkpoint(0, {})      # LoRA starts at identity: the base model
    t0, losses = time.time(), []
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad()
        for _ in range(args.batch):
            rec = recs[rng.integers(len(recs))]
            k = int(rng.integers(1, len(rec['clicks']) + 1))
            feats = cached_features(predictor, rec, cache, pool_dir)
            loss = distill_step(net, feats, rec['clicks'],
                                record_targets(rec, args.target, pool_dir), k, cb.DEVICE)
            (loss / args.batch).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f'  step {step:5d}  loss {np.mean(losses[-50 * args.batch:]):.4f}  '
                  f'{(time.time() - t0) / (step - start):.2f} s/step', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            best = checkpoint(step, best)
    print(f'best: step {best["step"]}  {cb.fmt(best)}')
    (out / 'best.json').write_text(json.dumps(best, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('targets', help='label the pool with the teacher')
    t.add_argument('--members', default='simpleclick,sam2')
    t.add_argument('--no-adapt', action='store_true')
    t.add_argument('--presses', type=int, default=3)
    t.add_argument('--pool', type=Path, default=ROOT / 'datasets' / 'lesion_distill')
    t.add_argument('--split', default='train')
    t.add_argument('--max-clicks', type=int, default=5)
    t.add_argument('--stop-iou', type=float, default=0.95)
    t.add_argument('--limit', type=int, default=None, help='label only the first N images')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--bench', type=Path, default=BENCH, help='benchmark whose session split adapts the teacher')

    s = sub.add_parser('train', help='train the SAM 2 student')
    s.add_argument('--targets', type=Path, required=True)
    s.add_argument('--target', choices=['teacher', 'accepted', 'gt'], default='teacher')
    s.add_argument('--steps', type=int, default=2000)
    s.add_argument('--batch', type=int, default=4)
    s.add_argument('--lr', type=float, default=2e-4)
    s.add_argument('--rank', type=int, default=8)
    s.add_argument('--eval-every', type=int, default=500)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--n-eval', type=int, default=None, help='score on the first N eval images')
    s.add_argument('--bench', type=Path, default=BENCH, help='benchmark the checkpoints are scored on')
    s.add_argument('--n-records', type=int, default=None,
                   help='train on a fixed subset of this many labelled images')
    s.add_argument('--select-split', default='eval',
                   help='split of --bench that picks the best checkpoint (val for the standard protocol)')
    s.add_argument('--out', type=Path, default=None)
    s.add_argument('--resume', action='store_true',
                   help='continue from the latest step checkpoint in --out')

    args = ap.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    (cmd_targets if args.cmd == 'targets' else cmd_train)(args)


if __name__ == '__main__':
    main()
