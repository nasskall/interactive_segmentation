"""
Score IMIS-Net with the same simulated user as click_benchmark.py.

IMIS-Net (Cheng et al., "Interactive Medical Image Segmentation: A Benchmark
Dataset and Baseline", CVPR 2025) is run unmodified from a separate clone of
https://github.com/uni-medical/IMIS-Bench; none of its code is copied here.
Its bundled package is also called ``segment_anything``, so this script runs
on its own and never imports Meta's package or this repo's predictors: only
the robot clicker (isegm.inference.clicker) is shared, and the per-image
JSON it writes is the one tools/paper_stats.py reads.

Each click goes to the centre of the largest error; IMIS-Net gets every click
so far plus its previous low-resolution logits, as in the authors'
predictor_example.ipynb. With --box it instead gets one box per object, the
same boxes tools/box_benchmark.py gives every other model.

Setup (once):
    git clone https://github.com/uni-medical/IMIS-Bench.git ../external/IMIS-Bench
    git -C ../external/IMIS-Bench checkout f4519c0f8f5a6e62e1a2bb9dde52be63c4a4fb30
    # checkpoint (Apache-2.0, the authors' Hugging Face upload, revision c3d2edd):
    #   https://huggingface.co/1Junlong/IMIS-Net/resolve/c3d2eddcc07537bc67cb97003905bf8a4bd072f5/IMISNet-B.pth
    # its model.py loads the CLIP tokenizer from ./clip-vit-base-patch32, so
    # put the tokenizer files of openai/clip-vit-base-patch32 there
    pip install monai transformers

Run from repo root, e.g.:
    python tools/imis_benchmark.py --imis-root ../external/IMIS-Bench \\
        --checkpoint ../external/IMIS-Bench/ckpt/IMISNet-B.pth \\
        --data datasets/isic2018_protocol --eval-split test --per-image results/isic/base_imis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from isegm.inference.clicker import Clicker    # noqa: E402  (numpy + cv2 only)
from prompt_boxes import box_scores, gt_box     # noqa: E402  (numpy only)


def load_split(data_dir: Path, split: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Same files and order as click_benchmark.load_manifest."""
    out = []
    for it in json.loads((data_dir / 'manifest.json').read_text())['items']:
        if it['split'] != split:
            continue
        bgr = cv2.imread(str(data_dir / 'images' / f"{it['name']}.jpg"))
        gt = cv2.imread(str(data_dir / 'masks' / f"{it['name']}.png"), cv2.IMREAD_GRAYSCALE)
        out.append((it['name'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt > 127))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def build_predictor(imis_root: Path, checkpoint: Path, device: str):
    # Their loader calls torch.load without weights_only (and maps to CUDA), so
    # the network is built without a checkpoint and the weights are loaded
    # here, as plain tensors only.
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except Exception as e:
        sys.exit(f'{checkpoint} does not load with weights_only=True ({e}); not loading it')
    if not (imis_root / 'clip-vit-base-patch32').is_dir():
        sys.exit(f'Missing {imis_root / "clip-vit-base-patch32"} (CLIP tokenizer files, see --help)')
    os.chdir(imis_root)                     # model.py finds the tokenizer by relative path
    sys.path.insert(0, str(imis_root))
    from segment_anything import sam_model_registry        # IMIS-Bench's own package
    from segment_anything.predictor import IMISPredictor
    from model import IMISNet

    sam = sam_model_registry['vit_b'](Namespace(image_size=1024, sam_checkpoint=None))
    sam.load_state_dict(state)
    sam.to(device)
    net = IMISNet(sam, test_mode=True, category_weights=None).to(device)
    net.eval()
    return IMISPredictor(_DecoderCallAdapter(net))


class _DecoderCallAdapter:
    """At commit f4519c0 the published IMISPredictor calls
    ``forward_decoder(features, image_size, prompt)`` but IMISNet defines
    ``forward_decoder(image_embedding, prompt)``. This drops the extra
    argument and passes every other attribute through, so neither file is
    edited."""

    def __init__(self, net):
        self._net = net

    def forward_decoder(self, image_embedding, image_size, prompt):
        return self._net.forward_decoder(image_embedding, prompt)

    def __getattr__(self, name):
        return getattr(self._net, name)


def click_curve(predictor, image, gt, max_clicks: int, make_clicker=None) -> list[float]:
    predictor.set_image(image)
    robot = make_clicker(gt, image) if make_clicker else Clicker(gt_mask=gt.astype(np.int32))
    pred = np.zeros_like(gt)
    logits, coords, labels, ious = None, [], [], []
    for _ in range(max_clicks):
        robot.make_next_click(pred)
        c = robot.clicks_list[-1]
        coords.append([int(c.coords[1]), int(c.coords[0])])     # (x, y)
        labels.append(1 if c.is_positive else 0)
        masks, logits, _ = predictor.predict(
            point_coords=np.array(coords), point_labels=np.array(labels),
            mask_input=logits, text=None, multimask_output=False)
        pred = np.asarray(masks).reshape(gt.shape) > 0.5
        ious.append(iou(pred, gt))
    return ious


def box_prediction(predictor, image, box: list[int]) -> np.ndarray:
    predictor.set_image(image)
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=np.array(box),
                                    mask_input=None, text=None, multimask_output=False)
    return np.asarray(masks).reshape(image.shape[:2]) > 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--imis-root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--eval-split', default='test')
    ap.add_argument('--n-eval', type=int, default=None)
    ap.add_argument('--max-clicks', type=int, default=10)
    ap.add_argument('--per-image', type=Path, required=True)
    ap.add_argument('--box', action='store_true', help='one box prompt per object instead of clicks')
    ap.add_argument('--jitter', type=float, default=0.0, help='box jitter (see tools/prompt_boxes.py)')
    ap.add_argument('--seed', type=int, default=0, help='box jitter / RClicks seed')
    ap.add_argument('--clicker', choices=['robot', 'rclicks'], default='robot',
                    help='simulated user (see tools/rclicks_clicker.py)')
    ap.add_argument('--rclicks-band', default='0,1')
    ap.add_argument('--rclicks-python', type=Path, default=ROOT.parent / 'external' / 'rclicks-venv' / 'bin' / 'python')
    ap.add_argument('--rclicks-root', type=Path, default=ROOT.parent / 'external' / 'rclicks')
    ap.add_argument('--rclicks-checkpoint', type=Path,
                    default=ROOT.parent / 'external' / 'rclicks' / 'ckpt' / 'clickability_model.pth')
    args = ap.parse_args()
    make_clicker = None
    if args.clicker == 'rclicks':
        from rclicks_clicker import RClicker, RClicksClient
        client = RClicksClient(args.rclicks_python.resolve(), args.rclicks_root.resolve(),
                               args.rclicks_checkpoint.resolve())
        band = tuple(float(v) for v in args.rclicks_band.split(','))
        make_clicker = lambda gt, image: RClicker(gt.astype(np.int32), image, client, band, args.seed)  # noqa: E731
    # Paths are resolved before build_predictor changes directory.
    imis_root, ckpt = args.imis_root.resolve(), args.checkpoint.resolve()
    data, out = args.data.resolve(), args.per_image.resolve()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evals = load_split(data, args.eval_split)[:args.n_eval]
    predictor = build_predictor(imis_root, ckpt, device)
    print(f'IMIS-Net on {device}; {data.name} {args.eval_split}: {len(evals)} eval objects', flush=True)

    if args.box:
        t0, scores, boxes = time.time(), {}, {}
        with torch.no_grad():
            for name, image, gt in evals:
                boxes[name] = gt_box(gt, name, args.jitter, args.seed)
                scores[name] = box_scores(box_prediction(predictor, image, boxes[name]), gt)
        print(f'  box jitter {args.jitter}: IoU {np.mean([s["iou"] for s in scores.values()]):.3f}  '
              f'Dice {np.mean([s["dice"] for s in scores.values()]):.3f}  '
              f'{(time.time() - t0) / len(evals):.1f} s/image')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'model': 'imis', 'prompt': 'box', 'jitter': args.jitter, 'seed': args.seed,
            'adapter': None, 'checkpoint': str(ckpt), 'data': str(data),
            'eval_split': args.eval_split, 'boxes': boxes, 'scores': scores}, indent=1))
        print(f'  per-image scores -> {out}')
        return

    t0, curves = time.time(), {}
    with torch.no_grad():
        for n, (name, image, gt) in enumerate(evals, 1):
            curves[name] = click_curve(predictor, image, gt, args.max_clicks, make_clicker)
            if n % 50 == 0 or n == len(evals):
                c = np.array(list(curves.values()))
                noc90 = np.mean([np.argmax(r >= 0.9) + 1 if (r >= 0.9).any() else args.max_clicks
                                 for r in c])
                print(f'  {n}/{len(evals)}  mIoU@1 {c[:, 0].mean():.3f}  NoC@90 {noc90:.2f}  '
                      f'{(time.time() - t0) / n:.1f} s/image', flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'model': 'imis', 'stage': 'baseline', 'adapter': None,
        'checkpoint': str(ckpt), 'data': str(data), 'eval_split': args.eval_split,
        'seed': args.seed, 'max_clicks': args.max_clicks, 'clicker': args.clicker,
        'rclicks_band': args.rclicks_band if args.clicker == 'rclicks' else None,
        'curves': curves}, indent=1))
    print(f'  per-image curves -> {out}')


if __name__ == '__main__':
    main()
