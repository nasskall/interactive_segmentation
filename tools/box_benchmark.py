"""
Box-prompt benchmark: one box per object, one mask, IoU and Dice.

The click benchmark (click_benchmark.py) cannot include MedSAM, whose public
checkpoint is trained for box prompts only, so every box-capable model is
also scored here on the same boxes (tools/prompt_boxes.py): a tight box
around the expert mask, or with --jitter each side moved by up to that
fraction of the box size (seeded by image name and --seed).

  sam, sam2   through this repo's predictors (--adapter for SAM 2 students)
  medsam      MedSAM (Ma et al., Nat. Commun. 15:654, 2024) with the
              preprocessing of the authors' MedSAM_Inference.py
IMIS-Net is scored on the same boxes by imis_benchmark.py --box.

Run from repo root, e.g.:
    python tools/box_benchmark.py --model medsam --checkpoint ../external/MedSAM/work_dir/MedSAM/medsam_vit_b.pth \\
        --data datasets/isic2018_protocol --eval-split test --per-image results/isic/box_medsam.json
    python tools/box_benchmark.py --model sam2 --adapter experiments/isic/accepted_s1/best.pt \\
        --data datasets/isic2018_protocol --eval-split test --jitter 0.1 --per-image results/isic/box10_accepted_s1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import click_benchmark as cb                       # noqa: E402
from prompt_boxes import box_scores, gt_box        # noqa: E402


# ── MedSAM ────────────────────────────────────────────────────────────────
# Preprocessing and decoding follow MedSAM_Inference.py from
# https://github.com/bowang-lab/MedSAM (commit d71e8a1), Copyright the MedSAM
# authors, Apache License 2.0; see THIRD_PARTY_NOTICES.md.

class MedSAM:
    def __init__(self, checkpoint: Path, device: str):
        from segment_anything import sam_model_registry
        # The official script passes checkpoint= (an unrestricted torch.load);
        # the weights are plain tensors, so load them here with weights_only.
        self.model = sam_model_registry['vit_b'](checkpoint=None)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image: np.ndarray, box: list[int]) -> np.ndarray:
        from skimage import transform
        H, W, _ = image.shape
        img = transform.resize(image, (1024, 1024), order=3, preserve_range=True,
                               anti_aliasing=True).astype(np.uint8)
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        tensor = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        embed = self.model.image_encoder(tensor)
        box_1024 = np.array([box]) / np.array([W, H, W, H]) * 1024
        box_t = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)[:, None, :]
        sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t, masks=None)
        logits, _ = self.model.mask_decoder(
            image_embeddings=embed, image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False)
        prob = F.interpolate(torch.sigmoid(logits), size=(H, W), mode='bilinear', align_corners=False)
        return prob.squeeze().cpu().numpy() > 0.5


# ── this repo's SAM / SAM 2 ──────────────────────────────────────────────

class RepoModel:
    def __init__(self, kind: str, adapter: Path | None):
        self.ctrl = cb.InteractiveController(cb.load_model(kind), cb.DEVICE,
                                             predictor_params={'brs_mode': 'NoBRS'},
                                             update_image_callback=lambda *a, **k: None)
        if adapter is not None:
            self.ctrl.load_adapter(str(adapter))

    def predict(self, image: np.ndarray, box: list[int]) -> np.ndarray:
        self.ctrl.set_image(image)
        self.ctrl.set_box(*box)
        pred = self.ctrl.current_object_prob > self.ctrl.prob_thresh
        self.ctrl.reset_last_object(update_image=False)
        return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model', choices=['sam', 'sam2', 'medsam'], required=True)
    ap.add_argument('--checkpoint', type=Path, default=None, help='MedSAM checkpoint (medsam_vit_b.pth)')
    ap.add_argument('--adapter', type=Path, default=None, help='LoRA adapter for sam2 (a student)')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--eval-split', default='test')
    ap.add_argument('--n-eval', type=int, default=None)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per-image', type=Path, required=True)
    args = ap.parse_args()

    if args.model == 'medsam':
        if args.checkpoint is None:
            sys.exit('--model medsam needs --checkpoint')
        model = MedSAM(args.checkpoint, cb.DEVICE)
    else:
        model = RepoModel(args.model, args.adapter)
    evals = cb.load_manifest(args.data)[args.eval_split][:args.n_eval]
    print(f'{args.model} on {cb.DEVICE}, box jitter {args.jitter}; '
          f'{args.data.name} {args.eval_split}: {len(evals)} objects', flush=True)

    t0, scores, boxes = time.time(), {}, {}
    for n, (name, image, gt) in enumerate(evals, 1):
        boxes[name] = gt_box(gt, name, args.jitter, args.seed)
        scores[name] = box_scores(model.predict(image, boxes[name]), gt)
        if n % 100 == 0 or n == len(evals):
            print(f'  {n}/{len(evals)}  IoU {np.mean([s["iou"] for s in scores.values()]):.3f}  '
                  f'Dice {np.mean([s["dice"] for s in scores.values()]):.3f}  '
                  f'{(time.time() - t0) / n:.2f} s/image', flush=True)

    args.per_image.parent.mkdir(parents=True, exist_ok=True)
    args.per_image.write_text(json.dumps({
        'model': args.model, 'prompt': 'box', 'jitter': args.jitter, 'seed': args.seed,
        'adapter': str(args.adapter) if args.adapter else None,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'data': str(args.data), 'eval_split': args.eval_split,
        'boxes': boxes, 'scores': scores}, indent=1))
    print(f'  per-image scores -> {args.per_image}')


if __name__ == '__main__':
    main()
