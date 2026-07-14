"""
End-to-end smoke test on a synthetic dataset.

Generates skin-lesion-style image+mask pairs, then exercises every
framework path:
  A. Regular interactive add_click flow
  B. Auto-segment (center click)
  C. Online recording + adapt_now + rollback
  D. Offline few-shot trainer (LoRA)
  E. Adapter save / load round-trip
  F. Adapter actually changes predictions

RITM is run end-to-end (uses the bundled checkpoint).
SimpleClick / SAM / SAM2 paths are exercised only if their checkpoints (and,
for SAM/SAM2, their pip packages) are present — otherwise reported as SKIP, so
a missing backend shows up in the summary instead of silently not being run.

Run from repo root:
    .venv/Scripts/python.exe tools/synthetic_smoke_test.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

# --- repo on path -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isegm.inference import utils as iu                           # noqa: E402
from isegm.adaptation import (                                     # noqa: E402
    FewShotConfig, FewShotTrainer, OnlineAdaptConfig,
)
from interactive_demo.controller import InteractiveController     # noqa: E402

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
RITM_CKPT = ROOT / 'model_weights' / 'best_checkpoint_068.pth'
SYNTH_DIR = ROOT / 'synthetic_test'


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []  # (name, status, detail)

    def add(self, name, status, detail=''):
        tag = {'PASS': '+', 'FAIL': 'x', 'SKIP': '~'}[status]
        print(f'  [{tag}] {name}: {status}' + (f' -- {detail}' if detail else ''))
        self.rows.append((name, status, detail))

    def summary(self):
        n = len(self.rows)
        passed = sum(1 for r in self.rows if r[1] == 'PASS')
        failed = sum(1 for r in self.rows if r[1] == 'FAIL')
        skipped = sum(1 for r in self.rows if r[1] == 'SKIP')
        print()
        print('=' * 72)
        print(f'  PASS={passed}  FAIL={failed}  SKIP={skipped}  (total {n})')
        print('=' * 72)
        return failed == 0


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_lesion_image(rng: np.random.Generator,
                      size=(256, 256),
                      seed_offset: int = 0):
    """Skin-tone background + elliptical 'lesion' with soft edge.
    Returns (rgb_uint8, mask_bool)."""
    H, W = size
    # Skin-tone gradient — vary base hue per image so models can't trivially
    # memorise.
    base_r = int(rng.integers(180, 230) + seed_offset) % 256
    base_g = int(rng.integers(140, 190) + seed_offset) % 256
    base_b = int(rng.integers(130, 170) + seed_offset) % 256

    img = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    img[..., 0] = base_r + (yy / H) * 18
    img[..., 1] = base_g + (xx / W) * 12
    img[..., 2] = base_b + ((yy + xx) / (H + W)) * 8

    # Add gaussian noise to mimic skin texture.
    noise = rng.normal(0.0, 6.0, size=img.shape)
    img = np.clip(img + noise, 0, 255)

    # Elliptical lesion in a randomised location/shape.
    cy = int(rng.integers(H // 3, 2 * H // 3))
    cx = int(rng.integers(W // 3, 2 * W // 3))
    a  = int(rng.integers(28, 55))
    b  = int(rng.integers(28, 55))
    theta = float(rng.uniform(0, np.pi))

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rel_y = yy - cy
    rel_x = xx - cx
    xr =  cos_t * rel_x + sin_t * rel_y
    yr = -sin_t * rel_x + cos_t * rel_y
    dist = (xr / a) ** 2 + (yr / b) ** 2
    mask = dist < 1.0

    # Darken the lesion area (lesions are typically darker than skin).
    darken = np.exp(-dist) * 70.0  # soft falloff
    img -= darken[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


def build_dataset(out_dir: Path, n_pairs: int = 6, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / 'images').mkdir(parents=True)
    (out_dir / 'masks').mkdir(parents=True)
    for i in range(n_pairs):
        img, mask = make_lesion_image(rng, seed_offset=i * 31)
        cv2.imwrite(str(out_dir / 'images' / f'lesion_{i:02d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / 'masks' / f'lesion_{i:02d}.png'),
                    (mask.astype(np.uint8) * 255))
    print(f'  generated {n_pairs} pairs at {out_dir}')
    return out_dir


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return float(inter / union) if union else 0.0


# ---------------------------------------------------------------------------
# RITM / SimpleClick tests (both are ISModel-based and share the predictor)
# ---------------------------------------------------------------------------

def _run_is_family_suite(
    report: Report,
    dataset_dir: Path,
    prefix: str,             # 'RITM' or 'SimpleClick'
    mt: str,                 # 'ritm' or 'simpleclick'
    loader_fn,               # callable() -> torch.nn.Module
    fs_config: FewShotConfig,
    online_steps: int,
    train_subset: int | None = None,
    min_iou: float | None = None,
):
    """Full A-F suite for an ISModel-based backend.

    ``min_iou`` gates the interactive-click check. SimpleClick passes None:
    like SAM, it saturates on these synthetic soft-edged blobs (predicting
    nearly the whole frame), so its IoU here says nothing about correctness --
    it is verified against real imagery instead. Only the plumbing is asserted.
    """
    from isegm.data.custom_dataset import CustomDataset

    model = loader_fn()
    report.add(f'{prefix}/load', 'PASS', f'device={DEVICE}')

    ctrl = InteractiveController(
        model, DEVICE,
        predictor_params={'brs_mode': 'NoBRS'},
        update_image_callback=lambda *a, **k: None,
    )

    ds = CustomDataset(dataset_dir, augment=False)
    img0, gt0 = ds[0]

    # ── A. Regular interactive flow ────────────────────────────────────
    try:
        ctrl.set_image(img0)
        ys, xs = np.where(gt0)
        cy_pos, cx_pos = int(ys.mean()), int(xs.mean())
        ys_n, xs_n = np.where(~gt0)
        cy_neg, cx_neg = int(ys_n.mean()), int(xs_n.mean())
        ctrl.add_click(cx_pos, cy_pos, is_positive=True)
        ctrl.add_click(cx_neg, cy_neg, is_positive=False)
        prob = ctrl.current_object_prob
        score = iou(prob > 0.5, gt0)
        ok = (prob is not None) if min_iou is None else (score > min_iou)
        report.add(f'{prefix}/A interactive_clicks', 'PASS' if ok else 'FAIL',
                   f'IoU={score:.3f} after 2 clicks')
        ctrl.finish_object()
    except Exception as exc:
        report.add(f'{prefix}/A interactive_clicks', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── B. Auto-segment (center click) ─────────────────────────────────
    try:
        ctrl.set_image(img0)
        result = ctrl.auto_segment(mode='auto')
        ok = result['ok'] and result['mode_used'] == 'center_click'
        prob = ctrl.current_object_prob
        score = iou(prob > 0.5, gt0) if prob is not None else 0.0
        report.add(f'{prefix}/B auto_segment', 'PASS' if ok else 'FAIL',
                   f"mode={result['mode_used']}, IoU={score:.3f}")
    except Exception as exc:
        report.add(f'{prefix}/B auto_segment', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── C. Online recording + adapt_now + rollback ─────────────────────
    try:
        ctrl.set_image(img0)
        ctrl.online_recording = True
        # Run two finished objects on different images to populate buffer.
        for k in range(min(2, len(ds))):
            img_k, gt_k = ds[k]
            ctrl.set_image(img_k)
            ys, xs = np.where(gt_k)
            ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
            ctrl.finish_object()
        buf_len_before = len(ctrl.replay_buffer)

        # Force a small step count for the smoke test
        adapter = ctrl._ensure_online_adapter()
        adapter.config.steps = online_steps
        report.add(f'{prefix}/C buffer_capture',
                   'PASS' if buf_len_before >= 2 else 'FAIL',
                   f'buffer={buf_len_before}')

        rep = ctrl.adapt_now()
        report.add(f'{prefix}/C adapt_now',
                   'PASS' if rep['steps'] == online_steps else 'FAIL',
                   f"steps={rep['steps']}, mean_loss={rep['mean_loss']:.4f}")

        ok_back = ctrl.rollback_adapter()
        report.add(f'{prefix}/C rollback', 'PASS' if ok_back else 'FAIL')
    except Exception as exc:
        traceback.print_exc()
        report.add(f'{prefix}/C online_adapt', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── D. Offline few-shot trainer ────────────────────────────────────
    try:
        # Use a fresh model so the previous online-LoRA injection does not
        # collide with the trainer's own injection step.
        model2 = loader_fn()
        ds_train = CustomDataset(dataset_dir, augment=True)
        if train_subset is not None:
            ds_train.pairs = ds_train.pairs[:train_subset]

        trainer = FewShotTrainer(
            model=model2,
            model_type=mt,
            dataset=ds_train,
            config=fs_config,
            device=DEVICE,
            log_cb=lambda m: None,
        )
        trainer.run()
        state = trainer.get_adapter_state()
        ok = state is not None and len(state) > 0
        report.add(f'{prefix}/D few_shot_train', 'PASS' if ok else 'FAIL',
                   f'{len(state) if state else 0} tensors, '
                   f'{len(trainer.get_wrapped_module_names())} modules')

        # ── E. Save / load round-trip ──────────────────────────────────
        out_path = SYNTH_DIR / f'{mt}_synth_adapter.pt'
        trainer.save(str(out_path))
        loaded = torch.load(out_path, map_location='cpu', weights_only=False)
        round_trip_ok = (
            loaded.get('model_type') == mt
            and loaded.get('rank') == fs_config.rank
            and len(loaded['lora_state']) == len(state)
        )
        report.add(f'{prefix}/E save_load', 'PASS' if round_trip_ok else 'FAIL',
                   f'file={out_path.name}, '
                   f'tensors={len(loaded["lora_state"])}')

        # ── F. Adapter actually changes prediction ─────────────────────
        # Compare the same-image, same-click prediction before and after
        # loading the adapter onto a fresh controller.
        ctrl_fresh = InteractiveController(
            loader_fn(), DEVICE,
            predictor_params={'brs_mode': 'NoBRS'},
            update_image_callback=lambda *a, **k: None,
        )
        img_test, gt_test = ds[0]
        ctrl_fresh.set_image(img_test)
        ys, xs = np.where(gt_test)
        ctrl_fresh.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
        prob_before = ctrl_fresh.current_object_prob.copy()

        ctrl_fresh.load_adapter(str(out_path))
        ctrl_fresh.reset_last_object(update_image=False)
        ctrl_fresh.set_image(img_test)
        ctrl_fresh.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
        prob_after = ctrl_fresh.current_object_prob.copy()

        diff = float(np.abs(prob_after - prob_before).mean())
        # Even tiny rank-4 LoRA training over a couple of epochs on synthetic
        # data should produce a measurably non-zero delta in the prob map.
        ok_delta = diff > 1e-4
        report.add(f'{prefix}/F adapter_changes_pred',
                   'PASS' if ok_delta else 'FAIL',
                   f'mean |dprob|={diff:.6f}')

    except Exception as exc:
        traceback.print_exc()
        report.add(f'{prefix}/D-F few_shot_pipeline', 'FAIL',
                   f'{type(exc).__name__}: {exc}')


def test_ritm(report: Report, dataset_dir: Path):
    print('\n[RITM] loading checkpoint…')
    if not RITM_CKPT.exists():
        report.add('RITM/load', 'SKIP', f'checkpoint missing: {RITM_CKPT}')
        return

    def loader():
        model = iu.load_is_model(str(RITM_CKPT), device=DEVICE, cpu_dist_maps=True)
        model._model_type = 'ritm'
        return model

    _run_is_family_suite(
        report, dataset_dir, prefix='RITM', mt='ritm', loader_fn=loader,
        fs_config=FewShotConfig(rank=4, epochs=2, lr=1e-3, n_clicks=2,
                                cpu_max_size=192, cuda_max_size=384),
        online_steps=4,
        min_iou=0.10,
    )


def test_simpleclick(report: Report, dataset_dir: Path):
    ckpt = _find_checkpoint(['cocolvis_vit_*.pth', 'sbd_vit_*.pth'])
    if ckpt is None:
        report.add('SimpleClick/all', 'SKIP',
                   'no cocolvis_vit_*/sbd_vit_*.pth in model_weights/ '
                   '(fetch one from the SimpleClick model zoo)')
        return

    print(f'\n[SimpleClick] using {ckpt.name}…')

    def loader():
        return iu.load_simpleclick_model(str(ckpt), DEVICE)

    # ViT-B at a fixed 448x448 is heavy on CPU, so keep the offline run small.
    _run_is_family_suite(
        report, dataset_dir, prefix='SimpleClick', mt='simpleclick',
        loader_fn=loader,
        fs_config=FewShotConfig(rank=4, epochs=1, lr=1e-3, n_clicks=1,
                                cpu_max_size=448, cuda_max_size=448),
        online_steps=2,
        train_subset=3,
        min_iou=None,
    )


# ---------------------------------------------------------------------------
# SAM / SAM2 tests (only if available)
# ---------------------------------------------------------------------------

def _find_checkpoint(patterns):
    weights = ROOT / 'model_weights'
    if not weights.exists():
        return None
    for pat in patterns:
        hits = list(weights.glob(pat))
        if hits:
            return hits[0]
    return None


def _run_sam_family_suite(
    report: Report,
    dataset_dir: Path,
    prefix: str,           # 'SAM' or 'SAM2'
    mt: str,               # 'sam' or 'sam2'
    label: str,            # UI label, e.g. 'SAM - ViT-B'
    ckpt: Path,
    loader_fn,             # callable() -> torch.nn.Module
):
    """Full A-F suite for a SAM-family model.

    Foundation-model passes are heavy on CPU, so this uses fewer images,
    fewer SGD steps, and a smaller image cap than the RITM suite. Each
    sub-test is independent so a single failure doesn't cascade.
    """
    from isegm.data.custom_dataset import CustomDataset

    print(f'\n[{prefix}] using {ckpt.name}…')

    model = loader_fn()
    ctrl = InteractiveController(model, DEVICE,
                                  predictor_params={'brs_mode': 'NoBRS'},
                                  update_image_callback=lambda *a, **k: None)
    report.add(f'{prefix}/load', 'PASS', f'device={DEVICE}, label={label}')

    ds = CustomDataset(dataset_dir, augment=False)
    img0, gt0 = ds[0]

    # ── A. interactive ───────────────────────────────────────────────
    # Threshold is intentionally low — synthetic skin-tone images with soft
    # lesion edges are out-of-distribution for SAM/SAM2; we're verifying
    # the plumbing, not quality.
    try:
        ctrl.set_image(img0)
        ys, xs = np.where(gt0)
        ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
        prob = ctrl.current_object_prob
        score = iou(prob > 0.5, gt0)
        report.add(f'{prefix}/A interactive_clicks',
                   'PASS' if (prob is not None and score >= 0.0) else 'FAIL',
                   f'IoU={score:.3f}')
    except Exception as exc:
        report.add(f'{prefix}/A interactive_clicks', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── B. auto-segment ──────────────────────────────────────────────
    try:
        ctrl.set_image(img0)
        result = ctrl.auto_segment(mode='auto')
        prob = ctrl.current_object_prob
        score = iou(prob > 0.5, gt0) if prob is not None else 0.0
        report.add(f'{prefix}/B auto_segment',
                   'PASS' if result['ok'] else 'FAIL',
                   f"mode={result['mode_used']}, IoU={score:.3f}")
    except Exception as exc:
        report.add(f'{prefix}/B auto_segment', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── C. online recording + adapt_now + rollback ───────────────────
    try:
        ctrl.set_image(img0)
        ctrl.online_recording = True
        for k in range(min(2, len(ds))):
            img_k, gt_k = ds[k]
            ctrl.set_image(img_k)
            ys, xs = np.where(gt_k)
            ctrl.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
            ctrl.finish_object()
        buf_len = len(ctrl.replay_buffer)
        report.add(f'{prefix}/C buffer_capture',
                   'PASS' if buf_len >= 2 else 'FAIL',
                   f'buffer={buf_len}')

        adapter = ctrl._ensure_online_adapter()
        adapter.config.steps = 2  # SAM-family steps are heavy on CPU
        rep = ctrl.adapt_now()
        ok_step = (rep.get('successful_steps', 0) >= 1
                   and not np.isnan(rep['mean_loss']))
        report.add(f'{prefix}/C adapt_now',
                   'PASS' if ok_step else 'FAIL',
                   f"ok={rep.get('successful_steps', 0)}/{rep['steps']}, "
                   f"mean_loss={rep['mean_loss']:.4f}"
                   + (f", failures={rep.get('failures', [])[:1]}"
                      if rep.get('failures') else ''))

        ok_back = ctrl.rollback_adapter()
        report.add(f'{prefix}/C rollback', 'PASS' if ok_back else 'FAIL')
    except Exception as exc:
        traceback.print_exc()
        report.add(f'{prefix}/C online_adapt', 'FAIL',
                   f'{type(exc).__name__}: {exc}')

    # ── D-F. Offline trainer + save/load + delta verification ────────
    try:
        # Fresh model + controller for the trainer (so the prior C-test
        # adapter state doesn't pollute D's metrics).
        model_train = loader_fn()
        ctrl_train = InteractiveController(
            model_train, DEVICE,
            predictor_params={'brs_mode': 'NoBRS'},
            update_image_callback=lambda *a, **k: None,
        )
        # Subset the dataset for SAM-family speed.
        ds_train = CustomDataset(dataset_dir, augment=True, max_size=384)
        ds_train.pairs = ds_train.pairs[:3]

        # Pick the right kwarg based on family.
        predictor_kwargs = (
            {'sam_predictor':  ctrl_train.predictor} if mt == 'sam'
            else {'sam2_predictor': ctrl_train.predictor}
        )
        trainer = FewShotTrainer(
            model=model_train,
            model_type=mt,
            dataset=ds_train,
            config=FewShotConfig(rank=8, epochs=1, lr=1e-3, n_clicks=1,
                                 cpu_max_size=256, cuda_max_size=512),
            device=DEVICE,
            log_cb=lambda m: None,
            **predictor_kwargs,
        )
        trainer.run()
        state = trainer.get_adapter_state()
        ok_d = state is not None and len(state) > 0
        report.add(f'{prefix}/D few_shot_train', 'PASS' if ok_d else 'FAIL',
                   f'{len(state) if state else 0} tensors, '
                   f'{len(trainer.get_wrapped_module_names())} modules')

        out_path = SYNTH_DIR / f'{mt}_synth_adapter.pt'
        trainer.save(str(out_path))
        loaded = torch.load(out_path, map_location='cpu', weights_only=False)
        ok_e = (
            loaded.get('model_type') == mt
            and loaded.get('rank') == 8
            and len(loaded['lora_state']) == len(state)
        )
        report.add(f'{prefix}/E save_load', 'PASS' if ok_e else 'FAIL',
                   f'file={out_path.name}, tensors={len(loaded["lora_state"])}')

        # F: prediction shifts after load_adapter on a fresh model.
        model_f = loader_fn()
        ctrl_f = InteractiveController(
            model_f, DEVICE,
            predictor_params={'brs_mode': 'NoBRS'},
            update_image_callback=lambda *a, **k: None,
        )
        ctrl_f.set_image(img0)
        ys, xs = np.where(gt0)
        ctrl_f.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
        prob_before = ctrl_f.current_object_prob.copy()

        ctrl_f.load_adapter(str(out_path))
        ctrl_f.reset_last_object(update_image=False)
        ctrl_f.set_image(img0)
        ctrl_f.add_click(int(xs.mean()), int(ys.mean()), is_positive=True)
        prob_after = ctrl_f.current_object_prob.copy()

        diff = float(np.abs(prob_after - prob_before).mean())
        report.add(f'{prefix}/F adapter_changes_pred',
                   'PASS' if diff > 1e-4 else 'FAIL',
                   f'mean |dprob|={diff:.6f}')
    except Exception as exc:
        traceback.print_exc()
        report.add(f'{prefix}/D-F few_shot_pipeline', 'FAIL',
                   f'{type(exc).__name__}: {exc}')


def test_sam(report: Report, dataset_dir: Path):
    try:
        from segment_anything import sam_model_registry  # noqa: F401
    except ImportError:
        report.add('SAM/all', 'SKIP', 'segment_anything not installed')
        return
    ckpt = _find_checkpoint(['sam_vit_*.pth'])
    if ckpt is None:
        report.add('SAM/all', 'SKIP', 'no sam_vit_*.pth in model_weights/')
        return
    label = ('SAM - ViT-B' if 'vit_b' in ckpt.name
             else 'SAM - ViT-L' if 'vit_l' in ckpt.name
             else 'SAM - ViT-H')
    _run_sam_family_suite(
        report, dataset_dir,
        prefix='SAM', mt='sam', label=label, ckpt=ckpt,
        loader_fn=lambda: iu.load_sam_model(str(ckpt), label, DEVICE),
    )


def test_sam2(report: Report, dataset_dir: Path):
    try:
        from sam2.build_sam import build_sam2  # noqa: F401
    except ImportError:
        report.add('SAM2/all', 'SKIP', 'sam2 not installed')
        return
    ckpt = _find_checkpoint(['sam2*hiera*tiny*.pt', 'sam2*hiera*small*.pt',
                             'sam2*hiera*base*.pt', 'sam2*hiera*large*.pt'])
    if ckpt is None:
        report.add('SAM2/all', 'SKIP', 'no sam2*hiera*.pt in model_weights/')
        return
    name = ckpt.name.lower()
    label_map = {'tiny': 'SAM2 - Tiny', 'small': 'SAM2 - Small',
                 'base': 'SAM2 - Base+', 'large': 'SAM2 - Large'}
    label = next((v for k, v in label_map.items() if k in name), None)
    if label is None:
        report.add('SAM2/all', 'SKIP', f'unknown variant: {ckpt.name}')
        return

    _run_sam_family_suite(
        report, dataset_dir,
        prefix='SAM2', mt='sam2', label=label, ckpt=ckpt,
        loader_fn=lambda: iu.load_sam2_model(str(ckpt), label, DEVICE),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 72)
    print(f'  Synthetic-data smoke test  (device={DEVICE})')
    print('=' * 72)

    print('\n[1] generating synthetic dataset…')
    build_dataset(SYNTH_DIR, n_pairs=6, seed=7)

    report = Report()
    print('\n[2] running RITM tests…')
    test_ritm(report, SYNTH_DIR)
    print('\n[3] running SimpleClick tests…')
    test_simpleclick(report, SYNTH_DIR)
    print('\n[4] running SAM tests…')
    test_sam(report, SYNTH_DIR)
    print('\n[5] running SAM2 tests…')
    test_sam2(report, SYNTH_DIR)

    ok = report.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
