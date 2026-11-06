"""
Clickability-map worker for tools/rclicks_clicker.py.

Runs the RClicks clickability model (Antonov et al., "RClicks: Realistic
Click Simulation for Benchmarking Interactive Segmentation", NeurIPS 2024,
https://github.com/emb-ai/rclicks, MIT) in its own environment: the model
needs mmsegmentation 1.2.2 (bundled in that repository), mmengine and
mmcv-lite 2.1, and the repository's package also ships a fork of RITM's
``isegm``, which would shadow this repo's. Only its ``nets`` module is loaded
(the package ``__init__`` is skipped) and its code is used unmodified.

Protocol on stdin/stdout: each request and reply is a 4-byte big-endian
length followed by an .npz payload. Request: image (H x W x 3 uint8 RGB),
gt (H x W bool), error (H x W bool). Reply: map (H x W float32, 0..1),
as the model's ``apply`` returns it. Started by RClicksClient; not run by hand.

Setup (once): see the docstring of tools/rclicks_clicker.py.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch


class _MissingOps(types.ModuleType):
    """Stand-in for mmcv.ops when mmcv is installed without compiled ops
    (mmcv-lite). mmsegmentation imports a few ops at package import time
    (point_sample, PSAMask, deform convs, ...) that the clickability network
    never runs; any actual use raises instead of computing something wrong."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)

        def missing(*args, **kwargs):
            raise RuntimeError(f'mmcv.ops.{name} needs the full mmcv build; the clickability '
                               'model should not use it')
        return missing


def load_model(root: Path, checkpoint: Path, device: str):
    try:
        import mmcv.ops  # noqa: F401
    except ImportError:
        sys.modules['mmcv.ops'] = _MissingOps('mmcv.ops')
    for p in (root / 'mmsegmentation', root / 'transalnet'):
        sys.path.insert(0, str(p))
    pkg = types.ModuleType('rclicks')           # skip rclicks/__init__.py
    pkg.__path__ = [str(root / 'rclicks' / 'rclicks')]
    sys.modules['rclicks'] = pkg
    import albumentations as A
    from rclicks import nets

    model = nets.SegNeXtSaliency()
    with torch.no_grad():                       # builds lazily created layers, as their loader does
        model(torch.randn(1, 3, 100, 100), torch.randn(1, 2, 100, 100))
    # Their loaders call torch.load without weights_only; the checkpoint is
    # plain tensors, so it is loaded here with weights_only=True.
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.normalization = A.Compose([A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225],
                                                 max_pixel_value=1.0)])
    model.device = device
    model.to(device).eval()
    return lambda image, gt, error: nets.SegNeXtSaliencyApply.apply(model, image, gt, error)


def read_msg(stream) -> dict | None:
    head = stream.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack('>I', head)
    return dict(np.load(io.BytesIO(stream.read(n))))


def write_msg(stream, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    stream.write(struct.pack('>I', len(data)) + data)
    stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rclicks-root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    apply = load_model(args.rclicks_root.resolve(), args.checkpoint.resolve(), args.device)
    print(f'rclicks worker ready on {args.device}', file=sys.stderr, flush=True)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    with torch.no_grad():
        while (req := read_msg(stdin)) is not None:
            out = apply(req['image'], req['gt'], req['error'])
            write_msg(stdout, map=out[0][0].detach().cpu().numpy().astype(np.float32))


if __name__ == '__main__':
    main()
