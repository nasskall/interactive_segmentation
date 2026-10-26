"""
Runs one project-model training job, outside the web server.

    python -m server.labelling.train_worker <model folder>/job.json [--fit module:function]

``TrainingRunner`` writes ``job.json`` and starts this, directly or through
``ISEG_TRAIN_LAUNCHER`` (e.g. as a Slurm job on a GPU). It never touches the
database: it reports through ``progress.json`` (phase, done / total, loss,
and the Slurm job id when there is one) and ends with ``result.json``
(``ok`` and the result, or ``error``, or ``cancelled``) in the same folder,
which the server reads. SIGTERM (cancel; Slurm forwards it) makes the job
stop at its next step. Exit code: 0 done, 1 failed, 3 cancelled.
``--fit`` runs another training function (tests).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from server.labelling.storage import ImageStore
from server.labelling.training import FITS

PATH_KEYS = ('models_dir', 'weights_dir', 'out_dir')


def _write(path: Path, data: dict, tries: int = 50) -> None:
    """Atomically, so the server never reads half a file. On Windows the
    rename fails while the server has the file open to read it; retry."""
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data), encoding='utf-8')
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def load_job(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding='utf-8'))
    job = {k: Path(v) if k in PATH_KEYS else v for k, v in spec.items()}
    job['images'] = ImageStore(Path(spec['items_dir']))
    if not job.get('device'):
        import torch
        job['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('job', type=Path)
    ap.add_argument('--fit', help='module:function to run instead of the kind\'s training')
    args = ap.parse_args()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    job = load_job(args.job)
    out = Path(job['out_dir'])
    slurm_job = os.environ.get('SLURM_JOB_ID')
    print(f"training project model {job['id']} ({job['kind']}/{job['arch']}) on {job['device']}"
          + (f', Slurm job {slurm_job}' if slurm_job else ''), flush=True)
    last = {'at': 0.0, 'phase': None}

    def report(phase: str, done: int, total: int, loss=None) -> None:
        # Throttled within a phase: training loops call this every step.
        if (phase == last['phase'] and time.monotonic() - last['at'] < 2
                and done != total):
            return
        last.update(at=time.monotonic(), phase=phase)
        try:
            _write(out / 'progress.json', {'phase': phase, 'done': done, 'total': total,
                                           'loss': loss, 'slurm_job': slurm_job}, tries=5)
        except OSError as exc:      # progress is best effort; never fail the training
            print(f'progress not written: {exc}', flush=True)

    report('starting', 0, 0)
    if args.fit:
        module, _, name = args.fit.partition(':')
        fit = getattr(importlib.import_module(module), name)
    else:
        fit = FITS['student' if job['arch'] == 'unet' else job['kind']]
    try:
        result = fit(job, report, stop.is_set)
    except Exception as exc:
        if stop.is_set():
            _write(out / 'result.json', {'cancelled': True})
            print('cancelled', flush=True)
            return 3
        traceback.print_exc()
        _write(out / 'result.json', {'error': f'{type(exc).__name__}: {exc}'})
        return 1
    _write(out / 'result.json', {'ok': True, **result})
    print(f"done: {result.get('adapter_file')}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
