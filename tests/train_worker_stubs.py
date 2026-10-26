"""Training functions for tests/test_train_worker.py, run inside a real
train_worker process (``--fit train_worker_stubs:<name>``)."""

from __future__ import annotations

import os
import time
from pathlib import Path


def ok(job, report, should_stop):
    report('training', 5, 10, loss=0.5)
    (job['out_dir'] / 'adapter.pt').write_bytes(b'adapter')
    # What the worker rebuilt from job.json, for the test to check.
    return {'metrics': {'device': job['device'], 'images_root': str(job['images'].root),
                        'paths': all(isinstance(job[k], Path) for k in ('out_dir', 'models_dir')),
                        'train': len(job['train']), 'slurm': os.environ.get('SLURM_JOB_ID')},
            'adapter_file': 'adapter.pt'}


def fails(job, report, should_stop):
    raise ValueError('bad data')


def crashes(job, report, should_stop):
    os._exit(5)


def slow(job, report, should_stop):
    deadline = time.monotonic() + 60
    while not should_stop() and time.monotonic() < deadline:
        report('training', 1, 100, loss=1.0)
        time.sleep(0.05)
    raise RuntimeError('stopped')
