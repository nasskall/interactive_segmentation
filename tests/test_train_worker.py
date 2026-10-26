"""Project-model training in a separate worker process (server.labelling.
train_worker), directly or through a launcher such as Slurm, with the server
copying its progress and result into the database."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from server.labelling import training
from server.labelling.db import Database, now
from server.labelling.storage import ImageStore

TESTS = Path(__file__).resolve().parent
USER = {'id': 1, 'username': 'admin'}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    # The worker imports the stubs from this folder.
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(filter(None, [str(TESTS),
                                                                     os.environ.get('PYTHONPATH')])))
    db = Database(tmp_path / 'labelling.db')
    with closing(db.connect()) as conn, conn:
        conn.execute("INSERT INTO users (username, display_name, password_hash, role, created_at) "
                     "VALUES ('admin', 'Admin', 'x', 'admin', ?)", (now(),))
        conn.execute("INSERT INTO projects (name, task_type, created_at) "
                     "VALUES ('p', 'instance_segmentation', ?)", (now(),))
    r = training.TrainingRunner(db, ImageStore(tmp_path / 'items'), tmp_path / 'models',
                                tmp_path / 'weights', 'cpu', launcher=[], worker_device='cpu')
    return r


def _start(runner, stub: str) -> int:
    runner.worker_fit = f'train_worker_stubs:{stub}'
    records = [{'item_id': i, 'clicks': [[1, 2, True]]} for i in range(10)]
    return runner.start(1, USER, 'interactive', 'SAM2 - Tiny', 'sam2.pt', None, 20, records)


def _row(runner, model_id) -> dict:
    with closing(runner.db.connect()) as conn:
        return training.model_out(conn.execute('SELECT * FROM project_models WHERE id = ?',
                                               (model_id,)).fetchone())


def test_worker_trains_and_the_server_registers_it(runner):
    m = _start(runner, 'ok')
    runner.join(60)
    row = _row(runner, m)
    assert row['status'] == 'done', row['error']
    assert row['metrics'] == {'device': 'cpu', 'images_root': str(runner.images.root),
                              'paths': True, 'train': 8, 'slurm': None}
    assert row['progress']['phase'] == 'training' and row['progress']['done'] == 5
    out = runner.models_dir / 'project_1' / f'model_{m}'
    assert sorted(p.name for p in out.iterdir()) == ['adapter.pt']      # job files cleaned up
    assert row['adapter_sha256'] == training.file_sha256(out / 'adapter.pt')
    log = training.log_path(runner.models_dir, 1, m).read_text()
    assert f'training project model {m}' in log and 'done: adapter.pt' in log


def test_launcher_runs_the_worker_as_a_scheduler_job(runner):
    runner.launcher = [sys.executable, str(TESTS / 'fake_srun.py')]
    m = _start(runner, 'ok')
    runner.join(60)
    row = _row(runner, m)
    assert row['status'] == 'done', row['error']
    assert row['metrics']['slurm'] == '4242' and row['progress']['slurm_job'] == '4242'
    log = training.log_path(runner.models_dir, 1, m).read_text()
    assert f'fake srun: iseg-m{m} | iseg project model {m} project 1 user admin' in log


@pytest.mark.parametrize('stub, error', [('fails', 'ValueError: bad data'),
                                         ('crashes', 'worker exited with code 5')])
def test_failures_are_recorded_with_the_reason(runner, stub, error):
    m = _start(runner, stub)
    runner.join(60)
    row = _row(runner, m)
    assert row['status'] == 'failed' and error in row['error']
    assert not (runner.models_dir / 'project_1' / f'model_{m}').exists()
    assert training.log_path(runner.models_dir, 1, m).exists()        # kept for diagnosis


def test_cancel_stops_the_worker(runner):
    m = _start(runner, 'slow')
    deadline = time.monotonic() + 60
    while (_row(runner, m)['progress'] or {}).get('phase') != 'training':
        assert time.monotonic() < deadline, 'worker never reported progress'
        time.sleep(0.2)
    assert runner.cancel(m)
    runner.join(60)
    row = _row(runner, m)
    assert row['status'] == 'cancelled' and row['error'] is None
    assert runner.running_id is None
