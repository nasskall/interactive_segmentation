"""Project models: SAM 2 fine-tuned on a project's own labels.

An admin starts a training run for a project; it runs in a background
thread (one at a time: training is CPU/GPU heavy) and is recorded in the
``project_models`` table, which is the in-app model register: base
checkpoint and its SHA-256, the adapter file and its SHA-256, which items
were held out, and the held-out scores. Two kinds:

``interactive``: SAM 2 that takes clicks, trained on the project's assisted
mask / polygon annotations that recorded their clicks
(``annotations.prompts``) for a single object made with clicks only (no box,
no automatic mask); the target is the mask as saved, so a labeller's
corrections are what the model learns. Scored against the base model with a
simulated labeller; a finished one can be chosen as the project's assist
model.

``automatic``: SAM 2 that segments every mask / polygon class with no clicks
(``isegm.adaptation.promptless``), trained on the project's images marked
done (their labels are taken as complete: a class with no label on such an
image is learned as absent). Scored by per-class IoU; a finished one can be
chosen to propose labels, which a labeller accepts or rejects one by one.

Items, not objects, are split: ``HELD_OUT_FRACTION`` of the items (seeded
shuffle) are held out for scoring and never trained on.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from server.labelling import audit
from server.labelling.db import Database, now
from server.labelling.export import rasterize
from server.labelling.storage import ImageStore

MIN_OBJECTS = 10            # trainable objects an interactive model needs
MIN_IMAGES = 10             # done images an automatic model needs
HELD_OUT_FRACTION = 0.2
MAX_STEPS = 5000


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def default_steps(train_objects: int) -> int:
    """About 50 steps (200 samples) per object, between 300 and 2000."""
    return int(min(2000, max(300, 50 * train_objects)))


def default_auto_steps(train_images: int) -> int:
    """About 30 steps (120 samples) per image, between 300 and 2000."""
    return int(min(2000, max(300, 30 * train_images)))


def model_out(r) -> dict:
    return {'id': r['id'], 'project_id': r['project_id'], 'kind': r['kind'],
            'status': r['status'],
            'classes': json.loads(r['classes']) if r['classes'] else None,
            'base_model_type': r['base_model_type'], 'base_checkpoint': r['base_checkpoint'],
            'base_sha256': r['base_sha256'], 'steps': r['steps'],
            'progress': json.loads(r['progress']) if r['progress'] else None,
            'train_objects': r['train_objects'], 'train_items': r['train_items'],
            'held_out_items': json.loads(r['held_out_items']),
            'metrics': json.loads(r['metrics']) if r['metrics'] else None,
            'adapter_file': r['adapter_file'], 'adapter_sha256': r['adapter_sha256'],
            'error': r['error'], 'created_by': r['created_by'], 'created_at': r['created_at'],
            'finished_at': r['finished_at']}


# ── training data ─────────────────────────────────────────────────────────

def trainable_objects(conn, project_id: int) -> list[dict]:
    """The project's annotations a model can learn from: one object, made
    with clicks only, saved as a mask or polygon."""
    out = []
    for r in conn.execute(
            'SELECT a.id, a.item_id, a.geometry, a.prompts, i.files, i.width, i.height '
            'FROM annotations a JOIN items i ON i.id = a.item_id '
            'JOIN classes c ON c.id = a.class_id '
            "WHERE i.project_id = ? AND a.prompts IS NOT NULL "
            "AND c.label_type IN ('mask', 'polygon') ORDER BY a.id", (project_id,)):
        prompts = json.loads(r['prompts'])
        objects = prompts.get('objects', [])
        if len(objects) != 1 or not objects[0]['clicks'] or objects[0].get('box'):
            continue
        files = json.loads(r['files'])
        if prompts.get('image') not in files:
            continue
        out.append({'annotation_id': r['id'], 'item_id': r['item_id'],
                    'file': files[prompts['image']], 'width': r['width'], 'height': r['height'],
                    'geometry': json.loads(r['geometry']),
                    # stored as [x, y, positive]; training takes (y, x, positive)
                    'clicks': [(int(y), int(x), bool(p)) for x, y, p in objects[0]['clicks']]})
    return out


def split_items(objects: list[dict], seed: int = 0) -> tuple[list[dict], list[dict], list[int]]:
    """(train objects, held-out objects, held-out item ids)."""
    items = sorted({o['item_id'] for o in objects})
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    held = set(items[:max(1, round(HELD_OUT_FRACTION * len(items)))]) if len(items) > 1 else set()
    return ([o for o in objects if o['item_id'] not in held],
            [o for o in objects if o['item_id'] in held], sorted(held))


def data_summary(conn, project_id: int) -> dict:
    objects = trainable_objects(conn, project_id)
    train, held, held_items = split_items(objects)
    assisted = conn.execute(
        "SELECT COUNT(*) FROM annotations a JOIN items i ON i.id = a.item_id "
        "WHERE i.project_id = ? AND a.source = 'assisted'", (project_id,)).fetchone()[0]
    problems = []
    if len(objects) < MIN_OBJECTS:
        problems.append(f'Needs at least {MIN_OBJECTS} objects labelled with clicks '
                        f'(has {len(objects)}).')
    if not held or not train:
        problems.append('Needs labelled objects on at least two images.')
    return {'assisted_annotations': assisted, 'trainable_objects': len(objects),
            'items': len({o['item_id'] for o in objects}), 'train_objects': len(train),
            'held_out_objects': len(held), 'held_out_items': len(held_items),
            'default_steps': default_steps(len(train)), 'min_objects': MIN_OBJECTS,
            'max_steps': MAX_STEPS, 'problems': problems}


def automatic_data(conn, project_id: int) -> tuple[list[dict], list[dict]]:
    """(classes an automatic model predicts, in output order; the project's
    done single images with their mask / polygon labels)."""
    classes = [{'id': r['id'], 'name': r['name']} for r in conn.execute(
        "SELECT id, name FROM classes WHERE project_id = ? AND label_type IN ('mask', 'polygon') "
        'ORDER BY position, id', (project_id,))]
    index = {c['id']: i for i, c in enumerate(classes)}
    items = []
    for it in conn.execute("SELECT * FROM items WHERE project_id = ? AND status = 'done' "
                           "AND kind = 'single' ORDER BY id", (project_id,)):
        labels = [(index[a['class_id']], json.loads(a['geometry'])) for a in conn.execute(
            'SELECT class_id, geometry FROM annotations WHERE item_id = ? ORDER BY id', (it['id'],))
            if a['class_id'] in index]
        items.append({'item_id': it['id'], 'file': json.loads(it['files'])['image'],
                      'width': it['width'], 'height': it['height'], 'labels': labels})
    return classes, items


def automatic_summary(conn, project_id: int) -> dict:
    classes, items = automatic_data(conn, project_id)
    train, held, held_items = split_items(items)
    problems = []
    if not classes:
        problems.append('Needs a mask or polygon class.')
    if len(items) < MIN_IMAGES:
        problems.append(f'Needs at least {MIN_IMAGES} images marked done (has {len(items)}).')
    if items and not any(it['labels'] for it in items):
        problems.append('The done images have no mask or polygon labels.')
    return {'classes': classes, 'done_images': len(items),
            'labels': sum(len(it['labels']) for it in items), 'train_images': len(train),
            'held_out_images': len(held_items), 'default_steps': default_auto_steps(len(train)),
            'min_images': MIN_IMAGES, 'max_steps': MAX_STEPS, 'problems': problems}


# ── runner ────────────────────────────────────────────────────────────────

FitFn = Callable[[dict, Callable[..., None], Callable[[], bool]], dict]


class TrainingRunner:
    """Runs one training job at a time in a background thread. ``fit(job,
    report, should_stop)`` does the work and returns ``{'metrics': ...,
    'adapter_file': ...}``; by default ``FITS[kind]`` (tests set ``fit``)."""

    def __init__(self, db: Database, images: ImageStore, models_dir: Path,
                 weights_dir: Path, device: str, fit: Optional[FitFn] = None):
        self.db, self.images, self.device = db, images, device
        self.models_dir, self.weights_dir = Path(models_dir), Path(weights_dir)
        self.fit = fit
        self._thread: Optional[threading.Thread] = None
        self._running_id: Optional[int] = None
        self._cancel = threading.Event()
        self._start_lock = threading.Lock()
        with closing(db.connect()) as conn, conn:
            # A server restart kills any run that was in progress.
            conn.execute("UPDATE project_models SET status = 'failed', "
                         "error = 'Interrupted by a server restart', finished_at = ? "
                         "WHERE status IN ('queued', 'running')", (now(),))

    @property
    def running_id(self) -> Optional[int]:
        return self._running_id if self._thread and self._thread.is_alive() else None

    def start(self, project_id: int, user, kind: str, base_model_type: str,
              base_checkpoint: str, base_sha256: Optional[str], steps: int,
              records: list[dict], classes: Optional[list[dict]] = None) -> int:
        """``records``: objects (interactive) or images (automatic), each
        with an ``item_id``; ``classes``: an automatic model's outputs."""
        # Requests run in a threadpool: check and start under one lock.
        with self._start_lock:
            if self.running_id is not None:
                raise RuntimeError(f'Model {self._running_id} is still training; '
                                   'wait for it or cancel it')
            return self._start(project_id, user, kind, base_model_type, base_checkpoint,
                               base_sha256, steps, records, classes)

    def _start(self, project_id, user, kind, base_model_type, base_checkpoint, base_sha256,
               steps, records, classes) -> int:
        train, held, held_items = split_items(records)
        objects = (len(train) if kind == 'interactive'
                   else sum(len(r['labels']) for r in train))
        with closing(self.db.connect()) as conn, conn:
            cur = conn.execute(
                'INSERT INTO project_models (project_id, kind, status, base_model_type, '
                'base_checkpoint, base_sha256, steps, train_objects, train_items, held_out_items, '
                "classes, created_by, created_at) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, "
                '?, ?)',
                (project_id, kind, base_model_type, base_checkpoint, base_sha256, steps, objects,
                 len({r['item_id'] for r in train}), json.dumps(held_items),
                 json.dumps(classes) if classes is not None else None, user['id'], now()))
            model_id = cur.lastrowid
            audit.record(conn, 'model.train', user=user, target=f'project_model:{model_id}',
                         detail={'project_id': project_id, 'kind': kind,
                                 'base_checkpoint': base_checkpoint, 'steps': steps,
                                 'train_objects': objects, 'held_out_items': held_items})
        job = {'id': model_id, 'project_id': project_id, 'kind': kind, 'classes': classes,
               'base_model_type': base_model_type,
               'base_checkpoint': base_checkpoint, 'base_sha256': base_sha256, 'steps': steps,
               'train': train, 'held': held, 'images': self.images, 'device': self.device,
               'weights_dir': self.weights_dir,
               'out_dir': self.models_dir / f'project_{project_id}' / f'model_{model_id}'}
        self._cancel.clear()
        self._running_id = model_id
        self._thread = threading.Thread(target=self._work, args=(job,), daemon=True,
                                        name=f'train-project-model-{model_id}')
        self._thread.start()
        return model_id

    def cancel(self, model_id: int) -> bool:
        if self.running_id != model_id:
            return False
        self._cancel.set()
        return True

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _set(self, model_id: int, **cols) -> None:
        with closing(self.db.connect()) as conn, conn:
            sets = ', '.join(f'{k} = ?' for k in cols)
            conn.execute(f'UPDATE project_models SET {sets} WHERE id = ?',
                         (*cols.values(), model_id))

    def _work(self, job: dict) -> None:
        model_id = job['id']
        self._set(model_id, status='running')
        last = [0.0]

        def report(phase: str, done: int, total: int, loss: Optional[float] = None) -> None:
            # Throttled: the training loop calls this every step.
            if time.monotonic() - last[0] < 2 and done != total:
                return
            last[0] = time.monotonic()
            self._set(model_id, progress=json.dumps(
                {'phase': phase, 'done': done, 'total': total, 'loss': loss}))

        fit = self.fit or FITS[job['kind']]
        try:
            result = fit(job, report, self._cancel.is_set)
        except Exception as exc:
            cancelled = self._cancel.is_set()
            self._set(model_id, status='cancelled' if cancelled else 'failed',
                      error=None if cancelled else f'{type(exc).__name__}: {exc}',
                      finished_at=now())
            shutil.rmtree(job['out_dir'], ignore_errors=True)
            audit.record_now(self.db, 'model.train_cancelled' if cancelled else 'model.train_failed',
                             target=f'project_model:{model_id}',
                             detail=None if cancelled else {'error': str(exc)[:500]})
            return
        adapter = job['out_dir'] / result['adapter_file']
        sha = file_sha256(adapter)
        self._set(model_id, status='done', metrics=json.dumps(result['metrics']),
                  adapter_file=str(adapter.relative_to(self.models_dir).as_posix()),
                  adapter_sha256=sha, finished_at=now())
        audit.record_now(self.db, 'model.train_done', target=f'project_model:{model_id}',
                         detail={'adapter_sha256': sha, 'metrics': result['metrics']})


def _examples(objects: list[dict], images: ImageStore):
    from isegm.adaptation.project_training import Example

    def loader(o):
        def load():
            mask = rasterize(o['geometry'], o['width'], o['height']).astype(bool)
            return images.load_rgb(o['item_id'], o['file']), mask
        return load
    return [Example(f"a{o['annotation_id']}", loader(o), o['clicks']) for o in objects]


def fit_sam2(job: dict, report, should_stop) -> dict:
    """Score the base model on the held-out objects, fine-tune it on the
    training objects, score it again, save the adapter."""
    import torch

    from isegm.adaptation import project_training as pt
    from isegm.inference import utils

    net = utils.load_model_by_type(str(job['weights_dir'] / job['base_checkpoint']),
                                   job['base_model_type'], job['device'])
    train, held = _examples(job['train'], job['images']), _examples(job['held'], job['images'])
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    report('scoring base model', 0, len(held))
    base = pt.evaluate(net, held, job['device'], should_stop=should_stop)
    try:
        payload = pt.train(net, train, job['steps'], job['device'], out / 'features',
                           report=report, should_stop=should_stop)
    finally:
        shutil.rmtree(out / 'features', ignore_errors=True)
    report('scoring trained model', 0, len(held))
    trained = pt.evaluate(net, held, job['device'], should_stop=should_stop)
    metrics = {'base': base, 'trained': trained}
    payload['project_training'] = {
        'project_model_id': job['id'], 'project_id': job['project_id'],
        'base_model_type': job['base_model_type'], 'base_checkpoint': job['base_checkpoint'],
        'base_sha256': job['base_sha256'], 'steps': job['steps'], 'metrics': metrics}
    torch.save(payload, out / 'adapter.pt')
    return {'metrics': metrics, 'adapter_file': 'adapter.pt'}


def _auto_examples(items: list[dict], n_classes: int, images: ImageStore):
    from isegm.adaptation.promptless import Example

    def loader(it):
        def load():
            masks = np.zeros((n_classes, it['height'], it['width']), bool)
            for c, geom in it['labels']:
                masks[c] |= rasterize(geom, it['width'], it['height']).astype(bool)
            return images.load_rgb(it['item_id'], it['file']), masks
        return load
    return [Example(f"i{it['item_id']}", loader(it)) for it in items]


def fit_promptless(job: dict, report, should_stop) -> dict:
    """Train a click-free SAM 2 on the training images, score it per class on
    the held-out ones, save the adapter with its class prompts."""
    import torch

    from isegm.adaptation import promptless as pl
    from isegm.inference import utils

    net = utils.load_model_by_type(str(job['weights_dir'] / job['base_checkpoint']),
                                   job['base_model_type'], job['device'])
    n = len(job['classes'])
    train = _auto_examples(job['train'], n, job['images'])
    held = _auto_examples(job['held'], n, job['images'])
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    try:
        prompts, payload = pl.train(net, n, train, job['steps'], job['device'], out / 'features',
                                    report=report, should_stop=should_stop)
    finally:
        shutil.rmtree(out / 'features', ignore_errors=True)
    report('scoring', 0, len(held))
    metrics = {'trained': pl.evaluate(net, prompts, held, job['device'], should_stop=should_stop)}
    payload['project_training'] = {
        'project_model_id': job['id'], 'project_id': job['project_id'], 'kind': 'automatic',
        'classes': job['classes'], 'base_model_type': job['base_model_type'],
        'base_checkpoint': job['base_checkpoint'], 'base_sha256': job['base_sha256'],
        'steps': job['steps'], 'metrics': metrics}
    torch.save(payload, out / 'adapter.pt')
    return {'metrics': metrics, 'adapter_file': 'adapter.pt'}


FITS: dict[str, FitFn] = {'interactive': fit_sam2, 'automatic': fit_promptless}
