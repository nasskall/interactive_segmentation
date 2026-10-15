"""SQLite storage for labelling projects.

One file (``<data_dir>/labelling.db``) holds users, projects, classes, items
and annotations; image files live next to it (see ``storage.py``). Each
request opens its own connection, which SQLite handles well at the scale of
a labelling team.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TASK_TYPES = ('classification', 'detection', 'semantic_segmentation',
              'instance_segmentation', 'change_detection')
LABEL_TYPES = ('tag', 'box', 'polygon', 'mask')
ROLES = ('admin', 'labeller')
ITEM_STATUSES = ('todo', 'in_progress', 'done')

# Which label types each task type can use.
ALLOWED_LABEL_TYPES = {
    'classification': ('tag',),
    'detection': ('box',),
    'semantic_segmentation': ('mask', 'polygon'),
    'instance_segmentation': ('mask', 'polygon'),
    'change_detection': ('mask', 'polygon', 'box'),
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN {ROLES}),
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    task_type   TEXT NOT NULL CHECK (task_type IN {TASK_TYPES}),
    multi_label INTEGER NOT NULL DEFAULT 0,
    protocol    TEXT NOT NULL DEFAULT '',
    created_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classes (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name       TEXT NOT NULL COLLATE NOCASE,
    color      TEXT NOT NULL,
    label_type TEXT NOT NULL CHECK (label_type IN {LABEL_TYPES}),
    position   INTEGER NOT NULL DEFAULT 0,
    UNIQUE (project_id, name)
);

CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('single', 'pair')),
    name        TEXT NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    files       TEXT NOT NULL,
    assignee_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    status      TEXT NOT NULL DEFAULT 'todo' CHECK (status IN {ITEM_STATUSES}),
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_project ON items(project_id);
CREATE INDEX IF NOT EXISTS items_assignee ON items(assignee_id);

CREATE TABLE IF NOT EXISTS annotations (
    id          INTEGER PRIMARY KEY,
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    labeller_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    geometry    TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('manual', 'assisted')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS annotations_item ON annotations(item_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        # FastAPI may run a request's dependency and endpoint on different
        # threadpool threads; each connection is still used by one request.
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        return conn
