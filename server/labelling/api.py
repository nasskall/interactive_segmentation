"""REST endpoints for labelling projects (all under /api).

Admins manage users, projects, classes and items. Labellers see the projects
in which they have assigned items, and annotate those items. Every endpoint
except setup, login and status needs ``Authorization: Bearer <token>``.

The first admin is created through ``POST /auth/setup`` with a one-time
setup token that the server prints on startup while no users exist.

Assisted labelling: ``POST /items/{id}/assist`` opens an interactive
segmentation session (the existing ``/api/sessions`` endpoints) on the
item's image. Saving an annotation with ``from_session`` turns that
session's mask into the class's label type (mask, box or polygon) at the
original resolution, then clears the session for the next object. The
annotation records which model proposed it (``model``: type, checkpoint and
SHA-256).

Security events and every change to accounts, projects, items and
annotations go to the hash-chained audit log (``audit.py``), which admins
read through ``GET /audit`` and check with ``GET /audit/verify``. Repeated
failed sign-ins lock the username and the client address for a while.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from server.labelling import audit, auth, export, geometry
from server.labelling.db import (ALLOWED_LABEL_TYPES, ITEM_STATUSES, ROLES, TASK_TYPES,
                                 Database, now)
from server.labelling.storage import ImageError, ImageStore, decode, extension
from server.sessions import ModelManager

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
USERNAME_PATTERN = r'^[A-Za-z0-9_.-]{3,32}$'
COLOR_PATTERN = r'^#[0-9a-fA-F]{6}$'


# ── request bodies ────────────────────────────────────────────────────────

MAX_PASSWORD_LENGTH = 1024


class SetupIn(BaseModel):
    setup_token: str
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=auth.MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str = Field(min_length=1, max_length=100)


class LoginIn(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)


class UserIn(BaseModel):
    username: str = Field(pattern=USERNAME_PATTERN)
    password: str = Field(min_length=auth.MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str = Field(min_length=1, max_length=100)
    role: Literal[ROLES] = 'labeller'


class UserPatch(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    password: Optional[str] = Field(default=None, min_length=auth.MIN_PASSWORD_LENGTH,
                                    max_length=MAX_PASSWORD_LENGTH)
    role: Optional[Literal[ROLES]] = None
    active: Optional[bool] = None


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    task_type: Literal[TASK_TYPES]
    multi_label: bool = False
    protocol: str = ''


class ProjectPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    multi_label: Optional[bool] = None
    protocol: Optional[str] = None


class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: str = Field(pattern=COLOR_PATTERN)
    label_type: Optional[Literal[('tag', 'box', 'polygon', 'mask')]] = None
    position: Optional[int] = None


class ClassPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    color: Optional[str] = Field(default=None, pattern=COLOR_PATTERN)
    position: Optional[int] = None


class ItemPatch(BaseModel):
    assignee_id: Optional[int] = None
    status: Optional[Literal[ITEM_STATUSES]] = None


class AssignIn(BaseModel):
    user_id: Optional[int]
    item_ids: list[int] = Field(min_length=1)


class AnnotationIn(BaseModel):
    class_id: int
    geometry: Optional[dict] = None
    from_session: Optional[str] = None


class AnnotationPatch(BaseModel):
    class_id: Optional[int] = None
    geometry: Optional[dict] = None


# ── row → JSON ────────────────────────────────────────────────────────────

def user_out(r: sqlite3.Row) -> dict:
    return {'id': r['id'], 'username': r['username'], 'display_name': r['display_name'],
            'role': r['role'], 'active': bool(r['active']), 'created_at': r['created_at']}


def project_out(r: sqlite3.Row) -> dict:
    return {'id': r['id'], 'name': r['name'], 'task_type': r['task_type'],
            'multi_label': bool(r['multi_label']), 'protocol': r['protocol'],
            'allowed_label_types': list(ALLOWED_LABEL_TYPES[r['task_type']]),
            'created_at': r['created_at']}


def class_out(r: sqlite3.Row) -> dict:
    return {'id': r['id'], 'project_id': r['project_id'], 'name': r['name'],
            'color': r['color'], 'label_type': r['label_type'], 'position': r['position']}


def item_out(r: sqlite3.Row) -> dict:
    files = json.loads(r['files'])
    return {'id': r['id'], 'project_id': r['project_id'], 'kind': r['kind'],
            'name': r['name'], 'width': r['width'], 'height': r['height'],
            'images': {role: f"/api/items/{r['id']}/image/{role}" for role in files},
            'assignee_id': r['assignee_id'], 'status': r['status'],
            'annotation_count': r['annotation_count'] if 'annotation_count' in r.keys() else None,
            'created_at': r['created_at']}


def annotation_out(r: sqlite3.Row) -> dict:
    return {'id': r['id'], 'item_id': r['item_id'], 'class_id': r['class_id'],
            'labeller_id': r['labeller_id'], 'geometry': json.loads(r['geometry']),
            'source': r['source'],
            'model': json.loads(r['model']) if r['model'] else None,
            'created_at': r['created_at'], 'updated_at': r['updated_at']}


# ── router ────────────────────────────────────────────────────────────────

@dataclass
class LabellingContext:
    db: Database
    images: ImageStore
    manager: ModelManager
    setup_token: Optional[str] = None
    # assist session id -> (item id, user id)
    assist_sessions: dict[str, tuple[int, int]] = field(default_factory=dict)
    throttle: auth.LoginThrottle = field(default_factory=auth.LoginThrottle)
    # FastAPI dependencies, shared with the segmentation endpoints in app.py.
    get_db: Optional[Callable] = None
    current_user: Optional[Callable] = None
    admin_user: Optional[Callable] = None


async def remember_client_ip(request: Request) -> None:
    """Router dependency: the client address the audit log records."""
    audit.client_ip.set(request.client.host if request.client else '')


def build_router(data_dir: Path, manager: ModelManager) -> tuple[APIRouter, LabellingContext]:
    data_dir = Path(data_dir)
    ctx = LabellingContext(Database(data_dir / 'labelling.db'),
                           ImageStore(data_dir / 'items'), manager)
    with ctx.db.connect() as conn:
        if conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
            ctx.setup_token = secrets.token_urlsafe(16)

    router = APIRouter(prefix='/api', dependencies=[Depends(remember_client_ip)])
    bearer = HTTPBearer(auto_error=False)

    # ── dependencies ──────────────────────────────────────────────────────

    def get_db() -> Iterator[sqlite3.Connection]:
        conn = ctx.db.connect()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
                     conn: sqlite3.Connection = Depends(get_db, scope='function')) -> sqlite3.Row:
        user = auth.user_for_token(conn, creds.credentials) if creds else None
        if user is None:
            raise HTTPException(401, 'Not signed in', headers={'WWW-Authenticate': 'Bearer'})
        return user

    def admin_user(user: sqlite3.Row = Depends(current_user)) -> sqlite3.Row:
        if user['role'] != 'admin':
            raise HTTPException(403, 'Admins only')
        return user

    ctx.get_db, ctx.current_user, ctx.admin_user = get_db, current_user, admin_user

    # ── access helpers ────────────────────────────────────────────────────

    def get_project(conn, project_id: int, user) -> sqlite3.Row:
        p = conn.execute('SELECT * FROM projects WHERE id = ?', (project_id,)).fetchone()
        if p is not None and user['role'] != 'admin':
            assigned = conn.execute('SELECT 1 FROM items WHERE project_id = ? AND assignee_id = ?',
                                    (project_id, user['id'])).fetchone()
            p = p if assigned else None
        if p is None:
            raise HTTPException(404, 'Project not found')
        return p

    def get_item(conn, item_id: int, user) -> sqlite3.Row:
        it = conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()
        if it is None or (user['role'] != 'admin' and it['assignee_id'] != user['id']):
            raise HTTPException(404, 'Item not found')
        return it

    def get_class(conn, class_id: int, project_id: int) -> sqlite3.Row:
        c = conn.execute('SELECT * FROM classes WHERE id = ? AND project_id = ?',
                         (class_id, project_id)).fetchone()
        if c is None:
            raise HTTPException(400, f'Class {class_id} is not in this project')
        return c

    def get_annotation(conn, ann_id: int, user) -> tuple[sqlite3.Row, sqlite3.Row]:
        a = conn.execute('SELECT * FROM annotations WHERE id = ?', (ann_id,)).fetchone()
        if a is None:
            raise HTTPException(404, 'Annotation not found')
        it = get_item(conn, a['item_id'], user)
        if user['role'] != 'admin' and a['labeller_id'] != user['id']:
            raise HTTPException(403, "You can only change your own annotations")
        return a, it

    def check_assignee(conn, user_id: Optional[int]) -> None:
        if user_id is None:
            return
        u = conn.execute('SELECT active FROM users WHERE id = ?', (user_id,)).fetchone()
        if u is None or not u['active']:
            raise HTTPException(400, f'User {user_id} does not exist or is inactive')

    def active_admins(conn) -> int:
        return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1"
                            ).fetchone()[0]

    # ── auth ──────────────────────────────────────────────────────────────

    @router.get('/auth/status')
    def auth_status():
        return {'needs_setup': ctx.setup_token is not None}

    @router.post('/auth/setup', status_code=201)
    def setup(body: SetupIn, conn: sqlite3.Connection = Depends(get_db, scope='function')):
        if ctx.setup_token is None:
            raise HTTPException(409, 'Setup is already done')
        if not secrets.compare_digest(body.setup_token, ctx.setup_token):
            raise HTTPException(403, 'Wrong setup token')
        cur = conn.execute(
            'INSERT INTO users (username, display_name, password_hash, role, created_at) '
            "VALUES (?, ?, ?, 'admin', ?)",
            (body.username, body.display_name, auth.hash_password(body.password), now()))
        ctx.setup_token = None
        user = conn.execute('SELECT * FROM users WHERE id = ?', (cur.lastrowid,)).fetchone()
        audit.record(conn, 'auth.setup', user=user, target=f"user:{user['id']}")
        return {'token': auth.issue_token(conn, user['id']), 'user': user_out(user)}

    @router.post('/auth/login')
    def login(body: LoginIn, conn: sqlite3.Connection = Depends(get_db, scope='function')):
        keys = (f'user:{body.username.lower()}', f'ip:{audit.client_ip.get()}')
        wait = ctx.throttle.locked(*keys)
        if wait:
            audit.record_now(ctx.db, 'auth.login_blocked', username=body.username)
            raise HTTPException(429, f'Too many failed sign-ins; try again in '
                                     f'{int(wait) // 60 + 1} min',
                                headers={'Retry-After': str(int(wait) + 1)})
        user = conn.execute('SELECT * FROM users WHERE username = ?', (body.username,)).fetchone()
        if (user is None or not user['active']
                or not auth.verify_password(body.password, user['password_hash'])):
            locked = ctx.throttle.failed(*keys)
            audit.record_now(ctx.db, 'auth.login_failed', username=body.username,
                             detail={'locked': True} if locked else None)
            raise HTTPException(401, 'Wrong username or password')
        ctx.throttle.succeeded(*keys)
        audit.record(conn, 'auth.login', user=user)
        return {'token': auth.issue_token(conn, user['id']), 'user': user_out(user)}

    @router.post('/auth/logout', status_code=204)
    def logout(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
               conn: sqlite3.Connection = Depends(get_db, scope='function')):
        if creds:
            user = auth.user_for_token(conn, creds.credentials)
            auth.revoke_token(conn, creds.credentials)
            if user is not None:
                audit.record(conn, 'auth.logout', user=user)
        return Response(status_code=204)

    @router.get('/auth/me')
    def me(user=Depends(current_user)):
        return user_out(user)

    # ── users ─────────────────────────────────────────────────────────────

    @router.get('/users')
    def list_users(_=Depends(admin_user), conn=Depends(get_db, scope='function')):
        return [user_out(r) for r in conn.execute('SELECT * FROM users ORDER BY username')]

    @router.post('/users', status_code=201)
    def create_user(body: UserIn, admin=Depends(admin_user), conn=Depends(get_db, scope='function')):
        try:
            cur = conn.execute(
                'INSERT INTO users (username, display_name, password_hash, role, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (body.username, body.display_name, auth.hash_password(body.password),
                 body.role, now()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f'Username {body.username!r} is taken')
        audit.record(conn, 'user.create', user=admin, target=f'user:{cur.lastrowid}',
                     detail={'username': body.username, 'role': body.role})
        return user_out(conn.execute('SELECT * FROM users WHERE id = ?',
                                     (cur.lastrowid,)).fetchone())

    @router.patch('/users/{user_id}')
    def update_user(user_id: int, body: UserPatch, admin=Depends(admin_user),
                    conn=Depends(get_db, scope='function')):
        u = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if u is None:
            raise HTTPException(404, 'User not found')
        loses_admin = (u['role'] == 'admin' and u['active']
                       and (body.role == 'labeller' or body.active is False))
        if loses_admin and active_admins(conn) <= 1:
            raise HTTPException(409, 'At least one active admin must remain')
        if body.display_name is not None:
            conn.execute('UPDATE users SET display_name = ? WHERE id = ?',
                         (body.display_name, user_id))
        if body.role is not None:
            conn.execute('UPDATE users SET role = ? WHERE id = ?', (body.role, user_id))
        if body.active is not None:
            conn.execute('UPDATE users SET active = ? WHERE id = ?', (int(body.active), user_id))
        if body.password is not None:
            conn.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                         (auth.hash_password(body.password), user_id))
        if body.password is not None or body.active is False:
            conn.execute('DELETE FROM tokens WHERE user_id = ?', (user_id,))
        changes = {k: v for k, v in body.model_dump(exclude_unset=True).items() if k != 'password'}
        if body.password is not None:
            changes['password_changed'] = True
        audit.record(conn, 'user.update', user=admin, target=f'user:{user_id}', detail=changes)
        return user_out(conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone())

    # ── projects ──────────────────────────────────────────────────────────

    @router.get('/projects')
    def list_projects(user=Depends(current_user), conn=Depends(get_db, scope='function')):
        if user['role'] == 'admin':
            rows = conn.execute('SELECT * FROM projects ORDER BY name')
        else:
            rows = conn.execute(
                'SELECT * FROM projects WHERE id IN '
                '(SELECT project_id FROM items WHERE assignee_id = ?) ORDER BY name',
                (user['id'],))
        return [project_out(r) for r in rows]

    @router.post('/projects', status_code=201)
    def create_project(body: ProjectIn, user=Depends(admin_user), conn=Depends(get_db, scope='function')):
        try:
            cur = conn.execute(
                'INSERT INTO projects (name, task_type, multi_label, protocol, created_by, '
                'created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (body.name, body.task_type, int(body.multi_label), body.protocol,
                 user['id'], now()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f'A project named {body.name!r} already exists')
        audit.record(conn, 'project.create', user=user, target=f'project:{cur.lastrowid}',
                     detail={'name': body.name, 'task_type': body.task_type})
        return project_out(conn.execute('SELECT * FROM projects WHERE id = ?',
                                        (cur.lastrowid,)).fetchone())

    @router.get('/projects/{project_id}')
    def read_project(project_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        return project_out(get_project(conn, project_id, user))

    @router.patch('/projects/{project_id}')
    def update_project(project_id: int, body: ProjectPatch, user=Depends(admin_user),
                       conn=Depends(get_db, scope='function')):
        get_project(conn, project_id, user)
        try:
            for col in ('name', 'protocol', 'multi_label'):
                value = getattr(body, col)
                if value is not None:
                    conn.execute(f'UPDATE projects SET {col} = ? WHERE id = ?',
                                 (int(value) if col == 'multi_label' else value, project_id))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f'A project named {body.name!r} already exists')
        audit.record(conn, 'project.update', user=user, target=f'project:{project_id}',
                     detail=body.model_dump(exclude_unset=True))
        return project_out(get_project(conn, project_id, user))

    @router.delete('/projects/{project_id}', status_code=204)
    def delete_project(project_id: int, user=Depends(admin_user), conn=Depends(get_db, scope='function')):
        p = get_project(conn, project_id, user)
        item_ids = [r['id'] for r in conn.execute('SELECT id FROM items WHERE project_id = ?',
                                                  (project_id,))]
        conn.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        audit.record(conn, 'project.delete', user=user, target=f'project:{project_id}',
                     detail={'name': p['name'], 'items': len(item_ids)})
        conn.commit()
        for item_id in item_ids:
            ctx.images.delete(item_id)
        return Response(status_code=204)

    @router.get('/projects/{project_id}/progress')
    def project_progress(project_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        get_project(conn, project_id, user)
        status = {s: 0 for s in ITEM_STATUSES}
        for r in conn.execute('SELECT status, COUNT(*) n FROM items WHERE project_id = ? '
                              'GROUP BY status', (project_id,)):
            status[r['status']] = r['n']
        per_labeller = [dict(r) for r in conn.execute(
            'SELECT u.id user_id, u.display_name, COUNT(i.id) items, '
            "SUM(i.status = 'done') done, "
            '(SELECT COUNT(*) FROM annotations a JOIN items i2 ON i2.id = a.item_id '
            ' WHERE i2.project_id = ? AND a.labeller_id = u.id) annotations '
            'FROM users u JOIN items i ON i.assignee_id = u.id AND i.project_id = ? '
            'GROUP BY u.id ORDER BY u.display_name', (project_id, project_id))]
        if user['role'] != 'admin':
            # Labellers see the project totals but only their own row.
            per_labeller = [r for r in per_labeller if r['user_id'] == user['id']]
        return {'items': sum(status.values()), 'by_status': status,
                'by_labeller': per_labeller}

    # ── classes ───────────────────────────────────────────────────────────

    @router.get('/projects/{project_id}/classes')
    def list_classes(project_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        get_project(conn, project_id, user)
        return [class_out(r) for r in conn.execute(
            'SELECT * FROM classes WHERE project_id = ? ORDER BY position, id', (project_id,))]

    @router.post('/projects/{project_id}/classes', status_code=201)
    def create_class(project_id: int, body: ClassIn, user=Depends(admin_user),
                     conn=Depends(get_db, scope='function')):
        p = get_project(conn, project_id, user)
        allowed = ALLOWED_LABEL_TYPES[p['task_type']]
        label_type = body.label_type or allowed[0]
        if label_type not in allowed:
            raise HTTPException(400, f"{p['task_type']} projects use "
                                     f"{', '.join(allowed)} labels, not {label_type!r}")
        position = body.position
        if position is None:
            position = conn.execute('SELECT COALESCE(MAX(position) + 1, 0) FROM classes '
                                    'WHERE project_id = ?', (project_id,)).fetchone()[0]
        try:
            cur = conn.execute(
                'INSERT INTO classes (project_id, name, color, label_type, position) '
                'VALUES (?, ?, ?, ?, ?)',
                (project_id, body.name, body.color.lower(), label_type, position))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f'Class {body.name!r} already exists in this project')
        audit.record(conn, 'class.create', user=user, target=f'class:{cur.lastrowid}',
                     detail={'project_id': project_id, 'name': body.name,
                             'label_type': label_type})
        return class_out(conn.execute('SELECT * FROM classes WHERE id = ?',
                                      (cur.lastrowid,)).fetchone())

    @router.patch('/classes/{class_id}')
    def update_class(class_id: int, body: ClassPatch, user=Depends(admin_user),
                     conn=Depends(get_db, scope='function')):
        if conn.execute('SELECT 1 FROM classes WHERE id = ?', (class_id,)).fetchone() is None:
            raise HTTPException(404, 'Class not found')
        try:
            for col in ('name', 'color', 'position'):
                value = getattr(body, col)
                if value is not None:
                    value = value.lower() if col == 'color' else value
                    conn.execute(f'UPDATE classes SET {col} = ? WHERE id = ?', (value, class_id))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f'Class {body.name!r} already exists in this project')
        audit.record(conn, 'class.update', user=user, target=f'class:{class_id}',
                     detail=body.model_dump(exclude_unset=True))
        return class_out(conn.execute('SELECT * FROM classes WHERE id = ?',
                                      (class_id,)).fetchone())

    @router.delete('/classes/{class_id}', status_code=204)
    def delete_class(class_id: int, force: bool = False, user=Depends(admin_user),
                     conn=Depends(get_db, scope='function')):
        if conn.execute('SELECT 1 FROM classes WHERE id = ?', (class_id,)).fetchone() is None:
            raise HTTPException(404, 'Class not found')
        used = conn.execute('SELECT COUNT(*) FROM annotations WHERE class_id = ?',
                            (class_id,)).fetchone()[0]
        if used and not force:
            raise HTTPException(409, f'{used} annotations use this class; '
                                     'pass force=true to delete them too')
        conn.execute('DELETE FROM classes WHERE id = ?', (class_id,))
        audit.record(conn, 'class.delete', user=user, target=f'class:{class_id}',
                     detail={'annotations_deleted': used})
        return Response(status_code=204)

    # ── items ─────────────────────────────────────────────────────────────

    def read_upload(upload: UploadFile) -> tuple[bytes, str, np.ndarray]:
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f'{upload.filename} is too large')
        try:
            return data, extension(upload.filename), decode(data)
        except ImageError as exc:
            raise HTTPException(400, f'{upload.filename}: {exc}')

    def insert_item(conn, project_id: int, kind: str, name: str,
                    parts: dict[str, tuple[bytes, str, np.ndarray]]) -> sqlite3.Row:
        h, w = next(iter(parts.values()))[2].shape[:2]
        cur = conn.execute(
            'INSERT INTO items (project_id, kind, name, width, height, files, created_at) '
            "VALUES (?, ?, ?, ?, ?, '{}', ?)", (project_id, kind, name, w, h, now()))
        item_id = cur.lastrowid
        try:
            files = {role: ctx.images.save(item_id, role, data, ext)
                     for role, (data, ext, _) in parts.items()}
        except OSError:
            ctx.images.delete(item_id)
            raise
        conn.execute('UPDATE items SET files = ? WHERE id = ?', (json.dumps(files), item_id))
        return conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()

    @router.post('/projects/{project_id}/items', status_code=201)
    def upload_items(project_id: int,
                     images: list[UploadFile] = File(default=[]),
                     before: Optional[UploadFile] = File(default=None),
                     after: Optional[UploadFile] = File(default=None),
                     name: Optional[str] = Form(default=None),
                     user=Depends(admin_user), conn=Depends(get_db, scope='function')):
        """Single-image projects take one or more ``images``; change
        detection takes one ``before`` / ``after`` pair per request."""
        p = get_project(conn, project_id, user)
        created = []
        if p['task_type'] == 'change_detection':
            if images or before is None or after is None:
                raise HTTPException(400, 'Change detection items need a before and an after image')
            b, a = read_upload(before), read_upload(after)
            if b[2].shape[:2] != a[2].shape[:2]:
                raise HTTPException(400, 'Before and after images must be the same size '
                                         f'({b[2].shape[1]}x{b[2].shape[0]} vs '
                                         f'{a[2].shape[1]}x{a[2].shape[0]})')
            item_name = name or Path(before.filename or 'pair').stem
            created.append(insert_item(conn, project_id, 'pair', item_name,
                                       {'before': b, 'after': a}))
        else:
            if not images or before is not None or after is not None:
                raise HTTPException(400, 'Upload one or more files in the "images" field')
            decoded = [(u.filename, read_upload(u)) for u in images]
            for filename, part in decoded:
                item_name = (name if len(decoded) == 1 and name else Path(filename or 'image').stem)
                created.append(insert_item(conn, project_id, 'single', item_name,
                                           {'image': part}))
        audit.record(conn, 'item.upload', user=user, target=f'project:{project_id}',
                     detail={'item_ids': [r['id'] for r in created]})
        return [item_out(r) for r in created]

    @router.get('/projects/{project_id}/items')
    def list_items(project_id: int, status: Optional[Literal[ITEM_STATUSES]] = None,
                   assignee_id: Optional[int] = None,
                   user=Depends(current_user), conn=Depends(get_db, scope='function')):
        get_project(conn, project_id, user)
        sql = ('SELECT items.*, (SELECT COUNT(*) FROM annotations a WHERE a.item_id = items.id) '
               'annotation_count FROM items WHERE project_id = ?')
        args = [project_id]
        if user['role'] != 'admin':
            sql, args = sql + ' AND assignee_id = ?', args + [user['id']]
        elif assignee_id is not None:
            sql, args = sql + ' AND assignee_id = ?', args + [assignee_id]
        if status is not None:
            sql, args = sql + ' AND status = ?', args + [status]
        return [item_out(r) for r in conn.execute(sql + ' ORDER BY id', args)]

    @router.post('/projects/{project_id}/assign')
    def assign_items(project_id: int, body: AssignIn, user=Depends(admin_user),
                     conn=Depends(get_db, scope='function')):
        get_project(conn, project_id, user)
        check_assignee(conn, body.user_id)
        marks = ','.join('?' * len(body.item_ids))
        found = conn.execute(f'SELECT COUNT(*) FROM items WHERE project_id = ? AND id IN ({marks})',
                             [project_id, *body.item_ids]).fetchone()[0]
        if found != len(set(body.item_ids)):
            raise HTTPException(400, 'Some items are not in this project')
        conn.execute(f'UPDATE items SET assignee_id = ? WHERE id IN ({marks})',
                     [body.user_id, *body.item_ids])
        audit.record(conn, 'item.assign', user=user, target=f'project:{project_id}',
                     detail={'assignee_id': body.user_id, 'item_ids': sorted(set(body.item_ids))})
        return {'assigned': len(set(body.item_ids))}

    @router.get('/items/{item_id}')
    def read_item(item_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        return item_out(get_item(conn, item_id, user))

    @router.patch('/items/{item_id}')
    def update_item(item_id: int, body: ItemPatch, user=Depends(current_user),
                    conn=Depends(get_db, scope='function')):
        get_item(conn, item_id, user)
        if 'assignee_id' in body.model_fields_set:
            if user['role'] != 'admin':
                raise HTTPException(403, 'Only admins assign items')
            check_assignee(conn, body.assignee_id)
            conn.execute('UPDATE items SET assignee_id = ? WHERE id = ?',
                         (body.assignee_id, item_id))
        if body.status is not None:
            conn.execute('UPDATE items SET status = ? WHERE id = ?', (body.status, item_id))
        audit.record(conn, 'item.update', user=user, target=f'item:{item_id}',
                     detail=body.model_dump(exclude_unset=True))
        return item_out(conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone())

    @router.delete('/items/{item_id}', status_code=204)
    def delete_item(item_id: int, user=Depends(admin_user), conn=Depends(get_db, scope='function')):
        it = get_item(conn, item_id, user)
        conn.execute('DELETE FROM items WHERE id = ?', (item_id,))
        audit.record(conn, 'item.delete', user=user, target=f'item:{item_id}',
                     detail={'project_id': it['project_id'], 'name': it['name']})
        conn.commit()
        ctx.images.delete(item_id)
        return Response(status_code=204)

    @router.get('/items/{item_id}/image/{role}')
    def item_image(item_id: int, role: str, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        files = json.loads(get_item(conn, item_id, user)['files'])
        if role not in files:
            raise HTTPException(404, f'This item has no {role!r} image')
        data, media = ctx.images.for_browser(item_id, files[role])
        return Response(data, media_type=media)

    # ── assisted labelling ────────────────────────────────────────────────

    @router.post('/items/{item_id}/assist', status_code=201)
    def start_assist(item_id: int, image: str = 'image', user=Depends(current_user),
                     conn=Depends(get_db, scope='function')):
        """Open an interactive segmentation session on one of the item's
        images (``image``, or ``before`` / ``after`` for pairs)."""
        it = get_item(conn, item_id, user)
        files = json.loads(it['files'])
        if image not in files:
            raise HTTPException(400, f"Choose one of: {', '.join(files)}")
        session = ctx.manager.create_session(ctx.images.load_rgb(item_id, files[image]),
                                             f"{it['name']}_{image}", owner_id=user['id'])
        ctx.assist_sessions = {sid: v for sid, v in ctx.assist_sessions.items()
                               if ctx.manager.has_session(sid)}
        ctx.assist_sessions[session.id] = (item_id, user['id'])
        return session.state()

    def mask_from_session(sid: str, item, user) -> np.ndarray:
        owner = ctx.assist_sessions.get(sid)
        if owner != (item['id'], user['id']):
            raise HTTPException(400, 'from_session must be an assist session you opened '
                                     'on this item')
        session = ctx.manager.get(sid)
        with ctx.manager.lock:
            mask = session.controller.result_mask > 0
            h, w = item['height'], item['width']
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            # Start the next object from a clean slate (keeps SAM's embedding).
            session.controller.clear_objects()
        return mask

    # ── annotations ───────────────────────────────────────────────────────

    @router.get('/items/{item_id}/annotations')
    def list_annotations(item_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        get_item(conn, item_id, user)
        return [annotation_out(r) for r in conn.execute(
            'SELECT * FROM annotations WHERE item_id = ? ORDER BY id', (item_id,))]

    @router.post('/items/{item_id}/annotations', status_code=201)
    def create_annotation(item_id: int, body: AnnotationIn, user=Depends(current_user),
                          conn=Depends(get_db, scope='function')):
        """Give either ``geometry`` or ``from_session`` (tags need neither)."""
        it = get_item(conn, item_id, user)
        p = conn.execute('SELECT * FROM projects WHERE id = ?', (it['project_id'],)).fetchone()
        cls = get_class(conn, body.class_id, p['id'])
        if (body.geometry is None) == (body.from_session is None) and cls['label_type'] != 'tag':
            raise HTTPException(400, 'Give either geometry or from_session')
        try:
            if body.from_session is not None:
                geom = geometry.mask_to_geometry(mask_from_session(body.from_session, it, user),
                                                 cls['label_type'])
                source, model = 'assisted', json.dumps(ctx.manager.provenance())
            else:
                geom = geometry.validate(body.geometry or {'type': 'tag'}, cls['label_type'],
                                         it['width'], it['height'])
                source, model = 'manual', None
        except geometry.GeometryError as exc:
            raise HTTPException(400, str(exc))

        if cls['label_type'] == 'tag':
            same = conn.execute('SELECT 1 FROM annotations WHERE item_id = ? AND class_id = ?',
                                (item_id, cls['id'])).fetchone()
            if same:
                raise HTTPException(409, f"The item is already tagged {cls['name']!r}")
            if not p['multi_label']:
                # Single-label classification: the new tag replaces the old one.
                conn.execute('DELETE FROM annotations WHERE item_id = ?', (item_id,))

        ts = now()
        cur = conn.execute(
            'INSERT INTO annotations (item_id, class_id, labeller_id, geometry, source, model, '
            'created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (item_id, cls['id'], user['id'], json.dumps(geom), source, model, ts, ts))
        audit.record(conn, 'annotation.create', user=user, target=f'annotation:{cur.lastrowid}',
                     detail={'item_id': item_id, 'class_id': cls['id'], 'source': source})
        if it['status'] == 'todo':
            conn.execute("UPDATE items SET status = 'in_progress' WHERE id = ?", (item_id,))
        return annotation_out(conn.execute('SELECT * FROM annotations WHERE id = ?',
                                           (cur.lastrowid,)).fetchone())

    @router.patch('/annotations/{ann_id}')
    def update_annotation(ann_id: int, body: AnnotationPatch, user=Depends(current_user),
                          conn=Depends(get_db, scope='function')):
        a, it = get_annotation(conn, ann_id, user)
        class_id = body.class_id if body.class_id is not None else a['class_id']
        cls = get_class(conn, class_id, it['project_id'])
        geom = body.geometry if body.geometry is not None else json.loads(a['geometry'])
        try:
            geom = geometry.validate(geom, cls['label_type'], it['width'], it['height'])
        except geometry.GeometryError as exc:
            raise HTTPException(400, str(exc))
        conn.execute('UPDATE annotations SET class_id = ?, geometry = ?, updated_at = ? '
                     'WHERE id = ?', (class_id, json.dumps(geom), now(), ann_id))
        audit.record(conn, 'annotation.update', user=user, target=f'annotation:{ann_id}',
                     detail={'item_id': it['id'], 'class_id': class_id, 'source': a['source'],
                             'geometry_changed': body.geometry is not None})
        return annotation_out(conn.execute('SELECT * FROM annotations WHERE id = ?',
                                           (ann_id,)).fetchone())

    @router.delete('/annotations/{ann_id}', status_code=204)
    def delete_annotation(ann_id: int, user=Depends(current_user), conn=Depends(get_db, scope='function')):
        a, _ = get_annotation(conn, ann_id, user)
        conn.execute('DELETE FROM annotations WHERE id = ?', (ann_id,))
        audit.record(conn, 'annotation.delete', user=user, target=f'annotation:{ann_id}',
                     detail={'item_id': a['item_id'], 'class_id': a['class_id'],
                             'source': a['source']})
        return Response(status_code=204)

    # ── exports ───────────────────────────────────────────────────────────

    @router.get('/projects/{project_id}/export')
    def export_project(project_id: int, format: Literal[export.FORMATS],
                       status: Literal[('all', *ITEM_STATUSES)] = 'all',
                       include_images: bool = True,
                       user=Depends(admin_user), conn=Depends(get_db, scope='function')):
        """Download the project's labels: a CSV file, or a zip holding COCO
        JSON, PNG masks or YOLO labels (plus the images unless
        ``include_images`` is false). ``status`` limits it to items with that
        status."""
        p = get_project(conn, project_id, user)
        if format not in export.TASK_FORMATS[p['task_type']]:
            raise HTTPException(400, f"{p['task_type']} projects export as "
                                     f"{', '.join(export.TASK_FORMATS[p['task_type']])}")
        sql, args = 'SELECT * FROM items WHERE project_id = ?', [project_id]
        if status != 'all':
            sql, args = sql + ' AND status = ?', args + [status]
        items = [{**dict(r), 'files': json.loads(r['files'])}
                 for r in conn.execute(sql + ' ORDER BY id', args)]
        annotations: dict[int, list[dict]] = {}
        for r in conn.execute('SELECT a.* FROM annotations a JOIN items i ON i.id = a.item_id '
                              'WHERE i.project_id = ? ORDER BY a.id', (project_id,)):
            annotations.setdefault(r['item_id'], []).append(
                {**dict(r), 'geometry': json.loads(r['geometry'])})
        data = export.ExportData(
            project=dict(p),
            classes=[dict(r) for r in conn.execute(
                'SELECT * FROM classes WHERE project_id = ? ORDER BY position, id', (project_id,))],
            items=items, annotations=annotations,
            users={r['id']: r['display_name'] for r in conn.execute('SELECT * FROM users')})
        audit.record(conn, 'project.export', user=user, target=f'project:{project_id}',
                     detail={'format': format, 'status': status, 'include_images': include_images,
                             'items': len(items)})
        stem = f"{export.safe_name(p['name'])}_{format}"
        if format == 'csv':
            return Response(export.export_csv(data), media_type='text/csv',
                            headers={'Content-Disposition': f'attachment; filename="{stem}.csv"'})
        fd, tmp = tempfile.mkstemp(suffix='.zip')
        os.close(fd)
        try:
            export.write_zip(Path(tmp), format, data, ctx.images, include_images)
        except BaseException:
            os.unlink(tmp)
            raise
        return FileResponse(tmp, media_type='application/zip', filename=f'{stem}.zip',
                            background=BackgroundTask(os.unlink, tmp))

    # ── audit log ─────────────────────────────────────────────────────────

    @router.get('/audit')
    def read_audit(limit: int = 200, before_id: Optional[int] = None, action: Optional[str] = None,
                   user_id: Optional[int] = None, _=Depends(admin_user),
                   conn=Depends(get_db, scope='function')):
        """Newest first; page with ``before_id``. ``action`` matches a prefix
        (``auth.`` for every sign-in event)."""
        sql, args = 'SELECT * FROM audit_log WHERE 1 = 1', []
        if before_id is not None:
            sql, args = sql + ' AND id < ?', args + [before_id]
        if action:
            sql, args = sql + ' AND substr(action, 1, ?) = ?', args + [len(action), action]
        if user_id is not None:
            sql, args = sql + ' AND user_id = ?', args + [user_id]
        limit = max(1, min(limit, 1000))
        return [audit.row_out(r) for r in conn.execute(sql + ' ORDER BY id DESC LIMIT ?',
                                                        args + [limit])]

    @router.get('/audit/verify')
    def verify_audit(_=Depends(admin_user), conn=Depends(get_db, scope='function')):
        return audit.verify(conn)

    return router, ctx
