"""
End-to-end check of the web app through its real UI (Playwright, headless
Chromium), against a fresh server instance with an empty data folder.

It signs in, goes through every screen and control the app offers (users,
projects of all five task types, classes, uploads, assignment, the labelling
workspace with every tool, quick segment with every model and ensemble,
project-model training of every kind including a cancel, auto-labelling,
exports, settings, audit log, and the labeller's view), and checks the result
of each step through the API. Every check is reported PASS / FAIL / SKIP, with
a screenshot for each failure; one failure does not stop the others.

Start it with deploy/e2e_check.sh, which runs a separate server instance on
its own port and data folder, so live users and data are not touched. Needs
the ``playwright`` package and its Chromium (pip install playwright;
python -m playwright install chromium), in any Python; nothing else from this
repository.

    python tools/e2e_check.py --url http://127.0.0.1:8021 --setup-token TOKEN \\
        --out /tmp/e2e-report [--no-train] [--steps 20] [--train-timeout 1800]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

PASSWORD = 'e2e-check-password-1'
LAB_PASSWORD = 'e2e-labeller-pass-1'
W, H = 320, 240                 # synthetic image size
UI_TIMEOUT = 20_000             # ms, ordinary UI waits
MODEL_TIMEOUT = 180_000         # ms, first click after a model switch (CPU)


# ---------------------------------------------------------------------------
# Synthetic images (pure Python PNG, no imaging packages needed)
# ---------------------------------------------------------------------------

def _png(width: int, height: int, rgb_rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
    raw = b''.join(b'\x00' + row for row in rgb_rows)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def make_image(seed: int, n_blobs: int, shift: int = 0) -> tuple[bytes, list[tuple[int, int, int, int]]]:
    """A light, slightly noisy background with ``n_blobs`` dark ellipses that
    do not overlap. Returns the PNG and the ellipses as (cx, cy, ax, ay)."""
    rng = random.Random(seed)
    blobs: list[tuple[int, int, int, int]] = []
    while len(blobs) < n_blobs:
        ax, ay = rng.randint(28, 46), rng.randint(24, 40)
        cx = rng.randint(ax + 12, W - ax - 12) + shift
        cy = rng.randint(ay + 12, H - ay - 12)
        cx = max(ax + 4, min(W - ax - 4, cx))
        if all(abs(cx - x) > ax + bx + 10 or abs(cy - y) > ay + by + 10 for x, y, bx, by in blobs):
            blobs.append((cx, cy, ax, ay))
    base = (rng.randint(200, 225), rng.randint(165, 190), rng.randint(150, 170))
    rows = []
    for y in range(H):
        row = bytearray()
        for x in range(W):
            shade = 1.0
            for cx, cy, ax, ay in blobs:
                d = ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
                if d < 1.3:
                    shade = min(shade, 0.25 + 0.75 * max(0.0, (d - 0.7) / 0.6))
            n = rng.randint(-6, 6)
            row += bytes(max(0, min(255, int(c * shade) + n + (x + y) // 40)) for c in base)
        rows.append(bytes(row))
    return _png(W, H, rows), blobs


# ---------------------------------------------------------------------------
# Runner: checks, screenshots, report
# ---------------------------------------------------------------------------

class Skip(Exception):
    pass


class Run:
    def __init__(self, out: Path, base: str):
        self.out = out
        self.base = base.rstrip('/')
        self.results: list[dict] = []
        self.page: Page | None = None
        self.token: str | None = None
        self.started = time.time()

    @contextmanager
    def check(self, name: str):
        t = time.time()
        print(f'  .. {name}', flush=True)
        try:
            yield
        except Skip as exc:
            self._add(name, 'SKIP', str(exc), t)
        except Exception as exc:  # noqa: BLE001 - every failure is reported, then the run goes on
            shot = None
            if self.page is not None:
                shot = self.out / f'fail-{len(self.results) + 1:03d}.png'
                try:
                    self.page.screenshot(path=str(shot), full_page=True)
                except Exception:  # noqa: BLE001
                    shot = None
            detail = f'{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ""}'
            self._add(name, 'FAIL', detail, t, shot,
                      traceback.format_exc(limit=4))
        else:
            self._add(name, 'PASS', '', t)

    def _add(self, name, status, detail, t, shot=None, tb=None):
        self.results.append({'check': name, 'status': status, 'detail': detail,
                             'seconds': round(time.time() - t, 1),
                             'screenshot': shot.name if shot else None, 'traceback': tb})
        mark = {'PASS': '+', 'FAIL': 'x', 'SKIP': '~'}[status]
        print(f'  [{mark}] {name}' + (f'  ({detail})' if detail else ''), flush=True)

    # ── API, as the signed-in browser user ─────────────────────────────────
    def api(self, method: str, path: str, expect_status: int | None = 200, **kw):
        headers = {'Authorization': f'Bearer {self.token}'} if self.token else {}
        r = self.page.request.fetch(f'{self.base}/api{path}', method=method, headers=headers, **kw)
        if expect_status is not None and r.status != expect_status:
            raise AssertionError(f'{method} {path} -> {r.status}: {r.text()[:300]}')
        ctype = r.headers.get('content-type', '')
        return r.json() if 'json' in ctype else r.body()

    def refresh_token(self):
        self.token = self.page.evaluate("() => localStorage.getItem('iseg.token')")

    def report(self) -> int:
        counts = {s: sum(r['status'] == s for r in self.results) for s in ('PASS', 'FAIL', 'SKIP')}
        (self.out / 'report.json').write_text(json.dumps(
            {'counts': counts, 'minutes': round((time.time() - self.started) / 60, 1),
             'results': self.results}, indent=1))
        lines = [f"{r['status']:4s}  {r['check']}" + (f"  -- {r['detail']}" if r['detail'] else '')
                 + (f"  [{r['screenshot']}]" if r['screenshot'] else '') for r in self.results]
        (self.out / 'report.txt').write_text('\n'.join(lines) + '\n')
        print('\n' + '=' * 70)
        for r in self.results:
            if r['status'] != 'PASS':
                print(f"{r['status']:4s}  {r['check']}  -- {r['detail']}")
        print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
              f"in {(time.time() - self.started) / 60:.1f} min. Report: {self.out}/report.txt")
        return 1 if counts['FAIL'] else 0


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def button(page: Page, name: str, exact: bool = True):
    return page.get_by_role('button', name=name, exact=exact)


def canvas_point(page: Page, sel: str, x: float, y: float, w: int = W, h: int = H):
    box = page.locator(sel).first.bounding_box()
    assert box, f'{sel} not visible'
    return box['x'] + (x + 0.5) / w * box['width'], box['y'] + (y + 0.5) / h * box['height']


def click_canvas(page: Page, sel: str, x: float, y: float, right: bool = False, **kw):
    idle(page)
    px, py = canvas_point(page, sel, x, y, **kw)
    page.mouse.click(px, py, button='right' if right else 'left')


def drag_canvas(page: Page, sel: str, x0, y0, x1, y1, **kw):
    idle(page)
    a, b = canvas_point(page, sel, x0, y0, **kw), canvas_point(page, sel, x1, y1, **kw)
    # Paced like a person: the canvas keeps the drag in React state, which
    # only updates between events once the page has rendered.
    page.mouse.move(*a)
    page.mouse.down()
    page.wait_for_timeout(60)
    for i in range(1, 9):
        page.mouse.move(a[0] + (b[0] - a[0]) * i / 8, a[1] + (b[1] - a[1]) * i / 8)
        page.wait_for_timeout(40)
    page.mouse.up()


def blur(page: Page):
    """Keyboard shortcuts are ignored while a form field has the focus."""
    page.evaluate('() => document.activeElement && document.activeElement.blur()')


def idle(page: Page):
    """Wait until the page has no request in flight: both canvases ignore
    clicks while busy, and every busy-dependent button is disabled then.
    Quick segment: 'Load image…'; workspace: 'Mark done & next' / 'Reopen'."""
    probe = page.get_by_role('button', name='Load image…').or_(
        page.get_by_role('button', name='Mark done & next')).or_(
        page.get_by_role('button', name='Reopen'))
    expect(probe.first).to_be_enabled(timeout=MODEL_TIMEOUT)


def key(page: Page, k: str):
    blur(page)
    page.keyboard.press(k)


def goto(run: Run, hash_: str):
    """Open a page fresh: a hash-only change to the current URL would not reload it."""
    url = f'{run.base}/#{hash_}'
    if run.page.url == url:
        run.page.reload()
    else:
        run.page.goto(url)
    run.page.wait_for_load_state('domcontentloaded')


def no_error(page: Page):
    """The app shows failures in .error-text, and in the status footer of the
    workspace and quick segment (footer.status.error); none may be visible."""
    errs = [t for t in page.locator('.error-text, footer.status.error').all_inner_texts() if t.strip()]
    assert not errs, f'UI error: {errs[0]}'


ANNOT = '.annot-stage canvas'


def class_row(page: Page, name: str):
    """A row of the classes editor. Its name field is uncontrolled, so its value
    attribute is the saved name; the add form's field must not match."""
    return page.locator('ul.class-editor > li').filter(
        has=page.locator(f'input.grow[value="{name}"]'))
SEG = 'canvas.seg-canvas'


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

class E2E:
    def __init__(self, run: Run, args, browser):
        self.run, self.args, self.browser = run, args, browser
        self.page = run.page
        self.images: dict[str, list] = {}       # file name -> ellipses
        self.files = run.out / 'images'
        self.files.mkdir(exist_ok=True)
        self.projects: dict[str, int] = {}
        self.users: dict[str, int] = {}

    # ── data ──────────────────────────────────────────────────────────────
    def write_images(self, prefix: str, n: int, blobs=2, seed0=0, shift=0) -> list[Path]:
        paths = []
        for i in range(n):
            data, el = make_image(seed0 + i, blobs if isinstance(blobs, int) else blobs[i], shift)
            p = self.files / f'{prefix}_{i + 1:02d}.png'
            p.write_bytes(data)
            self.images[p.name] = el
            paths.append(p)
        return paths

    def items(self, project: str) -> list[dict]:
        return self.run.api('GET', f'/projects/{self.projects[project]}/items')

    def anns(self, item_id: int) -> list[dict]:
        return self.run.api('GET', f'/items/{item_id}/annotations')

    # ── setup and accounts ────────────────────────────────────────────────
    def setup_admin(self):
        p, r = self.page, self.run
        with r.check('Setup screen creates the first admin'):
            goto(r, '/')
            expect(p.get_by_text('Create the first admin')).to_be_visible(timeout=UI_TIMEOUT)
            expect(button(p, 'Create admin')).to_be_disabled()
            p.get_by_label('Setup token').fill(self.args.setup_token)
            p.get_by_label('Username').fill('e2e_admin')
            p.get_by_label('Display name').fill('E2E Admin')
            p.get_by_label('Password').fill('short')
            expect(button(p, 'Create admin')).to_be_disabled()
            p.get_by_label('Password').fill(PASSWORD)
            button(p, 'Create admin').click()
            expect(p.get_by_text('E2E Admin (admin)')).to_be_visible(timeout=UI_TIMEOUT)
            expect(p.get_by_text('No projects yet')).to_be_visible()
            r.refresh_token()
            assert r.api('GET', '/auth/status')['needs_setup'] is False

        with r.check('Sign out, wrong password is refused, sign in again'):
            button(p, 'Sign out').click()
            expect(p.get_by_role('heading', name='Sign in')).to_be_visible(timeout=UI_TIMEOUT)
            p.get_by_label('Username').fill('e2e_admin')
            p.get_by_label('Password').fill('not-the-password')
            button(p, 'Sign in').click()
            expect(p.locator('.error-text')).to_be_visible(timeout=UI_TIMEOUT)
            p.get_by_label('Password').fill(PASSWORD)
            button(p, 'Sign in').click()
            expect(p.get_by_text('E2E Admin (admin)')).to_be_visible(timeout=UI_TIMEOUT)
            r.refresh_token()

    def users_page(self):
        p, r = self.page, self.run
        with r.check('Users: add two labellers'):
            p.get_by_role('link', name='Users').click()
            for name, user in (('E2E Labeller', 'e2e_lab'), ('E2E Second', 'e2e_lab2')):
                form = p.locator('form, section, .card').filter(has=button(p, 'Add user')).last
                form.get_by_label('Display name').fill(name)
                form.get_by_label('Username').fill(user)
                form.get_by_label('Password').fill(LAB_PASSWORD)
                button(p, 'Add user').click()
                expect(p.get_by_text(f'Created {name}.')).to_be_visible(timeout=UI_TIMEOUT)
            users = {u['username']: u for u in r.api('GET', '/users')}
            assert users['e2e_lab']['role'] == 'labeller' and users['e2e_lab2']['active']
            self.users = {k: v['id'] for k, v in users.items()}

        row = p.locator('tr', has_text='e2e_lab2')
        with r.check('Users: change role to admin and back'):
            row.locator('select').select_option('admin')
            expect(p.get_by_text('E2E Second is now an admin')).to_be_visible(timeout=UI_TIMEOUT)
            row.locator('select').select_option('labeller')
            expect(p.get_by_text('E2E Second is now a labeller')).to_be_visible(timeout=UI_TIMEOUT)
            assert {u['username']: u['role'] for u in r.api('GET', '/users')}['e2e_lab2'] == 'labeller'

        with r.check('Users: reset a password'):
            row.get_by_role('button', name='Reset password').click()
            row.get_by_placeholder('New password (12+)').fill(LAB_PASSWORD + 'x')
            row.get_by_role('button', name='Set').click()
            expect(p.get_by_text('Password changed for E2E Second')).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Users: deactivate and reactivate'):
            row.get_by_role('button', name='Deactivate').click()
            row.get_by_role('button', name='Yes').click()
            expect(p.get_by_text('E2E Second was deactivated.')).to_be_visible(timeout=UI_TIMEOUT)
            row.get_by_role('button', name='Reactivate').click()
            expect(row.get_by_role('button', name='Deactivate')).to_be_visible(timeout=UI_TIMEOUT)
            assert {u['username']: u['active'] for u in r.api('GET', '/users')}['e2e_lab2']

    # ── projects ──────────────────────────────────────────────────────────
    def new_project(self, key: str, name: str, task_label: str, protocol: str = '',
                    multi: bool = False):
        p, r = self.page, self.run
        goto(r, '/')
        button(p, 'New project').click()
        p.get_by_label('Name').fill(name)
        p.locator('label.task-option', has_text=task_label).click()   # the radio itself is hidden
        if multi:
            p.get_by_label('Multi-label: an image can have several classes').check()
        if protocol:
            p.locator('label', has_text='Protocol').locator('textarea').fill(protocol)
        button(p, 'Create project').click()
        p.wait_for_url('**/classes', timeout=UI_TIMEOUT)
        pid = next(x['id'] for x in r.api('GET', '/projects') if x['name'] == name)
        self.projects[key] = pid
        return pid

    def add_class(self, name: str, label_type: str | None = None):
        p = self.page
        # The form clears itself after the previous add; wait for that first.
        expect(p.get_by_placeholder('New class name')).to_have_value('', timeout=UI_TIMEOUT)
        p.get_by_placeholder('New class name').fill(name)
        sel = p.get_by_title('Label type')
        if label_type and sel.count():
            sel.select_option(label_type)
        button(p, 'Add class').click()
        expect(class_row(p, name)).to_have_count(1, timeout=UI_TIMEOUT)

    def upload(self, paths: list[Path]):
        p = self.page
        p.locator('input[type=file]').first.set_input_files([str(x) for x in paths])
        expect(p.get_by_text(f'{len(paths)} image{"s" if len(paths) > 1 else ""} ready to upload')).to_be_visible()
        button(p, 'Upload').click()
        expect(p.get_by_text(f'Uploaded {len(paths)} of {len(paths)} images.')).to_be_visible(timeout=120_000)

    def project_a(self):
        """Instance segmentation: classes, uploads, assignment, deletion."""
        p, r = self.page, self.run
        with r.check('Projects: create an instance-segmentation project'):
            self.new_project('A', 'E2E instance', 'Instance segmentation',
                             protocol='Outline every dark blob.')
            expect(p.get_by_text('No classes yet')).to_be_visible()

        cid = lambda n: next(c['id'] for c in r.api('GET', f'/projects/{self.projects["A"]}/classes')
                             if c['name'] == n)
        with r.check('Classes: add mask and polygon classes'):
            self.add_class('blob', 'mask')
            self.add_class('outline', 'polygon')
            self.add_class('temp', 'mask')
            cls = {c['name']: c['label_type'] for c in r.api('GET', f'/projects/{self.projects["A"]}/classes')}
            assert cls == {'blob': 'mask', 'outline': 'polygon', 'temp': 'mask'}, cls

        with r.check('Classes: rename and recolour save on leaving the field'):
            box = class_row(p, 'temp').locator('input.grow')
            box.fill('temp2')
            box.press('Tab')
            colour = class_row(p, 'temp2') \
                .locator('input[type=color]').first
            colour.fill('#123456')
            colour.blur()
            p.wait_for_timeout(800)
            c = next(c for c in r.api('GET', f'/projects/{self.projects["A"]}/classes') if c['id'] == cid('temp2'))
            assert c['color'].lower() == '#123456', c

        with r.check('Classes: move up and down'):
            order = lambda: [c['name'] for c in sorted(r.api('GET', f'/projects/{self.projects["A"]}/classes'),
                                                        key=lambda c: c['position'])]
            row = class_row(p, 'temp2')
            row.get_by_title('Move up').click()
            p.wait_for_timeout(800)
            assert order() == ['blob', 'temp2', 'outline'], order()
            row = class_row(p, 'temp2')
            row.get_by_title('Move down').click()
            p.wait_for_timeout(800)
            assert order() == ['blob', 'outline', 'temp2'], order()

        with r.check('Classes: delete a class with annotations ("Delete them too")'):
            # A labelled class first refuses, then deletes with its annotations.
            self.upload_in('A', self.write_images('a_tmp', 1, blobs=1, seed0=900))
            item = self.items('A')[0]['id']
            r.api('POST', f'/items/{item}/annotations', expect_status=201,
                  data={'class_id': cid('outline'), 'geometry': {
                      'type': 'polygon', 'points': [[10, 10], [60, 10], [60, 50], [10, 50]]}})
            goto(r, f'/projects/{self.projects["A"]}/classes')
            row = class_row(p, 'outline')
            row.get_by_role('button', name='Delete').click()
            row.get_by_role('button', name='Yes').click()
            expect(p.get_by_text('Has annotations.')).to_be_visible(timeout=UI_TIMEOUT)
            button(p, 'Cancel').click()
            row = class_row(p, 'temp2')
            row.get_by_role('button', name='Delete').click()
            row.get_by_role('button', name='Yes').click()
            expect(class_row(p, 'temp2')).to_have_count(0, timeout=UI_TIMEOUT)
            # Now force-delete the polygon class with its annotation, then re-add it.
            row = class_row(p, 'outline')
            row.get_by_role('button', name='Delete').click()
            row.get_by_role('button', name='Yes').click()
            button(p, 'Delete them too').click()
            expect(class_row(p, 'outline')).to_have_count(0, timeout=UI_TIMEOUT)
            assert self.anns(item) == []
            self.add_class('outline', 'polygon')
            r.api('DELETE', f'/items/{item}', expect_status=204)

        with r.check('Images: upload 15 images through the drop zone'):
            for it in self.items('A'):      # leftovers of a failed check above
                r.api('DELETE', f"/items/{it['id']}", expect_status=204)
            goto(r, f'/projects/{self.projects["A"]}')
            self.upload(self.write_images('a', 15, blobs=[1 + (i % 2) for i in range(15)], seed0=10))
            assert len(self.items('A')) == 15

        with r.check('Images: status filter buttons'):
            for label in ('All (15)', 'todo (15)', 'in progress (0)', 'done (0)'):
                button(p, label).click()
            button(p, 'All (15)').click()
            no_error(p)

        lab = self.users.get('e2e_lab')
        with r.check('Images: select, assign, unassign, spread, delete'):
            # Each action clears the selection, so every step selects again.
            rows = p.locator('tbody tr')
            select_all = p.get_by_title('Select all shown')
            rows.nth(0).click()
            rows.nth(1).click()
            expect(p.get_by_text('2 selected')).to_be_visible()
            p.locator('.assign-bar select').select_option(label='E2E Labeller')
            button(p, 'Assign').click()
            expect(p.get_by_text('Assigned 2 items to E2E Labeller')).to_be_visible(timeout=UI_TIMEOUT)
            assert sum(i['assignee_id'] == lab for i in self.items('A')) == 2
            select_all.check()
            button(p, 'Unassign').click()
            expect(p.get_by_text('to nobody')).to_be_visible(timeout=UI_TIMEOUT)
            assert all(i['assignee_id'] is None for i in self.items('A'))
            select_all.check()
            p.locator('.assign-bar summary', has_text='Spread evenly').click()
            spread = p.locator('.assign-bar details')
            for name in ('E2E Labeller', 'E2E Second'):
                spread.get_by_label(name).check()
            spread.get_by_role('button', name='Spread', exact=False).click()
            expect(p.get_by_text('Spread 15 items over 2 labellers')).to_be_visible(timeout=UI_TIMEOUT)
            got = [i['assignee_id'] for i in self.items('A')]
            assert got.count(lab) in (7, 8) and got.count(self.users['e2e_lab2']) in (7, 8), got
            expect(p.get_by_text('Select images to assign or delete them.')).to_be_visible()
            rows.last.click()
            expect(p.get_by_text('1 selected')).to_be_visible()
            button(p, 'Delete').click()
            button(p, 'Yes').click()
            expect(p.get_by_text('Deleted 1 item', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
            assert len(self.items('A')) == 14

        with r.check('Images: assignee filter'):
            ids = [i['id'] for i in self.items('A')]
            r.api('POST', f'/projects/{self.projects["A"]}/assign', data={'user_id': None, 'item_ids': ids})
            r.api('POST', f'/projects/{self.projects["A"]}/assign', data={'user_id': lab, 'item_ids': ids[-2:]})
            goto(r, f'/projects/{self.projects["A"]}')
            sel = p.locator('select').filter(has=p.locator('option[value="any"]'))
            sel.select_option(label='E2E Labeller')
            expect(p.locator('tbody tr')).to_have_count(2, timeout=UI_TIMEOUT)
            sel.select_option('none')
            expect(p.locator('tbody tr')).to_have_count(12, timeout=UI_TIMEOUT)
            sel.select_option('any')

        with r.check('Settings, Team and Export tabs open'):
            for tab in ('Team', 'Export', 'Settings', 'Classes', 'Images'):
                p.get_by_role('link', name=tab, exact=True).click()
                no_error(p)
            p.get_by_role('link', name='Team', exact=True).click()
            expect(p.locator('td', has_text='E2E Labeller').first).to_be_visible(timeout=UI_TIMEOUT)

    def upload_in(self, key: str, paths: list[Path]):
        goto(self.run, f'/projects/{self.projects[key]}')
        self.upload(paths)

    # ── workspace ─────────────────────────────────────────────────────────
    def open_item(self, item_id: int):
        p = self.page
        goto(self.run, f'/items/{item_id}')
        expect(p.locator('.item-head')).to_be_visible(timeout=UI_TIMEOUT)

    def wait_model_ready(self):
        expect(self.page.get_by_text('model ready', exact=False)).to_be_visible(timeout=MODEL_TIMEOUT)

    def assist_object(self, item: dict, blob, save_with_enter: bool = False):
        """Click inside one ellipse and save it under the current class."""
        p = self.page
        cx, cy = blob[0], blob[1]
        click_canvas(p, ANNOT, cx, cy)
        save = p.get_by_role('button', name='Save', exact=False).filter(has_text='(Enter)')
        expect(save).to_be_enabled(timeout=MODEL_TIMEOUT)
        n = len(self.anns(item['id']))
        if save_with_enter:
            key(p, 'Enter')
        else:
            save.click()
        deadline = time.time() + 30
        while len(self.anns(item['id'])) == n and time.time() < deadline:
            p.wait_for_timeout(300)
        assert len(self.anns(item['id'])) == n + 1, 'annotation was not saved'

    def label_project_a(self):
        """Label 12 of the 14 images with assisted clicks and mark them done."""
        p, r = self.page, self.run
        items = sorted(self.items('A'), key=lambda i: i['id'])
        self.a_items = items
        first = items[0]

        with r.check('Workspace: model session starts, Assist is the default tool'):
            self.open_item(first['id'])
            self.wait_model_ready()
            expect(p.get_by_title('Assist (A)')).to_have_class('active')
            expect(p.get_by_text('Outline every dark blob.')).to_be_visible()

        with r.check('Workspace: positive click, negative click, Undo, Clear'):
            blob = self.images[first['name'] + ('' if first['name'].endswith('.png') else '')] \
                if first['name'] in self.images else self.images[f"{first['name']}.png"]
            cx, cy, ax, ay = blob[0]
            click_canvas(p, ANNOT, cx, cy)
            expect(button(p, 'Clear')).to_be_enabled(timeout=MODEL_TIMEOUT)
            click_canvas(p, ANNOT, min(W - 2, cx + ax + 15), cy, right=True)
            expect(button(p, 'Undo')).to_be_enabled(timeout=UI_TIMEOUT)
            button(p, 'Undo').click()
            p.wait_for_timeout(500)
            button(p, 'Clear').click()
            expect(button(p, 'Clear')).to_be_disabled(timeout=UI_TIMEOUT)
            no_error(p)

        with r.check('Workspace: Ctrl+Z and Escape shortcuts'):
            click_canvas(p, ANNOT, cx, cy)
            expect(button(p, 'Clear')).to_be_enabled(timeout=MODEL_TIMEOUT)
            click_canvas(p, ANNOT, cx + 5, cy)
            expect(button(p, 'Undo')).to_be_enabled(timeout=UI_TIMEOUT)
            p.wait_for_timeout(400)
            key(p, 'Control+z')
            p.wait_for_timeout(600)
            key(p, 'Escape')
            expect(button(p, 'Clear')).to_be_disabled(timeout=UI_TIMEOUT)

        self.blobs_of = lambda it: self.images[it['name']] if it['name'] in self.images \
            else self.images[f"{it['name']}.png"]
        with r.check('Workspace: label 12 images with clicks (button and Enter), Mark done & next'):
            for n, it in enumerate(items[:12]):
                if n:
                    expect(p.locator('.item-head strong')).to_have_text(it['name'], timeout=UI_TIMEOUT)
                    self.wait_model_ready()
                for k, blob in enumerate(self.blobs_of(it)):
                    self.assist_object(it, blob, save_with_enter=bool(k))
                button(p, 'Mark done & next').click()
                deadline = time.time() + 20
                while time.time() < deadline and \
                        next(x for x in self.items('A') if x['id'] == it['id'])['status'] != 'done':
                    p.wait_for_timeout(300)
                assert next(x for x in self.items('A') if x['id'] == it['id'])['status'] == 'done'
            assert sum(len(self.anns(i['id'])) for i in items[:12]) >= 18

        last = items[12]
        with r.check('Workspace: polygon tool (click points, close on the first point)'):
            self.open_item(last['id'])
            p.get_by_title('Polygon (P)').click()
            cx, cy, ax, ay = self.blobs_of(last)[0]
            pts = [(cx - ax, cy - ay), (cx + ax, cy - ay), (cx + ax, cy + ay), (cx - ax, cy + ay)]
            for x, y in pts:
                click_canvas(p, ANNOT, x, y)
            click_canvas(p, ANNOT, *pts[0])
            expect(p.get_by_text('polygon · 4 pts')).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Workspace: polygon Backspace and Enter keys'):
            for x, y in pts[:3] + [(cx, cy + ay + 5)]:
                click_canvas(p, ANNOT, x, y)
            key(p, 'Backspace')
            key(p, 'Enter')
            expect(p.get_by_text('polygon · 3 pts')).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Workspace: select tool, change class, hide/show, delete'):
            p.get_by_title('Select (V)').click()
            click_canvas(p, ANNOT, cx, cy - ay + 3)
            expect(p.locator('.selected-panel')).to_be_visible(timeout=UI_TIMEOUT)
            button(p, 'Delete (Del)').click()
            expect(p.get_by_text('Annotations (1)')).to_be_visible(timeout=UI_TIMEOUT)
            p.locator('.ann-list li').first.click()
            expect(p.locator('.selected-panel select')).to_be_visible()
            hide = p.locator('.ann-list li').first.get_by_title('Hide')
            hide.click()
            p.locator('.ann-list li').first.get_by_title('Show').click()
            key(p, 'Delete')
            expect(p.get_by_text('Annotations (0)')).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Workspace: tool and class keys (A R P V, 1-9), opacity slider'):
            key(p, 'v')
            expect(p.get_by_title('Select (V)')).to_have_class('active')
            key(p, 'p')
            expect(p.get_by_title('Polygon (P)')).to_have_class('active')
            key(p, '1')
            key(p, 'a')
            expect(p.get_by_title('Assist (A)')).to_have_class('active')
            p.locator('label', has_text='Fill opacity').locator('input').fill('0.8')
            expect(p.get_by_text('Fill opacity: 0.80')).to_be_visible()

        with r.check('Workspace: zoom buttons, keys and mouse wheel'):
            zin = p.get_by_label('Zoom in')
            zin.click()
            zin.click()
            fit = p.get_by_title('Fit the image (0)')
            expect(fit).to_be_enabled()
            fit.click()
            expect(fit).to_be_disabled()
            key(p, '+')
            expect(fit).to_be_enabled()
            key(p, '0')
            expect(fit).to_be_disabled()
            p.mouse.move(*canvas_point(p, ANNOT, W / 2, H / 2))
            p.mouse.wheel(0, -300)
            expect(fit).to_be_enabled()
            p.get_by_label('Zoom out').click()
            key(p, '0')

        with r.check('Workspace: Prev/Next buttons and [ ] keys'):
            button(p, '‹ Prev').click()
            expect(p.locator('.item-head strong')).to_have_text(items[11]['name'], timeout=UI_TIMEOUT)
            button(p, 'Next ›').click()
            expect(p.locator('.item-head strong')).to_have_text(last['name'], timeout=UI_TIMEOUT)
            key(p, '[')
            expect(p.locator('.item-head strong')).to_have_text(items[11]['name'], timeout=UI_TIMEOUT)
            key(p, ']')
            expect(p.locator('.item-head strong')).to_have_text(last['name'], timeout=UI_TIMEOUT)

        with r.check('Workspace: Reopen a done item and mark it done again'):
            self.open_item(items[0]['id'])
            button(p, 'Reopen').click()
            expect(button(p, 'Mark done & next')).to_be_visible(timeout=UI_TIMEOUT)
            button(p, 'Mark done & next').click()
            p.wait_for_timeout(800)
            assert next(x for x in self.items('A') if x['id'] == items[0]['id'])['status'] == 'done'

    # ── models: shared model in the workspace, quick segment ──────────────
    def model_info(self) -> dict:
        return self.run.api('GET', '/models')

    def load_model_ui(self, mtype: str, ckpt: str | None, scope):
        """Pick a model in a ModelChooser and press Load for everyone."""
        scope.get_by_label('Type').select_option(mtype)
        if ckpt:
            scope.get_by_label('Checkpoint').select_option(ckpt)
        footer = self.page.locator('footer.status')
        before = footer.inner_text() if footer.count() else ''
        scope.get_by_role('button', name='Load for everyone').click()
        if footer.count():
            # The footer reads "<label>…" while busy and "<label> · N ms" /
            # "... done in N ms" (or an error) when the load and refresh are over.
            deadline = time.time() + MODEL_TIMEOUT / 1000
            while time.time() < deadline:
                t = footer.inner_text()
                if t != before and (t.endswith(' ms') or 'error' in (footer.get_attribute('class') or '')):
                    break
                self.page.wait_for_timeout(200)
            no_error(self.page)
        deadline = time.time() + MODEL_TIMEOUT / 1000
        while time.time() < deadline:
            cur = self.model_info()['current']
            if cur['model_type'] == mtype and (ckpt is None or cur['checkpoint'] == ckpt):
                # The page then refreshes its sessions; clicks are ignored until it is idle.
                expect(scope.get_by_role('button', name='Load for everyone')).to_be_enabled(
                    timeout=MODEL_TIMEOUT)
                self.page.wait_for_timeout(500)
                return
            self.page.wait_for_timeout(500)
        no_error(self.page)
        raise AssertionError(f'{mtype} / {ckpt} did not become the current model')

    def singles(self) -> list[tuple[str, str]]:
        info = self.model_info()
        return [(t, c) for c, t in info['checkpoint_types'].items() if t]

    def quick_segment(self):
        p, r = self.page, self.run
        img = self.write_images('quick', 1, blobs=1, seed0=500)[0]
        cx, cy, ax, ay = self.images[img.name][0]

        with r.check('Quick segment: load an image'):
            goto(r, '/quick')
            p.locator('input[type=file]').set_input_files(str(img))
            expect(p.get_by_text(f'{img.stem} · {W}×{H}')).to_be_visible(timeout=UI_TIMEOUT)

        def one_object(label: str):
            click_canvas(p, SEG, cx, cy)
            expect(button(p, 'Finish object')).to_be_enabled(timeout=MODEL_TIMEOUT)
            expect(p.get_by_text('1 click(s) on current object', exact=False)).to_be_visible()
            no_error(p)

        with r.check('Quick segment: clicks, Undo, Reset, Finish, Auto-segment, Download mask'):
            one_object('first')
            click_canvas(p, SEG, min(W - 2, cx + ax + 15), cy, right=True)
            expect(p.get_by_text('2 click(s)', exact=False)).to_be_visible(timeout=MODEL_TIMEOUT)
            button(p, 'Undo').click()
            expect(p.get_by_text('1 click(s)', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
            button(p, 'Reset').click()
            expect(button(p, 'Finish object')).to_be_disabled(timeout=UI_TIMEOUT)
            one_object('again')
            button(p, 'Finish object').click()
            expect(p.get_by_text('1 finished', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
            button(p, 'Auto-segment').click()
            p.wait_for_timeout(1500)
            no_error(p)
            with p.expect_download() as dl:
                button(p, 'Download mask').click()
            data = Path(dl.value.path()).read_bytes()
            assert data[:8] == b'\x89PNG\r\n\x1a\n', 'mask is not a PNG'
            # Auto-segment leaves its object open; close it for the next checks.
            if button(p, 'Reset').is_enabled():
                button(p, 'Reset').click()
                expect(button(p, 'Finish object')).to_be_disabled(timeout=UI_TIMEOUT)

        with r.check('Quick segment: keyboard (Ctrl+Z, Space finish, Esc reset)'):
            blur(p)
            click_canvas(p, SEG, cx, cy)
            expect(button(p, 'Finish object')).to_be_enabled(timeout=MODEL_TIMEOUT)
            click_canvas(p, SEG, cx + 4, cy)
            expect(p.get_by_text('2 click(s)', exact=False)).to_be_visible(timeout=MODEL_TIMEOUT)
            key(p, 'Control+z')
            expect(p.get_by_text('1 click(s)', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
            key(p, 'Escape')
            expect(button(p, 'Finish object')).to_be_disabled(timeout=UI_TIMEOUT)
            click_canvas(p, SEG, cx, cy)
            expect(button(p, 'Finish object')).to_be_enabled(timeout=MODEL_TIMEOUT)
            key(p, ' ')
            expect(p.get_by_text('2 finished', exact=False)).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Quick segment: Space with no object open shows no error'):
            key(p, ' ')
            p.wait_for_timeout(800)
            no_error(p)

        with r.check('Quick segment: threshold, zoom-in, opacity and click radius settings'):
            sid = None
            thr = p.locator('label', has_text='Probability threshold').locator('input')
            thr.focus()
            thr.press('ArrowRight')
            thr.press('ArrowRight')
            p.wait_for_timeout(800)
            expect(p.get_by_text('Probability threshold: 0.60')).to_be_visible()
            zoom = p.get_by_label('Zoom-in (RITM / SimpleClick)')
            if zoom.is_enabled():
                # Controlled by the server's reply, so click and wait for it.
                zoom.click()
                expect(zoom).not_to_be_checked(timeout=UI_TIMEOUT)
                zoom.click()
                expect(zoom).to_be_checked(timeout=UI_TIMEOUT)
            p.locator('label', has_text='Overlay opacity').locator('input').fill('0.3')
            p.locator('label', has_text='Click radius').locator('input').fill('8')
            expect(p.get_by_text('Click radius: 8')).to_be_visible()
            no_error(p)

        with r.check('Quick segment: zoom buttons and wheel'):
            fit = p.get_by_title('Fit the image (0)')
            p.get_by_label('Zoom in').click()
            expect(fit).to_be_enabled()
            key(p, '0')
            expect(fit).to_be_disabled()

        info = self.model_info()
        base_type, base_ckpt = info['current']['model_type'], info['current']['checkpoint']
        with r.check('Model picker lists only types that have a checkpoint'):
            types = p.get_by_label('Type').locator('option').all_inner_texts()
            avail = {t for c, t in info['checkpoint_types'].items() if t}
            assert set(types) - {'Ensemble'} == avail, (types, avail)

        for mtype, ckpt in sorted(self.singles()):
            with r.check(f'Quick segment with {mtype} ({ckpt})'):
                self.load_model_ui(mtype, ckpt, p.locator('.sidebar, aside, body').first)
                expect(p.get_by_text(f'Active: {mtype} · {ckpt}', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
                button(p, 'Reset').click() if button(p, 'Reset').is_enabled() else None
                one_object(mtype)
                click_canvas(p, SEG, min(W - 2, cx + ax + 15), cy, right=True)
                expect(p.get_by_text('2 click(s)', exact=False)).to_be_visible(timeout=MODEL_TIMEOUT)
                button(p, 'Finish object').click()
                no_error(p)

        cands = info['ensemble']['candidates']
        fusions = info['ensemble']['fusions']
        if len(cands) < 2:
            with r.check('Quick segment with an ensemble'):
                raise Skip('needs two checkpoints')
            fusions = []
        for fusion in fusions:
            for tta in ((False, True) if fusion == 'staple' else (False,)):
                n = min(3, len(cands))
                with r.check(f'Quick segment with an ensemble of {n} ({fusion}{", TTA" if tta else ""})'):
                    p.get_by_label('Type').select_option('Ensemble')
                    boxes = p.locator('fieldset.members input[type=checkbox]')
                    for i in range(boxes.count()):
                        boxes.nth(i).uncheck() if boxes.nth(i).is_checked() else None
                    for i in range(n):
                        boxes.nth(i).check()
                    p.get_by_label('Combine').select_option(fusion)
                    tbox = p.get_by_label('Test-time augmentation')
                    tbox.check() if tta else tbox.uncheck()
                    button(p, 'Load for everyone').click()
                    deadline = time.time() + MODEL_TIMEOUT / 1000
                    while time.time() < deadline:
                        cur = self.model_info()['current']
                        if cur['model_type'] == 'Ensemble' and cur.get('fusion') == fusion \
                                and cur.get('tta') == tta:
                            break
                        p.wait_for_timeout(500)
                    else:
                        no_error(p)
                        raise AssertionError('ensemble did not load')
                    if button(p, 'Reset').is_enabled():
                        button(p, 'Reset').click()
                    one_object('ensemble')
                    expect(p.get_by_text('Model weights', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
                    button(p, 'Finish object').click()
                    no_error(p)

        with r.check('Model load refuses a mismatched type and checkpoint (API)'):
            r.api('POST', '/models/load', expect_status=400,
                  data={'model_type': 'SAM2 - Tiny', 'checkpoint': base_ckpt}) \
                if base_type != 'SAM2 - Tiny' else None

        with r.check('Restore the default model'):
            self.load_model_ui(base_type, base_ckpt, p.locator('body'))

    def box_prompt(self):
        """Box prompts in the workspace need SAM or SAM 2 as the shared model."""
        p, r = self.page, self.run
        sams = [(t, c) for t, c in self.singles() if t.startswith('SAM')]
        base = self.model_info()['current']
        it = self.a_items[13]
        for mtype, ckpt in sams:
            with r.check(f'Workspace: box prompt with {mtype}'):
                self.open_item(it['id'])
                p.locator('details summary', has_text='Model:').click()
                self.load_model_ui(mtype, ckpt, p.locator('details', has_text='Model:'))
                cx, cy, ax, ay = self.blobs_of(it)[0]
                drag_canvas(p, ANNOT, cx - ax - 4, cy - ay - 4, cx + ax + 4, cy + ay + 4)
                save = p.get_by_role('button', name='Save', exact=False).filter(has_text='(Enter)')
                expect(save).to_be_enabled(timeout=MODEL_TIMEOUT)
                n = len(self.anns(it['id']))
                save.click()
                p.wait_for_timeout(1500)
                assert len(self.anns(it['id'])) == n + 1
        if not sams:
            with r.check('Workspace: box prompt with SAM / SAM 2'):
                raise Skip('no SAM or SAM 2 checkpoint')
        with r.check('Workspace: box prompt with RITM gives a clear message'):
            if base['model_type'] != 'RITM':
                raise Skip('default model is not RITM')
            self.open_item(it['id'])
            p.locator('details summary', has_text='Model:').click()
            self.load_model_ui(base['model_type'], base['checkpoint'], p.locator('details', has_text='Model:'))
            cx, cy, ax, ay = self.blobs_of(it)[0]
            drag_canvas(p, ANNOT, cx - ax - 4, cy - ay - 4, cx + ax + 4, cy + ay + 4)
            expect(p.get_by_text('Box prompts need SAM or SAM 2', exact=False)).to_be_visible(timeout=UI_TIMEOUT)

    # ── project models ────────────────────────────────────────────────────
    def models_of(self, key='A') -> list[dict]:
        return self.run.api('GET', f'/projects/{self.projects[key]}/models')['models']

    def wait_model(self, model_id: int, until=('done', 'failed', 'cancelled'), timeout=None) -> dict:
        timeout = timeout or self.args.train_timeout
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            m = next(m for m in self.models_of() if m['id'] == model_id)
            phase = (m.get('progress') or {}).get('phase')
            if (m['status'], phase) != last:
                print(f'       model #{model_id}: {m["status"]}' + (f' ({phase})' if phase else ''), flush=True)
                last = (m['status'], phase)
            if m['status'] in until:
                return m
            time.sleep(5)
        raise AssertionError(f'model #{model_id} still {last} after {timeout} s')

    def train(self, kind_label: str, steps: int) -> dict:
        p = self.page
        goto(self.run, f'/projects/{self.projects["A"]}/models')
        button(p, kind_label).click()
        p.get_by_label('Training steps').fill(str(steps))
        expect(button(p, 'Train')).to_be_enabled(timeout=UI_TIMEOUT)
        before = {m['id'] for m in self.models_of()}
        button(p, 'Train').click()
        deadline = time.time() + 30
        while time.time() < deadline:
            new = [m for m in self.models_of() if m['id'] not in before]
            if new:
                return new[0]
            p.wait_for_timeout(500)
        no_error(p)
        raise AssertionError('training did not start')

    def project_models(self):
        p, r = self.page, self.run
        if self.args.no_train:
            with r.check('Project models: training'):
                raise Skip('--no-train')
            return
        info = r.api('GET', f'/projects/{self.projects["A"]}/models')
        if not info.get('base_models'):
            with r.check('Project models: training'):
                raise Skip('no SAM 2 checkpoint on the server')

        with r.check('Models tab: kinds, counts and prerequisites'):
            goto(r, f'/projects/{self.projects["A"]}/models')
            for kind in ('Interactive', 'Automatic'):
                button(p, kind).click()
                no_error(p)
                expect(button(p, 'Train')).to_be_enabled()
            # A student needs a finished automatic model first: the tab says so
            # in red and keeps Train disabled.
            button(p, 'Student').click()
            expect(p.get_by_text('Needs a finished automatic SAM 2 model to learn from',
                                 exact=False)).to_be_visible()
            expect(button(p, 'Train')).to_be_disabled()
            button(p, 'Interactive').click()
            expect(p.get_by_text('labelled objects on', exact=False)).to_be_visible()

        trained: dict[str, dict] = {}
        for label, kind in (('Interactive', 'interactive'), ('Automatic', 'automatic')):
            with r.check(f'Train an {kind} model on the GPU'):
                m = self.wait_model(self.train(label, self.args.steps)['id'])
                assert m['status'] == 'done', f"{m['status']}: {m.get('error')}"
                assert m['metrics'], 'no held-out scores'
                trained[kind] = m

        with r.check('Train a student model on the GPU (taught by the automatic model)'):
            if 'automatic' not in trained:
                raise Skip('no automatic model to teach it')
            m = self.wait_model(self.train('Student', self.args.steps)['id'])
            assert m['status'] == 'done', f"{m['status']}: {m.get('error')}"
            trained['student'] = m

        with r.check('Cancel a running training; its Slurm job leaves the queue'):
            m = self.train('Interactive', 3000)
            m = self.wait_model(m['id'], until=('running', 'done', 'failed', 'cancelled'), timeout=600)
            assert m['status'] == 'running', m['status']
            job = None
            for _ in range(60):
                m = next(x for x in self.models_of() if x['id'] == m['id'])
                job = (m.get('progress') or {}).get('slurm_job') or m.get('slurm_job')
                if (m.get('progress') or {}).get('phase') not in (None, 'waiting for the scheduler'):
                    break
                time.sleep(5)
            goto(r, f'/projects/{self.projects["A"]}/models')
            p.locator('tr', has_text=f"#{m['id']} ·").get_by_role('button', name='Cancel').click()
            m = self.wait_model(m['id'], timeout=120)
            assert m['status'] == 'cancelled', m['status']
            if shutil.which('squeue'):
                time.sleep(5)
                # Job names are iseg-m<model>-<pid>; none of this model's may be left.
                q = subprocess.run(['squeue', '-h', '-o', '%j'], capture_output=True, text=True).stdout
                left = [j for j in q.split() if j.startswith(f"iseg-m{m['id']}-")]
                assert not left, f'Slurm job still queued: {left}'
            else:
                print('       (no squeue here: Slurm queue not checked)', flush=True)

        with r.check('Download each trained model as a zip'):
            if not trained:
                raise Skip('nothing trained')
            goto(r, f'/projects/{self.projects["A"]}/models')
            for kind, m in trained.items():
                row = p.locator('tr', has_text=f"#{m['id']} ·")
                with p.expect_download() as dl:
                    row.get_by_role('button', name='Download').click()
                names = zipfile.ZipFile(dl.value.path()).namelist()
                assert any('README' in n for n in names) and any('card' in n for n in names), (kind, names)
                if kind == 'student':
                    assert 'model.onnx' in names, names

        with r.check('Use the interactive model for labelling; workspace assists with it'):
            if 'interactive' not in trained:
                raise Skip('no interactive model')
            m = trained['interactive']
            p.locator('tr', has_text=f"#{m['id']} ·").get_by_role('button', name='Use for labelling').click()
            expect(p.get_by_text(f"project model #{m['id']}", exact=False).first).to_be_visible(timeout=UI_TIMEOUT)
            it = self.a_items[13]
            self.open_item(it['id'])
            expect(p.get_by_title('This project assists with its own trained model')).to_be_visible(
                timeout=MODEL_TIMEOUT)
            self.assist_object(it, self.blobs_of(it)[0])
            goto(r, f'/projects/{self.projects["A"]}/models')
            button(p, 'Use the shared model').click()
            expect(p.get_by_text('the shared model', exact=False)).to_be_visible(timeout=UI_TIMEOUT)

        for kind in ('automatic', 'student'):
            with r.check(f'Auto-label with the {kind} model: propose, accept, reject, accept all'):
                if kind not in trained:
                    raise Skip(f'no {kind} model')
                m = trained[kind]
                goto(r, f'/projects/{self.projects["A"]}/models')
                p.locator('tr', has_text=f"#{m['id']} ·").get_by_role(
                    'button', name='Use for auto-labelling').click()
                expect(p.get_by_text(f"Auto-labelling: project model #{m['id']}", exact=False)).to_be_visible(
                    timeout=UI_TIMEOUT)
                it = self.a_items[12]
                self.open_item(it['id'])
                button(p, 'Propose labels').click()
                expect(p.get_by_text('Accept all').or_(p.get_by_text('found nothing', exact=False))) \
                    .to_be_visible(timeout=MODEL_TIMEOUT)
                no_error(p)
                n = len(self.anns(it['id']))
                if p.get_by_title('Accept').count():
                    p.get_by_title('Accept').first.click()
                    p.wait_for_timeout(1000)
                    assert len(self.anns(it['id'])) == n + 1
                    if p.get_by_title('Reject').count():
                        p.get_by_title('Reject').first.click()
                    if p.get_by_title('Accept').count():
                        button(p, 'Accept all').click()
                        p.wait_for_timeout(1500)
                    button(p, 'Done').click() if button(p, 'Done').count() else button(p, 'Reject all').click()
                goto(r, f'/projects/{self.projects["A"]}/models')
                button(p, 'Turn off').click()
                expect(p.get_by_text('Auto-labelling: off', exact=False)).to_be_visible(timeout=UI_TIMEOUT)

        with r.check('Delete a model that is not in use'):
            if 'student' not in trained:
                raise Skip('no student model')
            m = trained['student']
            row = p.locator('tr', has_text=f"#{m['id']} ·")
            row.get_by_role('button', name='Delete').click()
            row.get_by_role('button', name='Yes').click()
            p.wait_for_timeout(1000)
            assert all(x['id'] != m['id'] for x in self.models_of())

    # ── the other task types ──────────────────────────────────────────────
    def detection(self):
        p, r = self.page, self.run
        with r.check('Detection project: box class, upload'):
            self.new_project('B', 'E2E detection', 'Object detection')
            self.add_class('car')
            goto(r, f'/projects/{self.projects["B"]}')
            self.upload(self.write_images('b', 3, blobs=1, seed0=300))
        it = sorted(self.items('B'), key=lambda i: i['id'])[0]
        cx, cy, ax, ay = self.images[it['name']][0] if it['name'] in self.images else self.images[it['name'] + '.png'][0]
        with r.check('Workspace: freehand box (drag)'):
            self.open_item(it['id'])
            expect(p.get_by_title('Box (R)')).to_have_class('active')
            drag_canvas(p, ANNOT, cx - ax, cy - ay, cx + ax, cy + ay)
            expect(p.get_by_text('box ', exact=False).first).to_be_visible(timeout=UI_TIMEOUT)
            assert len(self.anns(it['id'])) == 1
        with r.check('Workspace: fixed-size box (click to place)'):
            p.get_by_label('Fixed size (click to place)').check()
            p.get_by_label('Width (px)').fill('40')
            p.get_by_label('Height (px)').fill('30')
            click_canvas(p, ANNOT, 60, 60)
            p.wait_for_timeout(1000)
            boxes = [a['geometry']['bbox'] for a in self.anns(it['id'])]
            assert any(round(b[2]) == 40 and round(b[3]) == 30 for b in boxes), boxes
            p.get_by_label('Fixed size (click to place)').uncheck()
        with r.check('Workspace: Assist saves a box for a box class'):
            p.get_by_title('Assist (A)').click()
            idle(p)
            it2 = dict(it)
            self.assist_object(it2, (cx, cy))
            assert self.anns(it['id'])[-1]['geometry']['type'] == 'box'
        with r.check('Workspace: select a box with the Select tool and delete it with Del'):
            p.get_by_title('Select (V)').click()
            click_canvas(p, ANNOT, 60, 60)
            expect(p.locator('.selected-panel')).to_be_visible(timeout=UI_TIMEOUT)
            n = len(self.anns(it['id']))
            key(p, 'Delete')
            p.wait_for_timeout(800)
            assert len(self.anns(it['id'])) == n - 1
        r.api('PATCH', f'/items/{it["id"]}', data={'status': 'done'})
        self.exports('B', ('coco', 'yolo', 'csv'))

    def classification(self):
        p, r = self.page, self.run
        for key, name, multi in (('C', 'E2E multi-label', True), ('C1', 'E2E single-label', False)):
            with r.check(f'Classification project ({"multi" if multi else "single"}-label): tags'):
                self.new_project(key, name, 'Classification', multi=multi)
                self.add_class('cat')
                self.add_class('dog')
                goto(r, f'/projects/{self.projects[key]}')
                self.upload(self.write_images(key.lower(), 2, blobs=1, seed0=400))
                it = sorted(self.items(key), key=lambda i: i['id'])[0]
                self.open_item(it['id'])
                expect(p.locator('.annot-stage canvas')).to_have_count(0)   # a plain image, no tools
                p.locator('.class-btn', has_text='cat').click()
                p.wait_for_timeout(800)
                p.locator('.class-btn', has_text='dog').click()
                p.wait_for_timeout(800)
                tags = sorted(a['class_id'] for a in self.anns(it['id']))
                if multi:
                    assert len(tags) == 2, tags
                    p.locator('.class-btn', has_text='cat').click()
                    p.wait_for_timeout(800)
                    assert len(self.anns(it['id'])) == 1
                else:
                    assert len(tags) == 1, f'single-label image has {len(tags)} tags'
                button(p, 'Mark done & next').click()
                p.wait_for_timeout(800)
        self.exports('C', ('csv',))

    def change_detection(self):
        p, r = self.page, self.run
        with r.check('Change detection: classes, before/after pair upload'):
            self.new_project('D', 'E2E change', 'Change detection')
            self.add_class('new', 'mask')
            self.add_class('area', 'box')
            goto(r, f'/projects/{self.projects["D"]}')
            before = self.write_images('d_before', 2, blobs=1, seed0=600)
            after = self.write_images('d_after', 2, blobs=2, seed0=600)
            p.get_by_label('Before images').set_input_files([str(x) for x in before])
            p.get_by_label('After images').set_input_files([str(x) for x in after])
            button(p, 'Upload').click()
            expect(p.get_by_text('Uploaded 2 of 2 pairs.')).to_be_visible(timeout=120_000)
            items = self.items('D')
            assert len(items) == 2 and all(i['kind'] == 'pair' for i in items), items
        it = sorted(self.items('D'), key=lambda i: i['id'])[0]
        blobs = self.images['d_after_01.png']
        with r.check('Change detection: before / after / Compare, divider, B key'):
            self.open_item(it['id'])
            for b in ('before', 'after', 'Compare'):
                button(p, b).click()
            p.locator('label', has_text='Before ◂ divider ▸ after').locator('input').fill('0.3')
            button(p, 'after').click()
            key(p, 'b')
            expect(button(p, 'before')).to_have_class('active', timeout=UI_TIMEOUT)
            key(p, 'b')
            expect(button(p, 'after')).to_have_class('active', timeout=UI_TIMEOUT)
        with r.check('Change detection: assist on the after image, mask and box classes'):
            self.wait_model_ready()
            p.locator('.class-btn', has_text='new').click()
            self.assist_object(it, blobs[0])
            p.locator('.class-btn', has_text='area').click()
            self.assist_object(it, blobs[1])
            types = sorted(a['geometry']['type'] for a in self.anns(it['id']))
            assert types == ['box', 'mask'], types
            button(p, 'Mark done & next').click()
            p.wait_for_timeout(800)
        self.exports('D', ('coco', 'masks', 'csv'))

    def semantic(self):
        p, r = self.page, self.run
        with r.check('Semantic segmentation project: label one image'):
            self.new_project('E', 'E2E semantic', 'Semantic segmentation')
            self.add_class('region', 'mask')
            goto(r, f'/projects/{self.projects["E"]}')
            self.upload(self.write_images('e', 2, blobs=1, seed0=700))
            it = sorted(self.items('E'), key=lambda i: i['id'])[0]
            self.open_item(it['id'])
            self.wait_model_ready()
            self.assist_object(it, self.images['e_01.png'][0])
            button(p, 'Mark done & next').click()
            p.wait_for_timeout(800)
        self.exports('E', ('coco', 'masks', 'yolo', 'csv'))

    # ── exports ───────────────────────────────────────────────────────────
    def exports(self, key: str, formats):
        p, r = self.page, self.run
        labels = {'coco': 'COCO JSON', 'masks': 'PNG masks', 'yolo': 'YOLO', 'csv': 'CSV'}
        for fmt in formats:
            with r.check(f'Export {key}: {labels[fmt]}'):
                goto(r, f'/projects/{self.projects[key]}/export')
                p.get_by_label(labels[fmt], exact=False).first.check()
                with p.expect_download() as dl:
                    button(p, 'Download').click()
                path = Path(dl.value.path())
                if fmt == 'csv':
                    rows = list(csv.reader(io.StringIO(path.read_text())))
                    assert len(rows) >= 2, rows
                    continue
                names = zipfile.ZipFile(path).namelist()
                want = {'coco': 'annotations.json', 'yolo': 'data.yaml', 'masks': '.png'}[fmt]
                assert any(n.endswith(want) for n in names), names
        with r.check(f'Export {key}: all images, without the images in the zip'):
            if formats == ('csv',):
                raise Skip('CSV only')
            goto(r, f'/projects/{self.projects[key]}/export')
            p.get_by_label(labels[formats[0]], exact=False).first.check()
            p.get_by_label('Only images marked done').uncheck()
            p.get_by_label('Include the images in the zip').uncheck()
            with p.expect_download() as dl:
                button(p, 'Download').click()
            names = zipfile.ZipFile(dl.value.path()).namelist()
            assert not any(n.lower().endswith(('.jpg', '.jpeg')) or n.startswith('images/') for n in names), names

    # ── labeller's view ───────────────────────────────────────────────────
    def labeller(self):
        r = self.run
        ctx = self.browser.new_context(viewport={'width': 1400, 'height': 900}, accept_downloads=True)
        lp = ctx.new_page()
        admin_page, admin_token = r.page, r.token
        r.page = lp
        try:
            with r.check('Labeller: signs in, sees only the assigned project'):
                lp.goto(f'{r.base}/')
                lp.get_by_label('Username').fill('e2e_lab')
                lp.get_by_label('Password').fill(LAB_PASSWORD)
                button(lp, 'Sign in').click()
                expect(lp.get_by_text('E2E Labeller (labeller)')).to_be_visible(timeout=UI_TIMEOUT)
                r.refresh_token()
                expect(lp.get_by_role('link', name='Users')).to_have_count(0)
                expect(lp.get_by_role('link', name='Audit log')).to_have_count(0)
                names = [x['name'] for x in r.api('GET', '/projects')]
                assert names == ['E2E instance'], names
            with r.check('Labeller: no admin tabs, protocol shown, only assigned images'):
                lp.get_by_text('E2E instance').first.click()
                expect(lp.get_by_text('assigned to you', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
                expect(lp.get_by_role('link', name='Classes', exact=True)).to_have_count(0)
                expect(lp.locator('summary', has_text='Protocol')).to_be_visible()
                expect(lp.locator('tbody tr')).to_have_count(2)
            with r.check('Labeller: labels an image and marks it done'):
                button(lp, 'Start labelling').click()
                expect(lp.get_by_text('model ready', exact=False)).to_be_visible(timeout=MODEL_TIMEOUT)
                name = lp.locator('.item-head strong').inner_text()
                item = next(i for i in r.api('GET', f'/projects/{self.projects["A"]}/items') if i['name'] == name)
                self.page = lp
                self.assist_object(item, self.blobs_of(item)[0])
                expect(lp.locator('details summary', has_text='Model:')).to_have_count(0)
                button(lp, 'Mark done & next').click()
                lp.wait_for_timeout(800)
            with r.check('Labeller: admin-only API calls are refused'):
                r.api('GET', '/users', expect_status=403)
                r.api('POST', '/models/load', expect_status=403,
                      data={'model_type': 'RITM', 'checkpoint': 'x.pth'})
                r.api('GET', '/audit', expect_status=403)
        finally:
            self.page = admin_page
            r.page, r.token = admin_page, admin_token
            ctx.close()

    # ── settings, audit, login link ───────────────────────────────────────
    def settings_and_audit(self):
        p, r = self.page, self.run
        with r.check('Settings: rename, protocol, Save'):
            goto(r, f'/projects/{self.projects["E"]}/settings')
            p.get_by_label('Name', exact=True).fill('E2E semantic renamed')
            p.get_by_label('Protocol', exact=True).fill('New protocol')
            button(p, 'Save').click()
            expect(p.get_by_text('Saved')).to_be_visible(timeout=UI_TIMEOUT)
            pr = r.api('GET', f'/projects/{self.projects["E"]}')
            assert pr['name'] == 'E2E semantic renamed' and pr['protocol'] == 'New protocol'
        with r.check('Settings: delete a project after typing its name'):
            p.get_by_placeholder('E2E semantic renamed').fill('E2E semantic renamed')
            button(p, 'Delete project').click()
            p.wait_for_timeout(1000)
            assert all(x['id'] != self.projects['E'] for x in r.api('GET', '/projects'))
        with r.check('Audit log: every filter, integrity check'):
            goto(r, '/audit')
            sel = p.locator('select').first
            for opt in ('Sign-ins', 'Accounts', 'Projects', 'Classes', 'Items', 'Annotations', 'Models',
                        'All events'):
                sel.select_option(label=opt)
                p.wait_for_timeout(400)
                no_error(p)
            button(p, 'Check integrity').click()
            expect(p.get_by_text('Intact: all', exact=False)).to_be_visible(timeout=UI_TIMEOUT)
            if button(p, 'Load older').count():
                button(p, 'Load older').click()
                no_error(p)
        with r.check('Signed out: every page, quick segment included, asks to sign in'):
            ctx = self.browser.new_context()
            lp = ctx.new_page()
            try:
                for h in ('/', '/quick', '/users', f'/projects/{self.projects["A"]}'):
                    lp.goto(f'{r.base}/#{h}')
                    expect(lp.get_by_role('heading', name='Sign in')).to_be_visible(timeout=UI_TIMEOUT)
                # The sign-in page must not offer anything that works without an account.
                expect(lp.get_by_role('link', name='Quick segment')).to_have_count(0)
            finally:
                ctx.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--url', required=True)
    ap.add_argument('--setup-token', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-train', action='store_true', help='skip project-model training')
    ap.add_argument('--steps', type=int, default=20, help='training steps per model (min 10)')
    ap.add_argument('--train-timeout', type=int, default=1800, help='seconds per training')
    ap.add_argument('--headed', action='store_true')
    ap.add_argument('--only', help='comma-separated section names (development)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run = Run(args.out, args.url)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={'width': 1400, 'height': 900}, accept_downloads=True)
        run.page = ctx.new_page()
        run.page.set_default_timeout(UI_TIMEOUT)
        console = []
        run.page.on('console', lambda m: m.type == 'error' and console.append(m.text))
        run.page.on('pageerror', lambda e: console.append(f'pageerror: {e}'))
        e = E2E(run, args, browser)
        sections = [e.setup_admin, e.users_page, e.project_a, e.label_project_a, e.quick_segment,
                    e.box_prompt, e.project_models, e.detection, e.classification,
                    e.change_detection, e.semantic, e.labeller, e.settings_and_audit]
        if args.only:
            keep = set(args.only.split(','))
            sections = [s for s in sections if s.__name__ in keep or s.__name__ == 'setup_admin']
        for section in sections:
            print(f'\n== {section.__name__}', flush=True)
            try:
                section()
            except Exception:  # noqa: BLE001 - a broken section must not stop the rest
                with run.check(f'{section.__name__}: section setup'):
                    raise
        with run.check('No JavaScript errors in the browser console'):
            real = [c for c in console if 'Failed to load resource' not in c]
            assert not real, real[:5]
        browser.close()
    sys.exit(run.report())


if __name__ == '__main__':
    main()
