"""REST API for interactive segmentation.

Run from the repository root:

    python -m server.main            # http://127.0.0.1:8000, docs at /docs

Endpoints are synchronous on purpose: FastAPI runs them in its threadpool and
ModelManager.lock serialises every call that touches the network.

With labelling enabled (``data_dir``, always the case from ``server.main``)
every segmentation endpoint needs a signed-in account: a session belongs to
the account that opened it and is invisible to everyone else, and only
admins may switch the shared model (each switch is audit-logged). Without
``data_dir`` there are no accounts and the endpoints are open; that mode is
for tests and single-user use on 127.0.0.1 only.

Every response carries security headers (CSP, no framing, no MIME
sniffing, no caching of API responses; HSTS when served over HTTPS).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.labelling import audit
from server.labelling.api import remember_client_ip
from server.sessions import ENSEMBLE, ModelManager, Session, SessionNotFound, encode_png

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# The React build loads only its own scripts and styles; overlays and masks
# arrive as data: / blob: URLs.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    'Content-Security-Policy': CONTENT_SECURITY_POLICY,
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Cross-Origin-Opener-Policy': 'same-origin',
}


def _no_accounts() -> None:
    """Stand-in user dependency when labelling (and so accounts) is off."""
    return None


class ClickIn(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    positive: bool = True


class BoxIn(BaseModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)


class SettingsIn(BaseModel):
    prob_thresh: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    use_zoom_in: Optional[bool] = None


class EnsembleMemberIn(BaseModel):
    model_type: str
    checkpoint: str


class LoadModelIn(BaseModel):
    """A single model (``checkpoint``), or ``model_type`` "Ensemble" with
    ``members``, ``fusion`` and ``tta``."""
    model_type: str
    checkpoint: Optional[str] = None
    members: Optional[list[EnsembleMemberIn]] = None
    fusion: str = 'staple'
    tta: bool = False


def create_app(manager: ModelManager, static_dir: Optional[Path] = None,
               data_dir: Optional[Path] = None) -> FastAPI:
    """``data_dir`` enables labelling projects (database + uploaded images);
    without it only the stateless segmentation endpoints are served."""
    app = FastAPI(title='Interactive Segmentation API', version='1.0',
                  dependencies=[Depends(remember_client_ip)])
    app.state.manager = manager
    app.state.labelling = None
    if data_dir is not None:
        from server.labelling.api import build_router
        router, app.state.labelling = build_router(data_dir, manager)
        app.include_router(router)
    labelling = app.state.labelling
    signed_in = labelling.current_user if labelling else _no_accounts
    admin_only = labelling.admin_user if labelling else _no_accounts

    def own(sid: str, user) -> Session:
        """The session, if ``user`` opened it (any session without accounts)."""
        s = manager.get(sid)
        if user is not None and s.owner_id != user['id']:
            raise SessionNotFound(sid)
        return s

    @app.middleware('http')
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith('/api'):
            response.headers.setdefault('Cache-Control', 'no-store')
        if request.url.scheme == 'https':
            response.headers.setdefault('Strict-Transport-Security',
                                        'max-age=31536000; includeSubDomains')
        return response

    # The Vite dev server proxies /api, so CORS is only needed when the
    # frontend is hosted on a different origin.
    origins = [o for o in os.environ.get('ISEG_CORS_ORIGINS', '').split(',') if o]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=['*'], allow_headers=['*'])

    @app.exception_handler(SessionNotFound)
    def _session_not_found(request: Request, exc: SessionNotFound):
        return JSONResponse(status_code=404, content={'detail': 'Session not found'})

    # ── models ─────────────────────────────────────────────────────────────

    @app.get('/api/models')
    def get_models(user=Depends(signed_in)):
        return manager.model_info()

    @app.post('/api/models/load')
    def load_model(body: LoadModelIn, user=Depends(admin_only)):
        """Switch the model every session uses (admins only)."""
        try:
            if body.model_type == ENSEMBLE:
                manager.load_ensemble([(m.model_type, m.checkpoint) for m in body.members or []],
                                      body.fusion, body.tta)
            elif body.checkpoint is None:
                raise ValueError('checkpoint is required')
            else:
                manager.load_model(body.model_type, body.checkpoint)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f'Failed to load model: {exc}')
        if labelling:
            audit.record_now(labelling.db, 'model.load', user=user,
                             detail=manager.provenance())
        return manager.model_info()

    # ── sessions ───────────────────────────────────────────────────────────

    @app.post('/api/sessions', status_code=201)
    def create_session(image: UploadFile = File(...), user=Depends(signed_in)):
        data = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image too large')
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, 'Could not decode image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(image.filename or 'image'))[0]
        name = ''.join(c for c in name if c.isalnum() or c in '-_.') or 'image'
        session = manager.create_session(rgb, name,
                                         owner_id=user['id'] if user is not None else None)
        return session.state()

    @app.get('/api/sessions/{sid}')
    def get_session(sid: str, user=Depends(signed_in)):
        return own(sid, user).state()

    @app.delete('/api/sessions/{sid}', status_code=204)
    def delete_session(sid: str, user=Depends(signed_in)):
        own(sid, user)
        manager.delete(sid)
        return Response(status_code=204)

    @app.get('/api/sessions/{sid}/image')
    def get_image(sid: str, user=Depends(signed_in)):
        """The (possibly downscaled) image that click coordinates refer to."""
        return Response(encode_png(own(sid, user).image), media_type='image/png')

    @app.post('/api/sessions/{sid}/clicks')
    def add_click(sid: str, click: ClickIn, user=Depends(signed_in)):
        s = own(sid, user)
        h, w = s.image.shape[:2]
        if click.x >= w or click.y >= h:
            raise HTTPException(400, f'Click ({click.x}, {click.y}) outside {w}x{h} image')
        with manager.lock:
            s.controller.add_click(click.x, click.y, click.positive)
            return s.state()

    @app.post('/api/sessions/{sid}/box')
    def set_box(sid: str, box: BoxIn, user=Depends(signed_in)):
        """Box prompt for the current object (SAM / SAM 2 only)."""
        s = own(sid, user)
        h, w = s.image.shape[:2]
        if not (box.x0 < box.x1 <= w and box.y0 < box.y1 <= h):
            raise HTTPException(400, f'Box must have x0 < x1 <= {w} and y0 < y1 <= {h}')
        with manager.lock:
            if not s.controller.supports_box:
                raise HTTPException(422, 'Box prompts need a SAM or SAM 2 model')
            s.controller.set_box(box.x0, box.y0, box.x1, box.y1)
            return s.state()

    @app.post('/api/sessions/{sid}/undo')
    def undo(sid: str, user=Depends(signed_in)):
        s = own(sid, user)
        with manager.lock:
            s.controller.undo_click()
            return s.state()

    @app.post('/api/sessions/{sid}/finish')
    def finish_object(sid: str, user=Depends(signed_in)):
        s = own(sid, user)
        with manager.lock:
            s.controller.finish_object()
            return s.state()

    @app.post('/api/sessions/{sid}/reset')
    def reset_object(sid: str, user=Depends(signed_in)):
        s = own(sid, user)
        with manager.lock:
            s.controller.reset_last_object(update_image=False)
            return s.state()

    @app.post('/api/sessions/{sid}/auto-segment')
    def auto_segment(sid: str, user=Depends(signed_in)):
        s = own(sid, user)
        with manager.lock:
            result = s.controller.auto_segment(mode='auto')
            if not result['ok']:
                raise HTTPException(422, f"No mask produced: {result.get('reason')}")
            return s.state()

    @app.patch('/api/sessions/{sid}/settings')
    def update_settings(sid: str, body: SettingsIn, user=Depends(signed_in)):
        s = own(sid, user)
        manager.update_settings(s, body.prob_thresh, body.use_zoom_in)
        with manager.lock:
            return s.state()

    @app.get('/api/sessions/{sid}/mask')
    def download_mask(sid: str, user=Depends(signed_in)):
        """Label mask at the original upload resolution."""
        s = own(sid, user)
        with manager.lock:
            png = s.full_res_mask_png()
        return Response(png, media_type='image/png', headers={
            'Content-Disposition': f'attachment; filename="{s.filename}_mask.png"',
        })

    # Serve the built React app (frontend/dist) at / when present.
    if static_dir is not None and Path(static_dir).is_dir():
        app.mount('/', StaticFiles(directory=static_dir, html=True), name='frontend')

    return app
