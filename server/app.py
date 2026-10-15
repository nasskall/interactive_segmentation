"""REST API for interactive segmentation.

Run from the repository root:

    python -m server.main            # http://127.0.0.1:8000, docs at /docs

Endpoints are synchronous on purpose: FastAPI runs them in its threadpool and
ModelManager.lock serialises every call that touches the network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.sessions import ModelManager, SessionNotFound, encode_png

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class ClickIn(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    positive: bool = True


class SettingsIn(BaseModel):
    prob_thresh: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    use_zoom_in: Optional[bool] = None


class LoadModelIn(BaseModel):
    model_type: str
    checkpoint: str


def create_app(manager: ModelManager, static_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title='Interactive Segmentation API', version='1.0')
    app.state.manager = manager

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
    def get_models():
        return manager.model_info()

    @app.post('/api/models/load')
    def load_model(body: LoadModelIn):
        try:
            manager.load_model(body.model_type, body.checkpoint)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(500, f'Failed to load model: {exc}')
        return manager.model_info()

    # ── sessions ───────────────────────────────────────────────────────────

    @app.post('/api/sessions', status_code=201)
    def create_session(image: UploadFile = File(...)):
        data = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image too large')
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, 'Could not decode image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(image.filename or 'image'))[0]
        name = ''.join(c for c in name if c.isalnum() or c in '-_.') or 'image'
        session = manager.create_session(rgb, name)
        return session.state()

    @app.get('/api/sessions/{sid}')
    def get_session(sid: str):
        return manager.get(sid).state()

    @app.delete('/api/sessions/{sid}', status_code=204)
    def delete_session(sid: str):
        manager.delete(sid)
        return Response(status_code=204)

    @app.get('/api/sessions/{sid}/image')
    def get_image(sid: str):
        """The (possibly downscaled) image that click coordinates refer to."""
        return Response(encode_png(manager.get(sid).image), media_type='image/png')

    @app.post('/api/sessions/{sid}/clicks')
    def add_click(sid: str, click: ClickIn):
        s = manager.get(sid)
        h, w = s.image.shape[:2]
        if click.x >= w or click.y >= h:
            raise HTTPException(400, f'Click ({click.x}, {click.y}) outside {w}x{h} image')
        with manager.lock:
            s.controller.add_click(click.x, click.y, click.positive)
            return s.state()

    @app.post('/api/sessions/{sid}/undo')
    def undo(sid: str):
        s = manager.get(sid)
        with manager.lock:
            s.controller.undo_click()
            return s.state()

    @app.post('/api/sessions/{sid}/finish')
    def finish_object(sid: str):
        s = manager.get(sid)
        with manager.lock:
            s.controller.finish_object()
            return s.state()

    @app.post('/api/sessions/{sid}/reset')
    def reset_object(sid: str):
        s = manager.get(sid)
        with manager.lock:
            s.controller.reset_last_object(update_image=False)
            return s.state()

    @app.post('/api/sessions/{sid}/auto-segment')
    def auto_segment(sid: str):
        s = manager.get(sid)
        with manager.lock:
            result = s.controller.auto_segment(mode='auto')
            if not result['ok']:
                raise HTTPException(422, f"No mask produced: {result.get('reason')}")
            return s.state()

    @app.patch('/api/sessions/{sid}/settings')
    def update_settings(sid: str, body: SettingsIn):
        s = manager.get(sid)
        manager.update_settings(s, body.prob_thresh, body.use_zoom_in)
        with manager.lock:
            return s.state()

    @app.get('/api/sessions/{sid}/mask')
    def download_mask(sid: str):
        """Label mask at the original upload resolution."""
        s = manager.get(sid)
        with manager.lock:
            png = s.full_res_mask_png()
        return Response(png, media_type='image/png', headers={
            'Content-Disposition': f'attachment; filename="{s.filename}_mask.png"',
        })

    # Serve the built React app (frontend/dist) at / when present.
    if static_dir is not None and Path(static_dir).is_dir():
        app.mount('/', StaticFiles(directory=static_dir, html=True), name='frontend')

    return app
