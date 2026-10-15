// Typed client for the FastAPI backend (server/app.py).

export interface Click {
  x: number;
  y: number;
  positive: boolean;
}

export interface SessionState {
  session_id: string;
  filename: string;
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  clicks: Click[];
  object_count: number;
  has_incomplete_object: boolean;
  can_undo: boolean;
  prob_thresh: number;
  use_zoom_in: boolean;
  overlay: string; // data: URL, RGBA PNG
}

export interface ModelInfo {
  model_types: string[];
  checkpoints: string[];
  current: { model_type: string; checkpoint: string | null };
  device: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  getModels: () => request<ModelInfo>('/models'),
  loadModel: (model_type: string, checkpoint: string) =>
    request<ModelInfo>('/models/load', json('POST', { model_type, checkpoint })),

  createSession: (file: File) => {
    const form = new FormData();
    form.append('image', file);
    return request<SessionState>('/sessions', { method: 'POST', body: form });
  },
  getSession: (sid: string) => request<SessionState>(`/sessions/${sid}`),
  deleteSession: (sid: string) => request<void>(`/sessions/${sid}`, { method: 'DELETE' }),
  imageUrl: (sid: string) => `/api/sessions/${sid}/image`,
  maskUrl: (sid: string) => `/api/sessions/${sid}/mask`,

  addClick: (sid: string, click: Click) =>
    request<SessionState>(`/sessions/${sid}/clicks`, json('POST', click)),
  undo: (sid: string) => request<SessionState>(`/sessions/${sid}/undo`, json('POST')),
  finish: (sid: string) => request<SessionState>(`/sessions/${sid}/finish`, json('POST')),
  reset: (sid: string) => request<SessionState>(`/sessions/${sid}/reset`, json('POST')),
  autoSegment: (sid: string) => request<SessionState>(`/sessions/${sid}/auto-segment`, json('POST')),
  updateSettings: (sid: string, settings: { prob_thresh?: number; use_zoom_in?: boolean }) =>
    request<SessionState>(`/sessions/${sid}/settings`, json('PATCH', settings)),
};
