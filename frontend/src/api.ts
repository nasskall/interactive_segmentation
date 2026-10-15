// Typed client for the FastAPI backend (server/app.py, server/labelling/api.py).

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
  box: [number, number, number, number] | null;
  supports_box: boolean;
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

// ── labelling ──────────────────────────────────────────────────────────────

export type TaskType =
  | 'classification'
  | 'detection'
  | 'semantic_segmentation'
  | 'instance_segmentation'
  | 'change_detection';
export type LabelType = 'tag' | 'box' | 'polygon' | 'mask';
export type ItemStatus = 'todo' | 'in_progress' | 'done';

export interface User {
  id: number;
  username: string;
  display_name: string;
  role: 'admin' | 'labeller';
  active: boolean;
}

export interface Project {
  id: number;
  name: string;
  task_type: TaskType;
  multi_label: boolean;
  protocol: string;
  allowed_label_types: LabelType[];
}

export interface LabelClass {
  id: number;
  project_id: number;
  name: string;
  color: string;
  label_type: LabelType;
  position: number;
}

export interface Item {
  id: number;
  project_id: number;
  kind: 'single' | 'pair';
  name: string;
  width: number;
  height: number;
  images: Record<string, string>; // role -> URL ('image', or 'before' / 'after')
  assignee_id: number | null;
  status: ItemStatus;
}

export interface Rle {
  size: [number, number];
  counts: number[];
}

export type Geometry =
  | { type: 'tag' }
  | { type: 'box'; bbox: [number, number, number, number] }
  | { type: 'polygon'; points: [number, number][] }
  | { type: 'mask'; rle: Rle };

export interface Annotation {
  id: number;
  item_id: number;
  class_id: number;
  labeller_id: number | null;
  geometry: Geometry;
  source: 'manual' | 'assisted';
  created_at: string;
}

export interface Progress {
  items: number;
  by_status: Record<ItemStatus, number>;
  by_labeller: { user_id: number; display_name: string; items: number; done: number;
                 annotations: number }[];
}

export const TASK_LABELS: Record<TaskType, string> = {
  classification: 'Classification',
  detection: 'Object detection',
  semantic_segmentation: 'Semantic segmentation',
  instance_segmentation: 'Instance segmentation',
  change_detection: 'Change detection',
};

// ── transport ──────────────────────────────────────────────────────────────

const TOKEN_KEY = 'iseg.token';

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

let token: string | null = readToken();
let onUnauthorized: () => void = () => {};

export function setToken(t: string | null) {
  token = t;
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: stay signed in for this tab only */
  }
}

export const hasToken = () => token !== null;
export const setUnauthorizedHandler = (fn: () => void) => { onUnauthorized = fn; };

async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(path.startsWith('/api') ? path : `/api${path}`, { ...init, headers });
  if (!res.ok) {
    if (res.status === 401 && token) onUnauthorized();
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
  return res;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await raw(path, init);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

/** Object URL for an image that needs the auth header (revoke when done). */
export async function fetchImageUrl(url: string): Promise<string> {
  const res = await raw(url);
  return URL.createObjectURL(await res.blob());
}

export const api = {
  // models + interactive sessions
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
  setBox: (sid: string, box: { x0: number; y0: number; x1: number; y1: number }) =>
    request<SessionState>(`/sessions/${sid}/box`, json('POST', box)),
  undo: (sid: string) => request<SessionState>(`/sessions/${sid}/undo`, json('POST')),
  finish: (sid: string) => request<SessionState>(`/sessions/${sid}/finish`, json('POST')),
  reset: (sid: string) => request<SessionState>(`/sessions/${sid}/reset`, json('POST')),
  autoSegment: (sid: string) => request<SessionState>(`/sessions/${sid}/auto-segment`, json('POST')),
  updateSettings: (sid: string, settings: { prob_thresh?: number; use_zoom_in?: boolean }) =>
    request<SessionState>(`/sessions/${sid}/settings`, json('PATCH', settings)),

  // accounts
  authStatus: () => request<{ needs_setup: boolean }>('/auth/status'),
  setup: (body: { setup_token: string; username: string; password: string; display_name: string }) =>
    request<{ token: string; user: User }>('/auth/setup', json('POST', body)),
  login: (username: string, password: string) =>
    request<{ token: string; user: User }>('/auth/login', json('POST', { username, password })),
  logout: () => request<void>('/auth/logout', json('POST')),
  me: () => request<User>('/auth/me'),
  listUsers: () => request<User[]>('/users'),

  // projects
  listProjects: () => request<Project[]>('/projects'),
  getProject: (id: number) => request<Project>(`/projects/${id}`),
  progress: (id: number) => request<Progress>(`/projects/${id}/progress`),
  listClasses: (projectId: number) => request<LabelClass[]>(`/projects/${projectId}/classes`),
  listItems: (projectId: number, status?: ItemStatus) =>
    request<Item[]>(`/projects/${projectId}/items${status ? `?status=${status}` : ''}`),

  // items + annotations
  getItem: (id: number) => request<Item>(`/items/${id}`),
  setItemStatus: (id: number, status: ItemStatus) =>
    request<Item>(`/items/${id}`, json('PATCH', { status })),
  startAssist: (itemId: number, image: string) =>
    request<SessionState>(`/items/${itemId}/assist?image=${encodeURIComponent(image)}`,
                          json('POST')),
  listAnnotations: (itemId: number) => request<Annotation[]>(`/items/${itemId}/annotations`),
  createAnnotation: (itemId: number,
                     body: { class_id: number; geometry?: Geometry; from_session?: string }) =>
    request<Annotation>(`/items/${itemId}/annotations`, json('POST', body)),
  updateAnnotation: (id: number, body: { class_id?: number; geometry?: Geometry }) =>
    request<Annotation>(`/annotations/${id}`, json('PATCH', body)),
  deleteAnnotation: (id: number) => request<void>(`/annotations/${id}`, { method: 'DELETE' }),
};
