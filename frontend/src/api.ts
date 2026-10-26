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
  ensemble_weights: Record<string, number> | null; // ensemble: weight per model, latest click
  object_count: number;
  has_incomplete_object: boolean;
  can_undo: boolean;
  prob_thresh: number;
  use_zoom_in: boolean;
  model: string | null; // set when the session runs a project model instead of the shared one
  overlay: string; // data: URL, RGBA PNG
}

export const ENSEMBLE = 'Ensemble';
export type Fusion = 'mean' | 'weighted' | 'weighted_conf' | 'staple';

export interface ModelRef {
  model_type: string;
  checkpoint: string;
}

export interface ModelInfo {
  model_types: string[];
  checkpoints: string[];
  current: {
    model_type: string;
    checkpoint: string | null;
    supports_box: boolean;
    // Ensemble only:
    members?: ModelRef[];
    fusion?: Fusion;
    tta?: boolean;
  };
  device: string;
  ensemble: { fusions: Fusion[]; max_members: number; candidates: ModelRef[] };
}

export type LoadModelBody =
  | ModelRef
  | { model_type: typeof ENSEMBLE; members: ModelRef[]; fusion: Fusion; tta: boolean };

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
  assist_model_id: number | null; // project model used for assisted labelling
  auto_model_id: number | null; // automatic project model that proposes labels
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
  annotation_count: number | null;
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
  model: AnnotationModel | null; // assisted only: the model that proposed it
  prompts: Prompts | null; // assisted only: the clicks (and box) behind each object
  created_at: string;
}

export interface Prompts {
  image: string;
  objects: { clicks: [number, number, boolean][]; box: [number, number, number, number] | null }[];
}

export interface AnnotationModel {
  model_type: string;
  checkpoint?: string | null;
  sha256?: string | null;
  fusion?: Fusion;
  members?: { model_type: string; checkpoint: string; sha256: string | null }[];
  adapter?: { project_model_id: number; file: string; sha256: string };
}

// ── project models ─────────────────────────────────────────────────────────

export type ModelStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
export type ModelKind = 'interactive' | 'automatic';
// What can be trained: a student is an automatic model with arch 'unet'.
export type TrainKind = ModelKind | 'student';

export interface Scores {
  objects: number;
  'iou@1': number;
  'iou@3': number;
  'iou@5': number;
  'noc@85': number;
  'noc@90': number;
}

// Automatic models: IoU of each class (in `classes` order) on the held-out images.
export interface AutoScores {
  images: number;
  iou: number[];
  miou: number;
}

export interface ProjectModel {
  id: number;
  project_id: number;
  kind: ModelKind;
  arch: 'sam2' | 'unet'; // unet: standalone student
  teacher_id: number | null; // students: the automatic SAM 2 model that taught them
  classes: { id: number; name: string }[] | null; // automatic: its outputs
  status: ModelStatus;
  base_model_type: string;
  base_checkpoint: string;
  base_sha256: string | null;
  steps: number;
  progress: { phase: string; done: number; total: number; loss: number | null } | null;
  train_objects: number;
  train_items: number;
  held_out_items: number[];
  metrics: { base: Scores; trained: Scores } | { trained: AutoScores; teacher?: AutoScores }
    | null;
  adapter_file: string | null;
  adapter_sha256: string | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface ProjectModels {
  models: ProjectModel[];
  assist_model_id: number | null;
  auto_model_id: number | null;
  auto_data: {
    classes: { id: number; name: string }[];
    done_images: number;
    labels: number;
    train_images: number;
    held_out_images: number;
    default_steps: number;
    min_images: number;
    max_steps: number;
    problems: string[];
  };
  student_data: {
    teacher_id: number | null;
    done_images: number;
    unlabelled_images: number;
    train_images: number;
    held_out_images: number;
    default_steps: number;
    min_images: number;
    max_steps: number;
    problems: string[];
  };
  data: {
    assisted_annotations: number;
    trainable_objects: number;
    items: number;
    train_objects: number;
    held_out_objects: number;
    held_out_items: number;
    default_steps: number;
    min_objects: number;
    max_steps: number;
    problems: string[];
  };
  base_models: ModelRef[];
  training_id: number | null;
}

export interface Proposal {
  index: number;
  class_id: number;
  geometry: Geometry;
  score: number; // mean probability inside the region
  area: number;  // share of the image
}

export interface ProposalSet {
  set: string;
  model_id: number;
  proposals: Proposal[];
}

export interface AuditEntry {
  id: number;
  at: string;
  user_id: number | null;
  username: string | null;
  action: string;
  target: string;
  detail: Record<string, unknown>;
  ip: string;
}

export interface Progress {
  items: number;
  by_status: Record<ItemStatus, number>;
  by_labeller: { user_id: number; display_name: string; items: number; done: number;
                 annotations: number }[];
}

export type ExportFormat = 'coco' | 'masks' | 'yolo' | 'csv';

// Mirrors TASK_FORMATS in server/labelling/export.py.
export const TASK_FORMATS: Record<TaskType, ExportFormat[]> = {
  classification: ['csv'],
  detection: ['coco', 'yolo', 'csv'],
  semantic_segmentation: ['coco', 'masks', 'yolo', 'csv'],
  instance_segmentation: ['coco', 'masks', 'yolo', 'csv'],
  change_detection: ['coco', 'masks', 'csv'],
};

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
  loadModel: (body: LoadModelBody) => request<ModelInfo>('/models/load', json('POST', body)),

  createSession: (file: File) => {
    const form = new FormData();
    form.append('image', file);
    return request<SessionState>('/sessions', { method: 'POST', body: form });
  },
  getSession: (sid: string) => request<SessionState>(`/sessions/${sid}`),
  deleteSession: (sid: string) => request<void>(`/sessions/${sid}`, { method: 'DELETE' }),
  imageUrl: (sid: string) => `/api/sessions/${sid}/image`,
  downloadMask: async (sid: string) => {
    const res = await raw(`/sessions/${sid}/mask`);
    const name = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') ?? '')?.[1];
    return { blob: await res.blob(), filename: name ?? 'mask.png' };
  },

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
  createUser: (body: { username: string; password: string; display_name: string;
                       role: User['role'] }) =>
    request<User>('/users', json('POST', body)),
  updateUser: (id: number, body: { display_name?: string; password?: string;
                                   role?: User['role']; active?: boolean }) =>
    request<User>(`/users/${id}`, json('PATCH', body)),

  // projects
  listProjects: () => request<Project[]>('/projects'),
  getProject: (id: number) => request<Project>(`/projects/${id}`),
  createProject: (body: { name: string; task_type: TaskType; multi_label: boolean;
                          protocol: string }) =>
    request<Project>('/projects', json('POST', body)),
  updateProject: (id: number, body: { name?: string; multi_label?: boolean; protocol?: string;
                                      assist_model_id?: number | null;
                                      auto_model_id?: number | null }) =>
    request<Project>(`/projects/${id}`, json('PATCH', body)),
  deleteProject: (id: number) => request<void>(`/projects/${id}`, { method: 'DELETE' }),
  progress: (id: number) => request<Progress>(`/projects/${id}/progress`),
  listClasses: (projectId: number) => request<LabelClass[]>(`/projects/${projectId}/classes`),
  createClass: (projectId: number, body: { name: string; color: string; label_type?: LabelType }) =>
    request<LabelClass>(`/projects/${projectId}/classes`, json('POST', body)),
  updateClass: (id: number, body: { name?: string; color?: string; position?: number }) =>
    request<LabelClass>(`/classes/${id}`, json('PATCH', body)),
  deleteClass: (id: number, force = false) =>
    request<void>(`/classes/${id}${force ? '?force=true' : ''}`, { method: 'DELETE' }),
  listItems: (projectId: number, status?: ItemStatus) =>
    request<Item[]>(`/projects/${projectId}/items${status ? `?status=${status}` : ''}`),

  uploadImages: (projectId: number, files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append('images', f);
    return request<Item[]>(`/projects/${projectId}/items`, { method: 'POST', body: form });
  },
  uploadPair: (projectId: number, before: File, after: File, name?: string) => {
    const form = new FormData();
    form.append('before', before);
    form.append('after', after);
    if (name) form.append('name', name);
    return request<Item[]>(`/projects/${projectId}/items`, { method: 'POST', body: form });
  },
  assign: (projectId: number, userId: number | null, itemIds: number[]) =>
    request<{ assigned: number }>(`/projects/${projectId}/assign`,
                                  json('POST', { user_id: userId, item_ids: itemIds })),

  // items + annotations
  getItem: (id: number) => request<Item>(`/items/${id}`),
  deleteItem: (id: number) => request<void>(`/items/${id}`, { method: 'DELETE' }),
  setItemStatus: (id: number, status: ItemStatus) =>
    request<Item>(`/items/${id}`, json('PATCH', { status })),
  startAssist: (itemId: number, image: string) =>
    request<SessionState>(`/items/${itemId}/assist?image=${encodeURIComponent(image)}`,
                          json('POST')),
  listAnnotations: (itemId: number) => request<Annotation[]>(`/items/${itemId}/annotations`),
  createAnnotation: (itemId: number,
                     body: { class_id: number; geometry?: Geometry; from_session?: string;
                             from_proposal?: { set: string; index: number } }) =>
    request<Annotation>(`/items/${itemId}/annotations`, json('POST', body)),
  autoLabel: (itemId: number) =>
    request<ProposalSet>(`/items/${itemId}/auto-label`, json('POST')),
  updateAnnotation: (id: number, body: { class_id?: number; geometry?: Geometry }) =>
    request<Annotation>(`/annotations/${id}`, json('PATCH', body)),
  deleteAnnotation: (id: number) => request<void>(`/annotations/${id}`, { method: 'DELETE' }),

  // project models (admins)
  projectModels: (projectId: number) => request<ProjectModels>(`/projects/${projectId}/models`),
  trainProjectModel: (projectId: number, body: { kind: TrainKind; checkpoint?: string;
                                                  steps?: number; teacher_id?: number }) =>
    request<ProjectModel>(`/projects/${projectId}/models`, json('POST', body)),
  cancelProjectModel: (id: number) =>
    request<ProjectModel>(`/project-models/${id}/cancel`, json('POST')),
  deleteProjectModel: (id: number) => request<void>(`/project-models/${id}`, { method: 'DELETE' }),
  downloadProjectModel: async (id: number) => {
    const res = await raw(`/project-models/${id}/download`);
    const name = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') ?? '')?.[1];
    return { blob: await res.blob(), filename: name ?? `model_${id}.zip` };
  },

  // audit log (admins)
  listAudit: (opts: { action?: string; before_id?: number; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.action) q.set('action', opts.action);
    if (opts.before_id !== undefined) q.set('before_id', String(opts.before_id));
    if (opts.limit !== undefined) q.set('limit', String(opts.limit));
    return request<AuditEntry[]>(`/audit?${q}`);
  },
  verifyAudit: () => request<{ ok: boolean; rows: number; first_bad_id: number | null }>(
    '/audit/verify'),

  // exports
  exportProject: async (projectId: number, format: ExportFormat,
                        opts: { status: ItemStatus | 'all'; include_images: boolean }) => {
    const q = new URLSearchParams({ format, status: opts.status,
                                    include_images: String(opts.include_images) });
    const res = await raw(`/projects/${projectId}/export?${q}`);
    const name = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') ?? '')?.[1];
    return { blob: await res.blob(), filename: name ?? `export.${format === 'csv' ? 'csv' : 'zip'}` };
  },
};
