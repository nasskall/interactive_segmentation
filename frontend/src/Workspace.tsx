import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import AnnotCanvas, { type Tool } from './AnnotCanvas';
import { api, TASK_LABELS, type Annotation, type Geometry, type Item, type ItemStatus,
         type LabelClass, type LabelType, type ModelInfo, type Project, type SessionState,
         type User } from './api';
import { go, href, useAuthedImage, useImage } from './lib';

const TOOL_INFO: Record<Tool, { label: string; key: string; help: string }> = {
  assist: { label: 'Assist', key: 'a',
            help: 'Click to add, right-click to remove, drag a box (SAM). Enter saves.' },
  box: { label: 'Box', key: 'r', help: 'Drag to draw a box, or tick Fixed size and click.' },
  polygon: { label: 'Polygon', key: 'p',
             help: 'Click to add points; click the first point or press Enter to close. '
                   + 'Backspace removes a point.' },
  select: { label: 'Select', key: 'v', help: 'Click an annotation to select it.' },
};

// Label types each tool can produce.
const TOOL_TYPES: Record<Tool, LabelType[]> = {
  assist: ['mask', 'polygon', 'box'],
  box: ['box'],
  polygon: ['polygon'],
  select: [],
};

function toolsFor(project: Project, classes: LabelClass[]): Tool[] {
  if (project.task_type === 'classification') return [];
  const types = new Set(classes.map((c) => c.label_type));
  const tools: Tool[] = [];
  if (project.task_type === 'detection') tools.push('box');
  tools.push('assist');
  if (types.has('polygon')) tools.push('polygon');
  if (types.has('box') && !tools.includes('box')) tools.push('box');
  tools.push('select');
  return tools;
}

// The tool last used in each project, kept while moving between its items.
const lastTool = new Map<number, Tool>();

// Fixed-size boxes: one setting per project, remembered in this browser.
interface FixedBox { on: boolean; w: number; h: number }
const DEFAULT_FIXED_BOX: FixedBox = { on: false, w: 64, h: 64 };
const fixedBoxKey = (projectId: number) => `iseg.fixedBox.${projectId}`;

function loadFixedBox(projectId: number): FixedBox {
  try {
    const v = JSON.parse(localStorage.getItem(fixedBoxKey(projectId)) ?? 'null');
    if (v && typeof v.on === 'boolean' && v.w >= 1 && v.h >= 1) return { on: v.on, w: v.w, h: v.h };
  } catch {
    /* storage unavailable or bad value */
  }
  return DEFAULT_FIXED_BOX;
}

function saveFixedBox(projectId: number, v: FixedBox) {
  try {
    localStorage.setItem(fixedBoxKey(projectId), JSON.stringify(v));
  } catch {
    /* storage unavailable: keep it for this page only */
  }
}

const describe = (g: Geometry) =>
  g.type === 'box' ? `box ${Math.round(g.bbox[2])}×${Math.round(g.bbox[3])}`
    : g.type === 'polygon' ? `polygon · ${g.points.length} pts`
    : g.type === 'mask' ? 'mask' : 'tag';

interface Props {
  itemId: number;
  me: User;
}

export default function Workspace({ itemId, me }: Props) {
  const [item, setItem] = useState<Item | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [siblings, setSiblings] = useState<Item[]>([]);
  const [users, setUsers] = useState<Map<number, string>>(new Map());
  const [models, setModels] = useState<ModelInfo | null>(null);

  const [classId, setClassId] = useState<number | null>(null);
  const [tool, setTool] = useState<Tool>('assist');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [opacity, setOpacity] = useState(0.5);
  const [fixedBox, setFixedBox] = useState<FixedBox>(DEFAULT_FIXED_BOX);
  const [role, setRole] = useState('image');      // which image of the item is shown
  const [swipe, setSwipe] = useState<number | null>(null);
  const [sessions, setSessions] = useState<Record<string, SessionState>>({});

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const sessionIds = useRef<string[]>([]);
  // One model session per image of the item, created at most once.
  const pending = useRef<Record<string, Promise<SessionState>>>({});

  // ── loading ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let live = true;
    setItem(null);
    setSessions({});
    setSelectedId(null);
    setHidden(new Set());
    setSwipe(null);
    setError(null);
    (async () => {
      const it = await api.getItem(itemId);
      const [proj, cls, anns, items, mi] = await Promise.all([
        api.getProject(it.project_id), api.listClasses(it.project_id),
        api.listAnnotations(itemId), api.listItems(it.project_id), api.getModels()]);
      if (!live) return;
      setItem(it);
      setProject(proj);
      setFixedBox(loadFixedBox(proj.id));
      setClasses(cls);
      setAnnotations(anns);
      setSiblings(items);
      setModels(mi);
      setRole(it.kind === 'pair' ? 'after' : 'image');
      setClassId((cur) => (cls.some((c) => c.id === cur) ? cur : cls[0]?.id ?? null));
      const tools = toolsFor(proj, cls);
      const remembered = lastTool.get(proj.id);
      setTool(remembered && tools.includes(remembered) ? remembered : tools[0] ?? 'select');
      setStatus(`${it.name} · ${it.width}×${it.height}`);
      // Encode the image for the model now, so the first click is quick.
      if (tools.includes('assist')) {
        const r = it.kind === 'pair' ? 'after' : 'image';
        setStatus('Preparing the model for this image…');
        startSession(it.id, r)
          .then(() => live && setStatus(`${it.name} · ${it.width}×${it.height} · model ready`))
          .catch(() => {});
      }
    })().catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [itemId]);

  useEffect(() => {
    if (me.role !== 'admin') return;
    api.listUsers().then((us) => setUsers(new Map(us.map((u) => [u.id, u.display_name]))))
      .catch(() => {});
  }, [me.role]);

  // Close this item's model sessions when leaving it.
  useEffect(() => () => {
    for (const sid of sessionIds.current) api.deleteSession(sid).catch(() => {});
    sessionIds.current = [];
    pending.current = {};
  }, [itemId]);

  const classMap = useMemo(() => new Map(classes.map((c) => [c.id, c])), [classes]);
  const tools = project ? toolsFor(project, classes) : [];
  const cls = classId !== null ? classMap.get(classId) ?? null : null;
  const session = sessions[role] ?? null;
  const image = useAuthedImage(item?.images[role] ?? null);
  const before = useAuthedImage(item?.kind === 'pair' ? item.images.before : null);
  const overlay = useImage(session?.overlay ?? null);
  const index = siblings.findIndex((s) => s.id === itemId);
  const isSam = models?.current.model_type.startsWith('SAM') ?? false;

  // ── running backend calls ───────────────────────────────────────────────
  const run = useCallback(async (label: string, fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setStatus(`${label}…`);
    const t0 = performance.now();
    try {
      await fn();
      setStatus(`${label} · ${Math.round(performance.now() - t0)} ms`);
    } catch (e) {
      setError(`${label}: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, []);

  const putSession = (r: string, s: SessionState) => setSessions((all) => ({ ...all, [r]: s }));

  function startSession(id: number, r: string): Promise<SessionState> {
    if (!pending.current[r]) {
      pending.current[r] = api.startAssist(id, r).then((s) => {
        sessionIds.current.push(s.session_id);
        putSession(r, s);
        return s;
      });
      pending.current[r].catch(() => { delete pending.current[r]; });
    }
    return pending.current[r];
  }

  const ensureSession = async (): Promise<SessionState> => {
    if (sessions[role]) return sessions[role];
    if (!item) throw new Error('No item');
    setStatus('Preparing the model for this image…');
    return startSession(item.id, role);
  };

  const toSession = (s: SessionState, v: number) => Math.round((v * s.width) / item!.width);

  const assistClick = (x: number, y: number, positive: boolean) =>
    run(positive ? 'Add' : 'Remove', async () => {
      const s = await ensureSession();
      const sx = Math.min(toSession(s, x), s.width - 1), sy = Math.min(toSession(s, y), s.height - 1);
      putSession(role, await api.addClick(s.session_id, { x: sx, y: sy, positive }));
    });

  const assistBox = (x0: number, y0: number, x1: number, y1: number) =>
    run('Box prompt', async () => {
      if (!isSam) throw new Error('Box prompts need a SAM model; click instead, or switch model');
      const s = await ensureSession();
      putSession(role, await api.setBox(s.session_id, {
        x0: toSession(s, x0), y0: toSession(s, y0),
        x1: Math.min(toSession(s, x1), s.width), y1: Math.min(toSession(s, y1), s.height) }));
    });

  const addAnnotation = (a: Annotation) => {
    setAnnotations((all) => [...all, a]);
    setSelectedId(a.id);
    setItem((it) => (it && it.status === 'todo' ? { ...it, status: 'in_progress' } : it));
  };

  const saveAssisted = () => {
    if (!session?.has_incomplete_object || !item) return;
    if (!cls || !TOOL_TYPES.assist.includes(cls.label_type)) {
      setError('Pick a class to save this object under');
      return;
    }
    run(`Save ${cls.name}`, async () => {
      addAnnotation(await api.createAnnotation(item.id, { class_id: cls.id,
                                                          from_session: session.session_id }));
      putSession(role, await api.getSession(session.session_id));
    });
  };

  const saveGeometry = (geometry: Geometry, type: LabelType) => {
    if (!item) return;
    if (!cls || cls.label_type !== type) {
      setError(`Pick a ${type} class first`);
      return;
    }
    run(`Save ${cls.name}`, async () => {
      addAnnotation(await api.createAnnotation(item.id, { class_id: cls.id, geometry }));
    });
  };

  const sessionAction = (label: string, fn: (sid: string) => Promise<SessionState>) => {
    if (!session) return;
    run(label, async () => putSession(role, await fn(session.session_id)));
  };

  const deleteAnnotation = (id: number) =>
    run('Delete', async () => {
      await api.deleteAnnotation(id);
      setAnnotations((all) => all.filter((a) => a.id !== id));
      setSelectedId((cur) => (cur === id ? null : cur));
    });

  const changeClass = (a: Annotation, newClassId: number) =>
    run('Change class', async () => {
      const updated = await api.updateAnnotation(a.id, { class_id: newClassId });
      setAnnotations((all) => all.map((x) => (x.id === a.id ? updated : x)));
    });

  // Classification: tags are toggled (multi-label) or replaced (single).
  const toggleTag = (c: LabelClass) => {
    if (!item || !project) return;
    const existing = annotations.find((a) => a.class_id === c.id);
    run(existing ? `Remove ${c.name}` : `Tag ${c.name}`, async () => {
      if (existing) {
        await api.deleteAnnotation(existing.id);
      } else {
        await api.createAnnotation(item.id, { class_id: c.id });
      }
      setAnnotations(await api.listAnnotations(item.id));
      setItem((it) => (it && it.status === 'todo' ? { ...it, status: 'in_progress' } : it));
    });
  };

  const setItemStatus = (s: ItemStatus, thenNext = false) => {
    if (!item) return;
    run(s === 'done' ? 'Mark done' : 'Update status', async () => {
      const updated = await api.setItemStatus(item.id, s);
      setItem(updated);
      setSiblings((all) => all.map((x) => (x.id === updated.id ? updated : x)));
      if (thenNext) {
        const next = siblings.slice(index + 1).find((x) => x.status !== 'done')
          ?? siblings.find((x) => x.status !== 'done' && x.id !== item.id);
        if (next) go(href.item(next.id));
        else setStatus('All items in this project are done');
      }
    });
  };

  const loadModel = (modelType: string, checkpoint: string) =>
    run(`Load ${modelType}`, async () => {
      setModels(await api.loadModel(modelType, checkpoint));
      // Loading a model resets every session's current object.
      const fresh: Record<string, SessionState> = {};
      for (const [r, s] of Object.entries(sessions)) fresh[r] = await api.getSession(s.session_id);
      setSessions(fresh);
    });

  const pickClass = (c: LabelClass) => {
    if (project?.task_type === 'classification') toggleTag(c);
    else {
      setClassId(c.id);
      // A manual tool that cannot draw this class gives way to one that can.
      if (!TOOL_TYPES[tool].includes(c.label_type) && tool !== 'select') {
        setTool(c.label_type === 'box' && tools.includes('box') ? 'box' : 'assist');
      }
    }
  };

  const updateFixedBox = (patch: Partial<FixedBox>) => {
    const next = { ...fixedBox, ...patch };
    setFixedBox(next);
    if (project) saveFixedBox(project.id, next);
  };

  const pickTool = (t: Tool) => {
    setTool(t);
    if (project) lastTool.set(project.id, t);
    if (t !== 'select' && cls && !TOOL_TYPES[t].includes(cls.label_type)) {
      const fit = classes.find((c) => TOOL_TYPES[t].includes(c.label_type));
      if (fit) setClassId(fit.id);
    }
  };

  // ── keyboard ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) return;
      if (busy && e.key !== 'Escape') return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === 'z') {
        e.preventDefault();
        if (session?.can_undo) sessionAction('Undo', api.undo);
      } else if ((e.key === 'Enter' || e.key === ' ') && session?.has_incomplete_object) {
        e.preventDefault();
        saveAssisted();
      } else if (e.key === 'Escape') {
        if (session?.has_incomplete_object) sessionAction('Clear', api.reset);
        else setSelectedId(null);
      } else if (e.key === 'Delete' || (e.key === 'Backspace' && tool === 'select')) {
        if (selectedId !== null) deleteAnnotation(selectedId);
      } else if (/^[1-9]$/.test(e.key)) {
        const c = classes[Number(e.key) - 1];
        if (c) pickClass(c);
      } else if (k === 'b' && item?.kind === 'pair') {
        setRole((r) => (r === 'after' ? 'before' : 'after'));
      } else if (k === ']' || e.key === 'PageDown') {
        if (siblings[index + 1]) go(href.item(siblings[index + 1].id));
      } else if (k === '[' || e.key === 'PageUp') {
        if (index > 0) go(href.item(siblings[index - 1].id));
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const t = tools.find((x) => TOOL_INFO[x].key === k);
        if (t) pickTool(t);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (error && !item) {
    return <div className="page"><p className="error-text">{error}</p>
      <a href={href.projects()}>Back to projects</a></div>;
  }
  if (!item || !project) return <div className="page muted">Loading…</div>;

  const isClassification = project.task_type === 'classification';
  const tagged = new Set(annotations.map((a) => a.class_id));
  const selected = annotations.find((a) => a.id === selectedId) ?? null;
  const labellerName = (id: number | null) =>
    id === me.id ? 'you' : id === null ? 'deleted user' : users.get(id) ?? `user ${id}`;

  return (
    <div className="workspace">
      <aside className="sidebar">
        <nav className="crumbs">
          <a href={href.projects()}>Projects</a> › <a href={href.project(project.id)}>{project.name}</a>
        </nav>

        <section>
          <div className="item-head">
            <strong title={item.name}>{item.name}</strong>
            <span className={`badge status-${item.status}`}>{item.status.replace('_', ' ')}</span>
          </div>
          <p className="meta">
            {TASK_LABELS[project.task_type]} · {item.width}×{item.height}
            {index >= 0 && ` · ${index + 1} of ${siblings.length}`}
          </p>
          <div className="row">
            <button disabled={index <= 0} onClick={() => go(href.item(siblings[index - 1].id))}
                    title="Previous ( [ )">‹ Prev</button>
            <button disabled={index < 0 || index >= siblings.length - 1}
                    onClick={() => go(href.item(siblings[index + 1].id))}
                    title="Next ( ] )">Next ›</button>
          </div>
          {item.status === 'done'
            ? <button onClick={() => setItemStatus('in_progress')} disabled={busy}>Reopen</button>
            : <button className="primary" onClick={() => setItemStatus('done', true)}
                      disabled={busy}>Mark done &amp; next</button>}
        </section>

        {project.protocol && (
          <section>
            <details open>
              <summary><h2>Protocol</h2></summary>
              <p className="protocol">{project.protocol}</p>
            </details>
          </section>
        )}

        <section>
          <h2>Classes</h2>
          {classes.length === 0 && <p className="meta">This project has no classes yet.</p>}
          <ul className="class-list">
            {classes.map((c, i) => {
              const usable = isClassification || tool === 'select'
                || TOOL_TYPES[tool].includes(c.label_type);
              const active = isClassification ? tagged.has(c.id) : c.id === classId;
              return (
                <li key={c.id}>
                  <button className={`class-btn${active ? ' active' : ''}`}
                          disabled={busy || !usable} onClick={() => pickClass(c)}
                          title={usable ? undefined : `Not drawable with the ${tool} tool`}>
                    <span className="swatch" style={{ background: c.color }} />
                    <span className="grow">{c.name}</span>
                    {!isClassification && <span className="muted small">{c.label_type}</span>}
                    {i < 9 && <kbd>{i + 1}</kbd>}
                  </button>
                </li>
              );
            })}
          </ul>
          {isClassification && (
            <p className="meta">
              {project.multi_label ? 'Toggle every class that applies.' : 'Pick one class.'}
            </p>
          )}
        </section>

        {tools.length > 0 && (
          <section>
            <h2>Tools</h2>
            <div className="segmented">
              {tools.map((t) => (
                <button key={t} className={t === tool ? 'active' : ''} onClick={() => pickTool(t)}
                        title={`${TOOL_INFO[t].label} (${TOOL_INFO[t].key.toUpperCase()})`}>
                  {TOOL_INFO[t].label}
                </button>
              ))}
            </div>
            <p className="meta">{TOOL_INFO[tool].help}</p>
            {tool === 'assist' && (
              <>
                <div className="row">
                  <button disabled={busy || !session?.can_undo}
                          onClick={() => sessionAction('Undo', api.undo)}>Undo</button>
                  <button disabled={busy || !session?.has_incomplete_object}
                          onClick={() => sessionAction('Clear', api.reset)}>Clear</button>
                </div>
                <button className="primary" onClick={saveAssisted}
                        disabled={busy || !session?.has_incomplete_object}>
                  Save{cls ? ` as ${cls.name}` : ''} (Enter)
                </button>
                <ModelPicker models={models} admin={me.role === 'admin'} busy={busy}
                             onLoad={loadModel} />
              </>
            )}
            {tool === 'box' && (
              <>
                <label className="check">
                  <input type="checkbox" checked={fixedBox.on}
                         onChange={(e) => updateFixedBox({ on: e.target.checked })} />
                  Fixed size (click to place)
                </label>
                {fixedBox.on && (
                  <div className="row">
                    {(['w', 'h'] as const).map((k) => (
                      <label key={k}>{k === 'w' ? 'Width' : 'Height'} (px)
                        <input type="number" min={1} step={1}
                               max={k === 'w' ? item.width : item.height} value={fixedBox[k]}
                               onChange={(e) => {
                                 const v = Math.round(Number(e.target.value));
                                 if (v >= 1) updateFixedBox({ [k]: v });
                               }} />
                      </label>
                    ))}
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {item.kind === 'pair' && (
          <section>
            <h2>Images</h2>
            <div className="segmented">
              {['before', 'after'].map((r) => (
                <button key={r} className={role === r && swipe === null ? 'active' : ''}
                        onClick={() => { setRole(r); setSwipe(null); }}>{r}</button>
              ))}
              <button className={swipe !== null ? 'active' : ''}
                      onClick={() => { setRole('after'); setSwipe(swipe ?? 0.5); }}>Compare</button>
            </div>
            {swipe !== null && (
              <label>Before ◂ divider ▸ after
                <input type="range" min={0} max={1} step={0.01} value={swipe}
                       onChange={(e) => setSwipe(Number(e.target.value))} />
              </label>
            )}
            <p className="meta">B switches images. The model assists on the image shown
              {swipe !== null ? ' right of the divider (after)' : ''}.</p>
          </section>
        )}

        <section>
          <h2>Annotations ({annotations.length})</h2>
          <ul className="ann-list">
            {annotations.map((a) => {
              const c = classMap.get(a.class_id);
              return (
                <li key={a.id} className={a.id === selectedId ? 'selected' : ''}
                    onClick={() => setSelectedId(a.id === selectedId ? null : a.id)}>
                  <span className="swatch" style={{ background: c?.color }} />
                  <span className="grow">
                    {c?.name ?? 'deleted class'}
                    <span className="muted small"> · {describe(a.geometry)}
                      {a.source === 'assisted' && ' · assisted'} · {labellerName(a.labeller_id)}
                    </span>
                  </span>
                  {a.geometry.type !== 'tag' && (
                    <button className="icon" title={hidden.has(a.id) ? 'Show' : 'Hide'}
                            onClick={(e) => {
                              e.stopPropagation();
                              const next = new Set(hidden);
                              if (next.has(a.id)) next.delete(a.id); else next.add(a.id);
                              setHidden(next);
                            }}>{hidden.has(a.id) ? '◌' : '●'}</button>
                  )}
                </li>
              );
            })}
          </ul>
          {selected && (
            <div className="selected-panel">
              <label>Class
                <select value={selected.class_id} disabled={busy}
                        onChange={(e) => changeClass(selected, Number(e.target.value))}>
                  {classes.filter((c) => c.label_type === classMap.get(selected.class_id)?.label_type)
                    .map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </label>
              <button className="danger" disabled={busy}
                      onClick={() => deleteAnnotation(selected.id)}>Delete (Del)</button>
            </div>
          )}
          {!isClassification && (
            <label>Fill opacity: {opacity.toFixed(2)}
              <input type="range" min={0.1} max={1} step={0.05} value={opacity}
                     onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
          )}
        </section>

        <section className="help">
          {['1–9 class', ...tools.map((t) => `${TOOL_INFO[t].key.toUpperCase()} ${TOOL_INFO[t].label.toLowerCase()}`)].join(' · ')}
          <br />
          {isClassification ? '[ ] prev/next'
            : 'Wheel zoom · Space/middle-drag pan · Del delete · [ ] prev/next'}
        </section>
      </aside>

      <main className="stage">
        {isClassification ? (
          <ClassificationView image={image} />
        ) : (
          <AnnotCanvas
            width={item.width}
            height={item.height}
            image={image}
            swipeImage={swipe !== null ? before : null}
            swipe={swipe ?? undefined}
            annotations={annotations}
            classes={classMap}
            hidden={hidden}
            selectedId={selectedId}
            opacity={opacity}
            session={session}
            sessionOverlay={overlay}
            tool={tool}
            fixedBox={fixedBox.on ? [fixedBox.w, fixedBox.h] : null}
            drawColor={cls?.color ?? '#facc15'}
            busy={busy}
            onAssistClick={assistClick}
            onAssistBox={assistBox}
            onBox={(bbox) => saveGeometry({ type: 'box', bbox: bbox.map((v) => Math.round(v * 10) / 10) as [number, number, number, number] }, 'box')}
            onPolygon={(pts) => saveGeometry({ type: 'polygon', points: pts.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10] as [number, number]) }, 'polygon')}
            onSelect={setSelectedId}
          />
        )}
        <footer className={error ? 'status error' : 'status'}>{error ?? status}</footer>
      </main>
    </div>
  );
}

function ClassificationView({ image }: { image: HTMLImageElement | null }) {
  return (
    <div className="annot-stage">
      {image ? <img className="classify-img" src={image.src} alt="" /> : <span className="muted">Loading…</span>}
    </div>
  );
}

function ModelPicker({ models, admin, busy, onLoad }: {
  models: ModelInfo | null; admin: boolean; busy: boolean;
  onLoad: (modelType: string, checkpoint: string) => void;
}) {
  const [type, setType] = useState('');
  const [ckpt, setCkpt] = useState('');
  useEffect(() => {
    if (models) {
      setType(models.current.model_type);
      setCkpt(models.current.checkpoint ?? '');
    }
  }, [models]);
  if (!models) return null;
  const current = `${models.current.model_type} · ${models.device}`;
  if (!admin) return <p className="meta">Model: {current}</p>;
  return (
    <details>
      <summary className="meta">Model: {current}</summary>
      <label>Type
        <select value={type} onChange={(e) => setType(e.target.value)} disabled={busy}>
          {models.model_types.map((t) => <option key={t}>{t}</option>)}
        </select>
      </label>
      <label>Checkpoint
        <select value={ckpt} onChange={(e) => setCkpt(e.target.value)} disabled={busy}>
          {models.checkpoints.map((c) => <option key={c}>{c}</option>)}
        </select>
      </label>
      <button disabled={busy || !ckpt} onClick={() => onLoad(type, ckpt)}>Load for everyone</button>
    </details>
  );
}
