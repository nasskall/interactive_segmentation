// Admin screens: projects, classes, uploads, assignment, team progress, users.

import { useEffect, useMemo, useState, type DragEvent, type FormEvent } from 'react';
import { api, TASK_LABELS, type Item, type LabelClass, type LabelType, type Progress,
         type Project, type TaskType, type User } from './api';
import { go, href } from './lib';

const TASK_HELP: Record<TaskType, string> = {
  classification: 'One or more labels per image.',
  detection: 'Bounding boxes around objects.',
  semantic_segmentation: 'Pixel masks per class (or polygons).',
  instance_segmentation: 'A separate mask or polygon for each object.',
  change_detection: 'Before/after image pairs; mark what changed.',
};

const PALETTE = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#a855f7', '#ec4899', '#14b8a6',
                 '#eab308', '#6366f1', '#f97316'];

function useAsync() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, setError, run };
}

/** Two-step button: the first click asks, the second acts. No browser dialogs. */
export function ConfirmButton({ label, confirm, onConfirm, disabled }: {
  label: string; confirm: string; onConfirm: () => void; disabled?: boolean;
}) {
  const [asking, setAsking] = useState(false);
  if (!asking) {
    return <button className="danger" disabled={disabled} onClick={() => setAsking(true)}>{label}</button>;
  }
  return (
    <span className="confirm">
      <span className="small">{confirm}</span>
      <button className="danger-solid" onClick={() => { setAsking(false); onConfirm(); }}>Yes</button>
      <button onClick={() => setAsking(false)}>Cancel</button>
    </span>
  );
}

// ── new project ────────────────────────────────────────────────────────────

export function NewProject({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState('');
  const [task, setTask] = useState<TaskType>('instance_segmentation');
  const [multi, setMulti] = useState(false);
  const [protocol, setProtocol] = useState('');
  const { busy, error, run } = useAsync();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      const p = await api.createProject({ name: name.trim(), task_type: task, multi_label: multi,
                                          protocol });
      go(href.project(p.id, 'classes'));
    });
  };

  return (
    <form className="card form" onSubmit={submit}>
      <h2 className="form-title">New project</h2>
      <label>Name<input autoFocus value={name} onChange={(e) => setName(e.target.value)}
                        maxLength={200} /></label>
      <fieldset>
        <legend>Task</legend>
        <div className="task-grid">
          {(Object.keys(TASK_LABELS) as TaskType[]).map((t) => (
            <label key={t} className={`task-option${task === t ? ' active' : ''}`}>
              <input type="radio" name="task" checked={task === t} onChange={() => setTask(t)} />
              <strong>{TASK_LABELS[t]}</strong>
              <span className="muted small">{TASK_HELP[t]}</span>
            </label>
          ))}
        </div>
      </fieldset>
      {task === 'classification' && (
        <label className="check"><input type="checkbox" checked={multi}
                                        onChange={(e) => setMulti(e.target.checked)} />
          Multi-label: an image can have several classes</label>
      )}
      <label>Protocol <span className="muted small">(guidelines shown to labellers)</span>
        <textarea rows={4} value={protocol} onChange={(e) => setProtocol(e.target.value)} />
      </label>
      {error && <p className="error-text">{error}</p>}
      <div className="row-end">
        <button type="button" onClick={onCancel}>Cancel</button>
        <button className="primary" disabled={busy || !name.trim()}>Create project</button>
      </div>
    </form>
  );
}

// ── classes ────────────────────────────────────────────────────────────────

export function ClassesEditor({ project }: { project: Project }) {
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [name, setName] = useState('');
  const [color, setColor] = useState(PALETTE[0]);
  const [type, setType] = useState<LabelType>(project.allowed_label_types[0]);
  const [inUse, setInUse] = useState<number | null>(null);
  // Colours being picked, saved when the picker loses focus.
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const { busy, error, setError, run } = useAsync();

  const reload = async () => {
    const cs = await api.listClasses(project.id);
    setClasses(cs);
    setColor(PALETTE[cs.length % PALETTE.length]);
  };
  useEffect(() => { reload().catch((e) => setError(e.message)); }, [project.id]);

  const add = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      await api.createClass(project.id, { name: name.trim(), color, label_type: type });
      setName('');
      await reload();
    });
  };

  const update = (c: LabelClass, body: { name?: string; color?: string }) =>
    run(async () => {
      const updated = await api.updateClass(c.id, body);
      setClasses((all) => all.map((x) => (x.id === c.id ? updated : x)));
    });

  // Swap with the neighbour, renumbering positions to 0..n-1 on the way.
  const move = (i: number, d: -1 | 1) =>
    run(async () => {
      const order = classes.map((c) => c.id);
      [order[i], order[i + d]] = [order[i + d], order[i]];
      for (let k = 0; k < order.length; k++) {
        if (classes.find((c) => c.id === order[k])!.position !== k) {
          await api.updateClass(order[k], { position: k });
        }
      }
      await reload();
    });

  const remove = (c: LabelClass, force: boolean) =>
    run(async () => {
      try {
        await api.deleteClass(c.id, force);
        setInUse(null);
        await reload();
      } catch (e) {
        if ((e as Error).message.includes('annotations use this class')) setInUse(c.id);
        else throw e;
      }
    });

  return (
    <div className="stack">
      {classes.length === 0 && <p className="muted">No classes yet. Add the first one below.</p>}
      <ul className="class-editor">
        {classes.map((c, i) => (
          <li key={c.id}>
            <input type="color" value={drafts[c.id] ?? c.color} title="Colour"
                   onChange={(e) => setDrafts({ ...drafts, [c.id]: e.target.value })}
                   onBlur={() => {
                     const draft = drafts[c.id];
                     if (draft && draft !== c.color) update(c, { color: draft });
                     setDrafts((all) => { const next = { ...all }; delete next[c.id]; return next; });
                   }} />
            <input className="grow" defaultValue={c.name} key={`${c.id}-${c.name}`}
                   onBlur={(e) => e.target.value.trim() && e.target.value.trim() !== c.name
                     && update(c, { name: e.target.value.trim() })} />
            <span className="muted small type-tag">{c.label_type}</span>
            <button className="icon" title="Move up" disabled={busy || i === 0}
                    onClick={() => move(i, -1)}>↑</button>
            <button className="icon" title="Move down" disabled={busy || i === classes.length - 1}
                    onClick={() => move(i, 1)}>↓</button>
            {inUse === c.id
              ? <span className="confirm"><span className="small">Has annotations.</span>
                  <button className="danger-solid" onClick={() => remove(c, true)}>Delete them too</button>
                  <button onClick={() => setInUse(null)}>Cancel</button></span>
              : <ConfirmButton label="Delete" confirm={`Delete ${c.name}?`} disabled={busy}
                               onConfirm={() => remove(c, false)} />}
          </li>
        ))}
      </ul>
      <form className="class-add" onSubmit={add}>
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} title="Colour" />
        <input className="grow" placeholder="New class name" value={name}
               onChange={(e) => setName(e.target.value)} maxLength={100} />
        {project.allowed_label_types.length > 1 && (
          <select value={type} onChange={(e) => setType(e.target.value as LabelType)}
                  title="Label type">
            {project.allowed_label_types.map((t) => <option key={t}>{t}</option>)}
          </select>
        )}
        <button className="primary" disabled={busy || !name.trim()}>Add class</button>
      </form>
      <p className="meta">
        {project.allowed_label_types.length > 1
          ? `Each class is labelled as a ${project.allowed_label_types.join(' or a ')}. `
            + 'The model assist saves any of them; a class\'s type cannot change later.'
          : `Classes in this project are ${project.allowed_label_types[0]} labels.`}
        {' '}Names and colours save when you leave the field.
      </p>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

// ── settings ───────────────────────────────────────────────────────────────

export function ProjectSettings({ project, onSaved }: {
  project: Project; onSaved: (p: Project) => void;
}) {
  const [name, setName] = useState(project.name);
  const [protocol, setProtocol] = useState(project.protocol);
  const [multi, setMulti] = useState(project.multi_label);
  const [confirmName, setConfirmName] = useState('');
  const [saved, setSaved] = useState(false);
  const { busy, error, run } = useAsync();
  const dirty = name !== project.name || protocol !== project.protocol
    || multi !== project.multi_label;

  const save = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      onSaved(await api.updateProject(project.id, { name: name.trim(), protocol,
                                                    multi_label: multi }));
      setSaved(true);
    });
  };

  return (
    <div className="stack">
      <form className="card form" onSubmit={save}>
        <label>Name<input value={name} maxLength={200}
                          onChange={(e) => { setName(e.target.value); setSaved(false); }} /></label>
        <p className="meta">Task: {TASK_LABELS[project.task_type]} (fixed once created)</p>
        {project.task_type === 'classification' && (
          <label className="check"><input type="checkbox" checked={multi}
                                          onChange={(e) => { setMulti(e.target.checked); setSaved(false); }} />
            Multi-label</label>
        )}
        <label>Protocol
          <textarea rows={8} value={protocol}
                    onChange={(e) => { setProtocol(e.target.value); setSaved(false); }} />
        </label>
        {error && <p className="error-text">{error}</p>}
        <div className="row-end">
          {saved && !dirty && <span className="muted small">Saved</span>}
          <button className="primary" disabled={busy || !dirty || !name.trim()}>Save</button>
        </div>
      </form>

      <div className="card danger-zone">
        <h2 className="form-title">Delete project</h2>
        <p className="meta">Removes every image and annotation in it. This cannot be undone.
          Type the project name to confirm.</p>
        <div className="row">
          <input value={confirmName} onChange={(e) => setConfirmName(e.target.value)}
                 placeholder={project.name} />
          <button className="danger-solid" disabled={busy || confirmName !== project.name}
                  onClick={() => run(async () => { await api.deleteProject(project.id);
                                                   go(href.projects()); })}>
            Delete project
          </button>
        </div>
      </div>
    </div>
  );
}

// ── upload ─────────────────────────────────────────────────────────────────

const BATCH = 8;

/** Shared start of the two file names, e.g. site4_before.png + site4_after.png → site4. */
function pairName(before: File, after: File): string {
  const stem = (f: File) => f.name.replace(/\.[^.]+$/, '');
  const a = stem(before), b = stem(after);
  let n = 0;
  while (n < a.length && a[n] === b[n]) n++;
  return a.slice(0, n).replace(/[\s._-]+$/, '') || a;
}
const IMAGE_EXTS = /\.(png|jpe?g|bmp|tiff?|webp)$/i;

export function UploadPanel({ project, onUploaded }: {
  project: Project; onUploaded: () => void;
}) {
  const pairs = project.task_type === 'change_detection';
  const [files, setFiles] = useState<File[]>([]);
  const [after, setAfter] = useState<File[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [failures, setFailures] = useState<string[]>([]);
  const [over, setOver] = useState(false);
  const { busy, error, run } = useAsync();

  const pick = (list: FileList | null, set: (f: File[]) => void) => {
    if (!list) return;
    set([...list].filter((f) => IMAGE_EXTS.test(f.name))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })));
  };

  const upload = () =>
    run(async () => {
      const failed: string[] = [];
      if (pairs) {
        for (let i = 0; i < files.length; i++) {
          setProgress(`Uploading pair ${i + 1} of ${files.length}…`);
          try {
            await api.uploadPair(project.id, files[i], after[i], pairName(files[i], after[i]));
          } catch (e) {
            failed.push(`${files[i].name} + ${after[i].name}: ${(e as Error).message}`);
          }
        }
      } else {
        for (let i = 0; i < files.length; i += BATCH) {
          const batch = files.slice(i, i + BATCH);
          setProgress(`Uploading ${Math.min(i + BATCH, files.length)} of ${files.length}…`);
          try {
            await api.uploadImages(project.id, batch);
          } catch {
            // Retry one by one so a single bad file does not sink the batch.
            for (const f of batch) {
              try {
                await api.uploadImages(project.id, [f]);
              } catch (e) {
                failed.push(`${f.name}: ${(e as Error).message}`);
              }
            }
          }
        }
      }
      const total = files.length;
      setProgress(`Uploaded ${total - failed.length} of ${total}${pairs ? ' pairs' : ' images'}.`);
      setFailures(failed);
      setFiles([]);
      setAfter([]);
      onUploaded();
    });

  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    if (!pairs) pick(e.dataTransfer.files, setFiles);
  };

  const pairMismatch = pairs && files.length !== after.length;
  const ready = files.length > 0 && !pairMismatch;

  return (
    <div className="card stack">
      <h2 className="form-title">Add images</h2>
      {pairs ? (
        <>
          <div className="pair-pick">
            <label>Before images<input type="file" accept="image/*" multiple
                                       onChange={(e) => pick(e.target.files, setFiles)} /></label>
            <label>After images<input type="file" accept="image/*" multiple
                                      onChange={(e) => pick(e.target.files, setAfter)} /></label>
          </div>
          <p className="meta">Pairs are matched by file-name order; each pair must be the same size.
            {files.length > 0 && ` ${files.length} before, ${after.length} after.`}</p>
          {pairMismatch && after.length > 0 && <p className="error-text">Choose as many after images as before images.</p>}
          {ready && (
            <ul className="pair-list">
              {files.slice(0, 5).map((f, i) => <li key={f.name}>{f.name} → {after[i].name}</li>)}
              {files.length > 5 && <li className="muted">…and {files.length - 5} more</li>}
            </ul>
          )}
        </>
      ) : (
        <label className={`dropzone${over ? ' over' : ''}`}
               onDragOver={(e) => { e.preventDefault(); setOver(true); }}
               onDragLeave={() => setOver(false)} onDrop={drop}>
          <input type="file" accept="image/*" multiple hidden
                 onChange={(e) => { pick(e.target.files, setFiles); e.target.value = ''; }} />
          {files.length ? `${files.length} image${files.length > 1 ? 's' : ''} ready to upload`
            : 'Drop images here, or click to choose'}
          <span className="muted small">PNG, JPEG, TIFF, BMP or WebP</span>
        </label>
      )}
      <div className="row-end">
        {progress && <span className="muted small">{progress}</span>}
        {files.length > 0 && !busy && <button onClick={() => { setFiles([]); setAfter([]); }}>Clear</button>}
        <button className="primary" disabled={busy || !ready} onClick={upload}>Upload</button>
      </div>
      {failures.length > 0 && (
        <details open><summary className="error-text">{failures.length} failed</summary>
          <ul className="small">{failures.map((f) => <li key={f}>{f}</li>)}</ul></details>
      )}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

// ── assignment bar ─────────────────────────────────────────────────────────

export function AssignBar({ project, selected, items, labellers, onDone }: {
  project: Project; selected: number[]; items: Item[]; labellers: User[];
  onDone: (message: string) => Promise<void>;
}) {
  const [userId, setUserId] = useState<string>('');
  const [spread, setSpread] = useState<number[]>([]);
  const { busy, error, run } = useAsync();
  const n = selected.length;

  const assign = (uid: number | null) =>
    run(async () => {
      await api.assign(project.id, uid, selected);
      const who = uid === null ? 'nobody' : labellers.find((u) => u.id === uid)?.display_name;
      await onDone(`Assigned ${n} item${n > 1 ? 's' : ''} to ${who}.`);
    });

  // Round-robin over the chosen labellers, in item order.
  const distribute = () =>
    run(async () => {
      const ordered = items.filter((i) => selected.includes(i.id)).map((i) => i.id);
      const buckets = new Map<number, number[]>(spread.map((u) => [u, []]));
      ordered.forEach((id, k) => buckets.get(spread[k % spread.length])!.push(id));
      for (const [uid, ids] of buckets) if (ids.length) await api.assign(project.id, uid, ids);
      await onDone(`Spread ${n} items over ${spread.length} labellers.`);
    });

  const remove = () =>
    run(async () => {
      for (const id of selected) await api.deleteItem(id);
      await onDone(`Deleted ${n} item${n > 1 ? 's' : ''}.`);
    });

  if (!n) return <p className="meta">Select images to assign or delete them.</p>;
  return (
    <div className="assign-bar">
      <strong>{n} selected</strong>
      <span className="row">
        <select value={userId} onChange={(e) => setUserId(e.target.value)}>
          <option value="">Assign to…</option>
          {labellers.map((u) => <option key={u.id} value={u.id}>{u.display_name}</option>)}
        </select>
        <button disabled={busy || !userId} onClick={() => assign(Number(userId))}>Assign</button>
        <button disabled={busy} onClick={() => assign(null)}>Unassign</button>
      </span>
      {labellers.length > 1 && (
        <details className="spread">
          <summary>Spread evenly…</summary>
          <div className="spread-list">
            {labellers.map((u) => (
              <label key={u.id} className="check">
                <input type="checkbox" checked={spread.includes(u.id)}
                       onChange={(e) => setSpread(e.target.checked ? [...spread, u.id]
                                                  : spread.filter((x) => x !== u.id))} />
                {u.display_name}
              </label>
            ))}
            <button disabled={busy || spread.length < 2} onClick={distribute}>
              Spread {n} over {spread.length || '…'}
            </button>
          </div>
        </details>
      )}
      <ConfirmButton label="Delete" confirm={`Delete ${n} item${n > 1 ? 's' : ''} and their annotations?`}
                     disabled={busy} onConfirm={remove} />
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

// ── team progress ──────────────────────────────────────────────────────────

export function TeamProgress({ progress, items }: { progress: Progress | undefined; items: Item[] }) {
  const unassigned = items.filter((i) => i.assignee_id === null).length;
  if (!progress) return <p className="muted">Loading…</p>;
  return (
    <div className="stack">
      <table className="items">
        <thead><tr><th>Labeller</th><th>Assigned</th><th>Done</th><th>Annotations</th><th>Progress</th></tr></thead>
        <tbody>
          {progress.by_labeller.map((r) => (
            <tr key={r.user_id} className="static">
              <td>{r.display_name}</td>
              <td>{r.items}</td>
              <td>{r.done}</td>
              <td>{r.annotations}</td>
              <td className="bar-cell"><div className="progress">
                <span className="done" style={{ width: `${r.items ? (100 * r.done) / r.items : 0}%` }} />
              </div></td>
            </tr>
          ))}
        </tbody>
      </table>
      {progress.by_labeller.length === 0 && <p className="muted">Nobody is assigned yet. Select
        images on the Images tab and assign them.</p>}
      <p className="meta">{unassigned} of {items.length} images unassigned.</p>
    </div>
  );
}

// ── users ──────────────────────────────────────────────────────────────────

export function UsersPage({ me }: { me: User }) {
  const [users, setUsers] = useState<User[] | null>(null);
  const [form, setForm] = useState({ username: '', display_name: '', password: '',
                                     role: 'labeller' as User['role'] });
  const [resetFor, setResetFor] = useState<number | null>(null);
  const [newPassword, setNewPassword] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const { busy, error, setError, run } = useAsync();

  const reload = async () => setUsers(await api.listUsers());
  useEffect(() => { reload().catch((e) => setError(e.message)); }, []);

  const create = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      const u = await api.createUser({ ...form, username: form.username.trim(),
                                       display_name: form.display_name.trim() });
      setForm({ username: '', display_name: '', password: '', role: 'labeller' });
      setNotice(`Created ${u.display_name}. Share the username and password with them.`);
      await reload();
    });
  };

  const update = (u: User, body: Parameters<typeof api.updateUser>[1], message: string) =>
    run(async () => {
      await api.updateUser(u.id, body);
      setNotice(message);
      await reload();
    });

  const sorted = useMemo(() => [...(users ?? [])].sort((a, b) =>
    Number(b.active) - Number(a.active) || a.display_name.localeCompare(b.display_name)), [users]);

  return (
    <div className="page">
      <h1>Users</h1>
      <p className="muted">Admins manage projects and assign images; labellers see only what is
        assigned to them.</p>
      {notice && <p className="notice">{notice}</p>}
      {error && <p className="error-text">{error}</p>}
      <table className="items users">
        <thead><tr><th>Name</th><th>Username</th><th>Role</th><th>Status</th><th /></tr></thead>
        <tbody>
          {sorted.map((u) => (
            <tr key={u.id} className={`static${u.active ? '' : ' inactive'}`}>
              <td>{u.display_name}{u.id === me.id && <span className="muted small"> (you)</span>}</td>
              <td className="muted">{u.username}</td>
              <td>
                <select value={u.role} disabled={busy || !u.active || u.id === me.id}
                        title={u.id === me.id ? 'Another admin can change your role' : undefined}
                        onChange={(e) => update(u, { role: e.target.value as User['role'] },
                                                `${u.display_name} is now ${e.target.value === 'admin' ? 'an admin' : 'a labeller'}.`)}>
                  <option value="labeller">labeller</option>
                  <option value="admin">admin</option>
                </select>
              </td>
              <td>{u.active ? 'active' : <span className="muted">deactivated</span>}</td>
              <td><div className="actions">
                {resetFor === u.id ? (
                  <span className="row">
                    <input type="password" placeholder="New password (8+)" value={newPassword}
                           autoComplete="new-password" onChange={(e) => setNewPassword(e.target.value)} />
                    <button disabled={busy || newPassword.length < 8}
                            onClick={() => { update(u, { password: newPassword },
                                                    `Password changed for ${u.display_name}; they were signed out.`);
                                             setResetFor(null); setNewPassword(''); }}>Set</button>
                    <button onClick={() => { setResetFor(null); setNewPassword(''); }}>Cancel</button>
                  </span>
                ) : (
                  <>
                    <button disabled={busy} onClick={() => setResetFor(u.id)}>Reset password</button>
                    {u.active
                      ? u.id !== me.id && (
                          <ConfirmButton label="Deactivate" confirm="Sign out and block?"
                                         disabled={busy}
                                         onConfirm={() => update(u, { active: false },
                                                                 `${u.display_name} was deactivated.`)} />)
                      : <button disabled={busy}
                                onClick={() => update(u, { active: true }, `${u.display_name} was reactivated.`)}>
                          Reactivate</button>}
                  </>
                )}
              </div></td>
            </tr>
          ))}
        </tbody>
      </table>

      <form className="card form narrow" onSubmit={create}>
        <h2 className="form-title">Add user</h2>
        <div className="grid2">
          <label>Display name<input value={form.display_name}
                                    onChange={(e) => setForm({ ...form, display_name: e.target.value })} /></label>
          <label>Username<input value={form.username} placeholder="3–32 letters, digits, . _ -"
                                autoComplete="off"
                                onChange={(e) => setForm({ ...form, username: e.target.value })} /></label>
          <label>Password<input type="password" value={form.password} autoComplete="new-password"
                                placeholder="at least 8 characters"
                                onChange={(e) => setForm({ ...form, password: e.target.value })} /></label>
          <label>Role<select value={form.role}
                             onChange={(e) => setForm({ ...form, role: e.target.value as User['role'] })}>
            <option value="labeller">labeller</option><option value="admin">admin</option></select></label>
        </div>
        <div className="row-end">
          <button className="primary"
                  disabled={busy || !/^[A-Za-z0-9_.-]{3,32}$/.test(form.username.trim())
                            || !form.display_name.trim() || form.password.length < 8}>Add user</button>
        </div>
      </form>
    </div>
  );
}
