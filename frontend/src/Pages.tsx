import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { AssignBar, ClassesEditor, ExportPanel, ModelsPanel, NewProject, ProjectSettings, TeamProgress,
         UploadPanel } from './Admin';
import { api, MIN_PASSWORD, setToken, TASK_LABELS, type Item, type ItemStatus, type Progress, type Project,
         type User } from './api';
import { go, href, type ProjectTab } from './lib';

// ── sign-in ────────────────────────────────────────────────────────────────

export function Login({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await api.login(username, password);
      setToken(r.token);
      onSignedIn(r.user);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card auth" onSubmit={submit}>
      <h1>Sign in</h1>
      <label>Username<input autoFocus value={username} onChange={(e) => setUsername(e.target.value)}
                            autoComplete="username" /></label>
      <label>Password<input type="password" value={password} autoComplete="current-password"
                            onChange={(e) => setPassword(e.target.value)} /></label>
      {error && <p className="error-text">{error}</p>}
      <button className="primary" disabled={busy || !username || !password}>Sign in</button>
      <p className="meta">Or use <a href={href.quick()}>Quick segment</a> without an account.</p>
    </form>
  );
}

export function Setup({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [form, setForm] = useState({ setup_token: '', username: '', display_name: '',
                                     password: '' });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof form) => (e: { target: { value: string } }) =>
    setForm({ ...form, [k]: e.target.value });

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await api.setup({ ...form, setup_token: form.setup_token.trim() });
      setToken(r.token);
      onSignedIn(r.user);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card auth" onSubmit={submit}>
      <h1>Create the first admin</h1>
      <p className="meta">The server printed a one-time setup token in its console when it
        started. Paste it here.</p>
      <label>Setup token<input autoFocus value={form.setup_token} onChange={set('setup_token')} /></label>
      <label>Username<input value={form.username} onChange={set('username')}
                            autoComplete="username" placeholder="3–32 letters, digits, . _ -" /></label>
      <label>Display name<input value={form.display_name} onChange={set('display_name')} /></label>
      <label>Password<input type="password" value={form.password} onChange={set('password')}
                            autoComplete="new-password" placeholder={`at least ${MIN_PASSWORD} characters`} /></label>
      {error && <p className="error-text">{error}</p>}
      <button className="primary"
              disabled={busy || !form.setup_token || !form.username || !form.display_name
                        || form.password.length < MIN_PASSWORD}>Create admin</button>
    </form>
  );
}

// ── projects ───────────────────────────────────────────────────────────────

function ProgressBar({ p }: { p: Progress | undefined }) {
  if (!p) return <div className="progress" />;
  const pct = (n: number) => (p.items ? (100 * n) / p.items : 0);
  return (
    <>
      <div className="progress" title={`${p.by_status.done} done, ${p.by_status.in_progress} in progress, ${p.by_status.todo} to do`}>
        <span className="done" style={{ width: `${pct(p.by_status.done)}%` }} />
        <span className="doing" style={{ width: `${pct(p.by_status.in_progress)}%` }} />
      </div>
      <p className="meta">{p.by_status.done} of {p.items} done
        {p.by_status.in_progress > 0 && ` · ${p.by_status.in_progress} in progress`}</p>
    </>
  );
}

export function Projects({ me }: { me: User }) {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [progress, setProgress] = useState<Record<number, Progress>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listProjects()
      .then((ps) => {
        setProjects(ps);
        for (const p of ps) {
          api.progress(p.id).then((pr) => setProgress((all) => ({ ...all, [p.id]: pr })))
            .catch(() => {});
        }
      })
      .catch((e) => setError(e.message));
  }, []);

  const [creating, setCreating] = useState(false);
  return (
    <div className="page">
      <div className="page-head">
        <h1>Projects</h1>
        {me.role === 'admin' && !creating && (
          <button className="primary" onClick={() => setCreating(true)}>New project</button>
        )}
      </div>
      {creating && <NewProject onCancel={() => setCreating(false)} />}
      {error && <p className="error-text">{error}</p>}
      {projects?.length === 0 && !creating && (
        <p className="muted">
          {me.role === 'admin' ? 'No projects yet. Create the first one.'
            : 'Nothing is assigned to you yet.'}
        </p>
      )}
      <div className="cards">
        {projects?.map((p) => (
          <a key={p.id} className="card project-card" href={href.project(p.id)}>
            <strong>{p.name}</strong>
            <span className="muted small">{TASK_LABELS[p.task_type]}
              {p.task_type === 'classification' && p.multi_label && ' (multi-label)'}</span>
            <ProgressBar p={progress[p.id]} />
          </a>
        ))}
      </div>
    </div>
  );
}

const STATUS_FILTERS: (ItemStatus | 'all')[] = ['all', 'todo', 'in_progress', 'done'];
const TAB_LABELS: Record<ProjectTab, string> = {
  items: 'Images', classes: 'Classes', team: 'Team', models: 'Models', export: 'Export',
  settings: 'Settings',
};

export function ProjectPage({ projectId, tab, me }: { projectId: number; tab: ProjectTab; me: User }) {
  const admin = me.role === 'admin';
  const [project, setProject] = useState<Project | null>(null);
  const [items, setItems] = useState<Item[] | null>(null);
  const [progress, setProgress] = useState<Progress>();
  const [users, setUsers] = useState<User[]>([]);
  const [filter, setFilter] = useState<ItemStatus | 'all'>('all');
  const [assignee, setAssignee] = useState<string>('any');
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reloadItems = useCallback(async () => {
    const [its, pr] = await Promise.all([api.listItems(projectId), api.progress(projectId)]);
    setItems(its);
    setProgress(pr);
    setSelected((sel) => new Set([...sel].filter((id) => its.some((i) => i.id === id))));
  }, [projectId]);

  useEffect(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
    reloadItems().catch((e) => setError(e.message));
    if (admin) api.listUsers().then(setUsers).catch(() => {});
  }, [projectId, admin, reloadItems]);

  if (error) return <div className="page"><p className="error-text">{error}</p></div>;
  if (!project || !items) return <div className="page muted">Loading…</div>;

  const userName = new Map(users.map((u) => [u.id, u.display_name]));
  const labellers = users.filter((u) => u.active);
  const shown = items.filter((i) => (filter === 'all' || i.status === filter)
    && (assignee === 'any' || (assignee === 'none' ? i.assignee_id === null
                                                     : i.assignee_id === Number(assignee))));
  const next = items.find((i) => i.status !== 'done');
  const allShownSelected = shown.length > 0 && shown.every((i) => selected.has(i.id));
  const toggle = (id: number) => {
    const next2 = new Set(selected);
    if (next2.has(id)) next2.delete(id); else next2.add(id);
    setSelected(next2);
  };
  const toggleAll = () => {
    const next2 = new Set(selected);
    for (const i of shown) { if (allShownSelected) next2.delete(i.id); else next2.add(i.id); }
    setSelected(next2);
  };
  // Project models learn segmentation, so only projects with mask or polygon labels get them.
  const trainable = project.allowed_label_types.some((t) => t === 'mask' || t === 'polygon');
  const tabs: ProjectTab[] = !admin ? ['items']
    : ['items', 'classes', 'team', ...(trainable ? ['models' as const] : []), 'export', 'settings'];

  return (
    <div className="page">
      <nav className="crumbs"><a href={href.projects()}>Projects</a> › {project.name}</nav>
      <div className="page-head">
        <div>
          <h1>{project.name}</h1>
          <p className="muted">{TASK_LABELS[project.task_type]} · {items.length} images
            {!admin && ' assigned to you'}</p>
        </div>
        <button className="primary" disabled={!next} onClick={() => next && go(href.item(next.id))}>
          {next ? 'Start labelling' : items.length ? 'All done' : 'No images yet'}
        </button>
      </div>
      <ProgressBar p={progress} />

      {tabs.length > 1 && (
        <nav className="tabs">
          {tabs.map((t) => (
            <a key={t} href={href.project(project.id, t)} className={t === tab ? 'active' : ''}>
              {TAB_LABELS[t]}
            </a>
          ))}
        </nav>
      )}

      {tab === 'classes' && admin && <ClassesEditor project={project} />}
      {tab === 'settings' && admin && <ProjectSettings project={project} onSaved={setProject} />}
      {tab === 'team' && admin && <TeamProgress progress={progress} items={items} />}
      {tab === 'models' && admin && trainable && <ModelsPanel project={project} onChanged={setProject} />}
      {tab === 'export' && admin && <ExportPanel project={project} progress={progress} />}

      {(tab === 'items' || !admin) && (
        <>
          {!admin && project.protocol && (
            <details className="card"><summary>Protocol</summary>
              <p className="protocol">{project.protocol}</p></details>
          )}
          {admin && <UploadPanel project={project} onUploaded={() => { reloadItems(); }} />}
          <div className="list-tools">
            <div className="segmented filter">
              {STATUS_FILTERS.map((st) => (
                <button key={st} className={filter === st ? 'active' : ''} onClick={() => setFilter(st)}>
                  {st === 'all' ? `All (${items.length})`
                    : `${st.replace('_', ' ')} (${items.filter((i) => i.status === st).length})`}
                </button>
              ))}
            </div>
            {admin && (
              <select value={assignee} onChange={(e) => setAssignee(e.target.value)}>
                <option value="any">Anyone</option>
                <option value="none">Unassigned</option>
                {users.map((u) => <option key={u.id} value={u.id}>{u.display_name}</option>)}
              </select>
            )}
          </div>
          {admin && (
            <AssignBar project={project} selected={[...selected]} items={items} labellers={labellers}
                       onDone={async (msg) => { setSelected(new Set()); await reloadItems(); setNotice(msg); }} />
          )}
          {notice && <p className="notice">{notice}</p>}
          <table className="items">
            <thead>
              <tr>
                {admin && <th className="check-cell"><input type="checkbox" checked={allShownSelected}
                                                           onChange={toggleAll} title="Select all shown" /></th>}
                <th>Name</th><th>Size</th><th>Labels</th><th>Status</th>
                {admin && <th>Assigned to</th>}
              </tr>
            </thead>
            <tbody>
              {shown.map((i) => (
                <tr key={i.id} className={selected.has(i.id) ? 'selected' : ''}
                    onClick={() => (admin ? toggle(i.id) : go(href.item(i.id)))}>
                  {admin && <td className="check-cell"><input type="checkbox" checked={selected.has(i.id)}
                                                             onChange={() => toggle(i.id)}
                                                             onClick={(e) => e.stopPropagation()} /></td>}
                  <td><a href={href.item(i.id)} onClick={(e) => e.stopPropagation()}>{i.name}</a>
                    {i.kind === 'pair' && <span className="muted small"> · pair</span>}</td>
                  <td className="muted">{i.width}×{i.height}</td>
                  <td className="muted">{i.annotation_count ?? '—'}</td>
                  <td><span className={`badge status-${i.status}`}>{i.status.replace('_', ' ')}</span></td>
                  {admin && (
                    <td className="muted">{i.assignee_id === null ? '—'
                      : userName.get(i.assignee_id) ?? `user ${i.assignee_id}`}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          {shown.length === 0 && (
            <p className="muted">{items.length === 0
              ? (admin ? 'No images yet. Add some above.' : 'Nothing is assigned to you here.')
              : 'No images match this filter.'}</p>
          )}
        </>
      )}
    </div>
  );
}
