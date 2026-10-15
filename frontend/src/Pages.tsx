import { useEffect, useState, type FormEvent } from 'react';
import { api, setToken, TASK_LABELS, type Item, type ItemStatus, type Progress, type Project,
         type User } from './api';
import { go, href } from './lib';

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
                            autoComplete="new-password" placeholder="at least 8 characters" /></label>
      {error && <p className="error-text">{error}</p>}
      <button className="primary"
              disabled={busy || !form.setup_token || !form.username || !form.display_name
                        || form.password.length < 8}>Create admin</button>
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

  return (
    <div className="page">
      <h1>Projects</h1>
      {error && <p className="error-text">{error}</p>}
      {projects?.length === 0 && (
        <p className="muted">
          {me.role === 'admin'
            ? 'No projects yet. Project and user management screens come in the next phase; '
              + 'for now create projects, classes and items from the API docs at /docs.'
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

export function ProjectItems({ projectId, me }: { projectId: number; me: User }) {
  const [project, setProject] = useState<Project | null>(null);
  const [items, setItems] = useState<Item[] | null>(null);
  const [progress, setProgress] = useState<Progress>();
  const [users, setUsers] = useState<Map<number, string>>(new Map());
  const [filter, setFilter] = useState<ItemStatus | 'all'>('all');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getProject(projectId), api.listItems(projectId), api.progress(projectId)])
      .then(([p, its, pr]) => { setProject(p); setItems(its); setProgress(pr); })
      .catch((e) => setError(e.message));
    if (me.role === 'admin') {
      api.listUsers().then((us) => setUsers(new Map(us.map((u) => [u.id, u.display_name]))))
        .catch(() => {});
    }
  }, [projectId, me.role]);

  if (error) return <div className="page"><p className="error-text">{error}</p></div>;
  if (!project || !items) return <div className="page muted">Loading…</div>;

  const shown = filter === 'all' ? items : items.filter((i) => i.status === filter);
  const next = items.find((i) => i.status !== 'done');

  return (
    <div className="page">
      <nav className="crumbs"><a href={href.projects()}>Projects</a> › {project.name}</nav>
      <div className="page-head">
        <div>
          <h1>{project.name}</h1>
          <p className="muted">{TASK_LABELS[project.task_type]} · {items.length} items
            {me.role !== 'admin' && ' assigned to you'}</p>
        </div>
        <button className="primary" disabled={!next} onClick={() => next && go(href.item(next.id))}>
          {next ? 'Start labelling' : 'All done'}
        </button>
      </div>
      <ProgressBar p={progress} />
      {project.protocol && (
        <details className="card"><summary>Protocol</summary>
          <p className="protocol">{project.protocol}</p></details>
      )}
      <div className="segmented filter">
        {STATUS_FILTERS.map((s) => (
          <button key={s} className={filter === s ? 'active' : ''} onClick={() => setFilter(s)}>
            {s === 'all' ? `All (${items.length})` : `${s.replace('_', ' ')} (${items.filter((i) => i.status === s).length})`}
          </button>
        ))}
      </div>
      <table className="items">
        <thead>
          <tr><th>#</th><th>Name</th><th>Size</th><th>Status</th>
            {me.role === 'admin' && <th>Assigned to</th>}</tr>
        </thead>
        <tbody>
          {shown.map((i) => (
            <tr key={i.id} onClick={() => go(href.item(i.id))}>
              <td className="muted">{i.id}</td>
              <td><a href={href.item(i.id)}>{i.name}</a>{i.kind === 'pair' && <span className="muted small"> · pair</span>}</td>
              <td className="muted">{i.width}×{i.height}</td>
              <td><span className={`badge status-${i.status}`}>{i.status.replace('_', ' ')}</span></td>
              {me.role === 'admin' && (
                <td className="muted">{i.assignee_id === null ? '—'
                  : users.get(i.assignee_id) ?? `user ${i.assignee_id}`}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {shown.length === 0 && <p className="muted">No items here.</p>}
    </div>
  );
}
