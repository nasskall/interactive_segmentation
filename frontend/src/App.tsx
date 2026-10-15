import { useEffect, useState } from 'react';
import { api, hasToken, setToken, setUnauthorizedHandler, type User } from './api';
import { href, useRoute } from './lib';
import { UsersPage } from './Admin';
import { Login, ProjectPage, Projects, Setup } from './Pages';
import QuickSegment from './QuickSegment';
import Workspace from './Workspace';

type Auth = { state: 'loading' } | { state: 'setup' } | { state: 'signed-out' }
  | { state: 'signed-in'; user: User };

export default function App() {
  const route = useRoute();
  const [auth, setAuth] = useState<Auth>({ state: 'loading' });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setToken(null);
      setAuth({ state: 'signed-out' });
    });
    (async () => {
      const { needs_setup } = await api.authStatus();
      if (needs_setup) return setAuth({ state: 'setup' });
      if (!hasToken()) return setAuth({ state: 'signed-out' });
      try {
        setAuth({ state: 'signed-in', user: await api.me() });
      } catch {
        setAuth({ state: 'signed-out' });
      }
    })().catch((e) => setError(`Backend unreachable: ${e.message}`));
  }, []);

  const signedIn = (user: User) => setAuth({ state: 'signed-in', user });
  const signOut = async () => {
    await api.logout().catch(() => {});
    setToken(null);
    setAuth({ state: 'signed-out' });
  };

  const user = auth.state === 'signed-in' ? auth.user : null;
  let body;
  if (route.page === 'quick') body = <QuickSegment />;
  else if (error) body = <div className="page"><p className="error-text">{error}</p></div>;
  else if (auth.state === 'loading') body = <div className="page muted">Loading…</div>;
  else if (auth.state === 'setup') body = <div className="page"><Setup onSignedIn={signedIn} /></div>;
  else if (!user) body = <div className="page"><Login onSignedIn={signedIn} /></div>;
  else if (route.page === 'project') {
    body = <ProjectPage key={route.id} projectId={route.id} tab={route.tab} me={user} />;
  } else if (route.page === 'users' && user.role === 'admin') body = <UsersPage me={user} />;
  else if (route.page === 'item') body = <Workspace key={route.id} itemId={route.id} me={user} />;
  else body = <Projects me={user} />;

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href={href.projects()}>Interactive Labelling</a>
        <nav>
          <a className={!['quick', 'users'].includes(route.page) ? 'active' : ''}
             href={href.projects()}>Projects</a>
          {user?.role === 'admin' && (
            <a className={route.page === 'users' ? 'active' : ''} href={href.users()}>Users</a>
          )}
          <a className={route.page === 'quick' ? 'active' : ''} href={href.quick()}>Quick segment</a>
        </nav>
        <span className="grow" />
        {user && (
          <span className="user">
            {user.display_name} <span className="muted small">({user.role})</span>
            <button onClick={signOut}>Sign out</button>
          </span>
        )}
      </header>
      <div className="content">{body}</div>
    </div>
  );
}
