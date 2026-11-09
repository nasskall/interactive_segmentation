// Second sign-in step: setting up the authenticator app (QR code), entering
// its 6-digit code or a recovery code, the recovery codes shown once, and the
// Account page where they can be renewed.

import { useEffect, useState, type FormEvent } from 'react';
import { api, setToken, type Enrolment, type LoginStep, type SignedIn, type User } from './api';

const CODE_HELP = 'The 6-digit code your authenticator app shows for Interactive Labelling.';

/** Recovery codes, shown once, with a text-file download. */
export function RecoveryCodes({ codes, account }: { codes: string[]; account: string }) {
  const download = () => {
    const text = `Interactive Labelling recovery codes for ${account}\n`
      + `Each code works once, instead of the authenticator app.\n\n${codes.join('\n')}\n`;
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    const a = Object.assign(document.createElement('a'),
                            { href: url, download: `recovery-codes-${account}.txt` });
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="stack">
      <ul className="recovery-codes">{codes.map((c) => <li key={c}><code>{c}</code></li>)}</ul>
      <button type="button" onClick={download}>Download as a text file</button>
    </div>
  );
}

function CodeField({ value, onChange, recovery }: {
  value: string; onChange: (v: string) => void; recovery: boolean;
}) {
  return recovery
    ? <label>Recovery code
        <input autoFocus value={value} onChange={(e) => onChange(e.target.value)}
               autoComplete="off" spellCheck={false} placeholder="xxxx-xxxx" /></label>
    : <label>Code
        <input autoFocus value={value} inputMode="numeric" autoComplete="one-time-code"
               maxLength={7} placeholder="123456"
               onChange={(e) => onChange(e.target.value.replace(/[^0-9 ]/g, ''))} /></label>;
}

/** Everything between the password and a signed-in session. */
export function SecondFactor({ step, onSignedIn, onRestart }: {
  step: LoginStep; onSignedIn: (u: User) => void; onRestart: () => void;
}) {
  const [enrolment, setEnrolment] = useState<Enrolment | null>(null);
  const [code, setCode] = useState('');
  const [recovery, setRecovery] = useState(false);
  const [done, setDone] = useState<SignedIn | null>(null);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const enrolling = step.second_factor === 'enrol';

  useEffect(() => {
    if (!enrolling) return;
    api.enrol(step.challenge).then(setEnrolment).catch((e) => setError(e.message));
  }, [enrolling, step.challenge]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = enrolling ? await api.activate(step.challenge, code)
                          : await api.verifyCode(step.challenge, code);
      setToken(r.token);
      // New recovery codes, or a recovery code just used: show a page first.
      if (r.recovery_codes || recovery) setDone(r);
      else onSignedIn(r.user);
    } catch (err) {
      setError((err as Error).message);
      setCode('');
    } finally {
      setBusy(false);
    }
  };

  const expired = !!error && /expired/i.test(error);
  const restart = <button type="button" className="link" onClick={onRestart}>Start again</button>;

  if (done?.recovery_codes) {
    return (
      <div className="card auth">
        <h1>Save your recovery codes</h1>
        <p>If you lose your phone, sign in with one of these instead of a code from the app.
          Each one works once. They are shown only now.</p>
        <RecoveryCodes codes={done.recovery_codes} account={step.user.username} />
        <label className="check">
          <input type="checkbox" checked={saved} onChange={(e) => setSaved(e.target.checked)} />
          I have saved these codes somewhere safe</label>
        <button className="primary" disabled={!saved} onClick={() => onSignedIn(done.user)}>
          Continue</button>
      </div>
    );
  }
  if (done) {
    const left = done.recovery_codes_left ?? 0;
    return (
      <div className="card auth">
        <h1>Signed in with a recovery code</h1>
        <p>That code cannot be used again. {left} recovery code{left === 1 ? '' : 's'} left.</p>
        {left <= 3 && <p className="error-text">Make new ones on your Account page while you
          still have the app or a code.</p>}
        <button className="primary" onClick={() => onSignedIn(done.user)}>Continue</button>
      </div>
    );
  }

  return (
    <form className="card auth" onSubmit={submit}>
      {enrolling ? <>
        <h1>Set up two-step sign-in</h1>
        <p className="meta">Every account signs in with a password and a code from an
          authenticator app (Google or Microsoft Authenticator, Aegis, 2FAS, …).</p>
        <ol className="steps">
          <li className="qr">In the app, add an account and scan this code.
            {enrolment ? <>
              <img src={enrolment.qr} alt="QR code for the authenticator app" width={220} height={220} />
              <details><summary>Can't scan it?</summary>
                <p className="meta">Add it by hand as a time-based key:</p>
                <p><code className="secret">{enrolment.secret}</code></p>
                <p className="meta">Account {enrolment.account}, issuer {enrolment.issuer}.</p>
              </details>
            </> : !error && <p className="muted">Preparing…</p>}
          </li>
          <li>Type the 6-digit code it shows.</li>
        </ol>
      </> : <>
        <h1>Two-step sign-in</h1>
        <p className="meta">{recovery ? 'One of the recovery codes you saved when you set up '
                                        + 'the app.' : CODE_HELP}</p>
      </>}
      <CodeField value={code} onChange={setCode} recovery={recovery} />
      {error && <p className="error-text">{error}</p>}
      {expired
        ? <p className="meta">{restart}</p>
        : <button className="primary" disabled={busy || (enrolling && !enrolment)
                                                || code.replace(/\s/g, '').length < (recovery ? 8 : 6)}>
            {enrolling ? 'Turn on and sign in' : 'Sign in'}</button>}
      {!enrolling && (
        <button type="button" className="link" onClick={() => { setRecovery(!recovery); setCode(''); }}>
          {recovery ? 'Use the authenticator app instead' : 'Lost your phone? Use a recovery code'}
        </button>
      )}
      {!expired && <p className="meta small">Signing in as {step.user.display_name}. {restart}</p>}
    </form>
  );
}

/** The signed-in user's own account: two-step sign-in and recovery codes. */
export function AccountPage({ me }: { me: User }) {
  const [user, setUser] = useState<User>(me);
  const [code, setCode] = useState('');
  const [codes, setCodes] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.me().then(setUser).catch(() => {}); }, [codes]);

  const renew = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setCodes((await api.renewRecoveryCodes(code)).recovery_codes);
      setCode('');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const left = user.recovery_codes_left;
  return (
    <div className="page">
      <h1>Account</h1>
      <p className="meta">{user.display_name} · {user.username} · {user.role}</p>
      <section className="card stack">
        <h2>Two-step sign-in</h2>
        <p>On. Signing in needs your password and a code from your authenticator app.</p>
        {left !== undefined && (
          <p className={left <= 3 ? 'error-text' : ''}>
            {left} of 10 recovery codes left.</p>
        )}
        {codes ? <>
          <p><strong>Your new recovery codes.</strong> The old ones no longer work. Save these
            now; they are shown only once.</p>
          <RecoveryCodes codes={codes} account={user.username} />
          <button onClick={() => setCodes(null)}>Done</button>
        </> : (
          <form className="stack" onSubmit={renew}>
            <p className="meta">New recovery codes replace all the old ones. Confirm with a code
              from your authenticator app.</p>
            <CodeField value={code} onChange={setCode} recovery={false} />
            {error && <p className="error-text">{error}</p>}
            <button className="primary" disabled={busy || code.replace(/\s/g, '').length < 6}>
              Make new recovery codes</button>
          </form>
        )}
        <p className="meta">Lost both the app and your codes? An administrator can reset your
          two-step sign-in on the server, and you set it up again when you next sign in.</p>
      </section>
    </div>
  );
}
