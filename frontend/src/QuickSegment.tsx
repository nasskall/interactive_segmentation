import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type LoadModelBody, type ModelInfo, type SessionState } from './api';
import ModelChooser, { describeModel, describeWeights } from './ModelChooser';
import SegCanvas from './SegCanvas';

export default function QuickSegment() {
  const [models, setModels] = useState<ModelInfo | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Load an image to start.');
  const [error, setError] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0.5);
  const [clickRadius, setClickRadius] = useState(5);
  const [thresh, setThresh] = useState(0.5);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch((e) => setError(`Backend unreachable: ${e.message}`));
  }, []);

  // Runs one backend call at a time; the canvas ignores clicks while busy.
  const run = useCallback(async (label: string, fn: () => Promise<SessionState | void>) => {
    setBusy(true);
    setError(null);
    setStatus(`${label}…`);
    const t0 = performance.now();
    try {
      const next = await fn();
      if (next) setSession(next);
      setStatus(`${label} done in ${Math.round(performance.now() - t0)} ms`);
    } catch (e) {
      setError((e as Error).message);
      setStatus(`${label} failed`);
    } finally {
      setBusy(false);
    }
  }, []);

  const sid = session?.session_id;

  useEffect(() => { if (session) setThresh(session.prob_thresh); }, [session]);

  // Sent once the slider is released rather than on every drag step.
  const commitThresh = () => {
    if (sid && session && thresh !== session.prob_thresh)
      run('Threshold', () => api.updateSettings(sid, { prob_thresh: thresh }));
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    const old = sid;
    run(`Uploading ${file.name}`, async () => {
      const s = await api.createSession(file);
      if (old) api.deleteSession(old).catch(() => {});
      return s;
    });
  };

  const onLoadModel = (body: LoadModelBody) =>
    run(`Loading ${body.model_type}`, async () => {
      const m = await api.loadModel(body);
      setModels(m);
      // Loading a model drops in-progress clicks on the server; resync.
      if (sid) return api.getSession(sid);
    });

  const undo = useCallback(() => sid && run('Undo', () => api.undo(sid)), [sid, run]);
  const finish = useCallback(() => sid && run('Finish object', () => api.finish(sid)), [sid, run]);
  const reset = useCallback(() => sid && run('Reset object', () => api.reset(sid)), [sid, run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (busy || !sid || (e.target as HTMLElement).tagName === 'INPUT' ||
          (e.target as HTMLElement).tagName === 'SELECT') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
      else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); finish(); }
      else if (e.key === 'Escape') reset();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, sid, undo, finish, reset]);

  // Zoom-in is a RITM / SimpleClick predictor option.
  const zoomApplies = ['RITM', 'SimpleClick'].includes(models?.current.model_type ?? '');
  const weights = describeWeights(session?.ensemble_weights);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Quick segment</h1>
        <p className="meta">One image at a time, nothing is saved. Download the mask when done.</p>

        <section>
          <h2>Image</h2>
          <input ref={fileInput} type="file" accept="image/*" hidden
                 onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ''; }} />
          <button onClick={() => fileInput.current?.click()} disabled={busy}>Load image…</button>
          {session && (
            <p className="meta">
              {session.filename} · {session.original_width}×{session.original_height}
              {session.width !== session.original_width && ` (shown at ${session.width}×${session.height})`}
            </p>
          )}
        </section>

        <section>
          <h2>Model</h2>
          {models && (
            <>
              <ModelChooser models={models} busy={busy} buttonLabel="Load model"
                            onLoad={onLoadModel} />
              <p className="meta">Active: {describeModel(models)}</p>
            </>
          )}
        </section>

        <section>
          <h2>Segmentation</h2>
          <div className="row">
            <button onClick={undo} disabled={busy || !session?.can_undo}>Undo</button>
            <button onClick={reset} disabled={busy || !session?.has_incomplete_object}>Reset</button>
          </div>
          <button className="primary" onClick={finish}
                  disabled={busy || !session?.has_incomplete_object}>Finish object</button>
          <button onClick={() => sid && run('Auto-segment', () => api.autoSegment(sid))}
                  disabled={busy || !sid}>Auto-segment</button>
          <a className={`button${!sid || busy ? ' disabled' : ''}`}
             href={sid ? api.maskUrl(sid) : undefined} download>Download mask</a>
          {session && (
            <p className="meta">
              {session.object_count} finished · {session.clicks.length} click(s) on current object
            </p>
          )}
          {weights && <p className="meta" title="Weight of each model for the latest click">
            Model weights: {weights}</p>}
        </section>

        <section>
          <h2>Settings</h2>
          <label>
            Probability threshold: {thresh.toFixed(2)}
            <input type="range" min={0.05} max={0.95} step={0.05} disabled={!sid}
                   value={thresh}
                   onChange={(e) => setThresh(Number(e.target.value))}
                   onPointerUp={commitThresh} onKeyUp={commitThresh} />
          </label>
          <label className="check">
            <input type="checkbox" disabled={busy || !sid || !zoomApplies}
                   checked={session?.use_zoom_in ?? true}
                   onChange={(e) => sid && run('Zoom-in',
                     () => api.updateSettings(sid, { use_zoom_in: e.target.checked }))} />
            Zoom-in (RITM / SimpleClick)
          </label>
          <label>
            Overlay opacity: {alpha.toFixed(2)}
            <input type="range" min={0} max={1} step={0.05} value={alpha}
                   onChange={(e) => setAlpha(Number(e.target.value))} />
          </label>
          <label>
            Click radius: {clickRadius}
            <input type="range" min={2} max={12} step={1} value={clickRadius}
                   onChange={(e) => setClickRadius(Number(e.target.value))} />
          </label>
        </section>

        <section className="help">
          Left click: add to object · Right click: remove<br />
          Ctrl+Z undo · Space finish · Esc reset
        </section>
      </aside>

      <main className="stage">
        {session ? (
          <SegCanvas
            imageUrl={api.imageUrl(session.session_id)}
            overlayUrl={session.overlay}
            width={session.width}
            height={session.height}
            clicks={session.clicks}
            alpha={alpha}
            clickRadius={clickRadius}
            busy={busy}
            onClick={(x, y, positive) =>
              run(positive ? 'Positive click' : 'Negative click',
                () => api.addClick(session.session_id, { x, y, positive }))}
          />
        ) : (
          <div className="placeholder" onClick={() => fileInput.current?.click()}
               onDragOver={(e) => e.preventDefault()}
               onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0]); }}>
            Drop an image here or click to choose one
          </div>
        )}
        <footer className={error ? 'status error' : 'status'}>{error ?? status}</footer>
      </main>
    </div>
  );
}
