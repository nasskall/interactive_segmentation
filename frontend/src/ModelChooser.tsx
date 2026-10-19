// Model selection shared by Quick segment and the labelling workspace:
// one model + checkpoint, or an ensemble of several (see
// isegm/inference/predictors/ensemble.py).

import { useEffect, useState } from 'react';
import { ENSEMBLE, type Fusion, type LoadModelBody, type ModelInfo, type ModelRef } from './api';

const FUSION_LABELS: Record<Fusion, string> = {
  mean: 'Average',
  weighted: 'Weighted by agreement',
  weighted_conf: 'Weighted + per-pixel confidence',
  staple: 'STAPLE consensus (recommended)',
};

const key = (m: ModelRef) => `${m.model_type}|${m.checkpoint}`;

/** Model type, plus the checkpoint when another entry has the same type
 *  (e.g. two RITM checkpoints). */
function memberLabel(m: ModelRef, all: ModelRef[]): string {
  const shared = all.filter((o) => o.model_type === m.model_type).length > 1;
  return shared ? `${m.model_type} (${m.checkpoint.replace(/\.(pth|pt)$/, '')})` : m.model_type;
}

/** One-line description of the loaded model. */
export function describeModel(models: ModelInfo): string {
  const c = models.current;
  if (c.model_type !== ENSEMBLE) return `${c.model_type} · ${c.checkpoint ?? ''} · ${models.device}`;
  const members = c.members ?? [];
  const names = members.map((m) => memberLabel(m, members)).join(' + ');
  return `Ensemble (${names}) · ${FUSION_LABELS[c.fusion ?? 'staple']}`
    + `${c.tta ? ' · TTA' : ''} · ${models.device}`;
}

/** "RITM 0.21 · SAM2 - Tiny 0.79" for the latest click of an ensemble. */
export function describeWeights(weights: Record<string, number> | null | undefined): string | null {
  if (!weights) return null;
  return Object.entries(weights).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ');
}

export default function ModelChooser({ models, busy, buttonLabel, onLoad }: {
  models: ModelInfo; busy: boolean; buttonLabel: string; onLoad: (body: LoadModelBody) => void;
}) {
  const cur = models.current;
  const [type, setType] = useState(cur.model_type);
  const [ckpt, setCkpt] = useState(cur.checkpoint ?? models.checkpoints[0] ?? '');
  // Until an ensemble has been loaded, offer every available model.
  const [members, setMembers] = useState<Set<string>>(() => new Set(
    (cur.members ?? models.ensemble.candidates.slice(0, models.ensemble.max_members)).map(key)));
  const [fusion, setFusion] = useState<Fusion>('staple');
  const [tta, setTta] = useState(false);

  // Follow whatever the server reports as loaded.
  useEffect(() => {
    setType(cur.model_type);
    if (cur.checkpoint) setCkpt(cur.checkpoint);
    if (cur.members) setMembers(new Set(cur.members.map(key)));
    if (cur.fusion) setFusion(cur.fusion);
    if (cur.tta !== undefined) setTta(cur.tta);
  }, [cur.model_type, cur.checkpoint, cur.members, cur.fusion, cur.tta]);

  const { candidates, max_members: max, fusions } = models.ensemble;
  const isEnsemble = type === ENSEMBLE;
  const chosen = candidates.filter((c) => members.has(key(c)));
  const valid = isEnsemble ? chosen.length >= 2 && chosen.length <= max : !!ckpt;

  const toggle = (m: ModelRef) => {
    const next = new Set(members);
    if (next.has(key(m))) next.delete(key(m)); else next.add(key(m));
    setMembers(next);
  };
  const load = () => onLoad(isEnsemble
    ? { model_type: ENSEMBLE, members: chosen, fusion, tta }
    : { model_type: type, checkpoint: ckpt });

  return (
    <>
      <label>Type
        <select value={type} onChange={(e) => setType(e.target.value)} disabled={busy}>
          {models.model_types.map((t) => <option key={t}>{t}</option>)}
        </select>
      </label>
      {isEnsemble ? (
        <>
          <fieldset className="members">
            <legend>Models to combine (pick 2–{max})</legend>
            {candidates.map((c) => (
              <label key={key(c)} className="check" title={c.checkpoint}>
                <input type="checkbox" checked={members.has(key(c))} onChange={() => toggle(c)}
                       disabled={busy || (!members.has(key(c)) && chosen.length >= max)} />
                {memberLabel(c, candidates)}
              </label>
            ))}
            {candidates.length < 2 && (
              <p className="meta">Needs at least two checkpoints in model_weights/.</p>
            )}
          </fieldset>
          <label>Combine
            <select value={fusion} onChange={(e) => setFusion(e.target.value as Fusion)}
                    disabled={busy}>
              {fusions.map((f) => <option key={f} value={f}>{FUSION_LABELS[f]}</option>)}
            </select>
          </label>
          {fusion === 'staple' && chosen.length === 2 && (
            <p className="meta">STAPLE needs three or more models; with two it averages them.</p>
          )}
          <label className="check">
            <input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)}
                   disabled={busy} />
            Test-time augmentation
          </label>
          <p className="meta">Every click runs each model, so expect it to be slower than one
            model alone{tta ? ', more so with augmentation' : ''}.</p>
        </>
      ) : (
        <label>Checkpoint
          <select value={ckpt} onChange={(e) => setCkpt(e.target.value)} disabled={busy}>
            {models.checkpoints.map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>
      )}
      <button onClick={load} disabled={busy || !valid}>{buttonLabel}</button>
    </>
  );
}
