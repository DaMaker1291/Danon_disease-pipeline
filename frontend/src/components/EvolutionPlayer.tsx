/**
 * EvolutionPlayer — the 365-day cancer evolution and treatment-resistance
 * movie, rendered from the validated frame series in the `evolution`
 * artifact (danon.evolution_export, paired steering-vs-MTD on BraTS
 * case 01017).
 *
 * The timeline indexes frame_days (the engine's recorded days — dense in
 * the first two weeks, coarsening to 7-day steps late). Each frame is a
 * complete VTI (drug_A, drug_B, sensitive, resistant) re-attached to the
 * static geometry maps (tissue_class / diffusion / proliferative /
 * vessels, carried once per payload) and rendered through the same
 * VtiViewer canvas the Twin Planner uses — legend, scale bar, cutaway,
 * screenshot all work here too.
 *
 * Synced to the timeline: the resistant-fraction trajectories of BOTH
 * arms (the divergence between them IS the resistance story) with a
 * cursor at the current day, the current-frame headline (burden, R
 * fraction, dose state), and the PFS markers of both arms.
 *
 * Playback: play/pause with a speed control; the day cursor steps the
 * instanced voxel field client-side (no engine calls, no re-parse).
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import VtiViewer, { type VtiData } from './VtiViewer';
import { useArtifact } from '../utils/artifacts';
import { artifactStamp } from '../utils/artifacts';
import type { EvolutionArtifactData, EvolutionArmData } from '../types/artifacts';

/* ------------------------------------------------------------------ decode */

function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Parse one frame VTI's four dynamic fields (subset of VtiViewer's parser:
 * frames from the artifact never carry the static maps, so the strict
 * sensitive/resistant check is all that is needed). */
function parseFrame(b64: string): { fields: Record<string, Float32Array>; mask: Uint8Array } {
  const bytes = decodeBase64(b64);
  let text = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    text += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  const doc = new DOMParser().parseFromString(text, 'text/xml');
  const img = doc.querySelector('ImageData');
  if (!img) throw new Error('frame is not a VTK ImageData file');
  const fields: Record<string, Float32Array> = {};
  let mask = new Uint8Array(0);
  doc.querySelectorAll('DataArray').forEach((el) => {
    const name = el.getAttribute('Name') || '';
    const type = el.getAttribute('type');
    const raw = decodeBase64((el.textContent || '').trim());
    if (type === 'Float32') fields[name] = new Float32Array(raw.buffer.slice(0));
    else if (type === 'UInt8' && name === 'tumour_mask') mask = new Uint8Array(raw);
  });
  if (!fields.sensitive || !fields.resistant || !mask.length) {
    throw new Error('frame missing sensitive/resistant/mask');
  }
  return { fields, mask };
}

/** Decode the once-per-payload static geometry maps (Fortran <f4, x-fastest —
 * the exporter's layout). */
function decodeStaticMaps(maps: Record<string, string>, nPoints: number): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const [name, b64] of Object.entries(maps)) {
    out[name] = new Float32Array(decodeBase64(b64).buffer.slice(0, nPoints * 4));
  }
  return out;
}

function frameToVtiData(
  frame: { fields: Record<string, Float32Array>; mask: Uint8Array },
  statics: Record<string, Float32Array> | null,
  dims: [number, number, number],
  spacingMm: number,
): VtiData {
  const fields: Record<string, Float32Array> = { ...frame.fields };
  if (statics) {
    for (const [k, v] of Object.entries(statics)) {
      if (!fields[k]) fields[k] = v;
    }
  }
  return { dims, fields, mask: frame.mask, spacingMm };
}

/** Linear blend of two recorded frames (same grid, same fields).
 *
 * The artifact samples the engine at ~36 recorded days across the year;
 * playback interpolates the fields between two of them so the volume moves
 * continuously instead of stepping day-to-day. Scrubbing sets blend = 0, so
 * any frame you land on is an exact engine state — only the motion between
 * recorded days is interpolated, never a number that gets reported. */
function blendFrames(
  a: { fields: Record<string, Float32Array>; mask: Uint8Array },
  b: { fields: Record<string, Float32Array>; mask: Uint8Array },
  alpha: number,
): { fields: Record<string, Float32Array>; mask: Uint8Array } {
  const fields: Record<string, Float32Array> = {};
  for (const [k, fa] of Object.entries(a.fields)) {
    const fb = b.fields[k];
    if (!fb || fb.length !== fa.length) { fields[k] = fa; continue; }
    const out = new Float32Array(fa.length);
    for (let i = 0; i < fa.length; i++) out[i] = fa[i] + alpha * (fb[i] - fa[i]);
    fields[k] = out;
  }
  return { fields, mask: a.mask };
}

/* ------------------------------------------------------------------ format */

function num(x: number | null | undefined, d = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return x.toFixed(d).replace(/\B(?=(\d{3})+(?!$))/g, ',');
}
function pct(x: number | null | undefined, d = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(d)}%`;
}
function fmt(x: number | null | undefined, d = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1e5 || (a > 0 && a < 1e-3)) return x.toExponential(2);
  return x.toFixed(d);
}

/* ------------------------------------------------------------------- panel */

function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="panel" style={{ marginBottom: 14 }}>
      <div className="panel-title">{title}</div>
      {sub && <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '4px 0 12px', lineHeight: 1.5 }}>{sub}</div>}
      {children}
    </div>
  );
}

/* ---------------------------------------------------------------- component */

const SPEEDS = [0.5, 1, 2, 4] as const;

export default function EvolutionPlayer() {
  const { artifact, loading, error } = useArtifact<EvolutionArtifactData>('evolution');
  const [frameIdx, setFrameIdx] = useState(0);
  const [arm, setArm] = useState<'policy' | 'mtd'>('policy');
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [showMtd, setShowMtd] = useState(true);
  const [frameError, setFrameError] = useState<string | null>(null);
  /* Sub-frame playback offset in [0,1) between frameIdx and frameIdx+1.
   * Mirrored in a ref so the playback tick reads the live value instead of
   * a stale closure, and so the tick can advance the index itself. */
  const [blend, setBlend] = useState(0);
  const blendRef = useRef(0);
  const idxRef = useRef(0);
  const setBlendBoth = (v: number) => { blendRef.current = v; setBlend(v); };
  const timerRef = useRef<number | null>(null);

  const data: EvolutionArtifactData | null = artifact?.data ?? null;
  const armData: EvolutionArmData | null = data ? data.arms[arm] : null;

  /* Decode static maps once (geometry constants carried once per payload). */
  const statics = useMemo(() => {
    if (!data?.static_maps) return null;
    const n = data.grid[0] * data.grid[1] * data.grid[2];
    try { return decodeStaticMaps(data.static_maps, n); }
    catch { return null; }
  }, [data]);

  /* Decode only the frames the timeline can currently show: lazily per arm,
   * cached across frame changes. */
  const frameCache = useRef<Map<string, ReturnType<typeof parseFrame>>>(new Map());
  const frameAt = (idx: number, from: EvolutionArmData) => {
    const key = `${arm}:${idx}`;
    const hit = frameCache.current.get(key);
    if (hit) return hit;
    const f = parseFrame(from.frames_vti_b64[idx]);
    frameCache.current.set(key, f);
    return f;
  };
  const currentFrame = useMemo(() => {
    if (!data || !armData) return null;
    try {
      const cur = frameAt(frameIdx, armData);
      const n = armData.frames_vti_b64.length;
      /* Only touch state when clearing a real error: this memo also runs on
       * every blend tick, and a setState here would re-render each tick. */
      if (frameError !== null) setFrameError(null);
      if (blend <= 0 || frameIdx + 1 >= n) return cur;
      return blendFrames(cur, frameAt(frameIdx + 1, armData), blend);
    } catch (e: any) {
      setFrameError(e?.message || 'frame decode failed');
      return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, armData, arm, frameIdx, blend, frameError]);

  /* Free the cache when the arm changes — the other arm re-decodes on demand. */
  useEffect(() => { frameCache.current.clear(); setBlendBoth(0); }, [arm]);
  useEffect(() => { idxRef.current = frameIdx; }, [frameIdx]);

  /* Playback: step through frame_days with the chosen speed. */
  useEffect(() => {
    if (!playing || !armData) return;
    const n = armData.frame_days.length;
    timerRef.current = window.setInterval(() => {
      const next = blendRef.current + 0.24 * speed;
      if (next < 1) { setBlendBoth(next); return; }
      const i = idxRef.current;
      if (i + 1 >= n) { setPlaying(false); setBlendBoth(0); return; }
      idxRef.current = i + 1;
      setFrameIdx(i + 1);
      setBlendBoth(next - 1);
    }, 95);
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      /* Pause (or a speed change) snaps back to the recorded frame, so any
       * view you are stopped on is an exact engine state -- the volume then
       * agrees with the readouts, which are always indexed by recorded day. */
      setBlendBoth(0);
    };
  }, [playing, speed, armData]);

  /* Reset to day 0 when a new artifact loads. */
  useEffect(() => { setFrameIdx(0); setPlaying(false); setBlendBoth(0); }, [artifact]);

  /* Synced chart data: resistant fraction per day, both arms.
   * NOTE: every hook runs before any early return — the hook order must be
   * identical on the loading render and the loaded render. */
  const rChart = useMemo(() => {
    if (!data) return [] as { day: number; policy: number; mtd: number }[];
    const out: { day: number; policy: number; mtd: number }[] = [];
    for (let d = 0; d < data.days; d++) {
      out.push({ day: d + 1, policy: data.resistant_fraction.policy[d], mtd: data.resistant_fraction.mtd[d] });
    }
    return out;
  }, [data]);

  if (loading) {
    return <Panel title="🎞️ 365-Day Evolution & Resistance"><div style={{ color: 'var(--text2)' }}>Loading the 365-day artifact (validated engine frames; the file is fetched once, then frames decode on demand)…</div></Panel>;
  }
  if (error || !data || !armData) {
    return (
      <Panel title="🎞️ 365-Day Evolution & Resistance">
        <div style={{ color: 'var(--warning)', fontSize: 13, lineHeight: 1.6 }}>⚠ {error ?? 'artifact missing'}</div>
      </Panel>
    );
  }

  const frameDays = data.frame_days;
  const day = frameDays[Math.min(frameIdx, frameDays.length - 1)];
  const nextDay = frameDays[Math.min(frameIdx + 1, frameDays.length - 1)];
  const vtiData = currentFrame && data ? frameToVtiData(currentFrame, statics, data.grid, data.spacing_mm) : null;
  const p = data.arms.policy;
  const m = data.arms.mtd;

  /* Current-frame readouts (from the run's own arrays at this day). */
  const burdenP = p.burden_S[day] + p.burden_R[day];
  const burdenM = m.burden_S[day] + m.burden_R[day];
  const rFracP = data.resistant_fraction.policy[day];
  const rFracM = data.resistant_fraction.mtd[day];
  const curDoseA = armData.dose_A[day];
  const curDoseB = armData.dose_B[day];

  const armBadge = (a: 'policy' | 'mtd') => (
    <button
      key={a}
      className={`chip ${arm === a ? 'active' : ''}`}
      onClick={() => { setArm(a); setFrameIdx(Math.min(frameIdx, data.arms[a].frame_days.length - 1)); setBlendBoth(0); }}
      style={{ marginRight: 8 }}
    >
      <span className="chip-dot" style={{ background: a === 'policy' ? '#22c55e' : '#ef4444' }} />
      {a === 'policy' ? 'adaptive switch' : 'MTD control'}
    </button>
  );

  return (
    <div>
      <Panel
        title="🎞️ 365-Day Evolution & Treatment Resistance — every frame a validated engine state"
        sub={`${data.model}. ${data.geometry_note} Frame aggregates are cross-checked against the run's own daily curves before export (evolution_export.FrameSet.validate_against_trajectory).`}
      >
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          {armBadge('policy')}
          {armBadge('mtd')}
          <span className="badge-status live">● ENGINE FRAMES</span>
          <span className="badge-warn">CONSTANTS ASSUMED</span>
          <span className="badge">NOT A PATIENT PREDICTION</span>
        </div>

        {/* Transport bar */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          <button className={`nav-btn ${playing ? 'active' : ''}`} style={{ padding: '6px 14px' }}
            onClick={() => setPlaying((v) => !v)} disabled={!!frameError}>
            {playing ? '⏸ pause' : '▶ play'}
          </button>
          <div style={{ display: 'flex', gap: 4 }}>
            {SPEEDS.map((s) => (
              <button key={s} className={`chip ${speed === s ? 'active' : ''}`} onClick={() => setSpeed(s)}>
                {s}×
              </button>
            ))}
          </div>
          <input
            type="range" min={0} max={frameDays.length - 1} step={1} value={frameIdx}
            onChange={(e) => { setPlaying(false); setFrameIdx(parseInt(e.target.value, 10)); setBlendBoth(0); }}
            style={{ flex: '1 1 220px', minWidth: 180 }}
            aria-label="timeline"
          />
          <div className="mono" style={{ fontSize: 15, fontWeight: 650, minWidth: 110, textAlign: 'right' }}>
            day {day}<span style={{ color: 'var(--text3)', fontWeight: 400 }}> / {data.days}</span>
          </div>
        </div>

        {/* Headline row for the current frame */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginBottom: 12 }}>
          <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 12px', background: 'rgba(59,130,246,0.08)' }}>
            <div style={{ fontSize: 10, letterSpacing: '.8px', color: 'var(--text3)', textTransform: 'uppercase' }}>viewing</div>
            <div style={{ fontSize: 17, fontWeight: 650, color: arm === 'policy' ? 'var(--success)' : 'var(--danger)' }}>
              {arm === 'policy' ? 'adaptive switch' : 'MTD control'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>seed {data.seed} · same tumour, same physics</div>
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 12px' }}>
            <div style={{ fontSize: 10, letterSpacing: '.8px', color: 'var(--text3)', textTransform: 'uppercase' }}>burden @ day {day}</div>
            <div className="mono" style={{ fontSize: 17, fontWeight: 650 }}>{fmt(burdenP)}</div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>MTD control: {fmt(burdenM)}</div>
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 12px' }}>
            <div style={{ fontSize: 10, letterSpacing: '.8px', color: 'var(--text3)', textTransform: 'uppercase' }}>resistant fraction</div>
            <div className="mono" style={{ fontSize: 17, fontWeight: 650, color: rFracP > 0.5 ? 'var(--danger)' : 'var(--text)' }}>{pct(rFracP)}</div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>MTD control: {pct(rFracM)}</div>
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '10px 12px' }}>
            <div style={{ fontSize: 10, letterSpacing: '.8px', color: 'var(--text3)', textTransform: 'uppercase' }}>dose today (this arm)</div>
            <div className="mono" style={{ fontSize: 17, fontWeight: 650 }}>
              A {curDoseA > 0.5 ? '●' : '○'} · B {curDoseB > 0.5 ? '●' : '○'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>
              {curDoseB > 0.5 ? 'drug B active (switched)' : 'drug A only'}
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) minmax(320px, 1fr)', gap: 14 }}>
          {/* 3D volume at the current day */}
          <div style={{ minHeight: 560, display: 'flex', flexDirection: 'column' }}>
            {frameError ? (
              <div className="seq-empty">frame decode failed: {frameError}</div>
            ) : vtiData ? (
              <div style={{ flex: 1, border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                <VtiViewer
                  vtiBase64={null}
                  parsed={vtiData}
                  footerNote={
                    blend > 0
                      ? `blending day ${(day + blend * (nextDay - day)).toFixed(1)} between engine days ${day} and ${nextDay} of ${data.days} · ${arm === 'policy' ? 'adaptive' : 'MTD'} arm · volume interpolated, readouts stay on recorded days`
                      : `frame day ${day} of ${data.days} — exact engine state · ${arm === 'policy' ? 'adaptive' : 'MTD'} arm · seed ${data.seed} · same fields ParaView opens`
                  }
                />
              </div>
            ) : (
              <div className="seq-empty">decoding frame…</div>
            )}
          </div>

          {/* Synced resistance trajectory */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="panel" style={{ flex: 1 }}>
              <div className="panel-title">Resistant fraction — the divergence IS the resistance story</div>
              <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '4px 0 10px', lineHeight: 1.5 }}>
                R/(S+R) per day for both arms on the same tumour and seed. The cursor tracks the timeline;
                the dashed line is the switch-on threshold (15%). Reported as 0 below the ~1e8-cell detection
                floor: R/(S+R) hits 100% the moment the sensitive population is extinct, so a sub-detectable
                tumour is not 'resistant', it is too small for the ratio to mean anything — read the absolute
                burden figures above for those days.
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rChart} margin={{ top: 6, right: 14, bottom: 4, left: 0 }}>
                  <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)"
                    ticks={[1, 92, 183, 274, 365]} tickFormatter={(d: number) => `d${d}`} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" width={44}
                    tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                  <Tooltip
                    contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }}
                    formatter={(v: number) => pct(v)}
                    labelFormatter={(d: number) => `day ${d}`}
                  />
                  <ReferenceLine y={0.15} stroke="rgba(139,92,246,.6)" strokeDasharray="4 4"
                    label={{ value: 'switch ON 15%', fill: '#8b5cf6', fontSize: 10, position: 'insideTopRight' }} />
                  <ReferenceLine x={day + 1} stroke="#e2e8f0" strokeWidth={1.6}
                    label={{ value: `d${day}`, fill: '#e2e8f0', fontSize: 10, position: 'insideTopLeft' }} />
                  {m.pfs_day !== null && (
                    <ReferenceLine x={m.pfs_day + 1} stroke="rgba(239,68,68,.75)" strokeDasharray="6 3"
                      label={{ value: 'MTD PFS', fill: '#ef4444', fontSize: 10, position: 'insideTopLeft' }} />
                  )}
                  {showMtd && (
                    <Line type="monotone" dataKey="mtd" name="MTD control" stroke="#ef4444" strokeWidth={2} dot={false} />
                  )}
                  <Line type="monotone" dataKey="policy" name="adaptive switch" stroke="#22c55e" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <label className="vti-check" style={{ marginTop: 6 }}>
                <input type="checkbox" checked={showMtd} onChange={(e) => setShowMtd(e.target.checked)} />
                show MTD control curve
              </label>
            </div>

            {/* Arm outcome cards */}
            <div className="panel">
              <div className="panel-title">Outcome at the horizon (day {data.days})</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <div style={{ border: '1px solid rgba(34,197,94,.4)', borderRadius: 10, padding: '10px 12px', background: 'rgba(34,197,94,.07)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text2)' }}>adaptive switch</div>
                  <div className="mono" style={{ fontSize: 20, fontWeight: 650, color: 'var(--success)' }}>
                    {p.pfs_day !== null ? `PFS ${num(p.pfs_day)} d` : 'progression-free'}
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text2)', marginTop: 2 }}>
                    R {pct(data.resistant_fraction.policy[data.days - 1])} · toxicity {fmt(p.toxicity, 1)}
                  </div>
                </div>
                <div style={{ border: '1px solid rgba(239,68,68,.4)', borderRadius: 10, padding: '10px 12px', background: 'rgba(239,68,68,.07)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text2)' }}>MTD control</div>
                  <div className="mono" style={{ fontSize: 20, fontWeight: 650, color: 'var(--danger)' }}>
                    {m.pfs_day !== null ? `PFS ${num(m.pfs_day)} d` : 'progression-free'}
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text2)', marginTop: 2 }}>
                    R {pct(data.resistant_fraction.mtd[data.days - 1])} · toxicity {fmt(m.toxicity, 1)}
                  </div>
                </div>
              </div>
              <div style={{ marginTop: 10, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.6 }}>
                {data.r0_source.length > 180 ? `${data.r0_source.slice(0, 180)}…` : data.r0_source}
              </div>
            </div>
          </div>
        </div>
      </Panel>

      <Panel
        title="What the frames are"
        sub={artifact ? artifactStamp(artifact) : undefined}
      >
        <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text2)', fontSize: 12, lineHeight: 1.7 }}>
          {data.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
        <div style={{ marginTop: 10, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.7 }}>
          Frame budget: {data.budget.frames_per_arm} engine frames per arm (daily through day {data.budget.dense_until}, then
          coarsening; raise <span className="mono">build_evolution_artifact(max_frames=…)</span> for more). The slider steps where
          the          engine recorded, so every frame you land on is an exact engine state. Playback blends adjacent frames so the
          volume moves continuously between recorded days and snaps back to the recorded frame when paused; the daily
          curves and every readout stay one point per simulated day and are never interpolated.
        </div>
      </Panel>
    </div>
  );
}
