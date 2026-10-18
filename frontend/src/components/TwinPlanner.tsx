/**
 * TwinPlanner — upload a segmentation mask, run the paired 365-day research
 * simulation. It returns model outputs on that mask, never a treatment plan
 * for the person the scan came from.
 *
 * What this component promises: it renders ONLY what the /api/twin/plan
 * endpoint returns from the real simulation engine. The paired metrics,
 * daily curves, event table, and VTI export are engine outputs; the
 * provenance block (what is measured-from-your-file vs assumed vs refused)
 * is rendered verbatim. When no backend is running, the panel says so
 * instead of inventing a plausible plan — a fabricated number here would be
 * indistinguishable from a real one.
 */
import React, { useCallback, useRef, useState } from 'react';
import {
  Area, AreaChart, CartesianGrid, Legend, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

export interface TwinPlanResponse {
  geometry: {
    maskPath: string;
    gridShapeXyz: number[];
    totalVoxels: number;
    tumourVoxels: number;
    tumourVolumeMm3: number;
    voxelMm: number;
    boundingBoxXyz: { min: number[]; max: number[] };
    vasculatureSource: string;
    proliferativeSource: string;
    engineBudgetVoxels: number;
    budgetFraction: number;
    /* Provenance of the input file, reported beside the grid the engine ran. */
    originalGridShapeXyz?: number[];
    sourceVoxelMm?: number;
    sourceVoxelSizesMm?: number[];
    downsampleFactors?: number[];
    downsamplePerAxis?: number;
    inputSha256?: string;
  };
  scheduleText: string;
  dailyCurves: {
    days: number[];
    burden_S: (number | null)[];
    burden_R: (number | null)[];
    dose_A: number[];
    dose_B: number[];
    mtdBurdenTotal: number[];
  };
  pairedMetrics: {
    policyMedianPfsDays: number;
    mtdMedianPfsDays: number;
    pfsExtensionMedian: number | null;
    toxicityReductionMedian: number | null;
    nReplicates: number;
    policyPfsReplicates: number[];
    mtdPfsReplicates: number[];
    refPfsDay: number | null;
    refProgressed: boolean;
    refToxicity: number;
    mtdToxicity: number;
    refFinalResistantFraction: number;
  };
  events: { day: number; action: string; resistantFraction: number; burden: number }[];
  notes: string[];
  provenance: Record<string, string>;
}

interface TwinState {
  loading: boolean;
  data: TwinPlanResponse | null;
  error: string | null;
  vtiUrl: string | null;
  elapsed: number | null;
}

const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;

const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

interface Props { onVti?: (b64: string | null) => void }

export default function TwinPlanner({ onVti }: Props) {
  const [state, setState] = useState<TwinState>({ loading: false, data: null, error: null, vtiUrl: null, elapsed: null });
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = useCallback(async (file: File) => {
    setState({ loading: true, data: null, error: null, vtiUrl: null, elapsed: null });
    onVti?.(null);
    const t0 = performance.now();
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch('/api/twin/plan', { method: 'POST', body: fd });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail.slice(0, 400) || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      // The VTI blob is base64 ASCII inside the VTK XML: a plain Blob is a
      // valid ParaView file (the XML declares its own encoding).
      const blob = new Blob([payload.vtiBase64 ? atob(payload.vtiBase64) : ''], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      onVti?.(payload.vtiBase64 ?? null);
      setState({ loading: false, data: payload, error: null, vtiUrl: url, elapsed: (performance.now() - t0) / 1000 });
    } catch (e: any) {
      onVti?.(null);
      setState({ loading: false, data: null, error: e?.message || 'planning failed', vtiUrl: null, elapsed: null });
    }
  }, [onVti]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) submit(f);
  }, [submit]);

  const d = state.data;
  const curveData = d ? d.dailyCurves.days.map((day, i) => ({
    day,
    sensitive: d.dailyCurves.burden_S[i],
    resistant: d.dailyCurves.burden_R[i],
    mtd: d.dailyCurves.mtdBurdenTotal[i],
  })) : [];
  const doseData = d ? d.dailyCurves.days.map((day, i) => ({
    day,
    drugA: d.dailyCurves.dose_A[i],
    drugB: d.dailyCurves.dose_B[i],
  })) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ---------- upload zone ---------- */}
      {!d && (
        <div
          className={`upload-zone ${dragOver ? 'drag' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          style={{ minHeight: 220, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: state.loading ? 'wait' : 'pointer' }}
        >
          <input
            ref={inputRef} type="file" accept=".nii,.nii.gz" hidden
            onChange={e => { const f = e.target.files?.[0]; if (f) submit(f); }}
          />
          {state.loading ? (
            <>
              <div className="upload-zone-icon"><span className="spinner" /></div>
              <div className="upload-zone-text">Running paired 365-day twin simulation…</div>
              <div className="upload-zone-text" style={{ fontSize: 11 }}>intake → clonal competition → schedule extraction → VTI export</div>
            </>
          ) : (
            <>
              
              <div className="upload-zone-text">Drop a tumour segmentation mask (.nii / .nii.gz)</div>
              <div className="upload-zone-text" style={{ fontSize: 11 }}>
                any binary or label mask; nonzero = tumour · ≤4,000,000 voxels · auto-downsampled if larger
              </div>
              {!IS_LOCAL && (
                <div className="upload-zone-text" style={{ fontSize: 11, color: '#fbbf24', marginTop: 8 }}>
                  ⚠ live planning needs the local API (uvicorn api_server:app) — this deployed site renders precomputed engine artifacts
                </div>
              )}
            </>
          )}
        </div>
      )}

      {state.error && (
        <div className="panel glass"><div className="panel-title" style={{ color: '#f87171' }}>⚠ Simulation failed</div>
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{state.error}</pre></div>
      )}

      {/* ---------- results ---------- */}
      {d && (
        <>
          <div className="plan-metrics-row">
            <Metric label="Simulated steering median PFS" value={`${d.pairedMetrics.policyMedianPfsDays.toFixed(0)} d`} sub={`${d.pairedMetrics.nReplicates} paired replicates`} accent="cyan" />
            <Metric label="Simulated MTD median PFS" value={`${d.pairedMetrics.mtdMedianPfsDays.toFixed(0)} d`} sub="same geometry, same seeds" accent="blue" />
            <Metric label="Simulated PFS extension" value={pct(d.pairedMetrics.pfsExtensionMedian)} sub={d.pairedMetrics.pfsExtensionMedian === null ? 'no progression in control' : 'paired median over replicates'} accent="violet" />
            <Metric label="Simulated toxicity reduction" value={pct(d.pairedMetrics.toxicityReductionMedian)} sub={`steering ${d.pairedMetrics.refToxicity.toFixed(0)} vs MTD ${d.pairedMetrics.mtdToxicity.toFixed(0)} dose-days`} accent="cyan" />
          </div>
          <div className="plan-note">
            These four numbers are <b>model outputs</b> from the simulation on your uploaded mask under
            assumed engine parameters. They are not patient outcome predictions, and they are not a
            treatment recommendation or a schedule to give anyone.
          </div>

          <div className="content-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="panel glass">
              <div className="panel-title">Tumour burden — steering vs MTD control</div>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={curveData} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                  <defs>
                    <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.5} /><stop offset="100%" stopColor="#06b6d4" stopOpacity={0.05} />
                    </linearGradient>
                    <linearGradient id="gR" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.5} /><stop offset="100%" stopColor="#f43f5e" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                  <XAxis dataKey="day" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.3)', borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area type="monotone" dataKey="sensitive" stroke="#06b6d4" fill="url(#gS)" name="sensitive burden" strokeWidth={2} />
                  <Area type="monotone" dataKey="resistant" stroke="#f43f5e" fill="url(#gR)" name="resistant burden" strokeWidth={2} />
                  <Line type="monotone" dataKey="mtd" stroke="#94a3b8" strokeDasharray="6 3" dot={false} name="MTD total (control)" strokeWidth={1.5} />
                  {d.pairedMetrics.refPfsDay !== null && (
                    <ReferenceLine x={d.pairedMetrics.refPfsDay} stroke="#f43f5e" strokeDasharray="4 4"
                      label={{ value: 'progression', fill: '#f43f5e', fontSize: 11, position: 'insideTopRight' }} />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="panel glass">
              <div className="panel-title">Executed dose schedule (extracted from dose logs)</div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={doseData} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                  <XAxis dataKey="day" stroke="#64748b" fontSize={11} />
                  <YAxis domain={[0, 1]} ticks={[0, 1]} stroke="#64748b" fontSize={11} />
                  <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.3)', borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="stepAfter" dataKey="drugA" stroke="#3b82f6" dot={false} name="drug A" strokeWidth={2} />
                  <Line type="stepAfter" dataKey="drugB" stroke="#8b5cf6" dot={false} name="drug B" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="content-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="panel glass">
              <div className="panel-title">Uploaded geometry (measured from your file)</div>
              <table className="kv-table">
                <tbody>
                  <Row k="Grid (engine)" v={`${d.geometry.gridShapeXyz.join(' × ')} = ${d.geometry.totalVoxels.toLocaleString()} voxels`} />
                  {d.geometry.originalGridShapeXyz && d.geometry.originalGridShapeXyz.join() !== d.geometry.gridShapeXyz.join() && (
                    <Row k="Grid (your file)" v={`${d.geometry.originalGridShapeXyz.join(' × ')} = ${d.geometry.originalGridShapeXyz.reduce((a, b) => a * b, 1).toLocaleString()} voxels`} />
                  )}
                  <Row k="Tumour" v={`${d.geometry.tumourVoxels.toLocaleString()} voxels · ${d.geometry.tumourVolumeMm3.toLocaleString(undefined, { maximumFractionDigits: 0 })} mm³`} />
                  <Row
                    k="Voxel size (engine)"
                    v={d.geometry.downsamplePerAxis
                      ? `${d.geometry.voxelMm.toFixed(2)} mm · your file declared ${d.geometry.sourceVoxelMm?.toFixed(2)} mm`
                      : `${d.geometry.voxelMm.toFixed(2)} mm (from file header)`}
                  />
                  {d.geometry.downsampleFactors && (
                    <Row k="Downsampled" v={`block-max ×${d.geometry.downsamplePerAxis} per axis, tumour-preserving (a block containing tumour stays tumour)`} />
                  )}
                  <Row k="Bounding box" v={d.geometry.boundingBoxXyz.min.join(', ') + ' → ' + d.geometry.boundingBoxXyz.max.join(', ')} />
                  <Row k="Vasculature" v={d.geometry.vasculatureSource} />
                  <Row k="Proliferation" v={d.geometry.proliferativeSource} />
                  <Row k="Engine budget" v={`${(d.geometry.budgetFraction * 100).toFixed(1)}% of 4.0M voxel budget`} />
                  {d.geometry.inputSha256 && (
                    <Row k="Input SHA-256" v={`${d.geometry.inputSha256.slice(0, 16)}… (full hash in the API response)`} />
                  )}
                </tbody>
              </table>
              <div className="plan-actions">
                {state.vtiUrl && (
                  <a className="btn btn-run" href={state.vtiUrl} download="twin_fields.vti" style={{ textDecoration: 'none', textAlign: 'center' }}>
                    ⬇ Download ParaView .vti (final fields)
                  </a>
                )}
                <button className="btn btn-run" onClick={() => { onVti?.(null); setState({ loading: false, data: null, error: null, vtiUrl: null, elapsed: null }); }}>
                  ⟲ Simulate another mask
                </button>
              </div>
              <div className="plan-note">
                The .vti opens directly in ParaView: four point scalars (drug_A, drug_B, sensitive, resistant) plus the tumour mask,
                on {d.geometry.downsampleFactors
                  ? `the engine's downsampled grid (×${d.geometry.downsamplePerAxis} coarser per axis than your file)`
                  : "the uploaded file's own voxel grid"}. <b>No ParaView install? The 3D panel below is the same data rendered in-browser.</b>
              </div>
              {d.geometry.downsampleFactors && (
                <div className="plan-note" style={{ color: '#fbbf24' }}>
                  Your mask exceeded the engine's 4.0M-voxel budget, so this run used a block-max-downsampled copy.
                  The tumour volume above is the block-max volume on the coarser grid, so it can only be equal-or-larger
                  than the source, never smaller; the engine's constants are voxel-unit, so its voxel is the coarse one
                  ({d.geometry.voxelMm.toFixed(2)} mm, taken from the scaled header).
                </div>
              )}
            </div>

            <div className="panel glass">
              <div className="panel-title">Regime changes (extracted from the executed dose logs)</div>
              {d.events.length ? (
                <table className="kv-table">
                  <thead><tr><th>Day</th><th>Action</th><th>R fraction</th><th>Burden</th></tr></thead>
                  <tbody>
                    {d.events.map((ev, i) => (
                      <tr key={i}>
                        <td className="mono">{ev.day}</td>
                        <td>{ev.action}</td>
                        <td className="mono">{pct(ev.resistantFraction)}</td>
                        <td className="mono">{ev.burden.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <div className="seq-empty">None — the policy never changed regime.</div>}
            </div>
          </div>

          <div className="panel glass">
            <div className="panel-title">Compiled schedule text (engine output, verbatim)</div>
            <pre className="mono schedule-pre">{d.scheduleText}</pre>
          </div>

          <div className="panel glass" style={{ borderColor: 'rgba(245,158,11,0.35)' }}>
            <div className="panel-title" style={{ color: '#fbbf24' }}>⚖ What is measured vs assumed vs refused</div>
            <table className="kv-table">
              <tbody>
                {Object.entries(d.provenance).map(([k, v]) => (
                  <tr key={k}><td style={{ width: 130, color: '#93c5fd' }}>{k}</td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
            <ul className="plan-notes">
              {d.notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          </div>

          <div className="plan-elapsed mono">planning run: {state.elapsed?.toFixed(1)} s · {d.pairedMetrics.nReplicates} paired replicates · MTD control on identical seeds</div>
        </>
      )}
    </div>
  );
}

function Metric({ label, value, sub, accent }: { label: string; value: string; sub: string; accent: string }) {
  return (
    <div className={`panel glass plan-metric accent-${accent}`}>
      <div className="plan-metric-label">{label}</div>
      <div className="plan-metric-value">{value}</div>
      <div className="plan-metric-sub">{sub}</div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return <tr><td style={{ width: 130, color: '#93c5fd' }}>{k}</td><td className="mono" style={{ fontSize: 12 }}>{v}</td></tr>;
}
