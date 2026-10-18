/**
 * ScanGallery — the actual tumour scans this repository holds, rendered.
 *
 * Loads `artifacts/scans.json` (written by `_scan_gallery.py` from the cached
 * BraTS-2021 NIfTI files) and shows the real T1ce MRI of every committed case
 * with the provided expert segmentation drawn on top. The contours are labels,
 * not predictions: no engine output is composited onto these images, and the
 * artifact's provenance records `model_prediction_on_these_images` as
 * PROVEN_FAILED so a future edit cannot quietly start drawing one.
 *
 * Each card can also POST that case's own engine-ready mask (served under
 * `artifacts/masks/`, <1 KB each) to the live API and show the real run that
 * comes back. That run takes tens of seconds, so the button counts it up and
 * the card says so up front rather than leaving a visitor watching a dead
 * button.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ScansArtifactData, ScanCaseRow } from '../types/artifacts';
import type { TwinPlanResponse } from './TwinPlanner';
import { livePlanLines } from '../utils/livePlan';

const fmt = (n: number | null | undefined): string =>
  n === null || n === undefined ? '—' : n.toLocaleString('en-US');

const LABEL_NAMES: Record<string, string> = {
  '1': 'necrotic',
  '2': 'oedema',
  '4': 'enhancing',
};

/** Simulated horizon per policy, and roughly how long that takes to compute. */
const HORIZON_DAYS = 365;

/**
 * Measured, not guessed: all six served masks POSTed to a local `/api/twin/plan`
 * on an otherwise idle machine took 22.3 / 25.3 / 28.6 / 23.8 / 26.8 / 24.4 s
 * (mean 25.2). The wall time tracks grid size and fixed overhead more than
 * tumour voxel count — 01020, the largest mask at 3,623 voxels, was *faster*
 * than 01019 at 2,826. So the copy quotes the measured range rather than one
 * optimistic number, and a busy machine pushes it higher, which the per-second
 * counter shows honestly.
 */
const TYPICAL_RUN_RANGE = '20–30 s';

const STEPS = [
  {
    n: 1,
    body: (
      <>
        Read the scan. These are committed patient MRIs — the contours are the
        dataset's expert labels, not this engine's output.
      </>
    ),
  },
  {
    n: 2,
    body: (
      <>
        Press <strong>Run live plan</strong> on any case. It posts that
        patient's mask to the local engine and simulates two {HORIZON_DAYS}-day
        dosing policies side by side — measured at {TYPICAL_RUN_RANGE} on an idle
        machine, longer if it is busy.
      </>
    ),
  },
  {
    n: 3,
    body: (
      <>
        Results are recorded engine output from this repository — not a
        measurement of that patient. Provenance is at the bottom.
      </>
    ),
  },
];

interface RunState {
  loading: boolean;
  data: TwinPlanResponse | null;
  error: string | null;
  elapsed: number | null;
}

function LivePlanResult({ data }: { data: TwinPlanResponse }) {
  // The lines come from utils/livePlan.ts so _verify_live_plan.mjs can assert
  // this exact wording headlessly; the component adds no arithmetic of its own.
  const lines = livePlanLines(data);
  return (
    <div className="seq-meta scan-result" aria-live="polite">
      {lines.map((l, i) => (
        <span key={i} className={i === 0 ? 'mono' : undefined}>
          {l}
        </span>
      ))}
      <span style={{ color: 'var(--warning)' }}>
        recorded engine output from this repository — not a measurement of this patient
      </span>
    </div>
  );
}

function LivePlan({ row, base }: { row: ScanCaseRow; base: string }) {
  const [run, setRun] = useState<RunState>({
    loading: false,
    data: null,
    error: null,
    elapsed: null,
  });
  const [tick, setTick] = useState(0);

  // A tens-of-seconds wait with a frozen button reads as a hang. Count it up so
  // the visitor can see it is alive and roughly how much is left.
  useEffect(() => {
    if (!run.loading) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [run.loading]);

  const runPlan = useCallback(async () => {
    setRun({ loading: true, data: null, error: null, elapsed: null });
    setTick(0);
    const t0 = performance.now();
    try {
      const res = await fetch(`${base}artifacts/masks/${row.engine_mask_file}`);
      if (!res.ok) throw new Error(`mask not served (HTTP ${res.status})`);
      const fd = new FormData();
      fd.append(
        'file',
        new File([await res.blob()], row.engine_mask_file, {
          type: 'application/gzip',
        }),
      );
      const plan = await fetch('/api/twin/plan', { method: 'POST', body: fd });
      if (!plan.ok) throw new Error((await plan.text()).slice(0, 300) || `HTTP ${plan.status}`);
      setRun({
        loading: false,
        data: await plan.json(),
        error: null,
        elapsed: (performance.now() - t0) / 1000,
      });
    } catch (e: any) {
      setRun({
        loading: false,
        data: null,
        error: e?.message || 'planning failed',
        elapsed: null,
      });
    }
  }, [row.engine_mask_file, base]);

  const label = run.loading
    ? `Running 365-day twin… ${tick}s`
    : run.error
      ? 'Try again'
      : run.data
        ? 'Run it again'
        : 'Run live plan on this mask';

  return (
    <div>
      <button
        className="btn scan-run"
        onClick={runPlan}
        disabled={run.loading}
        aria-busy={run.loading}
      >
        {run.loading && <span className="spinner" aria-hidden="true" />}
        {label}
      </button>
      {run.loading && (
        <div className="scan-elapsed">
          2 policies × {HORIZON_DAYS} days · same grid, same seeds — measured{' '}
          {TYPICAL_RUN_RANGE} here, and it keeps counting
        </div>
      )}
      {run.error && (
        <div className="run-error" role="alert">
          Live API unavailable — {run.error}
          <div className="run-note">
            Start it with{' '}
            <span className="mono">uvicorn api_server:app --port 8000</span>, then press try
            again. The scans themselves need no backend.
          </div>
        </div>
      )}
      {run.data && (
        <>
          <LivePlanResult data={run.data} />
          {run.elapsed !== null && (
            <div className="scan-elapsed">took {run.elapsed.toFixed(1)} s in your browser</div>
          )}
        </>
      )}
    </div>
  );
}

export default function ScanGallery() {
  const { artifact, loading, error } = useArtifact<ScansArtifactData>('scans');

  if (loading)
    return (
      <div className="seq-empty">
        Loading the committed scans… if this stays, the artifact was never built — run{' '}
        <code>python _scan_gallery.py</code>.
      </div>
    );
  if (error)
    return (
      <div className="seq-empty">
        MRI scans could not be loaded ({error}). Rebuild with{' '}
        <code>python _scan_gallery.py</code>, which writes{' '}
        <code>artifacts/scans.json</code> and the PNGs next to it.
      </div>
    );
  if (!artifact) return null;

  const base = import.meta.env.BASE_URL;
  const d = artifact.data;

  return (
    <div className="content-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
        {/* First run: three sentences, then the scans. */}
        <div className="scan-steps">
          {STEPS.map((s) => (
            <div key={s.n} className="scan-step">
              <span className="scan-step-n">{s.n}</span>
              <span>{s.body}</span>
            </div>
          ))}
        </div>

        {/* Featured: three orthogonal views of the case every cohort runs on. */}
        <div className="panel glass">
          <div className="panel-title">
            Real T1ce MRI + expert segmentation — {d.standard_case}
            <span className="readonly-badge">MEASURED REAL DATA</span>
          </div>
          <figure className="scan-figure">
            <img
              src={`${base}artifacts/scans/${d.montage.file}`}
              alt={`${d.standard_case}: axial, coronal and sagittal T1ce slices with tumour contour`}
            />
            <figcaption className="scan-legend">
              <span>
                <i style={{ background: '#ff4136' }} /> whole tumour
              </span>
              <span>
                <i style={{ background: '#ffd400' }} /> enhancing
              </span>
              <span>
                <i style={{ background: '#9aa4b0' }} /> brain tissue (T1ce grayscale)
              </span>
              <span className="scan-legend-note">
                three orthogonal views · orientation follows the NIfTI array order
              </span>
            </figcaption>
          </figure>
          <div className="plan-note">{d.disclaimer}</div>
        </div>

        {/* Every committed case, with what the engine actually ran on. */}
        <div className="panel glass">
          <div className="panel-title">
            All {d.cases.length} committed cases
            <span className="readonly-badge">BraTS-2021 · cached in data/brats</span>
          </div>
          <div className="scan-grid">
            {d.cases.map((c) => (
              <div
                key={c.case}
                className={`glass scan-card${c.is_standard_case ? ' is-standard' : ''}`}
              >
                <div className="scan-figure">
                  <img
                    src={`${base}artifacts/scans/${c.image}`}
                    alt={`${c.case}: axial T1ce slice ${c.slice_index} with expert tumour contour`}
                    loading="lazy"
                  />
                </div>
                <div className="scan-card-title">
                  <span className="mono">{c.case}</span>
                  {c.is_standard_case && <span className="chip chip-btn">start here</span>}
                </div>
                <div className="scan-stats">
                  <div className="scan-stat">
                    <span>labelled voxels</span>
                    <strong>{fmt(c.seg_voxels_total)}</strong>
                  </div>
                  <div className="scan-stat">
                    <span>simulated</span>
                    <strong>{fmt(c.engine_mask_voxels)}</strong>
                  </div>
                </div>
                <div className="scan-meta mono">
                  grid {c.engine_grid_xyz ? c.engine_grid_xyz.join('×') : '—'} · axial slice{' '}
                  {c.slice_index} ·{' '}
                  {Object.entries(c.seg_voxels_by_label)
                    .map(([k, v]) => `${LABEL_NAMES[k] ?? k} ${fmt(v)}`)
                    .join(', ')}
                </div>
                <LivePlan row={c} base={base} />
              </div>
            ))}
          </div>
        </div>

        {/* Where the pixels came from, and what they are not. */}
        <div className="panel glass">
          <div className="panel-title">Provenance</div>
          <div className="seq-meta" style={{ lineHeight: 1.6 }}>
            <span>source: {artifact.source}</span>
            <span className="mono" style={{ wordBreak: 'break-all' }}>
              {artifactStamp(artifact)}
            </span>
          </div>
          <ul className="plan-notes" style={{ marginTop: 8 }}>
            {artifact.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
          <div className="plan-note">
            These are pre-operative scans. No post-operative imaging exists in this
            repository (claim #49), so no resection cavity or recurrence follow-up
            can be shown here — that is a refusal, not a missing feature.
          </div>
        </div>
      </div>
    </div>
  );
}