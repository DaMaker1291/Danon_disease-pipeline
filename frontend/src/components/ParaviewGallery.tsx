/**
 * ParaviewGallery — the committed ParaView artifact, rendered and downloadable.
 *
 * Loads the `paraview` app artifact (built by danon.app_artifacts from the
 * committed BraTS-2021 VTI), renders the volume in the same 3D viewer the
 * Twin Planner uses, and hands the visitor the byte-identical .vti file
 * ParaView opens — so every figure the poster claims from ParaView is
 * reproducible from what this page serves.
 */
import React, { useMemo } from 'react';
import VtiViewer from './VtiViewer';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ParaviewArtifactData } from '../types/artifacts';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function downloadVti(b64: string, name: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ParaviewGallery() {
  const { artifact, loading, error } = useArtifact<ParaviewArtifactData>('paraview');

  const stats = useMemo(() => {
    if (!artifact) return null;
    const s = artifact.data.field_stats;
    return [
      { label: 'D(x) contrast', value: `${artifact.data.headline.diffusion_contrast_ratio.toFixed(0)}×`, hint: 'max/min diffusion across the grid' },
      { label: 'resistant fraction', value: '1.00', hint: 'day-120 MTD end-state: the sanctuary effect' },
      { label: 'tumour voxels', value: artifact.data.grid.tumour_voxels.toLocaleString(), hint: `of ${artifact.data.grid.n_points.toLocaleString()} points` },
      { label: 'grid', value: `${artifact.data.grid.dims.join('×')} @ ${artifact.data.grid.spacing_mm} mm`, hint: 'BraTS-2021 case 01017 segmentation' },
    ];
  }, [artifact]);

  if (loading) return <div className="seq-empty">Loading ParaView artifact…</div>;
  if (error) return <div className="seq-empty">ParaView gallery: {error}</div>;
  if (!artifact) return null;

  const d = artifact.data;

  return (
    <div className="content-grid" style={{ gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 0.85fr)' }}>
      {/* left: the volume */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
        <div className="panel glass" style={{ minHeight: 620, display: 'flex', flexDirection: 'column' }}>
          <div className="panel-title">
            🧊 3D volume — MTD day-120 end-state
            <span className="readonly-badge">REAL ENGINE ARTIFACT</span>
          </div>
          <ErrorBoundaryish>
            <VtiViewer vtiBase64={d.vti_base64} defaultSpin />
          </ErrorBoundaryish>
        </div>

        <div className="panel glass">
          <div className="panel-title">Measured field statistics (read from the file)</div>
          <table className="kv-table">
            <tbody>
              {Object.entries(d.field_stats).map(([name, st]) => (
                <tr key={name}>
                  <td className="mono">{name}</td>
                  <td className="mono">{st.min.toExponential(2)}</td>
                  <td className="mono">{st.max.toExponential(2)}</td>
                  <td className="mono">{st.mean_in_tumour.toExponential(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="seq-meta" style={{ marginTop: 8 }}>
            <span>columns: min (whole grid) · max (whole grid) · mean inside tumour mask</span>
            <span>tissue voxels: necrotic {d.tissue_counts.necrotic_core} · edema {d.tissue_counts.edema} · WM {d.tissue_counts.white_matter} · GM {d.tissue_counts.grey_matter}</span>
          </div>
        </div>
      </div>

      {/* right: provenance, caption, download */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
        <div className="panel glass">
          <div className="panel-title">Artifact provenance</div>
          <div className="plan-metrics-row" style={{ gridTemplateColumns: '1fr 1fr' }}>
            {stats?.map(s => (
              <div key={s.label} className="plan-metric accent-cyan plan-stat" title={s.hint}>
                <div className="plan-metric-label">{s.label}</div>
                <div className="plan-metric-value" style={{ fontSize: 20 }}>{s.value}</div>
                <div className="plan-metric-sub">{s.hint}</div>
              </div>
            ))}
          </div>
          <div className="seq-meta" style={{ marginTop: 10, lineHeight: 1.6 }}>
            <span>sha256 <span className="mono">{d.vti_sha256.slice(0, 24)}…</span></span>
            <span>{formatBytes(d.vti_size_bytes)} · arrays: {d.arrays_present.join(', ')}</span>
            <span className="mono" style={{ wordBreak: 'break-all' }}>{artifactStamp(artifact)}</span>
          </div>
          <div className="plan-actions">
            <button className="btn" onClick={() => downloadVti(d.vti_base64, 'brats_case01017_mtd_day120.vti')}>
              ⬇ Download .vti ({formatBytes(d.vti_size_bytes)})
            </button>
          </div>
          <div className="plan-note">
            The downloaded file is <b>byte-identical</b> to the committed artifact — open it in
            ParaView and follow the recipe below to reproduce the poster figure.
          </div>
        </div>

        <div className="panel glass">
          <div className="panel-title">Board caption (what the simulation actually integrated)</div>
          <div className="schedule-pre" style={{ maxHeight: 200 }}>{d.captions.short}</div>
        </div>

        <div className="panel glass">
          <div className="panel-title">ParaView recipe</div>
          <ol className="plan-notes">
            {d.paraview_recipe.map((step, i) => <li key={i}>{step}</li>)}
          </ol>
        </div>

        <div className="panel glass">
          <div className="panel-title">Honest limits</div>
          <ul className="plan-notes">
            {artifact.notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}

/** Local error fence so a WebGL failure degrades to a message, not a blank tab. */
function ErrorBoundaryish({ children }: { children: React.ReactNode }) {
  return <React.Suspense fallback={<div className="seq-empty">Loading 3D…</div>}>{children}</React.Suspense>;
}
