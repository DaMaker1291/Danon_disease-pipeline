import React from 'react';
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ClinicalArtifactData } from '../types/artifacts';

/**
 * Modelled cardiomyocyte rescue, read from `artifacts/clinical.json`.
 *
 * This view used to draw an invented LVMI/survival curve from a fitted
 * exponential (`Math.exp(-0.015 * m)` …). It now renders the 8-ODE cell
 * simulator's own dose-response panel and per-mutation comparison, and says
 * in the footer exactly what a model of one cardiomyocyte is not.
 */
export default function ClinicalDashboard() {
  const { artifact, loading, error } = useArtifact<ClinicalArtifactData>('clinical');

  if (loading) return <div className="panel"><div className="panel-title">Modelled cardiomyocyte rescue</div><div className="seq-empty">Loading cell-simulator artifact…</div></div>;
  if (error || !artifact) return <div className="panel"><div className="panel-title">Modelled cardiomyocyte rescue</div><div className="seq-empty">⚠ {error}</div></div>;

  const { dose_response, mutation_comparison, benchmark, vector_copy_number } = artifact.data;

  const ucl = benchmark.ucl_trial_score;
  const ours = benchmark.platform_best_score;

  return (
    <div className="panel">
      <div className="panel-title">Modelled cardiomyocyte rescue — 8-ODE cell simulator</div>
      <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '4px 0 12px', lineHeight: 1.5 }}>
        Dose-response sweep and per-mutation comparison at VCN = {vector_copy_number}. Every point is a solved
        ODE system for one modelled cardiomyocyte — not a measurement.
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <span className="badge-status live">● ENGINE OUTPUT</span>
        <span className="badge-warn">MODEL, NOT ASSAY</span>
        <span className="badge">{dose_response.length} dose points</span>
        <span className="badge">{mutation_comparison.length} mutation classes</span>
      </div>

      <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text2)', marginBottom: 6 }}>
        Restored function vs vector copy number
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={dose_response} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="rgba(51,65,85,.4)" strokeDasharray="3 3" />
          <XAxis dataKey="vector_copy_number" tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)"
            label={{ value: 'vector copy number', fill: '#64748b', fontSize: 10, position: 'insideBottom', offset: -2 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" width={44} />
          <Tooltip contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 10.5, color: 'var(--text2)' }} />
          <Line type="monotone" dataKey="lamp2b_restoration_pct" name="LAMP2B restored (%)" stroke="#22c55e" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="glycogen_clearance_pct" name="glycogen clearance (%)" stroke="#3b82f6" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="autophagic_flux" name="autophagic flux (a.u.)" stroke="#a78bfa" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>

      <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text2)', margin: '14px 0 6px' }}>
        Functional cure score and modelled 5-year survival
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={dose_response} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="rgba(51,65,85,.4)" strokeDasharray="3 3" />
          <XAxis dataKey="vector_copy_number" tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" />
          <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" width={44} />
          <Tooltip contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 10.5, color: 'var(--text2)' }} />
          <ReferenceLine y={0.7} stroke="rgba(251,191,36,.7)" strokeDasharray="5 4"
            label={{ value: 'cure threshold', fill: '#fbbf24', fontSize: 10, position: 'insideTopRight' }} />
          <Line type="monotone" dataKey="functional_cure_score" name="cure score" stroke="#f59e0b" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="survival_5y" name="5-year survival (modelled)" stroke="#06b6d4" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>

      <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text2)', margin: '14px 0 6px' }}>
        By mutation class (VCN = {vector_copy_number})
      </div>
      <div className="mut-table-wrap">
        <table className="mut-table">
          <thead>
            <tr>
              <th>mutation</th><th>LAMP2 (%)</th><th>clearance (%)</th>
              <th>flux (%)</th><th>cure score</th><th>cure?</th><th>10-y survival</th>
            </tr>
          </thead>
          <tbody>
            {mutation_comparison.map((m) => (
              <tr key={m.mutation}>
                <td style={{ textTransform: 'capitalize' }}>{m.mutation}</td>
                <td>{m.lamp2_pct.toFixed(1)}</td>
                <td>{m.glycogen_clearance_pct.toFixed(1)}</td>
                <td>{m.flux_restoration_pct.toFixed(1)}</td>
                <td>{m.cure_score.toFixed(3)}</td>
                <td style={{ color: m.is_cure ? 'var(--success)' : 'var(--warning)' }}>{m.is_cure ? 'yes' : 'no'}</td>
                <td>{(m.survival_10y * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(typeof ucl === 'number' || typeof ours === 'number') && (
        <div style={{ display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
          <div className="stat-chip"><span>UCL trial score</span><strong>{typeof ucl === 'number' ? ucl.toFixed(3) : '—'}</strong></div>
          <div className="stat-chip ok"><span>this platform, best</span><strong>{typeof ours === 'number' ? ours.toFixed(3) : '—'}</strong></div>
          {typeof ucl === 'number' && typeof ours === 'number' && (
            <div className="stat-chip"><span>ratio</span><strong>{(ours / ucl).toFixed(2)}×</strong></div>
          )}
        </div>
      )}

      <div style={{ marginTop: 14, color: 'var(--text3)', fontSize: 11, lineHeight: 1.7 }}>
        {artifactStamp(artifact)}
        <ul style={{ margin: '8px 0 0', paddingLeft: 16 }}>
          {artifact.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      </div>
    </div>
  );
}
