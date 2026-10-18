import React from 'react';
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { TumorBoardData, Refusal, RefusedField } from '../types/tumorboard';
import type { TabId } from '../nav';

/**
 * The tumour board: five panels, and the refusals that make them readable.
 *
 * EVERY value here is read from `artifacts/tumorboard.json`, which is assembled
 * from recorded engine output -- the outcome cohort, the day frames, the
 * performance record and the live claims audit. Nothing is computed at render
 * time, and nothing is filled in when its record is missing: a failed fetch
 * says so.
 *
 * The panel that matters most is not the 3D one. It is `not_recorded` on the
 * patient card, which lists the fields a clinical dashboard normally leads
 * with -- MGMT methylation, Ki-67, 5-ALA residual, KPS -- and says, field by
 * field, that this repository holds no value for them, what would have to
 * produce one, and what a mock-up showed instead. A dashboard that renders those
 * chips without a source is the specific failure this artifact exists to make
 * impossible.
 */

const CARD: React.CSSProperties = {
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius)',
  padding: '12px 14px',
  minWidth: 0,
  background: 'linear-gradient(160deg, rgba(59,130,246,0.08), rgba(17,24,39,0.5))',
};

const REFUSED: React.CSSProperties = {
  border: '1px solid rgba(245,158,11,.42)',
  background: 'rgba(245,158,11,.07)',
  color: '#fcd34d',
  borderRadius: 10,
  padding: '10px 12px',
  fontSize: 12,
  lineHeight: 1.55,
};

const MONO: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11.5,
  fontVariantNumeric: 'tabular-nums',
};

function Panel({
  title, sub, badge, children,
}: { title: string; sub?: string; badge?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="panel" style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <div className="panel-title" style={{ margin: 0 }}>{title}</div>
        {badge}
      </div>
      {sub && (
        <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '6px 0 12px', lineHeight: 1.55 }}>{sub}</div>
      )}
      {children}
    </div>
  );
}

function Field({ label, value, source }: { label: string; value: React.ReactNode; source?: string }) {
  return (
    <div style={{ padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ fontSize: 10.5, letterSpacing: '.7px', color: 'var(--text3)', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, margin: '2px 0', color: 'var(--text)' }}>{value}</div>
      {source && <div style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.45 }}>{source}</div>}
    </div>
  );
}

function RefusedCard({ row }: { row: RefusedField }) {
  return (
    <div style={{ ...REFUSED, marginBottom: 8 }}>
      <div style={{ fontWeight: 650, color: '#fde68a' }}>
        {row.field} — <span style={{ fontWeight: 500 }}>{row.status}</span>
      </div>
      <div style={{ marginTop: 4 }}>{row.detail}</div>
      <div style={{ marginTop: 6, color: 'var(--text2)' }}>
        Would come from: <b style={{ color: 'var(--text)' }}>{row.would_come_from}</b>
      </div>
      <div style={{ marginTop: 4, color: 'var(--text3)' }}>
        A dashboard mock-up showed <code style={{ color: 'var(--danger)' }}>{row.quoted_in_the_dashboard_mockup}</code> here.
        Nothing in this repository produces that number.
      </div>
    </div>
  );
}

function RefusalRow({ row }: { row: Refusal }) {
  const instead = row.what_is_recorded_instead;
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: '10px 12px', marginBottom: 8 }}>
      <div style={{ fontSize: 12.5, color: 'var(--danger)', fontWeight: 600 }}>✕ {row.shown_in_the_mockup}</div>
      <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.55, margin: '5px 0' }}>{row.why_refused}</div>
      {instead !== null && instead !== undefined && (
        <pre style={{
          ...MONO, margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          color: 'var(--success)', background: 'rgba(2,6,23,.5)', padding: '8px 10px', borderRadius: 8,
        }}>
          {JSON.stringify(instead, null, 1)}
        </pre>
      )}
    </div>
  );
}

function RatioTable({ title, values }: { title: string; values: Record<string, number> }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px', marginBottom: 5 }}>
        {title}
      </div>
      <table className="mini-table">
        <thead><tr><th>grid</th><th style={{ textAlign: 'right' }}>speed-up</th></tr></thead>
        <tbody>
          {Object.entries(values).map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td style={{ textAlign: 'right', color: v > 1 ? 'var(--success)' : 'var(--warning)' }}>{v.toFixed(2)}×</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TumorBoard({ onNavigate }: { onNavigate?: (id: TabId) => void }) {
  const { artifact, loading, error } = useArtifact<TumorBoardData>('tumorboard');

  if (error) {
    return (
      <Panel title="Tumour board">
        <div style={{ color: 'var(--warning)', fontSize: 13, lineHeight: 1.6 }}>⚠ {error}</div>
      </Panel>
    );
  }
  if (loading || !artifact) {
    return <Panel title="Tumour board"><div style={{ color: 'var(--text2)' }}>Loading recorded panels…</div></Panel>;
  }

  const d = artifact.data;
  const kinetics = d.kinetics_panel;
  const days = kinetics.resistant_fraction.policy ?? [];
  const curve = days.map((v: number, i: number) => ({
    day: i + 1,
    adaptive: v,
    mtd: kinetics.resistant_fraction.mtd?.[i] ?? null,
  }));
  const arms = d.dosing_panel.arms;
  const acc = d.audit_panel.acceleration;
  const two = arms.policy && arms.mtd ? [['adaptive', arms.policy], ['MTD control', arms.mtd]] as const : [];
  const burdenScale = Math.max(d.dosing_panel.day_365.burden_policy, d.dosing_panel.day_365.burden_mtd);

  return (
    <div>
      {/* ---------------------------------------------------------- panel 5 */}
      <Panel
        title="Audit — this repository's own checks, counted live"
        sub={`${d.audit_panel.no_regulatory_status}`}
        badge={<span className="badge-status live">● RECORDED STATE OF THE CHECKS</span>}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 10 }}>
          <div style={CARD}>
            <div style={{ fontSize: 10.5, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px' }}>claims passing</div>
            <div style={{ fontSize: 25, fontWeight: 650, color: 'var(--success)' }}>
              {d.audit_panel.claims.passed}/{d.audit_panel.claims.total}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>
              {d.audit_panel.claims.failed_ids.length
                ? `failing: ${d.audit_panel.claims.failed_ids.join(', ')}`
                : 'no failing check; the registry is the count'}
            </div>
          </div>
          <div style={CARD}>
            <div style={{ fontSize: 10.5, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px' }}>test functions</div>
            <div style={{ fontSize: 25, fontWeight: 650 }}>{d.audit_panel.test_functions.toLocaleString()}</div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>counted by {d.audit_panel.counted_by}, not remembered</div>
          </div>
          <div style={CARD}>
            <div style={{ fontSize: 10.5, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px' }}>kernel equality</div>
            <div style={{ fontSize: 25, fontWeight: 650, color: acc.bit_identity_holds.fields && acc.bit_identity_holds.trajectory ? 'var(--success)' : 'var(--danger)' }}>
              {acc.bit_identity_holds.fields && acc.bit_identity_holds.trajectory ? 'EXACT' : 'BROKEN'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>accelerated operators vs their reference, day by day over 365 days</div>
          </div>
          <div style={CARD}>
            <div style={{ fontSize: 10.5, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px' }}>engine source hash</div>
            <div style={{ ...MONO, fontSize: 12, wordBreak: 'break-all' }}>{d.audit_panel.engine_source_hash.combined_sha256.slice(0, 32)}…</div>
            <div style={{ fontSize: 11, color: 'var(--text2)' }}>{d.audit_panel.engine_source_hash.scope}</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 14, marginTop: 14 }}>
          <RatioTable title="Laplacian, CPU, this machine" values={acc.laplacian_ratio_by_grid} />
          <RatioTable title="Divergence operator, CPU" values={acc.divergence_ratio_by_grid} />
          <div>
            <div style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px', marginBottom: 5 }}>
              UQ throughput strategies
            </div>
            <table className="mini-table">
              <thead><tr><th>strategy</th><th style={{ textAlign: 'right' }}>vs serial</th></tr></thead>
              <tbody>
                <tr>
                  <td>batched tensor, per-replicate streams</td>
                  <td style={{ textAlign: 'right', color: 'var(--warning)' }}>{acc.batch_per_replicate_ratio.toFixed(2)}×</td>
                </tr>
                <tr>
                  <td>batched tensor, one stream draw</td>
                  <td style={{ textAlign: 'right', color: acc.batch_one_stream_ratio > 1 ? 'var(--text)' : 'var(--warning)' }}>
                    {acc.batch_one_stream_ratio.toFixed(2)}×
                  </td>
                </tr>
                {Object.entries(acc.process_ratio_by_workers).map(([w, r]) => (
                  <tr key={w}>
                    <td>{w} OS processes</td>
                    <td style={{ textAlign: 'right', color: 'var(--success)' }}>{r.toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 6, lineHeight: 1.5 }}>
              statistic: {acc.statistic} · measured {acc.measured_at}. Widening the arrays is not the
              answer on this engine; widening the processes is, and it cannot change a number because
              every worker runs the unmodified serial engine.
            </div>
          </div>
        </div>
      </Panel>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(330px, 1fr))', gap: 14 }}>
        {/* -------------------------------------------------------- panel 1 */}
        <Panel
          title="Patient & data on file"
          sub={`${d.case.geometry_note}. This card is a segmentation, not a clinical record.`}
          badge={<span className="badge-warn">MOLECULAR DATA ABSENT</span>}
        >
          {d.patient_panel.recorded.map((f) => (
            <Field key={f.field} label={f.field} value={String(f.value)} source={f.source} />
          ))}
          <div style={{ marginTop: 12, fontSize: 11.5, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.7px' }}>
            Fields a clinical dashboard leads with, and what this repository has for them
          </div>
          <div style={{ marginTop: 8 }}>
            {d.patient_panel.not_recorded.map((row) => <RefusedCard key={row.field} row={row} />)}
          </div>
        </Panel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
          {/* ---------------------------------------------------- panel 2 */}
          <Panel
            title="Volume inventory"
            sub={d.volume_panel.note}
            badge={<span className="badge-status live">● RECORDED FRAMES</span>}
          >
            <table className="mini-table">
              <thead>
                <tr><th>arm</th><th>frames</th><th>seed</th><th style={{ textAlign: 'right' }}>PFS</th><th>progressed</th></tr>
              </thead>
              <tbody>
                {Object.entries(d.volume_panel.arm_frames).map(([arm, a]) => (
                  <tr key={arm}>
                    <td>{arm === 'policy' ? 'adaptive' : 'MTD'}</td>
                    <td>{a.n_frames}</td>
                    <td>{a.seed}</td>
                    <td style={{ textAlign: 'right' }}>{a.pfs_day === null ? 'not reached' : `${a.pfs_day} d`}</td>
                    <td style={{ color: a.progressed ? 'var(--danger)' : 'var(--success)' }}>{a.progressed ? 'YES' : 'no'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11.5, color: 'var(--text2)', lineHeight: 1.55, marginTop: 10 }}>
              Grid {d.volume_panel.grid?.join('×')} at {d.volume_panel.spacing_mm} mm; frames on{' '}
              {d.volume_panel.frame_days?.join(', ') ?? 'n/a'}. The 3D renderer for these same frames is
              the 365-Day Evolution screen — this panel inventories them rather than drawing a second
              viewer from a different code path.
            </div>
            {onNavigate && (
              <button className="nav-btn" style={{ alignSelf: 'flex-start', marginTop: 10 }} onClick={() => onNavigate('evolution')}>
                open the recorded frames →
              </button>
            )}
          </Panel>

          {/* ---------------------------------------------------- panel 3 */}
          <Panel
            title="Clonal kinetics — the quantity the switch rule acts on"
            sub={kinetics.note}
          >
            <ResponsiveContainer width="100%" height={190}>
              <LineChart data={curve} margin={{ top: 6, right: 10, bottom: 2, left: 0 }}>
                <CartesianGrid stroke="rgba(51,65,85,.4)" strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 10 }} stroke="rgba(51,65,85,.6)"
                  ticks={[1, 92, 183, 274, 365]} tickFormatter={(v: number) => `d${v}`} />
                <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} stroke="rgba(51,65,85,.6)" width={34} />
                <Tooltip
                  contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }}
                  formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                  labelFormatter={(v: number) => `day ${v}`}
                />
                <ReferenceLine y={0.5} stroke="rgba(248,113,113,.7)" strokeDasharray="4 4"
                  label={{ value: 'resistance takes over', fill: '#f87171', fontSize: 10, position: 'insideTopRight' }} />
                <Line type="monotone" dataKey="mtd" name="MTD control" stroke="#f59e0b" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="adaptive" name="adaptive switch" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 12, color: 'var(--text2)', marginTop: 8 }}>
              Tipping day (first day R/(S+R) &gt; 0.5):{' '}
              <b style={{ color: 'var(--text)' }}>adaptive {kinetics.tipping_day.policy ?? 'never'}</b>,{' '}
              <b style={{ color: 'var(--text)' }}>MTD {kinetics.tipping_day.mtd ?? 'never'}</b>.
            </div>
          </Panel>
        </div>
      </div>

      {/* ---------------------------------------------------------- panel 4 */}
      <Panel
        title="Adaptive schedule vs MTD control — the paired trial"
        sub={`Same geometry, same seeds, one policy changed. Switch rule: ${d.dosing_panel.switch_rule}. ${d.dosing_panel.no_recurrence_date}`}
        badge={
          d.dosing_panel.pfs_extension_is_lower_bound
            ? <span className="badge-warn">PFS EXTENSION IS A LOWER BOUND</span>
            : <span className="badge-status live">● RECORDED ENGINE OUTCOME</span>
        }
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
          {two.map(([label, a]) => (
            <div key={label} style={CARD}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div style={{ fontSize: 13.5, fontWeight: 650 }}>{label}</div>
                <code style={{ ...MONO, color: 'var(--text3)' }}>{a.label}</code>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 10 }}>
                <div>
                  <div style={{ fontSize: 10.5, color: 'var(--text3)' }}>MEDIAN PFS</div>
                  <div style={{ fontSize: 22, fontWeight: 650, fontVariantNumeric: 'tabular-nums' }}>{a.median_pfs_days.toFixed(0)} d</div>
                </div>
                <div>
                  <div style={{ fontSize: 10.5, color: 'var(--text3)' }}>PROGRESSED</div>
                  <div style={{ fontSize: 22, fontWeight: 650, color: a.n_progressed ? 'var(--danger)' : 'var(--success)' }}>
                    {a.n_progressed}/{a.n_replicates}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 10.5, color: 'var(--text3)' }}>Σ DOSE A / B</div>
                  <div style={{ ...MONO }}>{a.cumulative_drug_A.toFixed(0)} / {a.cumulative_drug_B.toFixed(0)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 10.5, color: 'var(--text3)' }}>TOXICITY PROXY</div>
                  <div style={{ ...MONO }}>{a.median_toxicity.toFixed(2)}</div>
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 10.5, color: 'var(--text3)', marginBottom: 3 }}>
                  DAY-365 BURDEN (cell-density units)
                </div>
                <div style={{ height: 8, background: 'rgba(2,6,23,.6)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    width: `${(100 * (label === 'adaptive' ? d.dosing_panel.day_365.burden_policy : d.dosing_panel.day_365.burden_mtd)) / burdenScale}%`,
                    height: '100%',
                    background: label === 'adaptive' ? 'var(--accent)' : 'var(--danger)',
                  }} />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>
                  {label === 'adaptive' ? d.dosing_panel.day_365.burden_policy.toFixed(2) : d.dosing_panel.day_365.burden_mtd.toFixed(2)}
                </div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text2)', marginTop: 10, lineHeight: 1.55 }}>
          Day-365 burden reduction vs MTD: <b style={{ color: 'var(--text)' }}>
            {(100 * d.dosing_panel.day_365.burden_reduction_vs_mtd).toFixed(2)}%
          </b>. {d.dosing_panel.day_365.toxicty_note}. Measured under the engine's declared constants,
          on one real case — not a patient prediction, and not a trial result.
        </div>
      </Panel>

      {/* ------------------------------------------------------- refusals */}
      <Panel
        title="What this dashboard was asked to show, and what it shows instead"
        sub="Each row is a figure a mock-up displayed. The reason is the field's, not a disclaimer's: the number has no producer in this repository, so it cannot be rendered honestly at any precision."
        badge={<span className="badge-fail">REFUSED</span>}
      >
        {d.refusals.map((row) => <RefusalRow key={row.shown_in_the_mockup} row={row} />)}
        <div style={{ fontSize: 11.5, color: 'var(--text2)', marginTop: 8, lineHeight: 1.6 }}>
          Not measured here, and therefore absent from every panel:{' '}
          {acc.not_measured.join(' · ')}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 10 }}>
          {artifactStamp(artifact)} · artifact generated {d.generated_utc} · {d.model}
        </div>
      </Panel>
    </div>
  );
}
