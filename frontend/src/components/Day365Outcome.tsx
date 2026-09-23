import React, { useEffect, useMemo, useState } from 'react';
import {
  Area, AreaChart, CartesianGrid, Legend, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { loadOutcomeCards } from '../utils/evotwin';
import type { OutcomeCard, OutcomeCardsPayload } from '../types/evotwin';

/* ------------------------------------------------------------------ format */

function fmt(x: number | null | undefined, d = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  if (Math.abs(x) >= 1e5 || (Math.abs(x) > 0 && Math.abs(x) < 1e-3)) return x.toExponential(2);
  return x.toFixed(d);
}
function num(x: number | null | undefined, d = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return x.toFixed(d).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
function pct(x: number | null | undefined, d = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(d)}%`;
}

const TONE: Record<string, string> = {
  good: 'var(--success)',
  bad: 'var(--danger)',
  warn: 'var(--warning)',
  acc: 'var(--accent)',
  acc2: 'var(--accent2)',
};

function Kpi({ label, value, meta, tone }: { label: string; value: string; meta: string; tone?: string }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        background: 'linear-gradient(160deg, rgba(59,130,246,0.10), rgba(17,24,39,0.55))',
        borderRadius: 'var(--radius)',
        padding: '13px 15px',
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 10.5, letterSpacing: '.9px', color: 'var(--text3)', textTransform: 'uppercase' }}>{label}</div>
      <div
        style={{
          fontSize: 26, fontWeight: 650, letterSpacing: '-0.5px', margin: '5px 0 3px',
          fontVariantNumeric: 'tabular-nums', color: tone ? TONE[tone] : 'var(--text)',
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.45 }}>{meta}</div>
    </div>
  );
}

function Panel({ title, sub, children, id }: { title: string; sub?: string; children: React.ReactNode; id?: string }) {
  return (
    <div className="panel" id={id} style={{ marginBottom: 14 }}>
      <div className="panel-title">{title}</div>
      {sub && <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '4px 0 12px', lineHeight: 1.5 }}>{sub}</div>}
      {children}
    </div>
  );
}

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="mini-table" style={{ width: '100%', minWidth: 520, fontSize: 12.5, fontVariantNumeric: 'tabular-nums' }}>
        <thead>
          <tr>{head.map((h, i) => (
            <th key={h} style={{ textAlign: i === 0 ? 'left' : 'right', color: 'var(--text3)', fontWeight: 600, fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '.7px', padding: '6px 8px' }}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td key={ci} style={{ textAlign: ci === 0 ? 'left' : 'right', padding: '6px 8px', borderTop: '1px solid var(--border)', color: ci === 0 ? 'var(--text2)' : 'var(--text)' }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------- charts */

function BurdenChart({ card, logScale }: { card: OutcomeCard; logScale: boolean }) {
  const p = card.arms.policy.daily;
  const m = card.arms.mtd.daily;
  const data = p.burden_total.map((v, i) => ({
    day: i + 1,
    policy: v,
    mtd: m.burden_total[i],
  }));
  const threshold = (p.burden_total[0] || 1) * (1 + 0.25);
  return (
    <ResponsiveContainer width="100%" height={248}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="rgba(51,65,85,.4)" strokeDasharray="3 3" />
        <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)"
          ticks={[1, 92, 183, 274, 365]} tickFormatter={(d: number) => `d${d}`} />
        <YAxis scale={logScale ? 'log' : 'linear'} domain={logScale ? [1e-19, 'auto'] : [0, 'auto']}
          allowDataOverflow={logScale} tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" width={58} />
        <Tooltip
          contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }}
          formatter={(v: number) => fmt(v, 3)}
          labelFormatter={(d: number) => `day ${d}`}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text2)' }} />
        <ReferenceLine y={threshold} stroke="rgba(148,163,184,.7)" strokeDasharray="5 4"
          label={{ value: 'RANO threshold', fill: '#94a3b8', fontSize: 10, position: 'insideTopRight' }} />
        <Line type="monotone" dataKey="mtd" name="MTD control" stroke="#ef4444" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="policy" name="adaptive switch" stroke="#22c55e" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ClonalChart({ card }: { card: OutcomeCard }) {
  const p = card.arms.policy.daily;
  const m = card.arms.mtd.daily;
  const data = p.resistant_fraction.map((v, i) => ({
    day: i + 1,
    policy: v,
    mtd: m.resistant_fraction[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={248}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="rgba(51,65,85,.4)" strokeDasharray="3 3" />
        <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)"
          ticks={[1, 92, 183, 274, 365]} tickFormatter={(d: number) => `d${d}`} />
        <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} stroke="rgba(51,65,85,.6)" width={44} />
        <Tooltip
          contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }}
          formatter={(v: number) => pct(v)}
          labelFormatter={(d: number) => `day ${d}`}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text2)' }} />
        <ReferenceLine y={0.15} stroke="rgba(139,92,246,.6)" strokeDasharray="4 4"
          label={{ value: 'switch ON 15%', fill: '#8b5cf6', fontSize: 10, position: 'insideTopRight' }} />
        <Line type="monotone" dataKey="mtd" name="MTD control" stroke="#f59e0b" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="policy" name="adaptive switch" stroke="#8b5cf6" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function DoseChart({ card }: { card: OutcomeCard }) {
  const p = card.arms.policy.daily;
  const data = p.dose_A.map((v, i) => ({ day: i + 1, drugA: v, drugB: p.dose_B[i] }));
  return (
    <ResponsiveContainer width="100%" height={132}>
      <AreaChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="rgba(51,65,85,.35)" strokeDasharray="3 3" />
        <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 10 }} stroke="rgba(51,65,85,.6)"
          ticks={[1, 92, 183, 274, 365]} tickFormatter={(d: number) => `d${d}`} />
        <YAxis domain={[0, 1]} ticks={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} stroke="rgba(51,65,85,.6)" width={30} />
        <Tooltip
          contentStyle={{ background: 'rgba(15,23,42,.95)', border: '1px solid var(--border)', borderRadius: 10, fontSize: 12 }}
          formatter={(v: number, n: string) => [v > 0.5 ? 'full dose' : 'withheld', n]}
          labelFormatter={(d: number) => `day ${d}`}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text2)' }} />
        <Area type="stepAfter" dataKey="drugA" name="Drug A" stackId="1" stroke="#3b82f6" fill="rgba(59,130,246,.45)" />
        <Area type="stepAfter" dataKey="drugB" name="Drug B" stackId="1" stroke="#f59e0b" fill="rgba(245,158,11,.45)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* -------------------------------------------------------------------- shell */

export default function Day365Outcome() {
  const [payload, setPayload] = useState<OutcomeCardsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [logScale, setLogScale] = useState(false);

  useEffect(() => {
    loadOutcomeCards().then(setPayload).catch((e: Error) => setError(e.message));
  }, []);

  const card = payload?.cases[Math.min(idx, (payload?.cases.length ?? 1) - 1)];

  const replicateRows = useMemo(() => {
    if (!card) return [];
    const rows: (string | number)[][] = [];
    (['policy', 'mtd'] as const).forEach((arm) => {
      const label = arm === 'policy' ? 'adaptive' : 'MTD';
      card.arms[arm].replicate_day_readouts.forEach((r) => {
        rows.push([
          `${label} · seed ${r.seed}`,
          r.pfs_day === null ? 'not reached' : `${num(r.pfs_day, 0)} d`,
          r.progressed ? 'YES' : 'no',
          num(r.day.burden_total),
          pct(r.day.resistant_fraction),
          num(r.day.dose_A_cumulative),
          num(r.day.dose_B_cumulative),
          fmt(r.toxicity, 1),
        ]);
      });
    });
    return rows;
  }, [card]);

  if (error) {
    return (
      <Panel title="365-Day Predicted Outcome">
        <div style={{ color: 'var(--warning)', fontSize: 13, lineHeight: 1.6 }}>⚠ {error}</div>
      </Panel>
    );
  }
  if (!payload || !card) {
    return <Panel title="365-Day Predicted Outcome"><div style={{ color: 'var(--text2)' }}>Loading outcome cards…</div></Panel>;
  }

  const h = card.day_365;
  const p = card.arms.policy.daily;
  const m = card.arms.mtd.daily;
  const i365 = Math.min(365, p.burden_total.length) - 1;

  const crossCheck: (string | number)[][] = [
    ['Tumour burden (cell density)', fmt(h.burden_policy, 3), fmt(h.burden_mtd, 3)],
    ['Resistant clone burden', fmt(p.burden_resistant[i365], 3), fmt(m.burden_resistant[i365], 3)],
    ['Sensitive clone burden', fmt(p.burden_sensitive[i365], 3), fmt(m.burden_sensitive[i365], 3)],
    ['Resistant fraction', fmt(h.resistant_fraction_policy, 3), fmt(h.resistant_fraction_mtd, 3)],
    ['Cumulative dose, Drug A', fmt(p.dose_A.reduce((a, b) => a + b, 0), 1), fmt(m.dose_A.reduce((a, b) => a + b, 0), 1)],
    ['Cumulative dose, Drug B', fmt(p.dose_B.reduce((a, b) => a + b, 0), 1), '0.0'],
    ['Median PFS (days)', fmt(h.median_pfs_policy, 1), fmt(h.median_pfs_mtd, 1)],
    ['Median toxicity proxy', fmt(card.arms.policy.median_toxicity, 2), fmt(card.arms.mtd.median_toxicity, 2)],
  ];

  return (
    <div>
      <Panel
        title="🧠 365-Day Predicted Outcome — exact engine state"
        sub={`Adaptive collateral switch vs paired MTD control on the same real geometry and seeds. ${payload.model}. Generated ${payload.generated_utc}.`}
      >
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <select
            value={idx}
            onChange={(e) => setIdx(Number(e.target.value))}
            style={{ background: 'var(--surface2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 10, padding: '8px 10px', fontSize: 13, minWidth: 280 }}
          >
            {payload.cases.map((c, i) => (
              <option key={`${c.case_id}-${c.patient_id}`} value={i}>
                {c.case_id} × {c.patient_id} (r0 = {c.intake.r0.toFixed(3)})
              </option>
            ))}
          </select>
          <button className={`nav-btn ${logScale ? 'active' : ''}`} onClick={() => setLogScale((v) => !v)} style={{ padding: '7px 12px' }}>
            log-scale burden
          </button>
          <span className="badge-status live">● ENGINE OUTPUT</span>
          <span className="badge-warn">CONSTANTS ASSUMED</span>
          <span className="badge">NOT A PATIENT PREDICTION</span>
        </div>

        <div style={{ color: 'var(--text2)', fontSize: 12, lineHeight: 1.6, marginBottom: 12 }}>
          {card.geometry.note} — engine grid {card.geometry.grid.join('×')},{' '}
          {num(card.geometry.tumour_voxels)} tumour voxels at {card.geometry.voxel_mm} mm.
          <br />
          Intake r0 = <b style={{ color: 'var(--text)' }}>{card.intake.r0.toFixed(4)}</b>: {card.intake.r0_source}
          <br />
          Replicates: seeds {card.seeds.join(', ')} · horizon {card.horizon_days} d · switch rule: {payload.switch_rule}
        </div>

        {h.clearance_note && (
          <div style={{ border: '1px solid rgba(245,158,11,.4)', background: 'rgba(245,158,11,.08)', color: '#fcd34d', borderRadius: 10, padding: '10px 13px', fontSize: 12.5, lineHeight: 1.55, marginBottom: 10 }}>
            ⚠ {h.clearance_note}
          </div>
        )}
        {h.pfs_extension_is_lower_bound && (
          <div style={{ border: '1px solid rgba(59,130,246,.35)', background: 'rgba(59,130,246,.08)', color: '#bfdbfe', borderRadius: 10, padding: '10px 13px', fontSize: 12.5, lineHeight: 1.55, marginBottom: 10 }}>
            The median-PFS extension is a <b>lower bound</b>: no replicate in the adaptive arm progressed inside the {h.day}-day
            horizon, so the true extension cannot be measured here — only bounded below.
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(165px, 1fr))', gap: 12 }}>
          <Kpi label="Progression-free at day 365" value={h.progression_free_policy ? 'YES' : 'NO'}
            meta={`adaptive switch · ${h.n_progressed_policy}/${h.n_replicates} replicates progressed`}
            tone={h.progression_free_policy ? 'good' : 'bad'} />
          <Kpi label="MTD control at day 365" value={h.progression_free_mtd ? 'yes' : 'NO'}
            meta={`${h.n_progressed_mtd}/${h.n_replicates} replicates progressed`} tone={h.progression_free_mtd ? undefined : 'bad'} />
          <Kpi label="Burden reduction vs MTD" value={pct(h.burden_reduction_vs_mtd)}
            meta={`policy ${fmt(h.burden_policy)} vs MTD ${num(h.burden_mtd)} cell-density units`} tone="acc" />
          <Kpi label="Median PFS" value={`${num(h.median_pfs_policy)} d`}
            meta={h.pfs_extension_label
              ? `vs MTD ${num(h.median_pfs_mtd)} d · ${h.pfs_extension_label}`
              : `vs MTD ${num(h.median_pfs_mtd)} d (extension not estimable)`}
            tone="acc2" />
          <Kpi label="Toxicity reduction" value={pct(h.toxicity_reduction)}
            meta="cumulative weighted dose proxy vs MTD" tone="good" />
          <Kpi label="Resistant fraction at 365" value={pct(h.resistant_fraction_policy)}
            meta={h.clearance_note ? 'not interpretable at this burden' : `MTD control: ${pct(h.resistant_fraction_mtd)}`}
            tone={h.clearance_note || h.resistant_fraction_policy > 0.5 ? 'warn' : 'good'} />
        </div>
      </Panel>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 14 }}>
        <Panel title="Tumour burden over 365 days"
          sub="Total cell density, reference replicate. The dashed line is the simulator's RANO progression threshold (burden +25%).">
          <BurdenChart card={card} logScale={logScale} />
        </Panel>
        <Panel title="Clonal composition"
          sub="Resistant fraction R/(S+R) — the quantity the switch rule acts on, and the one that is NOT clinically observable in real time.">
          <ClonalChart card={card} />
        </Panel>
      </div>

      <Panel title="Executed dosing" sub="Adaptive arm: Drug A → Drug B on the resistant-fraction thresholds. Control arm: continuous Drug A at MTD.">
        <DoseChart card={card} />
      </Panel>

      <Panel title="Exact state at day 365, per replicate"
        sub="Every value is read from the engine at day 365 of that replicate — no interpolation, no smoothing.">
        <DataTable
          head={['arm · replicate', 'PFS', 'progressed', 'burden @365', 'R fraction', 'Σ dose A', 'Σ dose B', 'toxicity']}
          rows={replicateRows}
        />
      </Panel>

      <Panel title="Cross-check — median across replicates" sub="Same run, same seeds, both arms, day 365.">
        <DataTable head={['quantity at day 365', 'adaptive switch', 'MTD control']} rows={crossCheck} />
      </Panel>

      <Panel title="What this does not show" sub="Carried from the engine's and the data-ingestion's own limitation records.">
        <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text2)', fontSize: 12, lineHeight: 1.7 }}>
          {card.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
        <div style={{ marginTop: 12, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.7 }}>
          <b style={{ color: 'var(--warning)' }}>Engine constants: ASSUMED.</b> {String(payload.engine_constants.provenance)}
          <br />
          Data provenance: {Object.entries(card.data_provenance).map(([k, v]) => `${k}=${v}`).join(' · ')}
        </div>
      </Panel>
    </div>
  );
}
