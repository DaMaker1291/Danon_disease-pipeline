/**
 * DosingConsole — drag a real calendar, get a real engine run.
 *
 * WHAT THIS IS NOT. It is not a mock-up console with plausible-looking curves.
 * Every number on screen is an outcome of POSTing the calendar to
 * /api/dosing/plan, which runs `danon.evosteer.TumorEvolutionSimulator` on
 * exactly the schedule drawn here. The engine's recorded dose series IS the
 * calendar (pinned by tests/test_dosing.py::TestIdentityWithEngine).
 *
 * WHAT IS ASSUMED. The neutrophil safety screen uses DECLARED parameters, not
 * measured ones, and the panel says so in those words. It is not an NCCN
 * threshold and confers no clinical safety. The card that carries the refusal
 * is rendered next to the number it qualifies, never in a footnote.
 *
 * WHEN THERE IS NO BACKEND the console says so. It never renders a
 * representative curve, because a fabricated result on this screen would be
 * indistinguishable from a real one — which is the one thing a dosing console
 * must never do.
 */
import React, { useCallback, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

interface ScheduleScore {
  name: string;
  schedule: { blocks: unknown[]; totalMgM2: number; onFraction: number; text: string };
  seeds: number[];
  nSeeds: number;
  medianAubc: number;
  iqrAubc: [number, number];
  medianDominanceDay: number;
  medianPfsDays: number;
  medianToxicity: number;
  nProgressed: number;
  feasible: boolean;
  ancNadir: number;
  engineRuns: number;
}

interface DosingResponse {
  mode: 'single' | 'search';
  geometry: { grid: number[]; tumourVoxels: number; phantomSeed: number; isPatientImaging: boolean };
  myelosuppressionModel: {
    ancBaseline: number;
    ancFloor: number;
    ancFloorSource: string;
    suppressionBudgetMgM2: number;
    budgetDefinition: string;
    kernel: {
      family: string; logMu: number; logSd: number; unitArea: boolean; fittedTo: string;
    };
    labelQuotes: Record<string, { quote: string; section: string }>;
    secondDrugPotency: string;
    anchors: Record<string, number | boolean | null>;
    status: string;
    isLabelThreshold: boolean;
    measuredInThisRepository: boolean;
    isClinicalDecisionSupport: boolean;
  };
  provenance: Record<string, string>;
  wallSeconds: number;
  // single
  calendar?: ScheduleScore;
  safety?: { feasible: boolean; ancNadir: number; floor: number; violations: string[] };
  mtdComparison?: ScheduleScore;
  beatsMtd?: boolean;
  notes?: string[];
  // search
  candidatesEvaluated?: number;
  candidatesInfeasible?: number;
  engineRuns?: number;
  beatsBaselines?: boolean;
  best?: ScheduleScore | null;
  scored?: ScheduleScore[];
  baselines?: Record<string, ScheduleScore>;
}

interface Props { }

const MONO: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', monospace", fontSize: 11.5, fontVariantNumeric: 'tabular-nums',
};

function Field({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div style={{ padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ fontSize: 10.5, letterSpacing: '.7px', color: 'var(--text3)', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, margin: '2px 0', color: 'var(--text)' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.45 }}>{sub}</div>}
    </div>
  );
}

function Refusal({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      border: '1px solid rgba(245,158,11,.42)', background: 'rgba(245,158,11,.07)',
      color: '#fcd34d', borderRadius: 10, padding: '10px 12px', fontSize: 12, lineHeight: 1.55,
    }}>
      {children}
    </div>
  );
}

/** One slider row with its live read-out. */
function Slider({
  label, value, min, max, step, suffix, onChange, format,
}: {
  label: string; value: number; min: number; max: number; step: number; suffix?: string;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span style={{ fontSize: 11, letterSpacing: '.6px', color: 'var(--text3)', textTransform: 'uppercase' }}>{label}</span>
        <span style={MONO}>{format ? format(value) : value}{suffix ?? ''}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%' }}
      />
    </div>
  );
}

export default function DosingConsole(_props: Props) {
  const [doseMgM2, setDoseMgM2] = useState(150);
  const [onDays, setOnDays] = useState(5);
  const [cycleDays, setCycleDays] = useState(26);
  const [nSeeds, setNSeeds] = useState(3);
  const [mode, setMode] = useState<'single' | 'search'>('single');

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DosingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const invalid = onDays > cycleDays;

  const run = useCallback(async () => {
    setLoading(true); setData(null); setError(null);
    try {
      const res = await fetch('/api/dosing/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode, dose_mg_m2: doseMgM2, on_days: onDays, cycle_days: cycleDays,
          days: 365, n_seeds: nSeeds, dose_level: 1.0,
        }),
      });
      if (!res.ok) throw new Error((await res.text()).slice(0, 400) || `HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: any) {
      setError(e?.message || 'the dosing endpoint did not answer');
    } finally {
      setLoading(false);
    }
  }, [mode, doseMgM2, onDays, cycleDays, nSeeds]);

  const cal = data?.calendar;
  const mtd = data?.mtdComparison;
  const searchRows = (data?.scored ?? []).map((s, i) => ({
    name: s.name.length > 26 ? s.name.slice(0, 26) + '…' : s.name,
    aubc: Number(s.medianAubc.toFixed(4)),
    feasible: s.feasible,
    i: i,
  }));

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 340px) minmax(0, 1fr)', gap: 18, alignItems: 'start' }}>
      {/* ---------------- controls ---------------- */}
      <div className="panel">
        <div className="panel-title" style={{ marginTop: 0 }}>Dosing calendar</div>
        <div style={{ color: 'var(--text2)', fontSize: 11.5, lineHeight: 1.55, marginBottom: 14 }}>
          Set the calendar, press run. The engine executes exactly this schedule.
        </div>

        <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
          {(['single', 'search'] as const).map(m => (
            <button
              key={m}
              className={`btn ${mode === m ? 'btn-run' : ''}`}
              style={{ flex: 1, fontSize: 11.5, padding: '6px 8px' }}
              onClick={() => { setMode(m); setData(null); setError(null); }}
            >
              {m === 'single' ? 'Run this calendar' : 'Search the grid'}
            </button>
          ))}
        </div>

        <Slider label="dose" value={doseMgM2} min={0} max={300} step={10}
          suffix=" mg/m²/d" onChange={setDoseMgM2} />
        <Slider label="days on" value={onDays} min={1} max={30} step={1}
          suffix=" d" onChange={setOnDays} />
        <Slider label="cycle" value={cycleDays} min={2} max={90} step={1}
          suffix=" d" onChange={setCycleDays} />
        <Slider label="seeds" value={nSeeds} min={1} max={12} step={1}
          suffix=" replicates" onChange={setNSeeds} />

        {invalid && (
          <Refusal>
            A holiday longer than the block is not a schedule — {onDays} days on with a{' '}
            {cycleDays}-day cycle has no off period. The server rejects this too.
          </Refusal>
        )}

        <button
          className="btn btn-run"
          style={{ width: '100%', marginTop: 12, padding: '9px 12px' }}
          disabled={loading || invalid}
          onClick={run}
        >
          {loading ? 'running the engine…' : mode === 'single' ? 'Run on the engine' : 'Search the grid'}
        </button>
        {data && (
          <div style={{ ...MONO, color: 'var(--text2)', marginTop: 8 }}>
            {data.wallSeconds}s wall · {data.geometry.tumourVoxels} voxels · {data.geometry.grid.join('×')}
          </div>
        )}
      </div>

      {/* ---------------- results ---------------- */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {error && (
          <Refusal>
            The endpoint did not answer: {error}. No curve is drawn, because there is
            nothing to draw. Start the API server and press run again.
          </Refusal>
        )}

        {!data && !error && !loading && (
          <div className="panel" style={{ color: 'var(--text2)', fontSize: 12, lineHeight: 1.6 }}>
            Nothing has been run yet. Every figure on this page is produced by the
            engine on request — this console will not show a representative curve.
          </div>
        )}

        {data && (
          <>
            {/* --- what is measured vs assumed --- */}
            <div className="panel">
              <div className="panel-title" style={{ marginTop: 0 }}>Provenance</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 8 }}>
                {Object.entries(data.provenance).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 11.5, lineHeight: 1.5 }}>
                    <span style={{ color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '.6px', fontSize: 10 }}>
                      {k.replace(/_/g, ' ')}
                    </span>
                    <div style={{
                      color: v.startsWith('NOT CLAIMED') || v.startsWith('DECLARED') ? '#fcd34d' : 'var(--text2)',
                    }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* --- single calendar --- */}
            {mode === 'single' && cal && (
              <>
                <div className="panel">
                  <div className="panel-title" style={{ marginTop: 0 }}>
                    {cal.schedule.text}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginTop: 12 }}>
                    <Field label="median AUBC" value={cal.medianAubc.toFixed(4)}
                      sub="area under burden ÷ day-0 burden; lower is better" />
                    <Field label="MTD AUBC" value={mtd ? mtd.medianAubc.toFixed(4) : '—'}
                      sub="continuous dosing, same seeds" />
                    <Field label="beats MTD?" value={data.beatsMtd ? 'yes' : 'no'}
                      sub={`measured on ${cal.nSeeds} replicates`} />
                    <Field label="resistance dominance" value={`day ${Math.round(cal.medianDominanceDay)}`}
                      sub="median day R fraction passes 0.5" />
                    <Field label="median PFS" value={`${cal.medianPfsDays.toFixed(0)} d`}
                      sub={`${cal.nProgressed} of ${cal.nSeeds} progressed`} />
                    <Field label="cumulative dose" value={`${cal.schedule.totalMgM2.toFixed(0)} mg/m²`}
                      sub={`on ${(cal.schedule.onFraction * 100).toFixed(0)}% of days`} />
                  </div>
                  {cal.medianAubc < (mtd?.medianAubc ?? Infinity) ? (
                    <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text2)' }}>
                      Interquartile AUBC {cal.iqrAubc[0].toFixed(4)}–{cal.iqrAubc[1].toFixed(4)}
                      {' '}across {cal.nSeeds} seeds ({cal.engineRuns} engine runs).
                    </div>
                  ) : (
                    <div style={{ marginTop: 12 }}>
                      <Refusal>
                        This calendar did <strong>not</strong> beat continuous MTD
                        (AUBC {cal.medianAubc.toFixed(4)} vs {mtd?.medianAubc.toFixed(4)})
                        on {cal.nSeeds} seeds. Shown as measured. Interquartile AUBC{' '}
                        {cal.iqrAubc[0].toFixed(4)}–{cal.iqrAubc[1].toFixed(4)}.
                      </Refusal>
                    </div>
                  )}
                </div>

                {/* --- the safety screen, with its disclaimer attached --- */}
                <div className="panel">
                  <div className="panel-title" style={{ marginTop: 0 }}>Safety screen</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginTop: 12 }}>
                    <Field label="modelled ANC nadir" value={`${cal.ancNadir.toFixed(2)} ×10⁹/L`}
                      sub={`screened against a ${data.myelosuppressionModel.ancFloor.toFixed(2)} floor`} />
                    <Field label="passes screen?" value={cal.feasible ? 'yes' : 'no'}
                      sub="anchored to the FDA label, still not a clinical rule" />
                    <Field
                      label="marrow budget"
                      value={`${data.myelosuppressionModel.suppressionBudgetMgM2.toFixed(1)} mg/m²`}
                      sub="reservoir of the label's own 150 ×5 q28d regimen"
                    />
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <Refusal>
                      The floor is the <strong>label's own quoted threshold</strong> —
                      “{data.myelosuppressionModel.labelQuotes.withhold_threshold?.quote}”
                      ({data.myelosuppressionModel.labelQuotes.withhold_threshold?.section}) — and
                      the marrow budget is <em>defined</em> as the burden produced by the
                      label's own regimen, so the screen's boundary lands on a real
                      protocol. What is still a modelling choice: the kernel shape
                      (μ={data.myelosuppressionModel.kernel.logMu},
                      σ={data.myelosuppressionModel.kernel.logSd}), fitted to{' '}
                      {data.myelosuppressionModel.kernel.fittedTo}; and drug B's marrow
                      potency, which is assumed and swept rather than measured.
                      Measured in this repository:{' '}
                      {String(data.myelosuppressionModel.measuredInThisRepository)}. Clinical
                      decision support:{' '}
                      {String(data.myelosuppressionModel.isClinicalDecisionSupport)}. A
                      schedule that passes here has not been shown to be safe for anyone.
                    </Refusal>
                  </div>
                  {(data.notes ?? []).map((n, i) => (
                    <div key={i} style={{ ...MONO, color: '#fcd34d', marginTop: 8 }}>{n}</div>
                  ))}
                </div>
              </>
            )}

            {/* --- search --- */}
            {mode === 'search' && (
              <div className="panel">
                <div className="panel-title" style={{ marginTop: 0 }}>Grid search</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginTop: 12 }}>
                  <Field label="candidates evaluated" value={data.candidatesEvaluated ?? 0}
                    sub={`${data.candidatesInfeasible ?? 0} screened out by the safety model`} />
                  <Field label="engine runs" value={data.engineRuns ?? 0}
                    sub="two-stage: screen, then confirm" />
                  <Field label="beat the baselines?" value={data.beatsBaselines ? 'yes' : 'no'}
                    sub="continuous MTD and the widest grid step" />
                </div>

                {data.beatsBaselines && data.best ? (
                  <div style={{ marginTop: 12, fontSize: 12.5, color: 'var(--text)' }}>
                    Best confirmed calendar: <strong>{data.best.name}</strong>, AUBC{' '}
                    {data.best.medianAubc.toFixed(4)} on {data.best.nSeeds} seeds.
                  </div>
                ) : (
                  <div style={{ marginTop: 12 }}>
                    <Refusal>
                      <strong>No calendar in the grid beat the baselines.</strong>{' '}
                      The search is reported as a null result rather than returning the
                      best cell it happened to evaluate. See the notes in the artifact.
                    </Refusal>
                  </div>
                )}

                {searchRows.length > 0 && (
                  <div style={{ height: 260, marginTop: 16 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={searchRows} margin={{ top: 8, right: 12, bottom: 60, left: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-35} textAnchor="end"
                          interval={0} stroke="var(--text3)" />
                        <YAxis tick={{ fontSize: 10 }} stroke="var(--text3)" />
                        <Tooltip />
                        <Bar dataKey="aubc" radius={[3, 3, 0, 0]}>
                          {searchRows.map((r) => (
                            <Cell key={r.i} fill={r.i === 0 && data.beatsBaselines ? '#22c55e' : '#64748b'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
                <div style={{ ...MONO, color: 'var(--text3)', marginTop: 8 }}>
                  median AUBC per confirmed calendar · lower is better
                </div>
              </div>
            )}

            <Refusal>
              Geometry is a synthetic phantom (seed {data.geometry.phantomSeed}), not
              patient imaging: <strong>{String(data.geometry.isPatientImaging)}</strong>.
              This console ranks schedules under assumed biology. It does not predict
              what will happen to a person.
            </Refusal>
          </>
        )}
      </div>
    </div>
  );
}