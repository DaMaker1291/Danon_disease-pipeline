import React, { useEffect, useMemo, useState } from 'react';
import { loadOutcomeCards } from '../utils/evotwin';
import { artifactStamp, useArtifact } from '../utils/artifacts';
import {
  INSUFFICIENT_REPLICATES_FLAG,
  MIN_PAIRS_FOR_AN_INTERVAL,
  NESTED_REPLICATES_FLAG,
  NOISE_FLOOR_FLAG,
  TRIAL_DESIGN_LIVE_SEED,
  computeLiveComparison,
  nestingCounts,
  pair,
  reconcileComparison,
} from '../utils/trialDesign';
import { DataTable, Panel, num } from './ui';
import type { OutcomeCardsPayload } from '../types/evotwin';
import type { IntervalFan, NoiseFloorState, TrialDesignData } from '../types/artifacts';

/*
 * The trial-design screen: rendering only.
 *
 * Every rule that decides what the numbers ARE lives in
 * `frontend/src/utils/trialDesign.ts`, and every recorded number comes from the
 * `trial_design` artifact. This file decides what the screen looks like and
 * nothing else -- it does not resample, aggregate or re-derive anything, so a
 * change to the noise-floor policy lands in one place instead of two.
 *
 * The comparison is computed live in the browser from the recorded outcome
 * cards. The rule-out panel is read from the record, because the ANC screen is a
 * Python model and re-implementing it here would create a second definition of
 * the same number. A screen that presents a rule-out as a licence decision is the
 * failure this whole repository refuses, so the rule-out is phrased against the
 * model's own declared screen and never as a regulatory boundary.
 */

const BRIEF_CALENDAR = 'brief_arithmetic_150x5_of_19';

/** The resolution state as a chip: the flag text, or a plain "resolves". */
function StateChip({ floor }: { floor: NoiseFloorState }) {
  if (!floor.flag) {
    return <span className="badge" style={{ color: 'var(--success)' }}>resolves</span>;
  }
  const overlap = floor.status === 'within_noise_floor';
  return (
    <span
      className={overlap ? 'badge-warn' : 'badge'}
      style={{
        fontSize: 10,
        whiteSpace: 'normal',
        textAlign: 'right',
        display: 'inline-block',
        lineHeight: 1.3,
        maxWidth: 260,
        color: overlap ? undefined : 'var(--warning)',
      }}
      title={floor.reason}
    >
      {floor.flag}
    </span>
  );
}

/** One interval bar on a shared domain, with the red zero line as the reference. */
function FanRow({
  label, ci, domain,
}: { label: string; ci: [number, number] | null; domain: [number, number] }) {
  const width = domain[1] - domain[0] || 1;
  const zeroPct = ((0 - domain[0]) / width) * 100;
  const crosses = ci !== null && ci[0] <= 0 && ci[1] >= 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
      <span style={{ width: 46, fontSize: 11, color: 'var(--text3)', fontVariantNumeric: 'tabular-nums' }}>
        {label}
      </span>
      <div style={{ position: 'relative', flex: 1, height: 14, background: 'rgba(148,163,184,.10)', borderRadius: 4 }}>
        <div
          style={{
            position: 'absolute',
            left: `${Math.max(0, Math.min(100, zeroPct))}%`,
            top: -2,
            bottom: -2,
            width: 1,
            background: 'rgba(204,95,87,.85)',
          }}
        />
        {ci && (
          <div
            style={{
              position: 'absolute',
              top: 2,
              bottom: 2,
              borderRadius: 3,
              left: `${Math.max(0, ((ci[0] - domain[0]) / width) * 100)}%`,
              width: `${Math.max(0.6, ((ci[1] - ci[0]) / width) * 100)}%`,
              background: crosses ? 'rgba(201,154,69,.75)' : 'rgba(76,141,219,.8)',
            }}
          />
        )}
      </div>
      <span style={{ width: 150, textAlign: 'right', fontSize: 11, color: 'var(--text2)', fontVariantNumeric: 'tabular-nums' }}>
        {pair(ci)}
      </span>
    </div>
  );
}

function Kpi({ label, value, meta, tone }: { label: string; value: string; meta: string; tone?: string }) {
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '11px 13px' }}>
      <div style={{ fontSize: 10.5, color: 'var(--text3)', letterSpacing: '.9px', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 650, margin: '4px 0', color: tone }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text2)' }}>{meta}</div>
    </div>
  );
}

function intervalPoints(fan: IntervalFan): number[] {
  return [fan.ci50, fan.ci80, fan.ci95]
    .filter((ci): ci is [number, number] => ci !== null)
    .flat();
}

export default function TrialDesign() {
  const [payload, setPayload] = useState<OutcomeCardsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { artifact, loading, error: artError } = useArtifact<TrialDesignData>('trial_design', 'artifacts');

  useEffect(() => {
    loadOutcomeCards().then(setPayload).catch((e: Error) => setError(e.message));
  }, []);

  const live = useMemo(() => (payload ? computeLiveComparison(payload) : null), [payload]);
  const recorded = artifact?.data ?? null;
  const check = useMemo(
    () => (live && recorded ? reconcileComparison(live, recorded) : null),
    [live, recorded],
  );

  // The domain the fans are drawn on: every live and recorded endpoint, zero
  // included, so "does this interval cross zero" is visible rather than a
  // judgement the eye has to make from digits.
  const domain = useMemo<[number, number]>(() => {
    const pts: number[] = [0, ...intervalPoints(live?.pooled.intervals ?? {
      ci50: null, ci80: null, ci95: null,
    })];
    if (recorded) pts.push(...intervalPoints(recorded.pooled.intervals));
    (live?.by_geometry ?? []).forEach((g) => pts.push(...intervalPoints(g.intervals)));
    (recorded?.by_geometry ?? []).forEach((g) => pts.push(...intervalPoints(g.intervals)));
    const lo = Math.min(...pts);
    const hi = Math.max(...pts);
    const pad = (hi - lo) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }, [live, recorded]);

  if (error) {
    return <Panel title="Trial design"><div style={{ color: 'var(--warning)', fontSize: 13 }}>⚠ {error}</div></Panel>;
  }
  if (!live) {
    return <Panel title="Trial design"><div style={{ color: 'var(--text2)' }}>Loading the recorded outcome cards…</div></Panel>;
  }

  const { pooled, by_geometry: byGeometry, cases, resolvability, inversion } = live;
  const briefRow = recorded?.anc_screen.candidates.find((c) => c.label === BRIEF_CALENDAR) ?? null;

  return (
    <div>
      <Panel
        title="The counterfactual — what the model did, under declared assumptions"
        sub={
          'Two arms on the same recorded geometry and the same seeds: the adaptive collateral-switch ' +
          `arm against continuous MTD. Switch rule: ${payload?.switch_rule ?? '—'}. Horizon ` +
          `${live.horizon_days} d. The endpoint is the model's own RANO >= 25% burden rule, so every ` +
          "figure below is a statement about the simulator's arithmetic — not a prediction about a " +
          'patient.'
        }
      >
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
          <span className="badge-status live">● COMPUTED LIVE IN THIS BROWSER</span>
          <span className="badge">ENGINE CONSTANTS ASSUMED</span>
          <span className="badge">
            PAIRED REPLICATES: {live.n_pairs} · INDEPENDENT UNITS: {pooled.nesting.independent_units}
          </span>
          <span className="badge-warn">NOT A PATIENT PREDICTION</span>
        </div>

        <DataTable
          head={['geometry', 'pairs (nested)', 'independent units', 'adaptive PFS median (d)', 'MTD PFS median (d)', 'mean paired difference (d)', 'resampled 95% interval (d)']}
          rows={byGeometry.map((g) => [
            g.case_id,
            String(g.n_pairs),
            String(g.nesting.independent_units),
            num(g.median_pfs_policy_days, 0),
            num(g.median_pfs_mtd_days, 0),
            num(g.mean_difference_days, 1),
            pair(g.intervals.ci95),
          ])}
        />
        <div style={{ marginTop: 8, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.6 }}>
          A geometry row pools every recorded case on that image; its PFS medians are the median across
          those cases of each case's own median PFS. The pair count and the number of independent units
          behind it are different numbers and are listed side by side, because a bootstrap over paired
          replicates assumes the pairs are exchangeable and in this record they are nested.
        </div>

        <div style={{ marginTop: 16, color: 'var(--text2)', fontSize: 12, lineHeight: 1.6 }}>
          <b style={{ color: 'var(--text)' }}>The simulated inversion, measured rather than assumed.</b>{' '}
          {inversion.definition}. {inversion.delay_note}.
        </div>
        <div style={{ marginTop: 8 }}>
          <DataTable
            head={['case', 'image', 'inversion day (adaptive)', 'inversion day (MTD)', 'delay (d)', 'endpoint censored']}
            rows={cases.map((c) => [
              c.patient_id,
              c.case_id,
              c.inversion_day_policy === null ? 'not reached' : `d${c.inversion_day_policy}`,
              c.inversion_day_mtd === null ? 'not reached' : `d${c.inversion_day_mtd}`,
              num(c.inversion_delay_days, 0),
              c.is_right_censored_lower_bound ? 'yes (lower bound)' : 'no',
            ])}
          />
        </div>
      </Panel>

      <Panel
        title="The noise floor — where the difference resolves, and where it does not"
        sub={
          'The same paired differences at three scales, because whether a simulated difference is ' +
          'resolvable is a property of the replicate count rather than of the comparison. A bar that ' +
          'crosses the red zero line is an arm comparison this record cannot separate.'
        }
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 12,
            marginBottom: 14,
          }}
        >
          <Kpi
            label={`pooled, ${live.n_pairs} pairs → ${pooled.nesting.independent_units} independent units`}
            value={`${num(pooled.median_difference_days, 1)} d`}
            meta="median paired difference"
          />
          <Kpi
            label="cases without an interval"
            value={`${live.n_cases_unresolvable} / ${cases.length}`}
            meta={`${MIN_PAIRS_FOR_AN_INTERVAL} pairs are needed before any interval exists`}
            tone={live.n_cases_unresolvable ? 'var(--warning)' : undefined}
          />
          <Kpi
            label="images without an interval"
            value={`${live.n_geometries_unresolvable} / ${byGeometry.length}`}
            meta="pooling the cases per image"
          />
          <Kpi
            label="independent units"
            value={`${num(pooled.nesting.independent_units, 0)}`}
            meta={`behind ${num(pooled.nesting.assumed_exchangeable, 0)} paired replicates the interval treats as exchangeable`}
            tone="var(--warning)"
          />
        </div>

        {/* The disclosure the audit requires: the pooled count is never shown
            without the number of units that actually stand behind it. */}
        <div
          style={{
            border: '1px solid rgba(201,154,69,.45)',
            background: 'rgba(201,154,69,.08)',
            borderRadius: 10,
            padding: '11px 14px',
            fontSize: 12.5,
            lineHeight: 1.65,
            marginBottom: 14,
          }}
        >
          <b style={{ color: '#fcd34d' }}>{NESTED_REPLICATES_FLAG}</b>{' '}
          The pooled row resamples <b>{num(pooled.nesting.assumed_exchangeable, 0)}</b> paired replicates
          as exchangeable. Behind them stand <b>{num(pooled.nesting.independent_units, 0)}</b> genuinely
          distinct units: {num(pooled.nesting.n_configurations, 0)} configurations —{' '}
          {num(pooled.nesting.configurations_per_geometry, 0)} synthetic initial resistant fractions on
          each of {num(pooled.nesting.n_geometries, 0)} imaging geometries — with{' '}
          {num(pooled.nesting.seeds_per_configuration, 0)} seeds each. {pooled.nesting.note}.
        </div>

        <div style={{ fontSize: 11.5, color: 'var(--text3)', marginBottom: 6, letterSpacing: '.6px', textTransform: 'uppercase' }}>
          pooled — resampled here (seed {TRIAL_DESIGN_LIVE_SEED})
        </div>
        <FanRow label="50%" ci={pooled.intervals.ci50} domain={domain} />
        <FanRow label="80%" ci={pooled.intervals.ci80} domain={domain} />
        <FanRow label="95%" ci={pooled.intervals.ci95} domain={domain} />

        <div style={{ marginTop: 14, fontSize: 11.5, color: 'var(--text3)', marginBottom: 6, letterSpacing: '.6px', textTransform: 'uppercase' }}>
          per image
        </div>
        {byGeometry.map((g) => (
          <div key={g.case_id} style={{ marginBottom: 10 }}>
            <div
              style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 3 }}
              title={g.nesting.note}
            >
              {g.case_id} · {g.n_pairs} pairs from {num(g.nesting.independent_units, 0)} independent
              units ({num(g.nesting.n_configurations, 0)} configurations ×{' '}
              {num(g.nesting.seeds_per_configuration, 0)} seeds) ·{' '}
              {g.noise_floor.status.replace(/_/g, ' ')}
            </div>
            <FanRow label="50%" ci={g.intervals.ci50} domain={domain} />
            <FanRow label="95%" ci={g.intervals.ci95} domain={domain} />
          </div>
        ))}

        <div style={{ marginTop: 12 }}>
          <DataTable
            head={['scale', 'pairs → independent units', 'state', 'separated', 'tied', 'other way']}
            rows={[
              [
                'pooled, whole record',
                `${live.n_pairs} → ${pooled.nesting.independent_units}`,
                <StateChip floor={pooled.noise_floor} />,
                String(pooled.noise_floor.n_separated),
                String(pooled.noise_floor.n_ties),
                String(pooled.noise_floor.n_violations),
              ],
              ...byGeometry.map((g) => [
                `one image (${g.case_id})`,
                `${g.n_pairs} → ${g.nesting.independent_units}`,
                <StateChip floor={g.noise_floor} />,
                String(g.noise_floor.n_separated),
                String(g.noise_floor.n_ties),
                String(g.noise_floor.n_violations),
              ]),
              ...cases.map((c) => [
                `one configuration (r0 ${num(c.r0, 4)} · ${c.patient_id} · ${c.case_id})`,
                `${c.n_pairs} → ${nestingCounts([c]).independent_units}`,
                <StateChip floor={c.noise_floor} />,
                String(c.noise_floor.n_separated),
                String(c.noise_floor.n_ties),
                String(c.noise_floor.n_violations),
              ]),
            ]}
          />
        </div>

        <div style={{ marginTop: 12, color: 'var(--text2)', fontSize: 12, lineHeight: 1.6 }}>
          <b style={{ color: 'var(--text)' }}>Reading this honestly.</b> {resolvability.note}. No pool in
          this record reaches the overlap state, whose label is {NOISE_FLOOR_FLAG}, and the state that
          needs fewer pairs than the method can use carries {INSUFFICIENT_REPLICATES_FLAG} — so the
          overlap branch is live but untriggered here.
        </div>
      </Panel>

      {check && (
        <Panel
          title="Cross-check — the live comparison against the recorded one"
          sub={
            'The comparison is computed here; the record was computed in Python over the same outcome ' +
            'cards with a different resampling generator, so the interval endpoints are not expected to ' +
            'match to the last digit. What must agree is every figure the browser can compute exactly, ' +
            'the prose summary each producer derives from its own counts, and the VERDICT each interval ' +
            'reaches.'
          }
        >
          <DataTable
            head={['quantity', 'live, in this browser', 'recorded', 'basis']}
            rows={check.rows.map((r) => [
              r.label,
              r.live,
              r.recorded,
              r.basis === 'exact' ? 'exact' : 'verdict',
            ])}
            minWidth={720}
          />
          {!check.agree && (
            <div
              style={{
                marginTop: 10,
                border: '1px solid rgba(201,154,69,.45)',
                background: 'rgba(201,154,69,.08)',
                color: '#fcd34d',
                borderRadius: 10,
                padding: '10px 13px',
                fontSize: 12.5,
                lineHeight: 1.55,
              }}
            >
              ⚠ The live comparison and the record do not agree on: {check.diverged.join('; ')}. For a
              row whose basis is a VERDICT, this is not a display bug and not a tie-break to be made
              here — a difference sitting at the edge of the noise floor shows up exactly this way, so
              read that row as unresolved. For a row whose basis is EXACT the two producers have drifted
              apart, and the recorded artifact is the record — the live side is the one that is wrong,
              and the fix is to make the two agree rather than to explain the gap.
            </div>
          )}
          <div style={{ marginTop: 10, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.6 }}>
            {artifact ? artifactStamp(artifact) : 'loading the record…'}
          </div>
        </Panel>
      )}

      {!recorded && !loading && (
        <Panel title="Rule-out — unavailable">
          <div style={{ color: 'var(--warning)', fontSize: 12.5, lineHeight: 1.6 }}>
            ⚠ {artError || 'the trial_design artifact is missing'}. The rule-out panel needs the recorded
            ANC screen, because that screen is a Python model and re-implementing it here would create a
            second definition of the same number.
          </div>
        </Panel>
      )}

      {recorded && (
        <Panel
          title="Rule-out — calendars the DECLARED ANC screen refuses"
          sub={
            'Every row is a declared calendar, not a search result. A refusal means THE MODEL refused ' +
            "this calendar under its declared parameters — a true statement about this repository's " +
            'arithmetic — and never a statement that a calendar is unsafe for a person, which is a ' +
            'different kind of claim and not one this screen can make.'
          }
        >
          <div
            style={{
              border: '1px solid rgba(201,154,69,.4)',
              background: 'rgba(201,154,69,.07)',
              borderRadius: 10,
              padding: '10px 13px',
              fontSize: 12.5,
              lineHeight: 1.6,
              marginBottom: 12,
            }}
          >
            <b style={{ color: '#fcd34d' }}>Declared screen, not a regulatory boundary.</b>{' '}
            {recorded.anc_screen.screen_status}. The screen permits a constant marrow-weighted burden of{' '}
            {num(recorded.anc_screen.budget_continuous_mg_m2, 2)} mg/m²/day because it DEFINES its budget
            by the label's own 150 mg/m² on days 1-5 of a 28-day cycle, so that regimen is the reference
            row rather than a proposal.
          </div>
          <DataTable
            head={['declared calendar', 'screen verdict', 'modelled nadir ANC', 'peak exposure / budget', 'assumed-potency drugs']}
            rows={recorded.anc_screen.candidates.map((c) => [
              <span title={c.note}>{c.label.replace(/_/g, ' ')}</span>,
              c.feasible
                ? <span className="badge" style={{ color: 'var(--success)' }}>permitted</span>
                : <span className="badge-warn">refused by the model</span>,
              num(c.nadir_anc, 3),
              num(c.peak_exposure_over_budget, 3),
              c.assumed_potency_drugs.length ? c.assumed_potency_drugs.join(', ') : 'none',
            ])}
            minWidth={720}
          />
          <div style={{ marginTop: 12, color: 'var(--text2)', fontSize: 12, lineHeight: 1.6 }}>
            {recorded.anc_screen.rule_out_note}.{' '}
            {briefRow
              ? briefRow.feasible
                ? "The design brief's own arithmetic — five days on, then a fourteen-day holiday — appears above as a row this screen PERMITS; check the holidays, because the label's own cycle is 28 days."
                : "The design brief's own arithmetic — five days on, then a fourteen-day holiday — appears above as one of the refused rows, because that holiday is a 19-day cycle against the label's 28 days; the calendar the brief quotes as its block is the label's regimen the adaptive arm is compared against, not an output of any search here."
              : "The design brief's own arithmetic is no longer among the rows above, so this screen no longer answers the question it was built to answer."}
          </div>
        </Panel>
      )}

      {recorded && (
        <Panel title="What this screen is, and what it is not">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--success)', letterSpacing: '.7px', textTransform: 'uppercase', marginBottom: 6 }}>
                what it is
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text2)', fontSize: 12, lineHeight: 1.7 }}>
                {recorded.what_this_screen_is.map((t) => <li key={t}>{t}</li>)}
              </ul>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--warning)', letterSpacing: '.7px', textTransform: 'uppercase', marginBottom: 6 }}>
                what it is not
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text2)', fontSize: 12, lineHeight: 1.7 }}>
                {recorded.what_this_screen_is_not.map((t) => <li key={t}>{t}</li>)}
              </ul>
            </div>
          </div>
          <div style={{ marginTop: 12, color: 'var(--text3)', fontSize: 11.5, lineHeight: 1.7 }}>
            {recorded.paired_difference_estimand}. Provenance:{' '}
            {Object.entries(artifact?.provenance ?? {}).map(([k, v]) => `${k}=${v}`).join(' · ')}
          </div>
        </Panel>
      )}
    </div>
  );
}
