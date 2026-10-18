/**
 * The live trial-design comparison.
 *
 * Pure: it takes the recorded outcome cards and returns the counterfactual, the
 * noise-floor states and the resolvability ladder. No React and no fetching, so
 * the rules that decide whether a simulated difference resolves live in exactly
 * one place -- `danon/trial_design.py` is the recorded mirror of those same
 * rules, and `reconcileComparison` is the only code that arbitrates between the
 * two producers.
 *
 * The two flags are the record's own labels: `danon/claims.py` checks that the
 * strings here are the strings its artifact carries, so the live statuses and
 * the recorded ones call an unresolved result by the same name.
 *
 * OWNERSHIP AND FLOW
 *
 *   outcome_cards.json ─┬─► computeLiveComparison   (this module) ─┐
 *                       └─► danon/trial_design.py   (the record)  ─┴─► reconcileComparison
 *
 *   danon/dosing.py, the declared ANC screen ─► the record's `anc_screen` ─► the view
 *
 *   this module      the comparison rules, the resolution states, the two labels
 *   the record       the ANC screen, plus a recorded copy of every figure above
 *   TrialDesign.tsx  rendering, and nothing else
 *
 * A new figure belongs in the two producers above, not in the view; a change to
 * a rule belongs here first, with the record updated to match.
 */

import type { OutcomeCard, OutcomeCardsPayload } from '../types/evotwin';
import type {
  IntervalFan,
  NestingBlock,
  NoiseFloorState,
  NoiseFloorStatus,
  TrialDesignCase,
  TrialDesignComparison,
  TrialDesignGroup,
} from '../types/artifacts';

/** Fixed so a resample here is reproducible; the record uses the same value. */
export const TRIAL_DESIGN_LIVE_SEED = 20261008;
/** Below this many pairs the percentile bootstrap refuses to answer. */
export const MIN_PAIRS_FOR_AN_INTERVAL = 5;
export const LEVELS: number[] = [0.5, 0.8, 0.95];

/** The widest interval still contains zero: the arms' spreads overlap. */
export const NOISE_FLOOR_FLAG = '[SIMULATED BENEFIT WITHIN NOISE FLOOR - UNRESOLVABLE]';
/** Too few pairs to form an interval at all: unassessed, not "within a floor". */
export const INSUFFICIENT_REPLICATES_FLAG = '[NO INTERVAL AT THIS REPLICATE COUNT - UNRESOLVED]';
/**
 * The pairs in this record are NESTED, not independent samples.
 *
 * A bootstrap over paired replicates treats them as exchangeable; here three
 * seeds share one configuration and three configurations share one tumour
 * geometry. The count and the nesting are disclosed together, always.
 */
export const NESTED_REPLICATES_FLAG = '[NESTED REPLICATES - NOT INDEPENDENT SAMPLES]';

const PAIRED_DIFFERENCE_ESTIMAND =
  'policy PFS - MTD PFS in days, over replicates paired by position within a case';
const INVERSION_DEFINITION =
  "the first 1-based day the simulated resistant fraction reaches the arm's own switch threshold";
const RESOLVABILITY_DEFINITION =
  "the same paired differences viewed at three scales -- within one case, pooled over one " +
  "image's cases, and pooled over the whole record -- because whether a simulated difference " +
  'is resolvable is a property of the replicate count, not of the comparison';

/**
 * The one nesting disclosure, mirrored word for word in `danon/trial_design.py`.
 *
 * A pool over a single geometry names that one image; the whole-record pool
 * names them all. One template means the two scales can differ only in the
 * counts they interpolate, never in what they say those counts mean.
 */
function nestingNote(
  subject: string,
  assumed: number,
  configurations: number,
  geometries: number,
  seeds: number,
  units: number,
): string {
  const scope =
    geometries > 1
      ? `${configurations} configurations across ${geometries} imaging geometries`
      : `${configurations} configurations on a single image`;
  return (
    `${subject} resamples ${assumed} paired replicates as exchangeable, but ` +
    `they come from ${scope}, ${seeds} seeds each: seeds within a ` +
    'configuration are the same model under a different seed rather than a ' +
    'new patient, and configurations sharing a geometry are the same tumour ' +
    'under different synthetic initial conditions, so the interval rests on ' +
    `at most ${units} nested units and is narrower than one over independent ` +
    'patients would be'
  );
}

/**
 * How many units a pool's pairs actually rest on.
 *
 * `assumed_exchangeable` is what the interval is computed over;
 * `independent_units` is how many genuinely distinct (geometry, synthetic
 * initial condition) configurations stand behind it. The two are reported
 * together so neither can be read as the other.
 */
export function nestingCounts(cases: TrialDesignCase[]): Omit<NestingBlock, 'note'> {
  const assumed = cases.reduce((total, c) => total + c.n_pairs, 0);
  const seedsPerConfiguration = cases.reduce((most, c) => Math.max(most, c.n_seeds), 0);

  const perGeometry = new Map<string, number>();
  const configurations = new Set<string>();
  cases.forEach((c) => {
    const geometry = String(c.case_id);
    perGeometry.set(geometry, (perGeometry.get(geometry) ?? 0) + 1);
    configurations.add(`${geometry}|${c.r0}`);
  });

  return {
    n_pairs: assumed,
    assumed_exchangeable: assumed,
    n_configurations: cases.length,
    independent_units: configurations.size,
    n_geometries: perGeometry.size,
    configurations_per_geometry: Math.max(0, ...perGeometry.values()),
    seeds_per_configuration: seedsPerConfiguration,
  };
}

/** The disclosure for one image pool: every pair on ONE tumour geometry. */
function poolNesting(cases: TrialDesignCase[]): NestingBlock {
  const counts = nestingCounts(cases);
  return {
    ...counts,
    note: nestingNote(
      "this pool's interval",
      counts.assumed_exchangeable,
      counts.n_configurations,
      counts.n_geometries,
      counts.seeds_per_configuration,
      counts.independent_units,
    ),
  };
}

/** The disclosure for the whole record: the pairs across every geometry. */
function recordNesting(cases: TrialDesignCase[]): NestingBlock {
  const counts = nestingCounts(cases);
  return {
    ...counts,
    note: nestingNote(
      'the pooled interval',
      counts.assumed_exchangeable,
      counts.n_configurations,
      counts.n_geometries,
      counts.seeds_per_configuration,
      counts.independent_units,
    ),
  };
}

const NO_DELAY_NOTE =
  'in the recorded replicates the two arms reach the switch threshold on the SAME simulated day ' +
  'in every case, so the model shows no simulated delay in this inversion -- the arms diverge ' +
  'after it, in burden, not in when it happens';
const SOME_DELAY_NOTE =
  'the recorded replicates show a non-zero simulated delay in the inversion day for at least one ' +
  'case; read the per-case rows rather than a summary';

/* --------------------------------------------------------------- resampling */

/** Mulberry32: a small deterministic PRNG, so a resample here is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile(sorted: number[], q: number): number {
  if (!sorted.length) return Number.NaN;
  const idx = (sorted.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Percentile bootstrap over the mean, mirroring `danon.campaign.bootstrap_ci_mean`. */
function bootstrapCiMean(
  samples: number[],
  seed: number,
  level: number,
  nBoot = 2000,
): [number, number] | null {
  if (samples.length < MIN_PAIRS_FOR_AN_INTERVAL) return null;
  const rnd = mulberry32(seed);
  const n = samples.length;
  const means: number[] = new Array(nBoot);
  for (let b = 0; b < nBoot; b += 1) {
    let s = 0;
    for (let i = 0; i < n; i += 1) s += samples[Math.floor(rnd() * n)];
    means[b] = s / n;
  }
  means.sort((x, y) => x - y);
  const alpha = (1 - level) / 2;
  return [percentile(means, alpha), percentile(means, 1 - alpha)];
}

function fan(diffs: number[], seed: number): IntervalFan {
  return {
    ci50: bootstrapCiMean(diffs, seed + 0, 0.5),
    ci80: bootstrapCiMean(diffs, seed + 1, 0.8),
    ci95: bootstrapCiMean(diffs, seed + 2, 0.95),
  };
}

/* ------------------------------------------------------------------ summary */

function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Does the interval fail to separate the arms, or does it not exist at all? */
function spansZero(ci: [number, number] | null): boolean {
  return ci !== null && ci[0] <= 0 && ci[1] >= 0;
}

/**
 * The three resolution states, kept distinct on purpose.
 *
 * `within_noise_floor` means the widest interval contains zero. `no_interval_at_
 * this_replicate_count` means there were too few pairs for the method to answer
 * at all -- a different statement, and calling a handful of all-positive pairs
 * "within a noise floor" would itself be an overclaim. Tie and violation counts
 * are reported beside the state rather than folded into it, because a set whose
 * interval excludes zero can still contain replicates that failed to separate.
 */
function noiseFloor(diffs: number[], intervals: IntervalFan): NoiseFloorState {
  const ci = intervals.ci95;
  const zeroSpanning = spansZero(ci);
  const nTies = diffs.filter((d) => d === 0).length;
  const nViolations = diffs.filter((d) => d < 0).length;
  const nSeparated = diffs.filter((d) => d > 0).length;
  const noPairSeparated = diffs.length > 0 && nSeparated === 0;

  let status: NoiseFloorStatus;
  let flag: string | null;
  let reason: string;
  if (ci === null) {
    status = 'no_interval_at_this_replicate_count';
    flag = INSUFFICIENT_REPLICATES_FLAG;
    reason =
      `${diffs.length} paired replicate(s) is below the ${MIN_PAIRS_FOR_AN_INTERVAL} ` +
      "the repository's own percentile method requires, so no interval exists here and " +
      'the difference is unassessed rather than resolvable';
  } else if (zeroSpanning || noPairSeparated) {
    status = 'within_noise_floor';
    flag = NOISE_FLOOR_FLAG;
    reason = zeroSpanning
      ? "the widest resampled interval contains zero, so the two arms' spreads overlap " +
        'and the simulated difference is not resolvable at this replicate count'
      : `not one of the ${diffs.length} paired replicate(s) separated the arms`;
  } else {
    status = 'resolves';
    flag = null;
    reason =
      `the widest interval excludes zero and ${nSeparated} of ${diffs.length} paired ` +
      `replicate(s) separated the arms; ${nTies} tied and ${nViolations} went the other way`;
  }
  return {
    status,
    ci_spans_zero: zeroSpanning,
    n_pairs: diffs.length,
    n_ties: nTies,
    n_violations: nViolations,
    n_separated: nSeparated,
    no_pair_separated: noPairSeparated,
    unresolvable_at_this_n: flag !== null,
    reason,
    flag,
  };
}

/* ------------------------------------------------------- the comparison */

function armPfs(card: OutcomeCard, arm: 'policy' | 'mtd', horizon: number): number[] {
  return card.arms[arm].replicate_day_readouts.map((r) =>
    r.pfs_day === null ? horizon : r.pfs_day,
  );
}

function inversionDay(series: number[], threshold: number): number | null {
  for (let i = 0; i < series.length; i += 1) {
    if (series[i] >= threshold) return i + 1;
  }
  return null;
}

function thresholdFromRule(rule: string): number {
  const m = /resistant fraction\s*>=\s*([0-9.]+)\s*%/.exec(rule || '');
  return m ? Number(m[1]) / 100 : 0.15;
}

/**
 * The members of each imaging-geometry pool, in first-seen order.
 *
 * Grouping holds the CASES rather than the flattened numbers, because a pool now
 * owns three things a flat list cannot answer: its nesting, its own cases' PFS
 * medians, and its interval. `String(...)` mirrors the record's own
 * `str(case[key])` grouping, so a case with no id lands in the same pool on both
 * sides.
 */
function membersByImage(cases: TrialDesignCase[]): Map<string, TrialDesignCase[]> {
  const out = new Map<string, TrialDesignCase[]>();
  cases.forEach((c) => {
    const key = String(c.case_id);
    const bucket = out.get(key) ?? [];
    bucket.push(c);
    out.set(key, bucket);
  });
  return out;
}

function present(values: (number | null)[]): number[] {
  return values.filter((v): v is number => v !== null);
}

/**
 * The counterfactual, the noise floor and the resolvability ladder, derived from
 * the recorded outcome cards alone.
 *
 * The pooled difference the reader sees first is a MEDIAN while the intervals are
 * over the MEAN (the repository's own S1 rule); both are reported so neither has
 * to stand for the other.
 */
export function computeLiveComparison(payload: OutcomeCardsPayload): TrialDesignComparison {
  const horizon = payload.cases[0]?.horizon_days ?? 365;
  const threshold = thresholdFromRule(payload.switch_rule);
  const pooledDiffs: number[] = [];

  const cases: TrialDesignCase[] = payload.cases.map((card, ci) => {
    const p = armPfs(card, 'policy', horizon);
    const m = armPfs(card, 'mtd', horizon);
    const n = Math.min(p.length, m.length);
    const diffs: number[] = [];
    for (let i = 0; i < n; i += 1) diffs.push(p[i] - m[i]);
    pooledDiffs.push(...diffs);

    const fanned = fan(diffs, TRIAL_DESIGN_LIVE_SEED + ci * 10);
    const invPolicy = inversionDay(card.arms.policy.daily.resistant_fraction, threshold);
    const invMtd = inversionDay(card.arms.mtd.daily.resistant_fraction, threshold);

    return {
      case_id: card.case_id,
      patient_id: card.patient_id,
      r0: card.intake.r0,
      n_pairs: diffs.length,
      n_seeds: Math.max(p.length, m.length),
      diffs_days: diffs,
      n_ties: diffs.filter((d) => d === 0).length,
      n_violations: diffs.filter((d) => d < 0).length,
      n_censored_policy_replicates: card.arms.policy.replicate_day_readouts.filter(
        (r) => r.pfs_day === null,
      ).length,
      is_right_censored_lower_bound: card.day_365.pfs_extension_is_lower_bound,
      median_pfs_policy_days: card.day_365.median_pfs_policy,
      median_pfs_mtd_days: card.day_365.median_pfs_mtd,
      ratio_of_medians_extension: card.day_365.pfs_extension,
      ratio_of_medians_label: card.day_365.pfs_extension_label,
      inversion_day_policy: invPolicy,
      inversion_day_mtd: invMtd,
      inversion_delay_days:
        invPolicy === null || invMtd === null ? null : invPolicy - invMtd,
      switch_threshold: threshold,
      intervals: fanned,
      noise_floor: noiseFloor(diffs, fanned),
    };
  });

  let gi = 0;
  const byGeometry: TrialDesignGroup[] = [];
  membersByImage(cases).forEach((members, caseId) => {
    const diffs = members.flatMap((m) => m.diffs_days);
    const fanned = fan(diffs, TRIAL_DESIGN_LIVE_SEED + 100 + gi * 10);
    gi += 1;
    byGeometry.push({
      case_id: caseId,
      n_pairs: diffs.length,
      mean_difference_days: mean(diffs),
      median_pfs_policy_days: median(present(members.map((m) => m.median_pfs_policy_days))),
      median_pfs_mtd_days: median(present(members.map((m) => m.median_pfs_mtd_days))),
      diffs_days: diffs,
      intervals: fanned,
      noise_floor: noiseFloor(diffs, fanned),
      nesting: poolNesting(members),
    });
  });

  const pooledFan = fan(pooledDiffs, TRIAL_DESIGN_LIVE_SEED);
  const pooledFloor = noiseFloor(pooledDiffs, pooledFan);

  const nCasesUnresolvable = cases.filter((c) => c.noise_floor.unresolvable_at_this_n).length;
  const nGeometriesUnresolvable = byGeometry.filter(
    (g) => g.noise_floor.unresolvable_at_this_n,
  ).length;

  const delays = cases
    .map((c) => c.inversion_delay_days)
    .filter((d): d is number => d !== null);
  const noDelay = delays.length > 0 && delays.every((d) => d === 0);
  const pairsPerImage = byGeometry[0]?.n_pairs ?? 0;

  return {
    n_cases: cases.length,
    n_pairs: pooledDiffs.length,
    horizon_days: horizon,
    bootstrap_seed: TRIAL_DESIGN_LIVE_SEED,
    levels: LEVELS,
    paired_difference_estimand: PAIRED_DIFFERENCE_ESTIMAND,
    pooled: {
      mean_difference_days: mean(pooledDiffs),
      median_difference_days: median(pooledDiffs),
      intervals: pooledFan,
      noise_floor: pooledFloor,
      nesting: recordNesting(cases),
    },
    cases,
    by_geometry: byGeometry,
    n_cases_unresolvable: nCasesUnresolvable,
    n_geometries_unresolvable: nGeometriesUnresolvable,
    resolvability: {
      definition: RESOLVABILITY_DEFINITION,
      min_pairs_for_an_interval: MIN_PAIRS_FOR_AN_INTERVAL,
      ladder: [
        {
          scale: 'within one case',
          groups: cases.length,
          pairs_per_group: cases[0]?.n_pairs ?? 0,
          groups_without_a_resolvable_interval: nCasesUnresolvable,
        },
        {
          scale: 'pooled over one image',
          groups: byGeometry.length,
          pairs_per_group: pairsPerImage,
          groups_without_a_resolvable_interval: nGeometriesUnresolvable,
        },
        {
          scale: 'pooled over the whole record',
          groups: 1,
          pairs_per_group: pooledDiffs.length,
          groups_without_a_resolvable_interval: pooledFloor.unresolvable_at_this_n ? 1 : 0,
        },
      ],
      note:
        'a pool without a resolvable interval is either below the ' +
        `${MIN_PAIRS_FOR_AN_INTERVAL} paired replicates the method needs, or an interval that ` +
        `still contains zero: ${nCasesUnresolvable} of ${cases.length} single-case pools, ` +
        `${nGeometriesUnresolvable} of ${byGeometry.length} image pools, and ` +
        `${pooledFloor.unresolvable_at_this_n ? 1 : 0} pooled whole-record row(s) are in that state`,
    },
    inversion: {
      definition: INVERSION_DEFINITION,
      n_cases_with_a_delay: delays.filter((d) => d !== 0).length,
      n_cases_with_equal_inversion_day: delays.filter((d) => d === 0).length,
      delay_note: noDelay ? NO_DELAY_NOTE : SOME_DELAY_NOTE,
    },
  };
}

/* ------------------------------------------------------ reconciliation */

/** What agreement means for a row when the two producers resample differently. */
export type ReconciliationBasis = 'exact' | 'verdict';

export interface ReconciliationRow {
  label: string;
  live: string;
  recorded: string;
  agrees: boolean;
  basis: ReconciliationBasis;
}

export interface Reconciliation {
  agree: boolean;
  rows: ReconciliationRow[];
  diverged: string[];
}

/** Format an interval fan member for display; `null` is a state, not a gap. */
export function pair(ci: [number, number] | null, d = 1): string {
  if (!ci) return 'no interval at this count';
  return `${ci[0].toFixed(d)} … ${ci[1].toFixed(d)}`;
}

function intervalAgrees(a: [number, number] | null, b: [number, number] | null): boolean {
  if (a === null || b === null) return a === b;
  return spansZero(a) === spansZero(b);
}

/**
 * Compare the live comparison against the recorded one.
 *
 * The endpoints are NOT expected to match: the two producers resample with
 * different generators. What must match is every figure the browser can compute
 * exactly, and the VERDICT each interval reaches -- whether it separates the
 * arms. A divergence is reported rather than resolved, because a difference
 * sitting at the edge of the noise floor is exactly the state this screen exists
 * to show.
 */
export function reconcileComparison(
  live: TrialDesignComparison,
  recorded: TrialDesignComparison,
): Reconciliation {
  const rows: ReconciliationRow[] = [];
  // Rows the browser can compute exactly: equality is the whole test.
  const add = (label: string, liveValue: string, recordedValue: string) => {
    rows.push({
      label,
      live: liveValue,
      recorded: recordedValue,
      agrees: liveValue === recordedValue,
      basis: 'exact',
    });
  };

  add('pooled pairs', String(live.n_pairs), String(recorded.n_pairs));
  add(
    'pooled median paired difference (d)',
    (live.pooled.median_difference_days ?? Number.NaN).toFixed(1),
    (recorded.pooled.median_difference_days ?? Number.NaN).toFixed(1),
  );
  add(
    'single-case pools without a resolvable interval',
    String(live.n_cases_unresolvable),
    String(recorded.n_cases_unresolvable),
  );
  add(
    'image pools without a resolvable interval',
    String(live.n_geometries_unresolvable),
    String(recorded.n_geometries_unresolvable),
  );
  add('resolvability note', live.resolvability.note, recorded.resolvability.note);
  add('inversion-day note', live.inversion.delay_note, recorded.inversion.delay_note);

  // The nesting disclosure is derived on BOTH sides from the same cards, so it
  // has to agree exactly -- a dropped or drifted disclosure is a disagreement.
  add(
    'pooled pairs (assumed exchangeable)',
    String(live.pooled.nesting.assumed_exchangeable),
    String(recorded.pooled.nesting.assumed_exchangeable),
  );
  add(
    'pooled independent units',
    String(live.pooled.nesting.independent_units),
    String(recorded.pooled.nesting.independent_units),
  );
  add(
    'image pools independent units',
    live.by_geometry.map((g) => g.nesting.independent_units).join(' + '),
    recorded.by_geometry.map((g) => g.nesting.independent_units).join(' + '),
  );
  add('pooled nesting note', live.pooled.nesting.note, recorded.pooled.nesting.note);
  add(
    'image-pool nesting note',
    live.by_geometry[0]?.nesting.note ?? '',
    recorded.by_geometry[0]?.nesting.note ?? '',
  );

  // The two resampled rows compare VERDICTS, and say so, so a mismatch in the
  // last digit is not reported as a disagreement.
  rows.push({
    label: `pooled ${Math.round((live.levels[live.levels.length - 1] ?? 0.95) * 100)}% interval (d)`,
    live: pair(live.pooled.intervals.ci95),
    recorded: pair(recorded.pooled.intervals.ci95),
    agrees: intervalAgrees(live.pooled.intervals.ci95, recorded.pooled.intervals.ci95),
    basis: 'verdict',
  });
  rows.push({
    label: 'pooled state',
    live: live.pooled.noise_floor.status,
    recorded: recorded.pooled.noise_floor.status,
    agrees: live.pooled.noise_floor.status === recorded.pooled.noise_floor.status,
    basis: 'verdict',
  });

  const diverged = rows.filter((r) => !r.agrees).map((r) => r.label);
  return { agree: diverged.length === 0, rows, diverged };
}
