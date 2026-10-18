/**
 * livePlan — pure formatting for the MRI Scans "run live plan" readout.
 *
 * Kept out of the component so it can be exercised headlessly (see
 * `_verify_live_plan.mjs`): the arithmetic here is what turns a raw API payload
 * into the sentence the visitor reads, and a divide-by-zero or a wrong
 * "N% lower" would be a false number on a page whose whole point is that its
 * numbers are not invented.
 */

export interface LivePlanData {
  geometry: { tumourVoxels: number; gridShapeXyz: number[] };
  pairedMetrics: {
    policyMedianPfsDays: number;
    mtdMedianPfsDays: number;
    pfsExtensionMedian: number | null;
    refToxicity: number;
    mtdToxicity: number;
    nReplicates: number;
  };
  dailyCurves: {
    burden_S: (number | null)[];
    burden_R: (number | null)[];
    mtdBurdenTotal: (number | null)[];
  };
}

export interface LivePlanSummary {
  voxels: string;
  grid: string;
  replicates: number;
  steeringBurden: number;
  mtdBurden: number;
  /** null when the arms cannot be compared (equal, or not lower) — never 0 or NaN. */
  reductionPct: number | null;
  censored: boolean;
}

const last = (a: (number | null)[]): number => {
  const v = a[a.length - 1];
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
};

export function summariseLivePlan(d: LivePlanData): LivePlanSummary {
  const steeringBurden = last(d.dailyCurves.burden_S) + last(d.dailyCurves.burden_R);
  const mtdBurden = last(d.dailyCurves.mtdBurdenTotal);
  const lower = steeringBurden < mtdBurden;
  return {
    voxels: d.geometry.tumourVoxels.toLocaleString('en-US'),
    grid: d.geometry.gridShapeXyz.join('×'),
    replicates: d.pairedMetrics.nReplicates,
    steeringBurden,
    mtdBurden,
    // Guarded on purpose: mtdBurden === 0 with a non-zero steering arm would
    // otherwise print "Infinity% lower".
    reductionPct: lower && mtdBurden > 0 ? (1 - steeringBurden / mtdBurden) * 100 : null,
    censored: d.pairedMetrics.pfsExtensionMedian === null,
  };
}

export function livePlanLines(d: LivePlanData): string[] {
  const s = summariseLivePlan(d);
  const m = d.pairedMetrics;
  return [
    `live run · ${s.voxels} voxels @ ${s.grid} · ${s.replicates} paired replicates`,
    `day-365 burden: steering ${s.steeringBurden.toFixed(0)} vs MTD ${s.mtdBurden.toFixed(0)} voxels` +
      (s.reductionPct === null ? '' : ` (${s.reductionPct.toFixed(1)}% lower)`),
    s.censored
      ? 'PFS: horizon-censored — no progression in either arm inside the 365-day horizon, so no extension is computable'
      : `simulated median PFS: steering ${m.policyMedianPfsDays.toFixed(0)} d vs MTD ${m.mtdMedianPfsDays.toFixed(0)} d (model output)`,
    `dose-days: steering ${m.refToxicity.toFixed(0)} vs MTD ${m.mtdToxicity.toFixed(0)}`,
  ];
}