/**
 * Headless assertions for the MRI Scans "run live plan" readout.
 *
 * There is no JS test runner in this project, so the pure module is compiled
 * with the project's own tsc and driven from node. The cases below are the ones
 * that would put a wrong number on the page:
 *
 *   - mtdBurden === 0        -> must not print Infinity%/NaN%
 *   - steering >= mtd        -> must not print a negative "lower"
 *   - nulls / short arrays   -> must not print NaN
 *   - pfsExtensionMedian null-> must say horizon-censored, not "0 d vs 0 d"
 *
 * Run:  node _verify_live_plan.mjs [--payload .tmp/plan.json]
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname ?? '.');
const out = mkdtempSync(join(tmpdir(), 'liveplan-'));
const failures = [];

function check(cond, msg) {
  console.log((cond ? '  ok   ' : '  FAIL ') + msg);
  if (!cond) failures.push(msg);
}

function build(d, grid = [14, 24, 24], voxels = 1774) {
  return {
    geometry: { tumourVoxels: voxels, gridShapeXyz: grid },
    pairedMetrics: {
      policyMedianPfsDays: 365,
      mtdMedianPfsDays: 365,
      pfsExtensionMedian: null,
      refToxicity: 361.7,
      mtdToxicity: 365,
      nReplicates: 5,
      ...(d.pairedMetrics ?? {}),
    },
    dailyCurves: {
      burden_S: [1, 2, 3],
      burden_R: [1, 2, 3],
      mtdBurdenTotal: [2, 4, 6],
      ...(d.dailyCurves ?? {}),
    },
  };
}

console.log('compiling frontend/src/utils/livePlan.ts with the project tsc…');
// Invoke the project's own TypeScript through node directly: npx at the repo
// root resolves nothing (typescript lives in frontend/node_modules), and the
// Windows .cmd shim makes spawnSync fail with EINVAL.
const TSC = join(ROOT, 'frontend', 'node_modules', 'typescript', 'bin', 'tsc');
check(existsSync(TSC), `project tsc found at ${TSC}`);
execFileSync(
  process.execPath,
  [
    TSC,
    'frontend/src/utils/livePlan.ts',
    '--outDir',
    out,
    '--module',
    'es2020',
    '--target',
    'es2020',
    '--moduleResolution',
    'bundler',
  ],
  { stdio: 'inherit', cwd: ROOT },
);
const mod = await import(pathToFileURL(join(out, 'livePlan.js')).href);
const { summariseLivePlan, livePlanLines } = mod;
check(typeof summariseLivePlan === 'function', 'summariseLivePlan exported');
check(typeof livePlanLines === 'function', 'livePlanLines exported');

// ---- the real payload the API returned for BraTS2021_01017 -----------------
const payloadArg = process.argv.indexOf('--payload');
const payloadPath = payloadArg > -1 ? process.argv[payloadArg + 1] : '.tmp/plan_01017.json';
if (existsSync(payloadPath)) {
  const real = JSON.parse(readFileSync(payloadPath, 'utf8'));
  const lines = livePlanLines(real);
  console.log('\nreal API payload ->');
  lines.forEach((l) => console.log('    ' + l));
  const joined = lines.join('\n');
  check(
    joined.includes(`${real.geometry.tumourVoxels.toLocaleString('en-US')} voxels @ ${real.geometry.gridShapeXyz.join('×')}`),
    'real run reports the geometry the API returned',
  );
  check(
    !/NaN|Infinity|undefined|null/.test(joined),
    'real run prints no NaN / Infinity / undefined / null',
  );
  const s = summariseLivePlan(real);
  const mtdEnd = real.dailyCurves.mtdBurdenTotal.at(-1);
  const steerEnd = real.dailyCurves.burden_S.at(-1) + real.dailyCurves.burden_R.at(-1);
  check(
    Math.abs(s.mtdBurden - mtdEnd) < 1e-9 && Math.abs(s.steeringBurden - steerEnd) < 1e-9,
    'day-365 burden matches the last day of the API curves',
  );
  check(
    s.censored === (real.pairedMetrics.pfsExtensionMedian === null),
    'censoring flag matches the API metric',
  );
} else {
  console.log(`\n(no real payload at ${payloadPath}; skipping the recorded-run case)`);
}

// ---- the failure-prone edges ----------------------------------------------
console.log('\nedge cases ->');

const zeroMtd = livePlanLines(build({ dailyCurves: { burden_S: [1], burden_R: [1], mtdBurdenTotal: [0] } }));
check(
  !zeroMtd.join('\n').includes('Infinity') && !zeroMtd.join('\n').includes('NaN'),
  'MTD arm at zero voxels prints no Infinity%/NaN%',
);
check(!zeroMtd[1].includes('lower'), 'MTD arm at zero voxels claims no reduction');

const equal = livePlanLines(build({ dailyCurves: { burden_S: [3], burden_R: [3], mtdBurdenTotal: [6] } }));
check(!equal[1].includes('lower'), 'identical arms print no percentage');

const worse = livePlanLines(build({ dailyCurves: { burden_S: [9], burden_R: [9], mtdBurdenTotal: [6] } }));
check(
  !worse[1].includes('lower') && !/-\d+(\.\d+)?\s*%/.test(worse[1]),
  `steering worse than MTD never prints a negative reduction ("${worse[1]}")`,
);
check(worse[1].includes('18') && worse[1].includes('6'), 'the raw numbers are still shown when there is no gain');

const nulls = livePlanLines(
  build({ dailyCurves: { burden_S: [null], burden_R: [null, null], mtdBurdenTotal: [null, null] } }),
);
check(!nulls.join('\n').includes('NaN'), 'null curve entries render as 0, not NaN');

const empty = livePlanLines(build({ dailyCurves: { burden_S: [], burden_R: [], mtdBurdenTotal: [] } }));
check(!empty.join('\n').includes('NaN'), 'empty curve arrays render as 0, not NaN');

const censored = livePlanLines(build({}));
check(censored[2].startsWith('PFS: horizon-censored'), 'null extension prints horizon-censored');
check(!/0 d vs 0 d/.test(censored[2]), 'null extension never prints "0 d vs 0 d"');

const uncensored = livePlanLines(
  build({ pairedMetrics: { pfsExtensionMedian: 0.2, policyMedianPfsDays: 300, mtdMedianPfsDays: 250 } }),
);
check(uncensored[2] === 'simulated median PFS: steering 300 d vs MTD 250 d (model output)', 'a real extension prints both medians');

const pct = summariseLivePlan(
  build({ dailyCurves: { burden_S: [11.639026374964425], burden_R: [0], mtdBurdenTotal: [758.6332348330848] } }),
);
check(
  pct.reductionPct !== null && Math.abs(pct.reductionPct - 98.5) < 0.05,
  `reduction percentage is computed, not guessed (${pct.reductionPct?.toFixed(2)}%)`,
);

console.log();
if (failures.length) {
  console.log(`${failures.length} CHECK(S) FAILED`);
  failures.forEach((f) => console.log('  - ' + f));
  process.exit(1);
}
console.log('livePlan: all checks passed');