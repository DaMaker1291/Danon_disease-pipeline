/**
 * Headless assertions for the evidence-tier display on the landing page.
 *
 * The bug this guards against: `tier` is `danon.cure.EvidenceStatus`, and the
 * Overview used to render the raw enum member as a chip coloured by the *audit
 * result*. Every claim on the "what this project refuses to claim" panel is
 * `proven_failed` or `unknown`, so they all rendered as green chips reading
 * "proven_failed" -- the colour reserved for measured real data, under a legend
 * that teaches a completely different three-term vocabulary. A visitor cannot
 * decode that, and green reads as a pass.
 *
 * So the assertions here are about the mapping NOT collapsing back:
 *
 *   - every tier present in the live artifact has a real display label
 *     (a new upstream enum member must not silently render as a bare slug)
 *   - no label contains an underscore, and no label is empty
 *   - `proven_failed` / `unknown` / `animal_only` never carry the `real` tone
 *   - every mapped tier's `source` names the defining module, so the tooltip
 *     points at code instead of paraphrasing it
 *   - an unrecognised tier degrades to `warn`, never to `real`
 *   - every claim the landing page highlights as a refusal is `warn` (the list
 *     is read from nav.ts, so it does not pin a count)
 *   - the ids `nav.ts` highlights still exist in the artifact (a typo there
 *     drops a card silently, because the row lookup filters missing ids out)
 *
 * Run:  node _verify_evidence_tier.mjs
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname ?? '.');
const out = mkdtempSync(join(tmpdir(), 'tier-'));
const failures = [];

function check(cond, msg) {
  console.log((cond ? '  ok   ' : '  FAIL ') + msg);
  if (!cond) failures.push(msg);
}

// ---- compile the pure module with the project's own tsc --------------------
console.log('compiling frontend/src/utils/evidenceTier.ts with the project tsc…');
const TSC = join(ROOT, 'frontend', 'node_modules', 'typescript', 'bin', 'tsc');
check(existsSync(TSC), `project tsc found at ${TSC}`);
execFileSync(
  process.execPath,
  [
    TSC,
    'frontend/src/utils/evidenceTier.ts',
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
const mod = await import(pathToFileURL(join(out, 'evidenceTier.js')).href);
const { EVIDENCE_TIERS, tierInfo } = mod;
check(typeof tierInfo === 'function', 'tierInfo exported');
check(typeof EVIDENCE_TIERS === 'object', 'EVIDENCE_TIERS exported');

// ---- the live artifact -----------------------------------------------------
const artifactPath = join(ROOT, 'frontend', 'public', 'artifacts', 'claims.json');
const claims = JSON.parse(readFileSync(artifactPath, 'utf8')).data.claims;
const tiersInUse = [...new Set(claims.map((c) => c.tier))].sort();
console.log(`\nartifact tiers in use: ${tiersInUse.join(', ')}`);

console.log('\nmapping ->');
for (const t of tiersInUse) {
  const info = tierInfo(t);
  // Declared, not the fallback. NOTE: the label is deliberately allowed to
  // equal the separated member name ("proven failed") -- `danon/claims.py`
  // warns that swapping vocabularies is how a provenance claim becomes
  // decoration, so the member names are kept and only made readable. What must
  // NOT happen is a tier falling through to the unrecognised-member branch,
  // which is how a new upstream enum member renders as a bare slug.
  check(
    !/not in this build/.test(info.source),
    `${t}: declared in EVIDENCE_TIERS (not the fallback branch) as "${info.label}"`,
  );
  check(!info.label.includes('_'), `${t}: label "${info.label}" has no underscore`);
  check(info.label.trim().length > 0, `${t}: label is not empty`);
  check(/danon\/cure\.py::EvidenceStatus/.test(info.source), `${t}: source names the defining module`);
}

// ---- the exact regression: refusals must not be painted as measured data ---
console.log('\ntone ->');
for (const t of ['proven_failed', 'unknown', 'animal_only']) {
  check(tierInfo(t).tone !== 'real', `${t}: never carries the "real" (measured-data) tone`);
}
check(tierInfo('established').tone === 'real', 'established: carries the "real" tone');

// ---- unknown members must degrade to warn, not to real ---------------------
console.log('\nunknown member ->');
const madeUp = tierInfo('some_new_upstream_tier');
check(madeUp.tone === 'warn', 'an unrecognised tier degrades to warn, never to real');
check(madeUp.label === 'some new upstream tier', `unrecognised tier is separated, not hidden: "${madeUp.label}"`);
check(/not in this build/.test(madeUp.source), 'unrecognised tier says it is not in this vocabulary');

// ---- ids the landing page highlights must resolve --------------------------
console.log('\nnav.ts ->');
const nav = readFileSync(join(ROOT, 'frontend', 'src', 'nav.ts'), 'utf8');
function idsOf(name) {
  const m = new RegExp(`export const ${name} = \\[([^\\]]*)\\]`).exec(nav);
  if (!m) return null;
  return [...m[1].matchAll(/'([^']+)'/g)].map((x) => x[1]);
}
const byId = new Map(claims.map((c) => [c.id, c]));
for (const [name, wantTone] of [
  ['OVERVIEW_REFUSALS', 'not-real'],
  ['OVERVIEW_RESULTS', null],
]) {
  const ids = idsOf(name);
  check(ids !== null, `${name} found in nav.ts`);
  for (const id of ids ?? []) {
    const c = byId.get(id);
    check(!!c, `${name}: "${id}" resolves to a claim in the artifact`);
    if (!c) continue;
    const info = tierInfo(c.tier);
    console.log(`       ${id} -> tier ${c.tier} (${info.label}, ${info.tone}) supported=${c.supported}`);
    if (wantTone === 'not-real') {
      check(
        info.tone !== 'real',
        `${name}: "${id}" is a refusal, so its chip is not the measured-data green`,
      );
    }
  }
}

console.log(
  failures.length === 0
    ? '\nevidenceTier: all checks passed'
    : `\nevidenceTier: ${failures.length} FAILED\n  - ${failures.join('\n  - ')}`,
);
process.exit(failures.length === 0 ? 0 : 1);
