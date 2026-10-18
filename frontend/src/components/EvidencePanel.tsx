import { useEffect, useMemo, useState } from 'react';
import { useArtifact } from '../utils/artifacts';
import type { ClaimsArtifactData, PairedExtensionData } from '../types/artifacts';
import { TIER_VOCABULARY_NOTE, tierInfo } from '../utils/evidenceTier';
import { loadOutcomeCards } from '../utils/evotwin';
import type { OutcomeCardsPayload } from '../types/evotwin';

/**
 * Evidence mode -- the inverse of a "hide the research clutter" switch.
 *
 * A clinical decision-support pitch asks a UI to strip the phase-plane, the
 * variance logs and the algorithmic warnings so a reader sees only "the
 * result". This panel is the one control that ADDS the epistemic surfaces to
 * the screen rather than removing them: the tier vocabulary, the live audit
 * status, the paired model comparison WITH its censoring, and the regulatory
 * gate. Turning it off removes this panel and nothing else -- the intended-use
 * banner, the per-page provenance header and the footer are rendered by App
 * whether this is on or off, so no switch here can hide a caveat.
 *
 * Every value is READ from a recorded artifact (artifacts/claims.json and
 * outputs_real/outcome_cards.json). Nothing is recomputed at render time, and a
 * figure the record does not carry -- a bootstrap interval over paired seeds --
 * is named as absent rather than estimated here, because a number that appears
 * in a UI with no artifact behind it is exactly the failure this mode exists to
 * prevent.
 *
 * The strings in this file are also scanned by danon/claims.py, which fails the
 * build if a shipped surface presents a fixed prescription, a mode that
 * suppresses the warnings, or a days-of-life figure. Keep that in mind before
 * rewording anything.
 */

function num(x: number | null | undefined, d = 0): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return x.toFixed(d).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function fmtPfs(pfs: number | null, horizon: number): string {
  return pfs === null ? `not reached within ${horizon} d` : `${num(pfs, 0)} d`;
}

export default function EvidencePanel() {
  const {
    artifact: claims,
    loading: claimsLoading,
    error: claimsError,
  } = useArtifact<ClaimsArtifactData>('claims');

  const {
    artifact: pairedExt,
    loading: pairedExtLoading,
    error: pairedExtError,
  } = useArtifact<PairedExtensionData>('paired_extension');

  const [cards, setCards] = useState<OutcomeCardsPayload | null>(null);
  const [cardsError, setCardsError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    loadOutcomeCards()
      .then((p) => {
        if (live) setCards(p);
      })
      .catch((e: Error) => {
        if (live) setCardsError(e.message);
      });
    return () => {
      live = false;
    };
  }, []);

  const card = cards?.cases[0] ?? null;
  const headline = card ? card.day_365 : null;

  const paired = useMemo(() => {
    if (!card) return [] as { seed: number; policy: number | null; mtd: number | null }[];
    const mtdBySeed = new Map<number, number | null>(
      card.arms.mtd.replicate_day_readouts.map((r) => [r.seed, r.pfs_day]),
    );
    return card.arms.policy.replicate_day_readouts.map((p) => ({
      seed: p.seed,
      policy: p.pfs_day,
      mtd: mtdBySeed.has(p.seed) ? (mtdBySeed.get(p.seed) ?? null) : null,
    }));
  }, [card]);

  return (
    <section className="evidence-panel glass" role="region" aria-label="Evidence mode">
      <div className="evidence-head">
        <span className="chip chip-real">EVIDENCE MODE</span>
        <strong>What every number here is — and is not</strong>
      </div>
      <p className="evidence-sub">
        This panel is the inverse of a switch that strips the algorithmic detail
        from the screen. It adds the vocabulary and the limits; it cannot remove
        the intended-use notice above, the per-page provenance header, or the
        footer, which are rendered whether this panel is open or closed.
      </p>

      <div className="evidence-grid">
        <div className="evidence-card">
          <div className="evidence-card-title">Bound-claim audit · live</div>
          {claimsError && (
            <div className="evidence-warn">claims artifact unavailable: {claimsError}</div>
          )}
          {claimsLoading && <div className="evidence-muted">loading…</div>}
          {claims && (
            <>
              <div className="evidence-figure">
                {claims.data.n_passed}/{claims.data.n_claims}
                <span> verified</span>
              </div>
              <div className="evidence-muted">
                {claims.data.n_failed} failing · read from {claims.source}, generated{' '}
                {claims.generated_utc}
              </div>
            </>
          )}
        </div>

        <div className="evidence-card">
          <div className="evidence-card-title">Honesty envelope · tiers</div>
          {claims ? (
            <ul className="evidence-list">
              {Object.entries(claims.data.provenance).map(([key, tier]) => (
                <li key={key}>
                  <code>{key}</code>
                  <span className={`chip chip-tier-${tierInfo(tier).tone}`}>
                    {tierInfo(tier).label}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="evidence-muted">loading…</div>
          )}
          <div className="evidence-muted">{TIER_VOCABULARY_NOTE}</div>
        </div>

        <div className="evidence-card">
          <div className="evidence-card-title">Adaptive switch vs paired MTD control</div>
          {cardsError && (
            <div className="evidence-warn">outcome cards unavailable: {cardsError}</div>
          )}
          {!cardsError && !card && <div className="evidence-muted">loading…</div>}
          {card && headline && (
            <>
              <div className="evidence-figure">
                {headline.pfs_extension_label ? headline.pfs_extension_label : 'not estimable'}
              </div>
              <div className="evidence-muted">
                median model PFS {num(headline.median_pfs_policy, 0)} d adaptive vs{' '}
                {num(headline.median_pfs_mtd, 0)} d MTD, over {headline.n_replicates}{' '}
                paired seeds on {card.case_id}.
              </div>
              {paired.length > 0 && (
                <table className="evidence-table">
                  <thead>
                    <tr>
                      <th>seed</th>
                      <th>adaptive</th>
                      <th>MTD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paired.map((r) => (
                      <tr key={r.seed}>
                        <td>{r.seed}</td>
                        <td>{fmtPfs(r.policy, card.horizon_days)}</td>
                        <td>{fmtPfs(r.mtd, card.horizon_days)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="evidence-muted">
                {headline.pfs_extension_is_lower_bound
                  ? 'The recorded extension is a censored LOWER BOUND, not a point ' +
                    'estimate: no adaptive-arm replicate progressed inside the ' +
                    `${headline.day}-day horizon, so the difference can only be bounded ` +
                    'below. This repository\'s own rule for this comparison ' +
                    '(danon/campaign.py::bootstrap_ci_mean, check S1) requires a ' +
                    'bootstrap 95% interval over paired seeds rather than a bare ' +
                    'number; that interval is not in this artifact, so it is named ' +
                    'as absent here and not computed at render time. It is a model ' +
                    'comparison, not an observed patient outcome, and the engine has ' +
                    'no patient-level validation.'
                  : 'The recorded extension is a model comparison across paired seeds, ' +
                    'not an observed patient outcome.'}
              </div>
            </>
          )}
        </div>

        <div className="evidence-card">
          <div className="evidence-card-title">Paired-seed PFS extension · recorded</div>
          {pairedExtError && (
            <div className="evidence-warn">
              paired-extension artifact unavailable: {pairedExtError}
            </div>
          )}
          {!pairedExtError && !pairedExt && pairedExtLoading && (
            <div className="evidence-muted">loading…</div>
          )}
          {pairedExt && (
            <>
              <div className="evidence-figure">
                {pairedExt.data.paired_mean_difference_ci95_days
                  ? `${num(pairedExt.data.paired_mean_difference_ci95_days[0], 0)}–${num(
                      pairedExt.data.paired_mean_difference_ci95_days[1],
                      0,
                    )} d`
                  : 'interval not recorded'}
                <span> bootstrap 95% over pairs</span>
              </div>
              <div className="evidence-muted">
                mean paired difference {num(pairedExt.data.paired_mean_difference_days, 1)} d
                over {pairedExt.data.n_pairs} replicate pairs in {pairedExt.data.n_cases} cases,
                computed from the recorded campaign log with no engine run (fixed
                seed {pairedExt.data.bootstrap_seed}).
              </div>
              <div className="evidence-muted">{pairedExt.data.lower_bound_note}</div>
              <div className="evidence-muted">
                {pairedExt.data.estimand_note} The outcome cards&apos; ratio-of-medians
                figure is {num(pairedExt.data.ratio_of_medians_extension_median, 3)} — a
                fraction, not days.
              </div>
              <div className="evidence-muted">{pairedExt.data.limit}</div>
            </>
          )}
        </div>

        <div className="evidence-card">
          <div className="evidence-card-title">Regulatory gate</div>
          <div className="evidence-body">
            Nothing here has been submitted to any regulator. What the clinical
            screens provide is <b>decision support</b>: the operating point, the
            decision curve and the provenance are on the same screen so a
            clinician can review the basis independently and override it. A
            surface that recommends a schedule which <b>drives</b> a
            clinician&apos;s decision, rather than supporting one and enabling
            independent review of its basis, fails the non-device criteria
            (FD&amp;C Act 520(o)(1)(E)) and is a device.
            Independent wet-lab validation, GLP safety profiling, and MHRA/FDA
            clearance before any therapeutic use would be required first.
          </div>
        </div>
      </div>
    </section>
  );
}
