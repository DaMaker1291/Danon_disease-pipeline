import React, { useMemo, useState } from 'react';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ClaimsArtifactData, ClaimRow } from '../types/artifacts';
import { tierInfo } from '../utils/evidenceTier';

/**
 * The verification page: the repository's claim registry, rendered.
 *
 * Reading the sentences is the point. Two different things are shown and are
 * deliberately kept apart:
 *
 * * the chip -- "verified" / "FAILING" -- is this repository's audit: a check
 *   in `danon/claims.py` agreed with the sentence at build time. It says
 *   nothing about whether the biology is right.
 * * the tier -- `established`, `proven_failed`, ... -- is
 *   `danon.cure.EvidenceStatus`: how well the mechanism is supported in
 *   humans. A claim can be verified against this repo while its mechanism is
 *   only `emerging_human`, or refused while the mechanism is `proven_failed`.
 *
 * Conflating those two is how a green tick starts implying clinical efficacy,
 * so the tier names its source rather than borrowing the audit's vocabulary.
 */
export default function Verification() {
  const { artifact, loading, error } = useArtifact<ClaimsArtifactData>('claims');
  const [tier, setTier] = useState<string>('all');
  const [only, setOnly] = useState<'all' | 'pass' | 'fail'>('all');
  // The registry covers the whole repository, which also holds a vector-design
  // study this app does not route. Default to this app's study and say how
  // many rows that hides, rather than quietly dropping or showing them.
  const [showAll, setShowAll] = useState(false);

  const all = artifact?.data.claims ?? [];
  const scoped = showAll ? all : all.filter((c) => c.study !== 'vector_design');
  const hidden = all.length - scoped.length;

  const tiers = useMemo(() => {
    const t = new Set(scoped.map((c) => c.tier));
    return ['all', ...Array.from(t).sort()];
  }, [scoped]);

  if (loading) return <div className="panel glass seq-empty">Loading the claim audit…</div>;
  if (error || !artifact) {
    return (
      <div className="panel glass">
        <div className="panel-title">Claim audit unavailable</div>
        <div className="seq-empty">
          {error ?? 'no artifact'}
          <div style={{ marginTop: 8 }}>
            Rebuild it with: python -c &quot;from danon.app_artifacts import write_app_artifacts;
            write_app_artifacts(keys=[&apos;claims&apos;])&quot;
          </div>
        </div>
      </div>
    );
  }

  const d = artifact.data;
  const nPassed = scoped.filter((c) => c.supported).length;
  const shown = scoped.filter(
    (c) =>
      (tier === 'all' || c.tier === tier) &&
      (only === 'all' || (only === 'pass' ? c.supported : !c.supported)),
  );

  return (
    <div className="content-grid">
      <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
        <div className="panel-title">
          Bound-claim audit
          <span className="overview-counter">
            {nPassed}/{scoped.length} verified · {scoped.length - nPassed} failing
          </span>
        </div>
        <p className="page-header-source">
          Each row is a sentence this project may publish, the check that backs it, and the check's
          result at build time. Passing means the code agrees with the sentence. It does not mean the
          biology is right, and a failed check is listed rather than dropped.
          {hidden > 0 && (
            <>
              {' '}The registry checks {all.length} sentences in total; {hidden} belong to a
              vector-design study this app does not route.{' '}
              <button className="chip chip-btn" onClick={() => setShowAll(true)}>show them</button>
            </>
          )}
          {showAll && all.length > scoped.length && (
            <>
              {' '}
              <button className="chip chip-btn" onClick={() => setShowAll(false)}>hide them</button>
            </>
          )}
        </p>

        <div className="verify-filters">
          <div className="verify-filter-group">
            <span className="verify-filter-label">tier</span>
            {tiers.map((t) => (
              <button
                key={t}
                className={`chip chip-btn ${tier === t ? 'chip-on' : ''}`}
                onClick={() => setTier(t)}
                title={t === 'all' ? 'every evidence tier' : tierInfo(t).source}
              >
                {t === 'all' ? 'all' : tierInfo(t).label}
              </button>
            ))}
          </div>
          <div className="verify-filter-group">
            <span className="verify-filter-label">result</span>
            {(['all', 'pass', 'fail'] as const).map((o) => (
              <button
                key={o}
                className={`chip chip-btn ${only === o ? 'chip-on' : ''}`}
                onClick={() => setOnly(o)}
              >
                {o}
              </button>
            ))}
          </div>
        </div>

        <div className="claim-list">
          {shown.map((c: ClaimRow) => (
            <div key={c.id} className={`claim-row ${c.supported ? '' : 'claim-row-fail'}`}>
              <span className={`chip ${c.supported ? 'chip-real' : 'chip-warn'}`}>
                {c.supported ? 'verified' : 'FAILING'}
              </span>
              <div style={{ minWidth: 0 }}>
                <div className="claim-sentence">{c.sentence}</div>
                <div className="claim-meta">
                  {c.id} · tier{' '}
                  <span title={tierInfo(c.tier).source}>{tierInfo(c.tier).label}</span>
                  {c.study !== 'tumour_twin' && ` · study ${c.study}`}
                  {' · backed by '}{c.substantiated_by}
                </div>
                <div className="claim-detail">{c.detail}</div>
                <div className="claim-limit">limit: {c.limit}</div>
              </div>
            </div>
          ))}
          {!shown.length && <div className="seq-empty">No claims match that filter.</div>}
        </div>
        <div className="overview-stamp">{artifactStamp(artifact)}</div>
      </div>

      <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
        <div className="panel-title">What passing this audit does not establish</div>
        <ul className="overview-list">
          {d.cannot_establish.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </div>
    </div>
  );
}
