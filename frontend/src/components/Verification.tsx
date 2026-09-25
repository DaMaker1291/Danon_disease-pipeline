import React, { useMemo, useState } from 'react';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ClaimsArtifactData, ClaimRow } from '../types/artifacts';

/**
 * The verification page: the repository's claim registry, rendered.
 *
 * Reading the sentences is the point. "ESTABLISHED" means a check in
 * `danon/claims.py` agreed with the sentence at build time -- it does not mean
 * the biology is right, and the page says so rather than leaving a green tick
 * to imply it.
 */
export default function Verification() {
  const { artifact, loading, error } = useArtifact<ClaimsArtifactData>('claims');
  const [tier, setTier] = useState<string>('all');
  const [only, setOnly] = useState<'all' | 'pass' | 'fail'>('all');

  const tiers = useMemo(() => {
    const t = new Set((artifact?.data.claims ?? []).map((c) => c.tier));
    return ['all', ...Array.from(t).sort()];
  }, [artifact]);

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
  const shown = d.claims.filter(
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
            {d.n_passed}/{d.n_claims} verified · {d.n_failed} failing
          </span>
        </div>
        <p className="page-header-source">
          Each row is a sentence this project may publish, the check that backs it, and the check's
          result at build time. Passing means the code agrees with the sentence. It does not mean the
          biology is right, and a failed check is listed rather than dropped.
        </p>

        <div className="verify-filters">
          <div className="verify-filter-group">
            <span className="verify-filter-label">tier</span>
            {tiers.map((t) => (
              <button
                key={t}
                className={`chip chip-btn ${tier === t ? 'chip-on' : ''}`}
                onClick={() => setTier(t)}
              >
                {t}
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
                  {c.id} · tier {c.tier} · backed by {c.substantiated_by}
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
