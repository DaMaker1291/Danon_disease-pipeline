import React from 'react';
import type { TranslationalReadiness } from '../types';

interface Props { readiness: TranslationalReadiness | null }

export default function TranslationalGateway({ readiness }: Props) {
  if (!readiness) {
    return (
      <div className="tg-wrap glass">
        <div className="tg-header warn">
          <span className="tg-badge">[ DESIGN CAPABLE · PRECLINICAL STEP 0.5 ]</span>
        </div>
        <div className="seq-empty">
          Run the pipeline to load the Translational Readiness Gateway — the explicit gate tracking
          physical wet-lab validation required before any clinical deployment.
        </div>
      </div>
    );
  }

  return (
    <div className="tg-wrap glass">
      <div className={`tg-header ${readiness.clinicalTrialEligibility ? 'ready' : 'warn'}`}>
        <span className="tg-badge">
          {readiness.clinicalTrialEligibility
            ? '[ CLINICAL TRIAL ELIGIBLE ]'
            : '[ DESIGN CAPABLE · PRECLINICAL STEP 0.5 ]'}
        </span>
        <span className="tg-stage">{readiness.currentStage}</span>
      </div>

      <div className="tg-bar">
        <i style={{ width: `${readiness.readinessScorePct}%` }} />
        <span className="tg-pct">{readiness.readinessScorePct.toFixed(0)}% physical validation complete</span>
      </div>

      <div className="tg-note">
        In-silico design validated across {readiness.totalGates} gates. {readiness.verifiedCount} of {readiness.totalGates} physical lab milestones supplied. <strong>No live-cell data → not clinic-validated.</strong>
      </div>

      <div className="tg-gates">
        {readiness.gates.map(g => (
          <div key={g.id} className={`tg-gate ${g.verified ? 'verified' : 'pending'}`}>
            <div className="tg-gate-top">
              <span className="tg-tier">{g.tier}</span>
              <span className={`tg-status ${g.verified ? 'ok' : 'pend'}`}>
                {g.verified ? 'VERIFIED' : g.pendingLabel}
              </span>
            </div>
            <div className="tg-gate-name">{g.name}</div>
            <div className="tg-gate-desc">{g.description}</div>
          </div>
        ))}
      </div>

      <div className="tg-next">
        <span className="tg-next-label">REQUIRED NEXT STEP</span>
        <span className="tg-next-step">{readiness.requiredNextStep}</span>
      </div>

      <div className="tg-disclaimer">
        Computational simulation only. Metrics are in-silico optimization models — not validated
        biological safety, toxicity, in-vivo efficacy, or curative potential. Requires independent
        wet-lab validation, GLP safety profiling, and MHRA/FDA clearance before any therapeutic use.
      </div>
    </div>
  );
}
