import React from 'react';
import { GROUPS, TABS, OVERVIEW_REFUSALS, OVERVIEW_RESULTS } from '../nav';
import type { TabId } from '../nav';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ClaimsArtifactData, ClaimRow } from '../types/artifacts';
import ErrorBoundary from './ErrorBoundary';

interface Props {
  onNavigate: (id: TabId) => void;
}

/**
 * The landing page.
 *
 * Its job is to stop a visitor from reading "recorded engine output" as
 * "measured in patients", and to make the two studies in this repository
 * legible before any figure is looked at. Every number that appears here comes
 * from the bound-claim audit, not from prose: if a claim stops passing, this
 * page says so.
 */
export default function Overview({ onNavigate }: Props) {
  const { artifact, loading, error } = useArtifact<ClaimsArtifactData>('claims');

  const byId = new Map<string, ClaimRow>(
    (artifact?.data.claims ?? []).map((c) => [c.id, c]),
  );
  const rows = (ids: string[]) => ids.map((id) => byId.get(id)).filter(Boolean) as ClaimRow[];

  return (
    <div className="overview">
      <section className="panel glass overview-intro">
        <h3>Two studies, one repository</h3>
        <div className="overview-two">
          <div>
            <h4>Patient tumour twin — glioblastoma</h4>
            <p>
              A 3D spatial engine: drug transport by a discretised diffusion PDE, logistic growth
              with per-voxel proliferation, Lotka–Volterra competition between a sensitive and a
              resistant clone, and Michaelis–Menten kill. It compares continuous maximum-tolerated
              dosing against an adaptive schedule that switches drug when the resistant fraction
              crosses a threshold.
            </p>
            <p className="overview-warn">
              It simulates a patient's course; it does not predict one. There is no dataset in this
              repository that pairs a baseline scan, the treatment actually given, and the observed
              outcome for the same patient, so no patient-level C-index is reported from the
              engine.
            </p>
          </div>
          <div>
            <h4>Vector design — AAV9-LAMP2B (Danon disease)</h4>
            <p>
              A design pipeline: capsid variants are sampled and ranked on a Pareto front, antibody
              epitopes are masked by Poisson–Boltzmann electrostatics on the deposited AAV9
              structure (PDB 3UX1), and the LAMP2B payload is recoded, screened for synthesis, and
              checked against junction immunogenicity.
            </p>
            <p className="overview-warn">
              The output is a designed sequence that has never been synthesised, expressed, or
              dosed. Every gate is a scorer in this repository, not an assay.
            </p>
          </div>
        </div>
      </section>

      <section className="panel glass">
        <div className="panel-title">How to read any screen here</div>
        <div className="overview-legend">
          <div className="legend-row">
            <span className="chip chip-real">measured real data</span>
            <span>Read from a real file: a BraTS volume, a GDC export, a deposited structure, a TCGA cohort.</span>
          </div>
          <div className="legend-row">
            <span className="chip">recorded engine output</span>
            <span>A run of a model in this repository, recorded to JSON so the deployed site works without a backend. Not an experiment.</span>
          </div>
          <div className="legend-row">
            <span className="chip">design-engine scores</span>
            <span>The pipeline's own scorers. Numbers describe the design, not a biological measurement.</span>
          </div>
          <div className="legend-row">
            <span className="chip chip-warn">refused / unmet</span>
            <span>Something the project cannot show, printed with the live count that refuses it rather than omitted.</span>
          </div>
        </div>
      </section>

      <ErrorBoundary label="ClaimHighlights">
        {loading && <div className="panel glass seq-empty">Loading the claim audit…</div>}
        {error && (
          <div className="panel glass">
            <div className="panel-title">Claim audit unavailable</div>
            <div className="seq-empty">{error}</div>
          </div>
        )}
        {artifact && (
          <>
            <section className="panel glass">
              <div className="panel-title">
                What this project refuses to claim
                <span className="overview-counter">{artifact.data.n_failed} of {artifact.data.n_claims} checks failing</span>
              </div>
              <div className="claim-list">
                {rows(OVERVIEW_REFUSALS).map((c) => (
                  <div key={c.id} className="claim-row">
                    <span className={`chip ${c.supported ? 'chip-real' : 'chip-warn'}`}>
                      {c.tier}
                    </span>
                    <div>
                      <div className="claim-sentence">{c.sentence}</div>
                      <div className="claim-meta">{c.substantiated_by} · checked by {c.check}</div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel glass">
              <div className="panel-title">What is actually measured</div>
              <div className="claim-list">
                {rows(OVERVIEW_RESULTS).map((c) => (
                  <div key={c.id} className="claim-row">
                    <span className={`chip ${c.supported ? 'chip-real' : 'chip-warn'}`}>
                      {c.supported ? 'verified' : 'not verified'}
                    </span>
                    <div>
                      <div className="claim-sentence">{c.detail}</div>
                      <div className="claim-meta">sentence: {c.sentence}</div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="overview-stamp">{artifactStamp(artifact)}</div>
            </section>

            <section className="panel glass">
              <div className="panel-title">What no result here can establish</div>
              <ul className="overview-list">
                {artifact.data.cannot_establish.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </section>
          </>
        )}
      </ErrorBoundary>

      <section className="panel glass">
        <div className="panel-title">Where to go</div>
        {GROUPS.filter((g) => g !== 'Start here').map((g) => (
          <div key={g} className="overview-group">
            <div className="overview-group-label">{g}</div>
            <div className="overview-cards">
              {TABS.filter((t) => t.group === g).map((t) => (
                <button key={t.id} className="overview-card" onClick={() => onNavigate(t.id)}>
                  <span className="overview-card-title">{t.label}</span>
                  <span className="overview-card-desc">{t.desc}</span>
                  <span className="chip">{t.status}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
