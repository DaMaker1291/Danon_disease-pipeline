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
 * "measured in patients". Every number that appears here comes from the
 * bound-claim audit, not from prose: if a claim stops passing, this page says
 * so.
 */
export default function Overview({ onNavigate }: Props) {
  const { artifact, loading, error } = useArtifact<ClaimsArtifactData>('claims');

  const mine = (artifact?.data.claims ?? []).filter((c) => c.study !== 'vector_design');
  const byId = new Map<string, ClaimRow>(mine.map((c) => [c.id, c]));
  const rows = (ids: string[]) => ids.map((id) => byId.get(id)).filter(Boolean) as ClaimRow[];

  return (
    <div className="overview">
      <section className="panel glass overview-intro">
        <h3>A 3D spatial tumour twin for glioblastoma</h3>
        <p>
          The engine models one patient's tumour as a voxel grid: drug moves by a discretised
          diffusion PDE, each voxel grows logistically, a sensitive and a resistant clone compete by
          Lotka–Volterra dynamics, and killing follows saturating (EC50) kinetics. On top of that it
          compares two policies over a 365-day horizon — continuous maximum-tolerated dosing, and an
          adaptive schedule that switches drug when the resistant fraction crosses a threshold.
        </p>
        <p className="overview-warn">
          It simulates a patient's course; it does not predict one. There is no dataset in this
          repository that pairs a baseline scan, the treatment actually given, and the observed
          outcome for the same patient, so no patient-level C-index is reported from the engine. The
          growth and kill constants are declared assumptions, and the resistant clone's starting
          fraction is an input, not a measurement.
        </p>
      </section>

      <section className="panel glass">
        <div className="panel-title">How to read any screen here</div>
        <div className="overview-legend">
          <div className="legend-row">
            <span className="chip chip-real">measured real data</span>
            <span>Read from a real file: a BraTS-2021 volume, a GDC export, a TCGA cohort, a published paper's table.</span>
          </div>
          <div className="legend-row">
            <span className="chip">recorded engine output</span>
            <span>A run of a model in this repository, recorded to JSON so the deployed site works without a backend. Not an experiment.</span>
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
                <span className="overview-counter">
                  {mine.filter((c) => !c.supported).length} of {mine.length} checks failing
                </span>
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
                      {/* The check's own wording is the claim; the sentence is the
                          lengthier published form, kept underneath. */}
                      <div className="claim-sentence">{c.sentence}</div>
                      <div className="claim-detail">{c.detail}</div>
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
