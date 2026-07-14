import React from 'react';
import type { PipelineResult, Phase } from '../types';

interface Props { result: PipelineResult | null }

function fmtAdvantage(x: number): string {
  if (x >= 1e12) return `${(x / 1e12).toFixed(2)} trillion×`;
  if (x >= 1e9) return `${(x / 1e9).toFixed(2)} billion×`;
  if (x >= 1e6) return `${(x / 1e6).toFixed(2)} million×`;
  return `${x.toFixed(0)}×`;
}

function PhaseCard({ p }: { p: Phase }) {
  return (
    <div className={`phase-card ${p.status} ${p.horizon === 2 ? 'h2' : ''}`}>
      <div className="phase-top">
        <span className="phase-id">P{p.id}</span>
        <span className={`phase-dot ${p.status}`} />
      </div>
      <div className="phase-name">{p.name}</div>
      <div className="phase-cat">{p.category}</div>
      <div className="phase-bar"><i style={{ width: `${p.score * 100}%` }} className={p.status} /></div>
      <div className="phase-metric">{p.metric}</div>
      <div className="phase-foot">
        <span>score {p.score.toFixed(2)}</span>
        <span className="sel">×{p.selectivityFactor.toFixed(2)}</span>
      </div>
    </div>
  );
}

function StatChip({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className={`stat-chip ${ok === undefined ? '' : ok ? 'ok' : 'bad'}`}>
      <span>{label}</span><strong>{value}</strong>
    </div>
  );
}

export default function PhaseMatrix({ result }: Props) {
  if (!result?.advancedMetrics) {
    return (
      <div className="seq-empty">
        {!result
          ? 'Run the pipeline (3D Capsid tab) to execute all 24 phases. This matrix renders the live per-phase scores, the six Horizon-2 modules, and the compounded combinatorial design-space advantage.'
          : 'Pipeline returned without advanced metrics. Re-run to populate Horizon-2 data.'}
      </div>
    );
  }
  return <PhaseMatrixInner result={result} />;
}

function PhaseMatrixInner({ result }: { result: PipelineResult }) {
  const am = result.advancedMetrics;

  return (
    <div className="phase-matrix">
      <div className="pm-header">
        <div className="pm-hero glass">
          <span className="pm-hero-label">Combinatorial design-space advantage</span>
          <span className="pm-hero-value">{fmtAdvantage(result.combinatorialAdvantage)}</span>
          <span className="pm-hero-sub">∏ selectivity factors across 24 orthogonal phases · {result.phasesPassed}/24 passed</span>
        </div>
        <div className="pm-hero-stats">
          <StatChip label="Capsid viable (DMS)" value={am.dms.capsidViable ? 'yes' : 'no'} ok={am.dms.capsidViable} />
          <StatChip label="Plasma soluble" value={am.solvation.plasmaSoluble ? 'yes' : 'no'} ok={am.solvation.plasmaSoluble} />
          <StatChip label="tAI" value={am.codon.tai.toFixed(3)} ok={am.codon.codonOptimized} />
          <StatChip label="Synthesizable" value={am.synthesis.synthesizable ? 'yes' : 'no'} ok={am.synthesis.synthesizable} />
        </div>
      </div>

      <div className="panel-title" style={{ marginTop: 6 }}>24-Phase Screening Matrix</div>
      <div className="phase-grid">
        {result.phases.map(p => <PhaseCard key={p.id} p={p} />)}
      </div>

      <div className="panel-title" style={{ marginTop: 16 }}>Horizon-2 Module Detail (Phases 19–24)</div>
      <div className="h2-grid">
        {/* PHASE 9 — vector topology */}
        <div className="h2-card glass">
          <h4>P9 · Vector Topology (Capacity Gate)</h4>
          <div className="h2-stats">
            <StatChip label="Strategy" value={am.vectorCapacity.strategy.includes("Single") ? "Single-Vector" : "Dual-Vector"} ok={am.vectorCapacity.strategy.includes("Single")} />
            <StatChip label="Cargo" value={`${am.vectorCapacity.cargoLengthBp} bp`} ok={am.vectorCapacity.cargoLengthBp <= 4700} />
            <StatChip label="Tox mult." value={`×${am.vectorCapacity.toxicityRiskMultiplier}`} ok={am.vectorCapacity.toxicityRiskMultiplier <= 1.1} />
          </div>
          <div className="h2-row"><span>justification</span><span className="mono">LAMP2B fits &lt; 4.7 kb → single vector avoids 2× dose / liver toxicity</span></div>
        </div>

        {/* PHASE 19 */}
        <div className="h2-card glass">
          <h4>P19 · Deep Mutational Scan Fitness</h4>
          <div className="h2-stats">
            <StatChip label="Viable" value={am.dms.capsidViable ? 'yes' : 'no'} ok={am.dms.capsidViable} />
            <StatChip label="Min fitness" value={am.dms.minDmsFitness.toFixed(2)} />
            <StatChip label="Lethal" value={String(am.dms.lethalMutations)} ok={am.dms.lethalMutations === 0} />
            <StatChip label="Pocket hits" value={String(am.dms.conservedPocketViolations)} ok={am.dms.conservedPocketViolations === 0} />
          </div>
          <table className="mini-table">
            <thead><tr><th>Pos</th><th>Sub</th><th>BLOSUM</th><th>Fitness</th></tr></thead>
            <tbody>
              {am.dms.mutationScores.slice(0, 6).map((m, i) => (
                <tr key={i} className={m.lethal ? 'bad-row' : ''}>
                  <td>{m.position}</td><td>{m.original_aa}→{m.mutated_aa}</td>
                  <td>{m.blosum_score}</td><td>{m.dms_fitness.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* PHASE 20 */}
        <div className="h2-card glass">
          <h4>P20 · Solvation Free Energy ΔΔG</h4>
          <div className="h2-stats">
            <StatChip label="ΔΔG total" value={`${am.solvation.ddgSolvTotal.toFixed(2)} kcal/mol`} ok={am.solvation.plasmaSoluble} />
            <StatChip label="Bound" value={`±${am.solvation.maxAllowedDdg}`} />
            <StatChip label="Aggregation risk" value={am.solvation.aggregationRisk.toFixed(2)} ok={am.solvation.aggregationRisk < 0.5} />
          </div>
          {am.solvation.regions.map(r => (
            <div key={r.region} className="h2-row">
              <span>{r.region.replace('_', '-')}</span>
              <span className="mono">WT {r.dgWildType.toFixed(1)} → Mut {r.dgMutant.toFixed(1)} (ΔΔG {r.ddgSolv >= 0 ? '+' : ''}{r.ddgSolv.toFixed(2)})</span>
            </div>
          ))}
        </div>

        {/* PHASE 21 */}
        <div className="h2-card glass">
          <h4>P21 · CpG Depletion / Anti-Silencing</h4>
          <div className="h2-stats">
            <StatChip label="CpG raw" value={am.smar.cpgDensityRaw.toFixed(2)} />
            <StatChip label="CpG depleted" value={am.smar.cpgDensityDepleted.toFixed(2)} ok={am.smar.cpgWithinThreshold} />
            <StatChip label="Reduction" value={`${am.smar.cpgReductionPct.toFixed(0)}%`} ok={am.smar.cpgReductionPct > 50} />
            <StatChip label="S/MAR" value={am.smar.secondarySmAR.shieldingPredicted ? '2° on' : '2° off'} />
          </div>
          <div className="h2-row"><span>primary</span><span className="mono">{am.smar.primaryStrategy}</span></div>
          <div className="h2-row"><span>note</span><span className="mono">synonymous recoding; S/MAR kept secondary (saves packaging budget)</span></div>
        </div>

        {/* PHASE 22 */}
        <div className="h2-card glass">
          <h4>P22 · Codon Elongation / tAI</h4>
          <div className="h2-stats">
            <StatChip label="tAI" value={am.codon.tai.toFixed(3)} ok={am.codon.tai >= am.codon.threshold} />
            <StatChip label="Min window" value={am.codon.minWindowTai.toFixed(2)} />
            <StatChip label="Codons" value={String(am.codon.nCodons)} />
            <StatChip label="Stalls" value={String(am.codon.stallSites)} ok={am.codon.stallSites === 0} />
          </div>
          <div className="h2-row"><span>threshold</span><span className="mono">tAI ≥ {am.codon.threshold} · mean elongation {am.codon.meanElongationRate.toFixed(2)}</span></div>
        </div>

        {/* PHASE 23 */}
        <div className="h2-card glass">
          <h4>P23 · HLA-DRB1 Junction Decoupler</h4>
          <div className="h2-stats">
            <StatChip label="Split site" value={`aa${am.hla.chosenSplitPosition}`} />
            <StatChip label="Strongest IC50" value={`${am.hla.strongestIc50Nm.toFixed(0)} nM`} ok={am.hla.decoupled} />
            <StatChip label="Hits" value={String(am.hla.highAffinityHits)} ok={am.hla.highAffinityHits === 0} />
            <StatChip label="Decoupled" value={am.hla.decoupled ? 'yes' : 'no'} ok={am.hla.decoupled} />
          </div>
          <table className="mini-table">
            <thead><tr><th>9-mer core</th><th>Score</th><th>IC50 (nM)</th></tr></thead>
            <tbody>
              {am.hla.binders.slice(0, 5).map((b, i) => (
                <tr key={i} className={b.immunogenic ? 'bad-row' : ''}>
                  <td className="mono">{b.core_9mer}</td><td>{b.binding_score.toFixed(2)}</td><td>{b.predicted_ic50_nm.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* PHASE 24 */}
        <div className="h2-card glass">
          <h4>P24 · Synthesis Feasibility Screen</h4>
          <div className="h2-stats">
            <StatChip label="GC" value={`${am.synthesis.gcContent.toFixed(0)}%`} ok={am.synthesis.gcContent >= am.synthesis.gcBounds[0] && am.synthesis.gcContent <= am.synthesis.gcBounds[1]} />
            <StatChip label="Length" value={`${am.synthesis.lengthBp} bp`} />
            <StatChip label="Synthesizable" value={am.synthesis.synthesizable ? 'yes' : 'no'} ok={am.synthesis.synthesizable} />
          </div>
          <div className="h2-row"><span>GC windows</span><span className="mono">{am.synthesis.gcMinWindow}–{am.synthesis.gcMaxWindow}% · {am.synthesis.outOfBoundsWindows} out of bound</span></div>
          <div className="h2-row"><span>repeats</span><span className="mono">{am.synthesis.homopolymerRuns} homopolymer · {am.synthesis.invertedRepeats} hairpin · {am.synthesis.directRepeats} direct</span></div>
        </div>
      </div>
    </div>
  );
}
