import React, { useState, useCallback } from 'react';
import { fetchEsmFitness, fetchEsmStructure } from '../utils/api';
import ErrorBoundary from './ErrorBoundary';

interface Props {
  sequence: string | null;
}

function ssColor(ss: string): string {
  if (ss === 'helix') return '#3b82f6';
  if (ss === 'sheet') return '#f59e0b';
  return '#64748b';
}

function accColor(acc: number): string {
  if (acc > 0.7) return '#3b82f6';
  if (acc > 0.4) return '#6ee7b7';
  return '#f87171';
}

function llrColor(llr: number): string {
  if (llr > 1.5) return '#ef4444';
  if (llr > 0.5) return '#94a3b8';
  if (llr > -0.5) return '#6ee7b7';
  return '#3b82f6';
}

export default function StructureViewer({ sequence }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [structData, setStructData] = useState<any>(null);
  const [fitnessData, setFitnessData] = useState<any>(null);
  const [mode, setMode] = useState<'ss' | 'accessibility' | 'fitness' | 'contacts'>('ss');

  const loadFeatures = useCallback(async (seq: string) => {
    setLoading(true);
    setError(null);
    try {
      const [struct, fitness] = await Promise.all([
        fetchEsmStructure(seq),
        fetchEsmFitness(seq),
      ]);
      setStructData(struct);
      setFitnessData(fitness);
    } catch (e) {
      setError((e as Error).message || 'ESM prediction failed');
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    if (sequence && sequence.length > 50) {
      loadFeatures(sequence);
    }
  }, [sequence, loadFeatures]);

  if (!sequence || sequence.length <= 50) {
    return (
      <div>
        <div className="panel-title" style={{ marginBottom: 6 }}>
          🧬 ESM-2 Structural Biology — AI-Powered Protein Analysis
        </div>
        <div className="seq-empty">
          Run the pipeline to generate an engineered capsid sequence. ESM-2 will predict
          secondary structure, solvent accessibility, contact maps, and per-residue fitness
          from the protein language model embeddings.
        </div>
      </div>
    );
  }

  const ss = structData?.secondaryStructure;
  const acc = structData?.solventAccessibility || [];
  const contacts = structData?.contactMap || [];
  const conf = structData?.confidence || [];
  const perRes = fitnessData?.perResidue || [];

  const helixCount = ss?.helix?.filter((v: number) => v > 0.4).length ?? 0;
  const sheetCount = ss?.sheet?.filter((v: number) => v > 0.4).length ?? 0;
  const beneficial = perRes.filter((r: any) => r.predictedClass === 'beneficial').length;
  const neutral = perRes.filter((r: any) => r.predictedClass === 'neutral').length;
  const deleterious = perRes.filter((r: any) => r.predictedClass === 'deleterious').length;

  return (
    <div>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        🧬 ESM-2 Structural Biology — Per-Residue AI Predictions
      </div>

      {loading && (
        <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>
          <div className="spinner" style={{ display: 'inline-block', marginBottom: 8 }} />
          <div>Running ESM-2 protein language model...</div>
        </div>
      )}

      {error && (
        <div style={{ padding: 16, color: '#f87171', background: '#1a0a0a', borderRadius: 8, border: '1px solid #7f1d1d' }}>
          ESM prediction failed: {error}
        </div>
      )}

      {structData && !loading && (
        <>
          <div className="score-grid" style={{ marginBottom: 8 }}>
            <div className="score-cell">
              <span>Helix content</span>
              <strong style={{ color: '#3b82f6' }}>{(structData.helixFraction * 100).toFixed(0)}%</strong>
            </div>
            <div className="score-cell">
              <span>Sheet content</span>
              <strong style={{ color: '#f59e0b' }}>{(structData.sheetFraction * 100).toFixed(0)}%</strong>
            </div>
            <div className="score-cell">
              <span>Contact pairs</span>
              <strong>{contacts.length}</strong>
            </div>
            <div className="score-cell">
              <span>Model confidence</span>
              <strong>{(structData.meanConfidence * 100).toFixed(0)}%</strong>
            </div>
            {fitnessData && (
              <>
                <div className="score-cell accent">
                  <span>ESM-2 mean fitness</span>
                  <strong>{fitnessData.meanFitness?.toFixed(2)}</strong>
                </div>
                <div className="score-cell">
                  <span>Mutation effects</span>
                  <strong>{beneficial}↑ {neutral}→ {deleterious}↓</strong>
                </div>
              </>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
            <button className={`nav-btn ${mode === 'ss' ? 'active' : ''}`}
              onClick={() => setMode('ss')} style={{ fontSize: 11, padding: '4px 12px' }}>
              Secondary Structure
            </button>
            <button className={`nav-btn ${mode === 'accessibility' ? 'active' : ''}`}
              onClick={() => setMode('accessibility')} style={{ fontSize: 11, padding: '4px 12px' }}>
              Solvent Accessibility
            </button>
            <button className={`nav-btn ${mode === 'fitness' ? 'active' : ''}`}
              onClick={() => setMode('fitness')} style={{ fontSize: 11, padding: '4px 12px' }}>
              ESM-2 Fitness
            </button>
            <button className={`nav-btn ${mode === 'contacts' ? 'active' : ''}`}
              onClick={() => setMode('contacts')} style={{ fontSize: 11, padding: '4px 12px' }}>
              Contact Map
            </button>
          </div>

          {mode === 'ss' && ss && (
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                Per-residue secondary structure prediction from ESM-2 embeddings (first 200 residues shown)
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {ss.helix.slice(0, 200).map((h: number, i: number) => {
                  const s = ss.sheet[i];
                  const l = ss.loop[i];
                  const dominant = h >= s && h >= l ? 'helix' : s >= l ? 'sheet' : 'loop';
                  return (
                    <div key={i} style={{
                      width: 5, height: 18,
                      background: ssColor(dominant),
                      opacity: dominant === 'helix' ? h : dominant === 'sheet' ? s : l,
                      borderRadius: 1,
                    }} title={`Res ${i + 263}: H=${(h * 100).toFixed(0)}% S=${(s * 100).toFixed(0)}% L=${(l * 100).toFixed(0)}%`} />
                  );
                })}
              </div>
              <div style={{ marginTop: 6, fontSize: 10, color: '#64748b', display: 'flex', gap: 12 }}>
                <span><span style={{ color: '#3b82f6' }}>■</span> α-helix ({helixCount})</span>
                <span><span style={{ color: '#f59e0b' }}>■</span> β-sheet ({sheetCount})</span>
                <span><span style={{ color: '#64748b' }}>■</span> loop/coil</span>
              </div>
            </div>
          )}

          {mode === 'accessibility' && (
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                Solvent accessibility — surface-exposed (blue) vs buried (red) residues
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {acc.slice(0, 200).map((a: number, i: number) => (
                  <div key={i} style={{
                    width: 5, height: 18,
                    background: accColor(a),
                    borderRadius: 1,
                  }} title={`Res ${i + 263}: accessibility ${(a * 100).toFixed(0)}%`} />
                ))}
              </div>
              <div style={{ marginTop: 6, fontSize: 10, color: '#64748b', display: 'flex', gap: 12 }}>
                <span><span style={{ color: '#3b82f6' }}>■</span> Surface-exposed (&gt;70%)</span>
                <span><span style={{ color: '#6ee7b7' }}>■</span> Intermediate (40–70%)</span>
                <span><span style={{ color: '#f87171' }}>■</span> Buried (&lt;40%)</span>
              </div>
            </div>
          )}

          {mode === 'fitness' && perRes.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                ESM-2 log-likelihood ratio — wild-type fitness at each position (positive = WT favored)
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {perRes.slice(0, 200).map((r: any, i: number) => (
                  <div key={i} style={{
                    width: 5,
                    height: Math.max(2, Math.abs(r.logLikelihoodRatio) * 3),
                    background: llrColor(r.logLikelihoodRatio),
                    alignSelf: r.logLikelihoodRatio > 0 ? 'flex-end' : 'flex-start',
                    borderRadius: 1,
                  }} title={`VP1 ${r.positionVp1} ${r.wildTypeAa}: LLR ${r.logLikelihoodRatio.toFixed(2)} (${r.predictedClass})`} />
                ))}
              </div>
              <div style={{ marginTop: 6, fontSize: 10, color: '#64748b', display: 'flex', gap: 12 }}>
                <span><span style={{ color: '#3b82f6' }}>■</span> Beneficial ({beneficial})</span>
                <span><span style={{ color: '#94a3b8' }}>■</span> Neutral ({neutral})</span>
                <span><span style={{ color: '#ef4444' }}>■</span> Deleterious ({deleterious})</span>
              </div>
            </div>
          )}

          {mode === 'contacts' && (
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                Predicted residue-residue contacts from ESM-2 attention (top {contacts.length} pairs)
              </div>
              {contacts.length === 0 ? (
                <div style={{ padding: 12, color: '#64748b', fontSize: 12 }}>No high-confidence contacts predicted for this sequence.</div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 4, maxHeight: 200, overflowY: 'auto' }}>
                  {contacts.slice(0, 50).map((c: any, i: number) => (
                    <div key={i} style={{
                      padding: '3px 6px', fontSize: 10, background: 'rgba(59,130,246,0.08)',
                      border: '1px solid rgba(59,130,246,0.2)', borderRadius: 4, color: '#94a3b8',
                    }}>
                      <span style={{ color: '#3b82f6' }}>{c.i + 263}</span> ↔ <span style={{ color: '#3b82f6' }}>{c.j + 263}</span>
                      <span style={{ float: 'right', color: '#6ee7b7' }}>{(c.probability * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 8, fontSize: 9px, color: '#475569' }}>
            ESM-2 ({structData.modelVersion || 'esm2_t6_8M_UR50D'}) · {structData.sequenceLength} residues · structural features predicted from protein language model embeddings
          </div>
        </>
      )}
    </div>
  );
}
