import React, { useState, useCallback } from 'react';
import CapsidViewer from './components/CapsidViewer';
import SequenceOutputPanel from './components/SequenceOutputPanel';
import LNPViewer from './components/LNPViewer';
import GeneViewer from './components/GeneViewer';
import MicrofluidicViewer from './components/MicrofluidicViewer';
import ParetoFrontChart from './components/ParetoFrontChart';
import ClinicalDashboard from './components/ClinicalDashboard';
import UploadPortal from './components/UploadPortal';
import TelemetryPanel from './components/TelemetryPanel';
import PhaseMatrix from './components/PhaseMatrix';
import TranslationalGateway from './components/TranslationalGateway';
import ErrorBoundary from './components/ErrorBoundary';
import { runPipeline } from './utils/api';
import type { FlowTelemetry, ParetoPoint, UploadResult, PipelineResult, PipelineConstraints } from './types';

type TabId = 'capsid' | 'phases' | 'lnp' | 'gene' | 'microfluidics' | 'clinical' | 'upload';
interface TabDef { id: TabId; label: string; icon: string; desc: string }

const TABS: TabDef[] = [
  { id: 'capsid', label: '3D Capsid', icon: '🧬', desc: 'AAV9 structural viewer' },
  { id: 'phases', label: '24 Phases', icon: '🔬', desc: 'Hyper-dimensional matrix' },
  { id: 'lnp', label: 'LNP Simulator', icon: '💊', desc: 'Lipid nanoparticle assembly' },
  { id: 'gene', label: 'Gene Construct', icon: '🧪', desc: 'LAMP2B transgene map' },
  { id: 'microfluidics', label: 'Microfluidics', icon: '💧', desc: 'Y-junction flow simulation' },
  { id: 'clinical', label: 'Clinical', icon: '📊', desc: 'Pareto fronts & outcomes' },
  { id: 'upload', label: 'Upload', icon: '📤', desc: 'Data ingress portal' },
];

const DEFAULT_CONSTRAINTS: PipelineConstraints = {
  maxHepaticAccumulation: 0.30,
  minCardiacTropism: 0.50,
  minImmuneEvasion: 0.50,
  lamp2bExpressionTarget: 0.70,
  candidatePool: 400,
  maxMutationsVrIv: 4,
  maxMutationsVrViii: 6,
  randomSeed: 42,
};

function Slider({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="ctl">
      <div className="ctl-head"><span>{label}</span><b>{unit === '%' ? `${(value * 100).toFixed(0)}%` : value}{unit && unit !== '%' ? unit : ''}</b></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<TabId>('capsid');
  const [constraints, setConstraints] = useState<PipelineConstraints>(DEFAULT_CONSTRAINTS);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [selectedSpike, setSelectedSpike] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);

  const setC = useCallback(<K extends keyof PipelineConstraints>(k: K, v: PipelineConstraints[K]) => {
    setConstraints(prev => ({ ...prev, [k]: v }));
  }, []);

  const execute = useCallback(async () => {
    setRunning(true); setError(null); setSelectedSpike(null);
    try {
      const r = await runPipeline(constraints);
      setResult(r);
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        setError('Pipeline timed out (120s). Try reducing the candidate pool size.');
      } else {
        setError(e?.message || 'pipeline failed');
      }
    } finally {
      setRunning(false);
    }
  }, [constraints]);

  const fallbackCandidates: ParetoPoint[] = result?.paretoFront ?? [];

  const telemetry: FlowTelemetry = {
    reynoldsNumberAqueous: 1.83, reynoldsNumberOrganic: 0.92, reynoldsNumberMixed: 1.62,
    wallShearStressPa: 0.32, maxShearStressPa: 0.48, mixingEfficiency: 0.72,
    pecletNumber: 8520, pressureDropPa: 125.0, flowRegime: 'laminar',
    shearStressSafe: true, frr: 0.33,
  };

  const renderCapsidTab = () => (
    <div className="capsid-layout">
      <aside className="run-panel glass">
        <div className="panel-title">⚙️ Target Constraints</div>
        <p className="run-sub">The pipeline generates the capsid — you set the design targets.</p>
        <Slider label="Max hepatic accumulation" value={constraints.maxHepaticAccumulation} min={0.05} max={0.6} step={0.01} unit="%" onChange={v => setC('maxHepaticAccumulation', v)} />
        <Slider label="Min cardiac tropism" value={constraints.minCardiacTropism} min={0.3} max={0.95} step={0.01} unit="%" onChange={v => setC('minCardiacTropism', v)} />
        <Slider label="Min immune evasion" value={constraints.minImmuneEvasion} min={0.3} max={0.95} step={0.01} unit="%" onChange={v => setC('minImmuneEvasion', v)} />
        <Slider label="LAMP2B expression target" value={constraints.lamp2bExpressionTarget} min={0.4} max={0.95} step={0.01} unit="%" onChange={v => setC('lamp2bExpressionTarget', v)} />
        <div className="ctl-divider" />
        <Slider label="Candidate pool" value={constraints.candidatePool} min={100} max={2000} step={100} onChange={v => setC('candidatePool', v)} />
        <Slider label="Max mutations VR-IV" value={constraints.maxMutationsVrIv} min={0} max={10} step={1} onChange={v => setC('maxMutationsVrIv', v)} />
        <Slider label="Max mutations VR-VIII" value={constraints.maxMutationsVrViii} min={0} max={16} step={1} onChange={v => setC('maxMutationsVrViii', v)} />
        <Slider label="Random seed" value={constraints.randomSeed} min={0} max={9999} step={1} onChange={v => setC('randomSeed', v)} />

        <button className="btn btn-run" onClick={execute} disabled={running}>
          {running ? <><span className="spinner" /> Optimizing…</> : '▶ Run Pipeline'}
        </button>
        {error && <div className="run-error">⚠ {error}</div>}
        <div className="run-note">NSGA-II Pareto · PDB 3J1S Poisson-Boltzmann · dual-region VR masking</div>
      </aside>

      <div className="panel glass capsid-canvas-panel">
        <CapsidViewer result={result} selectedSpike={selectedSpike} onSelectSpike={setSelectedSpike} />
      </div>

      <div className="panel glass seq-out-panel">
        <SequenceOutputPanel result={result} selectedSpike={selectedSpike} />
      </div>

      <div className="tg-col">
        <TranslationalGateway readiness={result?.translationalReadiness ?? null} />
      </div>

      {result?.sequence && (
        <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
          <ErrorBoundary label="StructureViewer">
            <StructureViewer sequence={result.sequence} />
          </ErrorBoundary>
        </div>
      )}
    </div>
  );

  const renderContent = () => {
    switch (tab) {
      case 'capsid': return renderCapsidTab();
      case 'phases':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
              <PhaseMatrix result={result} />
            </div>
          </div>
        );
      case 'lnp':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1', height: 560 }}><LNPViewer /></div>
            <div className="panel glass"><TelemetryPanel telemetry={telemetry} title="LNP Formulation Parameters" /></div>
          </div>
        );
      case 'gene':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1', height: 500 }}><GeneViewer /></div>
            <div className="panel glass"><TelemetryPanel telemetry={telemetry} title="Construct Specifications" /></div>
          </div>
        );
      case 'microfluidics':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1', height: 500 }}><MicrofluidicViewer /></div>
            <div className="panel glass"><TelemetryPanel telemetry={telemetry} title="Flow Telemetry" /></div>
          </div>
        );
      case 'clinical':
        return (
          <div className="content-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="panel glass">
              {fallbackCandidates.length
                ? <ParetoFrontChart candidates={fallbackCandidates} />
                : <div className="seq-empty">Run the pipeline (3D Capsid tab) to populate the Pareto front from real optimizer output.</div>}
            </div>
            <div className="panel glass"><ClinicalDashboard /></div>
          </div>
        );
      case 'upload':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1' }}><UploadPortal onResult={setUploadResult} /></div>
          </div>
        );
    }
  };

  return (
    <div className="app">
      <div className="bg-glow" style={{ left: '15%', top: '10%', width: '40vw', height: '40vw' }} />
      <div className="bg-glow" style={{ right: '10%', bottom: '5%', width: '35vw', height: '35vw', background: 'radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%)' }} />

      <header className="app-header glass">
        <div className="h-left">
          <div className="logo-wrap">
            <span className="logo-icon">🧬</span>
            <div><h1>Danon Pipeline</h1><span className="h-sub">AAV9-LAMP2B Discovery Platform</span></div>
          </div>
          <div className="h-badges">
            <span className="badge-status live">● LIVE</span>
            {result && <span className="badge-accent">{result.phasesPassed}/24 phases</span>}
            {result && result.combinatorialAdvantage >= 1e12 && (
              <span className="badge-accent violet">{(result.combinatorialAdvantage / 1e12).toFixed(1)}T× space</span>
            )}
            {result && (
              result.translationalReadiness?.clinicalTrialEligibility
                ? <span className="badge-accent">CLINIC ELIGIBLE</span>
                : <span className="badge-warn">PRECLINICAL STEP 0.5</span>
            )}
            <span className="badge-accent cyan">MHRA ILAP</span>
          </div>
        </div>
        <div className="h-right">
          <div className="status-indicator"><div className="status-pulse" /><span>{running ? 'Optimizing' : 'Pipeline Operational'}</span></div>
        </div>
      </header>

      <nav className="app-nav glass">
        {TABS.map(t => (
          <button key={t.id} className={`nav-btn ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
            <span className="nav-icon">{t.icon}</span>
            <span className="nav-label">{t.label}</span>
            <span className="nav-desc">{t.desc}</span>
          </button>
        ))}
      </nav>

      <main className="app-main"><ErrorBoundary label="TabContent">{renderContent()}</ErrorBoundary></main>

      <footer className="app-footer glass">
        <span>Danon Disease Pipeline v2.1 · AAV9-LAMP2B Cardiomyocyte Delivery</span>
        <span className="f-right">UCL/GOSH NCT03882437 Benchmark · 18-Phase Pareto Optimization</span>
      </footer>
    </div>
  );
}
