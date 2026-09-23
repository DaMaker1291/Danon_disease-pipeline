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
import StructureViewer from './components/StructureViewer';
import ErrorBoundary from './components/ErrorBoundary';
import Day365Outcome from './components/Day365Outcome';
import TwinPlanner from './components/TwinPlanner';
import VtiViewer from './components/VtiViewer';
import ParaviewGallery from './components/ParaviewGallery';
import { runPipeline } from './utils/api';
import { useArtifact } from './utils/artifacts';
import type { FlowTelemetry, ParetoPoint, UploadResult, PipelineResult, PipelineConstraints } from './types';
import type { MicrofluidicsArtifactData, PipelineArtifactData } from './types/artifacts';

type TabId = 'capsid' | 'phases' | 'lnp' | 'gene' | 'microfluidics' | 'clinical' | 'outcome' | 'upload' | 'twin' | 'paraview';
interface TabDef { id: TabId; label: string; icon: string; desc: string }

const TABS: TabDef[] = [
  { id: 'twin', label: 'MRI Twin Planner', icon: '🧠', desc: 'Scan in, treatment plan out' },
  { id: 'paraview', label: 'ParaView Gallery', icon: '🧊', desc: 'Real BraTS volumetric artifact' },
  { id: 'outcome', label: '365-Day Outcome', icon: '📅', desc: 'Exact day-365 engine state' },
  { id: 'capsid', label: '3D Capsid', icon: '🧬', desc: 'AAV9 structural viewer' },
  { id: 'phases', label: '24 Phases', icon: '🔬', desc: 'Hyper-dimensional matrix' },
  { id: 'lnp', label: 'LNP Simulator', icon: '💊', desc: 'Lipid nanoparticle assembly' },
  { id: 'gene', label: 'Gene Construct', icon: '🧪', desc: 'LAMP2B transgene map' },
  { id: 'microfluidics', label: 'Microfluidics', icon: '💧', desc: 'Y-junction flow simulation' },
  { id: 'clinical', label: 'Clinical', icon: '📊', desc: 'Pareto fronts & outcomes' },
  { id: 'upload', label: 'Upload', icon: '📤', desc: 'Sequence data ingress' },
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
  // Default view is the MRI twin planner: the front door of the app.
  const [tab, setTab] = useState<TabId>('twin');
  const [constraints, setConstraints] = useState<PipelineConstraints>(DEFAULT_CONSTRAINTS);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [selectedSpike, setSelectedSpike] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [twinVti, setTwinVti] = useState<string | null>(null);
  const setTwinVtiCb = useCallback((b64: string | null) => setTwinVti(b64), []);

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

  // Precomputed real runs, so every tab renders engine output with no backend.
  // A live pipeline result (from the API) always takes precedence.
  const { artifact: pipelineArtifact } = useArtifact<PipelineArtifactData>('pipeline');
  const { artifact: fluidicsArtifact } = useArtifact<MicrofluidicsArtifactData>('microfluidics');

  const effectiveResult = result ?? pipelineArtifact?.data ?? null;
  const artifactMode = result === null && pipelineArtifact !== null;
  const fallbackCandidates: ParetoPoint[] = effectiveResult?.paretoFront ?? [];

  const flow = fluidicsArtifact?.data.telemetry ?? null;
  const telemetry: FlowTelemetry | null = flow ? {
    reynoldsNumberAqueous: flow.reynolds_number_aqueous,
    reynoldsNumberOrganic: flow.reynolds_number_organic,
    reynoldsNumberMixed: flow.reynolds_number_mixed,
    wallShearStressPa: flow.wall_shear_stress_pa,
    maxShearStressPa: flow.max_shear_stress_pa,
    mixingEfficiency: flow.mixing_efficiency,
    pecletNumber: flow.peclet_number,
    pressureDropPa: flow.pressure_drop_pa,
    flowRegime: flow.flow_regime,
    shearStressSafe: flow.shear_stress_safe,
    frr: flow.frr,
  } : null;

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
        <CapsidViewer result={effectiveResult} selectedSpike={selectedSpike} onSelectSpike={setSelectedSpike} />
      </div>

      <div className="panel glass seq-out-panel">
        <SequenceOutputPanel result={effectiveResult} selectedSpike={selectedSpike} />
      </div>

      <div className="tg-col">
        <TranslationalGateway readiness={effectiveResult?.translationalReadiness ?? null} />
      </div>

      {effectiveResult?.sequence && (
        <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
          <ErrorBoundary label="StructureViewer">
            <StructureViewer sequence={effectiveResult.sequence} />
          </ErrorBoundary>
        </div>
      )}
    </div>
  );  const renderContent = () => {
    switch (tab) {
      case 'twin':
        return (
          <div className="content-grid" style={{ gridTemplateColumns: '1.05fr 0.95fr' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
              <ErrorBoundary label="TwinPlanner"><TwinPlanner onVti={setTwinVtiCb} /></ErrorBoundary>
            </div>
            <div className="panel glass" style={{ minHeight: 640, display: 'flex', flexDirection: 'column' }}>
              <div className="panel-title">3D evolution viewer — same fields ParaView opens</div>
              <ErrorBoundary label="VtiViewer"><VtiViewer vtiBase64={twinVti} /></ErrorBoundary>
            </div>
          </div>
        );
      case 'paraview':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="ParaviewGallery"><ParaviewGallery /></ErrorBoundary>
            </div>
          </div>
        );
      case 'capsid':
        return renderCapsidTab();
      case 'phases':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1' }}>
              <PhaseMatrix result={effectiveResult} />
            </div>
          </div>
        );
      case 'lnp':
        // The LNP view carries its own real formulation summary from the lnp
        // artifact; the flow-telemetry panel belongs on the microfluidics tab.
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1', height: 620 }}><LNPViewer /></div>
          </div>
        );
      case 'gene':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1', height: 560 }}><GeneViewer /></div>
          </div>
        );
      case 'microfluidics':
        return (
          <div className="content-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div className="panel glass"><MicrofluidicViewer /></div>
            <div className="panel glass">
              {telemetry
                ? <TelemetryPanel telemetry={telemetry} title="Flow Telemetry (engine artifact)" />
                : <div className="seq-empty">Microfluidics artifact not loaded — no telemetry to show.</div>}
            </div>
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
      case 'outcome':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="Day365Outcome"><Day365Outcome /></ErrorBoundary>
            </div>
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
            <div><h1>EvoTwin-10X</h1><span className="h-sub">Digital-Twin Oncology · MRI → Adaptive Treatment Plan</span></div>
          </div>
          <div className="h-badges">
            <span className="badge-status live">● LIVE</span>
            {artifactMode && <span className="badge-accent cyan">PRECOMPUTED ENGINE RUN</span>}
            {effectiveResult && <span className="badge-accent">{effectiveResult.phasesPassed}/24 phases</span>}
            {effectiveResult && effectiveResult.combinatorialAdvantage >= 1e12 && (
              <span className="badge-accent violet">{(effectiveResult.combinatorialAdvantage / 1e12).toFixed(1)}T× space</span>
            )}
            {effectiveResult && (
              effectiveResult.translationalReadiness?.clinicalTrialEligibility
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
        <span>
          EvoTwin-10X · spatial tumour-evolution twin engine · every number on this page is a recorded engine output or clearly labelled as refused
        </span>          <span className="f-right">paired steering-vs-MTD trials · ParaView VTI export · 975-test audit suite</span>
      </footer>
    </div>
  );
}
