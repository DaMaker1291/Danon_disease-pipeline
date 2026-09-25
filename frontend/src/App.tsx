import React, { useState, useCallback, useEffect } from 'react';
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
import EvolutionPlayer from './components/EvolutionPlayer';
import TwinPlanner from './components/TwinPlanner';
import VtiViewer from './components/VtiViewer';
import ParaviewGallery from './components/ParaviewGallery';
import Overview from './components/Overview';
import Verification from './components/Verification';
import PageHeader from './components/PageHeader';
import { runPipeline } from './utils/api';
import { useArtifact } from './utils/artifacts';
import { GROUPS, TABS } from './nav';
import type { TabId } from './nav';
import type { FlowTelemetry, ParetoPoint, UploadResult, PipelineResult, PipelineConstraints } from './types';
import type { MicrofluidicsArtifactData, PipelineArtifactData } from './types/artifacts';

/** Tab ids in the order the nav defines them, for URL parsing. */
const TAB_IDS = TABS.map((t) => t.id);
const DEFAULT_TAB: TabId = 'overview';

/**
 * Read the tab from the URL hash, so a view can be linked and the browser's
 * back/forward buttons work. An unknown hash falls back to the landing page
 * rather than rendering nothing.
 */
function tabFromHash(): TabId {
  const h = window.location.hash.replace(/^#\/?/, '');
  return (TAB_IDS as string[]).includes(h) ? (h as TabId) : DEFAULT_TAB;
}

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
  // The landing page is the front door; each tab is addressable by hash.
  const [tab, setTab] = useState<TabId>(() => tabFromHash());

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = useCallback((id: TabId) => {
    setTab(id);
    if (window.location.hash !== `#/${id}`) window.location.hash = `#/${id}`;
  }, []);
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
        <div className="panel-title">Target constraints</div>
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
        <div className="run-note">NSGA-II Pareto · PDB 3UX1 (AAV9) Poisson-Boltzmann · dual-region VR masking</div>
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
  );  const meta = TABS.find((t) => t.id === tab) ?? null;

  const renderContent = () => {
    switch (tab) {
      case 'overview':
        return <Overview onNavigate={navigate} />;
      case 'verification':
        return <Verification />;
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
      case 'evolution':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="EvolutionPlayer"><EvolutionPlayer /></ErrorBoundary>
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
            <div><h1>EvoTwin-10X</h1><span className="h-sub">Digital-Twin Oncology · MRI → Adaptive Treatment Plan</span></div>
          </div>
          {/*
            Badges state where the numbers on screen came from, and nothing
            else. Three that used to sit here were removed: a compounded
            "T× design space" (an arbitrary product -- see api_server.py), and
            an "MHRA ILAP" badge, which asserted a regulatory standing this
            project does not have (its own regulatory gate reports UNKNOWN and
            says it is not an MHRA determination).
          */}
          <div className="h-badges">
            {artifactMode
              ? <span className="badge-accent cyan">PRECOMPUTED ENGINE RUN</span>
              : running
                ? <span className="badge-warn">COMPUTING</span>
                : <span className="badge-status live">LIVE BACKEND RESULT</span>}
            {effectiveResult && effectiveResult.phaseSummary && (
              <span className="badge-accent">
                {effectiveResult.phaseSummary.nPass}/{effectiveResult.phaseSummary.nPhases} gates passed
              </span>
            )}
            {effectiveResult && (
              effectiveResult.translationalReadiness?.clinicalTrialEligibility
                ? <span className="badge-accent">READINESS GATES MET</span>
                : <span className="badge-warn">PRECLINICAL · GATES UNMET</span>
            )}
          </div>
        </div>
        <div className="h-right">
          {/* States where the numbers come from, not a vague "operational". */}
          <div className="status-indicator">
            <div className={running || result ? 'status-pulse' : 'status-idle'} />
            <span>
              {running
                ? 'Optimizing…'
                : result
                  ? 'Live backend result'
                  : artifactMode
                    ? 'Precomputed run (no backend)'
                    : 'No result loaded'}
            </span>
          </div>
        </div>
      </header>

      <nav className="app-nav glass">
        {GROUPS.map(g => (
          <div key={g} className="nav-group">
            <span className="nav-group-label">{g}</span>
            <div className="nav-group-items">
              {TABS.filter(t => t.group === g).map(t => (
                <button
                  key={t.id}
                  className={`nav-btn ${tab === t.id ? 'active' : ''}`}
                  onClick={() => navigate(t.id)}
                  title={t.desc}
                >
                  <span className="nav-label">{t.label}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <main className="app-main">
        <ErrorBoundary label="TabContent">
          {meta && meta.id !== 'overview' && meta.id !== 'verification' ? (
            <div className="tab-page">
              <PageHeader
                title={meta.label}
                subject={meta.group}
                purpose={meta.purpose}
                source={meta.source}
                status={meta.status}
                caveat={meta.caveat}
              />
              {renderContent()}
            </div>
          ) : (
            renderContent()
          )}
        </ErrorBoundary>
      </main>

      <footer className="app-footer glass">
        <span>
          EvoTwin-10X · spatial tumour-evolution twin engine · every number on this page is a recorded engine output or labelled as refused
        </span>
        {/* No test count here: a number in a footer is not re-derived by anything,
            and the one that used to sit here had drifted. The live count is the
            claim registry (danon/claims.py), which the audit checks. */}
        <span className="f-right">paired steering-vs-MTD trials · ParaView VTI export · claims verified in danon/claims.py</span>
      </footer>
    </div>
  );
}
