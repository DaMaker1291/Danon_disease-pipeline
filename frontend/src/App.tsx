import React, { useState, useCallback, useEffect } from 'react';
import UploadPortal from './components/UploadPortal';
import ErrorBoundary from './components/ErrorBoundary';
import Day365Outcome from './components/Day365Outcome';
import TrialDesign from './components/TrialDesign';
import EvolutionPlayer from './components/EvolutionPlayer';
import TwinPlanner from './components/TwinPlanner';
import DosingConsole from './components/DosingConsole';
import VtiViewer from './components/VtiViewer';
import ParaviewGallery from './components/ParaviewGallery';
import Overview from './components/Overview';
import TumorBoard from './components/TumorBoard';
import ScanGallery from './components/ScanGallery';
import Verification from './components/Verification';
import EvolutionForecast from './components/EvolutionForecast';
import PageHeader from './components/PageHeader';
import EvidencePanel from './components/EvidencePanel';
import { GROUPS, TABS } from './nav';
import type { TabId } from './nav';

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

export default function App() {
  // The landing page is the front door; each tab is addressable by hash.
  const [tab, setTab] = useState<TabId>(() => tabFromHash());
  // Evidence mode starts ON: this app's whole posture is that the caveats are
  // the product, so the switch can only ever ADD the evidence panel to the
  // screen -- turning it off never removes the intended-use notice, the
  // per-page provenance header or the footer.
  const [evidenceOpen, setEvidenceOpen] = useState(true);
  const [twinVti, setTwinVti] = useState<string | null>(null);
  const setTwinVtiCb = useCallback((b64: string | null) => setTwinVti(b64), []);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const navigate = useCallback((id: TabId) => {
    setTab(id);
    if (window.location.hash !== `#/${id}`) window.location.hash = `#/${id}`;
  }, []);

  const meta = TABS.find((t) => t.id === tab) ?? null;

  const renderContent = () => {
    switch (tab) {
      case 'overview':
        return <Overview onNavigate={navigate} />;
      case 'verification':
        return <Verification />;
      case 'tumorboard':
        return (
          <div className="content-grid" style={{ gridTemplateColumns: '1fr', gap: 0 }}>
            <ErrorBoundary label="TumorBoard" resetKey={tab}>
              <TumorBoard onNavigate={navigate} />
            </ErrorBoundary>
          </div>
        );
      case 'scans':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="ScanGallery" resetKey={tab}><ScanGallery /></ErrorBoundary>
            </div>
          </div>
        );
      case 'twin':
        return (
          <div className="content-grid" style={{ gridTemplateColumns: '1.05fr 0.95fr' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
              <ErrorBoundary label="TwinPlanner" resetKey={tab}><TwinPlanner onVti={setTwinVtiCb} /></ErrorBoundary>
            </div>
            <div className="panel glass" style={{ minHeight: 640, display: 'flex', flexDirection: 'column' }}>
              <div className="panel-title">3D evolution viewer — same fields ParaView opens</div>
              <ErrorBoundary label="VtiViewer" resetKey={tab}><VtiViewer vtiBase64={twinVti} /></ErrorBoundary>
            </div>
          </div>
        );
      case 'dosing':
        return (
          <ErrorBoundary label="DosingConsole" resetKey={tab}><DosingConsole /></ErrorBoundary>
        );
      case 'paraview':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="ParaviewGallery" resetKey={tab}><ParaviewGallery /></ErrorBoundary>
            </div>
          </div>
        );
      case 'outcome':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="Day365Outcome" resetKey={tab}><Day365Outcome /></ErrorBoundary>
            </div>
          </div>
        );
      case 'evolution':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="EvolutionPlayer" resetKey={tab}><EvolutionPlayer /></ErrorBoundary>
            </div>
          </div>
        );
      case 'trialdesign':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="TrialDesign" resetKey={tab}><TrialDesign /></ErrorBoundary>
            </div>
          </div>
        );
      case 'forecast':
        return (
          <div className="content-grid">
            <div style={{ gridColumn: '1 / -1' }}>
              <ErrorBoundary label="EvolutionForecast" resetKey={tab}>
                <EvolutionForecast />
              </ErrorBoundary>
            </div>
          </div>
        );
      case 'upload':
        return (
          <div className="content-grid">
            <div className="panel glass" style={{ gridColumn: '1 / -1' }}><UploadPortal /></div>
          </div>
        );
      default: {
        // Every TabId has a view. Adding a tab to nav.ts without a case here
        // narrows `tab` no longer to `never`, so this line stops compiling --
        // whereas before it rendered a page header over an empty body.
        const unhandled: never = tab;
        throw new Error(`no view is registered for tab ${String(unhandled)}`);
      }
    }
  };

  return (
    <div className="app">
      <header className="app-header glass">
        <div className="h-left">
          <div className="logo-wrap">
            <div>
              <h1>EvoTwin-10X</h1>
              <span className="h-sub">Glioblastoma tumour twin · decision support, clinician sign-off required</span>
            </div>
          </div>
          {/*
            No status badge here. The one that used to sit here reported the
            state of a design-pipeline run that this app no longer routes, and
            before that it said "Pipeline Operational" while rendering committed
            files. Each page states its own data provenance in its header.
          */}
        </div>
        <div className="h-right">
          <button
            className={`chip chip-btn ${evidenceOpen ? 'chip-on' : ''}`}
            onClick={() => setEvidenceOpen((v) => !v)}
            aria-pressed={evidenceOpen}
            title="Show the evidence behind every number: the tier vocabulary, the live audit, the paired model comparison and its censoring, and the regulatory gate. This switch only ever adds context; it cannot hide a caveat."
          >
            ● Evidence mode
          </button>
          <div className="status-indicator">
            <div className="status-idle" />
            <span>recorded engine runs · no backend required</span>
          </div>
        </div>
      </header>

      <div className="research-only-banner" role="note" aria-label="Intended-use notice">
        <strong>INTENDED USE — clinical decision support; a named clinician decides.</strong>{' '}
        Each screen states its evidence tier. The measured-patient screens (eg. Evolution Forecast)
        publish an operating point and a decision curve from the same audited record: act only within
        the threshold you have recalculated for your own prevalence, and record the reason for
        overriding or withholding. Simulation screens are planning aids, not patient-specific outcome
        predictions. No screen here acts autonomously or prescribes on its own; intended use,
        approved actions and hard guardrails are set out in
        clinical_twin/docs/clinical_use_model_card.md. This software has not been shown to stop or
        cure cancer or improve survival.
      </div>

      {evidenceOpen && <EvidencePanel />}

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
        <ErrorBoundary label="TabContent" resetKey={tab}>
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
          EvoTwin-10X · clinical decision support under clinical_twin/docs/clinical_use_model_card.md · no autonomous action, no patient outcome predictions
        </span>
        {/* No test count here: a number in a footer is not re-derived by anything,
            and the one that used to sit here had drifted. The live count is the
            claim registry (danon/claims.py), which the audit checks. */}
        <span className="f-right">synthetic/model-based comparisons · no demonstrated clinical benefit</span>
      </footer>
    </div>
  );
}
