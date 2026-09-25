/**
 * Navigation config, in one place so the header, the nav, and the landing
 * page cannot disagree about what a tab is.
 *
 * This app shows ONE study: the glioblastoma patient tumour twin. The
 * AAV9-LAMP2B vector-design pipeline that used to be routed here lives in
 * `src/vector_design/` and `danon/vector_design/` — preserved and runnable,
 * but not part of this product's navigation, because mixing a cardiac
 * gene-therapy design pipeline into an oncology tool made both harder to read.
 *
 * `status` says, per screen, what class of number the visitor is about to
 * read, because "recorded engine output" and "measured in real patients" are
 * different claims and the UI must not blur them.
 */

export type TabId =
  | 'overview'
  | 'twin'
  | 'evolution'
  | 'outcome'
  | 'upload'
  | 'paraview'
  | 'verification';

export type TabGroup =
  | 'Start here'
  | 'Patient twin (glioblastoma)'
  | 'Data & provenance';

export const GROUPS: TabGroup[] = [
  'Start here',
  'Patient twin (glioblastoma)',
  'Data & provenance',
];

/** What class of number a screen shows. Shown as a chip on every page. */
export type ScreenStatus =
  | 'recorded engine output'
  | 'measured real data'
  | 'live (needs local API)'
  | 'runs in your browser'
  | 'audit of this repository';

export interface TabDef {
  id: TabId;
  label: string;
  group: TabGroup;
  /** Nav tooltip and landing-page card text. */
  desc: string;
  /** Page header: what this screen is for. */
  purpose: string;
  /** Page header: where the numbers on this screen came from. */
  source: string;
  status: ScreenStatus;
  /** Page header chip when the screen has a hard precondition. */
  caveat?: string;
}

const SOURCE_TWIN =
  'BraTS-2021 segmentation + the engine, planned locally through api_server.py (uvicorn). '
  + 'On the deployed site the same fields render from the recorded evolution artifact.';

export const TABS: TabDef[] = [
  {
    id: 'overview',
    group: 'Start here',
    label: 'Overview',
    desc: 'What this is, what is measured, and what it refuses to claim',
    purpose:
      'A 3D spatial tumour-evolution engine for glioblastoma: drug transport, clonal competition, and the dosing schedule it derives. This page states which numbers on screen are measurements.',
    source: 'danon/claims.py — the bound-claim registry, re-run and recorded as artifacts/claims.json',
    status: 'audit of this repository',
  },
  {
    id: 'twin',
    group: 'Patient twin (glioblastoma)',
    label: 'MRI → Plan',
    desc: 'Upload a BraTS scan, get a dosing schedule',
    purpose:
      'Turn a patient mask into a spatial tumour state, simulate it forward under two dosing policies, and return the schedule.',
    source: SOURCE_TWIN,
    status: 'live (needs local API)',
    caveat: 'needs the local API',
  },
  {
    id: 'evolution',
    group: 'Patient twin (glioblastoma)',
    label: '365-Day Evolution',
    desc: 'Per-day engine frames, steering vs MTD',
    purpose:
      'Every frame is one simulated day: drug fields, sensitive and resistant burdens, and when resistance takes over under each policy.',
    source: 'danon/evolution_export.py over the engine, recorded as artifacts/evolution.json',
    status: 'recorded engine output',
  },
  {
    id: 'outcome',
    group: 'Patient twin (glioblastoma)',
    label: 'Day-365 State',
    desc: 'End-of-horizon burdens and outcome cards',
    purpose:
      'The day-365 end state and the per-case outcome cards, including progression-free survival under each policy.',
    source: 'engine runs recorded in outputs_real/outcome_cards.json',
    status: 'recorded engine output',
  },
  {
    id: 'upload',
    group: 'Patient twin (glioblastoma)',
    label: 'Data Intake',
    desc: 'NIfTI scans, sequences, availability manifests',
    purpose:
      'Inspect a file the way the intake would: NIfTI masks, FASTA/FASTQ, or a TCIA availability manifest. Nothing is uploaded or stored.',
    source: 'parsed in the browser (and identically by api_server.py when the API is running)',
    status: 'runs in your browser',
  },
  {
    id: 'paraview',
    group: 'Data & provenance',
    label: 'ParaView Artifact',
    desc: 'Committed BraTS volume, downloadable',
    purpose:
      'One committed day-120 tumour volume, rendered here and downloadable as the byte-identical .vti that ParaView opens.',
    source: 'real BraTS-2021 case, written by danon/imaging.py + ParaView VTI export',
    status: 'measured real data',
  },
  {
    id: 'verification',
    group: 'Data & provenance',
    label: 'Verification',
    desc: 'Every published claim and its check',
    purpose:
      'Every sentence this project may publish, the check that backs it, and whether that check passed when the audit last ran. Failed checks are listed, not dropped.',
    source: 'danon.claims.audit_claims(), recorded as artifacts/claims.json',
    status: 'audit of this repository',
  },
];

/** Claim ids surfaced on the landing page, in the order they are shown. */
export const OVERVIEW_REFUSALS = [
  'no_engine_c_index',
  'no_inflated_estimand',
  'no_rl_agent',
  'no_spatial_agreement_result',
];

/** Measured results the landing page highlights. */
export const OVERVIEW_RESULTS = [
  'heldout_discrimination',
  'stream4_cohort_resolution_is_a_design_calculation',
  'stream3_real_vasculature_measured_small_effect',
  'covariate_upgrades_measured_not_estimated',
];
