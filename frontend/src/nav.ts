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
  | 'tumorboard'
  | 'scans'
  | 'twin'
  | 'dosing'
  | 'evolution'
  | 'outcome'
  | 'trialdesign'
  | 'upload'
  | 'paraview'
  | 'forecast'
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
      'A tumour twin for exploring spatial tumour-evolution hypotheses as decision support for a care team, with every screen declaring its evidence tier and a clinician signing off. It does not identify a clinically effective schedule on its own, predict an individual patient outcome, or act without that sign-off.',
    source: 'danon/claims.py — the bound-claim registry, re-run and recorded as artifacts/claims.json',
    status: 'audit of this repository',
  },
  {
    id: 'tumorboard',
    group: 'Patient twin (glioblastoma)',
    label: 'Tumour Board',
    desc: 'Five panels a tumour board reads, and the fields refused for lack of a source',
    purpose:
      'The dashboard a tumour board would read: the data on file, the recorded frames, the clonal kinetics the switch rule acts on, the paired adaptive-vs-MTD outcome, and the audit bar — with every field this repository has no source for refused in place rather than filled in.',
    source:
      'artifacts/tumorboard.json: recorded outcome cards, recorded day frames, the measured engine performance record, the recorded claims audit and a live count of the test tree',
    status: 'recorded engine output',
    caveat: 'no molecular or intraoperative data exists for this case',
  },
  {
    id: 'scans',
    group: 'Patient twin (glioblastoma)',
    label: 'MRI Scans',
    desc: 'The six committed real BraTS-2021 cases, rendered with their expert segmentations',
    purpose:
      'Actual patient MRI from this repository: T1ce volumes with the provided BraTS expert segmentation drawn on top, and the engine grid each case was downsampled to. The contours are labels, not predictions.',
    source:
      'data/brats/*_{t1ce,seg}.nii.gz (BraTS-2021, cached locally), rendered by _scan_gallery.py into artifacts/scans.json',
    status: 'measured real data',
  },
  {
    id: 'twin',
    group: 'Patient twin (glioblastoma)',
    label: 'MRI → Simulate',
    desc: 'Run a research simulation on a NIfTI segmentation mask',
    purpose:
      'Run a paired, model-based comparison on an uploaded segmentation mask. The output is a research simulation, not a patient-specific prediction or treatment schedule.',
    source: SOURCE_TWIN,
    status: 'live (needs local API)',
    caveat: 'needs the local API',
  },
  {
    id: 'dosing',
    group: 'Patient twin (glioblastoma)',
    label: 'Dosing Console',
    desc: 'Explore hypothetical schedules in a research simulator',
    purpose:
      'Explore hypothetical calendars on a synthetic phantom and compare model outputs. The declared ANC screen is not a clinical safety rule; neither screen results nor ranking should guide a real dose.',
    source:
      'danon/dosing.py (calendar, safety screen, grid search) over danon/evosteer.py::TumorEvolutionSimulator, served by POST /api/dosing/plan',
    status: 'live (needs local API)',
    caveat: 'the safety screen uses DECLARED parameters, not NCCN thresholds',
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
      'The day-365 simulated end state and recorded model outcomes. These are not observed patient progression-free survival or survival estimates.',
    source: 'engine runs recorded in outputs_real/outcome_cards.json',
    status: 'recorded engine output',
  },
  {
    id: 'trialdesign',
    group: 'Patient twin (glioblastoma)',
    label: 'Trial Design',
    desc: 'The counterfactual, the noise floor, and the calendars a declared screen refuses',
    purpose:
      'What the model did under declared assumptions, where the two arms cannot be separated at the recorded replicate count, and which calendars the declared ANC screen refused. It names no calendar to administer.',
    source:
      'computed live in this browser from outputs_real/outcome_cards.json, beside the recorded danon/trial_design report served as artifacts/trial_design.json',
    status: 'runs in your browser',
    caveat: 'a model compared against itself, not patient-level validation',
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
    id: 'forecast',
    group: 'Data & provenance',
    label: 'Evolution Forecast',
    desc: 'Real paired tumours: does the earlier specimen predict the later change?',
    purpose:
      'The repository\u2019s one measured-in-patients result. Two questions are asked of paired GLASS specimens \u2014 is a variant measured in the primary seen again later, and how far does its fraction move \u2014 and each is scored against the dumbest honest baseline it must beat under repeated patient-level cross-validation, with the interval that decides resolvability drawn on the same axis. One result resolves (a modest AUC clear of chance); the other does not, and the page keeps both estimators so a reader can see what one split is worth. On top of the ranking, the screen publishes a clinical report for a tumour board: one call at a threshold declared before the cohort was ever read (sensitivity, specificity and both predictive values against this cohort\u2019s prevalence, with the counts beside them) and a decision curve weighing it against treating every row and treating none. It is decision support for a named clinician who may withhold it and records why; the intended use, approved actions and hard guardrails are in clinical_twin/docs/clinical_use_model_card.md, and no figure here is a treatment effect.',
    source:
      'paired real GLASS specimens (allele fractions computed from tumour/reference read counts), scored by danon.evolution_forecast and recorded as artifacts/evolution_forecast.json',
    status: 'measured real data',
    caveat: 'one public cohort, a declared 27-gene panel, no therapy or outcome joined; the pairing is inferred from the specimen id',
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
  'no_spatial_agreement_result',
];

/** Measured results the landing page highlights. */
export const OVERVIEW_RESULTS = [
  'heldout_discrimination',
  'rl_policy_trained_and_measured',
  'stream4_cohort_resolution_is_a_design_calculation',
  'stream3_real_vasculature_measured_small_effect',
  'covariate_upgrades_measured_not_estimated',
];
