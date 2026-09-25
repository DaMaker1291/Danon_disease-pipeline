/**
 * Navigation config, in one place so the header, the nav, and the landing
 * page cannot disagree about what a tab is.
 *
 * This repository holds two studies that share a codebase but not a disease:
 * a patient tumour twin (glioblastoma) and a vector-design pipeline
 * (AAV9-LAMP2B for Danon disease). `group` carries that boundary, and
 * `status` says, per screen, what class of number the visitor is about to
 * read -- because "recorded engine output" and "measured in real patients"
 * are different claims and the UI must not blur them.
 */

export type TabId =
  | 'overview'
  | 'twin'
  | 'evolution'
  | 'outcome'
  | 'upload'
  | 'capsid'
  | 'phases'
  | 'gene'
  | 'clinical'
  | 'lnp'
  | 'microfluidics'
  | 'paraview'
  | 'verification';

export type TabGroup =
  | 'Start here'
  | 'Patient twin (glioblastoma)'
  | 'Vector design (AAV9-LAMP2B)'
  | 'Fabrication'
  | 'Data & provenance';

export const GROUPS: TabGroup[] = [
  'Start here',
  'Patient twin (glioblastoma)',
  'Vector design (AAV9-LAMP2B)',
  'Fabrication',
  'Data & provenance',
];

/** What class of number a screen shows. Shown as a chip on every page. */
export type ScreenStatus =
  | 'recorded engine output'
  | 'measured real data'
  | 'design-engine scores'
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
  'braTS-2021 segmentation + the engine, planned locally through api_server.py (uvicorn). '
  + 'On the deployed site the same fields render from the recorded evolution artifact.';

export const TABS: TabDef[] = [
  {
    id: 'overview',
    group: 'Start here',
    label: 'Overview',
    desc: 'What this is, what is measured, and what it refuses to claim',
    purpose:
      'Two studies live in this repository: a patient tumour twin (glioblastoma) and an AAV9-LAMP2B vector-design pipeline (Danon disease). This page states what each screen shows and which numbers are measurements.',
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
    id: 'capsid',
    group: 'Vector design (AAV9-LAMP2B)',
    label: 'Capsid & Sequence',
    desc: 'PDB 3UX1 structure, masked VP1 design',
    purpose:
      'The deposited AAV9 capsid assembled from its own symmetry operators, and the VP1 design the optimizer selected with its substitutions.',
    source:
      'danon/structure.py (PDB 3UX1) + api_server.run_full_pipeline, recorded as artifacts/structure.json and artifacts/pipeline.json',
    status: 'design-engine scores',
  },
  {
    id: 'phases',
    group: 'Vector design (AAV9-LAMP2B)',
    label: 'Design Gates',
    desc: '24 gate scores and their composition',
    purpose:
      'The 24 screening gates over the selected design, their pass/warn/fail composition, and the Horizon-2 module detail.',
    source: 'api_server.run_full_pipeline, recorded as artifacts/pipeline.json',
    status: 'design-engine scores',
  },
  {
    id: 'gene',
    group: 'Vector design (AAV9-LAMP2B)',
    label: 'Transgene Map',
    desc: 'LAMP2B construct, promoters, UTRs',
    purpose: 'The single-vector LAMP2B construct: promoters, UTRs, miRNA detargeting sites, polyA.',
    source: 'danon.construct_builder, recorded as artifacts/construct.json',
    status: 'design-engine scores',
  },
  {
    id: 'clinical',
    group: 'Vector design (AAV9-LAMP2B)',
    label: 'Pareto & Readiness',
    desc: 'Optimizer front, translational gates',
    purpose:
      'The NSGA-II Pareto front over six competing objectives, the modelled cardiomyocyte rescue, and the translational gates that are still unmet.',
    source: 'artifacts/pipeline.json (front) + danon.cell_simulator 8-ODE model (artifacts/clinical.json)',
    status: 'recorded engine output',
  },
  {
    id: 'lnp',
    group: 'Fabrication',
    label: 'LNP Formulation',
    desc: 'Scored lipid composition and size',
    purpose:
      'The best-scoring lipid nanoparticle formulation out of the scored candidates: composition, predicted size, and which parts are drawn rather than computed.',
    source: 'danon.lnp_generator, recorded as artifacts/lnp.json',
    status: 'design-engine scores',
  },
  {
    id: 'microfluidics',
    group: 'Fabrication',
    label: 'Microfluidics',
    desc: 'Y-junction flow telemetry',
    purpose:
      'Y-junction mixing hydraulics: Reynolds, Peclet, shear against the structural-integrity limit, and the TypeScript port checked against the Python engine.',
    source: 'danon.microfluidics_core, recorded as artifacts/microfluidics.json',
    status: 'recorded engine output',
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
  'deposited_structure',
];
