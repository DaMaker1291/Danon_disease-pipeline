/**
 * Shapes of `outputs_real/outcome_cards.json` (served from /evotwin/).
 *
 * Produced by `danon.outcome_export.export_outcome_cards`: exact per-day engine
 * state for the adaptive collateral-switch arm and its paired MTD control, on
 * real BraTS 2021 (TCGA/TCIA lineage) patient geometry.
 */

export interface DayReadout {
  day: number;
  burden_sensitive: number;
  burden_resistant: number;
  burden_total: number;
  resistant_fraction: number;
  dose_A_today: number;
  dose_B_today: number;
  dose_A_cumulative: number;
  dose_B_cumulative: number;
}

export interface DailySeries {
  seed: number;
  burden_total: number[];
  burden_sensitive: number[];
  burden_resistant: number[];
  resistant_fraction: number[];
  dose_A: number[];
  dose_B: number[];
}

export interface ReplicateReadout {
  seed: number;
  pfs_day: number | null;
  progressed: boolean;
  toxicity: number;
  day: DayReadout;
}

export interface ArmPayload {
  label: string;
  median_pfs_days: number;
  n_progressed: number;
  median_toxicity: number;
  daily: DailySeries;
  replicate_day_readouts: ReplicateReadout[];
}

export interface Day365Headline {
  day: number;
  burden_policy: number;
  burden_mtd: number;
  burden_policy_over_mtd: number | null;
  burden_reduction_vs_mtd: number | null;
  resistant_fraction_policy: number;
  resistant_fraction_mtd: number;
  progression_free_policy: boolean;
  progression_free_mtd: boolean;
  n_progressed_policy: number;
  n_progressed_mtd: number;
  median_pfs_policy: number;
  median_pfs_mtd: number;
  pfs_ratio: number | null;
  pfs_extension: number | null;
  toxicity_reduction: number | null;
  n_replicates: number;
  initial_burden: number;
  burden_fraction_of_initial_policy: number | null;
  burden_fraction_of_initial_mtd: number | null;
  pfs_extension_is_lower_bound: boolean;
  pfs_extension_label: string | null;
  clearance_note: string | null;
}

export interface OutcomeCard {
  case_id: string;
  patient_id: string;
  intake: { r0: number; r0_source: string; r0_note: string };
  geometry: { note: string; grid: number[]; tumour_voxels: number; voxel_mm: number };
  seeds: number[];
  horizon_days: number;
  arms: { policy: ArmPayload; mtd: ArmPayload };
  day_365: Day365Headline;
  notes: string[];
  data_provenance: Record<string, string>;
}

export interface OutcomeCardsPayload {
  title: string;
  generated_utc: string;
  model: string;
  rano_progression_fraction: number;
  switch_rule: string;
  engine_constants: Record<string, unknown> & { provenance: string };
  provenance: Record<string, string>;
  cases: OutcomeCard[];
}
