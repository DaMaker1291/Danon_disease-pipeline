/**
 * Types for the tumour-board artifact (`artifacts/tumorboard.json`).
 *
 * The artifact carries three kinds of field and never mixes them: a RECORDED
 * value with the run that produced it, a REFUSAL with the reason this
 * repository has no such value, and the figure a dashboard mock-up showed for a
 * field that is being refused. The types say so: `RecordedField`, `RefusedField`
 * and `Refusal` are separate shapes, so a view cannot read a refused field as
 * if it were a measured one.
 */

export interface RecordedField {
  field: string;
  value: string | number;
  source: string;
}

export interface RefusedField {
  field: string;
  status: string;
  detail: string;
  would_come_from: string;
  quoted_in_the_dashboard_mockup: string;
}

export interface ArmOutcome {
  label: string;
  median_pfs_days: number;
  n_progressed: number;
  n_replicates: number;
  median_toxicity: number;
  cumulative_drug_A: number;
  cumulative_drug_B: number;
}

export interface Refusal {
  shown_in_the_mockup: string;
  why_refused: string;
  what_is_recorded_instead: unknown;
}

export interface EngineSourceHash {
  algorithm: string;
  scope: string;
  files: { path: string; sha256: string; bytes: number }[];
  combined_sha256: string;
  verify_with: string;
}

export interface TumorBoardData {
  generated_utc: string;
  model: string;
  case: {
    case_id: string;
    patient_id: string;
    grid: number[];
    voxel_mm: number;
    tumour_voxels: number;
    geometry_note: string;
    r0: number;
    r0_source: string;
  };
  patient_panel: {
    recorded: RecordedField[];
    not_recorded: RefusedField[];
  };
  volume_panel: {
    note: string;
    frame_days?: number[];
    grid?: number[];
    spacing_mm?: number;
    arm_frames: Record<string, { seed: number; n_frames: number; pfs_day: number | null; progressed: boolean }>;
  };
  kinetics_panel: {
    days?: number[];
    resistant_fraction: Record<string, number[]>;
    tipping_day: Record<string, number | null>;
    note: string;
  };
  dosing_panel: {
    switch_rule: string;
    rano_progression_fraction: number;
    arms: Record<string, ArmOutcome>;
    day_365: {
      burden_policy: number;
      burden_mtd: number;
      burden_reduction_vs_mtd: number;
      progression_free_policy: boolean;
      progression_free_mtd: boolean;
      toxicty_note: string;
    };
    pfs_extension_is_lower_bound: boolean;
    no_recurrence_date: string;
  };
  audit_panel: {
    claims: { passed: number; total: number; failed: number; failed_ids: string[]; source: string };
    test_functions: number;
    in_classes: number;
    counted_by: string;
    note: string;
    engine_source_hash: EngineSourceHash;
    acceleration: {
      laplacian_ratio_by_grid: Record<string, number>;
      divergence_ratio_by_grid: Record<string, number>;
      day_ratio_scalar: number;
      day_ratio_heterogeneous: number;
      batch_per_replicate_ratio: number;
      batch_one_stream_ratio: number;
      process_ratio_by_workers: Record<string, number>;
      bit_identity_holds: { fields: boolean; trajectory: boolean };
      measured_at: string;
      statistic: string;
      not_measured: string[];
    };
    no_regulatory_status: string;
  };
  refusals: Refusal[];
}
