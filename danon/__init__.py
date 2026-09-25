from danon.config import DanonConfig, danon_config
from danon.aav_generator import DanonAAVGenerator, DanonAAVCandidate
from danon.lnp_generator import DanonLNPGenerator, DanonLNPCandidate
from danon.tropism_filter import DanonTropismFilter
from danon.safety_engine import DanonSafetyEngine, DanonSafetyProfile, print_global_regulatory_disclaimer
from danon.translational_readiness import (
    TranslationalReadinessEngine, TranslationalReadinessResult, PreclinicalValidationMilestones,
)
from danon.wetlab_lims_tracker import WetlabLIMSTracker, ConstructOrder, AssayResult
from danon.mouse_study_simulator import MouseStudySimulator, MouseStudyResult
from danon.dual_vector_moi_optimizer import DualVectorMOIOptimizer, DualVectorMoiDesign
from danon.nab_assay_simulator import NAbAssaySimulator, NAbAssayResult
from danon.immunosuppression_protocol import ImmunosuppressionProtocol, ImmunosuppressionAssessment
from danon.cell_simulator import DanonCellSimulator, CellSimulationResult
from danon.epitope_masker import EpitopeMasker, ChargeMaskDesign
from danon.stoichiometric_calc import StoichiometricCalculator, DecoyOptimizationResult
from danon.promoter_spec import PromoterSpecEngine, PromoterSpec
from danon.platform_validator import PlatformValidator, RegulatoryAssessment
from danon.data_ingress import DataIngressEngine, IngressResult
from danon.microfluidics_core import MicrofluidicsCore, FlowTelemetry, MicrofluidicConfig
from danon.opentrons_compiler import OpentronsCompiler, OpentronsProtocol, RobotType
from danon.dms_fitness import DMSFitnessLayer, DMSFitnessResult
from danon.solvation_energy import SolvationEnergyEngine, SolvationResult
from danon.smar_insulator import SMARInsulatorEngine, CpGOptimizationEngine, CpGDepletionReport, calculate_cpg_density
from danon.codon_elongation import CodonElongationEngine, CodonElongationResult
from danon.hla_decoupler import HLADecoupler, HLADecouplerResult
from danon.synthesis_guard import SynthesisGuard, SynthesisResult
from danon.plasmid_builder import PlasmidAssembler, PlasmidAssemblyReport, AssembledVector

# Multi-agent CVD / ischemic heart disease deployment layer.
# These are new abstractions that reuse the existing Danon discovery primitives
# (generators, tropism/safety scoring, Pareto optimizer, regulatory validator,
# cell simulator) but parameterize them per agent and per disease cohort.
from danon.multiagent.core import (
    MultiAgentConfig,
    DiseaseCohort,
    SubAgentSpec,
    SubAgentTask,
    TaskObjective,
    TaskStatus,
    AgentJudgement,
    AgentJudgementOutcome,
)
from danon.multiagent.runtime import (
    SubAgentRuntime,
    DeploymentPlan,
    DeploymentManifest,
    deploy_agents,
    LocalDeploymentRunner,
    TaskPriority,
)
from danon.multiagent.benchmarks import (
    CVDHeartDiseaseCohort,
    IschemicHeartDiseaseCohort,
    CohortEndpoints,
    CohortCriteria,
)
from danon.multiagent.aggregator import (
    ResultAggregator,
    AggregatedStudyReport,
    ParetoRankedResult,
    normalize_candidate_score,
)

# Patient personalization layer: turns a blood panel into a patient-specific
# regimen by wiring the decoy / HLA / dose primitives per patient.
from danon.personalization import (
    PatientBloodPanel,
    PatientAssessment,
    PatientPersonalizationEngine,
    PersonalizedRegimen,
    PersonalizationTier,
    IschemicIndication,
    Transgene,
    TransgeneSpec,
    Modality,
    TRANSGENE_CATALOG,
    print_personalization_disclaimer,
)

# First-principles hemodynamics: derived constants, similarity, rig design.
from danon.physics import (
    BenchRigSolver,
    Constant,
    CoronarySegment,
    DimensionlessGroups,
    Fluid,
    Provenance,
    RigSpec,
    SimilarityReport,
    audit_constants,
    characterize,
    coronary_presets,
    provenance_summary,
    stokes_einstein_diffusion,
)

# Derived molecular / cell biology: transport theory, bond mechanics, kinetics.
from danon.biology import (
    DerivedAdhesion,
    StenosisGeometry,
    biological_provenance,
    bell_off_rate,
    cohesion_strength_kpa,
    damkohler_number,
    derive_adhesion,
    embolization_risk,
    enzyme_mass_removal_rate_kg_s,
    hydrodynamic_drag_force_n,
    injected_dose_required,
    noyes_whitney_rate_kg_s,
    particles_required_for_fraction,
    receptor_concentration,
    single_pass_adhesion_fraction,
    time_to_clear_fraction,
)

# Master controller: blood profiling -> molecular targeting -> swarm regression.
from danon.pipeline import (
    BloodPanel,
    BloodProfileResult,
    BloodProfiler,
    TargetingResult,
    MolecularTargeting,
    BloodToCurePipeline,
    PipelineReport,
)

# Benchtop rig telemetry: real acquisition, calibration, sim-vs-bench agreement.
from danon.telemetry import (
    AcquisitionSession,
    BenchMetrics,
    LineProtocol,
    ProtocolError,
    ReplayBackend,
    RigBackend,
    RigChannel,
    RigConfig,
    RigTrace,
    SensorCalibration,
    SerialBackend,
    SimulatedRigBackend,
    ValidationReport,
    clearance_for_reduction,
    default_rig_config,
    detect_pulse_peaks,
    nominal_reduction_pct,
    run_validation,
    validate_against_prediction,
    waveform_stats,
)

# Vectorised swarm ABM: stenotic coronary flow, plaque regression, embolism.
from danon.swarm import (
    ArteryGeometry,
    SwarmConfig,
    SwarmResult,
    SwarmSimulator,
    SwarmCampaign,
    CampaignReport,
    correlation_report,
)

# Pre-event prevention engine: published ASCVD risk equations, guideline
# triage, and the CTT lives-saved calculation. This is the part of Danon that
# addresses the actual cause of most preventable cardiac deaths -- the therapy
# already exists and is generic; what is missing is detection and triage.
from danon.cure import (
    EvidenceStatus,
    CureStage,
    CureRoadmapResult,
    StageAssessment,
    PlaqueRegressionResult,
    ScarRemodelingResult,
    DeliveryFeasibilityResult,
    assess_cure_roadmap,
    plaque_regression,
    scar_remodeling,
    regeneration_safety,
    regeneration_regime_comparison,
    angiogenesis_response,
    delivery_feasibility,
    cure_provenance,
    print_cure_disclaimer,
)

# Organ-level cardiovascular digital twin: scan+panel initialization with
# conflict detection, treatment counterfactuals from the validated models, and
# a NOT_MODELLED refusal layer for treatments with no effect-size model.
# Oncology multimodal data twin: cross-modality fusion with discordance and
# CHIP rules, kinetics from the patient's own serial data, the molecular
# lead-time window, and therapy counterfactuals that refuse off-label anchors
# and per-patient resistance prediction.
from danon.oncotwin import (
    CTDNA_LEAD_TIME_IQR_MONTHS,
    CTDNA_LEAD_TIME_MEDIAN_MONTHS,
    CTDNA_SENSITIVITY_CEILING,
    CHIP_GENES,
    GrowthKinetics,
    ImagingRecord,
    LiquidBiopsy,
    MolecularRecurrenceOutlook,
    MultimodalPatientRecord,
    Oncotwin,
    OncotwinReport,
    SystemicRecord,
    TherapyCounterfactual,
    TherapySpec,
    TwinVerdict as OncotwinVerdict,
    THERAPY_CATALOGUE,
    fuse_modalities,
    growth_kinetics,
    molecular_recurrence_outlook,
    oncotwin_provenance,
    simulate_line_of_therapy,
)

from danon.twin import (
    ScanInputs,
    TwinPatient,
    TwinVerdict,
    TwinTreatment,
    CounterfactualResult,
    TwinInitializationReport,
    CardiovascularTwin,
    simulate_treatment,
    risk_category_of,
)

# Validation-tier module: the three experimental proxies for testing a model
# against real cancer, with the statistics for tier 1 (Cox/log-rank/C-index on
# held-out patients), tier 2 (log-space IC50 agreement with units-bias
# detection), and tier 3 (pre-registered wet-lab predictions with cell-line
# caveats). Judges models; trains none.
from danon.validation import (
    IC50Pair,
    IC50ValidationResult,
    ScaleCheck,
    ValidationTier,
    CohortValidationResult,
    WetLabPrediction,
    WetLabValidationPlan,
    concordance_index,
    design_wetlab_validation,
    log2_transform_check,
    logrank_two_group,
    patient_level_split,
    tier_overview,
    validate_ic50_predictions,
    validate_survival_stratification,
    validation_provenance,
)

# Predictive measurement: real Cox/C-index holdout on the TCGA-GBM clinical
# export, Monte Carlo policy stability under input noise, and physical
# invariants on recorded trajectories -- plus the explicit verdict that this
# engine has no per-patient discrimination to report.
from danon.predictive import (
    Concordance,
    CoxFit,
    EngineDiscrimination,
    HoldoutReport,
    InvariantViolation,
    NoiseLevelResult,
    PatientRow,
    StabilityReport,
    check_invariants,
    cox_fit,
    engine_discrimination_verdict,
    harrell_c_index,
    kfold_holdout_c_index,
    load_clinical_holdout,
    load_cohort_rows,
    monte_carlo_policy_stability,
    predictive_provenance,
    what_predictive_measurement_cannot_prove,
)

# Planted ground-truth recovery: the pipeline is run against constants it
# planted itself, so the estimators behind every real-data number can be checked
# against a known answer rather than only against biology nobody knows.
from danon.recovery import (
    FORWARD_EULER_STABILITY_LIMIT,
    REFERENCE_GBM_MEDIAN_OS_DAYS,
    CoxRecovery,
    IdentifiabilityDiagnostic,
    IdentifiabilitySweepRow,
    RecoveryFailed,
    TransportFit,
    TransportRecovery,
    analytic_point_source_field,
    assert_cox_recovery,
    assert_transport_recovery,
    assess_transport_identifiability,
    cox_recovery,
    infer_times,
    oracle_c_index,
    planted_cox_cohort,
    recovery_provenance,
    recover_transport_parameters,
    render_recovery_report,
    transport_identifiability_sweep,
    transport_recovery,
    what_recovery_cannot_prove,
)

# Temporal-leakage and out-of-sample audit trace: structural assertions that
# the outcome never entered the design, a permutation null that must collapse
# held-out discrimination to chance, seeded reproducibility, and an explicit
# non-estimable section for the per-patient prediction that cannot exist here.
from danon.audit import (
    AuditCheck,
    PermutationNull,
    audit_provenance,
    determinism_audit,
    design_columns_audit,
    fold_membership_audit,
    ingestion_boundary,
    ingestion_boundary_notes,
    per_patient_prediction_audit,
    permutation_leakage_audit,
    permutation_null_c_index,
    render_audit_trace,
    run_audit,
    what_the_audit_cannot_establish,
)

# Retrospective multimodal benchmark: which of the three external tests the
# reachable public data can actually support, with the spatial test's refusal
# derived from recorded probe counts rather than asserted in prose.
from danon.multimodal import (
    CANDIDATE_GLIOMA_COLLECTIONS,
    EVIDENCE_PATH as MULTIMODAL_EVIDENCE_PATH,
    MIN_FOLLOWUP_INTERVAL_DAYS,
    BenchmarkTest,
    ModalityEvidence,
    ModalityStatus,
    PairedControlSummary,
    SpatialDiceVerdict,
    benchmark_tests,
    build_evidence,
    load_evidence,
    measure_collection_timepoints,
    load_paired_control_cards,
    measured_discrimination,
    multimodal_provenance,
    probe_collection,
    probe_local_inventory,
    probe_tcia,
    render_multimodal_report,
    save_evidence,
    spatial_dice_verdict,
    summarize_paired_control,
    what_multimodal_benchmark_cannot_show,
)

# Longitudinal MRI availability manifests: the per-(patient, timepoint)
# sequence-flag tables that make stream 4 countable at all, with the census the
# stream record and the intake endpoints are computed from.
from danon.availability import (
    CFB_GBM_DOI,
    CFB_GBM_MANIFEST,
    CFB_GBM_SOURCE,
    MIN_PAIR_INTERVAL_WEEKS,
    REFERENCE_SEQUENCE as AVAILABILITY_REFERENCE_SEQUENCE,
    SEQUENCE_COLUMNS,
    AvailabilityCensus,
    AvailabilityRow,
    census_from_text,
    load_mri_availability_manifest,
    looks_like_availability_manifest,
    mri_availability_census,
)

# Anisotropic, heterogeneous transport: the mm^2/day <-> voxel^2/day bridge,
# the tissue-class diffusion field, and the variable-coefficient operator
# div(D(x) grad C) that reduces exactly to the engine's scalar Laplacian on
# constant D.
from danon.anisotropy import (
    CELL_MOTILITY_MM2_PER_DAY_RANGE,
    LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
    PHANTOM_VOXEL_MM,
    REAL_DECIMATION,
    REAL_VOXEL_MM,
    TISSUE_DRUG_D_MM2_PER_DAY,
    TISSUE_LABELS,
    DiffusionField,
    TransportContrast,
    UnitAuditRow,
    anisotropy_provenance,
    d_to_engine,
    d_to_physical,
    divergence_d_grad,
    divergence_diag_tensor_grad,
    measure_transport_contrast,
    tissue_classes_from_geometry,
    unit_audit,
    unit_audit_report,
    voxel_area_mm2,
    what_anisotropy_cannot_show,
)

# Evaluation targets: the three proposed 0.90+ estimands (spatial Dice,
# responder AUC, in-vitro R^2), each evaluated against data that exists and
# reported as MEASURED or as a specific refusal carrying its counts.
from danon.eval_targets import (
    EGFRVIII_IS_STRUCTURAL,
    GDSC2_FITTED_PATH,
    GDSC2_FITTED_SOURCE,
    RULE_BASED_POLICIES,
    DiceResult,
    EgfrviiiReachability,
    GdscAudit,
    ResponderLabelAudit,
    ResponderTarget,
    TargetStatus,
    TargetVerdict,
    audit_responder_label,
    dice_degenerate_case_report,
    dice_similarity,
    eval_targets_provenance,
    in_vitro_target,
    load_gdsc_glioma,
    measured_ceiling_report,
    probe_egfrviii_reachability,
    responder_target,
    rl_policy_target,
    solver_convergence_target,
    spatial_expansion_target,
    target_verdicts,
    what_targets_cannot_show,
)

# Covariate intake: the four upgrades proposed for the real cohort, each with a
# leakage gate and a MEASURED delta-C rather than an estimated one. Includes the
# correction to the censoring defect in `predictive.load_cohort_rows`, which had
# discarded every censored patient and inflated the progression C-index.
from danon.cohort_features import (
    BASE_COVARIATES as COHORT_BASE_COVARIATES,
    KPS_BASELINE_TIMEPOINTS,
    KPS_POST_BASELINE_TIMEPOINTS,
    MGMT_METHYLATION_CUTOFF,
    MGMT_PROBES,
    TREATMENT_RECORDS_PATH,
    CovariateEffect,
    KpsCoverage,
    LeakageContrast,
    MgmtBetaResult,
    MgmtReachability,
    TimepointCensus,
    TreatmentExposure,
    baseline_kps,
    cohort_feature_provenance,
    cohort_feature_report,
    covariate_effect,
    extract_mgmt_beta,
    kps_coverage,
    kps_leakage_contrast,
    kps_timepoint_census,
    parse_kps,
    post_baseline_kps,
    probe_mgmt_reachability,
    probe_treatment_records,
    treatment_exposure,
    what_covariate_intake_cannot_show,
)

# Real pharmacology intake: measured potency from ChEMBL with censoring and
# units preserved, curated molecular descriptors, and transport quantities
# derived by named methods -- each carrying its own provenance tag.
from danon.pharmacology import (
    ASSUMED_TORTUOSITY,
    GLIOMA_LINES,
    ActivityRecord,
    BloodBrainEstimate,
    DiffusionEstimate,
    Ec50Comparison,
    MoleculeRecord,
    PharmacologyUnavailable,
    PotencySummary,
    clark_logbb,
    compare_engine_ec50,
    engine_ec50_unit,
    engine_voxel_mm,
    fetch_activities,
    fetch_molecule,
    load_glioma_potency,
    normalize_to_nM,
    parse_activities,
    parse_molecule,
    pharmacology_provenance,
    stokes_einstein_diffusion,
    summarize_potency,
    what_pharmacology_cannot_prove,
)

# Claims audit: every pitch claim bound to a live check that reads the
# repository, so a poster cannot drift from the code that backs it.
from danon.claims import (
    CHECKS,
    CLAIMS,
    Claim,
    ClaimAudit,
    CheckResult,
    audit_claims,
    claims_provenance,
    render_claims_markdown,
    what_these_claims_cannot_establish,
)

from danon.prevention import (
    CTT_MORTALITY_RR_PER_MMOL_L,
    CTT_MVE_RR_CI95,
    CTT_MVE_RR_PER_MMOL_L,
    LDL_THRESHOLD_VERY_HIGH,
    LPA_HIGH_MG_DL,
    LPA_HIGH_NMOL_L,
    MG_DL_PER_MMOL_L_CHOLESTEROL,
    FHClassification,
    FHInputs,
    FHResult,
    PopulationPreventionCampaign,
    PopulationProfile,
    PopulationReport,
    PreventionAction,
    RaceStratum,
    RiskAssessment,
    RiskEnhancers,
    RiskEquation,
    RiskFactorProfile,
    Sex,
    TherapyRung,
    TreatmentBenefit,
    TriageDecision,
    ascvd_risk_10y,
    assess,
    dutch_lipid_clinic_score,
    ldl_exposure_years,
    prevention_constant_audit,
    prevention_provenance,
    print_prevention_disclaimer,
    project_benefit,
    risk_category,
    routine_ldl_reduction,
    triage,
)

# Spatial evolutionary treatment scheduling: MTD vs adaptive dose modulation
# vs collateral switching in a 3D tumour with drug-diffusion sanctuaries.
from danon.evosteer import (
    ADAPTIVE_TTP_MONTHS_MCRPC,
    ADAPTIVE_TTP_RATIO_ANCHOR,
    MEASURED_ADAPTIVE_MTD_RATIO_BAND,
    RANO_PROGRESSION_FRACTION,
    AdaptiveDosePolicy,
    ArmResult,
    BasePolicy,
    CloneParams,
    CollateralSwitchPolicy,
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    DrugParams,
    MTDPolicy,
    PolicyKind,
    PolicyState,
    SweepReport,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
    TrialReport,
    evosteer_provenance,
    make_gbm_phantom,
    make_policy,
    print_evosteer_disclaimer,
    run_trial,
    sanctuary_sequestration,
    threshold_sweep,
    what_this_model_cannot_know,
)

# EvoTwin: cohort-level evolutionary-trap analysis on the EvoSteer engine --
# Kaplan-Meier endpoints, patient-varying cohorts, immune-clearance hypothesis
# (CheckMate 143 anchors the GBM immunotherapy context as PROVEN_FAILED).
# Schedule compiler: turns engine trajectories into the two EvoTwin outputs
# (ParaView-loadable spatial fields; a measured adaptive dosing schedule with
# paired metrics and the switch-rule observability limit stated on every report).
from danon.schedule import (
    POSTER_PITCH_CLAIMS,
    DosingScheduleReport,
    PatientIntake,
    ScheduleEvent,
    compile_dosing_schedule,
    export_fields_vti,
    extract_schedule_events,
    format_schedule_text,
    poster_summary,
    print_schedule_disclaimer,
    read_fields_vti,
    VtiFields,
    schedule_provenance,
    what_this_model_cannot_know as schedule_what_this_model_cannot_know,
)

# Clinical observation layer: closes the schedule's documented #1 limit --
# the switch trigger (intratumoural resistant fraction) is not directly
# observable, so schedules are driven by an ABC filter over episodic CSF
# ctDNA VAF draws (Miller 2019 channel) through a receding-horizon controller.
from danon.observer import (
    CSF_LIQUID_BIOPSY_ANCHOR,
    ControllerRunResult,
    ObservationModel,
    ObservedSwitchController,
    ObserverVerdict,
    PatientObservations,
    Posterior,
    ProjectedState,
    StateEstimate,
    SurrogateParams,
    abc_filter,
    estimate_state,
    mtd_dose_log,
    null_dose_log,
    observer_provenance,
    project_resistant_fraction,
    run_observed_trial,
    surrogate_burden_series,
    what_this_model_cannot_know as observer_what_this_model_cannot_know,
)

# Real-imaging intake: NIfTI-1 I/O + TumorGeometry from a segmentation mask.
# Derived maps are labelled proxies; the null vasculature is declared
# sanctuary-free BY CONSTRUCTION on every PatientGeometry.
from danon.imaging import (
    NiftiHeader,
    PatientGeometry,
    read_nifti,
    write_nifti,
    from_nifti_mask,
    imaging_provenance,
    imaging_what_this_model_cannot_know,
)

# High-ceiling predictive targets: the engine emulator (one-shot holdout
# R^2 vs the simulator) and the real-patient MGMT-status classifier, both
# rendered as honest TargetVerdicts. Physics-derived features live in
# danon.emulator_features.
from danon.high_ceiling import (
    EmulatorDataset,
    EmulatorResult,
    MgmtStatusResult,
    FEATURE_NAMES,
    FEATURE_BOUNDS,
    emulator_target,
    mgmt_status_target,
    load_or_build_emulator_dataset,
)
from danon.emulator_features import (
    FEATURE_ORDER,
    feature_matrix,
    geometry_features,
)

# Digital-twin planning bridge: uploaded segmentation mask -> paired 365-day
# simulation -> schedule text, daily curves, VTI export, UI payload. Honesty
# labels travel with every field.
from danon.twin_plan import (
    TwinPlanResult,
    DEFAULT_R0,
    DEFAULT_R0_SOURCE,
    downsample_mask_to_budget,
    plan_from_mask,
)

# MGMT promoter methylation: the beta -> resistance bridge (r0 = Baseline
# _Resistance * (1 - beta)), its engine sweep, and the published-call concordance.
from danon.mgmt import (
    BASELINE_RESISTANCE as MGMT_BASELINE_RESISTANCE,
    BetaArmOutcome,
    MgmtBetaPanel,
    MgmtResistanceSweep,
    clone_params_for_mgmt_beta,
    load_mgmt_beta_panel,
    load_published_mgmt_calls,
    mgmt_call_concordance,
    mgmt_provenance,
    mgmt_resistance,
    mgmt_resistance_sweep,
    render_mgmt_report,
    resistance_audit,
)

# Clinical datastream intake: four routine hospital tests -> engine
# parameters, with identity guarantees and live-count refusals.
# Per-day spatial evolution: the paired 365-day movie (validated frames +
# ParaView .pvd time collection), the capture/recording layer, and the
# static geometry maps carried once per payload.
from danon.evolution_export import (
    EvolutionExport,
    FrameSet,
    capture_frames,
    export_evolution_frames,
    export_paired_evolution,
    save_payload,
    snapshot_days,
    static_geometry_maps,
    write_pvd,
    write_pvd_series,
)

from danon.clinical_twin import (
    CLEARANCE_GAMMA,
    DEFAULT_ALA_R0,
    DEFAULT_ALA_R0_SOURCE,
    KI67_REF,
    STREAM3_COHORT_PATH,
    TwinStreamState,
    assemble_digital_twin_state,
    build_stream3_clearance,
    build_stream3_vessels,
    clinical_stream_census,
    decimate_t1ce_to_engine_grid,
    stream1_initial_state,
    stream2_clones,
    stream2_growth_rate,
    stream3_report,
    stream4_velocity,
)

# Verification campaign: millions of elementary property checks (per-voxel
# invariants, dose/burden bounds, determinism probes) accumulated across
# paired runs -- resumable, with bootstrap CIs. Verifies the IMPLEMENTATION,
# not the biology.
from danon.campaign import (
    CampaignConfig,
    CampaignStats,
    bootstrap_ci_mean,
    campaign_provenance,
    check_final_fields,
    check_trajectory,
    determinism_probe,
    format_campaign_report,
    run_campaign,
    run_campaign_chunk,
    what_the_campaign_cannot_prove,
)

# Parametric robustness sweep: paired policy-vs-MTD runs over a grid of
# perturbations of every ASSUMED constant. Reports which constant the
# conclusion rests on -- not whether the constant is right.
from danon.sensitivity import (
    MATERIAL_EFFECT_RATIO,
    PerturbedOutcome,
    SensitivityReport,
    TARGETS as SENSITIVITY_TARGETS,
    perturb as perturb_parameters,
    run_sensitivity,
    sensitivity_provenance,
    what_sensitivity_cannot_prove as sensitivity_what_this_analysis_cannot_prove,
)

# Day-365 outcome cards + the dashboard build: exact per-day engine state
# (both arms, real geometry) exported as JSON and rendered into a
# self-contained HTML dashboard with the data embedded.
from danon.outcome_export import (
    ArmRecord,
    ReplicateRun,
    export_outcome_cards,
    load_cards,
    outcome_export_provenance,
    regenerate_ui_artifacts,
    render_dashboard_html,
    run_case_outcome,
)

# The real AAV9 capsid structure (PDB 3UX1): deposited C-alpha coordinates and
# the deposit's own symmetry operators, replacing a fabricated coordinate table
# that had been attributed to the wrong serotype.
from danon.structure import (
    AAV9_STRUCTURE_ID,
    AAV9_STRUCTURE_REFERENCE,
    AAV9_STRUCTURE_TITLE,
    CaResidue,
    CapsidAssembly,
    MISLABELED_STRUCTURE_ACTUAL,
    MISLABELED_STRUCTURE_ID,
    SymmetryOperator,
    VP1_OFFSET,
    VR_REGION_WINDOWS,
    axis_lines,
    classify_symmetry_operators,
    ensure_structure_file,
    load_aav9_capsid,
    parse_ca_residues,
    parse_symmetry_operators,
    structure_path,
    structure_provenance,
    vr_coordinate_map,
    what_this_structure_cannot_tell_you,
)

# Every artifact the UI renders, generated by running the real modules.
from danon.app_artifacts import (
    app_artifacts_provenance,
    build_app_artifacts,
    write_app_artifacts,
    what_these_artifacts_cannot_tell_you,
)

# Real-outcome external check: reconstruct the TCGA-GBM clinical cohorts from
# the GDC open export, anchor the ingestion against the published survival
# band, and compare the model's MTD arm against the observed progression
# curve (a timescale sanity check, explicitly not validation).
from danon.retrospective import (
    PUBLISHED_GBM_OS_BAND_DAYS,
    PUBLISHED_GBM_OS_SOURCE,
    AnchorCheck,
    ClinicalCase,
    ObservedCohort,
    PlausibilityCheck,
    anchor_check,
    load_campaign_log,
    load_tcga_gbm_clinical,
    model_curve_from_campaign_log,
    plausibility_check,
    reconstruct_cohort,
    retrospective_provenance,
    what_this_check_cannot_do as retrospective_what_this_check_cannot_do,
)

from danon.evotwin import (
    CHECKMATE143_HR,
    CHECKMATE143_MOS_MONTHS,
    GBM_RECURRENT_MEDIAN_OS_MONTHS,
    LONG_TERM_CONTROL_DAYS,
    NEOANTIGEN_RANGE,
    ArmSummary,
    CohortParams,
    EvotwinReport,
    PatientCohortSimulator,
    PatientOutcome,
    TrapReport,
    cohort_km_curve,
    evolutionary_trap_analysis,
    evotwin_provenance,
    kaplan_meier,
    km_at,
    km_median,
    print_evotwin_disclaimer,
    run_evotwin_cohort,
    what_this_model_cannot_know as evotwin_what_this_model_cannot_know,
)

__all__ = [
    # Original Danon pipeline
    "DanonConfig", "danon_config",
    "DanonAAVGenerator", "DanonAAVCandidate",
    "DanonLNPGenerator", "DanonLNPCandidate",
    "DanonTropismFilter",
    "DanonSafetyEngine", "DanonSafetyProfile",
    "EpitopeMasker", "ChargeMaskDesign",
    "StoichiometricCalculator", "DecoyOptimizationResult",
    "PromoterSpecEngine", "PromoterSpec",
    "PlatformValidator", "RegulatoryAssessment",
    "DataIngressEngine", "IngressResult",
    "MicrofluidicsCore", "FlowTelemetry", "MicrofluidicConfig",
    "OpentronsCompiler", "OpentronsProtocol", "RobotType",
    "DMSFitnessLayer", "DMSFitnessResult",
    "SolvationEnergyEngine", "SolvationResult",
    "SMARInsulatorEngine", "CpGOptimizationEngine", "CpGDepletionReport",
    "CodonElongationEngine", "CodonElongationResult",
    "HLADecoupler", "HLADecouplerResult",
    "SynthesisGuard", "SynthesisResult",
    "print_global_regulatory_disclaimer",
    "TranslationalReadinessEngine", "TranslationalReadinessResult", "PreclinicalValidationMilestones",
    "WetlabLIMSTracker", "ConstructOrder", "AssayResult",
    "MouseStudySimulator", "MouseStudyResult",
    "DualVectorMOIOptimizer", "DualVectorMoiDesign",

    # Cure-roadmap evidence map: the four biological requirements for REVERSING
    # IHD, each anchored to the human trial record (GLAGOV/ASTEROID for plaque
    # regression; CUPID 2 and miR-199a marked PROVEN_FAILED). Evidence map, not
    # a therapy.
    "EvidenceStatus",
    "CureStage",
    "CureRoadmapResult",
    "StageAssessment",
    "PlaqueRegressionResult",
    "ScarRemodelingResult",
    "DeliveryFeasibilityResult",
    "assess_cure_roadmap",
    "plaque_regression",
    "scar_remodeling",
    "regeneration_safety",
    "regeneration_regime_comparison",
    "angiogenesis_response",
    "delivery_feasibility",
    "cure_provenance",
    "print_cure_disclaimer",
    "NAbAssaySimulator", "NAbAssayResult",
    "ImmunosuppressionProtocol", "ImmunosuppressionAssessment",
    "DanonCellSimulator", "CellSimulationResult",
    "PlasmidAssembler", "PlasmidAssemblyReport", "AssembledVector",

    # Multi-agent deployment layer
    "MultiAgentConfig",
    "DiseaseCohort",
    "SubAgentSpec",
    "SubAgentTask",
    "TaskObjective",
    "TaskStatus",
    "AgentJudgement",
    "AgentJudgementOutcome",
    "SubAgentRuntime",
    "DeploymentPlan",
    "DeploymentManifest",
    "deploy_agents",
    "LocalDeploymentRunner",
    "TaskPriority",
    "CVDHeartDiseaseCohort",
    "IschemicHeartDiseaseCohort",
    "CohortEndpoints",
    "CohortCriteria",
    "ResultAggregator",
    "AggregatedStudyReport",
    "ParetoRankedResult",
    "normalize_candidate_score",

    # Patient personalization layer
    "PatientBloodPanel",
    "PatientAssessment",
    "PatientPersonalizationEngine",
    "PersonalizedRegimen",
    "PersonalizationTier",
    "IschemicIndication",
    "Transgene",
    "TransgeneSpec",
    "Modality",
    "TRANSGENE_CATALOG",
    "print_personalization_disclaimer",

    # Hemodynamics / rig design
    "BenchRigSolver",
    "Constant",
    "CoronarySegment",
    "DimensionlessGroups",
    "Fluid",
    "Provenance",
    "RigSpec",
    "SimilarityReport",
    "audit_constants",
    "characterize",
    "coronary_presets",
    "provenance_summary",
    "stokes_einstein_diffusion",

    # Derived biology
    "DerivedAdhesion",
    "StenosisGeometry",
    "biological_provenance",
    "bell_off_rate",
    "cohesion_strength_kpa",
    "damkohler_number",
    "derive_adhesion",
    "embolization_risk",
    "enzyme_mass_removal_rate_kg_s",
    "hydrodynamic_drag_force_n",
    "injected_dose_required",
    "noyes_whitney_rate_kg_s",
    "particles_required_for_fraction",
    "receptor_concentration",
    "single_pass_adhesion_fraction",
    "time_to_clear_fraction",

    # Master Blood-to-Cure pipeline
    "BloodPanel",
    "BloodProfileResult",
    "BloodProfiler",
    "TargetingResult",
    "MolecularTargeting",
    "BloodToCurePipeline",
    "PipelineReport",

    # Swarm ABM layer
    "ArteryGeometry",
    "SwarmConfig",
    "SwarmResult",
    "SwarmSimulator",
    "SwarmCampaign",
    "CampaignReport",
    "correlation_report",

    # Benchtop rig telemetry
    "AcquisitionSession",
    "BenchMetrics",
    "LineProtocol",
    "ProtocolError",
    "ReplayBackend",
    "RigBackend",
    "RigChannel",
    "RigConfig",
    "RigTrace",
    "SensorCalibration",
    "SerialBackend",
    "SimulatedRigBackend",
    "ValidationReport",
    "clearance_for_reduction",
    "default_rig_config",
    "detect_pulse_peaks",
    "nominal_reduction_pct",
    "run_validation",
    "validate_against_prediction",
    "waveform_stats",

    # Pre-event prevention engine
    "ascvd_risk_10y",
    "risk_category",
    "assess",
    "triage",
    "project_benefit",
    "routine_ldl_reduction",
    "ldl_exposure_years",
    "dutch_lipid_clinic_score",
    "prevention_provenance",
    "prevention_constant_audit",
    "print_prevention_disclaimer",
    "RiskEquation",
    "RiskFactorProfile",
    "RiskEnhancers",
    "RiskAssessment",
    "Sex",
    "RaceStratum",
    "TherapyRung",
    "PreventionAction",
    "TreatmentBenefit",
    "TriageDecision",
    "FHInputs",
    "FHResult",
    "FHClassification",
    "PopulationProfile",
    "PopulationPreventionCampaign",
    "PopulationReport",
    "CTT_MVE_RR_PER_MMOL_L",
    "CTT_MVE_RR_CI95",
    "CTT_MORTALITY_RR_PER_MMOL_L",
    "LDL_THRESHOLD_VERY_HIGH",
    "LPA_HIGH_MG_DL",
    "LPA_HIGH_NMOL_L",
    "MG_DL_PER_MMOL_L_CHOLESTEROL",

    # Cardiovascular digital twin
    "ScanInputs",
    "TwinPatient",
    "TwinVerdict",
    "TwinTreatment",
    "CounterfactualResult",
    "TwinInitializationReport",
    "CardiovascularTwin",
    "simulate_treatment",
    "risk_category_of",

    # Oncology multimodal data twin
    "LiquidBiopsy",
    "ImagingRecord",
    "SystemicRecord",
    "MultimodalPatientRecord",
    "Oncotwin",
    "OncotwinReport",
    "GrowthKinetics",
    "MolecularRecurrenceOutlook",
    "TherapySpec",
    "TherapyCounterfactual",
    "THERAPY_CATALOGUE",
    "OncotwinVerdict",
    "CHIP_GENES",
    "CTDNA_LEAD_TIME_MEDIAN_MONTHS",
    "CTDNA_LEAD_TIME_IQR_MONTHS",
    "CTDNA_SENSITIVITY_CEILING",
    "fuse_modalities",
    "growth_kinetics",
    "molecular_recurrence_outlook",
    "simulate_line_of_therapy",
    "oncotwin_provenance",

    # Validation tiers (cohort / pharmacology / wet-lab)
    "ValidationTier",
    "ScaleCheck",
    "CohortValidationResult",
    "IC50Pair",
    "IC50ValidationResult",
    "WetLabPrediction",
    "WetLabValidationPlan",
    "concordance_index",
    "logrank_two_group",
    "log2_transform_check",
    "patient_level_split",
    "validate_survival_stratification",
    "validate_ic50_predictions",
    "design_wetlab_validation",
    "tier_overview",
    "validation_provenance",

    # EvoSteer: spatial evolutionary treatment scheduling
    "ADAPTIVE_TTP_MONTHS_MCRPC", "ADAPTIVE_TTP_RATIO_ANCHOR",
    "MEASURED_ADAPTIVE_MTD_RATIO_BAND", "RANO_PROGRESSION_FRACTION",
    "MTDPolicy", "AdaptiveDosePolicy", "CollateralSwitchPolicy",
    "BasePolicy", "PolicyKind", "PolicyState",
    "CloneParams", "DrugParams", "DEFAULT_CLONES", "DEFAULT_DRUGS",
    "TumorGeometry", "TumorEvolutionSimulator", "SimulationTrajectory",
    "TrialReport", "ArmResult", "SweepReport",
    "make_gbm_phantom", "make_policy", "run_trial", "threshold_sweep",
    "sanctuary_sequestration", "evosteer_provenance",
    "print_evosteer_disclaimer", "what_this_model_cannot_know",

    # EvoTwin: cohort evolutionary-trap layer
    "CHECKMATE143_MOS_MONTHS", "CHECKMATE143_HR",
    "GBM_RECURRENT_MEDIAN_OS_MONTHS", "LONG_TERM_CONTROL_DAYS", "NEOANTIGEN_RANGE",
    "CohortParams", "PatientCohortSimulator", "PatientOutcome",
    "ArmSummary", "EvotwinReport", "TrapReport",
    "kaplan_meier", "cohort_km_curve", "km_at", "km_median",
    "run_evotwin_cohort", "evolutionary_trap_analysis", "evotwin_provenance",
    "print_evotwin_disclaimer", "evotwin_what_this_model_cannot_know",

    # Schedule compiler: engine runs -> clinic-shaped deliverables
    "PatientIntake", "ScheduleEvent", "DosingScheduleReport",
    "compile_dosing_schedule", "extract_schedule_events", "export_fields_vti",
    "read_fields_vti", "VtiFields",
    "TwinPlanResult", "DEFAULT_R0", "DEFAULT_R0_SOURCE", "downsample_mask_to_budget", "plan_from_mask",
    "assemble_digital_twin_state", "build_stream3_vessels", "build_stream3_clearance",
    "stream1_initial_state", "stream2_clones", "stream2_growth_rate", "stream3_report",
    "stream4_velocity", "clinical_stream_census", "decimate_t1ce_to_engine_grid",
    "KI67_REF", "CLEARANCE_GAMMA", "STREAM3_COHORT_PATH", "TwinStreamState",
    "DEFAULT_ALA_R0", "DEFAULT_ALA_R0_SOURCE",
    "MGMT_BASELINE_RESISTANCE", "BetaArmOutcome", "MgmtBetaPanel", "MgmtResistanceSweep",
    "clone_params_for_mgmt_beta", "load_mgmt_beta_panel", "load_published_mgmt_calls",
    "mgmt_call_concordance", "mgmt_provenance", "mgmt_resistance", "mgmt_resistance_sweep",
    "render_mgmt_report", "resistance_audit",
    "EmulatorDataset", "EmulatorResult", "MgmtStatusResult", "FEATURE_NAMES", "FEATURE_BOUNDS",
    "emulator_target", "mgmt_status_target", "load_or_build_emulator_dataset",
    "FEATURE_ORDER", "feature_matrix", "geometry_features",
    "format_schedule_text", "schedule_provenance",
    "print_schedule_disclaimer", "schedule_what_this_model_cannot_know",
    "POSTER_PITCH_CLAIMS", "poster_summary",

    # Imaging intake + verification campaign
    "NiftiHeader", "PatientGeometry", "read_nifti", "write_nifti",
    "from_nifti_mask", "imaging_provenance", "imaging_what_this_model_cannot_know",
    "CampaignConfig", "CampaignStats", "run_campaign", "run_campaign_chunk",
    "check_trajectory", "check_final_fields", "determinism_probe",
    "bootstrap_ci_mean", "format_campaign_report", "campaign_provenance",

    # Parametric robustness: paired sweeps of every ASSUMED constant. Says
    # which constant the conclusion depends on; cannot say the constant is right.
    "MATERIAL_EFFECT_RATIO", "SensitivityReport", "PerturbedOutcome",
    "SENSITIVITY_TARGETS", "run_sensitivity", "sensitivity_provenance",
    "sensitivity_what_this_analysis_cannot_prove", "perturb_parameters",

    # External check against REAL clinical outcomes (TCGA-GBM via GDC):
    # ingestion anchor vs published GBM survival + control-arm plausibility.
    "ClinicalCase", "ObservedCohort", "AnchorCheck", "PlausibilityCheck",
    "load_tcga_gbm_clinical", "reconstruct_cohort", "anchor_check",
    "plausibility_check", "model_curve_from_campaign_log", "load_campaign_log",
    "retrospective_provenance", "retrospective_what_this_check_cannot_do",
    "PUBLISHED_GBM_OS_BAND_DAYS", "PUBLISHED_GBM_OS_SOURCE",

    # Planted ground-truth recovery (known-answer tests for the estimators)
    "FORWARD_EULER_STABILITY_LIMIT", "REFERENCE_GBM_MEDIAN_OS_DAYS", "CoxRecovery",
    "IdentifiabilityDiagnostic", "IdentifiabilitySweepRow", "RecoveryFailed",
    "TransportFit", "TransportRecovery", "analytic_point_source_field",
    "assert_cox_recovery", "assert_transport_recovery",
    "assess_transport_identifiability", "cox_recovery", "infer_times",
    "oracle_c_index", "planted_cox_cohort", "recovery_provenance",
    "recover_transport_parameters", "render_recovery_report",
    "transport_identifiability_sweep", "transport_recovery",
    "what_recovery_cannot_prove",

    # Leakage / out-of-sample audit trace
    "AuditCheck", "PermutationNull", "audit_provenance", "determinism_audit",
    "design_columns_audit", "fold_membership_audit", "ingestion_boundary",
    "ingestion_boundary_notes", "per_patient_prediction_audit",
    "permutation_leakage_audit", "permutation_null_c_index",
    "render_audit_trace", "run_audit", "what_the_audit_cannot_establish",

    # Retrospective multimodal benchmark (availability evidence + derived refusal)
    "CANDIDATE_GLIOMA_COLLECTIONS", "MULTIMODAL_EVIDENCE_PATH", "BenchmarkTest",
    "ModalityEvidence", "ModalityStatus", "PairedControlSummary",
    "SpatialDiceVerdict", "benchmark_tests",
    "build_evidence", "load_evidence", "load_paired_control_cards",
    "measure_collection_timepoints", "MIN_FOLLOWUP_INTERVAL_DAYS",
    "CFB_GBM_DOI", "CFB_GBM_MANIFEST", "CFB_GBM_SOURCE", "MIN_PAIR_INTERVAL_WEEKS",
    "AVAILABILITY_REFERENCE_SEQUENCE", "SEQUENCE_COLUMNS", "AvailabilityCensus",
    "AvailabilityRow", "census_from_text", "load_mri_availability_manifest",
    "looks_like_availability_manifest", "mri_availability_census",
    "measured_discrimination", "multimodal_provenance", "probe_collection",
    "probe_local_inventory", "probe_tcia", "render_multimodal_report",
    "save_evidence", "spatial_dice_verdict", "summarize_paired_control",
    "what_multimodal_benchmark_cannot_show",

    # Real pharmacology (ChEMBL/PubChem measured data + named-method derivation)
    "ASSUMED_TORTUOSITY", "GLIOMA_LINES", "ActivityRecord",
    "BloodBrainEstimate", "DiffusionEstimate", "Ec50Comparison",
    "MoleculeRecord", "PharmacologyUnavailable", "PotencySummary",
    "clark_logbb", "compare_engine_ec50", "engine_ec50_unit", "engine_voxel_mm",
    "fetch_activities", "fetch_molecule", "load_glioma_potency",
    "normalize_to_nM", "parse_activities", "parse_molecule",
    "pharmacology_provenance", "stokes_einstein_diffusion",
    "summarize_potency", "what_pharmacology_cannot_prove",

    # Evaluation targets (proposed 0.90+ estimands, each measured or refused)
    "EGFRVIII_IS_STRUCTURAL", "GDSC2_FITTED_PATH", "GDSC2_FITTED_SOURCE",
    "RULE_BASED_POLICIES", "DiceResult", "EgfrviiiReachability", "GdscAudit",
    "ResponderLabelAudit", "ResponderTarget", "TargetStatus", "TargetVerdict",
    "audit_responder_label", "dice_degenerate_case_report", "dice_similarity",
    "eval_targets_provenance", "in_vitro_target", "load_gdsc_glioma",
    "measured_ceiling_report", "probe_egfrviii_reachability", "responder_target",
    "rl_policy_target", "solver_convergence_target", "spatial_expansion_target",
    "target_verdicts", "what_targets_cannot_show",

    # Anisotropic transport (unit bridge + variable-coefficient operator)
    "CELL_MOTILITY_MM2_PER_DAY_RANGE", "LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY",
    "PHANTOM_VOXEL_MM", "REAL_DECIMATION", "REAL_VOXEL_MM",
    "TISSUE_DRUG_D_MM2_PER_DAY", "TISSUE_LABELS",
    "DiffusionField", "TransportContrast", "UnitAuditRow",
    "anisotropy_provenance", "d_to_engine", "d_to_physical",
    "divergence_d_grad", "divergence_diag_tensor_grad",
    "measure_transport_contrast", "tissue_classes_from_geometry",
    "unit_audit", "unit_audit_report", "voxel_area_mm2",
    "what_anisotropy_cannot_show",

    # Covariate intake (the four proposed cohort upgrades, each measured) and
    # the corrected censoring in the cohort loader
    "COHORT_BASE_COVARIATES", "KPS_BASELINE_TIMEPOINTS", "KPS_POST_BASELINE_TIMEPOINTS",
    "MGMT_METHYLATION_CUTOFF", "MGMT_PROBES", "TREATMENT_RECORDS_PATH",
    "CovariateEffect", "KpsCoverage", "LeakageContrast", "MgmtBetaResult",
    "MgmtReachability", "TimepointCensus", "TreatmentExposure",
    "baseline_kps", "cohort_feature_provenance", "cohort_feature_report",
    "covariate_effect", "extract_mgmt_beta", "kps_coverage",
    "kps_leakage_contrast", "kps_timepoint_census", "parse_kps",
    "post_baseline_kps", "probe_mgmt_reachability", "probe_treatment_records",
    "treatment_exposure", "what_covariate_intake_cannot_show",

    # Claims audit (pitch sentence <-> code, verified live)
    "CHECKS", "CLAIMS", "Claim", "ClaimAudit", "CheckResult", "audit_claims",
    "claims_provenance", "render_claims_markdown",
    "what_these_claims_cannot_establish",

    # Predictive measurement (held-out discrimination, noise robustness,
    # invariants) and the explicit no-C-index verdict
    "Concordance", "CoxFit", "EngineDiscrimination", "HoldoutReport",
    "InvariantViolation", "NoiseLevelResult", "PatientRow", "StabilityReport",
    "check_invariants", "cox_fit", "engine_discrimination_verdict",
    "harrell_c_index", "kfold_holdout_c_index", "load_clinical_holdout",
    "load_cohort_rows", "monte_carlo_policy_stability", "predictive_provenance",
    "what_predictive_measurement_cannot_prove",

    # Real AAV9 structure intake (PDB 3UX1) and the UI's artifact generators
    "AAV9_STRUCTURE_ID", "AAV9_STRUCTURE_REFERENCE", "AAV9_STRUCTURE_TITLE",
    "CaResidue", "CapsidAssembly", "SymmetryOperator", "VP1_OFFSET",
    "VR_REGION_WINDOWS", "MISLABELED_STRUCTURE_ID", "MISLABELED_STRUCTURE_ACTUAL",
    "axis_lines", "classify_symmetry_operators", "ensure_structure_file",
    "load_aav9_capsid", "parse_ca_residues", "parse_symmetry_operators",
    "structure_path", "structure_provenance", "vr_coordinate_map",
    "what_this_structure_cannot_tell_you", "build_app_artifacts",
    "write_app_artifacts", "app_artifacts_provenance",
    "what_these_artifacts_cannot_tell_you",

    # Day-365 outcome cards + dashboard (the UI's data contract)
    "ArmRecord", "ReplicateRun", "run_case_outcome", "export_outcome_cards",
    # evolution_export
    "EvolutionExport", "FrameSet", "capture_frames", "export_evolution_frames",
    "export_paired_evolution", "save_payload", "snapshot_days",
    "static_geometry_maps", "write_pvd", "write_pvd_series",
    "load_cards", "render_dashboard_html", "regenerate_ui_artifacts",
    "outcome_export_provenance",

    # Clinical observation layer (CSF VAF -> state estimate -> schedule)
    "CSF_LIQUID_BIOPSY_ANCHOR", "ControllerRunResult", "ObservationModel",
    "ObservedSwitchController", "ObserverVerdict", "PatientObservations",
    "Posterior", "ProjectedState", "StateEstimate", "SurrogateParams",
    "abc_filter", "estimate_state", "mtd_dose_log", "null_dose_log",
    "observer_provenance", "project_resistant_fraction", "run_observed_trial",
    "surrogate_burden_series", "observer_what_this_model_cannot_know",
    "what_the_campaign_cannot_prove",
]
