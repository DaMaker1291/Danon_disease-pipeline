"""
Oncology multimodal data twin: the deterministic skeleton of the pan-cancer
fusion architecture.

WHAT THIS IS
------------

The user's framing is right: the buildable frontier is FUSING data streams
(imaging, liquid biopsy, clinical) rather than simulating 37 trillion cells.
This module implements the parts of that architecture that can be built with
published constants and the patient's own measurements -- and it is explicit
about the part that cannot be built here.

Three honest layers:

1. **FUSION (buildable today, deterministic).** Cross-modality consistency:
   ctDNA rising while imaging is stable is a real clinical state (molecular
   recurrence, the lead-time window); lesion growth with falling ctDNA
   suggests pseudoprogression or necrosis; CHIP variants without matched
   leukocyte sequencing make a tumour-fraction call unreliable. This is the
   "connect the three languages" value, implemented as verifiable rules
   instead of learned weights.

2. **KINETICS (buildable today, mechanistic).** Serial ctDNA and serial
   lesion diameters are exponential-ish processes. Fitting log-linear growth
   to the patient's OWN measurements gives a tumour volume doubling time and
   an expected trajectory. Therapy counterfactuals model clearance
   (exponential decay) and regrowth at resistance, anchored to published
   cohort medians for time-to-progression.

3. **THE LEARNED PATTERN-MATCHER (NOT here, and we will not fake it).** The
   user's "match the patient against a learned universe of 10 million
   patients" requires 10 million patients, the compute to train on them, and
   the governance to deploy. None of that is in this repository. Every number
   below is either a published constant, derived from the patient's own
   serial data, or flagged ASSUMED. Where a foundation model would
   eventually answer, this module says so and returns what it can.

THE TWO REFUSALS THAT MATTER
----------------------------

* **No per-patient resistance mechanism.** "If we give Drug A the tumour will
  mutate resistance in 90 days" is not a computable claim. Resistance
  emergence is stochastic; what exists is cohort-level mechanism
  distributions at progression (e.g. on osimertinib, MET amplification in
  ~15% of cases at failure, C797S in ~7-25%). The model returns the
  distribution as background data and refuses to name THIS tumour's escape.

* **No trajectory from a single timepoint.** Kinetic claims require serial
  measurements. One ctDNA value says "detected"; only a series says
  "doubling in N weeks". The model refuses the second from the first.

References (embedded at the constants that use them):
  - ctDNA lead time: Abidoye et al. 2025 review of MRD testing in
    colorectal cancer -- median 5.3 months (IQR 3.0-16.4, max 28.7) before
    radiographic recurrence.
  - CHIP confound: Razavi et al. Nat Med 2019; standard mitigation is
    matched leukocyte sequencing (tumour-informed panels reduce it further).
  - Cohort PFS anchors: FLAURA (osimertinib, mPFS 18.9 mo), KEYNOTE-006
    (pembrolizumab, mPFS ~8.1 mo), FOLFOX in mCRC (mPFS ~9 mo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: Genes in which clonal-haematopoiesis variants masquerade as tumour ctDNA.
#: A variant in one of these WITHOUT matched leukocyte sequencing is not
#: attributable to the tumour.
CHIP_GENES = (
    "DNMT3A", "TET2", "ASXL1", "TP53", "JAK2", "SF3B1", "GNB1", "PPM1D",
)

#: Tumour types ranked by ctDNA shedding propensity (ordinal, not numbers:
#: reported shedding rates span orders of magnitude). Low-shedders are the
#: honesty constraint on any "ctDNA-negative means disease-free" claim.
SHEDDING_RANK = {
    "high": ("lung_nsclc", "colorectal", "melanoma", "urothelial", "ovarian"),
    "moderate": ("breast", "pancreatic", "gastric", "renal", "head_neck"),
    "low": ("glioma", "prostate_early", "thyroid_papillary"),
}
#: Glioma ctDNA is largely absent from PLASMA; the accessible fluid is CSF.
CSF_ONLY_TUMOURS = ("glioma",)

#: Serial ctDNA sensitivity ceiling for MRD detection: a material fraction of
#: patients never show detectable ctDNA even at relapse (non-shedders), so
#: the assay's negative predictive value is bounded below 100%. From pooled
#: MRD analyses in CRC: ~88-92% cumulative sensitivity to relapse.
CTDNA_SENSITIVITY_CEILING = 0.90


@dataclass
class LiquidBiopsy:
    """One ctDNA measurement.

    ``tumor_fraction_pct`` is the tumour-informed variant-allele burden from
    a personalised (tumour-informed) panel, which is what modern MRD assays
    report. A tissue-naive panel's "ctDNA positive" call is treated with
    suspicion when CHIP genes are involved and WBC matching was skipped.
    """

    day: int                          # days since baseline (e.g. surgery)
    tumor_fraction_pct: Optional[float] = None   # None = below assay LOQ
    below_loq: bool = False
    assay_loq_pct: float = 0.005
    matched_wbc_sequencing: bool = False
    chip_genes_detected: Tuple[str, ...] = ()

    def detected(self) -> bool:
        return not self.below_loq and self.tumor_fraction_pct is not None


@dataclass
class ImagingRecord:
    day: int
    lesion_longest_diameter_mm: Optional[float] = None
    lesion_count: Optional[int] = None
    scan_type: str = "ct"             # 'ct' | 'mri' | 'pet'
    radiologist_impression: str = "none"   # none | stable | progression | pseudoprogression


@dataclass
class SystemicRecord:
    """Clinical context. Deliberately light: this twin's dynamics come from
    the patient's serial measurements, not from demographic regression."""

    age_years: float = 60.0
    ecog: int = 0
    tumour_type: str = "lung_nsclc"
    stage: str = "IIIB"


@dataclass
class MultimodalPatientRecord:
    """The fusion input: the three languages, as the clinicians hold them."""

    patient_id: str
    clinical: SystemicRecord = field(default_factory=SystemicRecord)
    imaging: List[ImagingRecord] = field(default_factory=list)
    biopsies: List[LiquidBiopsy] = field(default_factory=list)

    def sorted_biopsies(self) -> List[LiquidBiopsy]:
        return sorted(self.biopsies, key=lambda b: b.day)

    def sorted_imaging(self) -> List[ImagingRecord]:
        return sorted(self.imaging, key=lambda s: s.day)


# ---------------------------------------------------------------------------
# Fusion: cross-modality consistency
# ---------------------------------------------------------------------------

@dataclass
class FusionReport:
    """The single predictive map the fragmentation argument is about."""

    discordances: List[str] = field(default_factory=list)
    chip_warnings: List[str] = field(default_factory=list)
    clinical_states: List[str] = field(default_factory=list)
    usable_biopsy_timepoints: int = 0
    usable_imaging_timepoints: int = 0
    kinetic_fit_possible: bool = False

    @property
    def is_molecular_recurrence(self) -> bool:
        return any("molecular recurrence" in s for s in self.clinical_states)

    @property
    def possible_pseudoprogression(self) -> bool:
        return any("pseudoprogression" in s for s in self.discordances)


def fuse_modalities(record: MultimodalPatientRecord) -> FusionReport:
    """Cross-modality consistency rules. These encode real clinical
    interpretation patterns as verifiable logic -- the fragment-joining the
    foundation model would learn, made explicit here."""
    report = FusionReport()
    biopsies = record.sorted_biopsies()
    imaging = record.sorted_imaging()
    report.usable_biopsy_timepoints = sum(1 for b in biopsies if b.detected() or b.below_loq)
    report.usable_imaging_timepoints = len(imaging)

    # --- CHIP attribution --------------------------------------------------
    for b in biopsies:
        if b.chip_genes_detected and not b.matched_wbc_sequencing:
            report.chip_warnings.append(
                f"day {b.day}: variants in {', '.join(b.chip_genes_detected)} "
                "without matched leukocyte sequencing -- the tumour-fraction "
                "call is not fully attributable to the tumour (Razavi 2019)."
            )

    # --- serial trajectories -> clinical states -----------------------------
    if len(biopsies) >= 2 and len(imaging) >= 1:
        last_bio = biopsies[-1]
        last_img = imaging[-1]
        rising = (
            last_bio.detected()
            and biopsies[0].detected()
            and last_bio.tumor_fraction_pct > biopsies[0].tumor_fraction_pct
        )
        stable_imaging = last_img.radiologist_impression in ("stable", "none")

        if rising and stable_imaging and last_img.lesion_longest_diameter_mm is not None:
            report.discordances.append(
                f"ctDNA rising ({biopsies[0].tumor_fraction_pct:.3f}% -> "
                f"{last_bio.tumor_fraction_pct:.3f}%) with stable imaging "
                f"({last_img.lesion_longest_diameter_mm:.1f} mm): treat the "
                "molecular signal as leading; imaging lags in the lead-time "
                "window."
            )
        if rising and last_img.lesion_longest_diameter_mm is None and last_bio.detected():
            report.clinical_states.append(
                "molecular recurrence: ctDNA detected after clearance with no "
                "radiographic lesion yet -- the lead-time window is open."
            )
        if (
            not rising
            and last_bio.detected()
            and last_img.radiologist_impression == "progression"
        ):
            report.discordances.append(
                "imaging progression with falling ctDNA: consider "
                "pseudoprogression or tumour necrosis (common under "
                "immunotherapy); confirm before switching therapy."
            )

    # --- single-timepoint honesty -------------------------------------------
    detected_series = [b for b in biopsies if b.detected()]
    if len(detected_series) >= 2:
        report.kinetic_fit_possible = True
    else:
        report.clinical_states.append(
            "fewer than two detectable ctDNA timepoints: growth kinetics are "
            "not identifiable. A single value supports detection status only, "
            "not trajectory."
        )

    # --- shedding ceiling ----------------------------------------------------
    tumour = record.clinical.tumour_type
    if tumour in CSF_ONLY_TUMOURS and biopsies:
        report.chip_warnings.append(
            f"{tumour}: plasma ctDNA is characteristically near-absent; the "
            "accessible fluid is CSF. A plasma-negative call carries almost "
            "no information for this tumour type."
        )
    elif tumour in SHEDDING_RANK["low"] and all(
        not b.detected() for b in biopsies
    ):
        report.discordances.append(
            f"{tumour} is a low-shedding tumour type and ctDNA was not "
            "detected: the sensitivity ceiling (~"
            f"{CTDNA_SENSITIVITY_CEILING:.0%} cumulative) means a negative "
            "assay cannot be read as absence of disease."
        )

    return report


# ---------------------------------------------------------------------------
# Kinetics from the patient's own serial data
# ---------------------------------------------------------------------------

@dataclass
class GrowthKinetics:
    """Volume doubling time estimated from the patient's own series.

    Two independent estimators are provided where data allow: lesion
    diameters (cubic volume growth) and serial ctDNA fractions (direct
    exponential fit). Agreement between them is a validation the model
    reports rather than assumes."""

    ct_dna_doubling_days: Optional[float] = None
    imaging_doubling_days: Optional[float] = None
    fold_growth_per_month: float = 0.0
    estimators_agree: Optional[bool] = None
    basis: str = ""


def _doubling_from_two_points(v0: float, v1: float, dt_days: float) -> float:
    if v0 <= 0 or v1 <= 0 or dt_days <= 0:
        raise ValueError("both values must be positive and timepoints distinct")
    rate = math.log(v1 / v0) / dt_days
    return math.log(2.0) / rate


#: Maximum gap between consecutive detected timepoints for them to count as
#: one contiguous kinetic segment. After a cleared interval (below LOQ), the
#: pre-clearance signal belongs to the PRIMARY tumour, not the re-emerging
#: clone, so it must not be pooled into the re-emergence slope.
_KINETIC_MAX_GAP_DAYS = 120.0


def _latest_contiguous_series(points: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Return the most recent run of detected points with no gap larger than
    _KINETIC_MAX_GAP_DAYS between neighbours."""
    if not points:
        return []
    tail: List[Tuple[int, float]] = [points[-1]]
    for pt in reversed(points[:-1]):
        if tail[-1][0] - pt[0] > _KINETIC_MAX_GAP_DAYS:
            break
        tail.append(pt)
    return list(reversed(tail))


def growth_kinetics(record: MultimodalPatientRecord) -> GrowthKinetics:
    """Fit doubling times from serial ctDNA and serial lesion diameters.

    ctDNA tumour fraction tracks tumour burden approximately exponentially,
    so ln(fraction) vs time is linear -- the slope IS the growth rate. Same
    for volume from diameter cubed.

    Only the latest CONTIGUOUS segment is fitted. Pooling across a cleared
    interval would mix the primary tumour's signal with the re-emerging
    clone's and dilute the doubling time -- an earlier version did exactly
    that and reported 242 days where the recent slope was ~40.
    """
    kinetics = GrowthKinetics()
    detected = [(b.day, b.tumor_fraction_pct) for b in record.sorted_biopsies() if b.detected()]
    series = _latest_contiguous_series(detected)
    if len(series) >= 2:
        f0, f1 = series[0][1], series[-1][1]
        dt = float(series[-1][0] - series[0][0])
        if f0 > 0 and f1 > 0 and dt > 0:
            kinetics.ct_dna_doubling_days = _doubling_from_two_points(f0, f1, dt)

    scans = [s for s in record.sorted_imaging() if s.lesion_longest_diameter_mm]
    if len(scans) >= 2:
        d0 = scans[0].lesion_longest_diameter_mm
        d1 = scans[-1].lesion_longest_diameter_mm
        dt = float(scans[-1].day - scans[0].day)
        if d0 > 0 and d1 > 0 and dt > 0:
            # Volume ~ diameter^3: convert the diameter ratio to a volume ratio.
            kinetics.imaging_doubling_days = _doubling_from_two_points(
                d0 ** 3, d1 ** 3, dt
            )

    if kinetics.ct_dna_doubling_days is not None:
        rate = math.log(2.0) / kinetics.ct_dna_doubling_days
        kinetics.fold_growth_per_month = math.exp(rate * 30.0)
        kinetics.basis = "serial ctDNA"
    elif kinetics.imaging_doubling_days is not None:
        rate = math.log(2.0) / kinetics.imaging_doubling_days
        kinetics.fold_growth_per_month = math.exp(rate * 30.0)
        kinetics.basis = "serial imaging volume"

    if kinetics.ct_dna_doubling_days and kinetics.imaging_doubling_days:
        lo = min(kinetics.ct_dna_doubling_days, kinetics.imaging_doubling_days)
        hi = max(kinetics.ct_dna_doubling_days, kinetics.imaging_doubling_days)
        kinetics.estimators_agree = hi / lo < 2.0

    return kinetics


# ---------------------------------------------------------------------------
# The molecular lead-time window
# ---------------------------------------------------------------------------

#: Published distribution of the ctDNA-before-imaging lead interval in CRC
#: MRD surveillance: median 5.3 months, IQR 3.0-16.4, maximum 28.7.
CTDNA_LEAD_TIME_MEDIAN_MONTHS = 5.3
CTDNA_LEAD_TIME_IQR_MONTHS = (3.0, 16.4)
CTDNA_LEAD_TIME_MAX_MONTHS = 28.7


@dataclass
class MolecularRecurrenceOutlook:
    state: str
    ct_dna_detected: bool
    radiographic_lesion_present: bool
    expected_radiographic_visibility_months: Tuple[float, float] = (0.0, 0.0)
    fold_growth_expected_at_visibility: float = 0.0
    notes: List[str] = field(default_factory=list)


def molecular_recurrence_outlook(
    record: MultimodalPatientRecord,
    fusion: Optional[FusionReport] = None,
) -> MolecularRecurrenceOutlook:
    """If ctDNA is positive and imaging is negative, when will imaging catch up?

    Two independent lines of evidence are combined and reported separately:

    1. The published lead-time distribution (external evidence).
    2. The patient's own measured doubling time (internal evidence): how much
       the ctDNA signal is expected to grow while imaging catches up.

    The absolute tumour volume is NOT reconstructed from the ctDNA fraction:
    shedding rates span orders of magnitude between patients, so a
    fraction-to-volume inversion would fabricate a number. Fold-growth of the
    SIGNAL is the honest quantity.
    """
    fusion = fusion or fuse_modalities(record)
    biopsies = record.sorted_biopsies()
    detected = [b for b in biopsies if b.detected()]
    imaging = record.sorted_imaging()
    last_img = imaging[-1] if imaging else None

    lesion_present = bool(
        last_img and last_img.lesion_longest_diameter_mm
        and last_img.lesion_longest_diameter_mm >= 5.0
    )

    outlook = MolecularRecurrenceOutlook(
        state="disease surveillance",
        ct_dna_detected=bool(detected) and detected[-1].detected(),
        radiographic_lesion_present=lesion_present,
    )

    if outlook.ct_dna_detected and not lesion_present:
        outlook.state = "molecular recurrence (lead-time window open)"
        lo, hi = CTDNA_LEAD_TIME_IQR_MONTHS
        outlook.expected_radiographic_visibility_months = (
            lo, min(hi, CTDNA_LEAD_TIME_MAX_MONTHS)
        )
        kinetics = growth_kinetics(record)
        if kinetics.fold_growth_per_month > 0:
            mid = (lo + hi) / 2.0
            outlook.fold_growth_expected_at_visibility = (
                kinetics.fold_growth_per_month ** mid
            )
        outlook.notes.append(
            "External anchor: in serial ctDNA surveillance, recurrence was "
            "detected a median of 5.3 months before radiographic recurrence "
            "(IQR 3.0-16.4). The window for THIS patient is assumed to fall "
            "in that interval; it is not predictable per-patient."
        )
        if outlook.fold_growth_expected_at_visibility:
            outlook.notes.append(
                f"At the patient's measured growth rate, the ctDNA signal is "
                f"expected to grow ~{outlook.fold_growth_expected_at_visibility:.1f}-fold "
                "over the median lead interval before imaging confirms."
            )

    return outlook


# ---------------------------------------------------------------------------
# Therapy counterfactuals: clearance, then regrowth
# ---------------------------------------------------------------------------

class TwinVerdict(str, Enum):
    MODELLED = "modelled"
    MODELLED_APPROXIMATE = "modelled_approximate"
    NOT_MODELLED = "not_modelled"


@dataclass
class TherapySpec:
    """A systemic therapy with a published cohort time-to-progression.

    These are COHORT statistics used as anchors for a lognormal individual
    spread -- not patient-level predictions. The spread is wide and is
    reported, not hidden."""

    name: str
    cohort_median_pfs_months: float
    #: Tumour types this therapy's cohort anchor was established in. The twin
    #: will not apply a lung-cancer PFS anchor to a colorectal patient: the
    #: median would be meaningless off-label.
    applicable_tumours: Tuple[str, ...] = ()
    #: Cohort-level resistance-mechanism distribution at progression,
    #: reported as background data. The model refuses to pick one for a
    #: specific patient.
    resistance_mechanisms: Tuple[Tuple[str, float], ...] = ()
    #: Effective therapy clears circulating signal over days-to-weeks. The
    #: tumour-burden decay half-life is a calibration target (ASSUMED).
    decay_half_life_days: float = 10.0
    response_rate: float = 0.0
    evidence_status: EvidenceStatus = EvidenceStatus.ESTABLISHED
    notes: List[str] = field(default_factory=list)


THERAPY_CATALOGUE: Dict[str, TherapySpec] = {
    "osimertinib_egfr_nslc": TherapySpec(
        name="osimertinib (first-line EGFR-mutant NSCLC)",
        cohort_median_pfs_months=18.9,                    # FLAURA
        applicable_tumours=("lung_nsclc",),
        resistance_mechanisms=(
            ("MET_amplification", 0.15),
            ("EGFR_C797S", 0.07),
            ("HER2_amplification", 0.05),
            ("small_cell_transformation", 0.05),
            ("other_or_unknown", 0.68),
        ),
        response_rate=0.80,
        evidence_status=EvidenceStatus.ESTABLISHED,
        notes=["Cohort anchor: FLAURA median PFS 18.9 months."],
    ),
    "pembrolizumab_melanoma": TherapySpec(
        name="pembrolizumab (anti-PD-1, melanoma)",
        cohort_median_pfs_months=8.1,                     # KEYNOTE-006
        applicable_tumours=("melanoma",),
        resistance_mechanisms=(("JAK1_JAK2_loss", 0.10), ("B2M_loss", 0.08), ("other_or_unknown", 0.82)),
        response_rate=0.40,
        decay_half_life_days=14.0,
        evidence_status=EvidenceStatus.ESTABLISHED,
        notes=[
            "Cohort anchor: KEYNOTE-006 median PFS ~8 months.",
            "Under immunotherapy, imaging may progress before ctDNA falls: "
            "pseudoprogression discordance is expected and fused separately.",
        ],
    ),
    "folfox_metastatic_crc": TherapySpec(
        name="FOLFOX (metastatic colorectal)",
        cohort_median_pfs_months=9.0,
        applicable_tumours=("colorectal",),
        resistance_mechanisms=(("RAS_BRAF_emergence", 0.20), ("other_or_unknown", 0.80)),
        response_rate=0.50,
        evidence_status=EvidenceStatus.ESTABLISHED,
        notes=["Cohort anchor: modified FOLFOX6 mPFS ~9 months in mCRC."],
    ),
}

#: Shape parameter of the assumed lognormal individual spread around the
#: cohort median PFS. ASSUMED: cohort PFS distributions are right-skewed and
#: trial-calibrated; sigma=0.6 gives an 80% interval of roughly 0.4x-2.5x the
#: median, which matches the shape of published Kaplan-Meier curves. It is a
#: modelling choice, not a measured constant.
PFS_LOGNORMAL_SIGMA = 0.6


@dataclass
class TherapyCounterfactual:
    therapy: str
    verdict: TwinVerdict
    ct_dna_nadir_day: int
    ct_dna_nadir_fold_from_baseline: float
    expected_molecular_progression_months: Tuple[float, float]   # 80% interval
    cohort_median_pfs_months: float
    patient_baseline_fold_growth_per_month: float
    resistance_mechanism_distribution: Tuple[Tuple[str, float], ...] = ()
    not_modelled_reason: Optional[str] = None
    approximations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def simulate_line_of_therapy(
    record: MultimodalPatientRecord,
    therapy_key: str,
) -> TherapyCounterfactual:
    """What the ctDNA trajectory would look like under a therapy.

    Modelled, from published anchors plus the patient's own kinetics:
      - clearance: exponential decay of the circulating signal from the
        patient's current level at the therapy's decay half-life;
      - regrowth: at the patient's PRE-TREATMENT doubling time (resistance
        clones grow; the model does not pretend to know the new clone's rate);
      - time-to-molecular-progression: lognormal around the cohort median
        with an explicitly ASSUMED shape.

    Refused, by design:
      - WHICH resistance mechanism will emerge in this patient;
      - any claim built from a single ctDNA timepoint.
    """
    baseline_growth = growth_kinetics(record).fold_growth_per_month
    spec = THERAPY_CATALOGUE.get(therapy_key)
    if spec is None:
        return TherapyCounterfactual(
            therapy=therapy_key,
            verdict=TwinVerdict.NOT_MODELLED,
            ct_dna_nadir_day=0,
            ct_dna_nadir_fold_from_baseline=1.0,
            expected_molecular_progression_months=(0.0, 0.0),
            cohort_median_pfs_months=0.0,
            patient_baseline_fold_growth_per_month=baseline_growth,
            not_modelled_reason=(
                f"'{therapy_key}' is not in the therapy catalogue. The twin "
                "runs only therapies with a published cohort time-to-"
                "progression anchor; refusing to invent one is the correct "
                "behaviour."
            ),
        )

    detected = [b for b in record.sorted_biopsies() if b.detected()]
    if len(detected) < 2:
        return TherapyCounterfactual(
            therapy=spec.name,
            verdict=TwinVerdict.NOT_MODELLED,
            ct_dna_nadir_day=0,
            ct_dna_nadir_fold_from_baseline=1.0,
            expected_molecular_progression_months=(0.0, 0.0),
            cohort_median_pfs_months=spec.cohort_median_pfs_months,
            patient_baseline_fold_growth_per_month=baseline_growth,
            not_modelled_reason=(
                "kinetic counterfactuals require at least two detectable "
                "ctDNA timepoints to establish the patient's own growth rate; "
                "with fewer, only detection status is known. Run a second "
                "assay before asking for a trajectory."
            ),
        )

    # Tumour-type gate: a cohort PFS anchor means nothing off-label, and the
    # twin must not lend false authority to an inapplicable number.
    if spec.applicable_tumours and record.clinical.tumour_type not in spec.applicable_tumours:
        return TherapyCounterfactual(
            therapy=spec.name,
            verdict=TwinVerdict.NOT_MODELLED,
            ct_dna_nadir_day=0,
            ct_dna_nadir_fold_from_baseline=1.0,
            expected_molecular_progression_months=(0.0, 0.0),
            cohort_median_pfs_months=spec.cohort_median_pfs_months,
            patient_baseline_fold_growth_per_month=baseline_growth,
            not_modelled_reason=(
                f"this therapy's cohort anchor was established in "
                f"{'/'.join(spec.applicable_tumours)}, not "
                f"{record.clinical.tumour_type}; applying that PFS median to "
                "this patient would lend false authority to an off-label "
                "number"
            ),
        )

    # Clearance: ~5 half-lives to a practical nadir floor (below assay LOQ).
    nadir_day = int(round(5.0 * spec.decay_half_life_days))
    nadir_fold = 2.0 ** (-5.0)

    # Regrowth: time to molecular progression around the cohort anchor.
    sigma = PFS_LOGNORMAL_SIGMA
    mu = math.log(spec.cohort_median_pfs_months) - sigma ** 2 / 2.0
    p10 = math.exp(mu - 1.2816 * sigma)
    p90 = math.exp(mu + 1.2816 * sigma)

    approximations = [
        "individual time-to-progression drawn around the cohort median with "
        "an ASSUMED lognormal shape (sigma=0.6); the patient's own draw is "
        "not predictable",
        "tumour-burden decay half-life is a calibration target, not a "
        "measured constant",
        "regrowth is modelled at the PRE-treatment doubling time; the "
        "resistant clone's actual rate is unknown",
    ]

    return TherapyCounterfactual(
        therapy=spec.name,
        verdict=TwinVerdict.MODELLED if spec.evidence_status is EvidenceStatus.ESTABLISHED
        else TwinVerdict.MODELLED_APPROXIMATE,
        ct_dna_nadir_day=nadir_day,
        ct_dna_nadir_fold_from_baseline=nadir_fold,
        expected_molecular_progression_months=(p10, p90),
        cohort_median_pfs_months=spec.cohort_median_pfs_months,
        patient_baseline_fold_growth_per_month=baseline_growth,
        resistance_mechanism_distribution=spec.resistance_mechanisms,
        approximations=approximations,
        notes=list(spec.notes) + [
            "The mechanism distribution above is COHORT background data at "
            "progression. Which mechanism THIS tumour uses is stochastic and "
            "is not predicted -- that claim would be fabrication, and early "
            "detection of the clone is what serial ctDNA exists for.",
        ],
    )


# ---------------------------------------------------------------------------
# The twin object
# ---------------------------------------------------------------------------


@dataclass
class OncotwinProvenance:
    @staticmethod
    def as_dict() -> Dict[str, EvidenceStatus]:
        return {
            "cross_modality_fusion_rules": EvidenceStatus.EMERGING_HUMAN,
            "ctdna_lead_time_distribution_crc": EvidenceStatus.ESTABLISHED,
            "chip_confound_and_wbc_matching": EvidenceStatus.ESTABLISHED,
            "log_linear_growth_fitting": EvidenceStatus.ESTABLISHED,
            "tumour_fraction_to_absolute_volume": EvidenceStatus.ANIMAL_ONLY,  # refused
            "cohort_pfs_anchors_flaura_keynote_folfox": EvidenceStatus.ESTABLISHED,
            "individual_pfs_lognormal_shape": EvidenceStatus.ANIMAL_ONLY,      # assumed
            "tumour_burden_decay_half_life": EvidenceStatus.ANIMAL_ONLY,       # assumed
            "per_patient_resistance_mechanism_prediction": EvidenceStatus.PROVEN_FAILED,  # refused
            "foundation_model_pattern_matcher": EvidenceStatus.UNKNOWN,        # absent
        }


def oncotwin_provenance() -> Dict[str, EvidenceStatus]:
    return OncotwinProvenance.as_dict()


@dataclass
class OncotwinReport:
    patient_id: str
    fusion: FusionReport
    kinetics: GrowthKinetics
    outlook: MolecularRecurrenceOutlook
    therapy_counterfactuals: Dict[str, TherapyCounterfactual]
    cannot_know: List[str]

    def render(self) -> str:
        lines = [
            f"ONCOLOGY MULTIMODAL DATA TWIN - {self.patient_id}",
            "=" * 62,
            "Fusion (cross-modality consistency):",
        ]
        for state in self.fusion.clinical_states:
            lines.append(f"  state    : {state}")
        for d in self.fusion.discordances:
            lines.append(f"  discord  : {d}")
        for c in self.fusion.chip_warnings:
            lines.append(f"  CHIP     : {c}")
        if self.fusion.usable_biopsy_timepoints:
            lines.append(
                f"  usable timepoints: {self.fusion.usable_biopsy_timepoints} ctDNA, "
                f"{self.fusion.usable_imaging_timepoints} imaging"
            )

        k = self.kinetics
        lines += ["", "Kinetics (from the patient's own serial data):"]
        if k.ct_dna_doubling_days:
            lines.append(f"  ctDNA volume-doubling time : {k.ct_dna_doubling_days:.0f} days")
        if k.imaging_doubling_days:
            lines.append(f"  imaging volume-doubling time: {k.imaging_doubling_days:.0f} days")
        if k.ct_dna_doubling_days and k.imaging_doubling_days:
            agree = "AGREE" if k.estimators_agree else "DISAGREE (>2x)"
            lines.append(f"  estimator agreement         : {agree}")
        if k.fold_growth_per_month:
            lines.append(
                f"  expected signal growth      : {k.fold_growth_per_month:.2f}-fold/month "
                f"(basis: {k.basis})"
            )

        o = self.outlook
        if o.state != "disease surveillance":
            lines += ["", f"State: {o.state}"]
            if o.expected_radiographic_visibility_months != (0.0, 0.0):
                lo, hi = o.expected_radiographic_visibility_months
                lines.append(
                    f"  radiographic visibility expected in ~{lo:.1f}-{hi:.1f} months "
                    "(published lead-time IQR)"
                )

        lines += ["", "Therapy counterfactuals (ctDNA trajectory anchors):"]
        for key, tc in self.therapy_counterfactuals.items():
            if tc.verdict is TwinVerdict.NOT_MODELLED:
                lines.append(f"  {key}: NOT MODELLED - {tc.not_modelled_reason[:70]}...")
                continue
            lo, hi = tc.expected_molecular_progression_months
            lines.append(
                f"  {key}: nadir day {tc.ct_dna_nadir_day} "
                f"({tc.ct_dna_nadir_fold_from_baseline:.4f}x baseline), "
                f"molecular progression {lo:.1f}-{hi:.1f} mo "
                f"(cohort median {tc.cohort_median_pfs_months:.1f})"
            )

        lines += ["", "What this model cannot know:"]
        lines += [f"  - {s}" for s in self.cannot_know]
        return "\n".join(lines)


class Oncotwin:
    """Assembled multimodal oncology twin."""

    def __init__(self, record: MultimodalPatientRecord):
        self.record = record
        self.fusion = fuse_modalities(record)
        self.kinetics = growth_kinetics(record)
        self.outlook = molecular_recurrence_outlook(record, self.fusion)
        self.cannot_know = self.what_this_model_cannot_know()

    def simulate_therapy(self, therapy_key: str) -> TherapyCounterfactual:
        return simulate_line_of_therapy(self.record, therapy_key)

    def run_therapy_catalogue(self) -> Dict[str, TherapyCounterfactual]:
        return {
            key: self.simulate_therapy(key) for key in THERAPY_CATALOGUE
        }

    def full_report(self) -> OncotwinReport:
        return OncotwinReport(
            patient_id=self.record.patient_id,
            fusion=self.fusion,
            kinetics=self.kinetics,
            outlook=self.outlook,
            therapy_counterfactuals=self.run_therapy_catalogue(),
            cannot_know=self.cannot_know,
        )

    def what_this_model_cannot_know(self) -> List[str]:
        return [
            "the 10-million-patient learned pattern-matcher: this repository "
            "contains zero training patients; every number here is a "
            "published constant or the patient's own measurement",
            "which resistance mechanism will emerge in this tumour: "
            "stochastic evolutionary biology; only cohort-level mechanism "
            "distributions exist",
            "absolute tumour volume from ctDNA fraction: shedding rates span "
            "orders of magnitude between patients, so the inversion would "
            "fabricate a number (fold-growth of the signal is used instead)",
            "tumour internal heterogeneity and immune microenvironment: "
            "below imaging resolution and unmeasured by plasma",
            "per-patient pharmacokinetics: dosing, exposure and toxicity are "
            "outside this twin",
        ]
