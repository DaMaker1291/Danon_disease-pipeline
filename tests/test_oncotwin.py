"""Tests for the oncology multimodal data twin.

Themes: fusion rules that match real clinical interpretation, kinetics
fitted from the patient's own serial data, and refusals (off-label anchors,
single-timepoint trajectories, per-patient resistance prediction).
"""

from __future__ import annotations

import pytest

from danon.oncotwin import (
    CTDNA_LEAD_TIME_IQR_MONTHS,
    CTDNA_SENSITIVITY_CEILING,
    PFS_LOGNORMAL_SIGMA,
    THERAPY_CATALOGUE,
    CHIP_GENES,
    FusionReport,
    ImagingRecord,
    LiquidBiopsy,
    MultimodalPatientRecord,
    Oncotwin,
    SystemicRecord,
    TwinVerdict,
    _latest_contiguous_series,
    fuse_modalities,
    growth_kinetics,
    molecular_recurrence_outlook,
    oncotwin_provenance,
    simulate_line_of_therapy,
)
from danon.cure import EvidenceStatus


def _molecular_recurrence_record() -> MultimodalPatientRecord:
    """Resected colorectal patient: cleared, then ctDNA re-emerges while
    imaging stays clean -- the lead-time window in its canonical form."""
    return MultimodalPatientRecord(
        patient_id="onco-test",
        clinical=SystemicRecord(age_years=63, tumour_type="colorectal", stage="III"),
        imaging=[
            ImagingRecord(day=30, radiologist_impression="none"),
            ImagingRecord(day=240, radiologist_impression="none"),
        ],
        biopsies=[
            LiquidBiopsy(day=30, tumor_fraction_pct=0.021),
            LiquidBiopsy(day=90, below_loq=True),
            LiquidBiopsy(day=210, below_loq=True),
            LiquidBiopsy(day=300, tumor_fraction_pct=0.019),
            LiquidBiopsy(day=360, tumor_fraction_pct=0.054),
        ],
    )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_molecular_recurrence_state_is_detected():
    report = fuse_modalities(_molecular_recurrence_record())
    assert report.is_molecular_recurrence


def test_rising_ctdna_with_stable_imaging_is_flagged():
    record = _molecular_recurrence_record()
    record.imaging[-1] = ImagingRecord(
        day=360, lesion_longest_diameter_mm=8.0, radiologist_impression="stable"
    )
    report = fuse_modalities(record)
    assert any("lead-time" in d for d in report.discordances)


def test_imaging_progression_with_falling_ctdna_flags_pseudoprogression():
    record = MultimodalPatientRecord(
        patient_id="pseu",
        clinical=SystemicRecord(tumour_type="melanoma"),
        imaging=[ImagingRecord(day=60, lesion_longest_diameter_mm=30.0,
                               radiologist_impression="progression")],
        biopsies=[
            LiquidBiopsy(day=10, tumor_fraction_pct=0.50),
            LiquidBiopsy(day=60, tumor_fraction_pct=0.10),
        ],
    )
    report = fuse_modalities(record)
    assert report.possible_pseudoprogression


def test_chip_without_wbc_matching_is_flagged_by_name():
    record = MultimodalPatientRecord(
        patient_id="chip",
        clinical=SystemicRecord(tumour_type="colorectal"),
        biopsies=[LiquidBiopsy(day=10, tumor_fraction_pct=0.3,
                               chip_genes_detected=("DNMT3A", "TET2"))],
    )
    report = fuse_modalities(record)
    assert report.chip_warnings
    assert "DNMT3A" in report.chip_warnings[0]
    assert "leukocyte" in report.chip_warnings[0]


def test_single_timepoint_refuses_trajectory_but_not_detection():
    record = MultimodalPatientRecord(
        patient_id="single",
        clinical=SystemicRecord(tumour_type="colorectal"),
        biopsies=[LiquidBiopsy(day=10, tumor_fraction_pct=0.30)],
    )
    report = fuse_modalities(record)
    assert not report.kinetic_fit_possible
    assert any("detection status" in s for s in report.clinical_states)


def test_low_shedding_tumour_negative_is_not_reassurance():
    """A low-shedding tumour type with a negative assay must not be read as
    disease-free: the ~90% cumulative sensitivity ceiling forbids it."""
    record = MultimodalPatientRecord(
        patient_id="low",
        clinical=SystemicRecord(tumour_type="prostate_early"),
        biopsies=[LiquidBiopsy(day=10, below_loq=True)],
    )
    report = fuse_modalities(record)
    assert any("low-shedding" in d for d in report.discordances)


def test_glioma_gets_the_csf_rule_which_supersedes_it():
    """Glioma plasma is near-absent by biology, so the CSF message is the
    strictly more informative warning and supersedes the generic ceiling."""
    record = MultimodalPatientRecord(
        patient_id="gli",
        clinical=SystemicRecord(tumour_type="glioma"),
        biopsies=[LiquidBiopsy(day=10, below_loq=True)],
    )
    report = fuse_modalities(record)
    assert any("CSF" in w for w in report.chip_warnings)


def test_sensitivity_ceiling_is_below_one():
    """The assay's negative predictive value is bounded below 100% because a
    material fraction of patients never shed detectable ctDNA."""
    assert 0.5 < CTDNA_SENSITIVITY_CEILING < 1.0


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

def test_doubling_time_from_two_points():
    days = growth_kinetics(_molecular_recurrence_record()).ct_dna_doubling_days
    # 0.019 -> 0.054 over 60 days: ~1.7 doublings -> ~40 days.
    assert days == pytest.approx(40.0, rel=0.15)


def test_kinetics_only_fit_the_latest_contiguous_segment():
    """Regression test.

    Pooling across the cleared interval (day 30 signal -> day 360 signal)
    mixes the primary tumour with the re-emerging clone and reported a 242-day
    doubling where the recent slope is ~40 days. The fit must use only the
    post-clearance segment.
    """
    days = growth_kinetics(_molecular_recurrence_record()).ct_dna_doubling_days
    assert days < 100.0


def test_latest_contiguous_segment_breaks_on_large_gaps():
    points = [(0, 1.0), (50, 2.0), (400, 2.0), (450, 4.0)]
    seg = _latest_contiguous_series(points)
    assert seg == [(400, 2.0), (450, 4.0)]


def test_imaging_doubling_uses_cubic_volume():
    record = MultimodalPatientRecord(
        patient_id="img",
        clinical=SystemicRecord(tumour_type="colorectal"),
        imaging=[
            ImagingRecord(day=0, lesion_longest_diameter_mm=10.0),
            ImagingRecord(day=100, lesion_longest_diameter_mm=20.0),
        ],
    )
    k = growth_kinetics(record)
    # Volume x8 in 100 days = 3 doublings -> 33.3 days.
    assert k.imaging_doubling_days == pytest.approx(100.0 / 3.0, rel=0.02)
    assert k.ct_dna_doubling_days is None


# ---------------------------------------------------------------------------
# Molecular recurrence outlook
# ---------------------------------------------------------------------------

def test_outlook_reports_published_lead_time_window():
    outlook = molecular_recurrence_outlook(_molecular_recurrence_record())
    assert outlook.state.startswith("molecular recurrence")
    lo, hi = outlook.expected_radiographic_visibility_months
    assert (lo, hi) == CTDNA_LEAD_TIME_IQR_MONTHS
    assert any("5.3" in n for n in outlook.notes)


def test_outlook_uses_patient_doubling_for_expected_signal_growth():
    outlook = molecular_recurrence_outlook(_molecular_recurrence_record())
    # Median of the IQR is ~9.7 months; at 1.69x/month that is a large fold.
    assert outlook.fold_growth_expected_at_visibility > 10.0


def test_lesion_present_outlook_is_not_molecular_recurrence():
    record = _molecular_recurrence_record()
    record.imaging[-1] = ImagingRecord(
        day=360, lesion_longest_diameter_mm=15.0, radiologist_impression="progression"
    )
    outlook = molecular_recurrence_outlook(record)
    assert not outlook.state.startswith("molecular recurrence")


# ---------------------------------------------------------------------------
# Therapy counterfactuals and the refusals
# ---------------------------------------------------------------------------

def test_on_label_therapy_is_modelled_with_cohort_anchor():
    tc = simulate_line_of_therapy(_molecular_recurrence_record(), "folfox_metastatic_crc")
    assert tc.verdict is TwinVerdict.MODELLED
    assert tc.ct_dna_nadir_day == 50
    assert tc.ct_dna_nadir_fold_from_baseline == pytest.approx(1 / 32.0)
    lo, hi = tc.expected_molecular_progression_months
    assert lo < tc.cohort_median_pfs_months < hi
    assert any("cohort" in n.lower() for n in tc.notes)


def test_off_label_therapy_is_refused_by_tumour_type():
    """Regression test: the catalogue originally offered an EGFR-NSCLC drug's
    PFS anchor to a colorectal patient. An off-label median lends false
    authority and must be refused."""
    tc = simulate_line_of_therapy(_molecular_recurrence_record(), "osimertinib_egfr_nslc")
    assert tc.verdict is TwinVerdict.NOT_MODELLED
    assert "lung_nsclc" in tc.not_modelled_reason
    assert "colorectal" in tc.not_modelled_reason


def test_unknown_therapy_is_refused():
    tc = simulate_line_of_therapy(_molecular_recurrence_record(), "tumour_shrinking_vibes")
    assert tc.verdict is TwinVerdict.NOT_MODELLED
    assert "catalogue" in tc.not_modelled_reason


def test_single_timepoint_patient_cannot_get_a_trajectory():
    record = MultimodalPatientRecord(
        patient_id="single",
        clinical=SystemicRecord(tumour_type="colorectal"),
        biopsies=[LiquidBiopsy(day=10, tumor_fraction_pct=0.30)],
    )
    tc = simulate_line_of_therapy(record, "folfox_metastatic_crc")
    assert tc.verdict is TwinVerdict.NOT_MODELLED
    assert "two detectable" in tc.not_modelled_reason


def test_resistance_mechanism_distribution_is_background_not_prediction():
    """The mechanism list must be reported as cohort data, and the notes must
    explicitly refuse to name THIS tumour's escape -- the Drug-A-in-90-days
    claim is stochastic biology and this module will not fake it."""
    tc = simulate_line_of_therapy(_molecular_recurrence_record(), "folfox_metastatic_crc")
    assert tc.resistance_mechanism_distribution
    total = sum(p for _, p in tc.resistance_mechanism_distribution)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert any("not predicted" in n or "stochastic" in n for n in tc.notes)


def test_lognormal_spread_is_declared_and_reasonable():
    """The individual spread around the cohort median is an ASSUMED shape,
    and the 80% interval it produces must actually bracket the median."""
    assert PFS_LOGNORMAL_SIGMA > 0
    spec = THERAPY_CATALOGUE["folfox_metastatic_crc"]
    tc = simulate_line_of_therapy(_molecular_recurrence_record(), "folfox_metastatic_crc")
    lo, hi = tc.expected_molecular_progression_months
    assert hi / lo > 2.0          # a wide, honest interval
    assert lo < spec.cohort_median_pfs_months < hi


# ---------------------------------------------------------------------------
# Provenance and the assembled twin
# ---------------------------------------------------------------------------

def test_provenance_marks_the_refusals():
    prov = oncotwin_provenance()
    assert prov["per_patient_resistance_mechanism_prediction"] is EvidenceStatus.PROVEN_FAILED
    assert prov["tumour_fraction_to_absolute_volume"] is EvidenceStatus.ANIMAL_ONLY
    assert prov["ctdna_lead_time_distribution_crc"] is EvidenceStatus.ESTABLISHED
    assert prov["foundation_model_pattern_matcher"] is EvidenceStatus.UNKNOWN


def test_full_report_renders_every_section():
    rep = Oncotwin(_molecular_recurrence_record()).full_report()
    text = rep.render()
    for section in (
        "ONCOLOGY MULTIMODAL DATA TWIN",
        "Fusion",
        "Kinetics",
        "molecular recurrence",
        "Therapy counterfactuals",
        "What this model cannot know",
    ):
        assert section in text


def test_cannot_know_names_the_missing_pattern_matcher():
    twin = Oncotwin(_molecular_recurrence_record())
    limits = twin.what_this_model_cannot_know()
    assert any("10-million-patient" in s for s in limits)
    assert any("resistance mechanism" in s for s in limits)
