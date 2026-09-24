"""Tests for the retrospective multimodal benchmark module.

Two things are pinned here:

* The recorded probe evidence in ``data/tcia_probe/multimodal_evidence.json`` is
  the *input* to the spatial verdict. These tests assert the counts that make
  the refusal true, so the refusal cannot quietly become stale prose.
* The verdict must be **derivable**: a synthetic evidence set carrying masks and
  a second timepoint has to flip it to SUPPORTED. If it does not, the refusal is
  an assertion dressed as a computation.
"""

from __future__ import annotations

import json
import os

import pytest

from danon.multimodal import (
    CANDIDATE_GLIOMA_COLLECTIONS,
    EVIDENCE_PATH,
    BenchmarkTest,
    ModalityEvidence,
    ModalityStatus,
    PairedControlSummary,
    SpatialDiceVerdict,
    benchmark_tests,
    build_evidence,
    load_evidence,
    load_paired_control_cards,
    measure_collection_timepoints,
    multimodal_provenance,
    probe_local_inventory,
    render_multimodal_report,
    save_evidence,
    spatial_dice_verdict,
    summarize_paired_control,
    what_multimodal_benchmark_cannot_show,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(EVIDENCE_PATH),
    reason="recorded probe evidence absent; run the probe to regenerate it",
)


def _ev(key, status, detail, name=None):
    return ModalityEvidence(
        key=key, name=name or key, status=status, source="test", license="test",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Recorded evidence
# ---------------------------------------------------------------------------


class TestRecordedEvidence:
    def test_evidence_loads_and_names_every_recorded_source(self):
        items = load_evidence()
        keys = {m.key for m in items}
        assert "tcia_nbia_api" in keys
        assert "braats2021_segmentations" in keys
        assert "gdc_tcga_gbm_clinical" in keys
        assert "chembl_measured_potency" in keys

    def test_tcga_gbm_is_recorded_as_retired(self):
        """The barcode route to matched imaging+genomics closed, and it is recorded."""
        by_key = {m.key: m for m in load_evidence()}
        tcia = by_key["tcia_nbia_api"]
        assert "TCGA-GBM" in (tcia.get("glioma_candidates_absent") or [])
        assert "TCGA-LGG" in (tcia.get("glioma_candidates_absent") or [])
        assert by_key["tcia_tcga_gbm"].status is ModalityStatus.RETIRED

    def test_upenn_gbm_records_the_mask_ground_and_drops_the_timepoint_ground(self):
        """One blocker is left, and it is the one the data supports.

        The entry used to record 0 multi-timepoint patients from a 2-patient
        sample, which made "the collection is single-timepoint" look derived.
        Scored over all 630 patients it is 41, so the mask is the only ground
        left. The old sample is kept under ``sampled_*`` names: deleting it
        would hide that a sample ever stood in for a cohort.
        """
        by_key = {m.key: m for m in load_evidence()}
        upenn = by_key["tcia_upenn_gbm"]
        assert int(upenn.get("n_seg_series")) == 0
        assert int(upenn.get("patients_with_multi_timepoint")) == 41
        assert int(upenn.get("patients_checked_for_timepoints")) == 630
        assert int(upenn.get("sampled_patients_checked")) == 2
        assert int(upenn.get("sampled_patients_with_multi_timepoint")) == 0
        assert int(upenn.get("n_patients")) > 100
        # the withdrawn ground must stay withdrawn
        assert not any("timepoint" in c for c in upenn.cannot_support)
        assert any("mask" in c for c in upenn.cannot_support)
        assert upenn.status is ModalityStatus.ABSENT

    def test_local_braats_inventory_is_single_timepoint(self):
        by_key = {m.key: m for m in load_evidence()}
        brats = by_key["braats2021_segmentations"]
        assert brats.status is ModalityStatus.LOCAL
        assert int(brats.get("n_segmentation_files")) > 0
        assert int(brats.get("timepoints_per_case")) == 1

    def test_local_clinical_inventory_has_a_real_case_count(self):
        by_key = {m.key: m for m in load_evidence()}
        clinical = by_key["gdc_tcga_gbm_clinical"]
        assert clinical.status is ModalityStatus.LOCAL
        assert int(clinical.get("n_cases")) > 100

    def test_every_recorded_collection_says_what_it_cannot_do(self):
        """A source with no stated limit is a source presented as unlimited."""
        for m in load_evidence():
            if m.key.startswith("tcia_"):
                assert m.cannot_support, f"{m.key} records no limitation"


# ---------------------------------------------------------------------------
# Full-cohort measurement (no sampling, no network)
# ---------------------------------------------------------------------------


UPENN_SERIES_LISTING = "data/tcia_probe/upenn_series.json"


@pytest.mark.skipif(
    not os.path.exists(UPENN_SERIES_LISTING),
    reason="cached UPENN-GBM series listing absent",
)
class TestCohortTimepointMeasurement:
    """The cohort-scored counts must reproduce what the evidence claims."""

    def test_recorded_counts_reproduce_from_the_listing(self):
        """No drift: the record is a function of the listing, not typed by hand."""
        measured = measure_collection_timepoints(UPENN_SERIES_LISTING)
        upenn = {m.key: m for m in load_evidence()}["tcia_upenn_gbm"]
        assert measured["n_patients"] == int(upenn.get("n_patients"))
        assert measured["n_series"] == int(upenn.get("n_series"))
        assert measured["n_patients_longitudinal"] == int(
            upenn.get("patients_with_multi_timepoint")
        )
        assert measured["n_patients_multi_date"] == int(
            upenn.get("patients_multi_date_any_interval")
        )
        assert measured["min_interval_days"] == int(upenn.get("min_followup_interval_days"))
        assert measured["n_patients_dsc_perfusion"] == int(
            upenn.get("n_patients_dsc_perfusion")
        )
        assert measured["n_patients_processed_perfusion"] == int(
            upenn.get("n_patients_processed_perfusion")
        )

    def test_the_criterion_is_what_separates_43_from_41(self):
        """Distinct dates overstate follow-up; the threshold is the correction."""
        strict = measure_collection_timepoints(UPENN_SERIES_LISTING, min_interval_days=28)
        loose = measure_collection_timepoints(UPENN_SERIES_LISTING, min_interval_days=1)
        assert strict["n_patients_multi_date"] == 43
        assert strict["n_patients_longitudinal"] == 41
        assert loose["n_patients_longitudinal"] == 43
        # the criterion travels with the number it produced
        assert strict["min_interval_days"] == 28
        assert loose["min_interval_days"] == 1

    def test_perfusion_covers_every_longitudinal_patient(self):
        """The join that makes streams 3 and 4 co-activatable, not just present."""
        measured = measure_collection_timepoints(UPENN_SERIES_LISTING)
        assert measured["n_dsc_series"] == 490
        assert measured["n_patients_dsc_perfusion"] == 456
        assert measured["n_processed_series"] == 2607
        assert measured["n_patients_processed_perfusion"] == 628
        assert measured["n_longitudinal_with_perfusion"] == measured[
            "n_patients_longitudinal"
        ]

    def test_a_patient_without_an_id_or_a_date_is_skipped(self):
        """Malformed rows must not inflate a cohort count."""
        rows = [
            {"PatientID": "", "StudyDate": "2004-01-01"},
            {"PatientID": "P1", "StudyDate": "not-a-date"},
            {"PatientID": "P1", "SeriesDate": "2004-02-01"},
            {"PatientID": "P2", "StudyDate": "2004-01-01"},
            {"PatientID": "P2", "StudyDate": "2004-04-01"},
        ]
        m = measure_collection_timepoints(rows)
        assert m["n_patients"] == 2  # the blank id is not a patient
        assert m["n_patients_multi_date"] == 2
        assert m["n_patients_longitudinal"] == 1  # only P2 has a real interval
        assert m["source_file"] == "in-memory listing"


# ---------------------------------------------------------------------------
# The verdict is derived, so it can flip
# ---------------------------------------------------------------------------


class TestDerivedVerdict:
    def test_verdict_refuses_on_the_recorded_evidence(self):
        v = spatial_dice_verdict()
        assert isinstance(v, SpatialDiceVerdict)
        assert v.supported is False
        assert "no reachable glioma imaging collection" in v.reason
        assert len(v.deciding_evidence) >= 2
        assert any("0 segmentation series" in e for e in v.deciding_evidence)
        assert len(v.what_would_be_needed) == 3

    def test_verdict_flips_when_a_qualifying_collection_appears(self):
        """The refusal must be a computation, not a hard-coded 'no'."""
        items = [
            _ev("tcia_demo", ModalityStatus.AVAILABLE, {
                "n_patients": 40, "n_series": 900, "modalities": ["MR", "SEG"],
                "n_seg_series": 40, "patients_checked_for_timepoints": 3,
                "patients_with_multi_timepoint": 2,
            }),
        ]
        v = spatial_dice_verdict(items)
        assert v.supported is True
        assert "carries both masks and a second timepoint" in v.reason
        assert v.what_would_be_needed == ()
        assert "SUPPORTED" in v.describe()

    def test_masks_alone_are_not_enough(self):
        items = [
            _ev("tcia_a", ModalityStatus.AVAILABLE, {
                "n_patients": 10, "n_series": 100, "n_seg_series": 10,
                "patients_checked_for_timepoints": 2,
                "patients_with_multi_timepoint": 0,
            }),
        ]
        assert spatial_dice_verdict(items).supported is False

    def test_a_second_timepoint_alone_is_not_enough(self):
        items = [
            _ev("tcia_a", ModalityStatus.AVAILABLE, {
                "n_patients": 10, "n_series": 100, "n_seg_series": 0,
                "patients_checked_for_timepoints": 2,
                "patients_with_multi_timepoint": 2,
            }),
        ]
        assert spatial_dice_verdict(items).supported is False

    def test_empty_evidence_refuses_rather_than_assuming_support(self):
        v = spatial_dice_verdict([])
        assert v.supported is False
        assert "no imaging collections recorded" in v.reason

    def test_retired_collections_are_listed_as_deciding_evidence(self):
        items = [
            _ev("tcia_old", ModalityStatus.RETIRED, {"collection": "TCGA-GBM"}),
            _ev("tcia_cur", ModalityStatus.ABSENT, {
                "n_patients": 5, "n_series": 50, "n_seg_series": 0,
                "patients_checked_for_timepoints": 2,
                "patients_with_multi_timepoint": 0,
            }),
        ]
        v = spatial_dice_verdict(items)
        assert any("RETIRED" in e for e in v.deciding_evidence)


# ---------------------------------------------------------------------------
# Test battery
# ---------------------------------------------------------------------------


class TestBenchmarkBattery:
    def test_three_tests_are_reported_with_statuses(self):
        tests = benchmark_tests()
        assert len(tests) == 3
        ids = {t.test_id: t for t in tests}
        assert set(ids) == {"test1_discrimination", "test2_spatial_dice", "test3_paired_control"}
        assert all(isinstance(t, BenchmarkTest) for t in tests)

    def test_spatial_test_is_refused_and_says_what_would_unblock_it(self):
        ids = {t.test_id: t for t in benchmark_tests()}
        spatial = ids["test2_spatial_dice"]
        assert spatial.status == "REFUSED"
        assert spatial.blocker
        assert "DICOM SEG" in spatial.blocker

    def test_discrimination_test_records_that_it_is_not_the_engine_to_whom_it_refers(self):
        """The C-index is for a covariate model; saying so is the honest part."""
        ids = {t.test_id: t for t in benchmark_tests()}
        d = ids["test1_discrimination"]
        assert d.status == "SUPPORTED"
        assert "engine itself has no per-patient prediction" in d.blocker

    def test_paired_control_test_records_that_it_is_not_a_treatment_effect(self):
        ids = {t.test_id: t for t in benchmark_tests()}
        p = ids["test3_paired_control"]
        assert p.status == "SUPPORTED"
        assert "not a treatment effect" in p.blocker

    def test_battery_follows_the_verdict_it_is_computed_from(self):
        items = [
            _ev("tcia_demo", ModalityStatus.AVAILABLE, {
                "n_patients": 40, "n_series": 900, "n_seg_series": 40,
                "patients_checked_for_timepoints": 3,
                "patients_with_multi_timepoint": 2,
            }),
        ]
        ids = {t.test_id: t for t in benchmark_tests(items)}
        assert ids["test2_spatial_dice"].status == "SUPPORTED"


# ---------------------------------------------------------------------------
# Report / provenance / limits
# ---------------------------------------------------------------------------


class TestReportAndLimits:
    def test_report_is_poster_ready_and_carries_the_refusal(self):
        report = render_multimodal_report()
        assert "Retrospective multimodal benchmark" in report
        assert "Test battery" in report
        assert "REFUSED" in report
        assert "## Limits" in report
        assert "test1_discrimination" in report

    def test_limits_distinguish_unavailability_from_negative_evidence(self):
        joined = " ".join(what_multimodal_benchmark_cannot_show()).lower()
        assert "absence of a test is not a negative result" in joined
        assert "not say the estimate is good" in joined
        assert "cannot be reported as a survival gain" in joined

    def test_provenance_marks_the_two_impossible_tests_proven_failed(self):
        prov = multimodal_provenance()
        assert prov["spatial_dice_agreement"].name == "PROVEN_FAILED"
        assert prov["engine_per_patient_discrimination"].name == "PROVEN_FAILED"
        assert prov["gdc_clinical_endpoints"].name == "ESTABLISHED"

    def test_provenance_vocabulary_matches_the_repo_enum(self):
        from danon.cure import EvidenceStatus

        for status in multimodal_provenance().values():
            assert status in set(EvidenceStatus)


# ---------------------------------------------------------------------------
# Round-trip and offline behaviour
# ---------------------------------------------------------------------------


class TestEvidenceIo:
    def test_save_load_round_trip(self, tmp_path):
        items = [
            _ev("tcia_x", ModalityStatus.ABSENT, {"n_seg_series": 0, "n_patients": 3},
                name="X"),
        ]
        p = str(tmp_path / "ev.json")
        save_evidence(items, p)
        back = load_evidence(p)
        assert len(back) == 1
        assert back[0].key == "tcia_x"
        assert back[0].status is ModalityStatus.ABSENT
        assert back[0].detail["n_patients"] == 3

    def test_missing_evidence_raises_rather_than_defaulting(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            load_evidence(str(tmp_path / "nope.json"))
        assert "will not invent availability numbers" in str(exc.value)

    def test_local_inventory_counts_real_files(self):
        items = probe_local_inventory()
        by_key = {m.key: m for m in items}
        assert by_key["chembl_measured_potency"].status is ModalityStatus.LOCAL
        assert int(by_key["chembl_measured_potency"].get("n_activity_records_in_first")) > 100

    def test_candidate_list_includes_the_two_expected_collections(self):
        assert "TCGA-GBM" in CANDIDATE_GLIOMA_COLLECTIONS
        assert "UPENN-GBM" in CANDIDATE_GLIOMA_COLLECTIONS


# ---------------------------------------------------------------------------
# Paired-control aggregation (test 3)
# ---------------------------------------------------------------------------


def _card(pf_pol, pf_mtd, pfs_ext, tox, b_pol, b_mtd):
    return {
        "day_365": {
            "progression_free_policy": pf_pol, "progression_free_mtd": pf_mtd,
            "pfs_extension": pfs_ext, "toxicity_reduction": tox,
            "burden_policy": b_pol, "burden_mtd": b_mtd,
        }
    }


class TestPairedControlAggregation:
    def test_counts_progression_free_arms_and_medians(self):
        payload = {"cases": [
            _card(True, False, 0.25, 0.15, 1e-18, 2000.0),
            _card(True, True, 0.00, 0.16, 2e-18, 1500.0),
            _card(True, False, 0.34, 0.14, 4e-13, 3600.0),
        ]}
        s = summarize_paired_control(payload)
        assert isinstance(s, PairedControlSummary)
        assert s.n_cards == 3
        assert s.n_progression_free_policy == 3
        assert s.n_progression_free_mtd == 1
        assert s.median_pfs_extension == pytest.approx(0.25)
        assert s.min_pfs_extension == pytest.approx(0.0)
        assert s.max_pfs_extension == pytest.approx(0.34)
        assert s.median_toxicity_reduction == pytest.approx(0.15)

    def test_censored_extensions_are_counted_and_flagged(self):
        """A horizon-censored extension is a lower bound, and must be labelled."""
        payload = {"cases": [
            _card(True, False, 0.25, 0.15, 1e-18, 2000.0),
            _card(True, True, 0.00, 0.16, 2e-18, 1500.0),
        ]}
        s = summarize_paired_control(payload)
        assert s.n_censored_extensions == 1
        assert any("lower bound" in n for n in s.notes)
        assert any("not a clinical benefit" in n for n in s.notes)

    def test_missing_extensions_do_not_crash_the_median(self):
        payload = {"cases": [dict(day_365={})]}
        s = summarize_paired_control(payload)
        assert s.n_cards == 1
        assert s.median_pfs_extension is None
        assert "no real-geometry cards" not in s.describe()

    def test_empty_payload_is_reported_as_such(self):
        s = summarize_paired_control({})
        assert s.n_cards == 0
        assert "no real-geometry cards available" in s.describe()

    def test_exported_cards_aggregate_to_the_reported_contrast(self):
        """The committed artifact must produce the headline numbers, not others."""
        cards = load_paired_control_cards()
        if cards is None:
            pytest.skip("no exported outcome cards on disk")
        s = summarize_paired_control(cards)
        assert s.n_cards == 9
        assert s.n_progression_free_policy == 9
        assert s.n_progression_free_mtd == 3
        assert s.median_pfs_extension == pytest.approx(0.267, abs=5e-3)
        # Re-measured after the resistance floors landed: the toxicity
        # reduction fell from 0.156 to 0.0132. The collateral switch used to
        # toggle on a resistant fraction computed for a sub-detectable tumour
        # (the degenerate R/(S+R)=1.0 of an eradicated-looking lesion), taking
        # drug-B holidays on the strength of a number that meant nothing.
        # With the detection floor in place it stops doing that, so the
        # adaptive arm's toxicity advantage largely disappears. The PFS
        # advantage does not.
        assert s.median_toxicity_reduction == pytest.approx(0.0132, abs=5e-3)
        assert s.n_censored_extensions == 6

    def test_report_without_measurements_stays_cheap_and_complete(self):
        cheap = render_multimodal_report(include_measured=False)
        assert "Measured outcomes" not in cheap
        assert "REFUSED" in cheap
        full = render_multimodal_report(include_measured=True)
        assert "Measured outcomes" in full
        assert "Test 1" in full and "Test 3" in full
        assert "age alone" in full
        assert "lower bound" in full
