"""Tests for the covariate-intake module.

Three classes of assertion, kept separate:

* **Real-export anchors.** The censoring recovery and the KPS leakage contrast
  are pinned against the cached GDC export, because both are claims about real
  data and a synthetic fixture cannot decide them.
* **Refusal semantics.** The timepoint gate, `'Unknown'` parsing, and the
  protocol-restriction refusal are pinned as refusals, so a later change that
  quietly starts using a post-treatment measurement fails here.
* **Synthetic fixtures.** Parsing, the MGMT file reader, reachability with an
  injected fetcher, and treatment counting are exercised offline.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.cohort_features import (
    BASE_COVARIATES,
    KPS_BASELINE_TIMEPOINTS,
    KPS_POST_BASELINE_TIMEPOINTS,
    MGMT_METHYLATION_CUTOFF,
    MGMT_PROBES,
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
    treatment_exposure,
    what_covariate_intake_cannot_show,
)
from danon.cure import EvidenceStatus
from danon.predictive import PatientRow, load_cohort_rows

CLINICAL_PATH = "data/gdc_clinical/tcga_gbm_cases.json"

pytestmark = pytest.mark.skipif(
    not os.path.exists(CLINICAL_PATH), reason="cached GDC clinical export absent"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseKps:
    def test_unknown_is_missing_not_zero(self):
        # 'Unknown' read as 0 would record a KPS of 0, which is a dead patient.
        assert parse_kps("Unknown") is None
        assert parse_kps("unknown") is None
        assert parse_kps("Not Reported") is None
        assert parse_kps("NA") is None
        assert parse_kps("") is None
        assert parse_kps(None) is None

    def test_numeric_strings_parse(self):
        assert parse_kps("80") == 80.0
        assert parse_kps(" 90 ") == 90.0
        assert parse_kps(70) == 70.0
        assert parse_kps("100") == 100.0
        assert parse_kps("40") == 40.0

    def test_out_of_range_is_refused_rather_than_clamped(self):
        assert parse_kps("-10") is None
        assert parse_kps("120") is None
        assert parse_kps("0") == 0.0   # a real (if terminal) recorded value


# ---------------------------------------------------------------------------
# The temporal gate
# ---------------------------------------------------------------------------


def _case(submitter_id: str, follow_ups: list) -> dict:
    return {"submitter_id": submitter_id, "follow_ups": follow_ups}


class TestTemporalGate:
    def test_baseline_kps_refuses_post_treatment_timepoints(self):
        cases = [
            _case("A", [{"timepoint_category": "Preoperative",
                         "karnofsky_performance_status": "80"}]),
            _case("B", [{"timepoint_category": "Post Adjuvant Therapy",
                         "karnofsky_performance_status": "50"}]),
            _case("C", [{"timepoint_category": "Post Initial Treatment",
                         "karnofsky_performance_status": "60"}]),
            _case("D", [{"timepoint_category": "Prior to Adjuvant Therapy",
                         "karnofsky_performance_status": "90"}]),
        ]
        base = baseline_kps(cases)
        assert base == {"A": 80.0, "D": 90.0}
        # The post-treatment values exist and are reachable, but only through
        # the contrast function -- never as a baseline feature.
        assert post_baseline_kps(cases) == {"B": 50.0}
        assert set(base) & {"B", "C", "D"} == {"D"}

    def test_unknown_is_dropped_from_coverage(self):
        cases = [
            _case("A", [{"timepoint_category": "Preoperative",
                         "karnofsky_performance_status": "Unknown"}]),
        ]
        assert baseline_kps(cases) == {}

    def test_the_two_timepoint_sets_are_disjoint(self):
        assert not (KPS_BASELINE_TIMEPOINTS & KPS_POST_BASELINE_TIMEPOINTS)

    def test_census_marks_kps_timepoints_as_carrying_no_day_stamp(self):
        cases = [
            _case("A", [{"timepoint_category": "Preoperative",
                         "karnofsky_performance_status": "80"}]),
            _case("B", [{"timepoint_category": "Follow-up",
                         "days_to_follow_up": 300,
                         "karnofsky_performance_status": None}]),
        ]
        census = {c.timepoint: c for c in kps_timepoint_census(cases)}
        assert census["Preoperative"].carries_day_stamp is False
        assert census["Preoperative"].n_numeric_kps == 1
        assert census["Follow-up"].carries_day_stamp is True
        assert census["Follow-up"].n_kps_records == 0


class TestRealExportKpsProperties:
    def test_kps_and_day_stamps_are_disjoint_on_the_real_export(self):
        """The structural fact that makes the timepoint gate necessary."""
        with open(CLINICAL_PATH) as fh:
            cases = json.load(fh)["hits"]
        census = {c.timepoint: c for c in kps_timepoint_census(cases)}
        for tp, row in census.items():
            if row.n_numeric_kps:
                assert not row.carries_day_stamp, (
                    f"{tp} carries both KPS and a day stamp; the gate's premise "
                    f"(no absolute day for KPS records) no longer holds"
                )

    def test_leakage_contrast_shows_post_treatment_outranking_pre_treatment(self):
        rows, _ = load_cohort_rows(endpoint="os")
        with open(CLINICAL_PATH) as fh:
            cases = json.load(fh)["hits"]
        contrast = kps_leakage_contrast(rows, cases)
        assert contrast.baseline_n >= 100
        assert contrast.post_baseline_n >= 50
        assert contrast.post_baseline_c > contrast.baseline_c
        assert "LEAKAGE" in contrast.verdict

    def test_kps_covers_a_minority_of_the_cohort(self):
        with open(CLINICAL_PATH) as fh:
            cases = json.load(fh)["hits"]
        cov = kps_coverage(cases)
        assert cov.n_baseline < cov.n_cohort
        assert 0.2 < cov.baseline_fraction < 0.8
        assert "KPS:" in str(cov)


# ---------------------------------------------------------------------------
# Measured covariate effect
# ---------------------------------------------------------------------------


def _rows(n: int, *, seed: int = 0) -> list:
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        out.append(
            PatientRow(
                submitter_id=f"P{i}",
                age_years=float(40 + rng.randint(0, 40)),
                male=int(rng.randint(0, 2)),
                is_glioblastoma=1,
                time_days=float(50 + rng.exponential(300)),
                event=int(rng.rand() < 0.7),
            )
        )
    return out


class TestCovariateEffect:
    def test_a_constant_covariate_is_reported_as_no_measurable_gain(self):
        rows = _rows(180)
        flat = {r.submitter_id: 80.0 for r in rows}
        eff = covariate_effect(rows, flat, name="kps_baseline")
        assert eff.n_patients == 180
        assert eff.n_dropped_by_availability == 0
        assert abs(eff.delta) < max(eff.base_sd, 1e-9)
        assert eff.verdict == "no measurable gain"

    def test_a_planted_covariate_that_tracks_the_outcome_shows_a_gain(self):
        rows = _rows(200, seed=3)
        rng = np.random.RandomState(11)
        # KPS is protective -- higher value, longer time -- but noisy, because a
        # covariate that tracks the outcome perfectly separates it and leaves
        # the partial likelihood unbounded. That case is pinned separately below.
        planted = {
            r.submitter_id: float(np.clip(70 + 0.05 * (r.time_days - 200) + rng.normal(0, 15), 0, 100))
            for r in rows
        }
        eff = covariate_effect(rows, planted, name="kps_baseline")
        assert eff.folds_complete
        assert eff.delta > 0
        assert eff.verdict == "gain exceeds fold noise"
        assert eff.hazard_ratio is not None and eff.hazard_ratio < 1.0

    def test_a_separating_covariate_is_not_estimable_rather_than_a_huge_gain(self):
        """A covariate that encodes the outcome makes the likelihood unbounded.

        The old behaviour was a silent nan whose verdict read as 'not estimable'
        with no explanation; `folds_complete` now makes the cause explicit.
        """
        rows = _rows(200, seed=5)
        planted = {r.submitter_id: float(np.clip(120 - r.time_days / 12, 0, 100)) for r in rows}
        eff = covariate_effect(rows, planted, name="kps_baseline")
        assert not np.isfinite(eff.augmented_c)
        assert not eff.folds_complete
        assert "did not converge" in eff.verdict
        assert any("unbounded" in n for n in eff.notes)

    def test_an_absent_covariate_is_refused_rather_than_fitted_on_nothing(self):
        rows = _rows(120)
        eff = covariate_effect(rows, {}, name="mgmt_methylation")
        assert eff.n_patients == 0
        assert eff.verdict == "not estimable"
        assert any("refused" in n for n in eff.notes)

    def test_requiring_a_covariate_reports_what_it_costs(self):
        rows = _rows(160)
        partial = {r.submitter_id: 75.0 for r in rows[:40]}
        eff = covariate_effect(rows, partial, name="kps_baseline")
        assert eff.n_patients == 40
        assert eff.n_dropped_by_availability == 120
        assert any("dropped" in n for n in eff.notes)
        assert any("not comparable" in n for n in eff.notes)


# ---------------------------------------------------------------------------
# MGMT
# ---------------------------------------------------------------------------


class TestMgmtParsing:
    def test_probe_pair_is_the_documented_one(self):
        assert set(MGMT_PROBES) == {"cg12434587", "cg12981137"}

    def _write(self, tmp_path, rows):
        path = tmp_path / "betas.txt"
        path.write_text("".join(f"{p}\t{v}\n" for p, v in rows))
        return str(path)

    def test_extracts_the_probe_pair_and_calls_methylated(self, tmp_path):
        path = self._write(tmp_path, [
            ("cg00000029", "0.5"),
            ("cg12434587", "0.9"),
            ("cg12981137", "0.8"),
            ("cg00000108", "0.1"),
        ])
        res = extract_mgmt_beta(path, submitter_id="X")
        assert res.probe_values == {"cg12434587": 0.9, "cg12981137": 0.8}
        assert res.missing_probes == []
        assert res.mean_beta == pytest.approx(0.85)
        assert res.call == "methylated"
        assert res.n_probes_in_file == 4

    def test_unmethylated_below_the_cutoff(self, tmp_path):
        path = self._write(tmp_path, [("cg12434587", "0.1"), ("cg12981137", "0.1")])
        res = extract_mgmt_beta(path)
        assert res.call == "unmethylated"
        assert res.mean_beta < MGMT_METHYLATION_CUTOFF

    def test_na_is_missing_and_never_imputed(self, tmp_path):
        path = self._write(tmp_path, [("cg12434587", "NA"), ("cg12981137", "0.7")])
        res = extract_mgmt_beta(path)
        assert "cg12434587" in res.missing_probes
        assert res.probe_values == {"cg12981137": 0.7}
        assert res.mean_beta == pytest.approx(0.7)

    def test_no_usable_probe_is_not_callable(self, tmp_path):
        path = self._write(tmp_path, [("cg00000029", "0.5")])
        res = extract_mgmt_beta(path)
        assert res.mean_beta is None
        assert res.call == "not callable: no usable probe"
        assert set(res.missing_probes) == set(MGMT_PROBES)

    def test_the_reader_works_on_a_real_gdc_file_when_present(self):
        real = "data/gdc_methylation/TCGA-76-4928_450k_betas.txt"
        if not os.path.exists(real):
            pytest.skip("real 450k beta file not cached")
        res = extract_mgmt_beta(real, submitter_id="TCGA-76-4928")
        assert res.n_probes_in_file > 400_000
        assert res.missing_probes == []
        assert 0.0 <= res.mean_beta <= 1.0


class TestMgmtReachability:
    def test_uses_the_cache_without_touching_the_network(self, tmp_path, monkeypatch):
        cache = tmp_path / "reach.json"
        cache.write_text(json.dumps({
            "n_beta_files": 450, "n_patients_total": 200, "n_patients_450k": 142,
            "n_patients_27k": 200, "all_open_access": True, "source": "cached",
        }))
        called = []

        def boom(url):
            called.append(url)
            raise AssertionError("network hit despite a warm cache")

        res = probe_mgmt_reachability(fetcher=boom, cache_path=str(cache))
        assert called == []
        assert res.n_patients_450k == 142
        assert "142" in res.verdict

    def test_counts_patients_not_files(self, tmp_path):
        """155 files across 142 patients: the file count is the wrong number."""
        def fetcher(url):
            if "Illumina+Human+Methylation+450" in url:
                total, n_pat, open_ = 155, 142, True
            elif "Methylation+27" in url:
                total, n_pat, open_ = 295, 200, True
            else:
                total, n_pat, open_ = 450, 200, True
            return {"data": {
                "pagination": {"total": total},
                "aggregations": {
                    "cases.submitter_id": {"buckets": [{"key": f"p{i}", "doc_count": 1} for i in range(n_pat)]},
                    "access": {"buckets": [{"key": "open" if open_ else "controlled", "doc_count": total}]},
                },
            }}

        res = probe_mgmt_reachability(fetcher=fetcher, cache_path=str(tmp_path / "r.json"), use_cache=False)
        assert res.n_patients_450k == 142
        assert res.n_beta_files == 450
        assert res.all_open_access is True


# ---------------------------------------------------------------------------
# Treatment exposure
# ---------------------------------------------------------------------------


def _tx(submitter_id, agents, *, therapy_type=None, start_day=None):
    records = []
    for agent in agents:
        records.append({
            "therapeutic_agents": agent,
            "treatment_type": therapy_type or "Pharmaceutical Therapy, NOS",
            "days_to_treatment_start": start_day,
        })
    return {"submitter_id": submitter_id, "diagnoses": [{"treatments": records}]}


class TestTreatmentExposure:
    def test_detects_radiotherapy_plus_temozolomide(self):
        hits = [
            _tx("A", ["Temozolomide"], therapy_type="Radiation Therapy, NOS"),
            _tx("B", ["Temozolomide"]),
            _tx("C", ["Bevacizumab"]),
        ]
        exp = treatment_exposure(hits)
        assert exp.n_cases_rt_plus_tmz == 1
        assert exp.n_cases_with_agent == 3
        assert exp.n_records == 3
        assert ("Temozolomide", 2) in exp.top_agents

    def test_unstamped_records_block_a_protocol_restriction(self):
        hits = [_tx("A", ["Temozolomide"], therapy_type="Radiation Therapy, NOS")]
        exp = treatment_exposure(hits)
        assert exp.n_records_with_start_day == 0
        assert exp.can_restrict_to_protocol is False
        assert "treatment-EXPOSURE" in exp.verdict

    def test_mostly_stamped_records_would_permit_a_restriction(self):
        hits = [_tx(f"P{i}", ["Temozolomide"], start_day=10) for i in range(10)]
        hits.append(_tx("Q", ["Temozolomide"]))
        exp = treatment_exposure(hits)
        assert exp.can_restrict_to_protocol is True

    def test_no_records_at_all_is_reported_not_divided_by_zero(self):
        """The cached clinical export carries no treatment node at all."""
        exp = treatment_exposure([])
        assert exp.n_records == 0
        assert exp.can_restrict_to_protocol is False
        assert "no treatment records" in exp.verdict
        assert str(exp)  # must not raise


# ---------------------------------------------------------------------------
# The report and its honesty surface
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_carries_no_estimated_delta(self):
        """No authored gain may appear; every delta in it was computed."""
        text = cohort_feature_report(n_folds=3)
        # The fabricated roadmap's estimates, which are not measurements.
        for forbidden in ("+0.05 to +0.08", "0.780 - 0.820", "0.760 - 0.790"):
            assert forbidden not in text
        assert "Nothing in this report is an estimate" in text

    def test_report_states_the_corrected_censoring(self):
        text = cohort_feature_report(n_folds=3)
        assert "censored at last contact" in text
        assert "no measurable gain" in text or "gain exceeds fold noise" in text

    def test_provenance_marks_the_refusals(self):
        prov = cohort_feature_provenance()
        assert prov["treatment_protocol_restriction"] is EvidenceStatus.PROVEN_FAILED
        assert prov["subregion_segmentation"] is EvidenceStatus.PROVEN_FAILED
        assert prov["mgmt_methylation_cutoff"] is EvidenceStatus.UNKNOWN
        assert prov["kps_measured_delta_c"] is EvidenceStatus.ESTABLISHED

    def test_cannot_show_list_is_specific(self):
        items = what_covariate_intake_cannot_show()
        assert len(items) >= 4
        assert any("not about the engine" in i for i in items)
        assert any("not treatment protocol" in i for i in items)


# ---------------------------------------------------------------------------
# The censoring defect
# ---------------------------------------------------------------------------


class TestCensoringRecovery:
    """The defect this module documents: the loader searched the wrong node."""

    def test_os_endpoint_recovers_censored_patients(self):
        rows, notes = load_cohort_rows(endpoint="os")
        events = sum(r.event for r in rows)
        censored = len(rows) - events
        assert censored > 0, "every patient is an event: censoring is broken again"
        assert censored >= 90
        assert any("censored at last contact" in n for n in notes)
        assert not any("WARNING" in n for n in notes)

    def test_progression_endpoint_recovers_censored_patients(self):
        rows, notes = load_cohort_rows(endpoint="progression")
        censored = len(rows) - sum(r.event for r in rows)
        assert censored > 0
        assert censored >= 150

    def test_no_patient_with_a_recorded_death_day_is_censored(self):
        """The invariant: a recorded death day is always an event.

        One patient in the export is recorded Dead with no `days_to_death`, and
        is censored at last contact because there is no event time available.
        That is counted in the notes rather than absorbed silently, so the
        assertion here is about patients who *do* have a death day.
        """
        import json as _json

        with open(CLINICAL_PATH) as fh:
            hits = _json.load(fh)["hits"]
        with_death_day = {
            str(h.get("submitter_id"))
            for h in hits
            if (h.get("demographic") or {}).get("days_to_death") is not None
        }
        rows, notes = load_cohort_rows(endpoint="os")
        censored = {r.submitter_id for r in rows if not r.event}
        assert censored
        assert not (censored & with_death_day)
        # The one dead-without-day patient is reported, not hidden.
        assert any("no days_to_death" in n for n in notes)

    def test_most_censored_patients_are_recorded_alive(self):
        import json as _json

        with open(CLINICAL_PATH) as fh:
            hits = _json.load(fh)["hits"]
        status = {
            str(h.get("submitter_id")): str((h.get("demographic") or {}).get("vital_status"))
            for h in hits
        }
        rows, _ = load_cohort_rows(endpoint="os")
        censored = [r.submitter_id for r in rows if not r.event]
        assert sum(1 for s in censored if status.get(s) == "Alive") >= 0.9 * len(censored)

    def test_all_censored_is_flagged_rather_than_silently_estimated(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"hits": [{
            "submitter_id": "A",
            "demographic": {"vital_status": "Alive", "age_at_index": 60, "sex_at_birth": "Male"},
            "diagnoses": [{"primary_diagnosis": "Glioblastoma"}],
            "follow_ups": [{"days_to_follow_up": 100}],
        }]}))
        rows, notes = load_cohort_rows(str(p), endpoint="os")
        assert len(rows) == 1 and rows[0].event == 0
        assert any("censored at last contact" in n for n in notes)

    def test_no_censoring_triggers_the_warning(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"hits": [{
            "submitter_id": "A",
            "demographic": {"vital_status": "Dead", "days_to_death": 200,
                            "age_at_index": 60, "sex_at_birth": "Female"},
            "diagnoses": [{"primary_diagnosis": "Glioblastoma"}],
            "follow_ups": [],
        }]}))
        _, notes = load_cohort_rows(str(p), endpoint="os")
        assert any("WARNING" in n for n in notes)
