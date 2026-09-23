"""Tests for the temporal-leakage and out-of-sample audit trace.

The permutation test is the load-bearing one: it refits on shuffled outcomes and
requires held-out discrimination to collapse to chance. That is a real leakage
detector rather than a bookkeeping assertion, and it is the reason this file
exists.

Permutations are subsampled (few patients, few draws) so the suite stays fast.
The resolution that costs is stated in the module and in the checks themselves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from danon.audit import (
    FORBIDDEN_DESIGN_COLUMNS,
    PERMITTED_DESIGN_COLUMNS,
    AuditCheck,
    PermutationNull,
    audit_provenance,
    design_columns_audit,
    determinism_audit,
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


@pytest.fixture(scope="module")
def small_rows():
    from danon.predictive import load_cohort_rows

    rows, _ = load_cohort_rows(endpoint="os")
    return list(rows[:80])


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------


class TestStructuralLeakage:
    def test_design_matrix_excludes_every_outcome_column(self):
        check = design_columns_audit()
        assert isinstance(check, AuditCheck)
        assert check.passed
        assert "forbidden columns present: none" in check.detail
        assert "unpermitted columns present: none" in check.detail

    def test_forbidden_and_permitted_sets_do_not_overlap(self):
        assert not set(PERMITTED_DESIGN_COLUMNS) & set(FORBIDDEN_DESIGN_COLUMNS)
        assert "time_days" in FORBIDDEN_DESIGN_COLUMNS
        assert "event" in FORBIDDEN_DESIGN_COLUMNS

    def test_fold_membership_scores_every_patient_exactly_once(self, small_rows):
        check = fold_membership_audit(small_rows, n_folds=5, seed=0)
        assert check.passed
        assert check.observed == float(len(small_rows))
        assert "test appearances 1-1" in check.detail
        assert "train appearances 4-4" in check.detail

    def test_fold_membership_is_reproducible_for_a_seed(self, small_rows):
        a = fold_membership_audit(small_rows, n_folds=4, seed=3)
        b = fold_membership_audit(small_rows, n_folds=4, seed=3)
        assert a.detail == b.detail
        assert a.passed and b.passed


# ---------------------------------------------------------------------------
# The leakage detector
# ---------------------------------------------------------------------------


class TestPermutationNull:
    def test_null_centres_at_chance_on_real_patients(self, small_rows):
        """The required result: shuffling outcomes destroys discrimination."""
        pn = permutation_null_c_index(
            small_rows, n_permutations=4, n_folds=4, seed=0, endpoint="os"
        )
        assert isinstance(pn, PermutationNull)
        assert pn.n_permutations >= 1
        assert abs(pn.null_median - 0.5) <= 0.10, pn.detail
        assert pn.null_low <= pn.null_median <= pn.null_high

    def test_real_covariates_beat_the_null_band(self, small_rows):
        """Strength, reported separately from the leakage requirement."""
        pn = permutation_null_c_index(
            small_rows, n_permutations=4, n_folds=4, seed=0, endpoint="os"
        )
        # Age and diagnosis carry genuine prognostic signal, so the observed
        # value should sit above the shuffled band.
        assert pn.observed > pn.null_high
        assert pn.separation > 0

    def test_permutation_audit_passes_when_the_null_is_centred(self, small_rows):
        check = permutation_leakage_audit(
            small_rows, n_permutations=4, n_folds=4, seed=0, endpoint="os"
        )
        assert check.passed
        assert check.null_center is not None
        assert "null centre" in check.describe()

    def test_reports_its_own_resolution_rather_than_a_p_value(self, small_rows):
        pn = permutation_null_c_index(
            small_rows, n_permutations=5, n_folds=4, seed=0, endpoint="os"
        )
        assert "permutation(s) of (time, event)" in pn.detail
        # The trace must not dress a 5-draw null up as a significance test.
        assert "p-value" not in pn.detail

    def test_too_few_patients_reports_undetermined_rather_than_raising(self):
        """Guarding the permutation loop but not the observed fit crashed the audit."""
        from danon.predictive import PatientRow

        rows = [
            PatientRow(f"p{i}", 50.0 + i, i % 2, 1, 300.0 + 10 * i, 1)
            for i in range(6)
        ]
        pn = permutation_null_c_index(rows, n_permutations=3, n_folds=5, seed=0)
        assert pn.n_permutations == 0
        assert math.isnan(pn.null_median)
        assert "undetermined" in pn.detail

    def test_an_undetermined_null_is_a_failure_not_a_pass(self):
        """'We could not tell' must never render as 'no leak'."""
        from danon.predictive import PatientRow

        rows = [
            PatientRow(f"p{i}", 50.0 + i, i % 2, 1, 300.0 + 10 * i, 1)
            for i in range(6)
        ]
        check = permutation_leakage_audit(rows, n_permutations=3, n_folds=5)
        assert check.passed is False
        assert "has not passed" in check.detail


# ---------------------------------------------------------------------------
# Determinism and the refusal
# ---------------------------------------------------------------------------


class TestDeterminismAndRefusal:
    def test_determinism_check_separates_seeds(self):
        check = determinism_audit(days=10)
        assert check.passed
        assert "reproduces exactly: True" in check.detail
        assert "diverges: True" in check.detail
        assert "claim of full determinism would be false" in check.detail

    def test_per_patient_prediction_is_reported_as_not_estimable(self):
        check = per_patient_prediction_audit()
        assert check.passed is True          # the audit passes by declaring the gap
        assert check.observed is None
        assert "NOT ESTIMABLE" in check.detail
        assert "would be fabricated" in check.detail

    def test_refusal_check_fails_if_a_per_patient_prediction_appears(self, monkeypatch):
        """A future capability must break this check, not slip past it."""
        import danon.predictive as pred

        monkeypatch.setattr(
            pred, "engine_discrimination_verdict",
            lambda: pred.EngineDiscrimination(
                estimable=True, c_index=0.8, reason="now estimable",
                what_would_be_needed=[],
            ),
        )
        check = per_patient_prediction_audit()
        assert check.passed is False


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_run_audit_returns_all_checks(self):
        checks = run_audit(n_permutations=3, n_folds=4, max_patients=80)
        ids = [c.id for c in checks]
        assert ids == [
            "design_matrix_excludes_outcome",
            "out_of_sample_fold_membership",
            "permutation_leakage_null",
            "seeded_reproducibility",
            "per_patient_relapse_prediction",
        ]
        assert all(isinstance(c, AuditCheck) for c in checks)

    def test_trace_is_generated_with_real_numbers(self):
        text = render_audit_trace(
            n_permutations=3, n_folds=4, max_patients=80, endpoint="os"
        )
        assert "PART 1: INGESTION BOUNDARY" in text
        assert "PART 2: OUT-OF-SAMPLE STRUCTURE" in text
        assert "PART 3: LEAKAGE DETECTION" in text
        assert "PART 4:" in text
        assert "null centre" in text
        assert "C = " in text
        # The honest absences must be printed, not merely omitted.
        assert "WHAT IS DELIBERATELY ABSENT" in text
        assert "No per-patient predicted-vs-actual relapse day" in text

    def test_trace_has_no_invented_per_patient_error_figure(self):
        text = render_audit_trace(
            n_permutations=3, n_folds=4, max_patients=80
        )
        # The shape of the fabricated log: a per-patient id plus a day-level
        # absolute error. Neither may appear.
        assert "TCGA-06-0150" not in text
        assert "absolute_percentage_error" not in text
        assert "HIGH_PRECISION_MATCH" not in text
        assert "relapse_prediction_error_days" not in text

    def test_ingestion_boundary_states_the_two_structural_gaps(self):
        b = ingestion_boundary()
        assert b["patient_level_imaging_outcome_join_available"] is False
        assert b["engine_per_patient_prediction_available"] is False
        assert b["single_cell_assay_available"] is False
        assert b["imaging_timepoints_available"] == 1

    def test_ingestion_boundary_notes_are_the_reasons(self):
        joined = " ".join(ingestion_boundary_notes()).lower()
        assert "one imaging timepoint" in joined
        assert "no patient in this repository carries baseline imaging" in joined
        assert "no per-patient prediction to blind" in joined


# ---------------------------------------------------------------------------
# Provenance and limits
# ---------------------------------------------------------------------------


class TestProvenanceAndLimits:
    def test_provenance_marks_the_gap_proven_failed(self):
        prov = audit_provenance()
        assert prov["per_patient_relapse_prediction"].name == "PROVEN_FAILED"
        assert prov["single_cell_input"].name == "PROVEN_FAILED"
        assert prov["permutation_leakage_null"].name == "ESTABLISHED"

    def test_limits_refuse_absence_of_evidence_as_evidence_of_absence(self):
        joined = " ".join(what_the_audit_cannot_establish()).lower()
        assert "absence of detectable leakage is not proof of absence" in joined
        assert "band, not a p-value" in joined
        assert "not an audit of the engine's biology" in joined
        assert "does not mean prospectively held out in time" in joined
