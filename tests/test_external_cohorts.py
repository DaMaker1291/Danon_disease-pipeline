"""The external-cohort module: measured where the files exist, refused elsewhere.

These tests pin the two claims the module makes. First, that the CGGA numbers
are a real transport of the repository's own TCGA-fitted model with frozen
coefficients -- recomputed here from the cached files, not read back from the
artifact. Second, that the four pipelines with no attachable metric stay
refused, with the live verdicts that refuse them, so a later edit cannot turn
'nothing to score' into a number.

The cached CGGA release is the source of truth; if it is replaced, the counts
asserted here move, which is the point: the claim sentence quotes them.
"""

import json
import os

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.eval_targets import TargetStatus
from danon.external_cohorts import (
    CGGA_325_ZIP,
    CGGA_693_ZIP,
    CGGA_OOD_RECORD,
    ENGINE_CAVEAT,
    PITCH_FIGURES,
    cgga_ood_report,
    external_cohorts_provenance,
    external_pipeline_matrix,
    pitch_corrections,
    read_cgga_clinical,
    what_external_data_cannot_show,
)

needs_cgga = pytest.mark.skipif(
    not (os.path.exists(CGGA_325_ZIP) and os.path.exists(CGGA_693_ZIP)),
    reason="CGGA clinical releases not cached under data/cgga",
)

HAS_RECORD = os.path.exists(CGGA_OOD_RECORD)


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


@needs_cgga
def test_the_325_release_is_not_325_glioblastomas():
    """The pitch's 'n=325 GBM' is a misread of a 325-row file."""
    cohort = read_cgga_clinical(CGGA_325_ZIP)
    assert cohort.n_table_rows == 325
    assert sum(1 for r in cohort.rows if r.is_gbm) == 85
    assert len(cohort.primary_gbm()) <= 85
    assert len(cohort.primary()) < cohort.n_table_rows


@needs_cgga
def test_the_693_release_is_read_and_its_counts_recorded():
    """140 GBM rows in the table, of which 7 have no usable endpoint."""
    cohort = read_cgga_clinical(CGGA_693_ZIP)
    assert cohort.n_table_rows == 693
    scorable = sum(1 for r in cohort.rows if r.is_gbm)
    assert scorable + cohort.dropped_histology.get("GBM", 0) == 140
    assert scorable == 133
    assert cohort.n_events > 0


@needs_cgga
@pytest.mark.parametrize("path", [CGGA_325_ZIP, CGGA_693_ZIP])
def test_drops_are_counted_not_imputed(path):
    cohort = read_cgga_clinical(path)
    accounted = (
        cohort.n
        + cohort.dropped_no_age
        + cohort.dropped_no_endpoint
        + cohort.dropped_no_sex
    )
    assert accounted == cohort.n_table_rows
    # every kept row carries an event flag and a positive time
    assert all(r.os_days > 0 and r.event in (0, 1) for r in cohort.rows)


@needs_cgga
def test_padded_sex_values_are_stripped_and_counted():
    """The 693 release carries 'Female ' / 'Male  ' in a handful of rows."""
    cohort = read_cgga_clinical(CGGA_693_ZIP)
    assert cohort.sex_values_normalised > 0
    assert cohort.dropped_no_sex == 0


# ---------------------------------------------------------------------------
# The transport test
# ---------------------------------------------------------------------------


@needs_cgga
def test_coefficients_are_the_tcga_fit_and_agree_with_the_recorded_reference():
    """No coefficient may be fitted on CGGA: compare with a direct TCGA fit."""
    from danon.predictive import cox_fit, load_cohort_rows

    report = cgga_ood_report((CGGA_325_ZIP,))
    rows, _ = load_cohort_rows(endpoint="os")
    X = np.array([[r.age_years, r.male, r.is_glioblastoma] for r in rows], dtype=float)
    t = np.array([r.time_days for r in rows], dtype=float)
    e = np.array([r.event for r in rows], dtype=int)
    fit = cox_fit(X, t, e, ("age_years", "male", "is_glioblastoma"))
    assert report.reference.names == ("age_years", "male", "is_glioblastoma")
    assert list(report.reference.beta) == pytest.approx(list(fit.coefficients), abs=1e-12)
    # the internal reference is the number the repository already records
    assert abs(report.reference.internal_c - 0.6559) < 5e-4
    assert report.reference.n == 595


@needs_cgga
def test_manual_rescore_reproduces_the_reported_c_index():
    """Risk = X @ frozen beta, scored with the repo's own Harrell C."""
    from danon.predictive import harrell_c_index

    report = cgga_ood_report((CGGA_325_ZIP,))
    cohort = report.cohorts[os.path.basename(CGGA_325_ZIP)]
    rows = cohort.primary()
    X = np.array([[r.age_years, r.male, r.is_gbm] for r in rows], dtype=float)
    risk = X @ np.array(report.reference.beta, dtype=float)
    time = np.array([r.os_days for r in rows], dtype=float)
    event = np.array([r.event for r in rows], dtype=int)
    c = harrell_c_index(risk, time, event)
    reported = report.tests[os.path.basename(CGGA_325_ZIP)][0]
    assert c.estimable
    assert reported.c_index == pytest.approx(c.c_index, abs=1e-12)
    assert (reported.n, reported.n_events) == (len(rows), int(event.sum()))


@needs_cgga
def test_gbm_subset_is_age_and_sex_because_the_dummy_is_constant():
    report = cgga_ood_report((CGGA_325_ZIP,))
    cohort = report.cohorts[os.path.basename(CGGA_325_ZIP)]
    gbm = cohort.primary_gbm()
    assert gbm and all(r.is_gbm == 1 for r in gbm)
    dummy_col = np.array([r.is_gbm for r in gbm], dtype=float)
    assert len(set(dummy_col)) == 1
    # the reported number for this subset is therefore C(age, sex) -- and the
    # test asserts it lands where age alone does, not above it
    reported = report.tests[os.path.basename(CGGA_325_ZIP)][1]
    assert abs(reported.c_index - reported.age_only_c_index) < 0.02


@needs_cgga
def test_transport_is_never_described_as_engine_validation():
    assert (
        external_cohorts_provenance()["ood_transport_is_engine_validation"]
        is EvidenceStatus.PROVEN_FAILED
    )
    report = cgga_ood_report((CGGA_325_ZIP,))
    assert ENGINE_CAVEAT in report.notes
    assert any("null" in n or "not a validation" in n for n in report.notes)


@needs_cgga
def test_report_is_deterministic():
    a = cgga_ood_report((CGGA_325_ZIP, CGGA_693_ZIP))
    b = cgga_ood_report((CGGA_325_ZIP, CGGA_693_ZIP))
    for key in a.tests:
        for ta, tb in zip(a.tests[key], b.tests[key]):
            assert ta.c_index == tb.c_index
            assert ta.n == tb.n


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_matrix_holds_five_pipelines_and_measures_only_the_transport():
    matrix = external_pipeline_matrix()
    assert [p.index for p in matrix] == [1, 2, 3, 4, 5]
    assert matrix[0].status is TargetStatus.MEASURED
    for p in matrix[1:]:
        assert p.status is not TargetStatus.MEASURED, p.name
        assert p.detail.strip()


def test_scrna_refusal_names_the_dataset_as_reachable():
    matrix = external_pipeline_matrix()
    p = matrix[1]
    assert p.status is TargetStatus.REFUSED_NO_DATA
    assert "GSE131928" in p.dataset
    assert "reachable" in p.detail
    assert "four" in p.detail  # no four-state prediction exists


def test_in_vitro_refusal_matches_the_live_verdict():
    from danon.eval_targets import in_vitro_target

    live = in_vitro_target()
    p = external_pipeline_matrix()[2]
    assert p.status is live.status
    assert "extrapolat" in p.detail.lower()


def test_dice_refusal_matches_the_live_verdict():
    from danon.eval_targets import spatial_expansion_target

    p = external_pipeline_matrix()[4]
    assert p.status is spatial_expansion_target().status
    assert p.status is not TargetStatus.MEASURED


def test_inverse_recovery_refusal_names_both_reasons():
    p = external_pipeline_matrix()[3]
    assert p.status is TargetStatus.REFUSED_NO_DATA
    assert "inverse-tensor estimator" in p.detail
    assert "self-consistency" in p.detail


def test_provenance_refuses_every_pitch_metric():
    prov = external_cohorts_provenance()
    for key in (
        "single_cell_state_shift_prediction",
        "in_vitro_ic50_replicator_fit",
        "brainweb_inverse_tensor_recovery",
        "growth_dice_forecast",
        "cgga_pitch_figures_are_results",
    ):
        assert prov[key] is EvidenceStatus.PROVEN_FAILED, key


def test_corrections_carry_every_pitch_figure_and_the_snippet_defects():
    text = json.dumps(pitch_corrections())
    for figure in PITCH_FIGURES.values():
        assert figure.split(":")[0][:20] in text or figure[:20] in text, figure
    assert "0.6412" in text
    assert "325" in text
    assert "0.035" in text
    assert "0.620" in text
    assert "lifelines" in text


def test_corrections_count_the_files_rather_than_repeating_the_pitch():
    counts = pitch_corrections()[0]["what_is_recorded_instead"]
    if not counts:  # cache absent: the correction is empty, not wrong
        pytest.skip("CGGA releases not cached")
    assert counts[os.path.basename(CGGA_325_ZIP)]["table_rows"] == 325
    assert counts[os.path.basename(CGGA_325_ZIP)]["histology_GBM_in_table"] == 85
    assert counts[os.path.basename(CGGA_325_ZIP)]["histology_GBM_scorable"] == 85
    assert counts[os.path.basename(CGGA_693_ZIP)]["table_rows"] == 693
    assert counts[os.path.basename(CGGA_693_ZIP)]["histology_GBM_in_table"] == 140
    assert counts[os.path.basename(CGGA_693_ZIP)]["histology_GBM_scorable"] == 133


def test_what_external_data_cannot_show_is_not_empty_and_names_the_engine():
    notes = what_external_data_cannot_show()
    assert notes and any("engine" in n for n in notes)


# ---------------------------------------------------------------------------
# The record and the claim check
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RECORD, reason="outputs_real/cgga_ood.json not built")
@needs_cgga
def test_record_matches_a_live_report():
    live = cgga_ood_report((CGGA_325_ZIP, CGGA_693_ZIP))
    with open(CGGA_OOD_RECORD) as fh:
        record = json.load(fh)
    for key, tests in live.tests.items():
        rows = record["tests"][key]
        for t, r in zip(tests, rows):
            assert (t.n, t.n_events) == (r["n"], r["n_events"])
            assert t.c_index == pytest.approx(r["c_index"], abs=1e-12)
            assert t.age_only_c_index == pytest.approx(r["age_only_c_index"], abs=1e-12)


@pytest.mark.skipif(not HAS_RECORD, reason="outputs_real/cgga_ood.json not built")
@needs_cgga
def test_claim_check_passes_on_the_cached_files():
    from danon.claims import check_cgga_transport_is_measured_not_asserted

    result = check_cgga_transport_is_measured_not_asserted()
    assert result.supported, result.detail
