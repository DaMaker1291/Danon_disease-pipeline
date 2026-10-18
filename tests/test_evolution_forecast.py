"""Unit and invariant tests for the real-specimen evolution forecast.

Three layers, deliberately:

* the EXAMPLE BUILDERS on hand-built spectra, where the right answer is known by
  construction -- including the exclusion that keeps task A a measurement;
* the METRICS on a cohort small enough to reason about, testing that a baseline
  is always reported beside the model and that an advantage inside its own
  interval does not read as one;
* the SHIPPED RECORD, re-derived from the cached release, so the artifact the
  app renders cannot drift from the code that produced it.

The point of the last layer is that a forecast which quietly stopped matching
its own record -- or a verdict that started claiming an advantage its interval
does not support -- fails here rather than reaching a screen.
"""

from __future__ import annotations

import json
import os

import pytest

from danon.cure import EvidenceStatus
from danon.evolution_forecast import (
    DetectionExample,
    ForecastError,
    bootstrap_mae_difference,
    calibration_is_within_tolerance,
    calibration_reading,
    calibration_summary,
    pooling_block,
    reliability_bins,
    change_examples,
    cross_validated_change,
    cross_validated_detection,
    detection_examples,
    evaluate_change,
    evaluate_detection,
    forecast_provenance,
    paired_spectra,
    records_from_cache,
    repeated_folds,
    timepoint_census,
    timepoint_of,
    what_this_forecast_cannot_say,
)
from danon.variants import Spectrum, VariantRecord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "glass_paired", "glass_mutations.json")
RECORD = os.path.join(ROOT, "outputs_real", "glass_evolution_forecast.json")


def rec(patient, sample, timepoint, gene, start, alt, ref, protein="p.Ala1Val"):
    return VariantRecord(
        patient_id=patient,
        sample_id=sample,
        timepoint=timepoint,
        gene=gene,
        protein_change=protein,
        chromosome="7",
        start=start,
        ref_allele="C",
        alt_allele="T",
        alt_count=alt,
        ref_count=ref,
        source="test fixture",
    )


def spec(patient, sample, timepoint, *records):
    return Spectrum(
        patient_id=patient, sample_id=sample, timepoint=timepoint, records=tuple(records)
    )


# ---------------------------------------------------------------------------
# The labelling rule and the loader
# ---------------------------------------------------------------------------


def test_timepoint_labels_follow_the_specimen_suffix():
    assert timepoint_of("GLSS-MD-0086-TP") == "Primary"
    assert timepoint_of("GLSS-MD-0086-R1") == "Recurrence"
    assert timepoint_of("GLSS-MD-0086-R12") == "Recurrence"
    # Anything else is UNLABELLED rather than guessed into a fold: calling an
    # unknown suffix a primary would pair a specimen with itself.
    assert timepoint_of("GLSS-MD-0086-XX") == "Unlabelled"
    assert timepoint_of("GLSS-MD-0086") == "Unlabelled"


def test_loader_drops_records_without_both_read_counts():
    raw = {
        "source": "test fixture",
        "records": [
            {
                "patientId": "P1", "sampleId": "P1-TP", "_gene": "TP53",
                "proteinChange": "p.Arg1*", "chr": "7", "startPosition": 100,
                "referenceAllele": "C", "variantAllele": "T",
                "tumorAltCount": 30, "tumorRefCount": 70,
            },
            {
                "patientId": "P1", "sampleId": "P1-TP", "_gene": "TP53",
                "proteinChange": "p.Arg2*", "chr": "7", "startPosition": 200,
                "referenceAllele": "C", "variantAllele": "T",
                "tumorAltCount": None, "tumorRefCount": 70,
            },
            {
                "patientId": "P1", "sampleId": "P1-TP", "_gene": "TP53",
                "proteinChange": "p.Arg3*", "chr": "7", "startPosition": 300,
                "referenceAllele": "C", "variantAllele": "T",
                "tumorAltCount": 0, "tumorRefCount": 0,
            },
        ],
    }
    records, dropped = records_from_cache(raw)
    assert len(records) == 1
    # A declared 0/0 depth is a DROPPED record: an allele fraction is not
    # defined without reads, and entering it as 0.0 would be an invented zero.
    assert dropped == 2
    assert records[0].allele_fraction == pytest.approx(0.30)


def test_pairs_are_within_a_patient_and_ordered():
    records = [
        rec("P1", "P1-TP", "Primary", "TP53", 100, 30, 70),
        rec("P1", "P1-R1", "Recurrence", "TP53", 100, 60, 40),
        rec("P2", "P2-TP", "Primary", "EGFR", 200, 10, 90),
    ]
    spectra, pairs = paired_spectra(records)
    assert len(spectra) == 3
    # P2 has no recurrence, so it contributes NO pair rather than an imputed one.
    assert len(pairs) == 1
    assert pairs[0][0].sample_id == "P1-TP"
    assert pairs[0][1].sample_id == "P1-R1"


# ---------------------------------------------------------------------------
# Task A: the exclusion is the honesty fix, so it is asserted directly
# ---------------------------------------------------------------------------


def test_task_a_scores_only_loci_the_primary_measured():
    primary = spec("P1", "P1-TP", "Primary",
                   rec("P1", "P1-TP", "Primary", "TP53", 100, 30, 70))
    recurrence = spec("P1", "P1-R1", "Recurrence",
                      rec("P1", "P1-R1", "Recurrence", "TP53", 100, 60, 40),
                      rec("P1", "P1-R1", "Recurrence", "EGFR", 200, 50, 50))
    rows = detection_examples([(primary, recurrence)])
    # TP53 is shared (detected again); EGFR was never in the primary, so it is
    # NOT a row -- a 'gained' variant is re-detected by definition.
    assert len(rows) == 1
    assert rows[0].variant.startswith("TP53")
    assert rows[0].detected_at_recurrence is True
    assert rows[0].primary_af == pytest.approx(0.30)
    assert rows[0].primary_depth == 100


def test_a_lost_variant_is_a_negative_row_carrying_its_measured_reading():
    primary = spec("P1", "P1-TP", "Primary",
                   rec("P1", "P1-TP", "Primary", "TP53", 100, 30, 70))
    recurrence = spec("P1", "P1-R1", "Recurrence")
    rows = detection_examples([(primary, recurrence)])
    assert len(rows) == 1
    assert rows[0].detected_at_recurrence is False
    assert rows[0].primary_af == pytest.approx(0.30)
    assert rows[0].recurrence_depth == 0


def test_no_gained_variant_can_enter_task_a():
    """The defect this task was rewritten to remove, asserted as an invariant."""
    primary = spec("P1", "P1-TP", "Primary",
                   rec("P1", "P1-TP", "Primary", "TP53", 100, 30, 70))
    recurrence = spec("P1", "P1-R1", "Recurrence",
                      rec("P1", "P1-R1", "Recurrence", "EGFR", 200, 50, 50))
    rows = detection_examples([(primary, recurrence)])
    assert all(not (r.primary_af == 0.0 and r.detected_at_recurrence) for r in rows)
    # The primary's variant is scored as LOST; the gained one is absent.
    assert [r.detected_at_recurrence for r in rows] == [False]


def test_task_b_uses_both_specimens_depths():
    primary = spec("P1", "P1-TP", "Primary",
                   rec("P1", "P1-TP", "Primary", "TP53", 100, 30, 70))
    recurrence = spec("P1", "P1-R1", "Recurrence",
                      rec("P1", "P1-R1", "Recurrence", "TP53", 100, 60, 40))
    rows = change_examples([(primary, recurrence)])
    assert len(rows) == 1
    assert rows[0].primary_af == pytest.approx(0.30)
    assert rows[0].primary_depth == 100
    assert rows[0].delta_af == pytest.approx(0.60 - 0.30)


# ---------------------------------------------------------------------------
# The scorers: a baseline always sits beside the model, and an interval decides
# ---------------------------------------------------------------------------


def _cohort(n_patients=24, seed=3):
    """A small cohort where a variant's primary fraction carries real signal."""
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n_patients):
        pid = f"P{i:03d}"
        primary_recs, recurrence_recs = [], []
        for j in range(6):
            start = 1000 + j * 10
            af = rng.uniform(0.05, 0.95)
            depth = 200
            alt = int(round(af * depth))
            if rng.random() < 0.5:
                # present in the primary: re-detected when the fraction is high,
                # and re-detected variants MOVE (so task B has resolvable
                # changes rather than a cohort of exact ties)
                primary_recs.append(rec(pid, f"{pid}-TP", "Primary", "TP53", start, alt, depth - alt))
                again = af > 0.5
                if again:
                    moved = max(1, int(round((af - 0.25) * depth)))
                    recurrence_recs.append(
                        rec(pid, f"{pid}-R1", "Recurrence", "TP53", start, moved, depth - moved)
                    )
            else:
                # never in the primary: excluded from task A, scored in task B only
                primary_recs.append(rec(pid, f"{pid}-TP", "Primary", "EGFR", start, alt, depth - alt))
                recurrence_recs.append(
                    rec(pid, f"{pid}-R1", "Recurrence", "EGFR", start, alt, depth - alt)
                )
        rows.append(
            (
                spec(pid, f"{pid}-TP", "Primary", *primary_recs),
                spec(pid, f"{pid}-R1", "Recurrence", *recurrence_recs),
            )
        )
    return rows


def test_detection_reports_every_baseline_beside_the_model():
    result = evaluate_detection(detection_examples(_cohort()))
    m = result.metrics
    for key in (
        "test_prevalence_detected",
        "baseline_majority_accuracy",
        "baseline_majority_auc",
        "auc_primary_af_only",
        "auc_primary_depth_only",
        "auc_fitted_logistic",
        "auc_fitted_logistic_ci_low",
        "auc_fitted_logistic_ci_high",
    ):
        assert key in m, key
    assert m["baseline_majority_auc"] == 0.5
    assert 0.0 <= m["auc_fitted_logistic"] <= 1.0
    assert m["auc_fitted_logistic_ci_low"] <= m["auc_fitted_logistic_ci_high"]


def test_detection_signal_beats_the_nothing_baseline_when_it_is_there():
    """With a real (if artificial) signal the fitted model should clear 0.5."""
    result = evaluate_detection(detection_examples(_cohort()))
    assert result.metrics["auc_fitted_logistic"] > 0.5
    assert result.metrics["auc_primary_af_only"] > 0.5


def test_change_reports_persistence_and_the_majority_direction():
    result = evaluate_change(change_examples(_cohort()))
    m = result.metrics
    for key in (
        "baseline_persistence_mae",
        "baseline_train_mean_mae",
        "model_fitted_mae",
        "mae_advantage_over_persistence",
        "mae_advantage_ci_low",
        "mae_advantage_ci_high",
        "mae_advantage_resolvable",
        "baseline_majority_sign_accuracy",
    ):
        assert key in m, key
    assert m["mae_advantage_over_persistence"] == pytest.approx(
        m["baseline_persistence_mae"] - m["model_fitted_mae"]
    )
    # The sign baseline is the MAJORITY direction, never a flattering 0.5.
    assert m["baseline_majority_sign_accuracy"] >= 0.5


def test_the_split_never_leaves_a_fold_empty():
    # One patient cannot be split: the split refuses rather than putting every
    # row of the single tumour on one side and calling it held out.
    rows = _cohort(n_patients=1)
    with pytest.raises(ForecastError):
        evaluate_detection(detection_examples(rows), train_fraction=0.7)
    with pytest.raises(ForecastError):
        evaluate_change(change_examples(rows), train_fraction=0.7)


def test_a_train_fraction_outside_the_unit_interval_is_refused():
    rows = _cohort(n_patients=6)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ForecastError):
            evaluate_detection(detection_examples(rows), train_fraction=bad)


def test_an_empty_cohort_is_refused_rather_than_scored():
    with pytest.raises(ForecastError):
        evaluate_detection([])
    with pytest.raises(ForecastError):
        evaluate_change([])


def test_bootstrap_difference_needs_enough_patients_for_an_interval():
    lo, hi, share = bootstrap_mae_difference([0.1, 0.2], [0.0, 0.0], ["a", "b"])
    assert (lo, hi, share) == (None, None, None)


def test_bootstrap_difference_is_positive_when_the_model_is_uniformly_better():
    # The model predicts the change exactly, so its error is 0 on every row while
    # persistence (predict no change) errs by 0.5: an advantage of +0.5 that no
    # resample of patients can turn negative.
    y = [0.5] * 20
    pred = [0.5] * 20
    patients = [f"P{i // 4}" for i in range(20)]
    lo, hi, share = bootstrap_mae_difference(pred, y, patients)
    assert lo is not None and lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)
    assert share == 1.0


def test_bootstrap_difference_is_zero_when_the_model_adds_nothing():
    # Persistence is the model: both err by |0.5|, so the advantage is 0 and the
    # interval cannot exclude it however the patients are resampled.
    y = [0.5] * 20
    pred = [0.0] * 20
    patients = [f"P{i // 4}" for i in range(20)]
    lo, hi, share = bootstrap_mae_difference(pred, y, patients)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)
    assert share == 0.0


# ---------------------------------------------------------------------------
# Repeated cross-validation: every patient scored once per repeat, unseen
# ---------------------------------------------------------------------------


def test_repeated_folds_partition_patients_every_repeat():
    ids = [f"P{i:03d}" for i in range(23)]
    splits = repeated_folds(ids, n_folds=5, n_repeats=3, seed=5)
    assert len(splits) == 15
    for repeat in range(3):
        batch = splits[repeat * 5:(repeat + 1) * 5]
        seen = [p for split in batch for p in split.test_patients]
        # Every patient is tested exactly once per repeat -- no patient is left
        # out of the estimate and none is tested twice.
        assert sorted(seen) == ids
        for split in batch:
            assert not (set(split.train_patients) & set(split.test_patients))
            assert sorted(split.train_patients + split.test_patients) == ids


def test_repeats_use_different_partitions():
    ids = [f"P{i:03d}" for i in range(20)]
    splits = repeated_folds(ids, n_folds=4, n_repeats=3, seed=7)
    first = tuple(splits[0].test_patients)
    others = {tuple(splits[i].test_patients) for i in range(4, 12)}
    # A repeated scheme whose repeats partitioned identically would report a
    # spread of zero by construction rather than by measurement.
    assert first not in others


def test_cross_validation_refuses_an_unsplittable_cohort():
    with pytest.raises(ForecastError):
        repeated_folds(["P1", "P2"], n_folds=5)
    with pytest.raises(ForecastError):
        repeated_folds([f"P{i}" for i in range(10)], n_folds=5, n_repeats=0)


def test_cross_validated_detection_reports_folds_and_a_pooled_interval():
    result = cross_validated_detection(detection_examples(_cohort(40)), n_folds=5, n_repeats=3)
    m = result.metrics
    assert m["n_folds_scored"] == 15
    assert 0.0 <= m["oof_auc"] <= 1.0
    assert m["oof_auc_ci_low"] <= m["oof_auc_ci_high"]
    assert m["folds_above_chance"] <= m["n_folds_scored"]
    assert m["share_of_folds_above_chance"] == pytest.approx(
        m["folds_above_chance"] / m["n_folds_scored"]
    )
    # The pooled unit is the ROW -- one score per measured locus, each carrying
    # its own label -- while the interval rests on the tumours, so the number of
    # rows and the number of patients are both recorded and neither is inferred
    # from the other.
    assert result.n_patients == 40
    assert m["pooled_rows"] == result.n_rows
    assert m["pooled_patients"] == result.n_patients
    assert result.pooling is not None and result.pooling["unit"] == "row"
    assert m["patient_endpoint_positive"] <= result.n_patients
    assert len(result.per_fold) == 15


def test_cross_validated_change_reports_folds_and_a_per_patient_advantage():
    result = cross_validated_change(change_examples(_cohort(40)), n_folds=5, n_repeats=3)
    m = result.metrics
    assert m["n_folds_scored"] == 15
    assert m["folds_beating_persistence"] <= m["n_folds_scored"]
    # The advantage is the difference of two per-patient MAEs, so it agrees with
    # the two numbers printed beside it.
    assert m["oof_mae_advantage"] == pytest.approx(
        m["persistence_mae_per_patient"] - m["model_mae_per_patient"]
    )
    assert m["oof_mae_advantage_ci_low"] <= m["oof_mae_advantage_ci_high"]


def test_cross_validation_is_deterministic_for_a_seed():
    rows = detection_examples(_cohort(30))
    first = cross_validated_detection(rows, n_folds=5, n_repeats=2, seed=99)
    second = cross_validated_detection(rows, n_folds=5, n_repeats=2, seed=99)
    assert first.metrics["oof_auc"] == pytest.approx(second.metrics["oof_auc"])
    third = cross_validated_detection(rows, n_folds=5, n_repeats=2, seed=100)
    assert len(third.per_fold) == len(first.per_fold)


# ---------------------------------------------------------------------------
# The shipped record: the artifact must equal a live re-derivation
# ---------------------------------------------------------------------------


pytestmark_has_cache = pytest.mark.skipif(
    not os.path.exists(CACHE), reason="the cached GLASS release is not present"
)


@pytest.mark.skipif(not os.path.exists(CACHE), reason="cached GLASS release absent")
def test_the_shipped_record_re_derives_from_the_cache():
    with open(CACHE, encoding="utf-8") as fh:
        raw = json.load(fh)
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)

    records, dropped = records_from_cache(raw)
    spectra, pairs = paired_spectra(records)
    live_a = evaluate_detection(detection_examples(pairs))
    live_b = evaluate_change(change_examples(pairs))

    # v5 is v4 plus the repaired POOLING: the version is pinned (rather than
    # accepted as any string) so that a record written by an older build -- one
    # that pooled a single score per patient and read the patient's label off
    # whichever row was written last, the defect that moved both published
    # numbers -- cannot sit here being re-derived as if it were this one.
    assert record["record_version"] == "glass_evolution_forecast_v5"
    assert record["source"]["records_used"] == len(records)
    assert record["source"]["records_dropped"] == dropped == 0
    assert record["source"]["patients"] == len({r.patient_id for r in records})
    assert record["source"]["spectra"] == len(spectra)

    for label, live, stored in (
        ("task A", live_a, record["task_a_detection_at_recurrence"]),
        ("task B", live_b, record["task_b_allele_fraction_change"]),
    ):
        assert stored["n_test_rows"] == live.n_test_rows, label
        for key, value in live.metrics.items():
            theirs = stored["metrics"][key]
            if isinstance(value, float):
                assert theirs == pytest.approx(value, abs=1e-9), f"{label}.{key}"
            else:
                assert theirs == value, f"{label}.{key}"


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_each_verdict_agrees_with_its_own_interval():
    """The invariant that stops a point estimate being published as skill.

    Read from the CROSS-VALIDATED estimator, which is the one the verdicts name.
    The sentence test uses mutually exclusive phrases rather than a substring on
    a word like UNRESOLVED, because a verdict may discuss an earlier reading
    while claiming the opposite.
    """
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    cv_a = record["task_a_cross_validated"]["metrics"]
    cv_b = record["task_b_cross_validated"]["metrics"]
    verdicts = record["verdict"]

    a_resolved = isinstance(cv_a["oof_auc_ci_low"], float) and cv_a["oof_auc_ci_low"] > 0.5
    b_resolved = cv_b["beats_persistence_resolvably"] is True
    assert verdicts["task_a_resolved"] is a_resolved
    assert verdicts["task_b_resolved"] is b_resolved
    assert ("BEATS prevalence" in verdicts["task_a"]) is a_resolved
    assert ("BEATS persistence" in verdicts["task_b"]) is b_resolved
    # Task A resolves on this cohort and task B does not: a check that asserted
    # one fixed outcome would be pinning prose, not honesty.
    assert a_resolved is True and b_resolved is False
    assert "NOT shown to beat prevalence" not in verdicts["task_a"]
    assert "does NOT beat persistence" in verdicts["task_b"]


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_the_two_estimators_disagree_and_both_are_kept():
    """One split was underpowered on task A and lucky on task B -- recorded, not hidden."""
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    single_a = record["task_a_detection_at_recurrence"]["metrics"]
    cv_a = record["task_a_cross_validated"]["metrics"]
    single_b = record["task_b_allele_fraction_change"]["metrics"]
    cv_b = record["task_b_cross_validated"]["metrics"]

    # Task A: the single split's interval contained chance, the pooled folds do not.
    assert single_a["auc_fitted_logistic_ci_low"] <= 0.5 < cv_a["oof_auc_ci_low"]
    assert cv_a["folds_above_chance"] == cv_a["n_folds_scored"]
    # Task B: a positive single-split advantage against a NEGATIVE fold median.
    assert single_b["mae_advantage_over_persistence"] > 0
    assert cv_b["fold_mae_advantage_median"] < 0
    assert cv_b["folds_beating_persistence"] * 2 < cv_b["n_folds_scored"]
    assert cv_b["beats_persistence_resolvably"] is False


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_every_fold_of_every_repeat_is_recorded():
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    for key in ("task_a_cross_validated", "task_b_cross_validated"):
        cv = record[key]
        assert len(cv["per_fold"]) == cv["n_repeats"] * cv["n_folds"]
        for fold in cv["per_fold"]:
            assert fold["n_train_patients"] > 0 and fold["n_test_patients"] > 0
            assert fold["n_test_rows"] > 0


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_the_pairing_assumption_is_counted_not_assumed():
    """The release declares no timepoint, so the suffix rule is an inference."""
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    census = record["timepoint_census"]
    assert census["n_specimen_rows"] > 600
    # Every specimen row in this release says the same thing, so the column
    # cannot be the basis of the pairing and the census must say so.
    assert census["declared_types_distinct"] == 1
    assert census["declared_type_carries_timepoint"] is False
    assert "suffix" in census["basis"]
    assert census["cannot_establish"]
    assert census["n_excluded_by_suffix"] >= 0
    assert "Primary Solid Tumor" in census["declared_types"]


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_the_recorded_cohort_is_the_real_one_and_not_a_toy():
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    src = record["source"]
    assert src["records_used"] > 1000
    assert src["patients"] > 200
    assert src["spectra"] > 500
    assert src["paired_specimen_pairs"] > 200


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_the_recorded_sign_skill_is_not_dressed_up():
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    b = record["task_b_allele_fraction_change"]["metrics"]
    # The model's sign accuracy on resolvable changes equals the majority
    # direction exactly -- no skill beyond always predicting a fall. Recorded,
    # not hidden: a claim of sign skill would have to show a margin here.
    assert b["model_sign_accuracy_on_resolvable"] == pytest.approx(
        b["baseline_majority_sign_accuracy"]
    )


@pytest.mark.skipif(not os.path.exists(CACHE), reason="cached GLASS release absent")
def test_the_worked_example_is_a_real_patient_row_and_its_lower_bound_holds():
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    example = record["worked_example"]
    assert example["patient_id"].startswith("GLSS-")
    assert 0.0 <= example["lower_bound_primary"] <= 1.0
    assert 0.0 <= example["lower_bound_recurrence"] <= 1.0
    assert example["primary_depth"] > 0


def test_the_provenance_refuses_therapy_outcome_and_patient_prediction():
    prov = forecast_provenance()
    assert prov["allele_fractions_measured_from_real_specimens"] is EvidenceStatus.ESTABLISHED
    assert prov["patient_level_split_enforced"] is EvidenceStatus.ESTABLISHED
    for key in ("change_is_caused_by_therapy", "engine_predicts_patients", "clinical_benefit"):
        assert prov[key] is EvidenceStatus.UNKNOWN, key
    text = what_this_forecast_cannot_say().lower()
    assert "not a treatment effect" in text
    assert "clinically" in text


# ---------------------------------------------------------------------------
# Calibration: whether the probability itself can be read
# ---------------------------------------------------------------------------


def _calibrated_fixture():
    """Nine risk levels with exactly the hit rate each one promises.

    The one construction where the right answer is known by construction: at
    every level the observed rate IS the predicted probability, so the correct
    reading is "calibrated" and any other answer is a defect.
    """
    scores: List[float] = []
    labels: List[bool] = []
    for tenth in range(1, 10):
        p = tenth / 10.0
        scores += [p] * 20
        hits = int(round(p * 20))
        labels += [True] * hits + [False] * (20 - hits)
    return scores, labels


def test_reliability_bins_cover_every_row_exactly_once():
    scores, labels = _calibrated_fixture()
    bins = reliability_bins(scores, labels, n_bins=5)
    assert len(bins) == 5
    assert [b["bin"] for b in bins] == [0, 1, 2, 3, 4]
    assert sum(b["n"] for b in bins) == len(scores)
    # Equal-count bins: the sizes cannot differ by more than one row.
    sizes = [b["n"] for b in bins]
    assert max(sizes) - min(sizes) <= 1
    # Lower bins hold lower predictions, so the table reads as a curve.
    preds = [b["mean_predicted"] for b in bins]
    assert preds == sorted(preds)


def test_a_calibrated_predictor_is_read_as_calibrated():
    scores, labels = _calibrated_fixture()
    summary = calibration_summary(scores, labels)
    assert summary["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert summary["calibration_intercept"] == pytest.approx(0.0, abs=0.05)
    # The probabilities carry information a single constant does not.
    assert summary["brier_skill_score"] > 0.0
    assert calibration_is_within_tolerance(summary) is True
    assert "the out-of-fold PROBABILITIES are calibrated" in calibration_reading(
        summary, n_bins=5
    )


def test_an_over_confident_predictor_is_reported_as_not_calibrated():
    """Same ranking, useless probabilities -- and the reading must say so."""
    scores, labels = _calibrated_fixture()
    # Preserves the ORDER (so discrimination is unchanged) while pushing every
    # probability towards certainty: the classic miscalibrated model.
    pushed = [min(0.99, max(0.01, 0.5 + (s - 0.5) * 3.0)) for s in scores]
    summary = calibration_summary(pushed, labels)
    assert summary["calibration_slope"] < 0.5  # far below the ideal 1
    assert calibration_is_within_tolerance(summary) is False
    reading = calibration_reading(summary, n_bins=5)
    assert "NOT calibrated" in reading
    # The two readings are mutually exclusive, which is what lets the audit
    # assert one polarity rather than pattern-matching prose.
    assert "the probabilities are calibrated" not in reading


def test_the_calibration_reading_follows_the_numbers_not_the_mood():
    """The sentence is derived, so moving the numbers moves the sentence."""
    scores, labels = _calibrated_fixture()
    good = calibration_summary(scores, labels)
    bad = dict(good)
    bad["calibration_slope"] = 0.2
    bad["calibration_intercept"] = 1.5
    assert "calibrated" in calibration_reading(good, n_bins=5)
    assert "NOT calibrated" in calibration_reading(bad, n_bins=5)
    with_missing = dict(good)
    with_missing["calibration_slope"] = None
    assert calibration_is_within_tolerance(with_missing) is False


@pytest.mark.skipif(not os.path.exists(RECORD), reason="forecast record absent")
def test_the_recorded_calibration_block_is_the_live_one():
    """Task A's probabilities are published with the table that qualifies them."""
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    cal = record["task_a_cross_validated"]["calibration"]
    summary = cal["summary"]
    assert cal["n_bins"] >= 3
    assert len(cal["bins"]) == cal["n_bins"]
    assert cal["unit"] == "row"
    # The bins cover every pooled ROW: the calibration is measured on the same
    # population the AUC ranks, which is the row and not a patient average.
    assert sum(b["n"] for b in cal["bins"]) == record[
        "task_a_cross_validated"
    ]["n_rows"]
    # The bins are the arithmetic a reader checks by hand: the difference is the
    # observed rate minus the mean predicted probability in that bin.
    for b in cal["bins"]:
        assert b["difference"] == pytest.approx(
            b["observed_rate"] - b["mean_predicted"], abs=1e-12
        )
    # The published sentence is the one these numbers produce, byte for byte.
    assert cal["reading"] == calibration_reading(summary, n_bins=cal["n_bins"])
    assert record["verdict"]["task_a_calibration"] == cal["reading"]
    # And the claim's own polarity agrees with the measurement.
    from danon.claims import CLAIMS

    sentence = next(
        c.sentence for c in CLAIMS if c.id == "evolution_forecast_is_scored_against_baselines"
    )
    wanted = (
        "the probabilities are calibrated"
        if calibration_is_within_tolerance(summary)
        else "the probabilities are NOT calibrated"
    )
    assert wanted in sentence
    # Task B predicts a change, not a chance, so no calibration block belongs
    # there: attaching one would be a category error the record must not carry.
    assert record["task_b_cross_validated"].get("calibration") is None


# ---------------------------------------------------------------------------
# The pooled unit: what a pooled score is, and the defect that made this matter
# ---------------------------------------------------------------------------


def _cohort_with_mixed_label_rows():
    """Half the patients hold BOTH labels across their own rows, by construction.

    Every patient has one locus that is re-detected; the even-numbered ones also
    hold a locus that is not, so for exactly half the cohort a single
    per-patient label does not exist.
    """
    rows = []
    for i in range(12):
        patient = f"P{i:02d}"
        rows.append(DetectionExample(patient, "V0", 0.4, 200, True, 180))
        if i % 2 == 0:
            rows.append(DetectionExample(patient, "V1", 0.05, 300, False, 0))
    return rows


def test_a_pooled_score_is_a_row_and_each_row_keeps_its_own_label():
    rows = _cohort_with_mixed_label_rows()
    result = cross_validated_detection(rows, n_folds=4, n_repeats=2)
    m = result.metrics
    # The unit is the row: a collapse to one score per patient would change this
    # number, which is exactly what the repaired defect did.
    assert m["pooled_rows"] == len(rows) == 18
    assert m["pooled_patients"] == 12
    assert result.pooling["unit"] == "row"
    assert result.n_rows == len(rows)


def test_a_patient_whose_rows_disagree_is_not_labelled_by_its_last_row():
    """The repaired defect, pinned by the counts it must publish.

    Six of the twelve patients hold a re-detected locus AND a lost one. Reading
    the last row alone calls 6 of the 12 positive; the only defensible patient
    endpoint (a re-detection anywhere) calls all 12, and the two disagree on
    exactly those six. If a future edit reintroduces a per-patient average over
    mixed labels, this block and this test move together.
    """
    rows = _cohort_with_mixed_label_rows()
    block = pooling_block(rows)
    assert block["n_patients"] == 12
    assert block["n_pooled_rows"] == 18
    assert block["mixed_label_patients"] == 6
    assert block["patients_positive_last_row"] == 6
    assert block["patients_positive_any_row"] == 12
    assert block["patients_disagreeing"] == 6
    # The sentence is derived from those counts, so it names them rather than
    # asserting a unit in prose.
    assert "6 of" in block["reading"]
    assert "18 such rows" in block["reading"]
    result = cross_validated_detection(rows, n_folds=4, n_repeats=2)
    assert result.pooling == block
    # The secondary endpoint is the 'any row' label, recorded as its own number.
    assert result.metrics["patient_endpoint_positive"] == 12


def test_the_recorded_pooling_block_is_the_live_one():
    """The unit and the label of every published pooled score are re-derived."""
    with open(RECORD, encoding="utf-8") as fh:
        record = json.load(fh)
    stored = record["task_a_cross_validated"]["pooling"]
    with open(CACHE, encoding="utf-8") as fh:
        raw = json.load(fh)
    spectra, pairs = paired_spectra(records_from_cache(raw)[0])
    rows = detection_examples(pairs)
    assert stored == pooling_block(rows)
    assert stored["unit"] == "row"
    assert stored["n_pooled_rows"] == len(rows)
    # The defect is DOCUMENTED rather than merely fixed: the record still says
    # how many patients hold both labels and how far apart the two candidate
    # patient labels are.
    assert stored["mixed_label_patients"] > 0
    assert stored["patients_disagreeing"] > 0
    assert stored["patients_positive_any_row"] != stored["patients_positive_last_row"]
