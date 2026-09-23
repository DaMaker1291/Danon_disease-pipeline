"""Tests for the validation-tier module.

Anchor tests:

- Harrell's C must stay in [0, 1]. The original implementation counted
  event SUBJECTS in the denominator while accumulating comparable PAIRS in
  the numerator, producing C = 2.92 on a demo dataset -- statistically
  impossible and silently flattering. The bound is pinned here.
- A sign-flipped risk score must produce C < 0.5 AND hazard ratio < 1: the
  two statistics must agree about a bad score rather than contradicting
  each other (which is what the C bug caused).
- IC50 validation must refuse sparse, duplicated, or too-narrow data
  instead of reporting a number; and a constant multiplicative units error
  must be caught by the bias note even though R^2 is shift-invariant and
  cannot see it.

Synthetic cohorts are generated from a latent risk with a known sign, so
each test states what the correct answer IS before asserting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.validation import (
    IC50Pair,
    ValidationTier,
    WetLabPrediction,
    concordance_index,
    design_wetlab_validation,
    log2_transform_check,
    patient_level_split,
    tier_overview,
    validate_ic50_predictions,
    validate_survival_stratification,
    validation_provenance,
)


# ---------------------------------------------------------------------------
# Synthetic cohort from known latent risk
# ---------------------------------------------------------------------------


def _make_cohort(seed=3, n=120, flip=False, noise=0.6, censor=0.28):
    """Latent risk -> time-to-event; risk score = latent + noise.

    High latent -> shorter survival (times = 24*exp(-0.7*latent)). A correct
    risk score is +latent (+ noise); flip=True hands the validator the
    sign-reversed score and the model MUST fail, not flatter.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    times = np.clip(24.0 * np.exp(-0.7 * latent + rng.normal(0, 0.5, n)), 2.0, 90.0)
    # `censor` is the CENSORING probability; event probability is 1 - censor.
    events = (rng.random(n) >= censor).astype(int)
    sign = -1.0 if flip else 1.0
    ids = [f"PTCGA-{i:03d}" for i in range(n)]
    scores = {p: sign * latent[i] + rng.normal(0, noise) for i, p in enumerate(ids)}
    return ids, times.tolist(), events.tolist(), scores


# ---------------------------------------------------------------------------
# Tier 1: concordance index
# ---------------------------------------------------------------------------


def test_c_index_bounded_on_perfect_and_antiperfect_scores():
    n = 60
    times = [float(i + 1) for i in range(n)]
    events = [1] * n
    perfect = [float(n - i) for i in range(n)]  # high score dies earliest
    anti = [float(i + 1) for i in range(n)]
    assert concordance_index(times, events, perfect) == pytest.approx(1.0)
    assert concordance_index(times, events, anti) == pytest.approx(0.0)


def test_c_index_stays_in_unit_interval_under_censoring():
    ids, times, events, scores = _make_cohort(seed=11)
    c = concordance_index(times, events, [scores[p] for p in ids])
    assert 0.0 <= c <= 1.0
    assert c > 0.6  # genuine signal must register on the risk-score scale


def test_c_index_for_random_score_is_near_chance():
    rng = np.random.default_rng(0)
    ids, times, events, _ = _make_cohort(seed=11)
    random_scores = list(rng.normal(0, 1, len(ids)))
    c = concordance_index(times, events, random_scores)
    assert 0.35 < c < 0.65


def test_c_index_requires_comparable_pairs():
    # Every subject censored at the same time -> no comparable pairs at all.
    with pytest.raises(ValueError, match="comparable"):
        concordance_index([10.0, 10.0, 10.0], [0, 0, 0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Tier 1: patient-level splitting
# ---------------------------------------------------------------------------


def test_patient_level_split_is_disjoint_deterministic_and_complete():
    ids = [f"P{i:02d}" for i in range(50)]
    train1, test1 = patient_level_split(ids, test_fraction=0.3, seed=7)
    train2, test2 = patient_level_split(ids, test_fraction=0.3, seed=7)
    assert not set(train1) & set(test1)
    assert set(train1) | set(test1) == set(ids)
    assert (train1, test1) == (train2, test2)


def test_patient_level_split_refuses_degenerate_inputs():
    with pytest.raises(ValueError, match="test_fraction"):
        patient_level_split([f"P{i}" for i in range(50)], test_fraction=0.0)
    with pytest.raises(ValueError, match="test_fraction"):
        patient_level_split([f"P{i}" for i in range(50)], test_fraction=1.0)
    with pytest.raises(ValueError, match="at least 5 patients"):
        patient_level_split(["P1", "P2"], test_fraction=0.5)


# ---------------------------------------------------------------------------
# Tier 1: expression scale check
# ---------------------------------------------------------------------------


def test_log2_check_flags_raw_counts_before_they_become_results():
    rng = np.random.default_rng(0)
    counts = rng.lognormal(5.0, 1.5, 500)  # values in the hundreds/thousands
    check = log2_transform_check(counts.tolist())
    assert not check.plausible_transformed
    assert "RAW COUNTS" in check.message


def test_log2_check_passes_log_scale_data():
    check = log2_transform_check([0.1, 3.2, 7.9, 12.4, 18.0, 5.5])
    assert check.plausible_transformed
    assert "plausible" in check.message


# ---------------------------------------------------------------------------
# Tier 1: end-to-end survival validation
# ---------------------------------------------------------------------------


def test_survival_validation_good_score_transfers_to_held_out_patients():
    ids, times, events, scores = _make_cohort(seed=3, flip=False)
    res = validate_survival_stratification(ids, times, events, scores, seed=1)
    assert res.c_index > 0.55
    assert res.hazard_ratio > 1.0
    assert res.logrank_p_value < 0.05
    # Statistics are computed on the held-out split only.
    assert res.n_patients == len(res.split[1])
    assert not set(res.split[0]) & set(res.split[1])


def test_survival_validation_flipped_score_fails_and_statistics_agree():
    ids, times, events, scores = _make_cohort(seed=3, flip=True)
    res = validate_survival_stratification(ids, times, events, scores, seed=1)
    assert res.c_index < 0.5
    assert res.hazard_ratio < 1.0
    assert "AT OR BELOW CHANCE" in res.c_index_interpretation


def test_c_index_and_hazard_ratio_agree_on_a_bad_score():
    """The regression that motivated the C fix: C was computed wrong while
    the hazard ratio was correct, so the two statistics contradicted each
    other about the same score. Both must now tell the same story."""
    for flip in (False, True):
        ids, times, events, scores = _make_cohort(seed=3, flip=flip)
        res = validate_survival_stratification(ids, times, events, scores, seed=1)
        assert (res.c_index < 0.5) == (res.hazard_ratio < 1.0)


def test_survival_validation_flags_raw_expression_scale():
    ids, times, events, scores = _make_cohort(seed=3)
    raw_counts = [1e5, 2e4, 3e3, 5e4] * 50
    res = validate_survival_stratification(
        ids, times, events, scores, expression_sample=raw_counts, seed=1
    )
    assert not res.scale_check.plausible_transformed
    assert any("RAW COUNTS" in n for n in res.notes)


def test_survival_validation_refuses_missing_scores():
    ids, times, events, scores = _make_cohort(seed=3, n=60)
    del scores[ids[0]]
    with pytest.raises(ValueError, match="risk_scores missing"):
        validate_survival_stratification(ids, times, events, scores, seed=1)


def test_survival_validation_refuses_duplicate_patient_ids():
    ids, times, events, scores = _make_cohort(seed=3, n=60)
    dup_ids = ids + [ids[0]]
    dup_times = times + [times[0]]
    dup_events = events + [events[0]]
    dup_scores = {**scores, ids[0]: 1.0}
    with pytest.raises(ValueError, match="duplicate"):
        validate_survival_stratification(dup_ids, dup_times, dup_events, dup_scores, seed=1)


def test_survival_validation_refuses_nan_scores():
    ids, times, events, scores = _make_cohort(seed=3, n=60)
    scores[ids[0]] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_survival_stratification(ids, times, events, scores, seed=1)


def test_survival_validation_refuses_cohort_with_too_few_events():
    ids, times, events, scores = _make_cohort(seed=3, n=40, censor=0.99)
    with pytest.raises(ValueError, match="events"):
        validate_survival_stratification(ids, times, events, scores, seed=1)


# ---------------------------------------------------------------------------
# Tier 2: IC50 validation against public screens
# ---------------------------------------------------------------------------


def _ic50_pairs(noise_scale=0.15, units_error=False, seed=5):
    """6 lines x 3 drugs, measured IC50 from a latent, predicted = measured
    (optionally scaled 1000x to simulate a uM-vs-nM units error) * small noise."""
    rng = np.random.default_rng(seed)
    lines = [f"LINE-{i}" for i in range(6)]
    drugs = ["drugA", "drugB", "drugC"]
    pairs = []
    for line in lines:
        for drug in drugs:
            measured = 10 ** rng.normal(0.5, 0.7)  # uM
            predicted = measured * (1000.0 if units_error else 1.0)
            predicted *= float(np.exp(rng.normal(0, noise_scale)))
            pairs.append(
                IC50Pair(
                    cell_line=line,
                    drug=drug,
                    predicted_ic50_um=predicted,
                    measured_ic50_um=measured,
                )
            )
    return pairs


def test_ic50_validation_supportive_on_good_predictions():
    res = validate_ic50_predictions(_ic50_pairs())
    assert res.r2 >= 0.60
    assert "supportive" in res.verdict
    assert res.n_pairs == 18
    assert res.n_distinct_lines == 6
    assert res.n_distinct_drugs == 3
    assert 0.8 < res.spearman_rho <= 1.0
    assert 0.0 <= res.ci95[0] <= res.ci95[1] <= 1.0


def test_ic50_validation_catches_units_error_via_bias_note():
    """R^2 is shift-invariant in log space, so a 1000x units error is
    invisible to it. The bias note is what catches it -- and it must."""
    res = validate_ic50_predictions(_ic50_pairs(units_error=True))
    assert res.r2 >= 0.60  # unchanged: R^2 cannot see the shift
    assert abs(res.bias_log10 - 3.0) < 0.05
    assert any("bias" in n and "log10" in n for n in res.notes)


def test_ic50_validation_refuses_sparse_data():
    with pytest.raises(ValueError, match="pairs"):
        validate_ic50_predictions(_ic50_pairs()[:7])


def test_ic50_validation_refuses_duplicate_measurements():
    pairs = _ic50_pairs()
    dup = IC50Pair(
        cell_line=pairs[0].cell_line,
        drug=pairs[0].drug,
        predicted_ic50_um=pairs[0].predicted_ic50_um * 2.0,
        measured_ic50_um=pairs[0].measured_ic50_um,
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_ic50_predictions(pairs + [dup])


def test_ic50_validation_refuses_too_few_distinct_lines():
    pairs = [p for p in _ic50_pairs() if p.cell_line in {"LINE-0", "LINE-1", "LINE-2", "LINE-3"}]
    assert len(pairs) == 12  # enough pairs, too few lines
    with pytest.raises(ValueError, match="cell lines"):
        validate_ic50_predictions(pairs)


def test_ic50_validation_refuses_too_few_distinct_drugs():
    pairs = [p for p in _ic50_pairs() if p.drug in {"drugA", "drugB"}]
    assert len(pairs) == 12  # enough pairs, too few drugs
    with pytest.raises(ValueError, match="drugs"):
        validate_ic50_predictions(pairs)


def test_ic50_pair_rejects_nonpositive_and_nonfinite_values():
    with pytest.raises(ValueError):
        IC50Pair(cell_line="L", drug="D", predicted_ic50_um=-1.0, measured_ic50_um=1.0)
    with pytest.raises(ValueError):
        IC50Pair(cell_line="L", drug="D", predicted_ic50_um=0.0, measured_ic50_um=1.0)
    with pytest.raises(ValueError):
        IC50Pair(cell_line="L", drug="D", predicted_ic50_um=float("nan"), measured_ic50_um=1.0)


# ---------------------------------------------------------------------------
# Tier 3: wet-lab plan guardrails
# ---------------------------------------------------------------------------


def test_wetlab_plan_attaches_lineage_caveats_for_known_lines():
    plan = design_wetlab_validation(
        predictions=[
            WetLabPrediction(
                statement="TMZ + hypoxia reduces viability by 60%",
                outcome_measure="CCK-8 viability, 72 h",
                predicted_value=40.0,
                acceptable_range=(25.0, 55.0),
                conditions={"hypoxia": "1% O2"},
            )
        ],
        cell_lines=["U87-MG", "LN229"],
    )
    joined = " | ".join(plan.caveats)
    assert "U87-MG" in joined
    assert "lineage" in joined          # U87 caveat warns about lineage drift
    assert "p53" in joined              # LN229 caveat notes p53 background


def test_wetlab_plan_refuses_predictionless_plans_and_bad_ranges():
    with pytest.raises(ValueError, match="no predictions"):
        design_wetlab_validation(predictions=[])
    with pytest.raises(ValueError, match="acceptable_range"):
        design_wetlab_validation(
            predictions=[
                WetLabPrediction(
                    statement="x",
                    outcome_measure="y",
                    predicted_value=1.0,
                    acceptable_range=(5.0, 5.0),
                )
            ]
        )


# ---------------------------------------------------------------------------
# Tier ladder and provenance
# ---------------------------------------------------------------------------


def test_tier_overview_covers_all_four_tiers_with_lab_flags():
    rows = tier_overview()
    assert [r["tier"] for r in rows] == [t.value for t in ValidationTier]
    by_tier = {r["tier"]: r for r in rows}
    assert by_tier["public_cohort"]["requires_lab"] == "False"
    assert by_tier["public_pharmacology"]["requires_lab"] == "False"
    assert by_tier["cell_line_wetlab"]["requires_lab"] == "True"
    assert by_tier["patient_derived"]["requires_lab"] == "True"
    for row in rows:
        assert row["proves"] and row["cannot_prove"]


def test_provenance_marks_cross_screen_and_u87_fidelity_as_failed():
    prov = validation_provenance()
    assert prov["cross_screen_ic50_comparability"] == EvidenceStatus.PROVEN_FAILED
    assert prov["cell_line_fidelity_u87"] == EvidenceStatus.PROVEN_FAILED
    assert prov["independent_cohort_survival_validation"] == EvidenceStatus.ESTABLISHED
    assert prov["gdsc_ctrp_ic50_benchmarking"] == EvidenceStatus.ESTABLISHED
