"""Tests for the evaluation-target module.

The point of this module is that proposed high scores are checked against data
that exists, so the tests are mostly about refusals being *earned*: the label
audit must find an empty field, the GDSC audit must find the extrapolations, and
no claimed 0.9x figure may appear as a measurement anywhere.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.eval_targets import (
    GDSC2_FITTED_PATH,
    GdscAudit,
    ResponderLabelAudit,
    TargetStatus,
    _auc,
    _logistic_fit,
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

CLINICAL_PATH = "data/gdc_clinical/tcga_gbm_cases.json"
has_clinical = os.path.exists(CLINICAL_PATH)
has_gdsc = os.path.exists(GDSC2_FITTED_PATH)


# ---------------------------------------------------------------------------
# Target 1 -- spatial expansion
# ---------------------------------------------------------------------------


class TestSpatialTarget:
    def test_refused_while_the_live_verdict_refuses(self):
        from danon.multimodal import spatial_dice_verdict

        v = spatial_expansion_target()
        if spatial_dice_verdict().supported:
            # A segmented collection appeared: the target must now demand a number.
            assert v.status is TargetStatus.MEASURED
            assert "measure it" in v.detail
        else:
            assert v.status is TargetStatus.REFUSED_NO_DATA
            assert v.measured is None
            assert not v.has_number
            assert len(v.blockers) >= 3

    def test_detail_reports_real_probe_counts(self):
        v = spatial_expansion_target()
        assert "collection(s) probed" in v.detail
        assert "segmentation series" in v.detail


# ---------------------------------------------------------------------------
# Target 2 -- responder label
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_clinical, reason="cached GDC clinical export absent")
class TestResponderLabel:
    def test_best_overall_response_is_entirely_absent(self):
        """The field exists in the schema and is empty in the data."""
        audit = audit_responder_label(CLINICAL_PATH)
        assert audit.n_cases > 500
        assert audit.n_best_overall_response == 0

    def test_label_is_heavily_imbalanced_and_the_base_rate_is_the_trap(self):
        audit = audit_responder_label(CLINICAL_PATH)
        assert audit.n_clean_label > 400
        assert audit.positive_fraction < 0.10
        # A constant predictor already scores above 0.9 accuracy: this is where
        # a "94% accurate" claim would come from with no model at all.
        assert audit.majority_accuracy > 0.90
        assert "94% accuracy" in audit.verdict

    def test_some_cases_carry_both_statuses(self):
        audit = audit_responder_label(CLINICAL_PATH)
        assert audit.n_ambiguous > 0
        assert audit.n_neither > 0

    def test_responder_label_object_prefers_the_constant_class(self):
        audit = ResponderLabelAudit(
            n_cases=100, n_best_overall_response=0, n_with_any_disease_response=90,
            n_clean_label=90, n_positive=5, n_ambiguous=0, n_neither=10,
        )
        assert audit.positive_fraction == pytest.approx(5 / 90)
        assert audit.majority_accuracy == pytest.approx(85 / 90)

    def test_empty_label_does_not_divide_by_zero(self):
        audit = ResponderLabelAudit(
            n_cases=10, n_best_overall_response=0, n_with_any_disease_response=0,
            n_clean_label=0, n_positive=0, n_ambiguous=0, n_neither=10,
        )
        assert np.isnan(audit.positive_fraction)
        assert str(audit.verdict)


@pytest.mark.skipif(not has_clinical, reason="cached GDC clinical export absent")
class TestResponderTarget:
    def test_auc_is_measured_and_its_ci_is_reported(self):
        rt = responder_target()
        assert 0.0 <= rt.auc <= 1.0
        lo, hi = rt.auc_ci
        assert lo <= rt.auc <= hi
        assert rt.n_patients >= 400
        assert rt.n_positive >= 10

    def test_the_ci_includes_chance_so_the_auc_is_not_a_finding(self):
        """With this few positives the interval spans 0.5; saying so is required."""
        rt = responder_target()
        lo, hi = rt.auc_ci
        assert lo < 0.5 < hi

    def test_the_reachable_flag_runs_backwards_against_survival(self):
        rt = responder_target()
        assert rt.label_tracks_survival_backwards
        assert rt.median_os_positive_days < rt.median_os_negative_days
        assert "INVERTED" in rt.describe()


class TestAucHelpers:
    def test_auc_is_one_for_a_perfect_separation_and_half_for_ties(self):
        y = np.array([0, 0, 1, 1])
        assert _auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
        assert _auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)
        assert _auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.0)

    def test_auc_is_not_estimable_with_one_class(self):
        assert np.isnan(_auc(np.array([0.1, 0.2]), np.array([1, 1])))

    def test_logistic_fit_recovers_a_planted_direction(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=(500, 1))
        y = (rng.rand(500) < 1 / (1 + np.exp(-3.0 * x[:, 0]))).astype(int)
        beta = _logistic_fit(x, y)
        assert beta[1] > 0


# ---------------------------------------------------------------------------
# Target 3 -- in-vitro dose response
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_gdsc, reason="GDSC fitted release not cached")
class TestGdscAudit:
    def test_reads_the_real_release_scale(self):
        audit = load_gdsc_glioma()
        assert audit.n_rows > 200_000
        assert audit.n_drugs > 200
        assert audit.n_cell_lines > 900
        assert audit.n_gbm_rows > 1_000
        assert audit.n_gbm_lines > 20

    def test_most_glioma_ic50s_are_extrapolated_beyond_the_tested_range(self):
        """The finding that makes the R^2 target a refusal, not a caveat."""
        audit = load_gdsc_glioma()
        assert audit.extrapolated_fraction > 0.5
        assert audit.n_inside_range > 0

    def test_every_temozolomide_glioma_ic50_is_extrapolated(self):
        audit = load_gdsc_glioma()
        assert audit.temozolomide_rows >= 20
        assert audit.temozolomide_above_max_tested == audit.temozolomide_rows

    def test_verdict_names_both_counts(self):
        audit = load_gdsc_glioma()
        assert "above the highest concentration tested" in audit.verdict
        assert "temozolomide" in audit.verdict

    def test_missing_file_is_reported_not_silently_skipped(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gdsc_glioma(str(tmp_path / "nope.csv"))

    def test_synthetic_audit_maths(self):
        a = GdscAudit(
            path="x", n_rows=10, n_drugs=2, n_cell_lines=3, n_gbm_rows=8,
            n_gbm_drugs=2, n_gbm_lines=3, n_inside_range=2,
            n_above_max_tested=6, n_below_min_tested=0,
            temozolomide_rows=3, temozolomide_above_max_tested=3,
        )
        assert a.extrapolated_fraction == pytest.approx(0.75)
        assert "75.0%" in a.verdict


class TestInVitroTarget:
    def test_refused_on_the_missing_unit(self):
        from danon.pharmacology import engine_ec50_unit

        v = in_vitro_target()
        assert engine_ec50_unit() is None
        assert v.status is TargetStatus.REFUSED_INCOMMENSURABLE
        assert v.measured is None
        assert any("no declared concentration unit" in b for b in v.blockers)

    def test_blockers_include_the_extrapolation_and_the_missing_raw_files(self):
        v = in_vitro_target()
        blob = " ".join(v.blockers)
        assert "extrapolated" in blob
        assert "GB" in blob  # the raw dose-response files were not downloaded

    def test_survives_a_missing_gdsc_cache(self, tmp_path):
        v = in_vitro_target(path=str(tmp_path / "absent.csv"))
        assert v.status is TargetStatus.REFUSED_INCOMMENSURABLE
        assert v.n == 0


# ---------------------------------------------------------------------------
# Spatial metric semantics
# ---------------------------------------------------------------------------


class TestDiceSimilarity:
    def test_perfect_overlap_and_total_miss(self):
        a = np.zeros((4, 4, 4))
        a[1, 1, 1] = 1
        assert dice_similarity(a, a).coefficient == pytest.approx(1.0)
        b = np.zeros((4, 4, 4))
        b[3, 3, 3] = 1
        assert dice_similarity(a, b).coefficient == pytest.approx(0.0)

    def test_both_empty_is_undefined_not_perfect(self):
        """The shortcut convention scores no prediction at 1.0; this must not."""
        z = np.zeros((4, 4, 4))
        result = dice_similarity(z, z)
        assert not result.estimable
        assert result.coefficient is None
        assert "no prediction was made" in result.reason
        assert "not estimable" in str(result)

    def test_empty_prediction_against_real_truth_is_a_miss(self):
        z = np.zeros((4, 4, 4))
        t = np.zeros((4, 4, 4))
        t[1, 1, 1] = 1
        result = dice_similarity(z, t)
        assert result.estimable
        assert result.coefficient == pytest.approx(0.0)

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError):
            dice_similarity(np.zeros((3, 3, 3)), np.zeros((4, 4, 4)))

    def test_the_shortcut_bug_is_reported(self):
        text = dice_degenerate_case_report()
        assert "PERFECT score for no" in text
        assert "not estimable" in text


# ---------------------------------------------------------------------------
# EGFRvIII reachability
# ---------------------------------------------------------------------------


class TestEgfrviii:
    def test_structural_variation_is_entirely_controlled(self):
        r = probe_egfrviii_reachability(use_cache=False)
        assert r.n_structural_variation_files > 0
        assert r.n_structural_variation_controlled == r.n_structural_variation_files
        assert r.reachable_open is False

    def test_open_maf_files_exist_but_cannot_carry_the_deletion(self):
        r = probe_egfrviii_reachability(use_cache=False)
        assert r.n_open_maf_files > 0
        assert "STRUCTURAL variant" in r.verdict
        assert "not obtainable from open data" in r.verdict

    def test_verdict_flips_if_somesv_becomes_open(self):
        from danon.eval_targets import EgfrviiiReachability

        r = EgfrviiiReachability(
            n_structural_variation_files=100, n_structural_variation_controlled=99,
            n_open_maf_files=464, n_maf_controlled=0, source="x",
        )
        assert r.reachable_open is True
        assert "verify" in r.verdict


# ---------------------------------------------------------------------------
# The RL tier
# ---------------------------------------------------------------------------


class TestRlTier:
    """Two live branches: refused without a trained artifact, measured with one.

    The refusal is exercised against a path that does not exist, so these tests
    keep working after the policy is trained -- the branch is real either way.
    """

    MISSING = "outputs_real/does_not_exist_rl_artifact.json"

    def test_refuses_when_no_training_artifact_exists(self):
        v = rl_policy_target(artifact_path=self.MISSING)
        assert v.status is TargetStatus.REFUSED_NO_DATA
        assert v.measured is None
        assert not v.has_number
        assert "no valid, reproducible RL evaluation artifact" in v.detail

    def test_the_dosing_policies_are_named_as_closed_form_when_refusing(self):
        v = rl_policy_target(artifact_path=self.MISSING)
        blob = v.detail + " " + " ".join(v.blockers)
        for name in ("MTDPolicy", "AdaptiveDosePolicy", "CollateralSwitchPolicy"):
            assert name in blob
        assert "constant" in blob

    def test_the_authored_reward_stays_a_recorded_limit(self):
        v = rl_policy_target(artifact_path=self.MISSING)
        assert any("author-defined" in b or "falsified" in b for b in v.blockers)

    def test_refuses_an_artifact_with_a_swapped_weight_hash(self, tmp_path):
        from danon.rl_steering import ARTIFACT_PATH, load_artifact

        art = load_artifact(ARTIFACT_PATH)
        if art is None:
            pytest.skip("train the policy first: python _train_evosteer_ppo.py")
        art["weights_sha256"] = "0" * 64
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(art), encoding="utf-8")
        verdict = rl_policy_target(artifact_path=str(path))
        assert verdict.status is TargetStatus.REFUSED_NO_DATA
        assert verdict.measured is None
        assert "hash mismatch" in verdict.detail

    def test_reports_the_measured_artifact_when_present(self):
        from danon.rl_steering import ARTIFACT_PATH, load_artifact

        if load_artifact(ARTIFACT_PATH) is None:
            pytest.skip("train the policy first: python _train_evosteer_ppo.py")
        v = rl_policy_target()
        if v.has_number:
            assert v.status is TargetStatus.MEASURED
        else:
            assert v.status is TargetStatus.REFUSED_ESTIMAND
            assert any("denominator is zero" in b for b in v.blockers)
        assert v.n >= 20
        assert "IN SILICO" in v.detail
        assert "historical 95.4% target not evaluated" in v.claimed
        # the measured branch keeps the remaining limits visible
        assert any("synthetic" in b for b in v.blockers)
        assert any("author-defined" in b for b in v.blockers)
        assert any("alternation" in b for b in v.blockers)
        if v.has_number:
            assert v.measured == pytest.approx(1.0051)


# ---------------------------------------------------------------------------
# The one measurable tier
# ---------------------------------------------------------------------------


class TestSolverTier:
    def test_measured_and_labelled_synthetic(self):
        v = solver_convergence_target()
        assert v.status is TargetStatus.MEASURED
        assert v.has_number
        assert v.measured > 0.99
        assert "SYNTHETIC" in v.name
        assert "no patient appears" in " ".join(v.blockers)

    def test_verdicts_split_measured_from_refused(self):
        verdicts = target_verdicts(with_gdsc=False, with_solver=True)
        measured = [v for v in verdicts if v.has_number]
        refused = [v for v in verdicts if not v.has_number]
        # the synthetic PDE stencil is always measurable; the RL tier joins it
        # only once a policy has been trained and evaluated
        rl = [v for v in verdicts if v.name.startswith("RL")][0]
        assert len(measured) == (2 if rl.has_number else 1)
        assert len(refused) >= 2
        for v in measured:
            assert v.name.startswith("PDE stencil") or v.name.startswith("RL")


# ---------------------------------------------------------------------------
# The honest ceiling table
# ---------------------------------------------------------------------------


class TestCeilingReport:
    def test_no_clinical_target_reports_a_measured_score(self):
        """The clinical tiers cannot produce a number here, whatever else can.

        The RL tier is the one exception once a policy has been trained, and it
        arrives labelled: a measured RL number must carry the IN SILICO note, so
        it cannot be read as a clinical score.
        """
        verdicts = target_verdicts(with_gdsc=False)
        assert verdicts
        clinical = [v for v in verdicts if not v.name.startswith("RL")]
        assert clinical
        assert not any(v.has_number for v in clinical)
        rl = [v for v in verdicts if v.name.startswith("RL")][0]
        if rl.has_number:
            assert "IN SILICO" in rl.detail
        else:
            assert rl.status is TargetStatus.REFUSED_NO_DATA

    def test_no_claimed_value_appears_as_a_measured_score(self):
        """Each proposed figure may appear as a CLAIM, never as a measurement.

        Quoting the claim is the point of the table; the invariant is that the
        measured line for a refused tier never carries one of them.
        """
        text = measured_ceiling_report(with_gdsc=False)
        # Point scores invented in the pitch must not appear at all: the table
        # quotes claims as ranges, so a bare 0.914 has no legitimate source.
        for invented in ("0.914", "0.938", "0.921", "0.9999"):
            assert invented not in text
        # A claim that IS quoted must sit on a `claimed` line, never a `measured`
        # one, and the measured line for a refused tier must say so.
        lines = text.splitlines()
        claimed_lines = [ln for ln in lines if "claimed" in ln]
        measured_lines = [ln for ln in lines if "measured" in ln]
        assert not any("95.4%" in ln for ln in measured_lines)
        # Most tiers have no number at all. The responder tier carries 0.6155
        # measured on the nearest reachable label while its status still refuses
        # the claim, which is the intended split -- so the invariant is that a
        # measured number on a refused tier never satisfies the claim.
        # Composition depends on the with_gdsc flag AND on whether a policy has
        # been trained, so this is a floor rather than an exact count: the
        # spatial tier never carries a number here, which is the part that has
        # to hold.
        assert sum("no number" in ln for ln in measured_lines) >= 1
        assert "refused_no_label" in text
        assert "Refusing a tier does not raise another" in text

    def test_solver_tier_appears_only_when_asked_and_is_labelled(self):
        assert "SYNTHETIC" not in measured_ceiling_report(with_gdsc=False)
        text = measured_ceiling_report(with_gdsc=False, with_solver=True)
        assert "SYNTHETIC" in text
        rl = [v for v in target_verdicts(with_gdsc=False) if v.name.startswith("RL")][0]
        assert f"{2 if rl.has_number else 1} measurable" in text

    def test_report_labels_each_proposed_claim_and_its_status(self):
        text = measured_ceiling_report(with_gdsc=False)
        assert "Dice 0.91-0.94" in text
        assert "AUC 0.91-0.94" in text
        assert "refused_no_data" in text
        assert "refused_no_label" in text

    def test_every_verdict_describes_itself(self):
        for v in target_verdicts(with_gdsc=False):
            assert v.name in v.describe()
            assert v.metric in v.describe()

    def test_provenance_marks_the_two_failed_estimands(self):
        prov = eval_targets_provenance()
        assert prov["spatial_dice_verdict"] is EvidenceStatus.PROVEN_FAILED
        assert prov["engine_ec50_unit_absence"] is EvidenceStatus.PROVEN_FAILED
        assert prov["gdsc_extrapolation_counts"] is EvidenceStatus.ESTABLISHED
        assert prov["hill_slope_vs_measured_curves"] is EvidenceStatus.UNKNOWN

    def test_cannot_show_refuses_the_additive_move(self):
        items = what_targets_cannot_show()
        assert len(items) >= 5
        assert any("cannot be added to" in i for i in items)
        assert any("lower bound" in i for i in items)
