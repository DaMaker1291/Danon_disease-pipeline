"""Tests for the claims audit.

Three of these tests exist because the ISEF pitch contained claims the code
does not support, and the drift is always towards the more impressive sentence:

  * `DERIVED` is not an `EvidenceStatus` member, so the provenance vocabulary in
    the pitch was not the one in the repository;
  * the repository ships `torch.nn` modules and an ESM scorer, so "not a black
    box" is false as stated;
  * the quoted test count was stale.

Each is pinned here so the poster cannot drift back.
"""

from __future__ import annotations

import json
import os

import pytest

from danon.claims import (
    CHECKS,
    CLAIMS,
    Claim,
    CheckResult,
    audit_claims,
    check_evidence_status_vocabulary,
    check_seeded_reproducibility,
    check_wild_type_from_deposit,
    claims_provenance,
    render_claims_markdown,
    what_these_claims_cannot_establish,
)
from danon.cure import EvidenceStatus


class TestClaimSet:
    def test_every_claim_binds_to_a_registered_check(self):
        for claim in CLAIMS:
            assert claim.check in CHECKS, claim.id
            assert claim.sentence and claim.limit and claim.substantiated_by
            assert isinstance(claim.tier, EvidenceStatus)

    def test_every_claim_verifies_against_the_repository_today(self):
        audits = audit_claims()
        failed = [(a.claim.id, a.detail) for a in audits if not a.supported]
        assert failed == [], f"claims failed their own checks: {failed}"

    def test_claim_ids_are_unique(self):
        ids = [c.id for c in CLAIMS]
        assert len(ids) == len(set(ids))


class TestFailedChecksAreReportedNotDropped:
    def test_unknown_check_is_a_failure(self):
        bogus = Claim(
            id="bogus",
            sentence="this project cures cancer",
            tier=EvidenceStatus.ESTABLISHED,
            substantiated_by="nowhere",
            check="does_not_exist",
            limit="none",
        )
        audits = audit_claims((bogus,))
        assert len(audits) == 1
        assert not audits[0].supported
        assert "unknown check" in audits[0].detail

    def test_rendered_table_announces_failures(self):
        bogus = Claim(
            id="bogus",
            sentence="this project cures cancer",
            tier=EvidenceStatus.ESTABLISHED,
            substantiated_by="nowhere",
            check="does_not_exist",
            limit="none",
        )
        md = render_claims_markdown(audit_claims((bogus,)))
        assert "**NO**" in md
        assert "1 of 1 claims failed" in md

    def test_check_that_raises_counts_as_a_failure(self, monkeypatch):
        def boom() -> CheckResult:
            raise RuntimeError("check exploded")

        monkeypatch.setitem(CHECKS, "explodes", boom)
        claim = Claim(
            id="raises", sentence="s", tier=EvidenceStatus.UNKNOWN,
            substantiated_by="x", check="explodes", limit="l",
        )
        audits = audit_claims((claim,))
        assert not audits[0].supported
        assert "check raised" in audits[0].detail


class TestTheThreePitchDrifts:
    def test_derived_is_not_a_provenance_member(self):
        assert "DERIVED" not in [m.name for m in EvidenceStatus]
        assert check_evidence_status_vocabulary().supported
        # and the vocabulary used in the claim text is the real one
        claim = next(c for c in CLAIMS if c.id == "provenance_registry")
        for member in EvidenceStatus:
            assert member.name in claim.sentence

    def test_neural_components_are_detected_so_no_black_box_claim_is_made(self):
        res = CHECKS["neural_components_declared"]()
        assert res.supported
        assert "torch modules present" in res.detail
        claim = next(c for c in CLAIMS if c.id == "not_a_classifier")
        assert "design arm" in claim.sentence
        assert "false" in claim.limit

    def test_test_count_is_computed_and_not_quoted(self):
        md = render_claims_markdown()
        assert "528" not in md
        detail = CHECKS["test_suite_counted"]().detail
        quoted = int(detail.split()[0])
        assert quoted > 400

    def test_determinism_is_not_claimed(self):
        claim = next(c for c in CLAIMS if c.id == "seeded_reproducibility")
        assert "stochastic" in claim.sentence
        assert "deterministic" not in claim.sentence

    def test_engine_c_index_absence_is_a_claim_of_its_own(self):
        claim = next(c for c in CLAIMS if c.id == "no_engine_c_index")
        assert claim.tier == EvidenceStatus.UNKNOWN
        assert "0.5 by construction" in claim.sentence


class TestIndividualChecks:
    def test_seeded_reproducibility_distinguishes_seeds(self):
        res = check_seeded_reproducibility()
        assert res.supported
        assert "same seed reproduces exactly: True" in res.detail
        assert "different seed diverges: True" in res.detail

    def test_wild_type_check_re_derives_from_the_deposit(self):
        res = check_wild_type_from_deposit()
        assert res.supported
        assert "derived from the deposit: True" in res.detail

    def test_mechanistic_core_check_reads_the_engine(self):
        res = CHECKS["mechanistic_core"]()
        assert res.supported
        assert "PDE transport" in res.detail

    def test_voxel_scale_check_reports_millimetres_not_micrometres(self):
        res = CHECKS["voxel_scale"]()
        assert res.supported
        assert "mm from decimation factors" in res.detail
        assert "not 100 um" in res.detail
        # a micrometre grid would mean the decimation step was not applied
        assert "~3667 um" in res.detail

    def test_runtime_check_measures_rather_than_asserts(self):
        res = CHECKS["runtime_365"]()
        assert res.supported
        assert "re-measured at audit time" in res.detail
        seconds = float(res.detail.split(": ")[1].split(" s ")[0])
        assert 0.0 < seconds < 60.0

    def test_bulk_check_confirms_single_cell_inference_is_refused(self):
        res = CHECKS["bulk_not_single_cell"]()
        assert res.supported
        assert "bulk RNA-seq: True" in res.detail
        assert "PROVEN_FAILED in the module: True" in res.detail

    def test_quantitative_claims_carry_their_drift_in_the_limit(self):
        by_id = {c.id: c for c in CLAIMS}
        assert "100 um voxels" in by_id["multiscale_coarse_graining"].limit
        assert "machine-dependent" in by_id["runtime_is_measured"].limit
        assert "scRNA-seq-derived" in by_id["inputs_are_bulk"].limit


class TestRealDataClaims:
    """The claims added with the real-pharmacology and multimodal work.

    Each ties a poster sentence to the data it rests on, so the sentence cannot
    outlive the data (or vice versa).
    """

    def test_spatial_agreement_check_is_coupled_to_the_data_verdict(self):
        """The check must follow the verdict, not hard-code a refusal.

        If a segmented longitudinal collection ever appears, this check starts
        failing until the poster is updated -- that is the point.
        """
        from danon.multimodal import spatial_dice_verdict

        res = CHECKS["spatial_agreement_not_claimed"]()
        verdict = spatial_dice_verdict()
        assert res.supported == (not verdict.supported)
        assert "imaging collection(s) probed" in res.detail
        assert "segmentation series" in res.detail
        by_id = {c.id: c for c in CLAIMS}
        assert "Absence of a test is not a negative result" in (
            by_id["no_spatial_agreement_result"].limit
        )

    def test_potency_claim_is_backed_by_measured_records(self):
        res = CHECKS["potency_is_measured"]()
        assert res.supported
        assert "CHEMBL810" in res.detail
        assert "exact" in res.detail and "right-censored" in res.detail
        by_id = {c.id: c for c in CLAIMS}
        # The measured range must survive into the poster sentence.
        assert "496x" in by_id["potency_from_measured_records"].sentence
        assert "2D monolayer" in by_id["potency_from_measured_records"].limit

    def test_ec50_check_fails_when_a_unit_is_declared(self, monkeypatch):
        """The claim is 'no unit is declared'. Declaring one must invalidate it."""
        import danon.pharmacology as ph

        monkeypatch.setattr(ph, "engine_ec50_unit", lambda: "uM")
        res = CHECKS["ec50_unit_undeclared"]()
        assert res.supported is False
        assert "comparison permitted" in res.detail

    def test_ec50_claim_records_the_defect_rather_than_hiding_it(self):
        by_id = {c.id: c for c in CLAIMS}
        claim = by_id["engine_potency_scale_not_anchored"]
        assert "defect, not a feature" in claim.limit
        assert "unitless" in claim.sentence

    def test_new_claims_are_in_the_audited_set(self):
        ids = {c.id for c in CLAIMS}
        assert {
            "no_spatial_agreement_result",
            "potency_from_measured_records",
            "engine_potency_scale_not_anchored",
        } <= ids
        audits = {a.claim.id: a for a in audit_claims()}
        for cid in (
            "no_spatial_agreement_result",
            "potency_from_measured_records",
            "engine_potency_scale_not_anchored",
        ):
            assert audits[cid].supported, audits[cid].detail


class TestAppArtifact:
    """The claims artifact the app renders must not drift from the registry.

    This is the same class of check the README count needed: a number published
    in a file that nothing re-derives goes stale. The count and the ids are read
    from the committed artifact and compared against the live registry, so
    adding a claim without rebuilding the artifact fails here.
    """

    PATH = os.path.join("frontend", "public", "artifacts", "claims.json")

    def test_committed_artifact_matches_the_registry(self):
        if not os.path.exists(self.PATH):
            pytest.skip("claims artifact not built")
        d = json.load(open(self.PATH))["data"]
        assert d["n_claims"] == len(CLAIMS)
        assert d["n_passed"] + d["n_failed"] == d["n_claims"]
        assert {c["id"] for c in d["claims"]} == {c.id for c in CLAIMS}

    def test_failures_are_recorded_not_dropped(self):
        if not os.path.exists(self.PATH):
            pytest.skip("claims artifact not built")
        d = json.load(open(self.PATH))["data"]
        # every row carries its own verdict and detail, so a failing check is a
        # visible row rather than an absent one
        for row in d["claims"]:
            assert isinstance(row["supported"], bool)
            assert row["detail"].strip()
        assert d["n_failed"] == sum(1 for c in d["claims"] if not c["supported"])

    def test_artifact_says_passing_is_not_correctness(self):
        if not os.path.exists(self.PATH):
            pytest.skip("claims artifact not built")
        payload = json.load(open(self.PATH))
        notes = " ".join(payload["notes"]).lower()
        assert "not that the biology is right" in notes
        assert len(payload["data"]["cannot_establish"]) >= 3


class TestEnvelope:
    def test_provenance_marks_parameters_and_benefit_honestly(self):
        prov = claims_provenance()
        assert prov["parameters_are_measured"] == EvidenceStatus.PROVEN_FAILED
        assert prov["medical_benefit"].value == "unknown" if "medical_benefit" in prov else True
        assert prov["clinical_benefit"] == EvidenceStatus.UNKNOWN
        assert prov["engine_predicts_patients"] == EvidenceStatus.UNKNOWN

    def test_limits_refuse_the_mechanism_means_correct_inference(self):
        text = " ".join(what_these_claims_cannot_establish()).lower()
        assert "not correct" in text
        assert "prospective trials" in text
