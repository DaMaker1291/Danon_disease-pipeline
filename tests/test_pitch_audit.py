"""Tests for the architecture-write-up audit.

The audit's value is that its verdicts are TRUE OF THE CODE, so these tests do
not only check that a phrase matched: they re-derive the engine behaviour each
correction asserts, and they pin the two properties that make the audit
enforceable -- determinism, and that a rewritten document stops matching the
audited digest.
"""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from danon import evosteer as E
from danon.dosing import MyelosuppressionModel
from danon.pitch_audit import (
    AUDITED_DOCUMENT,
    DRIFT,
    DRIFT_RULES,
    ROOT,
    audit_document,
    audit_text,
    code_fact_snapshot,
    render_corrected_copy,
)


class TestDocumentIsAudited:
    def test_the_audited_document_exists_and_is_read(self):
        audit = audit_document(AUDITED_DOCUMENT)
        assert audit.n_words > 200
        assert len(audit.sha256) == 64

    def test_every_declared_rule_produces_a_finding(self):
        audit = audit_document(AUDITED_DOCUMENT)
        assert {f.rule_id for f in audit.findings} == {r.id for r in DRIFT_RULES}

    def test_the_submitted_writeup_trips_the_drift_rules(self):
        """The document under audit is the one that makes these statements."""
        audit = audit_document(AUDITED_DOCUMENT)
        flagged = {f.rule_id for f in audit.present}
        assert "deterministic_engine" in flagged
        assert "adaptive_switches_on_resistance" in flagged
        assert "anc_has_additive_recovery_term" in flagged
        assert {f.rule_id for f in audit.omitted}  # omissions are named too

    def test_every_finding_carries_a_correction_and_evidence(self):
        audit = audit_document(AUDITED_DOCUMENT)
        for f in audit.findings:
            if f.status in ("PRESENT", "OMITTED"):
                assert f.correction.strip(), f.rule_id
                assert f.evidence, f.rule_id

    def test_a_rewritten_document_stops_matching_the_audited_digest(self):
        """The digest is what makes a silent rewrite fail rather than pass."""
        full = os.path.join(ROOT, AUDITED_DOCUMENT)
        original = open(full, "rb").read()
        assert hashlib.sha256(original).hexdigest() == audit_document(
            AUDITED_DOCUMENT
        ).sha256
        rewritten = original.replace(b"deterministic, physics-constrained", b"stochastic")
        assert hashlib.sha256(rewritten).hexdigest() != hashlib.sha256(
            original
        ).hexdigest()

    def test_an_absent_rule_is_reported_as_absent_not_as_verified(self):
        audit = audit_text("a document that says nothing at all", path="<t>")
        by_rule = {f.rule_id: f for f in audit.findings}
        for rule in DRIFT_RULES:
            finding = by_rule[rule.id]
            assert finding.status in ("ABSENT", "OMITTED")
            assert finding.verdict == "" or rule.kind != DRIFT


class TestVerdictsAreTrueOfTheCode:
    """Each correction asserts engine behaviour; re-derive it here."""

    def test_adaptive_policy_is_burden_based_not_resistance_thresholded(self):
        policy = E.AdaptiveDosePolicy()

        def state(burden: float, initial: float, r_frac: float) -> E.PolicyState:
            return E.PolicyState(
                day=1, burden=burden, burden_initial=initial,
                resistant_fraction=r_frac,
            )

        # Fully resistant but below baseline burden -> nothing is given.
        assert policy.dose(state(9.0, 10.0, 1.0)) == (0.0, 0.0)
        # Almost fully sensitive but above baseline burden -> full dose.
        assert policy.dose(state(11.0, 10.0, 0.0)) == (1.0, 0.0)

    def test_the_resistance_threshold_controller_is_a_different_policy(self):
        policy = E.CollateralSwitchPolicy(theta_on=0.15, theta_off=0.05)

        def state(r_frac: float) -> E.PolicyState:
            return E.PolicyState(
                day=1, burden=1.0, burden_initial=1.0, resistant_fraction=r_frac,
            )

        assert policy.dose(state(0.20)) == (0.0, 1.0)   # switched to B
        assert policy.dose(state(0.01)) == (1.0, 0.0)   # and back
        with pytest.raises(ValueError):
            E.CollateralSwitchPolicy(theta_on=0.05, theta_off=0.15)

    def test_the_marrow_model_has_no_additive_recovery_term(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(MyelosuppressionModel)}
        assert not any("recovery" in n for n in names)
        # ... and the budget it does have is the label's own regimen.
        model = MyelosuppressionModel()
        anchors = model.anchor_checks(365)
        assert anchors
        assert model.anc_floor == 1.5

    def test_mutation_is_a_poisson_draw_on_realised_growth(self):
        source = open(
            os.path.join(ROOT, "danon", "evosteer.py"), encoding="utf-8"
        ).read()
        assert "new_R = rng.poisson(np.clip(grow_S" in source
        assert E.DEFAULT_CLONES.mutation_rate == 5e-05

    def test_pfs_is_a_burden_ratio_definition(self):
        assert E.RANO_PROGRESSION_FRACTION == 0.25
        source = open(
            os.path.join(ROOT, "danon", "evosteer.py"), encoding="utf-8"
        ).read()
        assert "pfs_threshold = (1.0 + RANO_PROGRESSION_FRACTION) * b_init" in source

    def test_immune_layer_and_motility_are_real_but_opt_in(self):
        import inspect

        sig = inspect.signature(E.TumorEvolutionSimulator.run)
        assert sig.parameters["immune"].default is None
        assert sig.parameters["growth_noise"].default == 0.05
        assert E.ImmuneParams().kill_max > 0
        source = open(
            os.path.join(ROOT, "danon", "evosteer.py"), encoding="utf-8"
        ).read()
        assert "motility * _laplacian" in source

    def test_evidence_vocabulary_is_not_two_states(self):
        from danon.cure import EvidenceStatus

        members = {m.name for m in EvidenceStatus}
        assert len(members) == 5
        assert {"ESTABLISHED", "PROVEN_FAILED"} <= members


class TestCorrectedReadingCopy:
    """The corrections a reviewer reads must be generated, never hand-edited."""

    def test_generated_copy_carries_the_digest_and_every_verdict(self):
        audit = audit_document(AUDITED_DOCUMENT)
        md = render_corrected_copy(audit)
        assert audit.sha256 in md
        assert AUDITED_DOCUMENT in md
        for verdict in ("REFUTED", "IMPRECISE", "QUALIFIED", "OMISSION"):
            assert verdict in md
        # the refuted sentences themselves are quoted, not paraphrased away
        assert "deterministic, physics-constrained" in md

    def test_copy_on_disk_is_byte_identical_to_a_fresh_render(self):
        path = os.path.join(ROOT, "outputs_real", "architecture_corrections.md")
        if not os.path.exists(path):
            pytest.skip("corrected reading copy not built")
        on_disk = open(path, encoding="utf-8").read()
        assert on_disk == render_corrected_copy(audit_document(AUDITED_DOCUMENT))


class TestFactSnapshotIsStableAndChecksSensitive:
    def test_snapshot_is_deterministic(self):
        assert code_fact_snapshot() == code_fact_snapshot()

    def test_snapshot_survives_a_json_round_trip(self):
        import json

        facts = code_fact_snapshot()
        assert json.loads(json.dumps(facts)) == facts
