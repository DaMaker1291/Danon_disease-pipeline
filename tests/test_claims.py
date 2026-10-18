"""Tests for the claims audit.

These tests exist because the ISEF pitch contained claims the code does not
support, and the drift is usually towards the more impressive sentence:

  * `DERIVED` is not an `EvidenceStatus` member, so the provenance vocabulary in
    the pitch was not the one in the repository;
  * the repository ships `torch.nn` modules and an ESM scorer, so "not a black
    box" is false as stated;
  * the quoted test count was stale.

Each is pinned here so the poster cannot drift back. The drift runs BOTH ways,
though, and this file has a fourth case to prove it: the measured forecast
verdict was rewritten when repeated folds resolved one task, while the sentence
the app published stayed at the previous reading. Every number was re-derived on
every run; the words about them were not, so the audit passed while the UI told
a reader that a task the measurement had resolved was unresolved. A supported
claim is therefore not enough on its own -- the published sentence has to be the
live one, which is what `test_the_app_publishes_this_audit_s_own_words` asserts.
"""

from __future__ import annotations

import json
import os
import re

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


_AUDIT_DETAILS: dict = {}


def _audit_details_once() -> dict:
    """Run the audit once per session; it re-derives the real cohort's folds."""
    if not _AUDIT_DETAILS:
        _AUDIT_DETAILS.update({a.claim.id: a.detail for a in audit_claims()})
    return _AUDIT_DETAILS


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

    def test_the_app_publishes_this_audit_s_own_words(self):
        """The mirrored artifact is a quotation of the audit, so it must match.

        Scoped deliberately to the one claim whose detail is a *conclusion about
        a measurement* rather than a census: a detail that counts live things
        (the test functions, for instance) legitimately moves whenever the tree
        does, and rebuilding the artifact for every such move would train people
        to ignore this test. The conclusion here is the case that actually broke
        -- the record's verdict moved and the published sentence did not -- so it
        is pinned rather than the whole registry.
        """
        audited = _audit_details_once()
        claim_id = "evolution_forecast_is_scored_against_baselines"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        canonical = os.path.join(root, "outputs_real", "app_artifacts", "claims.json")
        with open(canonical, encoding="utf-8") as fh:
            payload = json.load(fh)
        published = {
            c["id"]: c["detail"] for c in payload["data"]["claims"]
        }
        assert published[claim_id] == audited[claim_id], (
            "the claims artifact publishes a detail that the live audit no "
            "longer produces; rebuild it with "
            "write_app_artifacts(keys=['claims'])"
        )
        mirror = os.path.join(
            root, "frontend", "public", "artifacts", "claims.json"
        )
        with open(mirror, "rb") as fh:
            mirrored = fh.read()
        with open(canonical, "rb") as fh:
            source = fh.read()
        assert mirrored == source, (
            "the frontend mirror is not the canonical artifact, so the app is "
            "not rendering the file the repository holds"
        )

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
        res = CHECKS["test_suite_counted"]()
        quoted = int(res.detail.split()[0])
        assert quoted > 400
        # The README's own suite row is part of the check now: it advertised a
        # count 1068 for a 1059-function suite, which only hand-editing could
        # have kept true, so it defers to this check instead.
        assert res.supported
        assert "defers to this check" in res.detail

    def test_a_remembered_suite_count_in_the_readme_is_caught(self, monkeypatch):
        import danon.claims as claims_mod

        real_read = claims_mod._read

        def doctored(relpath):
            if relpath == "README.md":
                return "| Full integration suite | **1068 passed, 0 failed** | x |"
            return real_read(relpath)

        monkeypatch.setattr(claims_mod, "_read", doctored)
        res = claims_mod.check_test_suite_is_counted_live()
        assert not res.supported
        assert "remembered count" in res.detail

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
            "bench_anchors_persisted_and_calibration_refused",
            "alternating_win_has_a_reversal_map",
            "architecture_writeup_audited_against_the_code",
            "motility_constant_is_grid_dependent_and_priced",
        } <= ids
        audits = {a.claim.id: a for a in audit_claims()}
        for cid in (
            "no_spatial_agreement_result",
            "potency_from_measured_records",
            "engine_potency_scale_not_anchored",
            "bench_anchors_persisted_and_calibration_refused",
            "alternating_win_has_a_reversal_map",
            "architecture_writeup_audited_against_the_code",
            "motility_constant_is_grid_dependent_and_priced",
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


class TestInputProvenanceClaim:
    """A twin run must be tied to the uploaded file's bytes, not just described."""

    def test_check_is_registered_and_passes(self):
        res = CHECKS["twin_plan_input_is_byte_tied"]()
        assert res.supported, res.detail
        assert "sha256" in res.detail
        assert "ingestion path applies it" in res.detail

    def test_claim_is_in_the_audited_set(self):
        by_id = {c.id: c for c in CLAIMS}
        assert "twin_plan_input_is_byte_tied" in by_id
        claim = by_id["twin_plan_input_is_byte_tied"]
        assert "not evidence that the run predicts a patient" in claim.limit
        assert "block-max volume" in claim.limit

    def test_a_reverted_ingestion_path_fails_the_check(self, monkeypatch):
        """If the endpoint stops applying the scaled spacing, this must fail.

        The guard is only worth having if its absence is detectable, so the
        check is exercised against a doctored endpoint rather than trusting that
        it would notice.
        """
        import danon.claims as claims_mod

        real_read = claims_mod._read

        def doctored(relpath):
            text = real_read(relpath)
            if relpath == "api_server.py":
                return text.replace("derived_voxel_sizes", "SOURCE_SPACING")
            return text

        monkeypatch.setattr(claims_mod, "_read", doctored)
        res = claims_mod.check_twin_plan_input_is_byte_tied()
        assert res.supported is False
        assert "does not apply the derived spacing" in res.detail


class TestBadgeInventoryIsAnswered:
    """The headline a data-mining route makes -- 'the badges flip' -- counted.

    Every open badge is classified by what would close it, so the claim cannot
    be answered with a generic promise, and a new UNKNOWN has to be classified
    rather than absorbed into the old sentence.
    """

    def test_the_inventory_is_derived_live_from_the_registry(self):
        from danon.claims import badge_inventory

        inv = badge_inventory()
        assert inv["n_open"] == len(inv["badges"])
        assert inv["n_open"] == inv["n_claim_tiers"] + inv["n_provenance_keys"]
        assert inv["n_claim_tiers"] > 0 and inv["n_provenance_keys"] > 0
        assert inv["closed_by_a_public_download"] == 0
        for name, entry in inv["badges"].items():
            assert entry["would_close_on"], name
            assert entry["what_would_close_it"], name

    def test_the_check_passes_and_none_close_on_public_data(self):
        res = CHECKS["badge_inventory_is_classified_not_promised"]()
        assert res.supported, res.detail
        assert "open badges" in res.detail
        assert "closed by any of the four public datasets" in res.detail
        assert res.detail.rstrip().endswith("0")
        # The two benefit badges are the ones no dataset reaches.
        assert "prospective_trial" in res.detail

    def test_the_sentence_count_matches_the_registry(self):
        from danon.claims import badge_inventory

        claim = {c.id: c for c in CLAIMS}[
            "badge_inventory_is_classified_not_promised"
        ]
        match = re.search(r"(\d+)\s+badges? are open", claim.sentence)
        assert match is not None
        assert int(match.group(1)) == badge_inventory()["n_open"]
        assert "NONE of them closes" in claim.sentence
        assert "future" not in claim.limit

    def test_an_unclassified_badge_fails_the_check(self, monkeypatch):
        """A new UNKNOWN must be answered, not absorbed by the old sentence."""
        import danon.claims as claims_mod

        trimmed = dict(claims_mod.BADGE_CLOSURE)
        trimmed.pop("no_engine_c_index")
        monkeypatch.setattr(claims_mod, "BADGE_CLOSURE", trimmed)
        res = claims_mod.check_badge_inventory_is_answered_not_asserted()
        assert res.supported is False
        assert "carry no classification" in res.detail

    def test_a_remembered_count_fails_the_check(self, monkeypatch):
        import dataclasses

        import danon.claims as claims_mod

        rewritten = tuple(
            dataclasses.replace(
                claim, sentence=claim.sentence.replace("9 badges are open", "6 badges are open")
            )
            if claim.id == "badge_inventory_is_classified_not_promised"
            else claim
            for claim in claims_mod.CLAIMS
        )
        monkeypatch.setattr(claims_mod, "CLAIMS", rewritten)
        res = claims_mod.check_badge_inventory_is_answered_not_asserted()
        assert res.supported is False
        assert "rewrite it rather than" in res.detail


class TestDrugUnitsAreAFreeChoiceNotTheBlocker:
    """The potency gap is a MEASURED potency, not a unit declaration.

    The kill law is homogeneous of degree 0 in (concentration, EC50) and the
    transport is linear in its source, so relabelling units is an exact no-op --
    measured end to end and recorded. The control is in the record too: scaling
    the EC50 alone must move the trajectory, or "identical" would describe a
    measurement that cannot see anything.
    """

    def test_the_check_passes_and_quotes_the_labels_own_numbers(self):
        res = CHECKS["ec50_unit_undeclared"]()
        assert res.supported, res.detail
        assert "degree-0 homogeneous" in res.detail
        assert "free choice" in res.detail
        assert "7.5 mcg/mL oral" in res.detail and "7.3 mcg/mL IV" in res.detail
        assert "clearance 5.5 L/hr/m2" in res.detail
        assert "a MEASURED potency, not a unit" in res.detail

    def test_the_sentence_says_a_unit_would_not_fix_it(self):
        claim = {c.id: c for c in CLAIMS}["engine_potency_scale_not_anchored"]
        assert "bit for bit" in claim.sentence
        assert "CLEARANCE in L/hr/m2" in claim.sentence
        assert "without making the constant measured" in claim.sentence
        # The limit has to keep both blockers that survive the relabelling.
        assert "PLASMA concentration" in claim.limit
        assert "still not a per-day kill rate" in claim.limit

    def test_the_label_sentences_are_really_in_the_record(self):
        import danon.claims as claims_mod

        record = claims_mod._drug_units_probe_record()
        assert record is not None, "run _drug_units_probe.py"
        rel = record["relabelling"]
        assert rel["burden_arrays_bit_identical_when_both_scaled"] is True
        assert rel["burden_arrays_differ_when_only_ec50_scaled"] is True
        assert rel["scale"] == 1024.0
        text = " ".join(record["label"]["sentences"]).lower()
        assert "7.5 mcg/ml" in text and "7.3 mcg/ml" in text
        # The label writes it "5.5 L/hr/m 2": the superscript arrives as a
        # separate token once the markup is stripped, so the prefix is what is
        # stable and what carries the unit.
        assert "5.5 l/hr" in text and "clearance" in text
        assert "half-life is 1.8 hours" in text
        # The label does NOT call 5.5 a Cmax: that is the claim being refuted.
        assert not __import__("re").search(r"cmax[^.]{0,80}5\.5", text)

    def test_the_check_fails_if_the_relabelling_stops_being_exact(self, monkeypatch):
        import danon.claims as claims_mod

        original = claims_mod._drug_units_probe_record()
        assert original is not None
        doctored = json.loads(json.dumps(original))
        doctored["relabelling"]["burden_arrays_bit_identical_when_both_scaled"] = False
        monkeypatch.setattr(claims_mod, "_drug_units_probe_record", lambda: doctored)
        res = claims_mod.check_ec50_unit_is_undeclared()
        assert res.supported is False
        assert "not a free choice" in res.detail

    def test_the_check_fails_if_the_label_calls_5_5_a_cmax(self, monkeypatch):
        import danon.claims as claims_mod

        original = claims_mod._drug_units_probe_record()
        assert original is not None
        doctored = json.loads(json.dumps(original))
        doctored["label"]["sentences"].append(
            "Following a dose of 150 mg/m2 the mean Cmax is 5.5 mcg/mL ."
        )
        monkeypatch.setattr(claims_mod, "_drug_units_probe_record", lambda: doctored)
        res = claims_mod.check_ec50_unit_is_undeclared()
        assert res.supported is False
        assert "contradicts" in res.detail


class TestLumiereIsRefusedForTheMetricNotForAvailability:
    """Reachable and longitudinal, and still not scorable -- pinned both ways.

    The LUMIERE collection is the one longitudinal GBM collection the NBIA probe
    cannot answer for, so its refusal cannot rest on the same evidence as the
    other refusals. Its own article list says why: the masks ship as an
    '...automated segmentations' article, so a Dice against them scores this
    engine against another algorithm.
    """

    def test_the_check_passes_and_names_the_automated_masks(self):
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported, res.detail
        assert "LUMIERE is reachable" in res.detail
        assert "masks are AUTOMATED" in res.detail
        assert "model-versus-model" in res.detail
        assert "response class rather than a segmentation" in res.detail

    def test_the_sentence_admits_the_licence_and_the_arms_it_cannot_test(self):
        by_id = {c.id: c for c in CLAIMS}
        claim = by_id["walk_forward_is_refused_and_reasoned"]
        assert "automated segmentations" in claim.sentence
        assert "616" in claim.sentence and "model-versus-model" in claim.sentence
        # Reachable is a claim about a metadata API, and the licence is
        # unresolved -- both survive into the limit rather than being dropped.
        assert "no image was downloaded and no licence reviewed" in claim.limit
        assert "CC BY 4.0" in claim.limit and "non-commercial" in claim.limit

    def test_the_check_fails_if_lumiere_is_declared_scorable(self, monkeypatch):
        """The refusal is live: declaring it scorable must break the claim."""
        import danon.walkforward as wf

        real = wf.longitudinal_census

        def doctored(**kwargs):
            census = real(**kwargs)
            entry = dict(census["sources"]["LUMIERE"])
            entry["scorable_for_a_spatial_dice"] = True
            census["sources"] = dict(census["sources"])
            census["sources"]["LUMIERE"] = entry
            return census

        monkeypatch.setattr(wf, "longitudinal_census", doctored)
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported is False
        assert "scorable for a spatial Dice" in res.detail

    def test_the_check_fails_if_the_probe_loses_the_automated_article(self, monkeypatch):
        """The annotation is checked against the probe, not taken on trust."""
        import danon.walkforward as wf

        real = wf.longitudinal_census

        def doctored(**kwargs):
            census = real(**kwargs)
            figshare = json.loads(json.dumps(census["recorded_figshare"]))
            figshare["LUMIERE"]["articles"] = [
                {"id": 21266241, "title": "LUMIERE dataset - Readme file", "doi": None}
            ]
            census["recorded_figshare"] = figshare
            return census

        monkeypatch.setattr(wf, "longitudinal_census", doctored)
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported is False
        assert "automated segmentations" in res.detail

    def test_the_check_fails_if_a_lumiere_record_is_cached_here(self, monkeypatch):
        """A cached asset forces the refusal to be re-derived, not kept."""
        import danon.walkforward as wf

        real = wf.longitudinal_census

        def doctored(**kwargs):
            census = real(**kwargs)
            entry = dict(census["sources"]["LUMIERE"])
            entry["files_in_tree"] = 2
            census["sources"] = dict(census["sources"])
            census["sources"]["LUMIERE"] = entry
            return census

        monkeypatch.setattr(wf, "longitudinal_census", doctored)
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported is False
        assert "LUMIERE record(s) are now cached" in res.detail

    def test_the_check_fails_if_the_glass_records_lose_their_evidence(self, monkeypatch):
        """The GLASS tripwire ran one way and now runs the other.

        Real records arrived, so the check no longer fires on their presence: it
        fires if the records, the channel that measures them, or the channel's
        own refusals go missing while a sentence claims they are measured here.
        """
        import danon.walkforward as wf

        real = wf.longitudinal_census

        def without_artifact(**kwargs):
            census = real(**kwargs)
            sources = dict(census["sources"])
            glass = dict(sources["GLASS"])
            channel = dict(glass.get("variant_channel") or {})
            channel["built"] = False
            glass["variant_channel"] = channel
            sources["GLASS"] = glass
            census["sources"] = sources
            return census

        def without_records(**kwargs):
            census = without_artifact(**kwargs)
            sources = dict(census["sources"])
            glass = dict(sources["GLASS"])
            glass["files_in_tree"] = 0
            sources["GLASS"] = glass
            census["sources"] = sources
            return census

        monkeypatch.setattr(wf, "longitudinal_census", without_artifact)
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported is False
        assert "artifact is missing" in res.detail

        monkeypatch.setattr(wf, "longitudinal_census", without_records)
        res = CHECKS["walk_forward_is_refused_and_reasoned"]()
        assert res.supported is False
        assert "no GLASS record exists in this tree any more" in res.detail


class TestEpistemicGuardrailProposalIsAudited:
    """An 'epistemic type system' pitch, audited against the code not adopted.

    The proposal claims that wrapping every parameter in a provenance type,
    enforcing units with pint, hashing inputs, and printing a data pedigree
    makes hallucination syntactically impossible. The audit records what already
    exists here in a STRONGER form, what would weaken honesty if adopted, and
    that the proposal's worked examples are fabricated.
    """

    def test_the_check_passes_and_summarises_the_verdicts(self):
        res = CHECKS["epistemic_guardrail_proposal_is_audited"]()
        assert res.supported, res.detail
        assert "statistical leakage detector present" in res.detail
        assert "pint undeclared and unimported" in res.detail
        assert "engine EC50 unit still undeclared" in res.detail
        assert "plain floats" in res.detail

    def test_the_provenance_type_exists_for_fluidics_not_for_the_engine(self):
        import dataclasses

        from danon.evosteer import DEFAULT_CLONES
        from danon.physics import Constant, Provenance

        assert dataclasses.is_dataclass(Constant)
        assert "provenance" in {f.name for f in dataclasses.fields(Constant)}
        assert hasattr(Provenance, "ASSUMED")
        # The engine's constants are deliberately NOT wrapped in that type.
        assert isinstance(DEFAULT_CLONES.growth_S, float)

    def test_pint_is_absent_and_the_unit_guard_is_live(self):
        import sys

        from danon.pharmacology import engine_ec50_unit

        import danon.evosteer  # noqa: F401  -- ensure the engine is imported

        assert "pint" not in sys.modules
        # A declared unit would make the potency refusal look resolved.
        assert engine_ec50_unit() is None

    def test_the_leakage_guard_is_statistical_not_a_hash_lock(self):
        from danon import audit as audit_mod

        assert hasattr(audit_mod, "permutation_leakage_audit")
        assert hasattr(audit_mod, "permutation_null_c_index")

    def test_the_sentence_and_limit_are_honest_about_the_proposal(self):
        claim = {c.id: c for c in CLAIMS}[
            "epistemic_guardrail_proposal_is_audited"
        ]
        assert "CLEARANCE in L/hr/m2" in claim.sentence
        assert "fails on real leakage" in claim.sentence
        # The limit keeps the door open rather than denying the idea.
        assert "DERIVED rather than self-asserted" in claim.limit
        assert "not a claim that the current design is sufficient" in claim.limit

    def test_the_fabrication_scan_can_fail(self, tmp_path):
        from danon.claims import scan_for_fabricated_epistemic_figures

        (tmp_path / "danon").mkdir()
        (tmp_path / "danon" / "x.py").write_text(
            "tier = '" + "PATIENT_" + "MEASURED" + "'\n", encoding="utf-8"
        )
        fe = tmp_path / "frontend" / "src"
        fe.mkdir(parents=True)
        (fe / "Panel.tsx").write_text(
            "ci 'widened by 32" + "%'\n" + "conc = '5.5 " + "ug/mL'\n",
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_epistemic_figures(str(tmp_path))
        assert scanned == 2
        assert len(problems) >= 3
        joined = " ".join(problems)
        assert "self-declared measured tier" in joined
        assert "widened by a fixed percent" in joined
        assert "5.5 ug/mL brain concentration" in joined

    def test_a_clean_tree_scans_without_problems(self, tmp_path):
        from danon.claims import scan_for_fabricated_epistemic_figures

        (tmp_path / "danon").mkdir()
        (tmp_path / "danon" / "x.py").write_text("x = 1.0\n", encoding="utf-8")
        problems, scanned = scan_for_fabricated_epistemic_figures(str(tmp_path))
        assert scanned == 1 and problems == []

    def test_the_check_fails_if_a_declared_measurement_tier_appears(
        self, monkeypatch
    ):
        import dataclasses

        import danon.claims as claims_mod
        import danon.evosteer as evosteer

        monkeypatch.setattr(
            evosteer,
            "DEFAULT_CLONES",
            dataclasses.replace(evosteer.DEFAULT_CLONES, growth_S="MEASURED"),
        )
        res = claims_mod.check_epistemic_guardrail_proposal_is_audited()
        assert res.supported is False
        assert "no longer a plain float" in res.detail

    def test_the_check_fails_if_the_engine_declares_an_ec50_unit(
        self, monkeypatch
    ):
        import danon.claims as claims_mod
        import danon.pharmacology as pharm

        monkeypatch.setattr(pharm, "engine_ec50_unit", lambda: "uM")
        res = claims_mod.check_epistemic_guardrail_proposal_is_audited()
        assert res.supported is False
        assert "declares an EC50 unit" in res.detail

    def test_the_check_fails_if_pint_is_loaded(self, monkeypatch):
        import sys

        import danon.claims as claims_mod

        monkeypatch.setitem(sys.modules, "pint", object())
        res = claims_mod.check_epistemic_guardrail_proposal_is_audited()
        assert res.supported is False
        assert "pint is loaded" in res.detail


class TestSamdPathwayIsNamedNotPromised:
    """A roadmap to FDA clearance is NAMED as context, never claimed as status.

    The repository is a research prototype; nothing has been submitted. The
    check fails if a shipped surface claims a clearance, if the engine grows a
    GPU/JAX backend, or if a benefit badge moves.
    """

    def test_the_check_passes(self):
        res = CHECKS["samd_pathway_is_named_not_promised"]()
        assert res.supported, res.detail
        assert "research prototype, not a device" in res.detail
        assert "benefit badges pinned" in res.detail

    def test_the_context_is_named_and_honest(self):
        from danon.claims import SAMD_CONTEXT

        assert "not a device" in SAMD_CONTEXT["status"]
        assert "520(o)(1)(E)" in SAMD_CONTEXT["cds_exemption"]
        assert "510(k)" in SAMD_CONTEXT["pathway"]
        assert "Predetermined Change Control Plan" in SAMD_CONTEXT["adaptive_updates"]
        for standard in ("IEC 62304", "ISO 13485", "ISO 14971"):
            assert standard in SAMD_CONTEXT["standards"]
        assert "numpy only" in SAMD_CONTEXT["engine"]

    def test_the_engine_is_numpy_and_the_anc_model_disclaims_cds(self):
        import inspect

        from danon.dosing import MyelosuppressionModel

        import danon.evosteer  # noqa: F401

        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        assert "nothing here is clinical decision support" in doc or (
            "not clinical decision support" in doc
        )
        # No accelerator backend anywhere in the engine modules.
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        for rel in ("danon/evosteer.py", "danon/anisotropy.py", "danon/accel.py"):
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
            assert "import jax" not in text and "import cupy" not in text

    def test_the_day_365_readout_stays_refuted_as_a_prediction(self):
        from danon.outcome_export import outcome_export_provenance

        assert (
            outcome_export_provenance()["day_365_readout_is_a_patient_prediction"]
            is EvidenceStatus.PROVEN_FAILED
        )

    def test_the_scan_separates_status_from_a_future_gate(self, tmp_path):
        from danon.claims import scan_for_fabricated_device_status

        (tmp_path / "danon").mkdir()
        (tmp_path / "danon" / "a.py").write_text(
            "note = 'an FDA-cleared SaMD'\n", encoding="utf-8"
        )
        fe = tmp_path / "frontend" / "src"
        fe.mkdir(parents=True)
        (fe / "b.tsx").write_text(
            "requires MHRA/FDA clearance before any therapeutic use\n",
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_device_status(str(tmp_path))
        assert scanned == 2
        assert len(problems) == 1
        assert "FDA clearance or approval as a status" in problems[0]
        assert "b.tsx" not in problems[0]

    def test_the_check_fails_if_the_engine_imports_jax(self, monkeypatch, tmp_path):
        import danon.claims as claims_mod

        (tmp_path / "danon").mkdir()
        (tmp_path / "danon" / "evosteer.py").write_text(
            "import jax\n", encoding="utf-8"
        )
        monkeypatch.setattr(claims_mod, "ROOT", str(tmp_path))
        res = claims_mod.check_samd_pathway_is_named_not_promised()
        assert res.supported is False
        assert "imports a JAX/CuPy backend" in res.detail

    def test_the_check_fails_if_a_benefit_badge_moves(self, monkeypatch):
        import danon.claims as claims_mod

        real = claims_mod.claims_provenance

        def doctored():
            prov = dict(real())
            prov["clinical_benefit"] = EvidenceStatus.ESTABLISHED
            return prov

        monkeypatch.setattr(claims_mod, "claims_provenance", doctored)
        res = claims_mod.check_samd_pathway_is_named_not_promised()
        assert res.supported is False
        assert "clinical_benefit is no longer UNKNOWN" in res.detail


class TestBenefitSafetyAndRuntimeAreMeasured:
    """A 'saves lives / guarantees safety / 10,000 in 30 s' pitch, checked.

    The benefit and safety claims are refused (they need trials), and the
    runtime is the measured sweep rather than a hopeful number.
    """

    def test_the_check_passes(self):
        res = CHECKS["benefit_safety_and_runtime_are_measured_not_asserted"]()
        assert res.supported, res.detail
        assert "survival claim refused" in res.detail
        assert "~1.4 h measured, not seconds" in res.detail

    def test_the_precedent_is_context_not_a_result(self):
        from danon.claims import TRANSLATION_PRECEDENT

        text = TRANSLATION_PRECEDENT.lower()
        assert "not a result of this engine" in text
        assert "phase 2a" in text and "non-randomised" in text
        assert "nct02415621" in text

    def test_the_safety_screen_is_label_anchored_and_disclaims_cds(self):
        import inspect

        from danon.dosing import LABEL_ANC_FLOOR, MyelosuppressionModel

        assert float(LABEL_ANC_FLOOR) == 1.5
        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        assert "clinical decision support" in doc

    def test_the_runtime_is_the_measured_sweep_not_seconds(self):
        import json
        import pathlib

        repo = pathlib.Path(__file__).resolve().parent.parent
        perf = json.loads((repo / "outputs_real" / "perf.json").read_text())
        sweep = perf["uq_sweep_extrapolation"]
        assert sweep["draws"] == 10000
        # 1.41 h on 4 workers; a seconds-scale figure would be two orders off.
        assert sweep["hours_on_4_workers"] > 0.5

    def test_the_scan_flags_fabrications_but_not_a_benign_guarantee(self, tmp_path):
        from danon.claims import scan_for_fabricated_benefit_claims

        (tmp_path / "danon").mkdir()
        (tmp_path / "danon" / "a.py").write_text(
            "x = 'the software saves lives'\n"
            "y = 'guarantees the schedule will not cause harm'\n"
            "z = 'secures an additional year of life'\n"
            "r = '10,000 schedules in 30 seconds'\n",
            encoding="utf-8",
        )
        (tmp_path / "danon" / "b.py").write_text(
            "# streams are wired with identity guarantees\n", encoding="utf-8"
        )
        problems, scanned = scan_for_fabricated_benefit_claims(str(tmp_path))
        assert scanned == 2
        joined = " ".join(problems)
        assert "saves lives" in joined
        assert "safety guarantee to a patient" in joined
        assert "added year of life" in joined
        assert "quoted in seconds" in joined
        assert "b.py" not in joined  # a benign 'identity guarantees' is not a claim

    def test_the_check_fails_if_the_sweep_is_quoted_in_seconds(self, monkeypatch, tmp_path):
        import json

        import danon.claims as claims_mod

        out = tmp_path / "outputs_real"
        out.mkdir(parents=True)
        (out / "perf.json").write_text(
            json.dumps(
                {"uq_sweep_extrapolation": {"draws": 10000, "hours_on_4_workers": 0.008}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(claims_mod, "ROOT", str(tmp_path))
        res = claims_mod.check_benefit_safety_and_runtime_are_measured_not_asserted()
        assert res.supported is False
        assert "seconds-scale figure" in res.detail


class TestProvenanceTiersAreDerivedFromEvidence:
    """A tier may be DERIVED, but only semantically -- never from presence or
    from the audit result."""

    def test_the_check_passes(self):
        res = CHECKS["provenance_tiers_are_derived_from_evidence_not_declared"]()
        assert res.supported, res.detail
        assert "LUMIERE rule" in res.detail
        assert "never a tier" in res.detail

    def test_a_presence_only_bundle_cannot_raise_a_tier(self):
        from danon.claims import evidence_justifies_a_tier_upgrade

        # LUMIERE: reachable, 91 patients, 2487 images, AUTOMATED masks.
        ok, why = evidence_justifies_a_tier_upgrade(
            {"exists": True, "count": 2487, "annotation": "automated"}
        )
        assert ok is False
        assert "expert" in why

        semantic = {
            "kind": "labelling_index",
            "annotation": "expert",
            "patient_linked": True,
            "units_checked": True,
        }
        assert evidence_justifies_a_tier_upgrade(semantic)[0] is True

    def test_the_r0_tier_derives_to_unknown_for_want_of_a_numerator(self):
        from danon.claims import derive_r0_knowledge_tier

        tier, reason = derive_r0_knowledge_tier()
        assert tier is EvidenceStatus.UNKNOWN
        assert "labelling index" in reason

    def test_derivations_read_artifacts_not_the_audit(self, monkeypatch):
        """Rule 1: a derivation must not call the audit (it would recurse)."""
        import danon.claims as claims_mod

        def boom(*_a, **_k):
            raise AssertionError("a derivation called the audit -- it recurses")

        monkeypatch.setattr(claims_mod, "audit_claims", boom)
        for key in list(claims_mod.DERIVED_TIERS):
            claims_mod.derived_tier(key)  # must not raise

    def test_a_tier_is_not_the_audit_result(self):
        """Rule 2: `supported` and `tier` are orthogonal."""
        by_id = {c.id: c for c in CLAIMS}
        claim = by_id["no_engine_c_index"]
        assert claim.tier is EvidenceStatus.UNKNOWN
        assert CHECKS[claim.check]().supported is True

    def test_a_presence_only_source_does_not_upgrade_but_a_semantic_one_does(
        self, monkeypatch, tmp_path
    ):
        import json

        import danon.claims as claims_mod

        out = tmp_path / "outputs_real"
        out.mkdir(parents=True)
        (out / "ki67_source_probe.json").write_text(
            json.dumps(
                {
                    "ingested_labelling_index_sources": [
                        {"exists": True, "count": 91, "annotation": "automated"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(claims_mod, "ROOT", str(tmp_path))
        assert claims_mod.derive_r0_knowledge_tier()[0] is EvidenceStatus.UNKNOWN

        (out / "ki67_source_probe.json").write_text(
            json.dumps(
                {
                    "ingested_labelling_index_sources": [
                        {
                            "kind": "labelling_index",
                            "annotation": "expert",
                            "patient_linked": True,
                            "units_checked": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert claims_mod.derive_r0_knowledge_tier()[0] is EvidenceStatus.ESTABLISHED

    def test_the_check_fails_if_the_audit_result_becomes_a_tier(self, monkeypatch):
        import danon.claims as claims_mod

        monkeypatch.setitem(
            claims_mod.DERIVED_TIERS,
            "claim_matches_code",
            lambda: (EvidenceStatus.ESTABLISHED, "from the audit"),
        )
        res = claims_mod.check_provenance_tiers_are_derived_from_evidence_not_declared()
        assert res.supported is False
        assert "recurses" in res.detail


class TestCohortSpecificationMatchesTheEngine:
    """A cohort specification handed to a hospital data steward is a document a
    protocol copies from, so every number it states is pinned to the engine and
    the pitch's own contradictions are caught."""

    CHECK = "cohort_specification_matches_the_engine_not_the_pitch"

    def test_the_check_passes_and_summarises_the_corrections(self):
        res = CHECKS[self.CHECK]()
        assert res.supported, res.detail
        assert "necrosis NOT zero" in res.detail
        assert "0.9310" in res.detail and "0.3905" in res.detail
        assert "41 retrievable" in res.detail
        assert "UNKNOWN" in res.detail

    def test_the_tissue_map_is_read_from_the_engine_not_the_pitch(self):
        import numpy as np

        from danon.realdata import _label_proliferation

        labels = np.zeros((1, 1, 3), dtype=np.int32)
        labels[0, 0, 0], labels[0, 0, 1], labels[0, 0, 2] = 4, 2, 1
        prolif, usable = _label_proliferation(labels, labels > 0)
        assert usable
        weights = (
            float(prolif[0, 0, 0]),
            float(prolif[0, 0, 1]),
            float(prolif[0, 0, 2]),
        )
        assert weights == (1.0, 0.6, 0.2)
        # The pitch maps the necrotic core to K=0; the engine does not.
        assert weights[2] > 0.0

    def test_the_voxel_cap_is_the_engines_own(self):
        from danon.imaging import MAX_TOTAL_VOXELS

        assert int(MAX_TOTAL_VOXELS) == 4_000_000

    def test_the_spec_calls_the_decimation_defect_a_defect(self):
        from danon.claims import COHORT_SPEC_CORRECTIONS

        with open(
            os.path.join("outputs_real", "grid_artifact.json"), encoding="utf-8"
        ) as fh:
            grid = json.load(fh)
        defect = float(grid["geometry"]["shipped_rule_dice_vs_axis_aligned"])
        documented = float(
            grid["geometry"]["levels"]["engine_3p6mm"]["dice_vs_source"]
        )
        assert abs(defect - 0.3905) < 1e-3
        assert documented > defect
        assert "DEFECT" in COHORT_SPEC_CORRECTIONS["the_penalty_is_inverted"]

    def test_the_spec_does_not_imply_the_data_does_not_exist(self):
        from danon.claims import COHORT_SPECIFICATION
        from danon.clinical_twin import clinical_stream_census

        s4 = clinical_stream_census()["stream4_sequential"]
        assert s4["n_pairs_cached"] == 0
        assert s4["n_pairs_retrievable"] == 41
        assert "RETRIEVABLE" in COHORT_SPECIFICATION["availability"]

    def test_the_check_fails_if_necrosis_becomes_zero(self, monkeypatch):
        import numpy as np

        import danon.realdata as realdata

        def zero_necrosis(labels_grid, mask):
            out = np.zeros(labels_grid.shape, dtype=float)
            out[labels_grid == 4] = 1.0
            out[labels_grid == 2] = 0.6
            out[labels_grid == 1] = 0.0
            return out, True

        monkeypatch.setattr(realdata, "_label_proliferation", zero_necrosis)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "necrosis" in res.detail.lower()

    def test_the_scan_flags_a_fabricated_cohort_claim(self, tmp_path):
        from danon.claims import scan_for_fabricated_cohort_spec_claims

        d = tmp_path / "danon"
        d.mkdir()
        (d / "spec.py").write_text(
            "The necrotic core maps to K = 0, the documented Dice penalty is "
            "0.3905, 0 scorable pairs by danon/audit.py, and N=30 powers it "
            "above the noise floor for uncensored patients.",
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_cohort_spec_claims(str(tmp_path))
        assert scanned == 1
        assert len(problems) == 4


class TestClinicalExecutionUiIsAudited:
    """A 'clinical execution UI' directive asks for four things this repository
    refuses -- a prescription block, a mode that hides the warnings, a silent
    instant sweep, and a days-of-life-secured figure. Each is pinned live here
    so a CDS surface cannot drift into a device claim."""

    CHECK = "clinical_execution_ui_is_audited_not_implemented"

    def test_the_check_passes_and_summarises_the_refusals(self):
        res = CHECKS[self.CHECK]()
        assert res.supported, res.detail
        assert "19-day" in res.detail
        assert "UNKNOWN" in res.detail and "PROVEN_FAILED" in res.detail
        assert "MHRA/FDA" in res.detail

    def test_the_prescription_is_the_label_regimen_not_an_optimum(self):
        from danon.claims import CLINICAL_UI_CORRECTIONS
        from danon.dosing import LABEL_CYCLE_DAYS, LABEL_DOSE_MG_M2, LABEL_ON_DAYS

        assert float(LABEL_DOSE_MG_M2) == 150.0
        assert int(LABEL_ON_DAYS) == 5
        # The engine's own label anchor is the 28-day cycle, so the directive's
        # '5 days on, then a 14-day holiday' (a 19-day cycle) contradicts it.
        assert int(LABEL_CYCLE_DAYS) == 28
        assert int(LABEL_ON_DAYS) + 14 != int(LABEL_CYCLE_DAYS)
        assert "19" in CLINICAL_UI_CORRECTIONS["the_cycle_arithmetic_is_wrong"]

    def test_the_safety_floor_is_a_screen_that_disclaims_cds(self):
        import inspect

        from danon.dosing import MyelosuppressionModel

        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        assert "clinical decision support" in doc
        assert hasattr(MyelosuppressionModel, "feasible")

    def test_the_lifespan_figure_is_refused(self):
        prov = claims_provenance()
        assert prov["clinical_benefit"] == EvidenceStatus.UNKNOWN
        assert prov["engine_predicts_patients"] == EvidenceStatus.UNKNOWN

    def test_the_sweep_is_measured_not_a_background_instant(self):
        with open(os.path.join("outputs_real", "perf.json"), encoding="utf-8") as fh:
            perf = json.load(fh)
        sweep = perf["uq_sweep_extrapolation"]
        assert int(sweep["draws"]) == 10000
        assert float(sweep["hours_on_4_workers"]) >= 0.5

    def test_the_frontend_gate_still_requires_clearance(self):
        p = os.path.join(
            "frontend", "src", "vector_design", "TranslationalGateway.tsx"
        )
        with open(p, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        assert "clearance before any therapeutic use" in text
        assert os.path.exists(
            os.path.join("frontend", "src", "utils", "evidenceTier.ts")
        )

    def test_the_check_fails_if_a_benefit_badge_is_moved(self, monkeypatch):
        import danon.claims as claims_mod

        def moved():
            return {
                "claim_matches_code": claims_mod.EvidenceStatus.ESTABLISHED,
                "mechanism_is_mechanistic": claims_mod.EvidenceStatus.ESTABLISHED,
                "parameters_are_measured": claims_mod.EvidenceStatus.PROVEN_FAILED,
                "engine_predicts_patients": claims_mod.EvidenceStatus.ESTABLISHED,
                "clinical_benefit": claims_mod.EvidenceStatus.ESTABLISHED,
            }

        monkeypatch.setattr(claims_mod, "claims_provenance", moved)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "off UNKNOWN" in res.detail

    def test_the_scan_flags_a_fabricated_clinical_ui_claim(self, tmp_path):
        from danon.claims import scan_for_fabricated_clinical_ui_claims

        d = tmp_path / "frontend" / "src"
        d.mkdir(parents=True)
        (d / "Panel.tsx").write_text(
            "Clinical mode: we hide the warnings. Recommended protocol: "
            "150 mg/m2 for 5 days. It secures +40 additional days of "
            "progression-free life.",
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_clinical_ui_claims(str(tmp_path))
        assert scanned == 1
        assert len(problems) == 4


class TestPairedPfsExtensionIsRecorded:
    """The adaptive-vs-MTD benefit is carried as two labelled, non-interchangeable
    estimands, and the paired one is DERIVED from the recorded replicates rather
    than asserted."""

    CHECK = "paired_pfs_extension_is_recorded_not_fabricated"

    def test_the_check_passes(self):
        res = CHECKS[self.CHECK]()
        assert res.supported, res.detail
        assert "bootstrap 95% interval" in res.detail
        assert "ratio of medians" in res.detail
        assert "UNKNOWN" in res.detail

    def test_the_interval_rederives_from_the_recorded_log(self):
        from danon.paired_extension import paired_pfs_extension_report

        r = paired_pfs_extension_report()
        assert r["n_cases"] == 18 and r["n_pairs"] == 54
        lo, hi = r["paired_mean_difference_ci95_days"]
        assert lo < r["paired_mean_difference_days"] < hi
        # deterministic on the fixed seed: the same bytes give the same interval
        assert paired_pfs_extension_report() == r

    def test_it_is_a_right_censored_lower_bound(self):
        from danon.paired_extension import paired_pfs_extension_report

        r = paired_pfs_extension_report()
        assert r["is_right_censored_lower_bound"] is True
        assert r["n_policy_replicates_at_horizon"] == r["n_pairs"]

    def test_the_two_estimands_are_different_quantities(self):
        from danon.paired_extension import paired_pfs_extension_report

        r = paired_pfs_extension_report()
        # days vs a per-case fraction -- not the same number under another unit
        assert r["paired_median_difference_days"] > 1.0
        assert 0.0 < r["ratio_of_medians_extension_median"] < 5.0
        assert "not interchangeable" in r["estimand_note"].lower()

    def test_the_committed_artifact_matches_the_report(self):
        path = os.path.join("frontend", "public", "artifacts", "paired_extension.json")
        if not os.path.exists(path):
            pytest.skip("paired_extension artifact not built")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)["data"]
        from danon.paired_extension import paired_pfs_extension_report

        live = paired_pfs_extension_report()
        assert data["n_pairs"] == live["n_pairs"]
        assert (
            data["paired_mean_difference_ci95_days"]
            == live["paired_mean_difference_ci95_days"]
        )

    def test_the_check_fails_when_the_interval_is_dropped(self, monkeypatch):
        import danon.app_artifacts as aa
        import danon.paired_extension as pe

        # Doctor the RECORDED payload: the guard rails against an artifact that
        # carries a bare point estimate where the interval belongs.
        doctored = {"data": dict(pe.paired_pfs_extension_report())}
        doctored["data"]["paired_mean_difference_ci95_days"] = None
        monkeypatch.setitem(aa._BUILT, "paired_extension", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "bare point estimate" in res.detail


class TestTrialDesignScreenIsDerived:
    """The trial-design screen states the counterfactual, shows the noise, and
    rules calendars out -- and stops short of naming one to give. Every figure is
    derived from the recorded paired replicates, and the rule-out is pinned to the
    DECLARED ANC screen rather than to a regulatory boundary."""

    CHECK = "trial_design_screen_is_derived_not_prescribed"

    def test_the_check_passes(self):
        res = CHECKS[self.CHECK]()
        assert res.supported, res.detail
        assert "noise floor" in res.detail
        assert "DECLARED ANC screen" in res.detail
        assert "UNKNOWN" in res.detail

    def test_the_counterfactual_rederives_from_the_outcome_cards(self):
        from danon.trial_design import trial_design_report

        r = trial_design_report()
        assert r["n_cases"] == 9 and r["n_pairs"] == 27
        lo, hi = r["pooled"]["intervals"]["ci95"]
        assert lo < r["pooled"]["mean_difference_days"] < hi
        # deterministic: no engine run, so the same bytes give the same numbers
        assert trial_design_report() == r

    def test_the_noise_floor_keeps_two_unresolved_states_distinct(self):
        from danon.trial_design import (
            INSUFFICIENT_REPLICATES_FLAG,
            MIN_PAIRS_FOR_AN_INTERVAL,
            NOISE_FLOOR_FLAG,
            trial_design_report,
        )

        assert NOISE_FLOOR_FLAG != INSUFFICIENT_REPLICATES_FLAG
        r = trial_design_report()
        # 3 pairs per case is below the method's own minimum, so every case is
        # unassessed rather than "within a noise floor" -- a different statement.
        assert r["n_cases"] * 3 == r["n_pairs"]
        assert MIN_PAIRS_FOR_AN_INTERVAL == 5
        statuses = {c["noise_floor"]["status"] for c in r["cases"]}
        assert statuses == {"no_interval_at_this_replicate_count"}
        assert all(c["noise_floor"]["flag"] == INSUFFICIENT_REPLICATES_FLAG for c in r["cases"])
        # the pooled scale does resolve, so the flag is not fired everywhere
        assert r["pooled"]["noise_floor"]["status"] == "resolves"
        assert r["pooled"]["noise_floor"]["flag"] is None
        assert r["pooled"]["noise_floor"]["n_separated"] > 0

    def test_the_resolvability_ladder_reports_three_scales(self):
        from danon.trial_design import trial_design_report

        r = trial_design_report()
        ladder = r["resolvability"]["ladder"]
        assert [rung["scale"] for rung in ladder] == [
            "within one case",
            "pooled over one image",
            "pooled over the whole record",
        ]
        assert ladder[0]["groups_without_a_resolvable_interval"] == len(r["cases"])
        assert ladder[1]["groups"] == len(r["by_geometry"])
        assert ladder[-1]["pairs_per_group"] == r["n_pairs"]

    def test_every_pool_owns_its_own_summary(self):
        """A pool carries its interval, its state and its PFS medians, so a
        consumer renders what the pool says instead of re-deriving it -- which is
        what keeps the browser's live groups and this record the same shape."""
        from danon.trial_design import trial_design_report

        r = trial_design_report()
        assert r["by_geometry"], "the record must carry at least one image pool"
        for group in r["by_geometry"]:
            for field in (
                "n_pairs",
                "mean_difference_days",
                "median_pfs_policy_days",
                "median_pfs_mtd_days",
                "intervals",
                "noise_floor",
            ):
                assert field in group, f"{group['case_id']} is missing {field}"
            # the pool's own pair count is the pool's own rows, not a parallel tally
            assert group["n_pairs"] == len(group["diffs_days"])
            assert group["n_pairs"] == r["resolvability"]["ladder"][1]["pairs_per_group"]
        assert r["by_geometry"][0]["n_pairs"] > r["cases"][0]["n_pairs"]
        # the live screen mirrors this template word for word; if it stops being
        # count-derived, the screen's own cross-check starts reporting a drift
        assert r["resolvability"]["note"].startswith(
            "a pool without a resolvable interval"
        )

    def test_the_rule_out_is_the_declared_screen_not_a_regulator(self):
        import inspect

        from danon.dosing import MyelosuppressionModel
        from danon.trial_design import declared_anc_ruleout

        anc = declared_anc_ruleout()
        assert anc["declared"] is True
        assert anc["is_a_regulatory_boundary"] is False
        assert anc["budget_continuous_mg_m2"] > 0
        assert "not clinical decision support" in (
            inspect.getdoc(MyelosuppressionModel) or ""
        ).lower() or "nothing here is clinical decision support" in (
            inspect.getdoc(MyelosuppressionModel) or ""
        ).lower()
        rows = {c["label"]: c for c in anc["candidates"]}
        # the brief's own '5 on / 14 off' arithmetic is a 19-day cycle and is
        # refused by the declared screen; the label's 5-of-28 row is permitted
        assert rows["brief_arithmetic_150x5_of_19"]["feasible"] is False
        assert rows["label_regimen_150x5_of_28"]["feasible"] is True
        assert anc["n_rejected"] >= 1

    def test_the_inversion_day_is_measured_and_shows_no_delay(self):
        from danon.trial_design import trial_design_report

        r = trial_design_report()
        delays = [
            c["inversion_delay_days"]
            for c in r["cases"]
            if c["inversion_delay_days"] is not None
        ]
        assert len(delays) == r["n_cases"]
        assert set(delays) == {0}
        assert r["inversion"]["n_cases_with_a_delay"] == 0
        assert "no simulated delay" in r["inversion"]["delay_note"]

    def test_the_policy_module_carries_the_recorded_labels(self):
        from danon.trial_design import (
            INSUFFICIENT_REPLICATES_FLAG,
            NOISE_FLOOR_FLAG,
        )

        # The flags belong to the module that owns the noise-floor policy, not
        # to the view that renders its output.
        p = os.path.join("frontend", "src", "utils", "trialDesign.ts")
        with open(p, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        assert NOISE_FLOOR_FLAG in text
        assert INSUFFICIENT_REPLICATES_FLAG in text
        measured = []
        for rel in (("utils", "trialDesign.ts"), ("components", "TrialDesign.tsx")):
            q = os.path.join("frontend", "src", *rel)
            with open(q, encoding="utf-8") as fh:
                measured.append(fh.read().lower())
        for body in measured:
            # neither the policy module nor the view publishes prescription or
            # hidden-benefit vocabulary
            assert "recommended protocol" not in body
            assert "additional days" not in body

    def test_the_committed_artifact_matches_the_report(self):
        path = os.path.join("frontend", "public", "artifacts", "trial_design.json")
        if not os.path.exists(path):
            pytest.skip("trial_design artifact not built")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)["data"]
        from danon.trial_design import trial_design_report

        live = trial_design_report()
        assert data["n_pairs"] == live["n_pairs"]
        assert data["pooled"]["mean_difference_days"] == live["pooled"]["mean_difference_days"]
        assert data["pooled"]["intervals"]["ci95"] == live["pooled"]["intervals"]["ci95"]
        assert data["anc_screen"]["rejected"] == live["anc_screen"]["rejected"]

    def test_the_pooled_count_discloses_the_nesting(self):
        """A count of pairs must never be published as a count of independent
        units: the record nests 3 seeds inside each of 9 configurations, and the
        configurations sit 3 to a geometry."""
        from danon.trial_design import trial_design_report

        r = trial_design_report()
        pooled = r["pooled"]["nesting"]
        assert pooled["assumed_exchangeable"] == r["n_pairs"] == 27
        assert pooled["independent_units"] == 9
        assert pooled["n_configurations"] == 9
        assert pooled["n_geometries"] == 3
        assert pooled["configurations_per_geometry"] == 3
        assert pooled["seeds_per_configuration"] == 3
        # the whole point: fewer independent units than pairs
        assert pooled["independent_units"] < pooled["assumed_exchangeable"]
        assert "at most 9 nested units" in pooled["note"]
        for group in r["by_geometry"]:
            nesting = group["nesting"]
            assert nesting["assumed_exchangeable"] == 9
            assert nesting["independent_units"] == 3
            assert nesting["n_geometries"] == 1
            assert nesting["independent_units"] < nesting["assumed_exchangeable"]

    def test_the_view_renders_the_nesting_disclosure(self):
        p = os.path.join("frontend", "src", "components", "TrialDesign.tsx")
        with open(p, encoding="utf-8") as fh:
            text = " ".join(fh.read().split())
        # references the label the policy module defines, rather than restating it
        assert "NESTED_REPLICATES_FLAG" in text
        for field in (
            "nesting.independent_units",
            "nesting.assumed_exchangeable",
            "nesting.note",
        ):
            assert field in text, field

    def test_a_count_presented_as_independent_is_caught(self):
        """The rule must catch the CLAIM -- a count of replicates asserted to be
        an independent sample -- however a careless editor phrases it, and must
        never catch the disclosure itself."""
        from danon.claims import scan_text_for_fabricated_independence
        from danon.trial_design import NESTED_REPLICATES_FLAG

        for bad in (
            "the pooled 27 independent paired replicates",
            "9 iid samples per arm",
            "the replicate pairs are independent",
            "27 replicates that are independent",
            "the 27 pairs, which are independent, extend PFS",
            # the four phrasings the previous rule evaded
            "the 27 replicates can be treated as independent",
            "independent across the 27 pairs",
            "27 replicates, each an independent draw",
            "the 27 replicates as independent draws",
            "the pooled interval rests on 27 independent observations",
        ):
            assert scan_text_for_fabricated_independence(bad), bad
        for good in (
            NESTED_REPLICATES_FLAG,
            "at most 9 nested units and is narrower",
            "the pairs are not independent",
            "27 paired replicates as exchangeable",
            # the honest unit count must NOT be flagged: "independent units" is
            # exactly the phrasing the disclosure exists to publish
            "at most 9 independent units stand behind the pooled interval",
            "27 pairs over at most 9 independent units",
            "the replicates are NESTED rather than independent",
            "The 27 paired replicates are NESTED, not a sample of independent patients",
            # honest statistical prose the shipped tree really carries
            "moves by 0.0219 across independent draws of that split",
            "Each draw is an independent sample of n_patients without replacement",
            "a count presented as an independent sample",
        ):
            assert not scan_text_for_fabricated_independence(good), good

    def test_the_check_fails_if_a_top_level_artifact_note_is_fabricated(
        self, monkeypatch
    ):
        """The artifact's prose is scanned WHOLE: a fabrication in its top-level
        notes -- outside `data` -- must still fail, and with one named reason."""
        import danon.app_artifacts as aa
        import danon.trial_design as td

        payload = {"data": dict(td.trial_design_report())}
        payload["notes"] = ["the 27 independent paired replicates extend PFS"]
        monkeypatch.setitem(aa._BUILT, "trial_design", payload)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "payload presents a count as an independent sample" in res.detail
        assert res.detail.count("payload presents a count as an independent sample") == 1

    def test_the_scan_reports_one_problem_per_fabrication_per_surface(self, tmp_path):
        """One fabrication on one surface is one problem -- the scan does not
        report the same root cause twice."""
        from danon.claims import scan_for_fabricated_independence_claims

        art = tmp_path / "outputs_real" / "app_artifacts"
        art.mkdir(parents=True)
        (art / "trial_design.json").write_text(
            '{"notes": ["the 27 independent paired replicates extend PFS"]}',
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_independence_claims(str(tmp_path))
        assert len(problems) == 1, problems
        assert "trial_design.json" in problems[0]

    def test_the_unit_surface_requirement_survives_a_rewording(self):
        """A legitimate rewording of the view's labels must not fail the check."""
        import pathlib
        from danon.claims import _view_unit_surface_problems

        real = " ".join(
            pathlib.Path("frontend", "src", "components", "TrialDesign.tsx")
            .read_text(encoding="utf-8")
            .split()
        )
        assert _view_unit_surface_problems(real) == []
        reworded = real.replace("pairs \u2192 independent units", "pairs / units")
        reworded = reworded.replace("INDEPENDENT UNITS:", "units:")
        reworded = reworded.replace(
            "nestingCounts([c]).independent_units", "unitsBehind(c).independent_units"
        )
        assert _view_unit_surface_problems(reworded) == []

    def test_the_unit_surface_requirement_fails_when_units_are_stripped(self):
        """The moment a count surface loses its units, the check must fail."""
        import pathlib
        from danon.claims import _view_unit_surface_problems

        real = " ".join(
            pathlib.Path("frontend", "src", "components", "TrialDesign.tsx")
            .read_text(encoding="utf-8")
            .split()
        )
        stripped = real.replace(
            "${live.n_pairs} \u2192 ${pooled.nesting.independent_units}", "String(live.n_pairs)"
        )
        stripped = stripped.replace(
            "${g.n_pairs} \u2192 ${g.nesting.independent_units}", "String(g.n_pairs)"
        )
        stripped = stripped.replace(
            "${c.n_pairs} \u2192 ${nestingCounts([c]).independent_units}", "String(c.n_pairs)"
        )
        assert _view_unit_surface_problems(stripped), "stripping the units must fail"

    def test_the_check_fails_if_recorded_prose_presents_the_count_as_independent(
        self, monkeypatch
    ):
        """The audit's second hole: the artifact's own free prose was neither
        template-pinned nor scanned, so a doctored bullet still passed. It must
        now fail with a named reason."""
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        bullets = list(report["what_this_screen_is_not"])
        bullets[0] = (
            "a sample of independent patients: the 27 independent paired "
            "replicates come from separate patients"
        )
        doctored["data"]["what_this_screen_is_not"] = bullets
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "independent sample" in res.detail

    def test_the_independence_scan_covers_readme_and_artifact_prose(self, tmp_path):
        """The scan must cover the README and the recorded artifacts, not only the
        source tree -- those are where a count is most often quoted verbatim."""
        from danon.claims import scan_for_fabricated_independence_claims

        (tmp_path / "README.md").write_text(
            "the pooled 27 independent paired replicates extend PFS", encoding="utf-8"
        )
        art = tmp_path / "outputs_real" / "app_artifacts"
        art.mkdir(parents=True)
        (art / "trial_design.json").write_text(
            '{"data": {"what_this_screen_is_not": ["9 iid samples per arm"]}}',
            encoding="utf-8",
        )
        problems, scanned = scan_for_fabricated_independence_claims(str(tmp_path))
        joined = " ".join(problems)
        assert "README.md" in joined, problems
        assert "trial_design.json" in joined, problems
        assert scanned >= 2

    def test_the_check_fails_if_the_nesting_disclosure_is_dropped(self, monkeypatch):
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        doctored["data"]["pooled"] = {
            k: v for k, v in report["pooled"].items() if k != "nesting"
        }
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "nesting disclosure" in res.detail

    def test_the_check_fails_if_the_disclosure_contradicts_its_own_rows(
        self, monkeypatch
    ):
        """The audit's hole: a doctored payload claimed 8 independent units
        behind 9 configurations and 7 imaging geometries behind 3 case ids, and
        the check returned supported. It must now re-derive both from the rows."""
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        pooled = dict(report["pooled"])
        nesting = dict(pooled["nesting"])
        # keep the note as the real one, so the FIELD contradiction is what fires
        nesting["independent_units"] = 8
        nesting["n_geometries"] = 7
        pooled["nesting"] = nesting
        doctored["data"]["pooled"] = pooled
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "independent_units" in res.detail
        assert "n_geometries" in res.detail

    def test_the_check_fails_if_a_note_disagrees_with_its_own_counts(self, monkeypatch):
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        pool = dict(report["pooled"])
        nesting = dict(pool["nesting"])
        nesting["note"] = nesting["note"].replace(
            "at most 9 nested units", "at most 8 nested units"
        )
        pool["nesting"] = nesting
        doctored["data"]["pooled"] = pool
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "note" in res.detail

    def test_the_check_fails_if_an_image_pool_miscounts_its_units(self, monkeypatch):
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        groups = [dict(g) for g in report["by_geometry"]]
        nesting = dict(groups[0]["nesting"])
        nesting["independent_units"] = 2  # its 3 configurations cannot be 2 units
        groups[0]["nesting"] = nesting
        doctored["data"]["by_geometry"] = groups
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "image pool" in res.detail

    def test_the_check_fails_if_the_rule_out_becomes_a_licence_claim(self, monkeypatch):
        import danon.app_artifacts as aa
        import danon.trial_design as td

        report = td.trial_design_report()
        doctored = {"data": dict(report)}
        anc = dict(report["anc_screen"])
        anc["is_a_regulatory_boundary"] = True
        doctored["data"]["anc_screen"] = anc
        monkeypatch.setitem(aa._BUILT, "trial_design", doctored)
        res = CHECKS[self.CHECK]()
        assert res.supported is False
        assert "regulatory" in res.detail


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
