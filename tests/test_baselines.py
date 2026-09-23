"""Tests for comparison baselines.

Every module in this package used to contain a method like

    def utcl_score(self):
        return 0.25

which manufactured a baseline constant and therefore manufactured the claimed
improvement over it. These tests pin the replacement contract:

* a baseline must be *produced by the same model* as the optimized score, so the
  only thing that differs between the two arms is the design decision;
* a baseline must be reproducible and independent of call order;
* and a baseline must not be a hardcoded constant, because a constant can never
  be falsified by evidence.

The dual-vector tests additionally guard a saturation bug: the headline metric
of that module was pinned at 1.0000 across the entire therapeutic dose range,
which made every ranking it produced meaningless.
"""
import numpy as np
import pytest

from danon.biology import WILD_TYPE_AAV9, biological_provenance
from danon.cell_simulator import DanonCellSimulator
from danon.dual_vector_moi_optimizer import DualVectorMOIOptimizer
from danon.immunosuppression_protocol import ImmunosuppressionProtocol
from danon.nab_assay_simulator import NAbAssaySimulator
from danon.stoichiometric_calc import StoichiometricCalculator


class TestBaselinesAreDerivedNotAsserted:
    def test_wild_type_reference_is_registered_as_an_assumption(self):
        provenance = biological_provenance()
        for key in WILD_TYPE_AAV9:
            assert key in provenance, f"{key} must appear in the provenance audit"
            assert getattr(provenance[key], "value", provenance[key]) == "assumed"

    def test_stoichiometric_baseline_uses_the_no_decoy_arm(self):
        calc = StoichiometricCalculator()
        # A seronegative patient gains nothing from decoys : decoys exist to soak
        # up pre-existing antibody, and there is none. Baseline == optimized here
        # is the correct answer, and the old hardcoded 0.10 could never say so.
        assert calc.utcl_score(2.0) == pytest.approx(calc.our_best_score(2.0))

    def test_stoichiometric_decoy_helps_at_high_titer(self):
        calc = StoichiometricCalculator()
        for titer in (200.0, 400.0, 1000.0):
            assert calc.our_best_score(titer) > calc.utcl_score(titer)

    def test_stoichiometric_scores_are_reproducible(self):
        calc = StoichiometricCalculator()
        assert calc.utcl_score(200.0) == calc.utcl_score(200.0)
        assert calc.our_best_score(200.0) == calc.our_best_score(200.0)

    def test_nab_baseline_sits_at_the_assay_noise_floor(self):
        assay = NAbAssaySimulator()
        # neutralization_escape is defined relative to the wild-type arm, so the
        # wild-type arm is zero up to independent per-well assay noise.
        assert assay.utcl_score() < 0.05
        assert assay.our_best_score() > 0.30

    def test_nab_scores_are_independent_of_call_order(self):
        """Both arms must be judged on the same donor panel."""
        first_baseline = NAbAssaySimulator()
        first_engineered = NAbAssaySimulator()
        # call in opposite orders on identical fresh instances
        a_baseline = first_baseline.utcl_score()
        first_baseline.our_best_score()
        first_engineered.our_best_score()
        a_baseline_again = first_engineered.utcl_score()
        assert a_baseline == pytest.approx(a_baseline_again)

    def test_immunosuppression_baseline_is_the_do_nothing_arm(self):
        imm = ImmunosuppressionProtocol()
        baseline = imm.utcl_score()
        best = imm.our_best_score()
        assert baseline < best, "a treatment protocol must beat no treatment"
        assert baseline > 0.0

    def test_cell_simulator_baseline_runs_the_ode_model(self):
        sim = DanonCellSimulator()
        baseline = sim.utcl_score()
        engineered = sim.our_best_score()
        assert 0.0 < baseline < 1.0
        assert engineered > baseline


class TestDualVectorMetricIsNotSaturated:
    def test_dual_hit_fraction_is_saturated_and_therefore_useless_for_ranking(self):
        """Regression test for the saturation bug.

        ``dual_hit_fraction`` is ~1.0 for every tropism because at therapeutic
        doses every cell receives many vector copies. Asserting the saturation
        keeps the documentation honest - if someone later makes the metric
        informative, this test should fail and be rewritten deliberately.
        """
        opt = DualVectorMOIOptimizer()
        hits = [opt.profile_dose_range(t, 0.25).expected_dual_hit
                for t in (0.05, 0.25, 0.5, 0.72, 0.9)]
        assert max(hits) - min(hits) < 0.01, "expected a saturated metric"
        assert min(hits) > 0.99

    def test_reconstituted_copy_number_discriminates(self):
        opt = DualVectorMOIOptimizer()
        mus = [opt.profile_dose_range(t, 0.25).expected_reconstituted_copies_per_cell
               for t in (0.05, 0.25, 0.5, 0.72, 0.9)]
        spread = (max(mus) - min(mus)) / min(mus)
        assert spread > 0.05, "the replacement metric must actually vary"
        assert all(mu > 0 for mu in mus), "copy number must be positive, not clipped to zero"

    def test_dual_vector_baseline_is_below_the_optimized_design(self):
        opt = DualVectorMOIOptimizer()
        assert opt.our_best_score() > opt.utcl_score()

    def test_dual_vector_scores_are_reproducible(self):
        opt = DualVectorMOIOptimizer()
        assert opt.utcl_score() == opt.utcl_score()
        assert opt.our_best_score() == opt.our_best_score()


class TestInverseFoldBaselines:
    def test_global_numpy_rng_is_not_polluted(self):
        """``np.random.seed`` inside a design call corrupted callers' streams."""
        from danon.inverse_fold import InverseFoldingEngine

        wt = ("MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
              "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPT"
              "KTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYA") * 4
        wt = wt[:750]

        np.random.seed(1234)
        expected = np.random.random()
        np.random.seed(1234)
        InverseFoldingEngine().design_candidate(0, wt)
        assert np.random.random() == expected

    def test_design_is_reproducible_for_a_candidate_id(self):
        from danon.inverse_fold import InverseFoldingEngine

        wt = ("MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
              "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPT"
              "KTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYA") * 4
        wt = wt[:750]

        a = InverseFoldingEngine().design_candidate(7, wt)
        b = InverseFoldingEngine().design_candidate(7, wt)
        assert a.mutated_sequence == b.mutated_sequence

    def test_edits_are_confined_to_the_target_region_and_change_the_residue(self):
        from danon.inverse_fold import InverseFoldingEngine

        wt = ("MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
              "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPT"
              "KTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYA") * 4
        wt = wt[:750]

        for design in InverseFoldingEngine().stream_designs(wt, 20, seed=0):
            for pos, original, new in design.mutations:
                assert 263 <= pos < 819
                assert original != new

    def test_panel_is_diverse_not_one_design_repeated(self):
        from danon.inverse_fold import InverseFoldingEngine

        wt = ("MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
              "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPT"
              "KTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYA") * 4
        wt = wt[:750]

        panel = InverseFoldingEngine().stream_designs(wt, 30, seed=0)
        assert len({d.mutated_sequence for d in panel}) > 20

    def test_baseline_requires_a_real_sequence(self):
        from danon.inverse_fold import InverseFoldingEngine

        with pytest.raises(ValueError):
            InverseFoldingEngine().stream_designs("", 5)


class TestCodonResultModelIsStrict:
    def test_codon_result_exposes_every_field_it_documents(self):
        """pydantic's default ``extra='ignore'`` silently dropped 7 fields."""
        from danon.codon_elongation import CodonElongationEngine
        from danon.codon_elongation import DEFAULT_LAMP2B_PEPTIDE

        result = CodonElongationEngine(min_tai=0.88).evaluate(protein=DEFAULT_LAMP2B_PEPTIDE)
        for field in ("tai", "tai_before", "tai_after", "min_window_tai",
                      "mean_elongation_rate", "slowest_codon_index", "codon_optimized",
                      "optimized_cds", "harmonized_cds", "min_index_threshold"):
            assert hasattr(result, field), f"{field} missing from CodonHarmonizationResult"

    def test_codon_result_rejects_unknown_fields(self):
        from danon.codon_elongation import CodonHarmonizationResult

        kwargs = dict(
            cds_length_nt=3, n_codons=1, tai=0.9, tai_before=0.9, tai_after=0.9,
            harmonization_score=0.0, harmonization_score_after=0.0,
            velocity_match_improvement=0.0, stall_sites_after=0,
            harmonization_threshold=0.7,
        )
        CodonHarmonizationResult(**kwargs)  # documented fields are fine
        with pytest.raises(Exception):
            CodonHarmonizationResult(**kwargs, typo_field=1.0)
