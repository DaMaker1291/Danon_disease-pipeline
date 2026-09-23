"""Integrity tests for the rebuilt models.

These exist specifically to catch *degenerate* models — the failure mode where a
simulator returns a constant regardless of its inputs and therefore cannot
inform any decision. The stoichiometric decoy calculator previously reported a
free-NAb fraction of exactly zero at every titer and saturated therapeutic
efficacy at 1.0 for all conditions; the tests below fail if that regresses.
"""

from __future__ import annotations

import pytest

from danon.stoichiometric_calc import StoichiometricCalculator
from danon.hla_decoupler import HLADecoupler, resolve_allele


@pytest.fixture
def stoich() -> StoichiometricCalculator:
    return StoichiometricCalculator()


# ---------------------------------------------------------------------------
# Decoy model must respond to its inputs
# ---------------------------------------------------------------------------

TITERS = [1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
RATIOS = [1.0, 10.0, 100.0]


class TestDecoyModelResponsiveness:
    def test_efficacy_decreases_with_titer(self, stoich: StoichiometricCalculator) -> None:
        """More pre-existing antibody must mean fewer active capsids."""
        effs = [
            stoich.simulate_decoy_dose(t, 5e13, 1.0).therapeutic_efficacy_score
            for t in TITERS
        ]
        for a, b in zip(effs, effs[1:]):
            assert b < a, f"efficacy not monotonic decreasing: {effs}"
        # and it must actually span a wide range, not be pinned
        assert max(effs) - min(effs) > 0.5

    def test_efficacy_increases_with_decoy_ratio(
        self, stoich: StoichiometricCalculator
    ) -> None:
        """Adding sacrificial empties must rescue therapeutic capsids."""
        for t in [200.0, 1000.0]:
            effs = [
                stoich.simulate_decoy_dose(t, 5e13, r).therapeutic_efficacy_score
                for r in RATIOS
            ]
            for a, b in zip(effs, effs[1:]):
                assert b > a, f"titer {t}: efficacy not increasing with ratio: {effs}"

    def test_efficacy_is_not_saturated(self, stoich: StoichiometricCalculator) -> None:
        """A model that always returns exactly 1.0 carries no information."""
        effs = {
            (t, r): stoich.simulate_decoy_dose(t, 5e13, r).therapeutic_efficacy_score
            for t in TITERS
            for r in RATIOS
        }
        assert len(set(round(v, 4) for v in effs.values())) > 5
        assert any(0.0 < v < 0.999 for v in effs.values())

    def test_all_outputs_are_bounded(self, stoich: StoichiometricCalculator) -> None:
        for t in TITERS:
            for r in RATIOS:
                p = stoich.simulate_decoy_dose(t, 5e13, r)
                for name, val in (
                    ("free_capsid_fraction", p.free_capsid_fraction),
                    ("bound_capsid_fraction", p.bound_capsid_fraction),
                    ("free_nab_fraction", p.free_nab_fraction),
                    ("complement_deposition_score", p.complement_deposition_score),
                    ("therapeutic_efficacy_score", p.therapeutic_efficacy_score),
                    ("overall_decoy_efficiency", p.overall_decoy_efficiency),
                ):
                    assert 0.0 <= val <= 1.0, f"{name}={val} out of range at t={t}, r={r}"

    def test_complement_risk_falls_with_decoy_ratio(
        self, stoich: StoichiometricCalculator
    ) -> None:
        """Immune-complex load is the RP-A501 safety mechanism; decoys must cut it."""
        high = stoich.simulate_decoy_dose(1000.0, 5e13, 1.0).complement_deposition_score
        low = stoich.simulate_decoy_dose(1000.0, 5e13, 200.0).complement_deposition_score
        assert low < high

    def test_optimal_ratio_tracks_titer(self, stoich: StoichiometricCalculator) -> None:
        """The headline personalization claim: ratio must depend on the patient."""
        serial = [stoich.optimize_ratio(t).optimal_empty_full_ratio for t in TITERS]
        assert len(set(round(r, 2) for r in serial)) > 2, (
            f"optimal ratio is effectively constant across titers: {serial}"
        )
        # and it must rise with titer
        assert serial[-1] > serial[0]

    def test_seronegative_needs_no_heavy_decoys(
        self, stoich: StoichiometricCalculator
    ) -> None:
        r = stoich.optimize_ratio(1.0)
        assert r.optimal_empty_full_ratio < 5.0

    def test_ratio_range_brackets_optimum(self, stoich: StoichiometricCalculator) -> None:
        for t in TITERS:
            r = stoich.optimize_ratio(t)
            lo, hi = r.recommended_ratio_range
            assert lo <= r.optimal_empty_full_ratio <= hi

    def test_high_titer_still_flagged_for_exclusion(
        self, stoich: StoichiometricCalculator
    ) -> None:
        r = stoich.optimize_ratio(2000.0)
        assert "EXCLUDE" in r.clinical_recommendation

    def test_zero_titer_is_benign(self, stoich: StoichiometricCalculator) -> None:
        p = stoich.simulate_decoy_dose(0.0, 5e13, 1.0)
        assert p.therapeutic_efficacy_score > 0.99
        assert p.complement_deposition_score <= 0.16


# ---------------------------------------------------------------------------
# Per-patient HLA screening
# ---------------------------------------------------------------------------


class TestAlleleResolution:
    @pytest.mark.parametrize(
        "allele,locus,supertype",
        [
            ("HLA-A*02:01", "I", "A2"),
            ("A*02", "I", "A2"),
            ("HLA-A*01:01", "I", "A1"),
            ("HLA-B*07:02", "I", "B7"),
            ("HLA-B*44:02", "I", "B44"),
            ("HLA-DRB1*15:01", "II", "DR15"),
            ("HLA-DRB1*04:01", "II", "DR4"),
            ("DRB1*07:01", "II", "DR7"),
        ],
    )
    def test_known_alleles_resolve(self, allele: str, locus: str, supertype: str) -> None:
        a = resolve_allele(allele)
        assert a.locus == locus
        assert a.supertype == supertype
        assert a.resolved

    def test_unsupported_locus_falls_back_visibly(self) -> None:
        a = resolve_allele("HLA-DQB1*06:02")
        assert not a.resolved

    def test_class_i_window_differs_from_class_ii(self) -> None:
        assert resolve_allele("HLA-A*02:01").window < resolve_allele("HLA-DRB1*15:01").window


class TestPatientHlaScreening:
    SEGMENTS = ["PKYVKQNTLKLAT", "YFLKAWTVAQK"]

    def test_default_mode_is_unchanged(self) -> None:
        """Cohort-level callers must still get the DRB1 supertype behaviour."""
        r = HLADecoupler().evaluate(self.SEGMENTS)
        assert r.high_affinity_hits >= 1
        assert not r.decoupled

    def test_patient_mode_reports_alleles(self) -> None:
        e = HLADecoupler(patient_alleles=("HLA-A*02:01", "HLA-DRB1*15:01"))
        r = e.evaluate(self.SEGMENTS)
        assert r.alleles_screened == ["HLA-A*02:01", "HLA-DRB1*15:01"]
        assert not r.unresolved_alleles
        assert r.per_allele_hits

    def test_different_haplotypes_give_different_calls(self) -> None:
        s = self.SEGMENTS
        a = HLADecoupler(patient_alleles=("HLA-A*02:01", "HLA-DRB1*15:01")).evaluate(s)
        b = HLADecoupler(patient_alleles=("HLA-B*07:02", "HLA-DRB1*07:01")).evaluate(s)
        assert (a.dominant_allele, a.high_affinity_hits) != (
            b.dominant_allele,
            b.high_affinity_hits,
        )

    def test_patient_report_marks_specificity(self) -> None:
        rep = HLADecoupler(patient_alleles=("HLA-A*02:01",)).patient_report(self.SEGMENTS)
        assert rep["personalized"] is True
        assert rep["deimmunization_targets"]
        assert rep["dominant_allele"] == "HLA-A*02:01"

    def test_unresolved_allele_blocks_personalized_flag(self) -> None:
        rep = HLADecoupler(patient_alleles=("HLA-DQB1*06:02",)).patient_report(self.SEGMENTS)
        assert rep["personalized"] is False
        assert rep["unresolved_alleles"]

    def test_deimmunization_targets_are_nonamers(self) -> None:
        rep = HLADecoupler(patient_alleles=("HLA-DRB1*15:01",)).patient_report(self.SEGMENTS)
        for t in rep["deimmunization_targets"]:
            assert len(t) == 9
