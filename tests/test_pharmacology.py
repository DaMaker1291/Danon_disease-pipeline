"""Tests for the real pharmacology intake module.

The recorded ChEMBL payloads in ``data/pharmacology/`` are real responses (166
temozolomide IC50 records, molecule CHEMBL810), so these tests exercise the
parse/normalise/summarise path on real data without touching the network.

Pinned here, because each was a live hazard while writing the module:

* censored records (``relation == '>'``) must not enter the point estimate --
  treating them as exact inflates every summary;
* unlabelled and mass-concentration units must be *counted as unconvertible*
  rather than dropped or guessed;
* ``fetch_*`` must raise rather than substitute a placeholder;
* the EC50 comparison must refuse when the engine declares no concentration
  unit, which it currently does not.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.pharmacology import (
    ASSUMED_TORTUOSITY,
    CACHE_DIR,
    GLIOMA_LINES,
    ActivityRecord,
    Ec50Comparison,
    PharmacologyUnavailable,
    PotencySummary,
    clark_logbb,
    compare_engine_ec50,
    engine_ec50_unit,
    engine_voxel_mm,
    fetch_activities,
    fetch_molecule,
    load_glioma_potency,
    normalize_to_nM,
    parse_activities,
    parse_molecule,
    pharmacology_provenance,
    stokes_einstein_diffusion,
    summarize_potency,
    what_pharmacology_cannot_prove,
)

CACHE_MOLECULE = os.path.join(CACHE_DIR, "temozolomide_molecule.json")
CACHE_ACTIVITIES = os.path.join(CACHE_DIR, "raw_CHEMBL810_IC50.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CACHE_MOLECULE) and os.path.exists(CACHE_ACTIVITIES)),
    reason="recorded ChEMBL cache absent; cannot exercise the real-data path",
)


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------


class TestUnitNormalisation:
    def test_molar_units_scale_to_nM(self):
        assert normalize_to_nM(1.0, "nM") == pytest.approx(1.0)
        assert normalize_to_nM(1.0, "uM") == pytest.approx(1000.0)
        assert normalize_to_nM(1.0, "µM") == pytest.approx(1000.0)
        assert normalize_to_nM(1.0, "M") == pytest.approx(1e9)
        assert normalize_to_nM(1.0, "pM") == pytest.approx(1e-3)

    def test_mass_concentration_requires_molecular_weight(self):
        """ug/mL has no molar meaning without a MW -- refuse, do not assume."""
        assert normalize_to_nM(1.0, "ug.mL-1") is None
        # tmz MW 194.15 -> 1 ug/mL = 1e6/194.15 nM = 5150.7 nM
        assert normalize_to_nM(1.0, "ug.mL-1", mw=194.15) == pytest.approx(
            1e6 / 194.15
        )

    def test_unknown_units_and_missing_values_are_none(self):
        assert normalize_to_nM(1.0, "nonsense") is None
        assert normalize_to_nM(None, "nM") is None
        assert normalize_to_nM(1.0, None) is None

    def test_none_means_could_not_convert_not_zero(self):
        assert normalize_to_nM(5.0, "%inhibition") is None


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------


def _act(value, relation, units="nM", target="U-251"):
    return ActivityRecord(
        activity_id=None, standard_type="IC50", value=value, units=units,
        relation=relation, target=target, year=2020, document_chembl_id=None,
        value_nM=normalize_to_nM(value, units),
    )


class TestCensoring:
    def test_right_censored_records_are_excluded_from_the_point_estimate(self):
        """Pinning the trap: '>' values are lower bounds, not measurements."""
        rows = [
            _act(100.0, "="), _act(200.0, "="), _act(300.0, "="),
            _act(1_000_000.0, ">"), _act(2_000_000.0, ">"),
        ]
        s = summarize_potency(rows)
        assert s.n_exact == 3
        assert s.n_right_censored == 2
        # Median of the exact values only, NOT of all five.
        assert s.median_exact_nM == pytest.approx(200.0)
        assert s.censored_fraction == pytest.approx(2 / 5)
        assert any("right-censored" in n for n in s.notes)

    def test_censored_bound_never_overstates_the_true_median(self):
        """A bound, not an estimate: substituting it can only lower the median.

        This assertion started as ``median_lower_bound <= median_exact``, which
        is false and not what "lower bound" means -- with a censored record added
        the median falls on a different order statistic and the two numbers are
        not comparable. The property that must hold is against the median of the
        *fully observed* data.
        """
        bounds = [100.0, 200.0, 300.0, 10_000.0]
        truths = [100.0, 200.0, 300.0, 25_000.0]
        rows = [_act(bounds[0], "="), _act(bounds[1], "="), _act(bounds[2], "="),
                _act(bounds[3], ">")]
        s = summarize_potency(rows)
        assert s.median_lower_bound_nM == pytest.approx(250.0)
        assert s.median_lower_bound_nM <= float(np.median(truths))
        # The exact-only median is a different number (selection, not error).
        assert s.median_exact_nM == pytest.approx(200.0)
        assert any("biased low by selection" in n for n in s.notes)

    def test_left_censoring_destroys_the_lower_bound_rather_than_lying(self):
        """With left-censored records the substitution is invalid, so it refuses.

        Left-censored true values are <= their bounds, so substituting them
        cannot lower the median and the result is no longer a lower bound.
        """
        rows = [_act(10.0, "<"), _act(100.0, "="), _act(200.0, "=")]
        s = summarize_potency(rows)
        assert s.n_left_censored == 1
        assert s.median_lower_bound_nM is None
        # describe() must not crash or print a bound it does not have.
        assert "lower bound" not in s.describe()

    def test_left_censored_is_counted_separately_from_exact(self):
        rows = [_act(10.0, "<"), _act(100.0, "=")]
        s = summarize_potency(rows)
        assert s.n_left_censored == 1
        assert s.n_exact == 1
        assert s.median_exact_nM == pytest.approx(100.0)

    def test_unconvertible_units_are_counted_not_dropped(self):
        rows = [
            _act(100.0, "="),
            ActivityRecord(
                activity_id=None, standard_type="IC50", value=1.0, units="nonsense",
                relation="=", target="U-251", year=2020, document_chembl_id=None,
                value_nM=None,
            ),
            ActivityRecord(
                activity_id=None, standard_type="IC50", value=None, units=None,
                relation=None, target="U-251", year=2020, document_chembl_id=None,
                value_nM=None,
            ),
        ]
        s = summarize_potency(rows)
        assert s.n_records == 3
        assert s.n_exact == 1
        assert s.n_unconvertible == 2
        assert any("not" in n and "convertible" in n for n in s.notes)

    def test_unrecognised_relation_is_not_assumed_to_be_exact(self):
        """A numeric value with a garbage relation must not be treated as '='."""
        r = _act(50.0, "~")
        s = summarize_potency([r])
        assert s.n_exact == 0
        assert s.n_unconvertible == 1


# ---------------------------------------------------------------------------
# Real recorded data
# ---------------------------------------------------------------------------


class TestRecordedChEMBLData:
    def test_molecule_record_is_real_and_phase_correct(self):
        mol = fetch_molecule("temozolomide", allow_network=False)
        assert mol.chembl_id == "CHEMBL810"
        assert mol.mw == pytest.approx(194.15)
        assert mol.smiles and "n1nnc2" in mol.smiles
        # max_phase is a top-level ChEMBL field; reading it from
        # molecule_properties silently returned None for a phase-4 drug.
        assert mol.max_phase == pytest.approx(4.0)
        assert len(mol.payload_sha256) == 64

    def test_glioma_potency_summary_matches_the_recorded_payload(self):
        mol, s = load_glioma_potency()
        assert mol.chembl_id == "CHEMBL810"
        assert s.n_records == 50
        assert s.n_exact == 31
        assert s.n_right_censored == 19
        assert s.n_unconvertible == 0
        assert s.most_potent_exact_nM == pytest.approx(6426.0)
        assert s.median_exact_nM == pytest.approx(134_970.0)
        assert s.fold_spread > 400
        assert set(s.targets) <= set(GLIOMA_LINES)
        assert "U-87 MG" in s.targets

    def test_widening_the_target_filter_strictly_adds_records(self):
        mol = fetch_molecule("temozolomide", allow_network=False)
        raw = fetch_activities(mol.chembl_id, "IC50", allow_network=False)
        acts = parse_activities(raw, mw=mol.mw)
        gli = summarize_potency(acts, targets=GLIOMA_LINES)
        allt = summarize_potency(acts)
        assert gli.n_records < allt.n_records
        assert allt.n_records == 166
        # Units are heterogeneous across the full set -- that is the point.
        assert allt.n_unconvertible > 0

    def test_selection_by_target_is_case_insensitive(self):
        mol = fetch_molecule("temozolomide", allow_network=False)
        raw = fetch_activities(mol.chembl_id, "IC50", allow_network=False)
        acts = parse_activities(raw, mw=mol.mw)
        a = summarize_potency(acts, targets=("u-251",))
        b = summarize_potency(acts, targets=("U-251",))
        assert a.n_records == b.n_records > 0

    def test_parse_is_deterministic(self):
        mol = fetch_molecule("temozolomide", allow_network=False)
        raw = fetch_activities(mol.chembl_id, "IC50", allow_network=False)
        first = summarize_potency(parse_activities(raw, mw=mol.mw))
        second = summarize_potency(parse_activities(raw, mw=mol.mw))
        assert first.median_exact_nM == second.median_exact_nM
        assert first.exact_nM == second.exact_nM


# ---------------------------------------------------------------------------
# Refusal semantics
# ---------------------------------------------------------------------------


class TestRefusesInsteadOfInventing:
    def test_missing_cache_offline_raises(self, tmp_path):
        with pytest.raises(PharmacologyUnavailable):
            fetch_molecule("temozolomide", cache_dir=str(tmp_path), allow_network=False)

    def test_missing_activity_cache_offline_raises(self, tmp_path):
        with pytest.raises(PharmacologyUnavailable):
            fetch_activities("CHEMBL810", "IC50", cache_dir=str(tmp_path),
                             allow_network=False)

    def test_error_message_names_the_path_and_the_refusal(self, tmp_path):
        with pytest.raises(PharmacologyUnavailable) as exc:
            fetch_activities("CHEMBL810", "IC50", cache_dir=str(tmp_path),
                             allow_network=False)
        msg = str(exc.value)
        assert "Refusing to invent" in msg
        assert "CHEMBL810" in msg

    def test_corrupt_cache_is_treated_as_absent_not_used(self, tmp_path):
        p = tmp_path / "temozolomide_molecule.json"
        p.write_text("{not json at all")
        with pytest.raises(PharmacologyUnavailable):
            fetch_molecule("temozolomide", cache_dir=str(tmp_path), allow_network=False)

    def test_network_failure_falls_back_to_cache_when_present(self):
        mol = fetch_molecule("temozolomide", allow_network=True,
                             fetcher=lambda url: (_ for _ in ()).throw(OSError("down")))
        assert mol.chembl_id == "CHEMBL810"


# ---------------------------------------------------------------------------
# Derived transport
# ---------------------------------------------------------------------------


class TestDerivedTransport:
    def test_stokes_einstein_matches_hand_calculation(self):
        d = stokes_einstein_diffusion(194.15)
        # r = (3*MW/(4*pi*rho*N_A))^(1/3) with MW in kg/mol
        assert d.radius_nm == pytest.approx(0.4256, abs=2e-3)
        assert d.d_free_water_mm2_per_day == pytest.approx(66.8, abs=1.0)

    def test_heavier_molecules_diffuse_more_slowly(self):
        small = stokes_einstein_diffusion(194.15)
        large = stokes_einstein_diffusion(10_000.0)
        assert large.d_free_water_mm2_per_day < small.d_free_water_mm2_per_day

    def test_tortuosity_reduces_effective_diffusion_by_lambda_squared(self):
        d = stokes_einstein_diffusion(194.15)
        assert d.d_tissue_mm2_per_day == pytest.approx(
            d.d_free_water_mm2_per_day / ASSUMED_TORTUOSITY ** 2
        )
        assert d.d_tissue_mm2_per_day < d.d_free_water_mm2_per_day

    def test_engine_unit_conversion_uses_the_repo_voxel_convention(self):
        d = stokes_einstein_diffusion(194.15)
        vox = engine_voxel_mm()
        assert vox == pytest.approx(11.0 / 3.0)
        assert d.in_engine_units(vox) == pytest.approx(
            d.d_tissue_mm2_per_day / vox ** 2
        )

    def test_authored_engine_diffusion_is_below_the_derived_estimate(self):
        """The finding, not an assertion of correctness.

        ``evosteer.DEFAULT_DRUGS.diffusion`` is 0.35 voxel^2/day (tagged
        UNKNOWN). Stokes-Einstein with a single tortuosity factor gives ~1.9.
        The authored constant being *lower* means the engine assumes drug
        penetrates tissue ~5x more slowly than free transport allows, which is
        conservative for sanctuary penetration. It is still an authored
        constant, and this test only records the gap so a future edit cannot
        silently widen or close it without notice.
        """
        from danon.evosteer import DEFAULT_DRUGS

        d = stokes_einstein_diffusion(194.15)
        derived = d.in_engine_units(engine_voxel_mm())
        assert derived == pytest.approx(1.94, abs=0.05)
        assert DEFAULT_DRUGS.diffusion[0] < derived
        assert 4.0 < derived / DEFAULT_DRUGS.diffusion[0] < 7.0

    def test_zero_or_negative_mw_is_rejected(self):
        with pytest.raises(ValueError):
            stokes_einstein_diffusion(0.0)
        with pytest.raises(ValueError):
            stokes_einstein_diffusion(-5.0)


class TestLogBB:
    def test_clark_value_for_temozolomide(self):
        bb = clark_logbb(alogp=-2.08, psa=108.17)
        assert bb.logbb == pytest.approx(-1.778, abs=0.001)

    def test_higher_psa_lowers_predicted_penetration(self):
        assert clark_logbb(-2.08, 108.17).logbb < clark_logbb(-2.08, 20.0).logbb

    def test_citation_and_counterexample_caveat_travel_with_the_number(self):
        bb = clark_logbb(-2.08, 108.17)
        assert "Clark" in bb.citation
        assert "counterexample" in bb.caveat
        assert "temozolomide" in bb.caveat.lower()


# ---------------------------------------------------------------------------
# EC50 comparison -- and its refusal
# ---------------------------------------------------------------------------


class TestEc50Comparison:
    def test_engine_declares_no_concentration_unit_so_comparison_refuses(self):
        """The real defect: EC50 has no unit, so no ratio to a molar IC50 exists."""
        _, s = load_glioma_potency()
        assert engine_ec50_unit() is None
        cmp = compare_engine_ec50(s, 0.30, engine_ec50_unit())
        assert isinstance(cmp, Ec50Comparison)
        assert cmp.comparable is False
        assert cmp.ratio_to_median is None
        assert "no declared concentration unit" in cmp.reason
        assert "REFUSED" in cmp.describe()

    def test_comparison_runs_once_a_unit_is_declared(self):
        _, s = load_glioma_potency()
        cmp = compare_engine_ec50(s, 0.30, "uM")
        assert cmp.comparable is True
        assert cmp.engine_ec50_nM == pytest.approx(300.0)
        # The engine would be assuming ~450x more potency than the measured median.
        assert cmp.ratio_to_median == pytest.approx(134_970.0 / 300.0, rel=1e-3)
        assert "engine EC50" in cmp.describe()

    def test_empty_summary_refuses(self):
        cmp = compare_engine_ec50(PotencySummary(standard_type="IC50", n_records=0),
                                  0.30, "uM")
        assert cmp.comparable is False
        assert "no exact measured potency" in cmp.reason

    def test_mass_concentration_unit_without_mw_refuses(self):
        _, s = load_glioma_potency()
        cmp = compare_engine_ec50(s, 1.0, "ug.mL-1", mw=None)
        assert cmp.comparable is False
        assert "molecular weight" in cmp.reason


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_every_quantity_is_typed(self):
        prov = pharmacology_provenance()
        assert prov["chembl_activity_records"].name == "ESTABLISHED"
        assert prov["stokes_einstein_free_diffusion"].name == "UNKNOWN"
        assert prov["engine_ec50_concentration_unit"].name == "UNKNOWN"

    def test_limits_mention_heterogeneous_assays_and_no_validation_claim(self):
        joined = " ".join(what_pharmacology_cannot_prove()).lower()
        assert "different assays" in joined
        assert "not evidence that the scheduling model is right" in joined
        assert "in-vitro ic50" in joined or "in-vitro" in joined
