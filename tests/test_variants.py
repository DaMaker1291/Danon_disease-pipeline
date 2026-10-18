"""The variant channel: contracts, refusals, and the real GLASS record.

Two layers are pinned here. The first is the channel's own contract, which needs
no data at all: a fraction is computed from counts, a pairing needs two specimens
of one patient, and the mapping publishes a provable lower bound while refusing a
point estimate, an upper bound and a k-bin summarisation. The second is the REAL
record in `outputs_real/glass_variants.json`, built by `_glass_variant_cohort.py`
from the public paired GLASS release: when it is present these tests re-derive its
counts from the cached raw records, and when it is absent they skip rather than
pass silently.
"""

from __future__ import annotations

import json
import os

import pytest

from danon.variants import (
    LOWER_BOUND_DERIVATION,
    POINT_FRACTION_REFUSED,
    TWO_BIN_SUMMARY_REFUSED,
    UPPER_BOUND_REFUSED,
    VARIANT_PANEL,
    VariantError,
    VariantRecord,
    Spectrum,
    clone_fraction_lower_bound,
    paired_change,
    paired_changes,
    spectra_from_records,
    summarise_onto_two_bins,
    variant_channel_provenance,
    what_this_channel_cannot_know,
)
from danon.cure import EvidenceStatus
from danon.walkforward import WalkForwardError, clonal_jsd

ARTIFACT = os.path.join("outputs_real", "glass_variants.json")
CACHE = os.path.join("data", "glass_paired", "glass_mutations.json")


def _record(**kw):
    base = dict(
        patient_id="P1",
        sample_id="P1-TP",
        timepoint="Primary",
        gene="MSH6",
        protein_change="R922*",
        chromosome="2",
        start=48010530,
        ref_allele="C",
        alt_allele="T",
        alt_count=53,
        ref_count=84,
        source="test",
    )
    base.update(kw)
    return VariantRecord(**base)


# ---------------------------------------------------------------------------
# The measured layer
# ---------------------------------------------------------------------------


def test_allele_fraction_is_computed_from_counts():
    rec = _record()
    assert rec.depth == 137
    assert rec.allele_fraction == pytest.approx(53 / 137)
    # A deep and a shallow read of the same fraction agree: the fraction is a
    # ratio, so sequencing depth cannot move it.
    assert _record(alt_count=5300, ref_count=8400).allele_fraction == pytest.approx(
        rec.allele_fraction
    )


def test_zero_depth_and_non_integer_counts_are_refused():
    with pytest.raises(VariantError):
        _record(alt_count=0, ref_count=0)
    with pytest.raises(VariantError):
        _record(alt_count=5.0)
    with pytest.raises(VariantError):
        _record(alt_count=True)
    with pytest.raises(VariantError):
        _record(ref_count=-3)
    with pytest.raises(VariantError):
        _record(gene="  ")


def test_a_spectrum_is_one_patient_and_one_specimen():
    with pytest.raises(VariantError):
        Spectrum(
            patient_id="P1",
            sample_id="P1-TP",
            timepoint="Primary",
            records=(_record(patient_id="P2"),),
        )
    with pytest.raises(VariantError):
        Spectrum(
            patient_id="P1",
            sample_id="P1-R1",
            timepoint="Recurrence",
            records=(_record(),),
        )
    # The same locus twice is refused: a spectrum cannot double a variant.
    with pytest.raises(VariantError):
        Spectrum(
            patient_id="P1",
            sample_id="P1-TP",
            timepoint="Primary",
            records=(_record(), _record()),
        ).keyed()


# ---------------------------------------------------------------------------
# The pairing layer: the measured longitudinal quantity
# ---------------------------------------------------------------------------


def _pair():
    primary = Spectrum(
        patient_id="P1", sample_id="P1-TP", timepoint="Primary",
        records=(_record(alt_count=10, ref_count=90),),
    )
    recurrence = Spectrum(
        patient_id="P1", sample_id="P1-R1", timepoint="Recurrence",
        records=(
            _record(sample_id="P1-R1", timepoint="Recurrence", alt_count=60, ref_count=40),
            _record(
                sample_id="P1-R1", timepoint="Recurrence", gene="TP53",
                chromosome="17", start=7577022, alt_count=20, ref_count=80,
            ),
        ),
    )
    return primary, recurrence


def test_paired_change_measures_shared_gained_and_lost():
    primary, recurrence = _pair()
    change = paired_change(primary, recurrence)
    key = "MSH6:2:48010530:C>T"
    assert change.n_shared == 1
    assert change.n_gained == 1 and change.gained == ("TP53:17:7577022:C>T",)
    assert change.n_lost == 0
    assert change.af_primary[key] == pytest.approx(0.10)
    assert change.af_recurrence[key] == pytest.approx(0.60)
    assert change.af_delta[key] == pytest.approx(0.50)
    assert change.rising() == (key,)


def test_a_lost_variant_is_a_lost_variant():
    primary, recurrence = _pair()
    swapped = paired_change(recurrence, primary)
    assert swapped.n_lost == 1
    assert swapped.lost == ("TP53:17:7577022:C>T",)


def test_pairing_needs_two_specimens_of_one_patient():
    primary, recurrence = _pair()
    with pytest.raises(VariantError):
        paired_change(primary, primary)
    other = Spectrum(
        patient_id="P2", sample_id="P2-R1", timepoint="Recurrence",
        records=(_record(patient_id="P2", sample_id="P2-R1"),),
    )
    with pytest.raises(VariantError):
        paired_change(primary, other)


def test_only_primary_to_recurrence_pairs_are_formed():
    """Two primaries are a duplicate, not a longitudinal change."""
    p1 = Spectrum(
        patient_id="P1", sample_id="P1-TP", timepoint="Primary",
        records=(_record(alt_count=10, ref_count=90),),
    )
    p2 = Spectrum(
        patient_id="P1", sample_id="P1-TP2", timepoint="Primary",
        records=(_record(sample_id="P1-TP2", alt_count=40, ref_count=60),),
    )
    r1 = Spectrum(
        patient_id="P1", sample_id="P1-R1", timepoint="Recurrence",
        records=(_record(sample_id="P1-R1", timepoint="Recurrence", alt_count=60, ref_count=40),),
    )
    changes = paired_changes([p1, p2, r1])
    assert {(c.primary, c.recurrence) for c in changes} == {
        ("P1-TP", "P1-R1"),
        ("P1-TP2", "P1-R1"),
    }


def test_spectra_group_records_by_patient_and_specimen():
    spectra = spectra_from_records(
        [_record(), _record(alt_count=11, ref_count=89, start=48010531)]
    )
    assert len(spectra) == 1
    assert len(spectra[0]) == 2
    assert spectra[0].sample_id == "P1-TP"


# ---------------------------------------------------------------------------
# The mapping: one provable bound, three refusals
# ---------------------------------------------------------------------------


def test_the_lower_bound_is_provable_and_the_derivation_is_returned():
    low, derivation = clone_fraction_lower_bound(0.387)
    assert low == pytest.approx(0.387)
    assert "lower bound" in derivation
    assert LOWER_BOUND_DERIVATION in derivation
    # With a declared purity the somatic assumption tightens it, and only ever
    # upward: a bound cannot fall when the measured fraction is divided by p <= 1.
    scaled, why = clone_fraction_lower_bound(0.387, purity=0.8)
    assert scaled == pytest.approx(0.387 / 0.8)
    assert scaled > low and "purity" in why


def test_the_mapping_refuses_a_point_an_upper_bound_and_a_two_bin_summary():
    assert "refused" in POINT_FRACTION_REFUSED
    assert "no copy number" in UPPER_BOUND_REFUSED
    assert "k bins" in TWO_BIN_SUMMARY_REFUSED
    with pytest.raises(VariantError):
        summarise_onto_two_bins(
            Spectrum(patient_id="P1", sample_id="P1-TP", timepoint="Primary")
        )
    # The engine's own scorer refuses the same support, so the refusal is a
    # property of the pair of modules rather than a comment in one of them.
    with pytest.raises(WalkForwardError):
        clonal_jsd(0.7, 0.3, [0.5, 0.3, 0.2])


def test_contradictory_purity_publishes_nothing():
    with pytest.raises(VariantError):
        clone_fraction_lower_bound(0.9, purity=0.5)
    with pytest.raises(VariantError):
        clone_fraction_lower_bound(0.2, purity=0.0)
    with pytest.raises(VariantError):
        clone_fraction_lower_bound(1.5)
    with pytest.raises(VariantError):
        clone_fraction_lower_bound(-0.1)


def test_provenance_names_the_links_that_are_not_measured():
    prov = variant_channel_provenance()
    assert prov["allele_fraction_from_read_counts"] is EvidenceStatus.ESTABLISHED
    assert prov["clone_fraction_lower_bound"] is EvidenceStatus.ESTABLISHED
    assert prov["tumour_purity"] is EvidenceStatus.UNKNOWN
    assert prov["point_clone_fraction"] is EvidenceStatus.PROVEN_FAILED
    assert prov["upper_bound_on_clone_fraction"] is EvidenceStatus.PROVEN_FAILED
    assert prov["two_bin_summarisation"] is EvidenceStatus.PROVEN_FAILED
    assert prov["clinical_benefit"] is EvidenceStatus.UNKNOWN
    assert prov["engine_predicts_patients"] is EvidenceStatus.UNKNOWN
    text = what_this_channel_cannot_know()
    assert "NOT a clonal reconstruction" in text
    assert "cannot say whether a change was caused by therapy" in text


def test_the_panel_is_declared_and_bounded():
    names = [g for g, _ in VARIANT_PANEL]
    assert "MSH6" in names and "MGMT" in names and "TP53" in names
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# The REAL record: re-derived when present, skipped when not built
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="glass_variants.json not built")
class TestTheRealGlassRecord:
    @pytest.fixture(scope="class")
    def record(self):
        return json.load(open(ARTIFACT, encoding="utf-8"))

    def test_the_source_is_the_public_paired_release(self, record):
        src = record["source"]
        assert src["study_id"] == "difg_glass"
        assert src["public_study"] is True
        assert src["panel_is_declared_not_genome_wide"] is True
        assert len(src["panel"]) == len(VARIANT_PANEL)

    def test_every_panel_call_succeeded(self, record):
        assert record["calls"], "no panel call is recorded"
        assert all(c["status"] == 200 for c in record["calls"]), record["calls"]

    def test_the_counts_are_the_records_own_arithmetic(self, record):
        m = record["measured"]
        assert m["n_mutation_records"] == sum(c["n_records"] for c in record["calls"])
        assert (
            m["n_records_with_allele_counts"] + m["n_records_dropped_no_counts_or_zero_depth"]
            == m["n_mutation_records"]
        )
        af = m["allele_fraction"]
        assert af["n"] == m["n_records_with_allele_counts"]
        assert 0.0 <= af["min"] <= af["median"] <= af["max"] <= 1.0

    def test_the_recorded_pairing_is_internally_consistent(self, record):
        pc = record["measured"]["paired_change"]
        assert pc["n_patients_with_a_pair"] > 0
        assert pc["n_pairs"] >= pc["n_patients_with_a_pair"]
        # A pair cannot contribute more shared variants than either specimen
        # carries; the totals are sums over pairs and must be non-negative.
        assert pc["variants_shared"] >= 0
        assert pc["variants_gained"] >= 0 and pc["variants_lost"] >= 0
        example = record["measured"]["worked_example"]
        if example is not None:
            assert example["af_recurrence"] == pytest.approx(
                example["af_primary"] + example["af_delta"]
            )
            assert example["primary"] != example["recurrence"]

    def test_the_record_refuses_the_mappings_it_cannot_make(self, record):
        refused = record["what_is_refused"]
        assert refused["point_clone_fraction"] == POINT_FRACTION_REFUSED
        assert refused["upper_bound_on_clone_fraction"] == UPPER_BOUND_REFUSED
        assert refused["two_bin_summarisation"] == TWO_BIN_SUMMARY_REFUSED
        assert record["provenance"]["point_clone_fraction"] == "proven_failed"
        assert record["provenance"]["engine_predicts_patients"] == "unknown"
        assert "NOT a clonal reconstruction" in record["cannot_know"]
        assert len(record["limitations"]) >= 4

    def test_the_cached_raw_records_reproduce_the_artifact(self, record):
        if not os.path.exists(CACHE):
            pytest.skip("raw cache absent")
        raw = json.load(open(CACHE, encoding="utf-8"))
        refs = [
            VariantRecord(
                patient_id=r.get("patientId", ""),
                sample_id=r["sampleId"],
                timepoint="x",
                gene=r["_gene"],
                protein_change=r.get("proteinChange") or "",
                chromosome=str(r.get("chr") or ""),
                start=int(r.get("startPosition") or 0),
                ref_allele=str(r.get("referenceAllele") or ""),
                alt_allele=str(r.get("variantAllele") or ""),
                alt_count=r["tumorAltCount"],
                ref_count=r["tumorRefCount"],
            )
            for r in raw["records"]
            if isinstance(r.get("tumorAltCount"), int)
            and isinstance(r.get("tumorRefCount"), int)
            and r["tumorAltCount"] + r["tumorRefCount"] > 0
        ]
        assert len(refs) == record["measured"]["n_records_with_allele_counts"]
        assert len({r.patient_id for r in refs}) == record["measured"]["n_patients"]
        assert len({r.sample_id for r in refs}) == record["measured"]["n_samples"]
