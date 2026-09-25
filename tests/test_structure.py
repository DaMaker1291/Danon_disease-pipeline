"""Tests for the real AAV9 capsid structure intake.

These tests are the reason the structural layer can be trusted: they re-derive
the numbers from the deposited PDB file itself, so a fabricated constant cannot
reappear without failing here. The counts below (60 operators, 20 three-fold
elements on 10 axis lines, 31,080 C-alpha atoms, 143.7 A outer radius) are
properties of PDB 3UX1, not choices made in this repository.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest

from danon.vector_design.aav_generator import WILD_TYPE_AAV9_CAPSID, VP1_FRAME_OFFSET, CAPSID_STRUCTURAL_INTERFACES
from danon.vector_design.structure import (
    AAV9_STRUCTURE_ID,
    EXPECTED_SHA256,
    MISLABELED_STRUCTURE_ACTUAL,
    MISLABELED_STRUCTURE_ID,
    VP1_OFFSET,
    VP1_POSITION_OFFSET,
    VR_REGION_WINDOWS,
    CapsidAssembly,
    axis_lines,
    classify_symmetry_operators,
    ensure_structure_file,
    file_sha256,
    load_aav9_capsid,
    parse_ca_residues,
    parse_symmetry_operators,
    structure_provenance,
    what_this_structure_cannot_tell_you,
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@pytest.fixture(scope="module")
def capsid() -> CapsidAssembly:
    return load_aav9_capsid()


@pytest.fixture(scope="module")
def pdb_path() -> str:
    path, _ = ensure_structure_file()
    return path


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------


class TestDepositedFile:
    def test_cached_file_matches_the_pinned_digest(self, pdb_path):
        assert file_sha256(pdb_path) == EXPECTED_SHA256

    def test_chain_a_covers_vp1_219_to_736(self, capsid):
        assert capsid.residues[0].resseq == 219
        assert capsid.residues[-1].resseq == 736
        assert len(capsid.residues) == 518

    def test_refuses_a_file_that_is_not_this_structure(self, pdb_path, tmp_path):
        """A swapped file must not silently produce numbers."""
        tampered = tmp_path / f"{AAV9_STRUCTURE_ID}.pdb"
        with open(pdb_path) as src, open(tampered, "w") as dst:
            for line in src:
                dst.write("REMARK swapped\n" if line.startswith("REMARK 350 BIOMOLECULE") else line)
        with pytest.raises(ValueError, match="SHA-256"):
            load_aav9_capsid(str(tampered), directory=str(tmp_path))

    def test_missing_file_without_download_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                ensure_structure_file(directory=d, allow_download=False)

    def test_parsers_reject_garbage(self, tmp_path):
        bad = tmp_path / "empty.pdb"
        bad.write_text("HEADER nothing here\n")
        with pytest.raises(ValueError):
            parse_ca_residues(str(bad))
        with pytest.raises(ValueError):
            parse_symmetry_operators(str(bad))


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


class TestSymmetry:
    def test_60_operators_with_icosahedral_multiplicities(self, capsid):
        counts: dict[int, int] = {}
        for op in capsid.classified:
            counts[op.order] = counts.get(op.order, 0) + 1
        assert counts == {1: 1, 2: 15, 3: 20, 5: 24}

    def test_axis_line_counts_are_the_real_ones(self, capsid):
        assert len(capsid.three_fold_axes) == 10
        assert len(capsid.five_fold_axes) == 6
        assert len(capsid.two_fold_axes) == 15

    def test_20_protrusions_on_10_lines(self, capsid):
        assert len(capsid.protrusion_axes()) == 20
        lines = capsid.three_fold_axes
        # every operator axis coincides with one of the 10 lines
        for axis in capsid.protrusion_axes():
            assert max(abs(float(axis @ ln)) for ln in lines) == pytest.approx(1.0, abs=1e-4)

    def test_axes_are_unit_vectors(self, capsid):
        for ax in capsid.three_fold_axes + capsid.five_fold_axes + capsid.two_fold_axes:
            assert np.linalg.norm(ax) == pytest.approx(1.0, abs=1e-9)

    def test_malformed_operator_set_is_rejected(self, capsid):
        dropped = capsid.operators[:-1]
        with pytest.raises(ValueError, match="icosahedral"):
            CapsidAssembly(capsid.residues, dropped)

    def test_non_icosahedral_angle_is_rejected(self, capsid):
        ops = capsid.operators.copy()
        rot = np.eye(4)
        theta = np.radians(30.0)
        rot[0, 0] = np.cos(theta)
        rot[0, 1] = -np.sin(theta)
        rot[1, 0] = np.sin(theta)
        rot[1, 1] = np.cos(theta)
        ops[5] = rot
        with pytest.raises(ValueError, match="icosahedral"):
            classify_symmetry_operators(ops)


# ---------------------------------------------------------------------------
# Assembly geometry
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_atom_count_is_subunits_times_residues(self, capsid):
        assert capsid.atom_count == 60 * 518

    def test_radii_match_the_deposit(self, capsid):
        # AAV9's T=1 shell: ~86 A inner edge of the resolved trace, 143.7 A out.
        assert capsid.inner_radius == pytest.approx(85.5, abs=0.5)
        assert capsid.mean_radius == pytest.approx(109.0, abs=0.5)
        assert capsid.outer_radius == pytest.approx(143.7, abs=0.5)

    def test_every_subunit_is_the_same_distance_profile(self, capsid):
        """The 60 copies are related by symmetry, so their radii must coincide."""
        per_subunit = capsid.assembly.reshape(60, 518, 3)
        r = np.linalg.norm(per_subunit - capsid.center, axis=2)
        assert np.allclose(r, r[0], atol=1e-6)

    def test_exposure_proxy_is_a_normalised_radius(self, capsid):
        e = capsid.exposure_proxy()
        assert e.min() > 0.0 and e.max() == pytest.approx(1.0)
        assert len(e) == len(capsid.residues)

    def test_contact_numbers_are_real_structural_counts(self, capsid):
        cn = capsid.contact_numbers()
        assert cn.min() > 0                      # every residue has neighbours
        assert cn.max() <= 59 * 518              # cannot exceed the assembly size
        # buried residues must have more neighbours than exposed ones
        e = capsid.exposure_proxy()
        buried = cn[e < np.quantile(e, 0.25)].mean()
        exposed = cn[e > np.quantile(e, 0.75)].mean()
        assert buried > exposed


# ---------------------------------------------------------------------------
# The design windows
# ---------------------------------------------------------------------------


class TestDesignWindows:
    def test_vr_iv_contains_the_real_loop_sequence(self, capsid):
        seq = "".join(THREE_TO_ONE[r.aa] for r in capsid.region_residues("VR_IV"))
        # AAV9 VR-IV carries the 7-mer insertion and the SGQNQQTLKFS motif
        assert "GQNQQTLKFS" in seq
        assert "KTINGSGQNQQ" in seq

    def test_vr_viii_contains_the_aav9_9mer_insertion(self, capsid):
        seq = "".join(THREE_TO_ONE[r.aa] for r in capsid.region_residues("VR_VIII"))
        assert "ATESYGQVATNHQSAQAQAQTG" in seq

    def test_windows_reach_the_capsid_surface(self, capsid):
        """The fabricated table put VR-IV ~38 A out; the real loop is the surface."""
        for region in VR_REGION_WINDOWS:
            radii = [float(np.linalg.norm(r.coord - capsid.center)) for r in capsid.region_residues(region)]
            assert max(radii) > 0.9 * capsid.outer_radius

    def test_vr_iv_window_owns_the_outermost_atom(self, capsid):
        outer = np.linalg.norm(capsid.ca - capsid.center, axis=1).max()
        iv = [float(np.linalg.norm(r.coord - capsid.center)) for r in capsid.region_residues("VR_IV")]
        assert max(iv) == pytest.approx(outer, abs=0.5)

    def test_coordinate_table_is_in_the_design_frame(self, capsid):
        """Table keys are 1-based design positions: sequence[position - 1].

        Consumers index `sequence[position - 1]`, so a table keyed in the
        0-based index frame silently edited the residue one position before
        the one it labelled. This test pins the frame the callers actually use.
        """
        for region, (lo, hi) in VR_REGION_WINDOWS.items():
            table = capsid.region_coordinate_table(region)
            assert min(table) == lo - VP1_POSITION_OFFSET
            assert max(table) == hi - VP1_POSITION_OFFSET
            # the residue a caller edits is the residue the key claims
            for pos in table:
                residue = next(
                    r for r in capsid.region_residues(region) if r.design_position == pos
                )
                assert residue.vp1_position == pos + VP1_POSITION_OFFSET
                assert WILD_TYPE_AAV9_CAPSID[pos - 1] == THREE_TO_ONE[residue.aa]

    def test_unknown_region_raises(self, capsid):
        with pytest.raises(KeyError):
            capsid.region_residues("VR_XII")


# ---------------------------------------------------------------------------
# The wild-type sequence must be the deposited protein
# ---------------------------------------------------------------------------


class TestWildTypeSequence:
    def test_wild_type_sequence_matches_deposit(self, capsid):
        """The repository's wild type is VP1 263-736 of PDB 3UX1, re-derived here.

        This is the test that keeps the correction in place. The constant it
        guards was previously a 557-residue string that shared no 5-residue run
        with real AAV9 and contained none of the serotype's conserved motifs.
        """
        derived = "".join(
            THREE_TO_ONE[r.aa] for r in capsid.residues if r.resseq >= VP1_FRAME_OFFSET
        )
        assert derived == WILD_TYPE_AAV9_CAPSID

    def test_wild_type_carries_the_conserved_aav_motifs(self):
        for motif in ("WGYFDFNRF", "FNIQVKEV", "FMIPQYGYL", "LYYLSK", "SGQNQQTLKFS"):
            assert motif in WILD_TYPE_AAV9_CAPSID, motif

    def test_stored_frame_equals_vp1_minus_263(self, capsid):
        assert VP1_FRAME_OFFSET == VP1_OFFSET == 263
        first = capsid.residues[0]
        assert first.resseq - VP1_FRAME_OFFSET + VP1_FRAME_OFFSET == first.vp1_position

    def test_structural_interfaces_index_the_sequence_in_frame(self, capsid):
        """Interface ranges must land inside the real sequence, on the right residues."""
        assert len(WILD_TYPE_AAV9_CAPSID) == 474
        for name, idx in CAPSID_STRUCTURAL_INTERFACES.items():
            assert idx, name
            assert min(idx) >= 0
            assert max(idx) < len(WILD_TYPE_AAV9_CAPSID), name
        vr8 = CAPSID_STRUCTURAL_INTERFACES["spike_region_VR_VIII"]
        assert "ATESYGQVATNHQSAQAQAQTG" in WILD_TYPE_AAV9_CAPSID[min(vr8):max(vr8) + 1]

    def test_the_old_id_is_recorded_not_erased(self):
        assert MISLABELED_STRUCTURE_ID == "3J1S"
        assert "serotype 2" in MISLABELED_STRUCTURE_ACTUAL


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_coordinates_are_established_and_proxies_are_not(self):
        prov = structure_provenance()
        assert prov["ca_coordinates"].value == "established"
        assert prov["exposure_proxy"].value == "unknown"
        assert prov["region_boundaries"].value == "unknown"

    def test_limits_name_the_calpha_model_and_the_proxy(self):
        text = " ".join(what_this_structure_cannot_tell_you()).lower()
        assert "c-alpha" in text
        assert "not solvent-accessible surface area" in text
        assert "engineered variants" in text
