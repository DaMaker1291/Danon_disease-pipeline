"""Tests for the longitudinal MRI availability-manifest intake.

The numbers pinned here are the committed CFB-GBM manifest's own counts. They
exist so the census cannot drift into prose, and so the two intake paths (the
file loader and the text census used by the API and the browser) are forced to
agree with each other rather than with a memory of each other.

The last test is a cross-language contract: the browser mirror in
``frontend/src/utils/availability.ts`` must declare the same reference sequence
and the same interval floor as this module, because the deployed app is static
and cannot call the Python census.
"""

from __future__ import annotations

import os
import re

import pytest

from danon.availability import (
    CFB_GBM_MANIFEST,
    MIN_PAIR_INTERVAL_WEEKS,
    REFERENCE_SEQUENCE,
    SEQUENCE_COLUMNS,
    AvailabilityRow,
    census_from_text,
    load_mri_availability_manifest,
    looks_like_availability_manifest,
    mri_availability_census,
)

TS_MIRROR = "frontend/src/utils/availability.ts"

pytestmark = pytest.mark.skipif(
    not os.path.exists(CFB_GBM_MANIFEST),
    reason="committed CFB-GBM availability manifest absent",
)


@pytest.fixture(scope="module")
def census():
    return mri_availability_census()


class TestCommittedCensus:
    """The measured cohort, as counts."""

    def test_cohort_shape(self, census):
        assert census.n_rows == 554
        assert census.n_patients == 264
        assert census.n_timepoints_by_label == {"t0": 264, "t1": 170, "t2": 120}
        assert census.n_patients_multi_timepoint == 170

    def test_usable_pairs_and_the_confound_split(self, census):
        assert census.sequence == "t1gd"
        assert census.min_interval_weeks == 4
        assert census.n_usable_pairs == 393
        assert census.n_patients_with_usable_pair == 168
        # same-machine vs cross-machine: a pair spanning two scanners is not a
        # clean follow-up, so the split is reported rather than summed away
        assert census.n_same_machine_pairs == 356
        assert census.n_cross_machine_pairs == 37
        assert census.n_same_machine_pairs + census.n_cross_machine_pairs == 393
        assert census.n_patients_with_cross_machine_pair == 20
        assert census.n_patients_only_same_machine == 148

    def test_intervals_and_scanners(self, census):
        assert census.interval_weeks_min == 4
        assert census.interval_weeks_median == 16.0
        assert census.interval_weeks_max == 55
        assert census.n_machines == {"AERA (1.5T)": 510, "VIDA (3T)": 44}

    def test_per_sequence_coverage(self, census):
        assert census.n_patients_per_sequence == {
            "t1gd": 393, "t1eg": 220, "t1tse": 4, "t2star": 287,
            "t2tse": 3, "flair": 386, "adc": 173,
        }

    def test_provenance_names_the_doi(self, census):
        assert "10.7937/v9pn-2f72" in census.source
        assert "Cancer Imaging Archive" in census.source

    def test_it_says_it_is_not_a_measurement(self, census):
        """A count of what exists must not read as a measurement of usability."""
        joined = " ".join(census.notes)
        assert "not downloaded" in joined or "not usable" in joined
        assert "confound" in joined

    def test_the_interval_floor_is_nearly_inert_below_four_weeks(self):
        """Measured, not assumed: the floor excludes exactly one pair.

        The first version of this test asserted that lowering the floor to 1
        week would recover all 170 multi-timepoint patients. It does not, and
        the reason matters: the 2 patients without a usable pair are missing
        the reference sequence at a second timepoint, not a long enough
        interval. The floor is a declared parameter whose effect happens to be
        one pair here, so it is reported rather than credited.
        """
        pairs_loose = mri_availability_census(min_interval_weeks=1).n_usable_pairs
        strict = mri_availability_census(min_interval_weeks=4)
        assert pairs_loose == 394
        assert strict.n_usable_pairs == 393  # one pair has a 2-week interval
        # the floor changes no patient at all at this threshold
        assert strict.n_patients_with_usable_pair == 168
        assert mri_availability_census(min_interval_weeks=1).n_patients_with_usable_pair == 168
        # ...and 168 is a SEQUENCE limit, not an interval limit
        assert strict.n_patients_multi_timepoint == 170

    def test_the_count_is_sensitive_to_the_floor_above_eight_weeks(self):
        """Where the criterion does bite, so the declared value is not cosmetic."""
        assert mri_availability_census(min_interval_weeks=8).n_patients_with_usable_pair == 166
        assert mri_availability_census(min_interval_weeks=16).n_patients_with_usable_pair == 132
        assert mri_availability_census(min_interval_weeks=24).n_patients_with_usable_pair == 99
        assert mri_availability_census(min_interval_weeks=52).n_patients_with_usable_pair == 1


class TestIntakePathsAgree:
    """Both intake paths must produce the same census for the same bytes."""

    def test_text_census_matches_file_census(self, census):
        with open(CFB_GBM_MANIFEST, encoding="utf-8-sig") as fh:
            text = fh.read()
        assert looks_like_availability_manifest(text)
        from_text = census_from_text(text)
        assert from_text.to_json_dict() != census.to_json_dict()  # source differs
        a = from_text.to_json_dict()
        b = census.to_json_dict()
        a.pop("source")
        b.pop("source")
        assert a == b

    def test_detection_is_content_based_not_extension_based(self):
        header = "id_patient\ttemporality\ttime_diff_t0 (weeks)\tt1gd\n"
        assert looks_like_availability_manifest(header + "1\tt0\t0\t1\n")
        assert not looks_like_availability_manifest("hello world\n")
        assert not looks_like_availability_manifest("a,b,c\n1,2,3\n")
        assert not looks_like_availability_manifest("id_patient\tt1gd\n1\t1\n")


class TestRefusals:
    """A census computed from a file we had to guess at is worse than none."""

    def test_missing_required_column_is_refused(self):
        with pytest.raises(ValueError, match="missing required column"):
            census_from_text("id_patient\tt1gd\n1\t1\n")

    def test_non_numeric_offset_is_refused_with_the_line(self):
        text = "id_patient\ttemporality\ttime_diff_t0 (weeks)\tt1gd\n1\tt0\tweeks\t1\n"
        with pytest.raises(ValueError, match="not numeric"):
            census_from_text(text)

    def test_duplicate_patient_timepoint_is_refused(self, tmp_path):
        p = tmp_path / "dup.tsv"
        p.write_text(
            "id_patient\ttemporality\ttime_diff_t0 (weeks)\tt1gd\n"
            "1\tt0\t0\t1\n1\tt0\t0\t1\n"
        )
        with pytest.raises(ValueError, match="duplicate row"):
            load_mri_availability_manifest(str(p))

    def test_empty_file_is_refused(self, tmp_path):
        p = tmp_path / "empty.tsv"
        p.write_text("")
        with pytest.raises(ValueError, match="no rows|empty"):
            load_mri_availability_manifest(str(p))

    def test_absent_file_is_refused_not_invented(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="will not invent"):
            load_mri_availability_manifest(str(tmp_path / "nope.tsv"))

    def test_header_only_is_refused(self):
        with pytest.raises(ValueError, match="no usable rows"):
            census_from_text("id_patient\ttemporality\ttime_diff_t0 (weeks)\tt1gd\n")


class TestCensusEdges:
    """Counting rules on hand-built rows, where the answer is knowable."""

    @staticmethod
    def _row(pid, tp, weeks, **seqs):
        flags = {c: False for c in SEQUENCE_COLUMNS}
        flags.update(seqs)
        return AvailabilityRow(
            patient_id=pid, timepoint=tp, offset_weeks=weeks,
            sequences=flags, machine="M1",
        )

    def test_a_single_timepoint_is_not_a_pair(self):
        rows = [self._row("p", "t0", 0, t1gd=True)]
        c = mri_availability_census(rows)
        assert c.n_patients_with_usable_pair == 0
        assert c.n_usable_pairs == 0
        assert c.interval_weeks_min is None

    def test_a_gap_below_the_threshold_is_not_a_pair(self):
        rows = [self._row("p", "t0", 0, t1gd=True), self._row("p", "t1", 3, t1gd=True)]
        assert mri_availability_census(rows).n_usable_pairs == 0
        assert mri_availability_census(rows, min_interval_weeks=3).n_usable_pairs == 1

    def test_a_missing_sequence_on_one_end_kills_the_pair(self):
        rows = [self._row("p", "t0", 0, t1gd=True), self._row("p", "t1", 8, t1gd=False)]
        assert mri_availability_census(rows).n_usable_pairs == 0

    def test_three_timepoints_give_three_pairs(self):
        rows = [
            self._row("p", "t0", 0, t1gd=True),
            self._row("p", "t1", 8, t1gd=True),
            self._row("p", "t2", 26, t1gd=True),
        ]
        c = mri_availability_census(rows)
        assert c.n_usable_pairs == 3
        assert c.n_patients_with_usable_pair == 1
        assert c.n_patients_multi_timepoint == 1

    def test_an_unknown_machine_is_not_assumed_equal(self):
        """Blank machine is not evidence of the same scanner."""
        rows = [
            AvailabilityRow("p", "t0", 0, {c: c == "t1gd" for c in SEQUENCE_COLUMNS}, ""),
            AvailabilityRow("p", "t1", 8, {c: c == "t1gd" for c in SEQUENCE_COLUMNS}, ""),
        ]
        c = mri_availability_census(rows)
        assert c.n_same_machine_pairs == 0
        assert c.n_cross_machine_pairs == 1


@pytest.mark.skipif(
    not os.path.exists(TS_MIRROR), reason="browser mirror absent"
)
def test_browser_mirror_declares_the_same_rules():
    """The static app cannot call the Python census, so the rules must match.

    Reads the TypeScript source for its declared constants and compares them
    with this module's. Cheap, and it catches the failure that matters: the two
    implementations quietly disagreeing about what counts as a pair.
    """
    with open(TS_MIRROR, encoding="utf-8") as fh:
        src = fh.read()
    ref = re.search(r"REFERENCE_SEQUENCE\s*=\s*'([^']+)'", src)
    floor = re.search(r"MIN_INTERVAL_WEEKS\s*=\s*(\d+)", src)
    assert ref and floor, "mirror no longer declares its rules as constants"
    assert ref.group(1) == REFERENCE_SEQUENCE
    assert int(floor.group(1)) == MIN_PAIR_INTERVAL_WEEKS
    # every sequence column the Python census scores must appear in the mirror
    for col in SEQUENCE_COLUMNS:
        assert f"'{col}'" in src, f"mirror omits sequence column {col}"
