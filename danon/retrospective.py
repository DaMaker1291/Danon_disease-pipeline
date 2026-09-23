"""
External check against REAL clinical outcomes (TCGA-GBM, open access).

WHAT THIS MODULE IS
-------------------
It reconstructs two observed cohorts from the GDC open-access clinical
export for project TCGA-GBM -- recorded progression (`follow_ups.
days_to_progression` with `progression_or_recurrence == "Yes"`) and overall
survival (`demographic.days_to_death` / `vital_status`) -- using the repo's
own Kaplan-Meier estimator, and then:

  1. ANCHOR CHECK: the reconstructed cohort median OS must land in the
     published glioblastoma standard-of-care band (Stupp-era ~14.6 months;
     TCGA-GBM cohort reports ~12-15 months). This validates the DATA
     HANDLING -- if the ingestion is wrong, the reconstructed median is
     wrong, and this catches it. It says nothing about the model.

  2. PLAUSIBILITY CHECK: the model's MTD arm (its standard-of-care proxy)
     is compared against the OBSERVED progression curve at fixed time
     points. A model whose control arm progresses orders of magnitude
     faster or slower than real patients is not describing this disease;
     this is the weakest possible external check, and it is labelled as
     such.

WHAT THIS MODULE IS NOT
-----------------------
Not validation. There is no patient-level matching (BraTS and GDC are
anonymised with different identifiers, so no case in the imaging arm can be
tied to a case in the clinical arm), treatment is heterogeneous and only
partly recorded, TCGA's dated progression field is sparsely populated
(267 of 617 cases), a large share of cases are censored with short
follow-up (which biases any KM median), and the model's endpoint is a
RANO-fraction burden threshold, not a radiological response read. All of
that is carried in the check's caveats, not buried here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evotwin import kaplan_meier, km_median


GDC_CLINICAL_PATH = "data/gdc_clinical/tcga_gbm_cases.json"

# Stupp 2005 (NEJM 352:987) reported 14.6 months median survival with
# radiotherapy + temozolomide; TCGA-GBM cohort reports cluster at 12-15
# months. The band is deliberately wide: it is an order-of-magnitude anchor
# for an ingestion check, not a hypothesis test.
PUBLISHED_GBM_OS_BAND_DAYS: Tuple[float, float] = (330.0, 550.0)
PUBLISHED_GBM_OS_SOURCE = (
    "Stupp et al., NEJM 352:987-996 (2005): 14.6 months median survival for "
    "radiotherapy + temozolomide; TCGA-GBM cohort reports 12-15 months"
)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicalCase:
    """One reconstructed case, both endpoints."""

    submitter_id: str
    progression_time: Optional[float]
    progressed: bool
    follow_up_time: Optional[float]
    os_time: Optional[float]
    vital_status: Optional[str]


@dataclass
class ObservedCohort:
    """The reconstructed real cohort, plus its own caveats."""

    source: str
    n_cases: int
    cases: List[ClinicalCase] = field(default_factory=list)
    progression_times: Tuple[float, ...] = ()
    progression_events: Tuple[int, ...] = ()
    os_times: Tuple[float, ...] = ()
    os_events: Tuple[int, ...] = ()
    km_progression: Dict[str, object] = field(default_factory=dict)
    km_os: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def n_progression_events(self) -> int:
        return int(sum(self.progression_events))

    @property
    def n_progression_censored(self) -> int:
        return len(self.progression_times) - self.n_progression_events

    @property
    def median_progression_days(self) -> Optional[float]:
        return km_median(self.km_progression) if self.km_progression else None

    @property
    def median_os_days(self) -> Optional[float]:
        return km_median(self.km_os) if self.km_os else None

    def progression_event_iqr(self) -> Optional[Tuple[float, float]]:
        """IQR of the DATED progression events only (censoring-free view).

        The KM median of this cohort is inflated by heavy, short-follow-up
        censoring, so the event-time IQR is reported alongside it as the
        more stable central comparison.
        """
        dated = [
            t for t, e in zip(self.progression_times, self.progression_events) if e == 1
        ]
        if not dated:
            return None
        q1, q3 = np.percentile(dated, [25, 75])
        return (float(q1), float(q3))

    def survival_at(self, km: Dict[str, object], day: float) -> float:
        """Step-function lookup of S(t) for a KM dict."""
        ts = np.asarray(km["t"], dtype=float)
        Ss = np.asarray(km["S"], dtype=float)
        if len(ts) == 0:
            raise ValueError("empty KM curve")
        idx = np.searchsorted(ts, day, side="right") - 1
        if idx < 0:
            return 1.0
        return float(Ss[idx])


def reconstruct_cohort(payload: Dict[str, object], source: str) -> ObservedCohort:
    """Build the two observed cohorts from a raw GDC cases payload.

    Progression arm: earliest dated progression record per case is the event
    time; a case with no dated progression is censored at its last follow-up.
    OS arm: `days_to_death` with `vital_status == "Dead"`, censored at last
    follow-up for `Alive` cases. Cases with neither usable endpoint are
    counted in `n_cases` but contribute to no curve -- they are not silently
    dropped into a zero.
    """
    hits = payload.get("hits", [])
    cases: List[ClinicalCase] = []
    prog_t: List[float] = []
    prog_e: List[int] = []
    os_t: List[float] = []
    os_e: List[int] = []

    for h in hits:
        sid = str(h.get("submitter_id") or h.get("id") or "unknown")
        follow_ups = h.get("follow_ups") or []
        dated: List[float] = []
        for fu in follow_ups:
            if (
                fu.get("progression_or_recurrence") == "Yes"
                and fu.get("days_to_progression") is not None
            ):
                dated.append(float(fu["days_to_progression"]))
        follow_up_times = [
            float(fu["days_to_follow_up"])
            for fu in follow_ups
            if fu.get("days_to_follow_up") is not None
        ]
        last_fu = max(follow_up_times) if follow_up_times else None

        prog_time: Optional[float] = min(dated) if dated else None
        progressed = prog_time is not None
        if progressed:
            prog_t.append(float(prog_time))
            prog_e.append(1)
        elif last_fu is not None:
            prog_t.append(float(last_fu))
            prog_e.append(0)

        demographic = h.get("demographic") or {}
        vital = demographic.get("vital_status")
        dtd = demographic.get("days_to_death")
        os_time: Optional[float] = None
        if vital == "Dead" and dtd is not None:
            os_time = float(dtd)
            os_t.append(os_time)
            os_e.append(1)
        elif vital == "Alive" and last_fu is not None:
            os_time = float(last_fu)
            os_t.append(os_time)
            os_e.append(0)

        cases.append(
            ClinicalCase(
                submitter_id=sid,
                progression_time=prog_time,
                progressed=progressed,
                follow_up_time=last_fu,
                os_time=os_time,
                vital_status=vital,
            )
        )

    if not prog_t:
        raise ValueError("no usable progression records in payload")
    if not os_t:
        raise ValueError("no usable survival records in payload")

    cohort = ObservedCohort(
        source=source,
        n_cases=len(cases),
        cases=cases,
        progression_times=tuple(prog_t),
        progression_events=tuple(prog_e),
        os_times=tuple(os_t),
        os_events=tuple(os_e),
        km_progression=kaplan_meier(prog_t, prog_e),
        km_os=kaplan_meier(os_t, os_e),
        notes=[
            "progression events are the recorded TCGA field "
            "(days_to_progression with progression_or_recurrence == 'Yes'); "
            "this field is inconsistently populated and its definition is not "
            "endpoint-matched to the model's burden threshold",
            "censoring is heavy and non-random (short follow-up), which "
            "inflates the KM median; the event-time IQR is reported as the "
            "more stable central comparator",
            "treatment is heterogeneous (surgery, radiotherapy, temozolomide, "
            "and later-line agents), so this cohort is not one regimen",
        ],
    )
    return cohort


def load_tcga_gbm_clinical(
    path: str = GDC_CLINICAL_PATH,
    source_note: str = "GDC open API, project TCGA-GBM (expand demographic, diagnoses, follow_ups)",
) -> ObservedCohort:
    """Load the cached GDC clinical export and reconstruct the cohorts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no GDC clinical export at {path}: fetch it with the GDC cases "
            "endpoint (project TCGA-GBM, expand=demographic,diagnoses,follow_ups) "
            "-- this module refuses to reconstruct a cohort from nothing"
        )
    with open(path) as fh:
        payload = json.load(fh)
    note = payload.get("source") or source_note
    return reconstruct_cohort(payload, source=note)


# ---------------------------------------------------------------------------
# Check 1: the data-handling anchor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorCheck:
    """Does the reconstruction recover the published number?"""

    median_os_days: Optional[float]
    band_days: Tuple[float, float]
    source: str
    ok: bool

    def message(self) -> str:
        med = "not reached" if self.median_os_days is None else f"{self.median_os_days:.0f} d"
        lo, hi = self.band_days
        verdict = "INSIDE" if self.ok else "OUTSIDE"
        return (
            f"reconstructed cohort median OS {med} is {verdict} the published "
            f"band [{lo:.0f}, {hi:.0f}] d ({self.source})"
        )


def anchor_check(
    cohort: ObservedCohort,
    band_days: Tuple[float, float] = PUBLISHED_GBM_OS_BAND_DAYS,
) -> AnchorCheck:
    """Validate the ingestion: a wrong reconstruction cannot hit the anchor."""
    med = cohort.median_os_days
    ok = med is not None and band_days[0] <= med <= band_days[1]
    return AnchorCheck(
        median_os_days=med,
        band_days=band_days,
        source=PUBLISHED_GBM_OS_SOURCE,
        ok=ok,
    )


# ---------------------------------------------------------------------------
# Check 2: model control arm vs the observed cohort
# ---------------------------------------------------------------------------


@dataclass
class PlausibilityCheck:
    """Model MTD arm vs observed progression curve (weakest check, labelled)."""

    label: str
    model_n: int
    model_events: int
    model_median_days: Optional[float]
    observed_median_days: Optional[float]
    observed_event_iqr: Optional[Tuple[float, float]]
    eval_table: List[Tuple[float, float, float, float]]  # day, obs S, model S, gap
    caveats: List[str] = field(default_factory=list)

    @property
    def max_abs_gap(self) -> float:
        return max((abs(g) for _, _, _, g in self.eval_table), default=0.0)

    @property
    def median_within_event_iqr(self) -> Optional[bool]:
        """Is the model's median progression time inside the real event IQR?"""
        if self.model_median_days is None or self.observed_event_iqr is None:
            return None
        lo, hi = self.observed_event_iqr
        return lo <= self.model_median_days <= hi

    def verdict(self) -> str:
        """A sentence that cannot overclaim, whatever the numbers say."""
        med = (
            "not reached"
            if self.model_median_days is None
            else f"{self.model_median_days:.0f} d"
        )
        obs = (
            "not reached"
            if self.observed_median_days is None
            else f"{self.observed_median_days:.0f} d"
        )
        inside = self.median_within_event_iqr
        if inside is None:
            placement = "not comparable (a median is unreached)"
        elif inside:
            placement = (
                "inside the observed dated-event IQR, i.e. CONSISTENT WITH the "
                "real timescale"
            )
        else:
            placement = (
                "OUTSIDE the observed dated-event IQR, i.e. the control arm "
                "does not match the observed timescale on this cohort"
            )
        return (
            f"[{self.label}] model MTD-arm median progression {med} (n={self.model_n}, "
            f"{self.model_events} events) vs observed cohort KM median {obs}: {placement}. "
            "This is a timescale sanity check on the CONTROL arm, not validation: "
            "no patient matching, no regimen matching, endpoints not matched."
        )

    def summary_table(self) -> str:
        lines = [
            f"PLAUSIBILITY CHECK -- {self.label}",
            f"{'day':>6s} {'observed S(t)':>14s} {'model S(t)':>12s} {'gap':>8s}",
        ]
        for day, obs, mod, gap in self.eval_table:
            lines.append(f"{day:>6.0f} {obs:>14.3f} {mod:>12.3f} {gap:>+8.3f}")
        lines.append(f"max |gap| = {self.max_abs_gap:.3f}")
        lines.append(self.verdict())
        for c in self.caveats:
            lines.append(f"  caveat: {c}")
        return "\n".join(lines)


def plausibility_check(
    observed: ObservedCohort,
    model_times: Sequence[float],
    model_events: Sequence[int],
    label: str = "model MTD arm vs observed TCGA-GBM progression",
    eval_days: Sequence[float] = (100.0, 200.0, 365.0),
) -> PlausibilityCheck:
    """Compare the model's control arm curve to the observed cohort curve.

    Raises when the model arm is empty: a check with no data on one side
    would otherwise render as an agreement.
    """
    if len(model_times) != len(model_events) or len(model_times) == 0:
        raise ValueError("model arm needs equal-length, non-empty times and events")
    if any(e not in (0, 1) for e in model_events):
        raise ValueError("events must be 0/1")

    model_km = kaplan_meier(list(model_times), list(model_events))
    table = []
    for day in eval_days:
        obs_s = observed.survival_at(observed.km_progression, day)
        mod_s = observed.survival_at(model_km, day)
        table.append((float(day), obs_s, mod_s, mod_s - obs_s))

    return PlausibilityCheck(
        label=label,
        model_n=len(model_times),
        model_events=int(sum(model_events)),
        model_median_days=km_median(model_km),
        observed_median_days=observed.median_progression_days,
        observed_event_iqr=observed.progression_event_iqr(),
        eval_table=table,
        caveats=[
            "the model endpoint is a RANO-fraction burden threshold crossing; "
            "the observed endpoint is a recorded clinical progression date",
            "the observed KM median is inflated by heavy short-follow-up "
            "censoring; the dated-event IQR is the comparator used above",
            "the model's MTD arm is a constant-parameter model, not a patient "
            "or a trial: agreement is a timescale sanity check only",
            "the model is NOT fitted to this cohort, so agreement is not a "
            "calibration result and disagreement is a re-calibration target",
        ],
    )


# ---------------------------------------------------------------------------
# Model-side curves from the real-geometry campaign log
# ---------------------------------------------------------------------------


def model_curve_from_campaign_log(
    entries: Sequence[Dict[str, object]],
    arm: str = "mtd",
    horizon_days: float = 365.0,
) -> Tuple[List[float], List[int]]:
    """Turn logged replicate PFS values into (times, events).

    A replicate whose value equals the horizon was censored (no progression
    inside the window); anything shorter is a progression event. Values
    above the horizon mean the log and the caller disagree about the window,
    which is a refusal, not a rounding.
    """
    if arm not in ("mtd", "policy"):
        raise ValueError("arm must be 'mtd' or 'policy'")
    key = "pfs_mtd_replicates" if arm == "mtd" else "pfs_policy_replicates"
    times: List[float] = []
    events: List[int] = []
    for e in entries:
        if key not in e:
            raise ValueError(f"log entry {e.get('case_id')!r} has no {key}")
        vals = e[key]
        if not vals:
            raise ValueError(
                f"log entry {e.get('case_id')!r} has no replicate values in {key}"
            )
        for v in vals:
            t = float(v)
            if t > horizon_days:
                raise ValueError(
                    f"replicate PFS {t} exceeds the stated horizon {horizon_days}: "
                    "the log and the caller disagree about the censoring window"
                )
            times.append(t)
            events.append(0 if t >= horizon_days else 1)
    if not times:
        raise ValueError("no replicate values in the supplied log entries")
    return times, events


def load_campaign_log(path: str = "outputs_real/campaign_log.json") -> List[Dict[str, object]]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError(f"campaign log at {path} is not a non-empty list")
    return data


# ---------------------------------------------------------------------------
# Honesty record
# ---------------------------------------------------------------------------


def retrospective_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "tcga_gbm_clinical_export_real": EvidenceStatus.ESTABLISHED,
        "ingestion_recovers_published_os_anchor": EvidenceStatus.ESTABLISHED,
        "tcga_dated_progression_field_quality": EvidenceStatus.UNKNOWN,
        "model_timescale_matches_patients": EvidenceStatus.UNKNOWN,
        "this_check_is_clinical_validation": EvidenceStatus.PROVEN_FAILED,
    }


def what_this_check_cannot_do() -> List[str]:
    return [
        "predict any individual's outcome: no BraTS case can be matched to a "
        "GDC case (different anonymisation), so there is no pair to score",
        "establish that the model's mechanism is right: matching a control-arm "
        "timescale is necessary, nowhere near sufficient",
        "substitute for a trial: the model has never been compared against "
        "patients who received the adaptive schedule (nobody has)",
        "resolve the progression endpoint: TCGA's dated progression field is "
        "sparse and not RANO-adjudicated",
    ]
